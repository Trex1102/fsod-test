import torch
from torch import nn


def _make_group_norm(num_channels, num_groups):
    groups = min(int(num_groups), int(num_channels))
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _build_norm(norm_type, num_channels, gn_num_groups):
    norm = str(norm_type).upper()
    if norm == "GN":
        return _make_group_norm(num_channels, gn_num_groups)
    if norm == "LN":
        return nn.GroupNorm(1, num_channels)
    if norm in {"", "NONE"}:
        return nn.Identity()
    raise ValueError("Unsupported RES5_ADAPTER.NORM: '{}'".format(norm_type))


class _AdapterBlock(nn.Module):
    """
    Lightweight bottleneck adapter: 1x1 -> 3x3 -> 1x1 with residual.
    Initialized as identity (zero-init last conv) so it starts with no effect.
    """

    def __init__(self, channels, inner_channels, norm_type, gn_num_groups, zero_init_last):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False)
        self.norm1 = _build_norm(norm_type, inner_channels, gn_num_groups)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _build_norm(norm_type, inner_channels, gn_num_groups)
        self.conv3 = nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if zero_init_last:
            nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        return self.conv3(y)


class Res5WithAdapters(nn.Module):
    """
    Wraps a res5 nn.Sequential with trainable adapter modules inserted after
    selected blocks. The original res5 block weights are frozen; only the
    adapter parameters update during fine-tuning.

    Forward pass per block:
        x = res5_block(x)
        x = x + gate * adapter(x)   # if adapter enabled for this block
    """

    def __init__(self, res5_seq, out_channels, cfg):
        super().__init__()

        adapter_cfg = cfg.MODEL.ROI_HEADS.RES5_ADAPTER
        after_blocks  = set(int(i) for i in adapter_cfg.AFTER_BLOCKS)
        ratio         = float(adapter_cfg.BOTTLENECK_RATIO)
        min_channels  = int(adapter_cfg.MIN_CHANNELS)
        norm_type     = str(adapter_cfg.NORM)
        gn_groups     = int(adapter_cfg.GN_NUM_GROUPS)
        use_gate      = bool(adapter_cfg.USE_GATE)
        gate_init     = float(adapter_cfg.GATE_INIT)
        zero_init     = bool(adapter_cfg.ZERO_INIT_LAST)

        # Store original blocks; freeze them here so the caller does not need
        # special-case logic — FREEZE_FEAT still freezes the adapter wrapper,
        # but the adapters' parameters are re-enabled in rcnn.py after the
        # general freeze call (see rcnn.py).
        self.blocks = nn.ModuleList(list(res5_seq.children()))

        inner = max(int(round(out_channels * ratio)), min_channels)
        inner = min(inner, out_channels)

        self.adapters  = nn.ModuleDict()
        self.use_gate  = use_gate
        self.gates     = nn.ParameterDict() if use_gate else None

        for idx in range(len(self.blocks)):
            if idx in after_blocks:
                self.adapters[str(idx)] = _AdapterBlock(
                    out_channels, inner, norm_type, gn_groups, zero_init
                )
                if use_gate:
                    self.gates[str(idx)] = nn.Parameter(
                        torch.tensor(gate_init, dtype=torch.float32).view(1, 1, 1, 1)
                    )

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
            key = str(idx)
            if key in self.adapters:
                delta = self.adapters[key](x)
                if self.use_gate:
                    delta = self.gates[key] * delta
                x = x + delta
        return x

    def adapter_parameters(self):
        """Returns only the adapter (and gate) parameters — useful for re-enabling
        grad after a blanket freeze of roi_heads.res5."""
        for p in self.adapters.parameters():
            yield p
        if self.gates is not None:
            for p in self.gates.parameters():
                yield p
