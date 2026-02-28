import torch
from torch import nn


def _make_group_norm(num_channels, num_groups):
    groups = min(int(num_groups), int(num_channels))
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _build_norm_2d(norm_type, num_channels, gn_num_groups):
    norm = str(norm_type).upper()
    if norm == "GN":
        return _make_group_norm(num_channels, gn_num_groups)
    if norm == "LN":
        # Channel-first LN-style normalization via GN(1, C).
        return nn.GroupNorm(1, num_channels)
    if norm in {"", "NONE"}:
        return nn.Identity()
    raise ValueError("Unsupported MODEL.BRANCH_ADAPTER.NORM: '{}'".format(norm_type))


class BottleneckAdapter(nn.Module):
    """
    1x1 reduce -> norm -> relu -> 3x3 -> norm -> relu -> 1x1 restore.
    """

    def __init__(self, in_channels, cfg):
        super().__init__()
        ratio = float(cfg.MODEL.BRANCH_ADAPTER.BOTTLENECK_RATIO)
        min_channels = int(cfg.MODEL.BRANCH_ADAPTER.MIN_CHANNELS)
        inner_channels = max(int(round(in_channels * ratio)), min_channels)
        inner_channels = min(inner_channels, in_channels)
        if inner_channels <= 0:
            raise ValueError("Invalid adapter inner channel size: {}".format(inner_channels))

        norm = cfg.MODEL.BRANCH_ADAPTER.NORM
        gn_num_groups = int(cfg.MODEL.BRANCH_ADAPTER.GN_NUM_GROUPS)

        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.norm1 = _build_norm_2d(norm, inner_channels, gn_num_groups)
        self.conv2 = nn.Conv2d(
            inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = _build_norm_2d(norm, inner_channels, gn_num_groups)
        self.conv3 = nn.Conv2d(inner_channels, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if cfg.MODEL.BRANCH_ADAPTER.ZERO_INIT_LAST:
            nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        return y


class BranchAdapter(nn.Module):
    """
    Forward-path branch decoupling adapters for RPN and ROI branches.
    """

    def __init__(self, cfg, shape_dict):
        super().__init__()
        self.cfg = cfg
        self.use_gate = bool(cfg.MODEL.BRANCH_ADAPTER.USE_GATE)
        self.shared = bool(cfg.MODEL.BRANCH_ADAPTER.SHARED)
        self.gate_init = float(cfg.MODEL.BRANCH_ADAPTER.GATE_INIT)
        self.rpn_features = tuple(cfg.MODEL.BRANCH_ADAPTER.RPN_FEATURES)
        self.roi_features = tuple(cfg.MODEL.BRANCH_ADAPTER.ROI_FEATURES)

        known_features = set(shape_dict.keys())
        for feat in set(self.rpn_features + self.roi_features):
            if feat not in known_features:
                raise ValueError(
                    "MODEL.BRANCH_ADAPTER feature '{}' is not in backbone outputs: {}".format(
                        feat, sorted(known_features)
                    )
                )

        if self.shared:
            shared_feats = sorted(set(self.rpn_features + self.roi_features))
            self.shared_adapters = nn.ModuleDict(
                {f: BottleneckAdapter(shape_dict[f].channels, cfg) for f in shared_feats}
            )
            if self.use_gate:
                self.shared_gates = nn.ParameterDict(
                    {
                        f: nn.Parameter(
                            torch.tensor(self.gate_init, dtype=torch.float32).view(1, 1, 1, 1)
                        )
                        for f in shared_feats
                    }
                )
            else:
                self.shared_gates = None
        else:
            self.rpn_adapters = nn.ModuleDict(
                {f: BottleneckAdapter(shape_dict[f].channels, cfg) for f in self.rpn_features}
            )
            self.roi_adapters = nn.ModuleDict(
                {f: BottleneckAdapter(shape_dict[f].channels, cfg) for f in self.roi_features}
            )
            if self.use_gate:
                self.rpn_gates = nn.ParameterDict(
                    {
                        f: nn.Parameter(
                            torch.tensor(self.gate_init, dtype=torch.float32).view(1, 1, 1, 1)
                        )
                        for f in self.rpn_features
                    }
                )
                self.roi_gates = nn.ParameterDict(
                    {
                        f: nn.Parameter(
                            torch.tensor(self.gate_init, dtype=torch.float32).view(1, 1, 1, 1)
                        )
                        for f in self.roi_features
                    }
                )
            else:
                self.rpn_gates = None
                self.roi_gates = None

    def _apply_adapters(self, features, feature_names, adapter_dict, gate_dict):
        out = dict(features)
        for feat in feature_names:
            if feat not in features:
                continue
            delta = adapter_dict[feat](features[feat])
            if self.use_gate:
                delta = gate_dict[feat] * delta
            out[feat] = features[feat] + delta
        return out

    def forward(self, features_rpn, features_roi):
        if self.shared:
            out_rpn = self._apply_adapters(
                features_rpn, self.rpn_features, self.shared_adapters, self.shared_gates
            )
            out_roi = self._apply_adapters(
                features_roi, self.roi_features, self.shared_adapters, self.shared_gates
            )
            return out_rpn, out_roi

        out_rpn = self._apply_adapters(
            features_rpn, self.rpn_features, self.rpn_adapters, self.rpn_gates
        )
        out_roi = self._apply_adapters(
            features_roi, self.roi_features, self.roi_adapters, self.roi_gates
        )
        return out_rpn, out_roi
