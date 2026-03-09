"""
Batch-Agnostic Few-Shot Detection (BA-FSD).

Replaces BatchNorm with GroupNorm + Weight Standardization in the backbone
to eliminate batch-size dependence. This makes training invariant to batch
size, addressing the core issue of BS=8 vs BS=16 performance gap.

Components:
1. WeightStandardizedConv2d: Conv2d with per-channel weight normalization
2. convert_bn_to_gn_ws(): Recursive model conversion
3. TaskAdaptiveNorm: Optional support-conditioned normalization for novel classes

Usage:
- Enable via cfg.MODEL.BATCH_AGNOSTIC.ENABLE = True
- Requires retraining from base stage (GN+WS changes feature statistics)
- GN+WS is applied during base training on abundant data -> no overfitting risk
- TAN (if enabled) adds a small hypernetwork for novel fine-tuning

This module is called from rcnn.py after model construction when the
BATCH_AGNOSTIC config is enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization.

    Normalizes the convolutional kernel weights to have zero mean and unit
    variance per output channel. This stabilizes training and works
    synergistically with GroupNorm.

    Reference: Qiao et al., "Micro-Batch Training with Batch-Channel
    Normalization and Weight Standardization", 2019.
    """

    def forward(self, x):
        weight = self.weight
        # Per-output-channel normalization: mean/std over (Cin, kH, kW)
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight_std = weight.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-5)
        weight = (weight - weight_mean) / weight_std
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class TaskAdaptiveNorm(nn.Module):
    """Task-Adaptive Normalization for novel classes.

    A small hypernetwork maps support set features to per-channel affine
    parameters (gamma, beta) for GroupNorm. This conditions the normalization
    on the novel task without adding per-class parameters.

    Initialized to identity transform (gamma=1, beta=0) so the model
    starts identical to vanilla GN and only deviates when gradients
    provide strong signal.
    """

    def __init__(
        self,
        num_channels: int,
        support_dim: int = 2048,
        bottleneck: int = 64,
    ):
        super().__init__()
        self.gn = nn.GroupNorm(32, num_channels, affine=False)
        self.num_channels = num_channels

        # Hypernetwork: support features -> affine params
        self.hyper = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(support_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, 2 * num_channels),
        )

        # Initialize to identity transform
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)
        # gamma=1 for the first num_channels biases
        self.hyper[-1].bias.data[:num_channels] = 1.0

        # Cached affine params (set once from support, used during inference)
        self._cached_gamma: Optional[torch.Tensor] = None
        self._cached_beta: Optional[torch.Tensor] = None

    def set_support_features(self, support_feat: torch.Tensor):
        """Compute and cache affine params from support features.

        Call once with the K-shot support features before inference.
        """
        with torch.no_grad():
            if support_feat.dim() == 2:
                # (N, D) -> add spatial dims for AdaptiveAvgPool2d
                support_feat = support_feat.unsqueeze(-1).unsqueeze(-1)
            # Average across support samples
            mean_feat = support_feat.mean(dim=0, keepdim=True)
            params = self.hyper(mean_feat)
            self._cached_gamma = params[0, : self.num_channels].view(1, -1, 1, 1)
            self._cached_beta = params[0, self.num_channels :].view(1, -1, 1, 1)

    def forward(self, x, support_feat=None):
        out = self.gn(x)

        if support_feat is not None:
            if support_feat.dim() == 2:
                support_feat = support_feat.unsqueeze(-1).unsqueeze(-1)
            params = self.hyper(support_feat)
            gamma = params[:, : self.num_channels].unsqueeze(-1).unsqueeze(-1)
            beta = params[:, self.num_channels :].unsqueeze(-1).unsqueeze(-1)
            out = gamma * out + beta
        elif self._cached_gamma is not None:
            out = self._cached_gamma * out + self._cached_beta

        return out


def convert_bn_to_gn_ws(
    model: nn.Module,
    num_groups: int = 32,
    weight_standardization: bool = True,
    convert_1x1: bool = False,
) -> int:
    """Replace all BN layers with GN and Conv2d with WS-Conv2d.

    Args:
        model: Model to convert (modified in-place)
        num_groups: Number of groups for GroupNorm
        weight_standardization: Whether to apply WS to Conv2d
        convert_1x1: Whether to apply WS to 1x1 convolutions

    Returns:
        Number of layers converted
    """
    count = 0

    for name, module in list(model.named_children()):
        # Recursively convert children first
        count += convert_bn_to_gn_ws(
            module, num_groups, weight_standardization, convert_1x1
        )

        # Convert BatchNorm variants to GroupNorm
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            num_channels = module.num_features
            gn = nn.GroupNorm(
                min(num_groups, num_channels), num_channels, affine=True
            )
            # Initialize GN affine from BN affine if available
            if module.weight is not None:
                gn.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                gn.bias.data.copy_(module.bias.data)
            setattr(model, name, gn)
            count += 1

        # Handle FrozenBatchNorm2d (detectron2-specific)
        elif type(module).__name__ == "FrozenBatchNorm2d":
            num_channels = module.num_features
            gn = nn.GroupNorm(
                min(num_groups, num_channels), num_channels, affine=True
            )
            setattr(model, name, gn)
            count += 1

        # Convert Conv2d to WeightStandardizedConv2d
        elif weight_standardization and isinstance(module, nn.Conv2d):
            # Skip 1x1 convs unless explicitly requested
            if not convert_1x1 and module.kernel_size == (1, 1):
                continue
            # Skip depthwise convolutions
            if module.groups == module.in_channels:
                continue

            ws_conv = WeightStandardizedConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            ws_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ws_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, ws_conv)
            count += 1

    return count


def apply_batch_agnostic(model: nn.Module, cfg) -> nn.Module:
    """Apply batch-agnostic modifications to the model.

    This is the main entry point called from the model construction
    pipeline when cfg.MODEL.BATCH_AGNOSTIC.ENABLE is True.

    Args:
        model: GeneralizedRCNN model
        cfg: Config node

    Returns:
        Modified model (in-place)
    """
    ba_cfg = cfg.MODEL.BATCH_AGNOSTIC
    num_groups = int(ba_cfg.GN_NUM_GROUPS)
    ws = bool(ba_cfg.WEIGHT_STANDARDIZATION)
    convert_1x1 = bool(ba_cfg.CONVERT_1X1)

    # Convert backbone
    backbone_count = convert_bn_to_gn_ws(
        model.backbone, num_groups, ws, convert_1x1
    )
    logger.info(
        "BA-FSD: Converted %d layers in backbone (GN groups=%d, WS=%s)",
        backbone_count,
        num_groups,
        ws,
    )

    # Convert ROI head (res5)
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "res5"):
        roi_count = convert_bn_to_gn_ws(
            model.roi_heads.res5, num_groups, ws, convert_1x1
        )
        logger.info("BA-FSD: Converted %d layers in roi_heads.res5", roi_count)

    return model
