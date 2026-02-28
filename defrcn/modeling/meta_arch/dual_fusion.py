import torch
import torch.nn.functional as F
from torch import nn


def _build_group_norm(num_channels):
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class DualFusionNeck(nn.Module):
    """
    Build two task-specific fused feature maps from multi-scale backbone outputs:
      - F_rpn for proposal generation
      - F_roi for ROI classification/regression
    """

    def __init__(self, cfg, input_shapes):
        super().__init__()
        fusion_cfg = cfg.MODEL.DUAL_FUSION

        self.in_features = list(fusion_cfg.IN_FEATURES)
        self.out_feature = fusion_cfg.OUT_FEATURE
        self.align_channels = int(fusion_cfg.ALIGN_CHANNELS)
        self.use_refine = bool(fusion_cfg.USE_REFINE)

        if self.out_feature not in input_shapes:
            raise KeyError("MODEL.DUAL_FUSION.OUT_FEATURE '{}' is missing from backbone outputs.".format(
                self.out_feature
            ))
        for feat in self.in_features:
            if feat not in input_shapes:
                raise KeyError("MODEL.DUAL_FUSION.IN_FEATURES contains missing feature '{}'.".format(feat))

        self.align_layers = nn.ModuleDict()
        for feat in self.in_features:
            in_channels = int(input_shapes[feat].channels)
            self.align_layers[feat] = nn.Conv2d(in_channels, self.align_channels, kernel_size=1, stride=1, padding=0)

        rpn_init_logits = torch.tensor(fusion_cfg.RPN_INIT_LOGITS, dtype=torch.float32)
        roi_init_logits = torch.tensor(fusion_cfg.ROI_INIT_LOGITS, dtype=torch.float32)
        if len(rpn_init_logits) != len(self.in_features):
            raise ValueError("MODEL.DUAL_FUSION.RPN_INIT_LOGITS length must match IN_FEATURES.")
        if len(roi_init_logits) != len(self.in_features):
            raise ValueError("MODEL.DUAL_FUSION.ROI_INIT_LOGITS length must match IN_FEATURES.")
        self.rpn_logits = nn.Parameter(rpn_init_logits)
        self.roi_logits = nn.Parameter(roi_init_logits)

        self.register_buffer(
            "rpn_level_mask",
            self._build_level_mask(fusion_cfg.RPN_LEVELS, self.in_features),
        )
        self.register_buffer(
            "roi_level_mask",
            self._build_level_mask(fusion_cfg.ROI_LEVELS, self.in_features),
        )

        if self.use_refine:
            self.rpn_refine = nn.Sequential(
                nn.Conv2d(self.align_channels, self.align_channels, kernel_size=3, stride=1, padding=1, bias=False),
                _build_group_norm(self.align_channels),
                nn.ReLU(inplace=True),
            )
            self.roi_refine = nn.Sequential(
                nn.Conv2d(self.align_channels, self.align_channels, kernel_size=3, stride=1, padding=1, bias=False),
                _build_group_norm(self.align_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.rpn_refine = nn.Identity()
            self.roi_refine = nn.Identity()

        out_channels = int(input_shapes[self.out_feature].channels)
        if self.align_channels == out_channels:
            self.rpn_out_proj = nn.Identity()
            self.roi_out_proj = nn.Identity()
        else:
            self.rpn_out_proj = nn.Conv2d(self.align_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.roi_out_proj = nn.Conv2d(self.align_channels, out_channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _build_level_mask(enabled_levels, ordered_levels):
        if enabled_levels is None:
            enabled_levels = ordered_levels
        enabled_levels = set(enabled_levels)
        mask = torch.tensor([lvl in enabled_levels for lvl in ordered_levels], dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("At least one fusion level must be enabled per branch.")
        return mask

    @staticmethod
    def _masked_softmax(logits, level_mask):
        min_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~level_mask, min_value)
        return torch.softmax(masked_logits, dim=0)

    def _aligned_inputs(self, features):
        target_h, target_w = features[self.out_feature].shape[-2:]
        aligned = []
        for feat in self.in_features:
            x = self.align_layers[feat](features[feat])
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(x)
        return aligned

    def get_fusion_weights(self):
        rpn_weights = self._masked_softmax(self.rpn_logits, self.rpn_level_mask).detach().cpu()
        roi_weights = self._masked_softmax(self.roi_logits, self.roi_level_mask).detach().cpu()
        return {
            "levels": list(self.in_features),
            "rpn": rpn_weights.tolist(),
            "roi": roi_weights.tolist(),
        }

    def forward(self, features):
        aligned = self._aligned_inputs(features)
        rpn_weights = self._masked_softmax(self.rpn_logits, self.rpn_level_mask)
        roi_weights = self._masked_softmax(self.roi_logits, self.roi_level_mask)

        fused_rpn = 0.0
        fused_roi = 0.0
        for idx, x in enumerate(aligned):
            fused_rpn = fused_rpn + rpn_weights[idx] * x
            fused_roi = fused_roi + roi_weights[idx] * x

        fused_rpn = self.rpn_out_proj(self.rpn_refine(fused_rpn))
        fused_roi = self.roi_out_proj(self.roi_refine(fused_roi))
        return fused_rpn, fused_roi
