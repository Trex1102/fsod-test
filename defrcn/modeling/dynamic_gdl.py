"""
Dynamic Gradient Decoupled Layer (DP-GDL).

Replaces DeFRCN's fixed gradient scaling (lambda_rpn=0, lambda_rcnn=0.75)
with input-dependent learned gating. The gate is a lightweight channel
attention module that learns to route gradients differently for
classification vs localization features.

Components:
1. DynamicGDL: Gradient scaling with learned per-channel lambda
2. DualPathwayRouter: Splits features for cls/loc with learned routing
3. orthogonality_loss(): Regularization for pathway orthogonality

Usage:
- Enable via cfg.MODEL.DYNAMIC_GDL.ENABLE = True
- Gate is trained during base training and FROZEN during novel fine-tuning
- Zero additional parameters during novel fine-tuning

This module replaces the fixed decouple_layer() calls in rcnn.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import logging

logger = logging.getLogger(__name__)


class DynamicGradientScale(Function):
    """Autograd function that scales gradients by a learned per-channel mask."""

    @staticmethod
    def forward(ctx, x, lambda_mask):
        ctx.save_for_backward(lambda_mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_mask,) = ctx.saved_tensors
        # lambda_mask: (1, C, 1, 1) or broadcastable
        return grad_output * lambda_mask, None


class GateNetwork(nn.Module):
    """Channel attention gate that produces per-channel gradient scaling.

    Maps feature statistics -> per-channel lambda in [0, 1].
    Initialized to approximate the original fixed lambda.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        init_lambda: float = 0.75,
    ):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

        # Initialize bias of last linear so that output ≈ init_lambda
        # Sigmoid(x) = init_lambda => x = log(init_lambda / (1 - init_lambda))
        init_logit = torch.log(
            torch.tensor(max(init_lambda, 1e-4) / max(1 - init_lambda, 1e-4))
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, init_logit.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-channel lambda mask.

        Args:
            x: feature map (B, C, H, W)

        Returns:
            lambda_mask: (B, C, 1, 1) in [0, 1]
        """
        lambdas = self.gate(x)  # (B, C)
        return lambdas.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)


class DynamicGDL(nn.Module):
    """Dynamic Gradient Decoupled Layer with learned gating.

    Replaces fixed lambda with input-dependent per-channel scaling.
    In forward pass: identity. In backward pass: scale by learned lambda.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        init_lambda: float = 0.75,
    ):
        super().__init__()
        self.gate_net = GateNetwork(channels, reduction, init_lambda)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lambda_mask = self.gate_net(x)
        return DynamicGradientScale.apply(x, lambda_mask)


class DualPathwayRouter(nn.Module):
    """Dual-pathway feature routing for classification and localization.

    Splits backbone features into classification-relevant (spatially pooled,
    translation-invariant) and localization-relevant (spatially preserved)
    pathways. A learned gate controls the routing.

    The routing gate is trained during base training and provides implicit
    feature specialization that transfers to novel classes.
    """

    def __init__(self, in_channels: int = 1024):
        super().__init__()
        # Classification pathway: spatial invariance via global avg pool
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )

        # Localization pathway: preserve spatial info
        self.loc_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True),
        )

        # Routing gate
        self.route_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid(),
        )
        # Initialize gate to 0.5 (equal routing)
        nn.init.zeros_(self.route_gate[-2].weight)
        nn.init.zeros_(self.route_gate[-2].bias)

    def forward(self, x: torch.Tensor):
        """Route features for classification and localization.

        Args:
            x: (B, C, H, W) feature map

        Returns:
            Blended feature map (B, C, H, W) with routing applied
        """
        gate = self.route_gate(x)  # (B, 1)

        # Classification features (broadcast spatially)
        cls_feat = self.cls_pool(x).flatten(1)  # (B, C)
        cls_feat = self.cls_proj(cls_feat)  # (B, C)
        cls_feat = cls_feat.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        # Localization features
        loc_feat = self.loc_proj(x)  # (B, C, H, W)

        # Gated blending
        gate_2d = gate.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        blended = gate_2d * cls_feat + (1 - gate_2d) * loc_feat

        return blended


def orthogonality_loss(cls_grad: torch.Tensor, loc_grad: torch.Tensor) -> torch.Tensor:
    """Penalize alignment between classification and localization gradients.

    Encourages the two pathways to learn complementary features by
    minimizing the squared cosine similarity of their gradient directions.

    Args:
        cls_grad: (B, C, ...) classification pathway gradients
        loc_grad: (B, C, ...) localization pathway gradients

    Returns:
        Scalar loss
    """
    cls_flat = cls_grad.flatten(1)
    loc_flat = loc_grad.flatten(1)
    cos_sim = F.cosine_similarity(cls_flat, loc_flat, dim=1)
    return cos_sim.pow(2).mean()


def build_dynamic_gdl_modules(cfg, backbone_shape):
    """Build dynamic GDL modules for RPN and RCNN branches.

    Returns dict of modules to be added to GeneralizedRCNN.
    """
    gdl_cfg = cfg.MODEL.DYNAMIC_GDL
    reduction = int(gdl_cfg.REDUCTION)
    rpn_init = float(gdl_cfg.RPN_INIT_LAMBDA)
    rcnn_init = float(gdl_cfg.RCNN_INIT_LAMBDA)

    rpn_features = cfg.MODEL.RPN.IN_FEATURES
    roi_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

    modules = {}

    # Dynamic GDL for RPN
    for feat_name in rpn_features:
        channels = backbone_shape[feat_name].channels
        modules[f"dynamic_gdl_rpn_{feat_name}"] = DynamicGDL(
            channels, reduction, rpn_init
        )

    # Dynamic GDL for RCNN
    for feat_name in roi_features:
        channels = backbone_shape[feat_name].channels
        modules[f"dynamic_gdl_rcnn_{feat_name}"] = DynamicGDL(
            channels, reduction, rcnn_init
        )

    # Optional dual pathway router
    if bool(gdl_cfg.DUAL_PATHWAY):
        for feat_name in roi_features:
            channels = backbone_shape[feat_name].channels
            modules[f"dual_pathway_{feat_name}"] = DualPathwayRouter(channels)

    return modules


def install_dynamic_gdl(model, cfg):
    """Install dynamic GDL modules into an existing GeneralizedRCNN.

    The dynamic modules work alongside the existing fixed GDL (decouple_layer).
    The fixed GDL provides stable base gradient scaling while the dynamic
    modules add a learned per-channel refinement on top.

    The model's _forward_once_ is patched to apply the DynamicGDL modules
    AFTER the standard decouple_layer + affine processing.

    Args:
        model: GeneralizedRCNN instance
        cfg: Config node

    Returns:
        Modified model
    """
    gdl_cfg = cfg.MODEL.DYNAMIC_GDL
    backbone_shape = model.backbone.output_shape()

    # Build and register dynamic modules, ensure they're on the model's device
    device = next(model.parameters()).device
    modules = build_dynamic_gdl_modules(cfg, backbone_shape)
    for name, module in modules.items():
        model.add_module(name, module.to(device))

    logger.info(
        "DP-GDL: Installed %d dynamic modules (reduction=%d, dual_pathway=%s)",
        len(modules),
        int(gdl_cfg.REDUCTION),
        bool(gdl_cfg.DUAL_PATHWAY),
    )

    # Store references for the patched forward
    model._dynamic_gdl_modules = modules
    model._dynamic_gdl_cfg = gdl_cfg

    # Monkey-patch _forward_once_ to apply dynamic GDL after standard processing
    original_forward = model._forward_once_

    def _forward_once_dynamic(batched_inputs, gt_instances=None):
        # Run the original forward (which applies fixed GDL + affine)
        # but intercept features after decouple to add dynamic refinement
        from defrcn.modeling.meta_arch.gdl import decouple_layer

        images = model.preprocess_image(batched_inputs)
        features = model.backbone(images.tensor)

        features_rpn = features
        features_rcnn = features
        if model.dual_fusion is not None:
            fused_rpn, fused_roi = model.dual_fusion(features)
            features_rpn = dict(features)
            features_rcnn = dict(features)
            features_rpn[model.fusion_out_feature] = fused_rpn
            features_rcnn[model.fusion_out_feature] = fused_roi
        if model.branch_adapter is not None:
            features_rpn, features_rcnn = model.branch_adapter(
                features_rpn, features_rcnn
            )

        # RPN: fixed GDL + dynamic refinement
        features_de_rpn = features_rpn
        if model.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = model.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {}
            for k, v in features_rpn.items():
                if k in model.rpn_in_features:
                    x = decouple_layer(v, scale)
                    if k == model.rpn_affine_feature:
                        x = model.affine_rpn(x)
                    # Apply dynamic GDL refinement
                    gdl_key = f"dynamic_gdl_rpn_{k}"
                    if gdl_key in model._dynamic_gdl_modules:
                        x = model._dynamic_gdl_modules[gdl_key](x)
                    features_de_rpn[k] = x
                else:
                    features_de_rpn[k] = v

        proposals, proposal_losses = model.proposal_generator(
            images, features_de_rpn, gt_instances
        )

        # RCNN: fixed GDL + dynamic refinement + optional dual pathway
        features_de_rcnn = features_rcnn
        if model.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = model.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {}
            for k, v in features_rcnn.items():
                if k in model.roi_in_features:
                    x = decouple_layer(v, scale)
                    if k == model.roi_affine_feature:
                        x = model.affine_rcnn(x)
                    # Apply dynamic GDL refinement
                    gdl_key = f"dynamic_gdl_rcnn_{k}"
                    if gdl_key in model._dynamic_gdl_modules:
                        x = model._dynamic_gdl_modules[gdl_key](x)
                    # Optional dual pathway routing
                    dp_key = f"dual_pathway_{k}"
                    if dp_key in model._dynamic_gdl_modules:
                        x = model._dynamic_gdl_modules[dp_key](x)
                    features_de_rcnn[k] = x
                else:
                    features_de_rcnn[k] = v

        results, detector_losses = model.roi_heads(
            images, features_de_rcnn, proposals, gt_instances
        )

        return proposal_losses, detector_losses, results, images.image_sizes

    model._forward_once_ = _forward_once_dynamic
    return model


def freeze_dynamic_gdl(model):
    """Freeze all dynamic GDL parameters for novel fine-tuning."""
    if not hasattr(model, "_dynamic_gdl_modules"):
        return

    frozen_count = 0
    for name, module in model._dynamic_gdl_modules.items():
        for p in module.parameters():
            p.requires_grad = False
            frozen_count += 1

    logger.info("DP-GDL: Froze %d parameters for novel fine-tuning", frozen_count)
