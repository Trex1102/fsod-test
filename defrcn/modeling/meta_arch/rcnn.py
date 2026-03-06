import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from .dual_fusion import DualFusionNeck
from .branch_adapter import BranchAdapter
from defrcn.modeling.roi_heads import build_roi_heads

__all__ = ["GeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.rpn_in_features = tuple(cfg.MODEL.RPN.IN_FEATURES)
        self.roi_in_features = tuple(cfg.MODEL.ROI_HEADS.IN_FEATURES)
        if len(self.rpn_in_features) == 0 or len(self.roi_in_features) == 0:
            raise ValueError("MODEL.RPN.IN_FEATURES and MODEL.ROI_HEADS.IN_FEATURES must be non-empty.")
        self.rpn_affine_feature = self.rpn_in_features[0]
        self.roi_affine_feature = self.roi_in_features[0]
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_[self.rpn_affine_feature].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_[self.roi_affine_feature].channels, bias=True)
        self.dual_fusion = None
        self.fusion_out_feature = None
        if cfg.MODEL.DUAL_FUSION.ENABLE:
            self.dual_fusion = DualFusionNeck(cfg, self._SHAPE_)
            self.fusion_out_feature = cfg.MODEL.DUAL_FUSION.OUT_FEATURE
            if self.fusion_out_feature not in self.rpn_in_features:
                raise ValueError(
                    "MODEL.DUAL_FUSION.OUT_FEATURE must be in MODEL.RPN.IN_FEATURES, got '{}'.".format(
                        self.fusion_out_feature
                    )
                )
            if self.fusion_out_feature not in self.roi_in_features:
                raise ValueError(
                    "MODEL.DUAL_FUSION.OUT_FEATURE must be in MODEL.ROI_HEADS.IN_FEATURES, got '{}'.".format(
                        self.fusion_out_feature
                    )
                )
        self.branch_adapter = None
        if cfg.MODEL.BRANCH_ADAPTER.ENABLE:
            self.branch_adapter = BranchAdapter(cfg, self._SHAPE_)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            if cfg.MODEL.ROI_HEADS.RES5_ADAPTER.ENABLE:
                # Unfreeze only the adapter parameters — original res5 blocks stay frozen.
                for p in self.roi_heads.res5.adapter_parameters():
                    p.requires_grad = True
                print("froze res5 block parameters (adapters remain trainable)")
            else:
                print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features_rpn = features
        features_rcnn = features
        if self.dual_fusion is not None:
            fused_rpn, fused_roi = self.dual_fusion(features)
            features_rpn = dict(features)
            features_rcnn = dict(features)
            features_rpn[self.fusion_out_feature] = fused_rpn
            features_rcnn[self.fusion_out_feature] = fused_roi
        if self.branch_adapter is not None:
            features_rpn, features_rcnn = self.branch_adapter(features_rpn, features_rcnn)

        features_de_rpn = features_rpn
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {}
            for k, v in features_rpn.items():
                if k in self.rpn_in_features:
                    x = decouple_layer(v, scale)
                    if k == self.rpn_affine_feature:
                        x = self.affine_rpn(x)
                    features_de_rpn[k] = x
                else:
                    features_de_rpn[k] = v
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features_rcnn
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {}
            for k, v in features_rcnn.items():
                if k in self.roi_in_features:
                    x = decouple_layer(v, scale)
                    if k == self.roi_affine_feature:
                        x = self.affine_rcnn(x)
                    features_de_rcnn[k] = x
                else:
                    features_de_rcnn[k] = v
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
