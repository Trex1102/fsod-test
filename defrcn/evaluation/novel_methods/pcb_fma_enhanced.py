"""
PCB-FMA Enhanced: Support Augmentation + Temperature-Scaled Class-Competitive Similarity.

Wraps the base PCB (like vanilla PCB-FMA does) but with two key improvements:

1. Support Augmentation: Extracts DINOv2 CLS features from multiple augmented
   views of each support crop (original, horizontal flip, multi-scale center
   crops) and averages them.  This yields more robust prototypes, especially
   at K=1 where a single viewpoint is noisy.

2. Temperature-Scaled Class-Competitive Similarity: Instead of computing
   cosine similarity independently per class and mapping to [0,1], computes
   similarities to ALL novel prototypes simultaneously and applies
   softmax(sims / temperature).  This forces inter-class competition so that
   ambiguous features get low scores while clear matches get amplified.

This is a standalone wrapper -- it does NOT modify pcb_fma.py.
Inference-only, zero additional parameters at training time.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PCBFMAEnhanced:
    """PCB-FMA with Support Augmentation and Class-Competitive Similarity.

    Wraps the base PrototypicalCalibrationBlock and augments it with
    DINOv2 foundation model features, using augmented prototype building
    and class-competitive similarity scoring.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        enh_cfg = cfg.NOVEL_METHODS.PCB_FMA_ENHANCED

        self.det_weight = float(enh_cfg.DET_WEIGHT)
        self.fm_weight = float(enh_cfg.FM_WEIGHT)
        self.use_original_pcb = bool(enh_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(enh_cfg.ORIGINAL_PCB_WEIGHT)

        # Support augmentation config
        self.enable_flip = bool(enh_cfg.AUG_FLIP)
        self.enable_multicrop = bool(enh_cfg.AUG_MULTICROP)
        self.multicrop_scales = list(enh_cfg.AUG_MULTICROP_SCALES)
        self.multicrop_num = int(enh_cfg.AUG_MULTICROP_NUM)

        # Temperature-scaled class-competitive similarity
        self.temperature = float(enh_cfg.TEMPERATURE)
        self.competitive_mode = str(enh_cfg.COMPETITIVE_MODE).lower()

        # Normalize fusion weights
        if self.use_original_pcb:
            total = self.det_weight + self.fm_weight + self.original_pcb_weight
        else:
            total = self.det_weight + self.fm_weight
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        # Load foundation model (reuse FoundationModelFeatureExtractor)
        from .pcb_fma import FoundationModelFeatureExtractor

        self.fm_extractor = FoundationModelFeatureExtractor(
            model_name=str(enh_cfg.FM_MODEL_NAME),
            model_path=str(enh_cfg.FM_MODEL_PATH),
            feat_dim=int(enh_cfg.FM_FEAT_DIM),
            roi_size=int(enh_cfg.ROI_SIZE),
            batch_size=int(enh_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        # Build augmented FM prototypes from support set
        self.fm_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_augmented_prototypes()
        else:
            logger.warning(
                "PCB-FMA-Enhanced: FM not available, falling back to base PCB."
            )

    # ------------------------------------------------------------------
    # Support augmentation helpers
    # ------------------------------------------------------------------

    def _augment_crop(self, crop: np.ndarray) -> List[np.ndarray]:
        """Generate augmented views of a single RoI crop.

        Returns a list of numpy BGR images (not yet preprocessed).
        """
        views = [crop]  # original always included

        if self.enable_flip:
            views.append(cv2.flip(crop, 1))  # horizontal flip

        if self.enable_multicrop:
            h, w = crop.shape[:2]
            for scale in self.multicrop_scales:
                ch, cw = int(h * scale), int(w * scale)
                if ch < 2 or cw < 2:
                    continue
                # Center crop at this scale
                y0 = (h - ch) // 2
                x0 = (w - cw) // 2
                views.append(crop[y0 : y0 + ch, x0 : x0 + cw])
                # Corner crops for additional diversity
                if self.multicrop_num > 1:
                    views.append(crop[0:ch, 0:cw])  # top-left
                if self.multicrop_num > 2:
                    views.append(
                        crop[max(0, h - ch) : h, max(0, w - cw) : w]
                    )  # bottom-right

        return views

    def _extract_augmented_features_for_box(
        self, img: np.ndarray, box: np.ndarray
    ) -> Optional[torch.Tensor]:
        """Extract a single augmented feature vector for one GT box.

        Generates augmented views of the crop, batches them through
        DINOv2, and averages the CLS tokens.

        Args:
            img: BGR image (H, W, 3)
            box: [x1, y1, x2, y2] numpy array

        Returns:
            (feat_dim,) tensor or None
        """
        img_h, img_w = img.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2]))
        y2 = min(img_h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            crop = img[y1 : y1 + 1, x1 : x1 + 1]
        else:
            crop = img[y1:y2, x1:x2]

        views = self._augment_crop(crop)

        # Batch all views through DINOv2 in one forward pass
        batch_tensor = self.fm_extractor._preprocess_crops(views)
        with torch.no_grad():
            feats = self.fm_extractor.model(batch_tensor)  # (num_views, dim)

        # Average across augmented views
        return feats.mean(dim=0)

    # ------------------------------------------------------------------
    # Prototype building (augmented)
    # ------------------------------------------------------------------

    def _build_augmented_prototypes(self):
        """Build per-class FM prototypes using augmented support features."""
        logger.info("PCB-FMA-Enhanced: Building augmented FM prototypes...")

        class_features: Dict[int, List[torch.Tensor]] = {}

        dataloader = self.base_pcb.dataloader
        for index in range(len(dataloader.dataset)):
            inputs = [dataloader.dataset[index]]
            img = cv2.imread(inputs[0]["file_name"])
            if img is None:
                continue

            inst = inputs[0]["instances"]
            if len(inst) == 0:
                continue

            img_h, img_w = img.shape[:2]
            ratio = img_h / float(inst.image_size[0])
            gt_boxes = inst.gt_boxes.tensor.clone() * ratio
            labels = inst.gt_classes.cpu()

            boxes_np = gt_boxes.cpu().numpy()
            for i in range(len(labels)):
                aug_feat = self._extract_augmented_features_for_box(
                    img, boxes_np[i]
                )
                if aug_feat is None:
                    continue
                cls = int(labels[i].item())
                if cls not in class_features:
                    class_features[cls] = []
                class_features[cls].append(aug_feat.detach().cpu())

        # Aggregate per-class prototypes (mean, then L2-normalize)
        for cls, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.fm_prototypes[cls] = F.normalize(proto, dim=0)

        num_augs = 1 + int(self.enable_flip)
        if self.enable_multicrop:
            for _ in self.multicrop_scales:
                num_augs += 1 + max(0, self.multicrop_num - 1)
        logger.info(
            "PCB-FMA-Enhanced: Built augmented FM prototypes for %d classes "
            "(dim=%d, ~%d views/crop)",
            len(self.fm_prototypes),
            self.fm_extractor.feat_dim,
            num_augs,
        )

    # ------------------------------------------------------------------
    # Class-competitive similarity
    # ------------------------------------------------------------------

    def _compute_competitive_similarity(
        self, query_fm: torch.Tensor, cls: int
    ) -> float:
        """Compute class-competitive FM similarity.

        Instead of independent cosine mapped to [0,1], computes softmax
        over cosine similarities to ALL novel prototypes scaled by
        temperature.  Returns the softmax score for the target class.

        Args:
            query_fm: L2-normalized query feature (feat_dim,)
            cls: target class ID

        Returns:
            Similarity score in [0, 1]
        """
        if cls not in self.fm_prototypes:
            return 0.0

        all_cls_ids = sorted(self.fm_prototypes.keys())

        # Single class: fall back to independent cosine
        if len(all_cls_ids) <= 1:
            proto = self.fm_prototypes[cls].to(query_fm.device)
            sim = float(torch.dot(query_fm, proto).item())
            return (sim + 1.0) / 2.0

        # Compute cosine similarities to all novel prototypes
        sims = []
        target_idx = -1
        for idx, c in enumerate(all_cls_ids):
            proto = self.fm_prototypes[c].to(query_fm.device)
            sims.append(torch.dot(query_fm, proto))
            if c == cls:
                target_idx = idx

        if target_idx < 0:
            return 0.0

        sims_tensor = torch.stack(sims, dim=0)  # (num_classes,)

        if self.competitive_mode == "softmax":
            temp = max(self.temperature, 1e-6)
            competitive = F.softmax(sims_tensor / temp, dim=0)
            return float(competitive[target_idx].item())
        else:
            # "raw" mode: standard cosine mapped to [0,1] (same as vanilla FMA)
            return (float(sims_tensor[target_idx].item()) + 1.0) / 2.0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def execute_calibration(self, inputs, dts):
        """Execute calibration with augmented prototypes and competitive similarity."""
        # Fallback to base PCB if FM unavailable
        if not self.fm_extractor.available or not self.fm_prototypes:
            return self.base_pcb.execute_calibration(inputs, dts)

        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts

        scores = dts[0]["instances"].scores
        ileft = int((scores > self.base_pcb.pcb_upper).sum().item())
        iright = int((scores > self.base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = dts[0]["instances"].pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        pred_classes = dts[0]["instances"].pred_classes
        score_device = scores.device
        score_dtype = scores.dtype

        # Extract FM features for candidate RoIs (single view -- no augmentation at test time)
        box_tensor = pred_boxes.tensor
        fm_features = self.fm_extractor.extract_roi_features(img, box_tensor)

        # Also compute base PCB features for tri-modal fusion
        if self.use_original_pcb:
            img_h, img_w = img.shape[:2]
            boxes = [pred_boxes.to(self.base_pcb.device)]
            pcb_features = self.base_pcb.extract_roi_features(img, boxes)
            area_norm = self.base_pcb._normalized_area(box_tensor, img_h, img_w)

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb.exclude_cls:
                continue

            q_idx = i - ileft
            det_score = float(scores[i].item())

            # FM class-competitive similarity
            fm_sim = 0.0
            if cls in self.fm_prototypes and fm_features is not None:
                query_fm = F.normalize(fm_features[q_idx], dim=0)
                fm_sim = self._compute_competitive_similarity(query_fm, cls)

            # Original PCB similarity (optional tri-modal)
            pcb_sim = 0.0
            if self.use_original_pcb and cls in self.base_pcb.prototypes:
                proto_bank = self.base_pcb._select_proto_bank(
                    cls, float(area_norm[q_idx].item())
                )
                if proto_bank is not None:
                    pcb_sim = self.base_pcb._match_similarity(
                        pcb_features[q_idx], proto_bank
                    )
                    pcb_sim = (pcb_sim + 1.0) / 2.0

            # Tri-modal fusion
            if self.use_original_pcb:
                fused = (
                    self.det_weight * det_score
                    + self.fm_weight * fm_sim
                    + self.original_pcb_weight * pcb_sim
                )
            else:
                fused = self.det_weight * det_score + self.fm_weight * fm_sim

            fused = max(0.0, min(1.0, fused))
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_pcb_fma_enhanced(base_pcb, cfg):
    """Factory function to wrap PCB with enhanced FMA (Support Aug + Competitive Sim)."""
    return PCBFMAEnhanced(base_pcb, cfg)
