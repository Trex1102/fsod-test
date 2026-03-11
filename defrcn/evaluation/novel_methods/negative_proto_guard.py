"""
Negative Prototype Guard (NPG) for Few-Shot Object Detection.

Builds FM-space prototypes for base classes and suppresses detections
whose FM features are more similar to a base class prototype than to
any novel class prototype.  This addresses the dominant false-positive
failure mode in FSOD: base-class objects being misclassified as novel.

The guard wraps any PCB-like calibrator (vanilla PCB, PCB-FMA,
PCB-FMA-Patch, etc.) and applies confusion-based score suppression
*after* the inner calibrator has computed fused scores.

Architecture (post-calibration):
    max_novel_sim = max cosine similarity to novel FM prototypes
    max_base_sim  = max cosine similarity to base FM prototypes
    if max_base_sim > max_novel_sim + margin:
        score *= suppression_factor

Zero additional parameters.  Inference-time only.
"""

import cv2
import re
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NegativeProtoGuard:
    """Negative Prototype Guard that suppresses base-class false positives.

    Wraps any PCB calibrator.  Builds CLS prototypes for base classes
    in FM space and applies confusion-based score suppression.

    When stacked on top of PCB-FMA or PCB-FMA-Patch, the guard can
    share the DINOv2 model to avoid loading it twice.
    """

    def __init__(self, inner_pcb, cfg, fm_extractor=None):
        """
        Args:
            inner_pcb:    The wrapped calibrator (PrototypicalCalibrationBlock,
                          PCBFMA, PCBFMAPatch, …).
            cfg:          Full config node.
            fm_extractor: Optional shared FM extractor.  If *None* the guard
                          loads its own DINOv2 using NEG_PROTO_GUARD config.
        """
        self.inner_pcb = inner_pcb
        self.cfg = cfg
        npg_cfg = cfg.NOVEL_METHODS.NEG_PROTO_GUARD

        self.margin = float(npg_cfg.MARGIN)
        self.suppression_factor = float(npg_cfg.SUPPRESSION_FACTOR)
        self.max_base_samples = int(npg_cfg.MAX_BASE_SAMPLES_PER_CLASS)

        # ---- FM extractor (shared or own) ----
        if fm_extractor is not None and fm_extractor.available:
            self.fm_extractor = fm_extractor
            self._owns_extractor = False
            logger.info("NPG: Reusing FM extractor from parent wrapper.")
        else:
            from .pcb_fma import FoundationModelFeatureExtractor
            self.fm_extractor = FoundationModelFeatureExtractor(
                model_name=str(npg_cfg.FM_MODEL_NAME),
                model_path=str(npg_cfg.FM_MODEL_PATH),
                feat_dim=int(npg_cfg.FM_FEAT_DIM),
                roi_size=int(npg_cfg.ROI_SIZE),
                batch_size=int(npg_cfg.BATCH_SIZE),
                device=str(cfg.MODEL.DEVICE),
            )
            self._owns_extractor = True

        # ---- Resolve class sets ----
        self.base_cls_ids = self._get_base_class_ids()
        self.base_pcb_ref = self._get_base_pcb()

        # ---- Build prototypes ----
        self.base_prototypes: Dict[int, torch.Tensor] = {}
        self.novel_prototypes: Dict[int, torch.Tensor] = {}

        if self.fm_extractor.available and self.base_cls_ids:
            self._build_base_prototypes()
            self.novel_prototypes = self._resolve_novel_prototypes()
        else:
            reason = "FM not available" if not self.fm_extractor.available else "no base class IDs"
            logger.warning("NPG: %s — guard disabled.", reason)

    # ------------------------------------------------------------------
    # Helpers: navigate wrapper chain
    # ------------------------------------------------------------------

    def _get_base_pcb(self):
        """Walk the wrapper chain to the underlying PrototypicalCalibrationBlock."""
        pcb = self.inner_pcb
        while hasattr(pcb, "inner_pcb"):
            pcb = pcb.inner_pcb
        while hasattr(pcb, "base_pcb"):
            pcb = pcb.base_pcb
        return pcb

    def _get_base_class_ids(self) -> List[int]:
        """Derive base class IDs from dataset name.

        The original PCB.exclude_cls is only populated for 'test_all'
        datasets, so it's empty when testing on 'test_novel'.  We need
        base class IDs regardless, to build base FM prototypes for the
        confusion guard.  Derive them from the dataset name instead.
        """
        # First try exclude_cls (works for test_all datasets)
        pcb = self._get_base_pcb()
        if hasattr(pcb, "exclude_cls") and pcb.exclude_cls:
            return list(pcb.exclude_cls)

        # Derive from dataset name for test_novel datasets
        dsname = self.cfg.DATASETS.TEST[0]
        if "voc" in dsname:
            # VOC: base classes are always IDs 0-14 across all splits
            return list(range(0, 15))
        elif "coco" in dsname:
            # COCO: base class IDs (same as clsid_filter in calibration_layer)
            return [
                7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            ]
        return []

    # ------------------------------------------------------------------
    # FM feature extraction (handles both extractor types)
    # ------------------------------------------------------------------

    def _extract_cls_features(
        self, img: np.ndarray, boxes_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Extract CLS features, handling both extractor types.

        PatchFeatureExtractor.extract_roi_features → (patches, cls)
        FoundationModelFeatureExtractor.extract_roi_features → cls
        """
        result = self.fm_extractor.extract_roi_features(img, boxes_tensor)
        if result is None:
            return None
        if isinstance(result, tuple):
            return result[1]  # PatchFeatureExtractor → take CLS
        return result

    # ------------------------------------------------------------------
    # Prototype construction
    # ------------------------------------------------------------------

    def _derive_base_dataset_name(self) -> str:
        """Derive the base-class training dataset name from the novel dataset.

        Convention:
          Novel: voc_2007_trainval_novel{split}_Kshot_seedS
          Base:  voc_2007_trainval_base{split}
        """
        novel_ds = self.cfg.DATASETS.TRAIN[0]
        # VOC pattern
        m = re.match(r"(voc_\d+_trainval)_novel(\d+)_.*", novel_ds)
        if m:
            return f"{m.group(1)}_base{m.group(2)}"
        # COCO pattern
        m = re.match(r"(coco_trainval)_novel_.*", novel_ds)
        if m:
            return f"{m.group(1)}_base"
        return ""

    def _build_base_prototypes(self):
        """Build CLS prototypes for base classes from the base training dataset."""
        logger.info("NPG: Building base-class FM prototypes...")

        base_ds_name = self._derive_base_dataset_name()
        if not base_ds_name:
            logger.warning("NPG: Could not derive base dataset name from '%s'. Guard disabled.",
                           self.cfg.DATASETS.TRAIN[0])
            return

        try:
            from defrcn.dataloader import build_detection_test_loader
            base_loader = build_detection_test_loader(self.cfg, base_ds_name)
        except Exception as e:
            logger.warning("NPG: Could not load base dataset '%s': %s. Guard disabled.",
                           base_ds_name, e)
            return

        class_features: Dict[int, List[torch.Tensor]] = {}
        class_counts: Dict[int, int] = {}
        base_set = set(self.base_cls_ids)

        for index in range(len(base_loader.dataset)):
            # Early stop when every base class has enough samples
            if all(class_counts.get(c, 0) >= self.max_base_samples for c in base_set):
                break

            inputs = [base_loader.dataset[index]]
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

            # Keep only base boxes that still need more samples
            keep = torch.zeros(len(labels), dtype=torch.bool)
            for j in range(len(labels)):
                c = int(labels[j].item())
                if c in base_set and class_counts.get(c, 0) < self.max_base_samples:
                    keep[j] = True
            if not keep.any():
                continue

            keep_boxes = gt_boxes[keep]
            keep_labels = labels[keep]

            fm_feats = self._extract_cls_features(img, keep_boxes)
            if fm_feats is None:
                continue
            fm_feats = fm_feats.detach().cpu()

            for j in range(len(keep_labels)):
                c = int(keep_labels[j].item())
                if c not in class_features:
                    class_features[c] = []
                    class_counts[c] = 0
                class_features[c].append(fm_feats[j])
                class_counts[c] += 1

        # Aggregate mean prototypes
        for c, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.base_prototypes[c] = F.normalize(proto, dim=0)

        logger.info(
            "NPG: Built base prototypes for %d / %d base classes (samples: %s)",
            len(self.base_prototypes), len(base_set),
            {c: class_counts.get(c, 0) for c in sorted(self.base_prototypes.keys())},
        )

    def _resolve_novel_prototypes(self) -> Dict[int, torch.Tensor]:
        """Get novel-class CLS prototypes (reuse from inner PCB if available)."""
        pcb = self.inner_pcb
        # PCB-FMA stores them as fm_prototypes
        if hasattr(pcb, "fm_prototypes") and pcb.fm_prototypes:
            logger.info("NPG: Reusing %d novel FM prototypes from inner PCB (fm_prototypes).",
                        len(pcb.fm_prototypes))
            return pcb.fm_prototypes
        # PCB-FMA-Patch stores them as cls_prototypes
        if hasattr(pcb, "cls_prototypes") and pcb.cls_prototypes:
            logger.info("NPG: Reusing %d novel FM prototypes from inner PCB (cls_prototypes).",
                        len(pcb.cls_prototypes))
            return pcb.cls_prototypes
        # Fallback: build novel CLS prototypes ourselves
        return self._build_novel_fm_prototypes()

    def _build_novel_fm_prototypes(self) -> Dict[int, torch.Tensor]:
        """Build CLS prototypes for novel classes from the K-shot support set."""
        if not self.fm_extractor.available:
            return {}

        logger.info("NPG: Building novel-class FM prototypes from support set...")
        base_pcb = self.base_pcb_ref
        class_features: Dict[int, List[torch.Tensor]] = {}

        dataloader = base_pcb.dataloader
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

            fm_feats = self._extract_cls_features(img, gt_boxes)
            if fm_feats is None:
                continue
            fm_feats = fm_feats.detach().cpu()

            for j in range(len(labels)):
                c = int(labels[j].item())
                if c not in class_features:
                    class_features[c] = []
                class_features[c].append(fm_feats[j])

        prototypes = {}
        for c, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            prototypes[c] = F.normalize(proto, dim=0)

        logger.info("NPG: Built %d novel FM prototypes.", len(prototypes))
        return prototypes

    # ------------------------------------------------------------------
    # Calibration with guard
    # ------------------------------------------------------------------

    def execute_calibration(self, inputs, dts):
        """Run inner calibration, then apply negative prototype guard."""
        # Step 1: inner PCB calibration (tri-modal, patch-level, etc.)
        dts = self.inner_pcb.execute_calibration(inputs, dts)

        # Guard inactive → return as-is
        if not self.fm_extractor.available or not self.base_prototypes or not self.novel_prototypes:
            return dts

        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts

        instances = dts[0]["instances"]
        scores = instances.scores
        pred_classes = instances.pred_classes
        pred_boxes = instances.pred_boxes

        base_pcb = self.base_pcb_ref

        # Same score band as PCB calibration
        ileft = int((scores > base_pcb.pcb_upper).sum().item())
        iright = int((scores > base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        check_boxes = pred_boxes[ileft:iright]
        if len(check_boxes) == 0:
            return dts

        # Extract CLS features for candidate detections
        fm_feats = self._extract_cls_features(img, check_boxes.tensor)
        if fm_feats is None:
            return dts

        score_device = scores.device
        score_dtype = scores.dtype
        dev = self.fm_extractor.device if hasattr(self.fm_extractor, "device") else fm_feats.device

        suppressed = 0
        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in base_pcb.exclude_cls:
                continue

            q_idx = i - ileft
            q_feat = F.normalize(fm_feats[q_idx].to(dev), dim=0)

            # Max similarity to any novel prototype
            max_novel_sim = -1.0
            for nprot in self.novel_prototypes.values():
                sim = float(torch.dot(q_feat, nprot.to(dev)).item())
                if sim > max_novel_sim:
                    max_novel_sim = sim

            # Max similarity to any base prototype
            max_base_sim = -1.0
            for bprot in self.base_prototypes.values():
                sim = float(torch.dot(q_feat, bprot.to(dev)).item())
                if sim > max_base_sim:
                    max_base_sim = sim

            # Suppression trigger
            if max_base_sim > max_novel_sim + self.margin:
                old = float(scores[i].item())
                scores[i] = torch.tensor(
                    old * self.suppression_factor, device=score_device, dtype=score_dtype,
                )
                suppressed += 1

        if suppressed > 0:
            logger.debug(
                "NPG: Suppressed %d / %d detections (base-novel confusion guard).",
                suppressed, iright - ileft,
            )

        return dts

    def __getattr__(self, name):
        """Delegate to inner PCB for any non-overridden attributes."""
        return getattr(self.inner_pcb, name)


# ======================================================================
# Factory functions
# ======================================================================

def build_neg_proto_guard(base_pcb, cfg):
    """Wrap base PCB with Negative Prototype Guard (standalone)."""
    return NegativeProtoGuard(base_pcb, cfg)


def build_pcb_fma_patch_neg(base_pcb, cfg):
    """PCB-FMA-Patch + Negative Prototype Guard (combined).

    Builds PCB-FMA-Patch first (patch-level novel matching), then wraps
    it with NPG (base-class confusion guard).  The DINOv2 model is
    shared between the two wrappers to avoid loading it twice.
    """
    from .pcb_fma_patch import PCBFMAPatch

    # Step 1: patch-level FM matching (novel discrimination)
    patch_pcb = PCBFMAPatch(base_pcb, cfg)

    # Step 2: negative prototype guard (base confusion rejection)
    # Share the PatchFeatureExtractor — NPG only needs CLS tokens from it
    guard = NegativeProtoGuard(patch_pcb, cfg, fm_extractor=patch_pcb.fm_extractor)

    return guard


def build_pcb_fma_enhanced_neg(base_pcb, cfg):
    """PCB-FMA-Enhanced + Negative Prototype Guard (combined).

    Builds PCB-FMA-Enhanced first (augmented prototypes + competitive
    similarity for novel scoring), then wraps it with NPG (base-class
    confusion guard).  The DINOv2 model is shared between the two
    wrappers to avoid loading it twice.

    NPG also reuses the augmented novel prototypes from Enhanced
    (via fm_prototypes), so the guard benefits from better prototypes.
    """
    from .pcb_fma_enhanced import PCBFMAEnhanced

    # Step 1: enhanced FM scoring (augmented protos + competitive similarity)
    enhanced_pcb = PCBFMAEnhanced(base_pcb, cfg)

    # Step 2: negative prototype guard (base confusion rejection)
    # Share the FoundationModelFeatureExtractor to avoid loading DINOv2 twice
    guard = NegativeProtoGuard(enhanced_pcb, cfg, fm_extractor=enhanced_pcb.fm_extractor)

    return guard
