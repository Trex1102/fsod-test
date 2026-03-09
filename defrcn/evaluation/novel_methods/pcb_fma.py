"""
PCB with Foundation Model Alignment (PCB-FMA).

Replaces ImageNet ResNet-101 in PCB with a foundation model (DINOv2)
for better prototype features. The FM produces more generalizable features
that improve calibration, especially for underrepresented novel classes.

Key idea: Build prototypes in the FM feature space and compute calibration
scores there. The FM is frozen - zero additional parameters during novel
fine-tuning, making this purely an inference-time improvement.

Supports tri-modal fusion: det_score + FM_visual_sim + original_PCB_sim.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FoundationModelFeatureExtractor:
    """Extracts per-RoI features using a foundation model (DINOv2).

    Features are extracted by cropping each RoI from the image,
    resizing to the FM's input size, and taking the CLS token.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        model_path: str = "",
        feat_dim: int = 768,
        roi_size: int = 224,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.feat_dim = feat_dim
        self.roi_size = roi_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = None

        self._load_model(model_name, model_path)

    @staticmethod
    def _patch_dinov2_for_py38():
        """Patch cached DINOv2 source files for Python < 3.10 compatibility.

        DINOv2 uses PEP 604 union syntax (float | None) which requires Python 3.10+.
        This patches the cached files to use Optional[float] instead.
        """
        import sys
        if sys.version_info >= (3, 10):
            return

        import glob
        import os
        hub_dir = torch.hub.get_dir()
        dinov2_dirs = glob.glob(os.path.join(hub_dir, "facebookresearch_dinov2*"))
        for d in dinov2_dirs:
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as f:
                            content = f.read()
                        if "float | None" in content or "int | None" in content or "str | None" in content:
                            # Add from __future__ import annotations at top if not present
                            if "from __future__ import annotations" not in content:
                                content = "from __future__ import annotations\n" + content
                                with open(fpath, "w") as f:
                                    f.write(content)
                                logger.debug("Patched %s for Python 3.8 compatibility", fpath)
                    except Exception:
                        pass

    @staticmethod
    def _patch_dinov2_for_py38():
        """Patch cached DINOv2 source files for Python < 3.10 compatibility.

        DINOv2 uses PEP 604 union syntax (float | None) which requires Python 3.10+.
        This patches the cached files to use Optional[float] instead.
        """
        import sys
        if sys.version_info >= (3, 10):
            return

        import glob
        import os
        hub_dir = torch.hub.get_dir()
        dinov2_dirs = glob.glob(os.path.join(hub_dir, "facebookresearch_dinov2*"))
        for d in dinov2_dirs:
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as f:
                            content = f.read()
                        if "float | None" in content or "int | None" in content or "str | None" in content:
                            # Add from __future__ import annotations at top if not present
                            if "from __future__ import annotations" not in content:
                                content = "from __future__ import annotations\n" + content
                                with open(fpath, "w") as f:
                                    f.write(content)
                                logger.info("Patched %s for Python 3.8 compatibility", fpath)
                    except Exception:
                        pass

    def _load_model(self, model_name: str, model_path: str):
        """Load the foundation model."""
        try:
            if model_path:
                logger.info("Loading foundation model from local path: %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("Loading foundation model via torch.hub: %s", model_name)
                # Patch cached DINOv2 source for Python < 3.10 compatibility
                self._patch_dinov2_for_py38()
                try:
                    self.model = torch.hub.load(
                        "facebookresearch/dinov2",
                        model_name,
                        pretrained=True,
                    )
                except TypeError as e:
                    if "unsupported operand type" in str(e):
                        # First load downloaded repo but failed on PEP 604 syntax.
                        # Patch and retry.
                        logger.info("Patching DINOv2 for Python <3.10 compatibility and retrying...")
                        self._patch_dinov2_for_py38()
                        # Clear cached modules so patched files are re-imported
                        import sys as _sys
                        mods_to_remove = [k for k in _sys.modules if "dinov2" in k]
                        for k in mods_to_remove:
                            del _sys.modules[k]
                        self.model = torch.hub.load(
                            "facebookresearch/dinov2",
                            model_name,
                            pretrained=True,
                        )
                    else:
                        raise

            self.model = self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

            # Verify feature dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, self.roi_size, self.roi_size, device=self.device)
                out = self.model(dummy)
                actual_dim = out.shape[-1]
                if actual_dim != self.feat_dim:
                    logger.warning(
                        "FM feature dim mismatch: config says %d, actual is %d. Using actual.",
                        self.feat_dim, actual_dim,
                    )
                    self.feat_dim = actual_dim

            logger.info("Foundation model loaded successfully: %s (dim=%d)", model_name, self.feat_dim)

        except Exception as e:
            logger.warning("Failed to load foundation model '%s': %s. PCB-FMA will fall back to base PCB.", model_name, e)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess cropped images for the foundation model.

        DINOv2 expects ImageNet-normalized RGB tensors.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        tensors = []
        for crop in crops:
            # Resize to model input size
            resized = cv2.resize(crop, (self.roi_size, self.roi_size))
            # BGR -> RGB, HWC -> CHW, uint8 -> float32
            t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)
        batch = (batch - mean) / std
        return batch

    def extract_roi_features(
        self, img: np.ndarray, boxes_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Extract FM features for each RoI box.

        Args:
            img: BGR image (H, W, 3) as numpy array
            boxes_tensor: (N, 4) tensor of [x1, y1, x2, y2] boxes

        Returns:
            (N, feat_dim) tensor of FM features
        """
        if not self.available:
            return None

        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.feat_dim), device=self.device)

        boxes = boxes_tensor.cpu().numpy()
        img_h, img_w = img.shape[:2]

        # Crop each RoI
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                # Degenerate box: use 1x1 patch
                crops.append(img[y1:y1 + 1, x1:x1 + 1])
            else:
                crops.append(img[y1:y2, x1:x2])

        # Batch forward passes for efficiency
        all_features = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch_crops = crops[start : start + self.batch_size]
                batch_tensor = self._preprocess_crops(batch_crops)
                features = self.model(batch_tensor)  # (B, feat_dim) CLS token
                all_features.append(features)

        return torch.cat(all_features, dim=0)


class PCBFMA:
    """PCB with Foundation Model Alignment.

    Wraps the base PrototypicalCalibrationBlock and augments it with
    foundation model features for better prototype matching.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        fma_cfg = cfg.NOVEL_METHODS.PCB_FMA

        self.det_weight = float(fma_cfg.DET_WEIGHT)
        self.fm_weight = float(fma_cfg.FM_WEIGHT)
        self.use_original_pcb = bool(fma_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(fma_cfg.ORIGINAL_PCB_WEIGHT)

        # Normalize weights
        if self.use_original_pcb:
            total = self.det_weight + self.fm_weight + self.original_pcb_weight
        else:
            total = self.det_weight + self.fm_weight
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        # Load foundation model
        self.fm_extractor = FoundationModelFeatureExtractor(
            model_name=str(fma_cfg.FM_MODEL_NAME),
            model_path=str(fma_cfg.FM_MODEL_PATH),
            feat_dim=int(fma_cfg.FM_FEAT_DIM),
            roi_size=int(fma_cfg.ROI_SIZE),
            batch_size=int(fma_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        # Build FM-space prototypes from support set
        self.fm_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_fm_prototypes()
        else:
            logger.warning("PCB-FMA: FM not available, falling back to base PCB behavior.")

    def _build_fm_prototypes(self):
        """Build per-class prototypes in FM feature space from support set."""
        logger.info("PCB-FMA: Building FM-space prototypes from support set...")

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

            # Extract FM features for GT boxes
            fm_features = self.fm_extractor.extract_roi_features(img, gt_boxes)
            if fm_features is None:
                continue

            fm_features = fm_features.detach().cpu()
            for i in range(len(labels)):
                cls = int(labels[i].item())
                if cls not in class_features:
                    class_features[cls] = []
                class_features[cls].append(fm_features[i])

        # Aggregate per-class prototypes (simple mean)
        for cls, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.fm_prototypes[cls] = F.normalize(proto, dim=0)

        logger.info(
            "PCB-FMA: Built FM prototypes for %d classes (dim=%d)",
            len(self.fm_prototypes),
            self.fm_extractor.feat_dim,
        )

    def execute_calibration(self, inputs, dts):
        """Execute calibration with foundation model alignment."""
        # If FM not available, fall back to base PCB
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

        # Extract FM features for candidate RoIs
        box_tensor = pred_boxes.tensor
        fm_features = self.fm_extractor.extract_roi_features(img, box_tensor)

        # Also get base PCB features for original PCB score
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

            # FM visual similarity
            fm_sim = 0.0
            if cls in self.fm_prototypes and fm_features is not None:
                query_fm = F.normalize(fm_features[q_idx], dim=0)
                proto_fm = self.fm_prototypes[cls].to(query_fm.device)
                fm_sim = float(torch.dot(query_fm, proto_fm).item())
                # Map from [-1, 1] to [0, 1]
                fm_sim = (fm_sim + 1.0) / 2.0

            # Original PCB similarity (optional)
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


def build_pcb_fma(base_pcb, cfg):
    """Factory function to wrap PCB with Foundation Model Alignment."""
    return PCBFMA(base_pcb, cfg)
