"""
PCB-FMA with Patch-Level Local Feature Matching (PCB-FMA-Patch).

Extends PCB-FMA by using DINOv2 patch tokens for dense local matching
instead of CLS-only global matching. For each query RoI, similarity is
computed by matching each query patch to its nearest support patch
(DN4-style max-pool matching), capturing part-level discriminative cues.

This addresses a key limitation of CLS-based matching: global CLS tokens
collapse intra-class variation (e.g., a sofa with/without a person produces
very different CLS tokens but shares distinctive local patches like armrests
and cushion textures).

Architecture:
  fused_score = w_d * det_score + w_fm * patch_local_sim + w_pcb * pcb_sim

Zero additional parameters. Inference-time only.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PatchFeatureExtractor:
    """Extracts per-RoI patch tokens and CLS token using DINOv2.

    Unlike FoundationModelFeatureExtractor (CLS-only), this returns both
    the CLS token and all spatial patch tokens for local matching.
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
        self.num_patches = (roi_size // 14) ** 2  # ViT patch_size=14

        self._load_model(model_name, model_path)

    @staticmethod
    def _patch_dinov2_for_py38():
        """Patch cached DINOv2 source files for Python < 3.10 compatibility.

        DINOv2 uses PEP 604 union syntax (float | None) which requires
        Python 3.10+. This patches cached files to add
        ``from __future__ import annotations``.
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
                            if "from __future__ import annotations" not in content:
                                content = "from __future__ import annotations\n" + content
                                with open(fpath, "w") as f:
                                    f.write(content)
                                logger.debug("Patched %s for Python 3.8 compat", fpath)
                    except Exception:
                        pass

    def _load_model(self, model_name: str, model_path: str):
        """Load the DINOv2 model."""
        try:
            if model_path:
                logger.info("Loading patch FM from local path: %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("Loading patch FM via torch.hub: %s", model_name)
                self._patch_dinov2_for_py38()
                try:
                    self.model = torch.hub.load(
                        "facebookresearch/dinov2", model_name, pretrained=True,
                    )
                except TypeError as e:
                    if "unsupported operand type" in str(e):
                        self._patch_dinov2_for_py38()
                        import sys as _sys
                        mods_to_remove = [k for k in _sys.modules if "dinov2" in k]
                        for k in mods_to_remove:
                            del _sys.modules[k]
                        self.model = torch.hub.load(
                            "facebookresearch/dinov2", model_name, pretrained=True,
                        )
                    else:
                        raise

            self.model = self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

            # Verify dimensions using get_intermediate_layers
            with torch.no_grad():
                dummy = torch.randn(1, 3, self.roi_size, self.roi_size, device=self.device)
                out = self.model.get_intermediate_layers(
                    dummy, n=1, return_class_token=True,
                )
                patch_tokens, cls_token = out[0]
                actual_dim = cls_token.shape[-1]
                actual_patches = patch_tokens.shape[1]
                if actual_dim != self.feat_dim:
                    logger.warning(
                        "Patch FM dim mismatch: config %d, actual %d. Using actual.",
                        self.feat_dim, actual_dim,
                    )
                    self.feat_dim = actual_dim
                self.num_patches = actual_patches

            logger.info(
                "Patch FM loaded: %s (dim=%d, num_patches=%d)",
                model_name, self.feat_dim, self.num_patches,
            )

        except Exception as e:
            logger.warning(
                "Failed to load patch FM '%s': %s. Will fall back to base PCB.",
                model_name, e,
            )
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess cropped RoI images for DINOv2 (ImageNet normalisation)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (self.roi_size, self.roi_size))
            # BGR -> RGB, HWC -> CHW, uint8 -> float32
            t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)
        batch = (batch - mean) / std
        return batch

    def _crop_rois(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> List[np.ndarray]:
        """Crop RoIs from the image."""
        boxes = boxes_tensor.cpu().numpy()
        img_h, img_w = img.shape[:2]
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(img[y1:y1 + 1, x1:x1 + 1])
            else:
                crops.append(img[y1:y2, x1:x2])
        return crops

    def extract_roi_features(
        self, img: np.ndarray, boxes_tensor: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract patch tokens and CLS token for each RoI box.

        Args:
            img: BGR image (H, W, 3) as numpy array.
            boxes_tensor: (N, 4) tensor of [x1, y1, x2, y2] boxes.

        Returns:
            patch_tokens: (N, num_patches, feat_dim) or None on failure.
            cls_tokens:   (N, feat_dim) or None on failure.
        """
        if not self.available:
            return None, None

        if boxes_tensor.numel() == 0:
            return (
                torch.empty((0, self.num_patches, self.feat_dim), device=self.device),
                torch.empty((0, self.feat_dim), device=self.device),
            )

        crops = self._crop_rois(img, boxes_tensor)

        all_patches = []
        all_cls = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch_crops = crops[start:start + self.batch_size]
                batch_tensor = self._preprocess_crops(batch_crops)
                out = self.model.get_intermediate_layers(
                    batch_tensor, n=1, return_class_token=True,
                )
                patch_tokens, cls_token = out[0]
                all_patches.append(patch_tokens)
                all_cls.append(cls_token)

        return torch.cat(all_patches, dim=0), torch.cat(all_cls, dim=0)


class PCBFMAPatch:
    """PCB-FMA with patch-level local feature matching.

    Replaces CLS-only FM matching with dense patch-level matching
    for more discriminative prototype comparison.  Stores per-class
    support patch tokens and computes DN4-style max-pool local
    similarity during calibration.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        fma_cfg = cfg.NOVEL_METHODS.PCB_FMA_PATCH

        self.det_weight = float(fma_cfg.DET_WEIGHT)
        self.fm_weight = float(fma_cfg.FM_WEIGHT)
        self.use_original_pcb = bool(fma_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(fma_cfg.ORIGINAL_PCB_WEIGHT)
        self.top_k_patches = int(fma_cfg.TOP_K_PATCHES)
        self.bidirectional = bool(fma_cfg.BIDIRECTIONAL)
        self.cls_weight = float(fma_cfg.CLS_WEIGHT)

        # Normalise fusion weights
        if self.use_original_pcb:
            total = self.det_weight + self.fm_weight + self.original_pcb_weight
        else:
            total = self.det_weight + self.fm_weight
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        # Load foundation model (patch-level)
        self.fm_extractor = PatchFeatureExtractor(
            model_name=str(fma_cfg.FM_MODEL_NAME),
            model_path=str(fma_cfg.FM_MODEL_PATH),
            feat_dim=int(fma_cfg.FM_FEAT_DIM),
            roi_size=int(fma_cfg.ROI_SIZE),
            batch_size=int(fma_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        # Per-class prototypes in FM space
        self.patch_prototypes: Dict[int, torch.Tensor] = {}   # cls -> (K*N_patches, D) normalised
        self.cls_prototypes: Dict[int, torch.Tensor] = {}     # cls -> (D,) normalised

        if self.fm_extractor.available:
            self._build_prototypes()
        else:
            logger.warning("PCB-FMA-Patch: FM not available, falling back to base PCB.")

    # ------------------------------------------------------------------
    # Prototype construction
    # ------------------------------------------------------------------

    def _build_prototypes(self):
        """Build per-class patch-level and CLS prototypes from the support set."""
        logger.info("PCB-FMA-Patch: Building patch-level FM prototypes from support set...")

        class_patches: Dict[int, List[torch.Tensor]] = {}
        class_cls: Dict[int, List[torch.Tensor]] = {}

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

            patch_tokens, cls_tokens = self.fm_extractor.extract_roi_features(img, gt_boxes)
            if patch_tokens is None:
                continue

            patch_tokens = patch_tokens.detach().cpu()
            cls_tokens = cls_tokens.detach().cpu()

            for i in range(len(labels)):
                cls = int(labels[i].item())
                if cls not in class_patches:
                    class_patches[cls] = []
                    class_cls[cls] = []
                class_patches[cls].append(patch_tokens[i])   # (N_patches, D)
                class_cls[cls].append(cls_tokens[i])          # (D,)

        for cls in class_patches:
            # Patch prototype: concatenate all support patches, L2-normalise
            all_patches = torch.cat(class_patches[cls], dim=0)  # (K*N_patches, D)
            self.patch_prototypes[cls] = F.normalize(all_patches, dim=1)

            # CLS prototype: mean of CLS tokens, L2-normalise
            cls_proto = torch.stack(class_cls[cls], dim=0).mean(dim=0)
            self.cls_prototypes[cls] = F.normalize(cls_proto, dim=0)

        logger.info(
            "PCB-FMA-Patch: Built prototypes for %d classes (dim=%d, patches/class: %s)",
            len(self.patch_prototypes),
            self.fm_extractor.feat_dim,
            {c: int(p.shape[0]) for c, p in self.patch_prototypes.items()},
        )

    # ------------------------------------------------------------------
    # Local matching
    # ------------------------------------------------------------------

    def _local_match_score(
        self, query_patches: torch.Tensor, support_patches: torch.Tensor,
    ) -> float:
        """Compute DN4-style local matching score.

        For each query patch, find the most similar support patch (max-pool).
        Return the mean of these best-match cosine similarities.

        Args:
            query_patches:   (N_q, D) L2-normalised query patch tokens.
            support_patches: (N_s, D) L2-normalised support patch bank.

        Returns:
            Scalar similarity in [-1, 1].
        """
        sim_matrix = query_patches @ support_patches.T  # (N_q, N_s)

        if self.top_k_patches > 0 and self.top_k_patches < sim_matrix.shape[1]:
            topk_vals, _ = sim_matrix.topk(self.top_k_patches, dim=1)
            q2s = topk_vals.mean()
        else:
            q2s = sim_matrix.max(dim=1).values.mean()

        if self.bidirectional:
            if self.top_k_patches > 0 and self.top_k_patches < sim_matrix.shape[0]:
                topk_vals, _ = sim_matrix.topk(self.top_k_patches, dim=0)
                s2q = topk_vals.mean()
            else:
                s2q = sim_matrix.max(dim=0).values.mean()
            return float(((q2s + s2q) / 2.0).item())

        return float(q2s.item())

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def execute_calibration(self, inputs, dts):
        """Execute calibration with patch-level FM matching."""
        if not self.fm_extractor.available or not self.patch_prototypes:
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

        # Extract patch + CLS tokens for candidate RoIs
        box_tensor = pred_boxes.tensor
        query_patches, query_cls = self.fm_extractor.extract_roi_features(img, box_tensor)

        # Base PCB features (for original PCB similarity channel)
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

            # ----- FM patch-level similarity -----
            fm_sim = 0.0
            if cls in self.patch_prototypes and query_patches is not None:
                dev = self.fm_extractor.device
                q_p = F.normalize(query_patches[q_idx].to(dev), dim=1)  # (N_patches, D)
                s_p = self.patch_prototypes[cls].to(dev)

                patch_sim = self._local_match_score(q_p, s_p)

                # Optional CLS blending
                if self.cls_weight > 0 and cls in self.cls_prototypes:
                    q_c = F.normalize(query_cls[q_idx].to(dev), dim=0)
                    s_c = self.cls_prototypes[cls].to(dev)
                    cls_sim = float(torch.dot(q_c, s_c).item())
                    fm_sim = (1.0 - self.cls_weight) * patch_sim + self.cls_weight * cls_sim
                else:
                    fm_sim = patch_sim

                # Map cosine [-1, 1] -> [0, 1]
                fm_sim = (fm_sim + 1.0) / 2.0

            # ----- Original PCB similarity (optional) -----
            pcb_sim = 0.0
            if self.use_original_pcb and cls in self.base_pcb.prototypes:
                proto_bank = self.base_pcb._select_proto_bank(
                    cls, float(area_norm[q_idx].item()),
                )
                if proto_bank is not None:
                    pcb_sim = self.base_pcb._match_similarity(
                        pcb_features[q_idx], proto_bank,
                    )
                    pcb_sim = (pcb_sim + 1.0) / 2.0

            # ----- Tri-modal / bi-modal fusion -----
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


def build_pcb_fma_patch(base_pcb, cfg):
    """Factory function to wrap PCB with patch-level FM alignment."""
    return PCBFMAPatch(base_pcb, cfg)
