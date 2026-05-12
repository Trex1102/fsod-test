"""
PCB-FMA-Ring: Combined Foundation Model Alignment + Ring Background Rejection.

Motivation
----------
Two complementary signals, combined for the first time:

  1. FMA (foreground signal) -- DINOv2 CLS-token similarity to augmented
     class prototypes built from support crops.  This is the strongest
     available fg signal: temperature-scaled softmax across all novel
     prototypes forces inter-class competition.

  2. Ring transport (background rejection signal) -- for each query
     proposal we crop a background *ring* from the surrounding image
     context (expanded box minus the inner proposal area) and compute
     the patch-level transport cost from query patches to ring patches.
     If bg_cost is HIGH, the proposal does NOT look like its local
     background context -- it is likely genuine foreground.
     If bg_cost is LOW, the proposal blends into the background -- it
     is likely a false positive.

Unlike pure Counterfactual Transport (CT), the ring score here is a
STANDALONE signal (no fg_cost subtraction).  This removes the noisy
fg_cost that degrades CT at higher shots, while preserving the useful
bg-contrast signal from the ring.

Key design choices
------------------
* Single DINOv2 (dinov2_vitb14) forward call using
  ``get_intermediate_layers`` returns BOTH CLS token (for FMA) and
  patch tokens (for ring transport), so only ONE model is loaded.
* Two batched passes per image: (a) proposal crops, (b) ring crops.
* No fg patch bank needed -- removing the fg_cost term eliminates the
  per-class storage and the averaging complexity.
* Augmented prototype building (flip + multicrop) from support set for
  robust CLS prototypes (same as pcb_fma_enhanced).

Fusion
------
  fused = w_det * det_score + w_fma * fma_sim + w_ring * ring_score

  fma_sim   = softmax_i( cos(query_cls, proto_i) / fma_temperature ) [i=target class]
  ring_score = sigmoid( bg_cost / ring_temperature )
               where bg_cost = mean_p min_r dist(p, r), p in query patches,
               r in ring patches.

Default weights: DET=0.3, FMA=0.5, RING=0.2 (sum to 1.0).
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Feature extractor: CLS + patch tokens in one DINOv2 pass
# -----------------------------------------------------------------------

class PCBFMARingExtractor:
    """Single DINOv2 backbone that returns both CLS and patch tokens.

    Uses ``get_intermediate_layers(..., return_class_token=True)`` so one
    forward pass yields both signals needed for FMA and ring transport.
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
        self.num_patches = (roi_size // 14) ** 2

        self._load_model(model_name, model_path)

    @staticmethod
    def _patch_dinov2_for_py38():
        import sys
        if sys.version_info >= (3, 10):
            return
        import glob, os
        hub_dir = torch.hub.get_dir()
        for d in glob.glob(os.path.join(hub_dir, "facebookresearch_dinov2*")):
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath) as f:
                            content = f.read()
                        if (("float | None" in content or "int | None" in content
                                or "str | None" in content)
                                and "from __future__ import annotations" not in content):
                            with open(fpath, "w") as f:
                                f.write("from __future__ import annotations\n" + content)
                    except Exception:
                        pass

    def _load_model(self, model_name: str, model_path: str):
        try:
            if model_path:
                logger.info("PCB-FMA-Ring: loading backbone from %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("PCB-FMA-Ring: loading %s via torch.hub", model_name)
                self._patch_dinov2_for_py38()
                try:
                    self.model = torch.hub.load(
                        "facebookresearch/dinov2", model_name, pretrained=True,
                    )
                except TypeError as e:
                    if "unsupported operand type" in str(e):
                        self._patch_dinov2_for_py38()
                        import sys as _sys
                        for k in [k for k in _sys.modules if "dinov2" in k]:
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

            # Infer actual dims from a dummy forward
            with torch.no_grad():
                dummy = torch.randn(
                    1, 3, self.roi_size, self.roi_size, device=self.device
                )
                out = self.model.get_intermediate_layers(
                    dummy, n=1, return_class_token=True
                )
                patch_tokens, cls_token = out[0]
                actual_dim = cls_token.shape[-1]
                actual_patches = patch_tokens.shape[1]
                if actual_dim != self.feat_dim:
                    logger.warning(
                        "PCB-FMA-Ring dim mismatch: cfg %d, actual %d. Using actual.",
                        self.feat_dim, actual_dim,
                    )
                    self.feat_dim = actual_dim
                self.num_patches = actual_patches

            logger.info(
                "PCB-FMA-Ring backbone ready: %s (dim=%d, patches=%d)",
                model_name, self.feat_dim, self.num_patches,
            )

        except Exception as e:
            logger.warning(
                "PCB-FMA-Ring: failed to load '%s': %s. Will fall back to base PCB.",
                model_name, e,
            )
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (self.roi_size, self.roi_size))
            t = (
                torch.from_numpy(resized[:, :, ::-1].copy())
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(self.device)
        return (batch - mean) / std

    def _crop_rois(
        self, img: np.ndarray, boxes_tensor: torch.Tensor
    ) -> List[np.ndarray]:
        boxes = boxes_tensor.cpu().numpy()
        img_h, img_w = img.shape[:2]
        crops = []
        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(img_w, int(box[2]))
            y2 = min(img_h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(img[y1:y1 + 1, x1:x1 + 1])
            else:
                crops.append(img[y1:y2, x1:x2])
        return crops

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_cls_and_patch_tokens(
        self,
        img: np.ndarray,
        boxes_tensor: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Single forward pass → (cls_tokens, patch_tokens).

        Returns:
            cls_tokens:   (N, feat_dim)
            patch_tokens: (N, num_patches, feat_dim)
            or None on failure.
        """
        if not self.available:
            return None
        if boxes_tensor.numel() == 0:
            return (
                torch.empty((0, self.feat_dim), device=self.device),
                torch.empty((0, self.num_patches, self.feat_dim), device=self.device),
            )
        crops = self._crop_rois(img, boxes_tensor)
        all_cls, all_patches = [], []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = self._preprocess_crops(crops[start:start + self.batch_size])
                out = self.model.get_intermediate_layers(
                    batch, n=1, return_class_token=True
                )
                patch_tok, cls_tok = out[0]
                all_cls.append(cls_tok)
                all_patches.append(patch_tok)
        return torch.cat(all_cls, dim=0), torch.cat(all_patches, dim=0)

    def extract_patch_tokens_only(
        self,
        img: np.ndarray,
        boxes_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Patch tokens only — used for ring crops where CLS is not needed."""
        result = self.extract_cls_and_patch_tokens(img, boxes_tensor)
        return None if result is None else result[1]


# -----------------------------------------------------------------------
# Ring geometry helpers
# -----------------------------------------------------------------------

def _expand_box(
    box: np.ndarray, scale: float, img_h: int, img_w: int
) -> np.ndarray:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return np.array(
        [
            max(0, cx - w / 2.0),
            max(0, cy - h / 2.0),
            min(img_w, cx + w / 2.0),
            min(img_h, cy + h / 2.0),
        ],
        dtype=np.float32,
    )


def _ring_patch_mask(
    fg_box: np.ndarray, expanded_box: np.ndarray, n_patches_side: int
) -> torch.Tensor:
    """Boolean mask of patches in the expanded crop that are OUTSIDE the fg box."""
    ex1, ey1, ex2, ey2 = expanded_box
    fx1, fy1, fx2, fy2 = fg_box
    exp_w = max(ex2 - ex1, 1e-6)
    exp_h = max(ey2 - ey1, 1e-6)
    rel_x1 = (fx1 - ex1) / exp_w
    rel_y1 = (fy1 - ey1) / exp_h
    rel_x2 = (fx2 - ex1) / exp_w
    rel_y2 = (fy2 - ey1) / exp_h
    N = n_patches_side
    keep = []
    for r in range(N):
        for c in range(N):
            pc_x = (c + 0.5) / N
            pc_y = (r + 0.5) / N
            in_fg = (rel_x1 <= pc_x <= rel_x2) and (rel_y1 <= pc_y <= rel_y2)
            keep.append(not in_fg)
    return torch.tensor(keep, dtype=torch.bool)


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------

class PCBFMARing:
    """PCB-FMA + Ring Background Rejection.

    Combines:
      - DINOv2 CLS competitive-softmax similarity to augmented class
        prototypes (FMA signal, foreground-facing).
      - Ring transport background rejection: proposals that look like
        their local context score low; proposals distinct from context
        score high (ring signal, background-facing, standalone).
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        ring_cfg = cfg.NOVEL_METHODS.PCB_FMA_RING

        self.det_weight = float(ring_cfg.DET_WEIGHT)
        self.fma_weight = float(ring_cfg.FMA_WEIGHT)
        self.ring_weight = float(ring_cfg.RING_WEIGHT)

        # Normalize weights to sum to 1
        total = self.det_weight + self.fma_weight + self.ring_weight
        if total > 0:
            self.det_weight /= total
            self.fma_weight /= total
            self.ring_weight /= total

        self.ring_scale = float(ring_cfg.RING_SCALE)
        self.ring_temperature = float(ring_cfg.RING_TEMPERATURE)
        self.fma_temperature = float(ring_cfg.FMA_TEMPERATURE)

        # Support augmentation
        self.enable_flip = bool(ring_cfg.AUG_FLIP)
        self.enable_multicrop = bool(ring_cfg.AUG_MULTICROP)
        self.multicrop_scales = list(ring_cfg.AUG_MULTICROP_SCALES)
        self.multicrop_num = int(ring_cfg.AUG_MULTICROP_NUM)

        # Single backbone for both CLS and patch tokens
        self.fm_extractor = PCBFMARingExtractor(
            model_name=str(ring_cfg.FM_MODEL_NAME),
            model_path=str(ring_cfg.FM_MODEL_PATH),
            feat_dim=int(ring_cfg.FM_FEAT_DIM),
            roi_size=int(ring_cfg.ROI_SIZE),
            batch_size=int(ring_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        # Per-class CLS prototypes (augmented)
        self.fm_prototypes: Dict[int, torch.Tensor] = {}

        if self.fm_extractor.available:
            self._build_augmented_prototypes()
        else:
            logger.warning(
                "PCB-FMA-Ring: backbone unavailable, falling back to base PCB."
            )

    # ------------------------------------------------------------------
    # Support augmentation helpers (same as pcb_fma_enhanced)
    # ------------------------------------------------------------------

    def _augment_crop(self, crop: np.ndarray) -> List[np.ndarray]:
        views = [crop]
        if self.enable_flip:
            views.append(cv2.flip(crop, 1))
        if self.enable_multicrop:
            h, w = crop.shape[:2]
            for scale in self.multicrop_scales:
                ch, cw = int(h * scale), int(w * scale)
                if ch < 2 or cw < 2:
                    continue
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                views.append(crop[y0:y0 + ch, x0:x0 + cw])
                if self.multicrop_num > 1:
                    views.append(crop[0:ch, 0:cw])
                if self.multicrop_num > 2:
                    views.append(crop[max(0, h - ch):h, max(0, w - cw):w])
        return views

    def _extract_augmented_cls_for_box(
        self, img: np.ndarray, box: np.ndarray
    ) -> Optional[torch.Tensor]:
        """Augmented CLS feature for a single GT box.  Returns (feat_dim,) or None."""
        img_h, img_w = img.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2]))
        y2 = min(img_h, int(box[3]))
        crop = img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else img[y1:y1 + 1, x1:x1 + 1]

        views = self._augment_crop(crop)
        batch_tensor = self.fm_extractor._preprocess_crops(views)
        with torch.no_grad():
            # get_intermediate_layers returns (patch_tok, cls_tok) per layer
            out = self.fm_extractor.model.get_intermediate_layers(
                batch_tensor, n=1, return_class_token=True
            )
            _, cls_tok = out[0]   # (num_views, feat_dim)
        return cls_tok.mean(dim=0)  # average over augmented views

    # ------------------------------------------------------------------
    # Prototype building
    # ------------------------------------------------------------------

    def _build_augmented_prototypes(self):
        logger.info("PCB-FMA-Ring: building augmented CLS prototypes...")
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
                aug_feat = self._extract_augmented_cls_for_box(img, boxes_np[i])
                if aug_feat is None:
                    continue
                cls = int(labels[i].item())
                if cls not in class_features:
                    class_features[cls] = []
                class_features[cls].append(aug_feat.detach().cpu())

        for cls, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.fm_prototypes[cls] = F.normalize(proto, dim=0)

        logger.info(
            "PCB-FMA-Ring: built prototypes for %d classes (dim=%d)",
            len(self.fm_prototypes),
            self.fm_extractor.feat_dim,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_fma_sim(self, query_cls: torch.Tensor, cls: int) -> float:
        """Temperature-scaled competitive softmax FMA similarity."""
        if cls not in self.fm_prototypes:
            return 0.0

        all_cls_ids = sorted(self.fm_prototypes.keys())
        if len(all_cls_ids) <= 1:
            proto = self.fm_prototypes[cls].to(query_cls.device)
            sim = float(torch.dot(query_cls, proto).item())
            return (sim + 1.0) / 2.0

        sims, target_idx = [], -1
        for idx, c in enumerate(all_cls_ids):
            proto = self.fm_prototypes[c].to(query_cls.device)
            sims.append(torch.dot(query_cls, proto))
            if c == cls:
                target_idx = idx

        if target_idx < 0:
            return 0.0

        sims_tensor = torch.stack(sims, dim=0)
        temp = max(self.fma_temperature, 1e-6)
        competitive = F.softmax(sims_tensor / temp, dim=0)
        return float(competitive[target_idx].item())

    def _compute_ring_score(
        self,
        query_patches: torch.Tensor,
        ring_patches: torch.Tensor,
    ) -> float:
        """Standalone ring background score.

        bg_cost = mean over query patches of min cosine-distance to ring patches.
        High bg_cost → query is distinct from its context → likely FG → high ring_score.
        ring_score = sigmoid( bg_cost / ring_temperature )
        """
        if query_patches.shape[0] == 0 or ring_patches.shape[0] == 0:
            return 0.5  # neutral fallback

        # cosine distance matrix: (N_q, N_r)
        cost = 1.0 - (query_patches @ ring_patches.T)
        bg_cost = float(cost.min(dim=1).values.mean().item())

        temp = max(self.ring_temperature, 1e-6)
        x = max(-20.0, min(20.0, bg_cost / temp))
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def execute_calibration(self, inputs, dts):
        if not self.fm_extractor.available or not self.fm_prototypes:
            return self.base_pcb.execute_calibration(inputs, dts)

        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts

        img_h, img_w = img.shape[:2]
        n_patches_side = int(self.fm_extractor.roi_size // 14)

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
        box_tensor = pred_boxes.tensor
        boxes_np = box_tensor.cpu().numpy()

        # --- Pass 1: proposal crops → CLS tokens + patch tokens ---
        prop_result = self.fm_extractor.extract_cls_and_patch_tokens(img, box_tensor)

        # --- Pass 2: ring crops → patch tokens ---
        ring_boxes_np = np.array([
            _expand_box(boxes_np[j], self.ring_scale, img_h, img_w)
            for j in range(len(boxes_np))
        ])
        ring_boxes_tensor = torch.tensor(ring_boxes_np, dtype=torch.float32)
        ring_all_patches = self.fm_extractor.extract_patch_tokens_only(
            img, ring_boxes_tensor
        )

        dev = self.fm_extractor.device

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb.exclude_cls:
                continue

            q_idx = i - ileft
            det_score = float(scores[i].item())

            # --- FMA similarity (CLS token, competitive softmax) ---
            fma_sim = 0.5  # neutral fallback
            if prop_result is not None and cls in self.fm_prototypes:
                cls_tokens, _ = prop_result
                query_cls = F.normalize(cls_tokens[q_idx].to(dev), dim=0)
                fma_sim = self._compute_fma_sim(query_cls, cls)

            # --- Ring transport score (standalone bg_cost) ---
            ring_score = 0.5  # neutral fallback
            if prop_result is not None and ring_all_patches is not None:
                _, prop_patches = prop_result
                q_patches = F.normalize(prop_patches[q_idx].to(dev), dim=1)

                ring_mask = _ring_patch_mask(
                    boxes_np[q_idx], ring_boxes_np[q_idx], n_patches_side
                )
                ring_raw = ring_all_patches[q_idx].to(dev)
                ring_patches_masked = F.normalize(ring_raw[ring_mask], dim=1)

                ring_score = self._compute_ring_score(q_patches, ring_patches_masked)

            fused = (
                self.det_weight * det_score
                + self.fma_weight * fma_sim
                + self.ring_weight * ring_score
            )
            fused = max(0.0, min(1.0, fused))
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def __getattr__(self, name):
        return getattr(self.base_pcb, name)


def build_pcb_fma_ring(base_pcb, cfg):
    """Factory function for PCB-FMA-Ring."""
    return PCBFMARing(base_pcb, cfg)
