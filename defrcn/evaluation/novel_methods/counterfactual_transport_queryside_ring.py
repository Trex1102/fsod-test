"""
Direction 11: Foreground by Counterfactual Transport (query-side background).

The question: "can this proposal be explained away by its own local context?"
For each query proposal we compare two transport costs in DINOv2 patch space:

  fg_cost = how well the support fg patches cover the query proposal
  bg_cost = how well the query's own ring patches cover the query proposal

The background is taken from the SAME query image (ring around the proposal),
not from support images. This ensures the counterfactual is same-domain and
genuinely per-image: the signal is "does this proposal look more like the
support class, or like the surrounding context in this specific scene?"

Mathematical core:
  For proposal B with patch set P_B (DINOv2 patches of the crop):
    P_fg    = support foreground patches (from support GT crops)
    P_ring  = DINOv2 patches of the ring around B in the query image
              (expanded box minus the inner proposal area)

  Transport is query-centric (B as source):
    fg_cost = mean_p ( min_f  dist(p, f) )   p in P_B, f in P_fg
    bg_cost = mean_p ( min_r  dist(p, r) )   p in P_B, r in P_ring

    Delta(B) = bg_cost - fg_cost

  Large Delta:
    every proposal patch finds a close support-fg match (fg_cost low)
    AND is far from the local ring context (bg_cost high)
    => proposal is genuine foreground, not a background region.

  Final score:
    fused = w_d * det_score + w_ct * sigmoid(delta / temperature) + w_pcb * pcb_sim

Architecture:
  Two batched DINOv2 passes per query image (proposals + their rings).
  Zero additional parameters. Training-free.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CounterfactualTransportExtractor:
    """Extracts DINOv2 patch tokens for counterfactual transport computation.

    Self-contained DINOv2 loader following the established pattern.
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
        """Patch cached DINOv2 source files for Python < 3.10 compatibility."""
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
                    except Exception:
                        pass

    def _load_model(self, model_name: str, model_path: str):
        """Load the DINOv2 model."""
        try:
            if model_path:
                logger.info("Loading counterfactual FM from local path: %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("Loading counterfactual FM via torch.hub: %s", model_name)
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
                        "Counterfactual FM dim mismatch: config %d, actual %d. Using actual.",
                        self.feat_dim, actual_dim,
                    )
                    self.feat_dim = actual_dim
                self.num_patches = actual_patches

            logger.info(
                "Counterfactual FM loaded: %s (dim=%d, num_patches=%d)",
                model_name, self.feat_dim, self.num_patches,
            )

        except Exception as e:
            logger.warning(
                "Failed to load counterfactual FM '%s': %s. Will fall back to base PCB.",
                model_name, e,
            )
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess cropped images for DINOv2 (ImageNet normalisation)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (self.roi_size, self.roi_size))
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

    def extract_patch_tokens(
        self, img: np.ndarray, boxes_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Extract patch tokens for each RoI box.

        Returns:
            patch_tokens: (N, num_patches, feat_dim) or None on failure.
        """
        if not self.available:
            return None

        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.num_patches, self.feat_dim), device=self.device)

        crops = self._crop_rois(img, boxes_tensor)

        all_patches = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch_crops = crops[start:start + self.batch_size]
                batch_tensor = self._preprocess_crops(batch_crops)
                out = self.model.get_intermediate_layers(
                    batch_tensor, n=1, return_class_token=True,
                )
                patch_tokens, _ = out[0]
                all_patches.append(patch_tokens)

        return torch.cat(all_patches, dim=0)


def _expand_box(box: np.ndarray, scale: float, img_h: int, img_w: int) -> np.ndarray:
    """Expand a box [x1,y1,x2,y2] by a scale factor, clipping to image bounds."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, cx - w / 2.0)
    ny1 = max(0, cy - h / 2.0)
    nx2 = min(img_w, cx + w / 2.0)
    ny2 = min(img_h, cy + h / 2.0)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


class CounterfactualTransport:
    """Foreground by Counterfactual Transport.

    Scores proposals by the gap between foreground and background transport
    costs. True objects have low foreground transport cost and high background
    transport cost (large positive Delta). Background/partial proposals are
    well-explained by background patches, yielding small or negative Delta.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        ct_cfg = cfg.NOVEL_METHODS.COUNTERFACTUAL_TRANSPORT

        self.det_weight = float(ct_cfg.DET_WEIGHT)
        self.ct_weight = float(ct_cfg.CT_WEIGHT)
        self.use_original_pcb = bool(ct_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(ct_cfg.ORIGINAL_PCB_WEIGHT)

        self.ring_scale = float(ct_cfg.RING_SCALE)
        self.temperature = float(ct_cfg.TEMPERATURE)
        self.use_sinkhorn = bool(ct_cfg.USE_SINKHORN)
        self.sinkhorn_iters = int(ct_cfg.SINKHORN_ITERS)
        self.sinkhorn_reg = float(ct_cfg.SINKHORN_REG)

        # Normalise fusion weights
        if self.use_original_pcb:
            total = self.det_weight + self.ct_weight + self.original_pcb_weight
        else:
            total = self.det_weight + self.ct_weight
        if total > 0:
            self.det_weight /= total
            self.ct_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        # Load DINOv2
        self.fm_extractor = CounterfactualTransportExtractor(
            model_name=str(ct_cfg.FM_MODEL_NAME),
            model_path=str(ct_cfg.FM_MODEL_PATH),
            feat_dim=int(ct_cfg.FM_FEAT_DIM),
            roi_size=int(ct_cfg.ROI_SIZE),
            batch_size=int(ct_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        # Per-class foreground patch bank (background is query-side, built at inference)
        self.fg_patches: Dict[int, torch.Tensor] = {}   # cls -> (N_fg, D) normalised

        if self.fm_extractor.available:
            self._build_fg_bg_representations()
        else:
            logger.warning("Counterfactual Transport: FM not available, falling back to base PCB.")

    @staticmethod
    def _ring_patch_mask(
        fg_box: np.ndarray, expanded_box: np.ndarray, n_patches_side: int,
    ) -> torch.Tensor:
        """Return a boolean mask of patches in the expanded crop that are OUTSIDE the fg box.

        Args:
            fg_box: [x1, y1, x2, y2] original foreground box in image coords.
            expanded_box: [x1, y1, x2, y2] expanded (ring) crop in image coords.
            n_patches_side: number of patches per side (e.g. 16 for 224/14).

        Returns:
            mask: (n_patches_side**2,) bool tensor, True = ring patch (keep for bg).
        """
        ex1, ey1, ex2, ey2 = expanded_box
        fx1, fy1, fx2, fy2 = fg_box

        exp_w = max(ex2 - ex1, 1e-6)
        exp_h = max(ey2 - ey1, 1e-6)

        # Fg box in normalised expanded-crop coordinates [0, 1]
        rel_x1 = (fx1 - ex1) / exp_w
        rel_y1 = (fy1 - ey1) / exp_h
        rel_x2 = (fx2 - ex1) / exp_w
        rel_y2 = (fy2 - ey1) / exp_h

        N = n_patches_side
        keep = []
        for r in range(N):
            for c in range(N):
                # Patch centre in normalised coords
                pc_x = (c + 0.5) / N
                pc_y = (r + 0.5) / N
                in_fg = (rel_x1 <= pc_x <= rel_x2) and (rel_y1 <= pc_y <= rel_y2)
                keep.append(not in_fg)

        return torch.tensor(keep, dtype=torch.bool)

    def _build_fg_bg_representations(self):
        """Build per-class foreground patch banks from the support set.

        Background is no longer modelled from support images. Instead, at
        inference time the ring around each query proposal serves as the
        per-image background, making the counterfactual genuinely same-domain.
        """
        logger.info("Counterfactual Transport: Building fg patch banks from support set...")

        class_fg: Dict[int, List[torch.Tensor]] = {}

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

            fg_tokens = self.fm_extractor.extract_patch_tokens(img, gt_boxes)
            if fg_tokens is None:
                continue
            fg_tokens = fg_tokens.detach().cpu()

            for i in range(len(labels)):
                cls = int(labels[i].item())
                if cls not in class_fg:
                    class_fg[cls] = []
                class_fg[cls].append(F.normalize(fg_tokens[i], dim=1))

        for cls, patches in class_fg.items():
            self.fg_patches[cls] = torch.cat(patches, dim=0)

        logger.info(
            "Counterfactual Transport: Built fg for %d classes (fg patches: %s)",
            len(self.fg_patches),
            {c: int(v.shape[0]) for c, v in self.fg_patches.items()},
        )

    def _transport_cost(
        self, source: torch.Tensor, target: torch.Tensor,
    ) -> float:
        """Compute transport cost from source measure to target measure.

        Uses either Sinkhorn OT or greedy nearest-neighbor transport.

        Args:
            source: (N_s, D) L2-normalised source patches.
            target: (N_t, D) L2-normalised target patches.

        Returns:
            Scalar transport cost (lower = better match).
        """
        if source.shape[0] == 0 or target.shape[0] == 0:
            return float("inf")

        # Cost matrix: 1 - cosine_similarity (so 0 = identical, 2 = opposite)
        cost = 1.0 - (source @ target.T)  # (N_s, N_t)

        if self.use_sinkhorn:
            return self._sinkhorn_cost(cost)
        else:
            return self._greedy_transport_cost(cost)

    def _greedy_transport_cost(self, cost: torch.Tensor) -> float:
        """Simple greedy transport: each source patch maps to its nearest target.

        Returns mean of minimum costs (average nearest-neighbor distance).
        """
        # For each source patch, find minimum cost to any target patch
        min_costs = cost.min(dim=1).values  # (N_s,)
        return float(min_costs.mean().item())

    def _sinkhorn_cost(self, cost: torch.Tensor) -> float:
        """Sinkhorn optimal transport cost with entropic regularisation.

        Solves: min_T sum(T * C) + reg * sum(T * log(T))
        subject to T1 = a, T^T 1 = b (uniform marginals).
        """
        n_s, n_t = cost.shape
        reg = max(self.sinkhorn_reg, 1e-4)

        # Uniform marginals
        a = torch.ones(n_s, device=cost.device) / n_s
        b = torch.ones(n_t, device=cost.device) / n_t

        # Gibbs kernel
        K = torch.exp(-cost / reg)

        u = torch.ones(n_s, device=cost.device)
        for _ in range(self.sinkhorn_iters):
            v = b / (K.T @ u + 1e-8)
            u = a / (K @ v + 1e-8)

        # Transport plan
        T = u.unsqueeze(1) * K * v.unsqueeze(0)

        # Transport cost
        ot_cost = float(torch.sum(T * cost).item())
        return ot_cost

    def _compute_counterfactual_gap(
        self,
        query_patches: torch.Tensor,
        ring_patches: torch.Tensor,
        cls: int,
    ) -> float:
        """Compute the query-side counterfactual transport gap.

        Delta(B) = cost(B -> bg_ring) - cost(B -> fg_support)

        where:
          B        = query proposal patches (source for both transports)
          bg_ring  = DINOv2 patches from the ring around the query proposal
                     (local context from the SAME query image)
          fg_support = DINOv2 patches from the support GT boxes

        Transport is query-centric: for each query patch we find its
        nearest patch in the target set. This answers "can every part
        of the proposal be covered by fg (or bg)?"

        Large positive Delta:
          query patches are far from the local ring (not background)
          AND close to support fg (looks like the class)
          => true foreground proposal.
        """
        if cls not in self.fg_patches or ring_patches.shape[0] == 0:
            return 0.0

        dev = query_patches.device
        fg = self.fg_patches[cls].to(dev)

        # Query-centric: query patches are the source
        fg_cost = self._transport_cost(query_patches, fg)
        bg_cost = self._transport_cost(query_patches, ring_patches)

        return float(bg_cost - fg_cost)

    def _delta_to_score(self, delta: float) -> float:
        """Map the counterfactual gap to a [0, 1] score via sigmoid.

        score = sigmoid(delta / temperature)
        """
        temp = max(self.temperature, 1e-6)
        x = delta / temp
        # Clamp to avoid overflow
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def execute_calibration(self, inputs, dts):
        """Execute calibration with query-side counterfactual transport scoring.

        For each candidate proposal the background measure is the ring of
        patches immediately surrounding that proposal in the SAME query image.
        This ensures the counterfactual comparison is always same-domain:
        "does this proposal look more like the support class, or like its
        own local context?"
        """
        if not self.fm_extractor.available or not self.fg_patches:
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

        # Extract patch tokens for all candidate proposals (one batched DINOv2 pass)
        query_all_patches = self.fm_extractor.extract_patch_tokens(img, box_tensor)

        # Extract patch tokens for all candidate ring boxes (one batched DINOv2 pass)
        ring_boxes_np = np.array([
            _expand_box(boxes_np[j], self.ring_scale, img_h, img_w)
            for j in range(len(boxes_np))
        ])
        ring_boxes_tensor = torch.tensor(ring_boxes_np, dtype=torch.float32)
        ring_all_patches = self.fm_extractor.extract_patch_tokens(img, ring_boxes_tensor)

        # Base PCB features (optional additional channel)
        if self.use_original_pcb:
            boxes = [pred_boxes.to(self.base_pcb.device)]
            pcb_features = self.base_pcb.extract_roi_features(img, boxes)
            area_norm = self.base_pcb._normalized_area(box_tensor, img_h, img_w)

        dev = self.fm_extractor.device

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb.exclude_cls:
                continue

            q_idx = i - ileft
            det_score = float(scores[i].item())

            # ----- Query-side counterfactual transport score -----
            ct_sim = 0.5  # neutral default
            if (cls in self.fg_patches
                    and query_all_patches is not None
                    and ring_all_patches is not None):

                q_patches = F.normalize(query_all_patches[q_idx].to(dev), dim=1)

                # Ring patches: only those outside the proposal box
                ring_mask = self._ring_patch_mask(
                    boxes_np[q_idx], ring_boxes_np[q_idx], n_patches_side,
                )
                ring_raw = ring_all_patches[q_idx].to(dev)
                ring_patches = F.normalize(ring_raw[ring_mask], dim=1)

                delta = self._compute_counterfactual_gap(q_patches, ring_patches, cls)
                ct_sim = self._delta_to_score(delta)

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

            # ----- Fusion -----
            if self.use_original_pcb:
                fused = (
                    self.det_weight * det_score
                    + self.ct_weight * ct_sim
                    + self.original_pcb_weight * pcb_sim
                )
            else:
                fused = self.det_weight * det_score + self.ct_weight * ct_sim

            fused = max(0.0, min(1.0, fused))
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_counterfactual_transport_pcb(base_pcb, cfg):
    """Factory function to wrap PCB with counterfactual transport scoring."""
    return CounterfactualTransport(base_pcb, cfg)
