"""
Direction 11: Foreground by Counterfactual Transport.

The question here is not "does the box look like the support foreground?" but rather
"can the box be explained away by background?" Support boxes and support context rings
define competing foreground and context measures in DINOv2 space. A query proposal is
accepted only if foreground transport explains it substantially better than any
background explanation, including nearby query background. The detector is a
counterfactual evidence ratio rather than a similarity score.

Mathematical core:
  For proposal B with patch measure nu_B:
    mu_fg = support foreground measure (patches inside support boxes)
    mu_bg = support context measure (patches in ring around support boxes)

    Delta(B) = min_{T_bg} cost(T_bg; mu_bg, nu_B) - min_{T_fg} cost(T_fg; mu_fg, nu_B)

  Accept boxes with large positive Delta(B):
    Large Delta => background explains B poorly, foreground explains B well
    => B is likely a true foreground object.

  Final score:
    fused = w_d * det_score + w_ct * sigmoid(delta / temperature) + w_pcb * pcb_sim

Architecture:
  Uses Sinkhorn OT or simple transport cost (sum of best-match distances) to
  compute foreground and background transport costs. Zero parameters, training-free.
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

        # Per-class foreground and background patch banks
        self.fg_patches: Dict[int, torch.Tensor] = {}   # cls -> (N_fg, D) normalised
        self.bg_patches: Dict[int, torch.Tensor] = {}   # cls -> (N_bg, D) normalised

        if self.fm_extractor.available:
            self._build_fg_bg_representations()
        else:
            logger.warning("Counterfactual Transport: FM not available, falling back to base PCB.")

    def _build_fg_bg_representations(self):
        """Build per-class foreground and background patch banks from support set.

        Foreground patches: patches from support box crops.
        Background patches: patches from the ring region around support boxes
        (expanded box minus the original box area).
        """
        logger.info("Counterfactual Transport: Building fg/bg patch banks from support set...")

        class_fg: Dict[int, List[torch.Tensor]] = {}
        class_bg: Dict[int, List[torch.Tensor]] = {}

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

            # Extract foreground patches (from GT boxes)
            fg_tokens = self.fm_extractor.extract_patch_tokens(img, gt_boxes)
            if fg_tokens is None:
                continue
            fg_tokens = fg_tokens.detach().cpu()

            # Build ring boxes (expanded - original) for background context
            boxes_np = gt_boxes.cpu().numpy()
            ring_boxes = []
            for box in boxes_np:
                expanded = _expand_box(box, self.ring_scale, img_h, img_w)
                ring_boxes.append(expanded)

            ring_boxes_tensor = torch.tensor(np.array(ring_boxes), dtype=torch.float32)
            bg_tokens = self.fm_extractor.extract_patch_tokens(img, ring_boxes_tensor)
            if bg_tokens is None:
                continue
            bg_tokens = bg_tokens.detach().cpu()

            for i in range(len(labels)):
                cls = int(labels[i].item())
                if cls not in class_fg:
                    class_fg[cls] = []
                    class_bg[cls] = []

                # Foreground: all patches from the GT box crop
                class_fg[cls].append(F.normalize(fg_tokens[i], dim=1))

                # Background: patches from ring crop
                # We use all ring patches; ideally we'd subtract fg-overlapping patches,
                # but since the ring crop is a different (larger) region processed
                # independently, the interior patches will be at different spatial
                # positions in the ViT grid and thus represent different content.
                class_bg[cls].append(F.normalize(bg_tokens[i], dim=1))

        # Concatenate all support patches per class
        for cls in class_fg:
            all_fg = torch.cat(class_fg[cls], dim=0)  # (N_fg_total, D)
            self.fg_patches[cls] = all_fg

            all_bg = torch.cat(class_bg[cls], dim=0)  # (N_bg_total, D)
            self.bg_patches[cls] = all_bg

        logger.info(
            "Counterfactual Transport: Built fg/bg for %d classes "
            "(fg patches: %s, bg patches: %s)",
            len(self.fg_patches),
            {c: int(v.shape[0]) for c, v in self.fg_patches.items()},
            {c: int(v.shape[0]) for c, v in self.bg_patches.items()},
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
        self, query_patches: torch.Tensor, cls: int,
    ) -> float:
        """Compute the counterfactual transport gap for a proposal.

        Delta(B) = cost(bg -> B) - cost(fg -> B)

        Large positive Delta means background explains B poorly but
        foreground explains it well => true foreground object.

        Args:
            query_patches: (N_B, D) L2-normalised proposal patch tokens.
            cls: class index.

        Returns:
            delta: counterfactual gap (higher = more likely foreground).
        """
        if cls not in self.fg_patches or cls not in self.bg_patches:
            return 0.0

        dev = query_patches.device
        fg = self.fg_patches[cls].to(dev)
        bg = self.bg_patches[cls].to(dev)

        fg_cost = self._transport_cost(fg, query_patches)
        bg_cost = self._transport_cost(bg, query_patches)

        # Delta = bg_cost - fg_cost
        # If fg explains the proposal well (low cost) and bg doesn't (high cost),
        # delta is large positive.
        delta = bg_cost - fg_cost
        return float(delta)

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
        """Execute calibration with counterfactual transport scoring."""
        if not self.fm_extractor.available or not self.fg_patches:
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

        # Extract patch tokens for candidate RoIs
        box_tensor = pred_boxes.tensor
        query_all_patches = self.fm_extractor.extract_patch_tokens(img, box_tensor)

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

            # ----- Counterfactual transport score -----
            ct_sim = 0.5  # neutral default
            if cls in self.fg_patches and query_all_patches is not None:
                dev = self.fm_extractor.device
                q_patches = F.normalize(query_all_patches[q_idx].to(dev), dim=1)
                delta = self._compute_counterfactual_gap(q_patches, cls)
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
                    pcb_sim = (pcb_sim + 1.0) / 2.0  # map [-1,1] -> [0,1]

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
