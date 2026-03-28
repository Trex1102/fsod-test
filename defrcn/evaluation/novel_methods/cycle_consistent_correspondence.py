"""
Direction 3: Cycle-Consistent Support Correspondence as an Objectness Test.

Instead of asking whether a proposal resembles a support exemplar, this direction
asks whether the proposal can act as a consistent mediator between multiple support
exemplars. A real instance should support low-error cycles s_i -> B -> s_j in dense
DINOv2 correspondence space. Background or partial boxes usually match one support
superficially but fail multi-support cycle consistency. Detection therefore becomes
a relational consistency problem, not a metric matching problem.

Mathematical core:
  For each pair of supports (s_i, s_j) and proposal B:
    P_iB = soft correspondence from s_i to B  (N_si x N_B)
    P_Bj = soft correspondence from B to s_j  (N_B x N_sj)
    D_ij = direct correspondence from s_i to s_j  (N_si x N_sj)

  Cycle error:
    E_cycle(B) = sum_{i<j} ||P_Bj @ P_iB - D_ij||_F^2

  Coverage: fraction of support patches that have a good match in B.
    cov(B) = mean_i (fraction of s_i patches with max sim to B > threshold)

  Final score:
    S(B) = -E_cycle(B) + alpha * cov(B) + eta * log(o_CNN(B))

Architecture:
  fused_score = w_d * det_score + w_cycle * cycle_score + w_pcb * pcb_sim

Zero additional parameters. Inference-time only. Training-free.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CycleCorrespondenceExtractor:
    """Extracts DINOv2 patch tokens for cycle-consistency computation.

    Reuses the PatchFeatureExtractor pattern but is self-contained
    to avoid coupling with the PCB-FMA-Patch module.
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
                logger.info("Loading cycle-consistency FM from local path: %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("Loading cycle-consistency FM via torch.hub: %s", model_name)
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
                        "Cycle FM dim mismatch: config %d, actual %d. Using actual.",
                        self.feat_dim, actual_dim,
                    )
                    self.feat_dim = actual_dim
                self.num_patches = actual_patches

            logger.info(
                "Cycle FM loaded: %s (dim=%d, num_patches=%d)",
                model_name, self.feat_dim, self.num_patches,
            )

        except Exception as e:
            logger.warning(
                "Failed to load cycle FM '%s': %s. Will fall back to base PCB.",
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

        Args:
            img: BGR image (H, W, 3) as numpy array.
            boxes_tensor: (N, 4) tensor of [x1, y1, x2, y2] boxes.

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


class CycleConsistentCorrespondence:
    """Cycle-Consistent Support Correspondence as an Objectness Test.

    Scores proposals by how well they mediate correspondences between
    multiple support exemplars. True objects should produce low cycle
    error when used as intermediaries in support-to-support correspondence
    chains via DINOv2 patch-level features.

    Score:
      S(B) = -E_cycle(B) + alpha * cov(B) + eta * log(o_CNN(B))
      where E_cycle measures cycle inconsistency and cov measures
      support patch coverage.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        cc_cfg = cfg.NOVEL_METHODS.CYCLE_CONSISTENCY

        self.det_weight = float(cc_cfg.DET_WEIGHT)
        self.cycle_weight = float(cc_cfg.CYCLE_WEIGHT)
        self.use_original_pcb = bool(cc_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(cc_cfg.ORIGINAL_PCB_WEIGHT)

        self.coverage_alpha = float(cc_cfg.COVERAGE_ALPHA)
        self.coverage_threshold = float(cc_cfg.COVERAGE_THRESHOLD)
        self.correspondence_temperature = float(cc_cfg.CORRESPONDENCE_TEMPERATURE)
        self.use_hard_correspondence = bool(cc_cfg.USE_HARD_CORRESPONDENCE)

        # Normalise fusion weights
        if self.use_original_pcb:
            total = self.det_weight + self.cycle_weight + self.original_pcb_weight
        else:
            total = self.det_weight + self.cycle_weight
        if total > 0:
            self.det_weight /= total
            self.cycle_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        # Load DINOv2 for patch-level correspondences
        self.fm_extractor = CycleCorrespondenceExtractor(
            model_name=str(cc_cfg.FM_MODEL_NAME),
            model_path=str(cc_cfg.FM_MODEL_PATH),
            feat_dim=int(cc_cfg.FM_FEAT_DIM),
            roi_size=int(cc_cfg.ROI_SIZE),
            batch_size=int(cc_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        # Per-class support patch tokens: {cls: list of (N_patches, D) tensors}
        self.support_patches: Dict[int, List[torch.Tensor]] = {}
        # Precomputed direct support-support correspondence matrices
        # {cls: list of (N_si, N_sj) matrices for all pairs i<j}
        self.direct_correspondences: Dict[int, List[Tuple[int, int, torch.Tensor]]] = {}

        if self.fm_extractor.available:
            self._build_support_representations()
        else:
            logger.warning("Cycle-Consistency: FM not available, falling back to base PCB.")

    def _build_support_representations(self):
        """Build per-class support patch tokens and direct correspondence matrices."""
        logger.info("Cycle-Consistency: Building support patch tokens from support set...")

        class_patches: Dict[int, List[torch.Tensor]] = {}

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

            patch_tokens = self.fm_extractor.extract_patch_tokens(img, gt_boxes)
            if patch_tokens is None:
                continue

            patch_tokens = patch_tokens.detach().cpu()

            for i in range(len(labels)):
                cls = int(labels[i].item())
                if cls not in class_patches:
                    class_patches[cls] = []
                # L2-normalise each support's patches
                normed = F.normalize(patch_tokens[i], dim=1)  # (N_patches, D)
                class_patches[cls].append(normed)

        self.support_patches = class_patches

        # Precompute direct support-support correspondences for cycle error
        for cls, patches_list in class_patches.items():
            pairs = []
            n_supports = len(patches_list)
            for i in range(n_supports):
                for j in range(i + 1, n_supports):
                    # D_ij: soft correspondence from s_i to s_j
                    D_ij = self._compute_soft_correspondence(
                        patches_list[i], patches_list[j]
                    )
                    pairs.append((i, j, D_ij))
            self.direct_correspondences[cls] = pairs

        logger.info(
            "Cycle-Consistency: Built support for %d classes (supports/class: %s)",
            len(self.support_patches),
            {c: len(v) for c, v in self.support_patches.items()},
        )

    def _compute_soft_correspondence(
        self, source: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft correspondence matrix from source to target patches.

        Args:
            source: (N_s, D) L2-normalised patch tokens.
            target: (N_t, D) L2-normalised patch tokens.

        Returns:
            P: (N_s, N_t) soft correspondence matrix (rows sum to ~1).
        """
        # Cosine similarity matrix
        sim = source @ target.T  # (N_s, N_t)

        if self.use_hard_correspondence:
            # Hard (argmax) correspondence: one-hot rows
            idx = sim.argmax(dim=1)  # (N_s,)
            P = torch.zeros_like(sim)
            P.scatter_(1, idx.unsqueeze(1), 1.0)
        else:
            # Soft correspondence via temperature-scaled softmax
            temp = max(self.correspondence_temperature, 1e-6)
            P = F.softmax(sim / temp, dim=1)  # (N_s, N_t)

        return P

    def _compute_cycle_score(
        self, query_patches: torch.Tensor, cls: int,
    ) -> Tuple[float, float]:
        """Compute cycle consistency score for a proposal against class supports.

        For each pair (s_i, s_j):
          - P_iB: correspondence from s_i to B (proposal)
          - P_Bj: correspondence from B to s_j
          - D_ij: direct correspondence from s_i to s_j
          - cycle_error_ij = ||P_Bj @ P_iB - D_ij||_F^2

        Also computes coverage: fraction of support patches with good match in B.

        Args:
            query_patches: (N_B, D) L2-normalised proposal patch tokens.
            cls: class index.

        Returns:
            cycle_score: normalised cycle consistency score in [0, 1] (higher = better).
            coverage: support patch coverage in [0, 1].
        """
        if cls not in self.support_patches:
            return 0.0, 0.0

        supports = self.support_patches[cls]
        pairs = self.direct_correspondences.get(cls, [])

        if len(supports) < 2:
            # Single support: can't compute cycle, fall back to coverage only
            if len(supports) == 1:
                sim = supports[0] @ query_patches.T  # (N_s, N_B)
                cov = float((sim.max(dim=1).values > self.coverage_threshold).float().mean().item())
                return 0.5, cov  # Neutral cycle score
            return 0.0, 0.0

        dev = query_patches.device

        total_cycle_error = 0.0
        n_pairs = 0

        for (i, j, D_ij) in pairs:
            s_i = supports[i].to(dev)
            s_j = supports[j].to(dev)
            D_ij_dev = D_ij.to(dev)

            # P_iB: soft correspondence from s_i to B
            P_iB = self._compute_soft_correspondence(s_i, query_patches)
            # P_Bj: soft correspondence from B to s_j
            P_Bj = self._compute_soft_correspondence(query_patches, s_j)

            # Composed correspondence: s_i -> B -> s_j
            composed = P_iB @ P_Bj  # (N_si, N_sj) -- note: P_iB is (N_si, N_B), P_Bj is (N_B, N_sj)
            # Wait, P_iB is (N_si, N_B), but we want (N_si, N_sj)
            # Actually P_iB maps s_i patches to B patches: (N_si, N_B)
            # P_Bj maps B patches to s_j patches: (N_B, N_sj)
            # So composed = P_iB @ P_Bj = (N_si, N_sj) which is correct

            # Cycle error: ||composed - D_ij||_F^2
            diff = composed - D_ij_dev
            cycle_error = float(torch.sum(diff ** 2).item())

            # Normalise by matrix size to make comparable across different support sizes
            norm_factor = max(float(D_ij_dev.shape[0] * D_ij_dev.shape[1]), 1.0)
            total_cycle_error += cycle_error / norm_factor
            n_pairs += 1

        # Average cycle error across pairs
        avg_cycle_error = total_cycle_error / max(n_pairs, 1)

        # Convert to score: lower error = higher score
        # Use exponential mapping: score = exp(-gamma * error)
        # Typical cycle errors range from 0 (perfect) to ~2 (random)
        gamma = 5.0  # Controls sensitivity
        cycle_score = float(np.exp(-gamma * avg_cycle_error))

        # Compute coverage: mean over all supports of fraction of patches
        # that have a good match (cosine sim > threshold) in the proposal
        total_cov = 0.0
        for s_patches in supports:
            s_dev = s_patches.to(dev)
            sim = s_dev @ query_patches.T  # (N_s, N_B)
            max_sims = sim.max(dim=1).values  # (N_s,)
            cov = float((max_sims > self.coverage_threshold).float().mean().item())
            total_cov += cov
        coverage = total_cov / len(supports)

        # Combine: cycle_score already in [0,1], coverage in [0,1]
        # S(B) = -E_cycle + alpha * cov  (mapped to [0,1] space)
        combined = cycle_score * (1.0 - self.coverage_alpha) + coverage * self.coverage_alpha

        return float(combined), float(coverage)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def execute_calibration(self, inputs, dts):
        """Execute calibration with cycle-consistent correspondence scoring."""
        if not self.fm_extractor.available or not self.support_patches:
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

            # ----- Cycle consistency score -----
            cycle_sim = 0.0
            if cls in self.support_patches and query_all_patches is not None:
                dev = self.fm_extractor.device
                q_patches = F.normalize(query_all_patches[q_idx].to(dev), dim=1)
                cycle_sim, _ = self._compute_cycle_score(q_patches, cls)

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
                    + self.cycle_weight * cycle_sim
                    + self.original_pcb_weight * pcb_sim
                )
            else:
                fused = self.det_weight * det_score + self.cycle_weight * cycle_sim

            fused = max(0.0, min(1.0, fused))
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_cycle_consistent_pcb(base_pcb, cfg):
    """Factory function to wrap PCB with cycle-consistent correspondence scoring."""
    return CycleConsistentCorrespondence(base_pcb, cfg)
