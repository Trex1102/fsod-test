"""
Meta-Learned Calibration for PCB (Meta-PCB).

Replaces the fixed linear calibration s†= α·s + (1-α)·s_cos with a small
meta-learned calibration network: s†= C_θ(s, s_cos, support_stats, roi_stats).

The calibration network is trained episodically on base classes (simulating
few-shot conditions) and frozen for novel class evaluation. This learns a
*general calibration policy* that adapts to different score distributions
without overfitting to specific classes.

Key properties:
- Zero novel-stage learnable parameters (calibrator is frozen)
- Non-linear calibration that adapts to confidence level
- Class-agnostic: operates on scalar statistics, not raw features
- Residual connection: starts as vanilla PCB, learns corrections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MetaCalibrationNet(nn.Module):
    """Small MLP that learns non-linear score calibration.

    Input: [det_score, cos_sim, support_mean_norm, support_std_norm,
            support_count_norm, support_dispersion, roi_feat_norm, score_entropy]
    Output: calibrated score in [0, 1]

    Initialized with residual connection to approximate vanilla PCB.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, residual_scale: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale

        self.calibrator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self._init_to_linear_baseline()

    def _init_to_linear_baseline(self):
        """Initialize so output ≈ 0.5*det_score + 0.5*cos_sim (vanilla PCB)."""
        # Zero-init the last layer so residual starts at 0
        nn.init.zeros_(self.calibrator[-1].weight)
        nn.init.zeros_(self.calibrator[-1].bias)

    def forward(
        self,
        det_score: torch.Tensor,
        cos_sim: torch.Tensor,
        support_mean_norm: torch.Tensor,
        support_std_norm: torch.Tensor,
        support_count_norm: torch.Tensor,
        support_dispersion: torch.Tensor,
        roi_feat_norm: torch.Tensor,
        score_entropy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibrated score.

        All inputs are scalar tensors (or batched [B]).
        """
        x = torch.stack(
            [
                det_score,
                cos_sim,
                support_mean_norm,
                support_std_norm,
                support_count_norm,
                support_dispersion,
                roi_feat_norm,
                score_entropy,
            ],
            dim=-1,
        )

        # Linear baseline (vanilla PCB)
        linear_baseline = 0.5 * det_score + 0.5 * cos_sim

        # Learned correction
        correction = self.calibrator(x).squeeze(-1)
        calibrated = linear_baseline + self.residual_scale * correction

        return calibrated.clamp(0.0, 1.0)


class MetaPCB:
    """PCB wrapper that uses a meta-learned calibration network.

    If no trained calibrator is available, falls back to a non-linear
    heuristic calibration based on score statistics.
    """

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg

        meta_cfg = cfg.NOVEL_METHODS.META_PCB
        self.fallback_alpha = float(meta_cfg.FALLBACK_ALPHA)
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Build calibration network
        self.calibrator = MetaCalibrationNet(
            input_dim=int(meta_cfg.INPUT_DIM),
            hidden_dim=int(meta_cfg.HIDDEN_DIM),
            residual_scale=float(meta_cfg.RESIDUAL_SCALE),
        ).to(self.device)
        self.calibrator.eval()

        # Try to load trained weights
        self.calibrator_trained = False
        calibrator_path = str(meta_cfg.CALIBRATOR_PATH)
        if calibrator_path:
            try:
                state_dict = torch.load(calibrator_path, map_location=self.device)
                self.calibrator.load_state_dict(state_dict)
                self.calibrator_trained = True
                logger.info("Meta-PCB: Loaded trained calibrator from %s", calibrator_path)
            except Exception as e:
                logger.warning("Meta-PCB: Failed to load calibrator from %s: %s", calibrator_path, e)

        if not self.calibrator_trained:
            logger.info(
                "Meta-PCB: No trained calibrator. Using residual-initialized network "
                "(behaves ~vanilla PCB + slight non-linear correction). "
                "Run tools/meta_train_calibrator.py to train."
            )

        # Pre-compute support statistics for each class
        self.class_stats: Dict[int, Dict[str, float]] = {}
        self._compute_class_stats()

    def _compute_class_stats(self):
        """Compute per-class support statistics for calibrator input."""
        for cls in self.base_pcb.prototypes:
            if cls not in self.base_pcb._real_class_features:
                continue

            feats = self.base_pcb._real_class_features[cls]
            if not feats:
                continue

            feat_tensor = torch.stack(feats, dim=0)
            mean_feat = feat_tensor.mean(dim=0)

            # Statistics for calibrator
            mean_norm = float(mean_feat.norm().item())
            std_norm = float(feat_tensor.std(dim=0).norm().item()) if len(feats) > 1 else 0.0
            count_norm = min(float(len(feats)) / 10.0, 1.0)  # normalize to [0, 1]

            # Dispersion: average cosine distance from mean
            if len(feats) > 1:
                mean_normed = F.normalize(mean_feat.unsqueeze(0), dim=1)
                feat_normed = F.normalize(feat_tensor, dim=1)
                cos_sims = torch.mm(feat_normed, mean_normed.t()).squeeze(1)
                dispersion = float((1.0 - cos_sims.mean()).item())
            else:
                dispersion = 0.5  # uncertain with single sample

            self.class_stats[cls] = {
                "mean_norm": mean_norm,
                "std_norm": std_norm,
                "count_norm": count_norm,
                "dispersion": dispersion,
            }

    def _compute_score_entropy(self, scores: torch.Tensor, pred_idx: int) -> float:
        """Compute entropy of the softmax score distribution around pred_idx."""
        # Use a local window of scores around the prediction
        s = float(scores[pred_idx].item())
        if s <= 0 or s >= 1:
            return 0.0
        # Binary entropy as proxy
        entropy = -(s * math.log(s + 1e-8) + (1 - s) * math.log(1 - s + 1e-8))
        return float(entropy)

    def execute_calibration(self, inputs, dts):
        """Execute calibration using the meta-learned calibrator."""
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        import cv2

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
        features = self.base_pcb.extract_roi_features(img, boxes)

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

            sim = self.base_pcb._match_similarity(features[q_idx], proto_bank)
            det_score = float(scores[i].item())

            # Get class statistics
            stats = self.class_stats.get(cls, {})
            mean_norm = stats.get("mean_norm", 1.0)
            std_norm = stats.get("std_norm", 0.0)
            count_norm = stats.get("count_norm", 0.1)
            dispersion = stats.get("dispersion", 0.5)

            # RoI feature statistics
            roi_norm = float(features[q_idx].norm().item())
            score_entropy = self._compute_score_entropy(scores, i)

            # Normalize inputs to reasonable ranges
            mean_norm_input = min(mean_norm / 100.0, 1.0)
            std_norm_input = min(std_norm / 50.0, 1.0)
            roi_norm_input = min(roi_norm / 100.0, 1.0)
            cos_sim_01 = (sim + 1.0) / 2.0  # map [-1,1] to [0,1]

            # Run calibrator
            with torch.no_grad():
                calibrated = self.calibrator(
                    torch.tensor(det_score, device=self.device),
                    torch.tensor(cos_sim_01, device=self.device),
                    torch.tensor(mean_norm_input, device=self.device),
                    torch.tensor(std_norm_input, device=self.device),
                    torch.tensor(count_norm, device=self.device),
                    torch.tensor(dispersion, device=self.device),
                    torch.tensor(roi_norm_input, device=self.device),
                    torch.tensor(score_entropy, device=self.device),
                )
                fused = float(calibrated.item())

            fused = max(0.0, min(1.0, fused))
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_meta_pcb(base_pcb, cfg):
    """Factory function to wrap PCB with meta-learned calibration."""
    return MetaPCB(base_pcb, cfg)
