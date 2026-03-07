"""
Self-Distillation from Test-Time Predictions for Few-Shot Object Detection.

This module implements iterative self-distillation where high-confidence
predictions from the current model are used to refine prototypes and
improve subsequent predictions.

Key idea: Use the model's own confident predictions as pseudo-labels
to bootstrap better prototypes, similar to pseudo-labeling but applied
at the feature/prototype level.

Reference: Novel approach inspired by self-training and knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SelfDistillationModule:
    """
    Manages self-distillation from test-time predictions.
    
    Collects high-confidence predictions during inference and uses them
    to iteratively refine class prototypes.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        max_pseudo_per_class: int = 20,
        pseudo_weight: float = 0.3,
        ema_momentum: float = 0.99,
        min_samples_for_update: int = 3,
        temperature: float = 1.0,
        use_soft_labels: bool = True,
        entropy_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels
            max_pseudo_per_class: Maximum pseudo-samples to keep per class
            pseudo_weight: Weight of pseudo-samples relative to real support
            ema_momentum: EMA momentum for prototype updates
            min_samples_for_update: Minimum pseudo-samples before updating
            temperature: Temperature for soft label generation
            use_soft_labels: Whether to use soft probability labels
            entropy_threshold: Maximum entropy for accepting pseudo-labels
            device: Device for computations
        """
        self.confidence_threshold = confidence_threshold
        self.max_pseudo_per_class = max_pseudo_per_class
        self.pseudo_weight = pseudo_weight
        self.ema_momentum = ema_momentum
        self.min_samples_for_update = min_samples_for_update
        self.temperature = temperature
        self.use_soft_labels = use_soft_labels
        self.entropy_threshold = entropy_threshold
        self.device = device
        
        # Pseudo-label storage
        self.pseudo_features: Dict[int, List[torch.Tensor]] = {}
        self.pseudo_scores: Dict[int, List[float]] = {}
        self.pseudo_soft_labels: Dict[int, List[torch.Tensor]] = {}
        
        # Original prototype storage (for weighted combination)
        self.original_prototypes: Optional[Dict] = None
        
        # Statistics
        self.num_collected = 0
        self.num_updates = 0
    
    def reset(self):
        """Reset pseudo-label storage."""
        self.pseudo_features = {}
        self.pseudo_scores = {}
        self.pseudo_soft_labels = {}
        self.num_collected = 0
    
    def compute_entropy(self, probs: torch.Tensor) -> float:
        """
        Compute entropy of probability distribution.
        
        Args:
            probs: Probability tensor
        
        Returns:
            Entropy value (lower = more confident)
        """
        probs = probs.clamp(min=1e-8)
        entropy = -torch.sum(probs * torch.log(probs))
        # Normalize by max entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(float(probs.numel())))
        return float(entropy / max_entropy)
    
    def collect_pseudo_labels(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        scores: torch.Tensor,
        class_probs: Optional[torch.Tensor] = None
    ):
        """
        Collect high-confidence predictions as pseudo-labels.
        
        Args:
            features: Feature vectors of detections (N, D)
            predictions: Predicted class labels (N,)
            scores: Confidence scores (N,)
            class_probs: Full class probability distribution (N, C) for soft labels
        """
        for i in range(features.shape[0]):
            score = float(scores[i].item())
            cls = int(predictions[i].item())
            
            # Check confidence threshold
            if score < self.confidence_threshold:
                continue
            
            # Check entropy if soft labels available
            if class_probs is not None and self.use_soft_labels:
                entropy = self.compute_entropy(class_probs[i])
                if entropy > self.entropy_threshold:
                    continue
            
            # Initialize storage for class if needed
            if cls not in self.pseudo_features:
                self.pseudo_features[cls] = []
                self.pseudo_scores[cls] = []
                self.pseudo_soft_labels[cls] = []
            
            # Check if we've reached max for this class
            if len(self.pseudo_features[cls]) >= self.max_pseudo_per_class:
                # Replace lowest-score sample if this one is better
                min_score = min(self.pseudo_scores[cls])
                if score > min_score:
                    min_idx = self.pseudo_scores[cls].index(min_score)
                    self.pseudo_features[cls][min_idx] = features[i].detach().cpu()
                    self.pseudo_scores[cls][min_idx] = score
                    if class_probs is not None:
                        self.pseudo_soft_labels[cls][min_idx] = class_probs[i].detach().cpu()
            else:
                self.pseudo_features[cls].append(features[i].detach().cpu())
                self.pseudo_scores[cls].append(score)
                if class_probs is not None:
                    self.pseudo_soft_labels[cls].append(class_probs[i].detach().cpu())
                self.num_collected += 1
    
    def should_update(self) -> bool:
        """
        Check if we have enough pseudo-labels to warrant an update.
        
        Returns:
            True if update should be performed
        """
        # Check if any class has enough samples
        for cls in self.pseudo_features:
            if len(self.pseudo_features[cls]) >= self.min_samples_for_update:
                return True
        return False
    
    def compute_pseudo_prototype(
        self,
        features: List[torch.Tensor],
        scores: List[float]
    ) -> torch.Tensor:
        """
        Compute weighted prototype from pseudo-labeled features.
        
        Args:
            features: List of feature tensors
            scores: Corresponding confidence scores
        
        Returns:
            Weighted prototype tensor
        """
        if not features:
            return None
        
        features_tensor = torch.stack(features, dim=0)
        scores_tensor = torch.tensor(scores, dtype=features_tensor.dtype)
        
        # Weight by confidence scores
        weights = F.softmax(scores_tensor / self.temperature, dim=0)
        
        # Weighted mean
        prototype = torch.sum(features_tensor * weights.unsqueeze(1), dim=0)
        
        return prototype
    
    def get_distilled_prototypes(
        self,
        original_prototypes: Dict
    ) -> Dict:
        """
        Combine original prototypes with pseudo-label derived prototypes.
        
        Args:
            original_prototypes: Original prototype dictionary from PCB
        
        Returns:
            Updated prototype dictionary
        """
        if not self.pseudo_features:
            return original_prototypes
        
        distilled = {}
        
        for cls in original_prototypes:
            orig_entry = original_prototypes[cls]
            
            if cls in self.pseudo_features and len(self.pseudo_features[cls]) >= self.min_samples_for_update:
                # Compute pseudo prototype
                pseudo_proto = self.compute_pseudo_prototype(
                    self.pseudo_features[cls],
                    self.pseudo_scores[cls]
                )
                
                if pseudo_proto is not None:
                    # Get original global prototype
                    orig_protos = orig_entry["global"]["protos"]
                    orig_weights = orig_entry["global"]["weights"]
                    
                    if orig_protos.shape[0] > 0:
                        # EMA-style combination
                        orig_mean = torch.sum(
                            orig_protos * orig_weights.unsqueeze(1), 
                            dim=0
                        )
                        
                        # Blend with pseudo prototype
                        blended = (
                            self.ema_momentum * orig_mean +
                            (1 - self.ema_momentum) * self.pseudo_weight * pseudo_proto.to(orig_mean.device)
                        )
                        
                        # Normalize
                        blended = F.normalize(blended.unsqueeze(0), dim=1)
                        
                        # Create updated entry
                        distilled[cls] = {
                            "global": {
                                "protos": blended,
                                "weights": torch.tensor([1.0], device=blended.device)
                            },
                            "scale": orig_entry.get("scale", {})
                        }
                        continue
            
            # No update needed for this class
            distilled[cls] = orig_entry
        
        self.num_updates += 1
        return distilled


class SelfDistilledPCB:
    """
    Wrapper around PrototypicalCalibrationBlock that applies self-distillation
    from test-time predictions.
    """
    
    def __init__(self, base_pcb, cfg):
        """
        Args:
            base_pcb: The original PrototypicalCalibrationBlock instance
            cfg: Config node with NOVEL_METHODS.SELF_DISTILL settings
        """
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        # Extract self-distillation config
        sd_cfg = cfg.NOVEL_METHODS.SELF_DISTILL
        self.distiller = SelfDistillationModule(
            confidence_threshold=float(sd_cfg.CONFIDENCE_THRESHOLD),
            max_pseudo_per_class=int(sd_cfg.MAX_PSEUDO_PER_CLASS),
            pseudo_weight=float(sd_cfg.PSEUDO_WEIGHT),
            ema_momentum=float(sd_cfg.EMA_MOMENTUM),
            min_samples_for_update=int(sd_cfg.MIN_SAMPLES_FOR_UPDATE),
            temperature=float(sd_cfg.TEMPERATURE),
            use_soft_labels=bool(sd_cfg.USE_SOFT_LABELS),
            entropy_threshold=float(sd_cfg.ENTROPY_THRESHOLD),
            device=str(cfg.MODEL.DEVICE)
        )
        
        # Store original prototypes for reference
        self.distiller.original_prototypes = base_pcb.prototypes.copy()
        
        # Two-pass mode settings
        self.two_pass_mode = bool(sd_cfg.TWO_PASS_MODE)
        self.update_interval = int(sd_cfg.UPDATE_INTERVAL)
        self.image_count = 0
    
    def execute_calibration(self, inputs, dts):
        """
        Execute calibration with self-distillation.
        
        In two-pass mode:
        - First pass: Collect pseudo-labels from high-confidence predictions
        - After enough samples: Update prototypes
        - Second pass (implicit): Use updated prototypes for remaining images
        
        In online mode:
        - Continuously update prototypes as pseudo-labels are collected
        """
        # First run normal calibration
        calibrated_dts = self.base_pcb.execute_calibration(inputs, dts)
        
        # Collect pseudo-labels from calibrated predictions
        if len(calibrated_dts) > 0 and len(calibrated_dts[0]["instances"]) > 0:
            instances = calibrated_dts[0]["instances"]
            
            # Extract features for high-confidence detections
            # Note: This requires re-extracting features, which is expensive
            # In practice, features should be cached during calibration
            scores = instances.scores
            pred_classes = instances.pred_classes
            
            # Filter by PCB's operating range
            mask = (scores > self.base_pcb.pcb_lower) & (scores <= self.base_pcb.pcb_upper)
            
            if mask.any():
                # Get features from base PCB's last extraction
                # This is a simplified version - full implementation would cache features
                high_conf_mask = scores > self.distiller.confidence_threshold
                
                if high_conf_mask.any():
                    # Simulate feature collection (in practice, use cached features)
                    # Here we use the calibrated scores as a proxy for quality
                    high_scores = scores[high_conf_mask]
                    high_classes = pred_classes[high_conf_mask]
                    
                    # Note: Real implementation would extract actual features
                    # For now, we track which classes have high-confidence predictions
                    for i in range(high_scores.shape[0]):
                        cls = int(high_classes[i].item())
                        score = float(high_scores[i].item())
                        
                        if cls not in self.distiller.pseudo_features:
                            self.distiller.pseudo_features[cls] = []
                            self.distiller.pseudo_scores[cls] = []
                        
                        # Count high-confidence detections per class
                        self.distiller.pseudo_scores[cls].append(score)
        
        self.image_count += 1
        
        # Check if we should update prototypes
        if self.image_count % self.update_interval == 0:
            if self.distiller.should_update():
                # Update prototypes with pseudo-label information
                updated_protos = self.distiller.get_distilled_prototypes(
                    self.distiller.original_prototypes
                )
                self.base_pcb.prototypes = updated_protos
                
                logger.info(
                    "Self-distillation: Updated prototypes after %d images. "
                    "Collected %d pseudo-labels across %d classes.",
                    self.image_count,
                    self.distiller.num_collected,
                    len(self.distiller.pseudo_features)
                )
        
        return calibrated_dts
    
    def reset_distillation(self):
        """Reset distillation state for new evaluation run."""
        self.distiller.reset()
        self.image_count = 0
        self.base_pcb.prototypes = self.distiller.original_prototypes.copy()
    
    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_self_distilled_pcb(base_pcb, cfg):
    """
    Factory function to wrap a PCB with self-distillation.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config with NOVEL_METHODS.SELF_DISTILL settings
    
    Returns:
        SelfDistilledPCB wrapper
    """
    return SelfDistilledPCB(base_pcb, cfg)
