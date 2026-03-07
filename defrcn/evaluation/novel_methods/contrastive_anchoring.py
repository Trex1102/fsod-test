"""
Contrastive Prototype Anchoring for Few-Shot Object Detection.

This module implements a contrastive loss that anchors class prototypes
apart from each other while pulling same-class instances together.
Applied during fine-tuning to improve prototype discriminability.

Key idea: Use InfoNCE-style contrastive loss on prototype representations
to maximize inter-class separation while maintaining intra-class cohesion.

Reference: Novel approach inspired by supervised contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ContrastivePrototypeAnchor(nn.Module):
    """
    Contrastive loss module for anchoring class prototypes in feature space.
    
    During fine-tuning, this loss encourages:
    1. Same-class features to cluster around their prototype
    2. Different-class prototypes to be well-separated
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        temperature: float = 0.07,
        margin: float = 0.5,
        proto_momentum: float = 0.99,
        use_hard_negatives: bool = True,
        num_hard_negatives: int = 3,
        loss_weight: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of feature vectors
            temperature: Temperature for contrastive softmax
            margin: Margin for prototype separation
            proto_momentum: EMA momentum for prototype updates
            use_hard_negatives: Whether to mine hard negatives
            num_hard_negatives: Number of hard negatives per anchor
            loss_weight: Weight for the contrastive loss
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.margin = margin
        self.proto_momentum = proto_momentum
        self.use_hard_negatives = use_hard_negatives
        self.num_hard_negatives = num_hard_negatives
        self.loss_weight = loss_weight
        
        # Prototype bank (registered as buffer for state_dict)
        self.register_buffer('prototype_bank', None)
        self.register_buffer('prototype_counts', None)
        self.num_classes = 0
        
        # Projection head for contrastive space
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
    def initialize_prototypes(self, num_classes: int, device: str = 'cuda'):
        """
        Initialize prototype bank for given number of classes.
        
        Args:
            num_classes: Number of classes (including background if any)
            device: Device to create tensors on
        """
        self.num_classes = num_classes
        self.prototype_bank = torch.zeros(
            num_classes, self.feature_dim // 2,
            device=device
        )
        self.prototype_counts = torch.zeros(num_classes, device=device)
    
    def update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Update prototype bank with EMA of class features.
        
        Args:
            features: Feature vectors of shape (N, D)
            labels: Class labels of shape (N,)
        """
        if self.prototype_bank is None:
            return
        
        # Project features
        projected = self.projector(features)
        projected = F.normalize(projected, dim=1)
        
        # Update each class prototype
        for cls in torch.unique(labels):
            cls_idx = int(cls.item())
            if cls_idx < 0 or cls_idx >= self.num_classes:
                continue
            
            cls_mask = labels == cls
            cls_features = projected[cls_mask]
            cls_mean = cls_features.mean(dim=0)
            
            if self.prototype_counts[cls_idx] == 0:
                # First update - initialize directly
                self.prototype_bank[cls_idx] = cls_mean
            else:
                # EMA update
                self.prototype_bank[cls_idx] = (
                    self.proto_momentum * self.prototype_bank[cls_idx] +
                    (1 - self.proto_momentum) * cls_mean
                )
            
            self.prototype_counts[cls_idx] += cls_features.shape[0]
    
    def compute_infoNCE_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            features: Feature vectors of shape (N, D)
            labels: Class labels of shape (N,)
        
        Returns:
            Scalar loss value
        """
        if self.prototype_bank is None or features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Project features
        projected = self.projector(features)
        projected = F.normalize(projected, dim=1)
        
        # Normalize prototypes
        prototypes = F.normalize(self.prototype_bank, dim=1)
        
        # Compute similarity to all prototypes
        # Shape: (N, num_classes)
        similarities = torch.mm(projected, prototypes.t()) / self.temperature
        
        # Create target (index of correct class for each sample)
        # Filter out invalid labels
        valid_mask = (labels >= 0) & (labels < self.num_classes)
        if not valid_mask.any():
            return torch.tensor(0.0, device=features.device)
        
        valid_sims = similarities[valid_mask]
        valid_labels = labels[valid_mask]
        
        # InfoNCE loss (cross-entropy with prototypes as classes)
        loss = F.cross_entropy(valid_sims, valid_labels.long())
        
        return loss
    
    def compute_margin_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin-based separation loss between prototypes.
        
        Encourages prototypes to be at least `margin` apart in cosine distance.
        
        Args:
            features: Feature vectors of shape (N, D)
            labels: Class labels of shape (N,)
        
        Returns:
            Scalar loss value
        """
        if self.prototype_bank is None:
            return torch.tensor(0.0, device=features.device)
        
        # Normalize prototypes
        prototypes = F.normalize(self.prototype_bank, dim=1)
        
        # Compute pairwise similarities
        # Shape: (num_classes, num_classes)
        proto_sims = torch.mm(prototypes, prototypes.t())
        
        # Mask out diagonal (self-similarity)
        mask = 1.0 - torch.eye(self.num_classes, device=prototypes.device)
        
        # Margin loss: penalize similarities above (1 - margin)
        # Higher similarity = closer prototypes = bad
        margin_violations = F.relu(proto_sims - (1.0 - self.margin)) * mask
        
        # Average over all pairs
        num_pairs = mask.sum()
        if num_pairs > 0:
            loss = margin_violations.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=features.device)
        
        return loss
    
    def compute_hard_negative_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss focusing on hard negatives.
        
        Hard negatives are the closest prototypes of different classes.
        
        Args:
            features: Feature vectors of shape (N, D)
            labels: Class labels of shape (N,)
        
        Returns:
            Scalar loss value
        """
        if self.prototype_bank is None or features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Project features
        projected = self.projector(features)
        projected = F.normalize(projected, dim=1)
        
        # Normalize prototypes
        prototypes = F.normalize(self.prototype_bank, dim=1)
        
        # Compute similarities
        similarities = torch.mm(projected, prototypes.t())
        
        total_loss = torch.tensor(0.0, device=features.device)
        count = 0
        
        for i in range(features.shape[0]):
            cls = int(labels[i].item())
            if cls < 0 or cls >= self.num_classes:
                continue
            
            # Positive similarity
            pos_sim = similarities[i, cls]
            
            # Get negative similarities (all other classes)
            neg_mask = torch.ones(self.num_classes, dtype=torch.bool, device=features.device)
            neg_mask[cls] = False
            neg_sims = similarities[i, neg_mask]
            
            if neg_sims.numel() == 0:
                continue
            
            # Select hard negatives (highest similarity among negatives)
            k = min(self.num_hard_negatives, neg_sims.numel())
            hard_neg_sims, _ = torch.topk(neg_sims, k)
            
            # Triplet-style loss: push away hard negatives
            for neg_sim in hard_neg_sims:
                triplet_loss = F.relu(neg_sim - pos_sim + self.margin)
                total_loss = total_loss + triplet_loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        update_prototypes: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive anchoring loss.
        
        Args:
            features: Feature vectors of shape (N, D)
            labels: Class labels of shape (N,)
            update_prototypes: Whether to update prototype bank
        
        Returns:
            Dict with 'loss' and component losses
        """
        if update_prototypes:
            with torch.no_grad():
                self.update_prototypes(features.detach(), labels.detach())
        
        # Compute loss components
        infoNCE_loss = self.compute_infoNCE_loss(features, labels)
        margin_loss = self.compute_margin_loss(features, labels)
        
        if self.use_hard_negatives:
            hard_neg_loss = self.compute_hard_negative_loss(features, labels)
        else:
            hard_neg_loss = torch.tensor(0.0, device=features.device)
        
        # Combined loss
        total_loss = (
            infoNCE_loss + 
            0.5 * margin_loss + 
            0.5 * hard_neg_loss
        ) * self.loss_weight
        
        return {
            'loss': total_loss,
            'infoNCE_loss': infoNCE_loss,
            'margin_loss': margin_loss,
            'hard_neg_loss': hard_neg_loss
        }


class ContrastiveAnchoredPCB:
    """
    Wrapper around PrototypicalCalibrationBlock that uses contrastively
    learned prototype representations for improved calibration.
    """
    
    def __init__(self, base_pcb, cfg, contrastive_module: Optional[ContrastivePrototypeAnchor] = None):
        """
        Args:
            base_pcb: The original PrototypicalCalibrationBlock instance
            cfg: Config node with NOVEL_METHODS.CONTRASTIVE settings
            contrastive_module: Pre-trained ContrastivePrototypeAnchor (optional)
        """
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        if contrastive_module is not None:
            self.contrastive = contrastive_module
        else:
            # Create new module from config
            cont_cfg = cfg.NOVEL_METHODS.CONTRASTIVE
            self.contrastive = ContrastivePrototypeAnchor(
                feature_dim=int(cont_cfg.FEATURE_DIM),
                temperature=float(cont_cfg.TEMPERATURE),
                margin=float(cont_cfg.MARGIN),
                proto_momentum=float(cont_cfg.PROTO_MOMENTUM),
                use_hard_negatives=bool(cont_cfg.USE_HARD_NEGATIVES),
                num_hard_negatives=int(cont_cfg.NUM_HARD_NEGATIVES),
                loss_weight=float(cont_cfg.LOSS_WEIGHT)
            )
    
    def execute_calibration(self, inputs, dts):
        """
        Delegate to base PCB's execute_calibration.
        Contrastive learning happens during training, not inference.
        """
        return self.base_pcb.execute_calibration(inputs, dts)
    
    def get_training_loss_module(self) -> ContrastivePrototypeAnchor:
        """
        Return the contrastive module for use in training loop.
        """
        return self.contrastive
    
    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_contrastive_anchored_pcb(base_pcb, cfg, contrastive_module=None):
    """
    Factory function to wrap a PCB with contrastive anchoring.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config with NOVEL_METHODS.CONTRASTIVE settings
        contrastive_module: Optional pre-trained module
    
    Returns:
        ContrastiveAnchoredPCB wrapper
    """
    return ContrastiveAnchoredPCB(base_pcb, cfg, contrastive_module)
