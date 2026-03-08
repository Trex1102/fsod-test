"""
Feature Hallucination for Few-Shot Object Detection

This module implements feature hallucination based on transferring intra-class
variance from base classes to novel classes. The key idea is:

1. Collect RoI features from base class training data
2. Compute per-class statistics (mean, covariance) and cross-class variance patterns
3. For novel classes, use the few available samples + transferred variance to
   generate diverse synthetic features

This approach doesn't rely on external embeddings (like CLIP) and instead
learns variance patterns directly from the detection feature space.

Reference: 
- "Hallucinating Visual Instances in Total Absentia" (Zhang et al., ECCV 2020)
- Adapted for FSOD setting
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FeatureHallucinator(nn.Module):
    """
    Generates synthetic features for few-shot classes by:
    1. Learning a shared covariance structure from base classes
    2. Applying this variance to novel class prototypes
    
    Two modes supported:
    - 'gaussian': Sample from N(prototype, transferred_covariance)
    - 'delta': Add learned delta vectors to prototype (more structured)
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_deltas: int = 32,
        mode: str = "gaussian",
        variance_scale: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_deltas = num_deltas
        self.mode = mode
        self.variance_scale = variance_scale
        
        # Shared covariance (low-rank approximation for efficiency)
        # Full covariance would be 2048x2048 = 4M params
        # Low-rank: 2048 x rank = much smaller
        self.covariance_rank = min(256, feature_dim // 8)
        
        # Learnable components (initialized from base class statistics)
        self.register_buffer("global_mean", torch.zeros(feature_dim))
        self.register_buffer("global_std", torch.ones(feature_dim))
        
        # Low-rank covariance factors: Sigma ≈ U @ U.T + diag(D)
        self.register_buffer(
            "cov_U", torch.zeros(feature_dim, self.covariance_rank)
        )
        self.register_buffer("cov_D", torch.ones(feature_dim))
        
        # For delta mode: learned direction vectors
        if mode == "delta":
            self.delta_directions = nn.Parameter(
                torch.randn(num_deltas, feature_dim) * 0.1
            )
            self.delta_scales = nn.Parameter(torch.ones(num_deltas) * 0.5)
    
    def fit_from_base_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ):
        """
        Fit hallucinator from base class RoI features.
        
        Args:
            features: [N, D] tensor of RoI features from base classes
            labels: [N] tensor of class labels
            normalize: Whether to z-normalize features before fitting
        """
        device = features.device
        
        # Global statistics
        self.global_mean = features.mean(dim=0)
        self.global_std = features.std(dim=0).clamp(min=1e-6)
        
        if normalize:
            features_norm = (features - self.global_mean) / self.global_std
        else:
            features_norm = features - self.global_mean
        
        # Compute pooled within-class covariance
        # This captures the "typical" way features vary around their class mean
        unique_labels = labels.unique()
        within_cov_sum = torch.zeros(
            self.feature_dim, self.feature_dim, device=device
        )
        total_samples = 0
        
        for lbl in unique_labels:
            mask = labels == lbl
            class_feats = features_norm[mask]
            n = class_feats.shape[0]
            if n > 1:
                class_mean = class_feats.mean(dim=0, keepdim=True)
                centered = class_feats - class_mean
                within_cov_sum += centered.T @ centered
                total_samples += n - 1  # degrees of freedom
        
        if total_samples > 0:
            pooled_cov = within_cov_sum / total_samples
        else:
            pooled_cov = torch.eye(self.feature_dim, device=device)
        
        # Low-rank approximation via SVD
        # Sigma ≈ U @ U.T + diag(residual)
        try:
            U, S, _ = torch.svd_lowrank(pooled_cov, q=self.covariance_rank)
            self.cov_U = U * S.sqrt().unsqueeze(0)  # [D, rank]
            
            # Residual diagonal
            reconstructed = self.cov_U @ self.cov_U.T
            residual = torch.diag(pooled_cov) - torch.diag(reconstructed)
            self.cov_D = residual.clamp(min=1e-6)
        except Exception:
            # Fallback to diagonal covariance
            self.cov_U = torch.zeros(
                self.feature_dim, self.covariance_rank, device=device
            )
            self.cov_D = torch.diag(pooled_cov).clamp(min=1e-6)
        
        # For delta mode: initialize directions from principal components
        if self.mode == "delta" and hasattr(self, "delta_directions"):
            # Use top eigenvectors as delta directions
            try:
                _, S_full, V = torch.svd(pooled_cov)
                num_dirs = min(self.num_deltas, V.shape[1])
                self.delta_directions.data[:num_dirs] = V[:, :num_dirs].T
                self.delta_scales.data[:num_dirs] = S_full[:num_dirs].sqrt()
            except Exception:
                pass  # Keep random initialization
    
    @torch.no_grad()
    def generate(
        self,
        prototypes: torch.Tensor,
        num_per_class: int = 30,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate hallucinated features for given class prototypes.
        
        Args:
            prototypes: [C, D] class prototype features (mean of few-shot samples)
            num_per_class: Number of features to generate per class
            temperature: Scale factor for variance (higher = more diverse)
            
        Returns:
            features: [C * num_per_class, D] generated features
            labels: [C * num_per_class] class labels
        """
        device = prototypes.device
        num_classes = prototypes.shape[0]
        
        all_features = []
        all_labels = []
        
        for c in range(num_classes):
            proto = prototypes[c]  # [D]
            
            if self.mode == "gaussian":
                # Sample from N(proto, transferred_cov * temperature)
                features_c = self._sample_gaussian(
                    proto, num_per_class, temperature
                )
            else:  # delta mode
                features_c = self._sample_delta(
                    proto, num_per_class, temperature
                )
            
            all_features.append(features_c)
            all_labels.append(
                torch.full((num_per_class,), c, dtype=torch.long, device=device)
            )
        
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    
    def _sample_gaussian(
        self, proto: torch.Tensor, n: int, temperature: float
    ) -> torch.Tensor:
        """Sample from multivariate Gaussian with low-rank covariance."""
        device = proto.device
        scale = self.variance_scale * temperature
        
        # Sample: x = proto + U @ z1 + sqrt(D) * z2
        # where z1 ~ N(0, I_rank), z2 ~ N(0, I_D)
        z1 = torch.randn(n, self.covariance_rank, device=device)
        z2 = torch.randn(n, self.feature_dim, device=device)
        
        # Low-rank component
        low_rank = z1 @ self.cov_U.T  # [n, D]
        
        # Diagonal component
        diag = z2 * self.cov_D.sqrt()  # [n, D]
        
        # Combine
        samples = proto.unsqueeze(0) + scale * (low_rank + diag)
        
        # Apply ReLU to match RoI feature distribution (typically non-negative)
        samples = F.relu(samples)
        
        return samples
    
    def _sample_delta(
        self, proto: torch.Tensor, n: int, temperature: float
    ) -> torch.Tensor:
        """Generate by adding scaled delta directions to prototype."""
        device = proto.device
        scale = self.variance_scale * temperature
        
        # For each sample, randomly combine delta directions
        # This creates more structured variations
        weights = torch.randn(n, self.num_deltas, device=device)
        weights = F.softmax(weights * 2.0, dim=1)  # Sparse weighting
        
        # Weighted combination of deltas
        deltas = weights @ (self.delta_directions * self.delta_scales.unsqueeze(1))
        
        samples = proto.unsqueeze(0) + scale * deltas
        samples = F.relu(samples)
        
        return samples


def collect_roi_statistics(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 50000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect RoI features from base class data for hallucinator training.
    
    Args:
        model: Detection model with roi_heads
        dataloader: DataLoader for base class training data
        device: Device to use
        max_samples: Maximum number of RoI samples to collect
        
    Returns:
        features: [N, D] collected RoI features
        labels: [N] class labels
    """
    from detectron2.structures import Boxes
    
    model.eval()
    
    feat_chunks = []
    label_chunks = []
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if total >= max_samples:
                break
            
            images = model.preprocess_image(batch)
            backbone_feats = model.backbone(images.tensor)
            roi_feature_name = list(backbone_feats.keys())[0]
            feat_map = backbone_feats[roi_feature_name]
            
            for i, sample in enumerate(batch):
                if "instances" not in sample:
                    continue
                inst = sample["instances"].to(device)
                if len(inst) == 0:
                    continue
                
                gt_boxes = inst.gt_boxes.tensor
                gt_classes = inst.gt_classes
                
                # Extract RoI features for GT boxes
                pooled = model.roi_heads._shared_roi_transform(
                    [feat_map[i : i + 1]],
                    [Boxes(gt_boxes)],
                )
                pooled = pooled.mean(dim=[2, 3]).detach().cpu()
                
                feat_chunks.append(pooled)
                label_chunks.append(gt_classes.cpu())
                total += pooled.shape[0]
                
                if total >= max_samples:
                    break
    
    features = torch.cat(feat_chunks, dim=0)[:max_samples]
    labels = torch.cat(label_chunks, dim=0)[:max_samples]
    
    return features, labels


def build_hallucinated_feature_bank(
    novel_prototypes: torch.Tensor,
    novel_labels: torch.Tensor,
    hallucinator: FeatureHallucinator,
    num_gen_per_class: int = 30,
    temperature: float = 1.0,
    include_originals: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a feature bank using hallucinated features.
    
    Args:
        novel_prototypes: [C, D] mean features for each novel class
        novel_labels: [C] class indices
        hallucinator: Fitted FeatureHallucinator
        num_gen_per_class: Number of synthetic features per class
        temperature: Variance scale
        include_originals: Whether to include original prototypes in bank
        
    Returns:
        Dictionary with 'features' and 'labels' tensors
    """
    # Generate synthetic features
    gen_features, gen_labels = hallucinator.generate(
        novel_prototypes,
        num_per_class=num_gen_per_class,
        temperature=temperature,
    )
    
    if include_originals:
        # Expand prototypes to match label indexing
        features = torch.cat([novel_prototypes, gen_features], dim=0)
        labels = torch.cat([novel_labels, gen_labels], dim=0)
    else:
        features = gen_features
        labels = gen_labels
    
    return {
        "features": features,
        "labels": labels,
        "num_gen_per_class": num_gen_per_class,
        "temperature": temperature,
        "method": "hallucination",
    }
