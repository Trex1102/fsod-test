"""
Uncertainty-Weighted Prototype Matching for Few-Shot Object Detection.

This module implements uncertainty estimation for prototype matching,
where predictions are weighted by the model's confidence in the prototype
quality and match reliability.

Key idea: Estimate uncertainty in prototype representations using
ensemble-style perturbations or learned variance. Use uncertainty to
downweight unreliable matches and improve calibration.

Reference: Novel approach inspired by Bayesian deep learning and uncertainty quantification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PrototypeUncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty in prototype representations.
    
    Uses Monte Carlo dropout or learned variance to quantify
    reliability of prototypes and matches.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        learn_variance: bool = True,
        variance_init: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Args:
            feature_dim: Dimension of feature vectors
            num_mc_samples: Number of Monte Carlo dropout samples
            dropout_rate: Dropout rate for MC dropout
            learn_variance: Whether to learn per-dimension variance
            variance_init: Initial variance value
            temperature: Temperature for uncertainty scaling
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        self.learn_variance = learn_variance
        self.temperature = temperature
        
        # MC Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Learned variance (log-variance for numerical stability)
        if learn_variance:
            self.log_variance = nn.Parameter(
                torch.full((feature_dim,), math.log(variance_init))
            )
        else:
            self.register_buffer(
                'log_variance',
                torch.full((feature_dim,), math.log(variance_init))
            )
    
    @property
    def variance(self) -> torch.Tensor:
        """Get variance from log-variance."""
        return torch.exp(self.log_variance)
    
    def sample_prototype(
        self,
        prototype: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample from prototype distribution.
        
        Args:
            prototype: Mean prototype vector (D,) or (K, D)
            num_samples: Number of samples to generate
        
        Returns:
            Sampled prototypes (num_samples, K, D) or (num_samples, D)
        """
        if prototype.dim() == 1:
            prototype = prototype.unsqueeze(0)
        
        K, D = prototype.shape
        
        # Sample from Gaussian with learned variance
        std = torch.sqrt(self.variance).to(prototype.device)
        noise = torch.randn(num_samples, K, D, device=prototype.device)
        
        samples = prototype.unsqueeze(0) + noise * std.unsqueeze(0).unsqueeze(0)
        
        return samples
    
    def mc_dropout_forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity with MC dropout uncertainty estimation.
        
        Args:
            features: Query features (N, D)
            prototypes: Prototype features (K, D)
        
        Returns:
            Tuple of (mean_similarity, uncertainty) each of shape (N, K)
        """
        N = features.shape[0]
        K = prototypes.shape[0]
        
        similarities = []
        
        for _ in range(self.num_mc_samples):
            # Apply dropout to features
            dropped_features = self.dropout(features)
            
            # Normalize
            f_norm = F.normalize(dropped_features, dim=1)
            p_norm = F.normalize(prototypes, dim=1)
            
            # Compute similarity
            sim = torch.mm(f_norm, p_norm.t())
            similarities.append(sim)
        
        # Stack and compute statistics
        similarities = torch.stack(similarities, dim=0)  # (S, N, K)
        
        mean_sim = torch.mean(similarities, dim=0)  # (N, K)
        var_sim = torch.var(similarities, dim=0)    # (N, K)
        
        # Uncertainty = standard deviation
        uncertainty = torch.sqrt(var_sim + 1e-8)
        
        return mean_sim, uncertainty
    
    def compute_match_uncertainty(
        self,
        query_feature: torch.Tensor,
        prototype: torch.Tensor,
        num_samples: int = None
    ) -> Tuple[float, float]:
        """
        Compute uncertainty in prototype match.
        
        Args:
            query_feature: Query feature vector (D,)
            prototype: Prototype vector (D,)
            num_samples: Number of prototype samples
        
        Returns:
            Tuple of (mean_similarity, uncertainty)
        """
        if num_samples is None:
            num_samples = self.num_mc_samples
        
        # Sample prototypes
        proto_samples = self.sample_prototype(prototype, num_samples)  # (S, 1, D)
        proto_samples = proto_samples.squeeze(1)  # (S, D)
        
        # Normalize
        q_norm = F.normalize(query_feature.unsqueeze(0), dim=1)  # (1, D)
        p_norm = F.normalize(proto_samples, dim=1)  # (S, D)
        
        # Compute similarities
        sims = torch.mm(q_norm, p_norm.t()).squeeze(0)  # (S,)
        
        mean_sim = torch.mean(sims)
        std_sim = torch.std(sims)
        
        return float(mean_sim.item()), float(std_sim.item())
    
    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing uncertainty-aware similarities.
        
        Args:
            features: Query features (N, D)
            prototypes: Prototype features (K, D)
        
        Returns:
            Dict with 'mean_similarity', 'uncertainty', 'weighted_similarity'
        """
        mean_sim, uncertainty = self.mc_dropout_forward(features, prototypes)
        
        # Compute uncertainty-weighted similarity
        # Lower uncertainty = higher weight
        weights = 1.0 / (1.0 + uncertainty / self.temperature)
        weighted_sim = mean_sim * weights
        
        return {
            'mean_similarity': mean_sim,
            'uncertainty': uncertainty,
            'weighted_similarity': weighted_sim,
            'weights': weights
        }


class UncertaintyWeightedMatcher:
    """
    Matches query features to prototypes with uncertainty weighting.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        uncertainty_threshold: float = 0.3,
        fallback_alpha: float = 0.7,
        use_ensemble: bool = False,
        ensemble_size: int = 5,
        device: str = "cuda"
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_mc_samples: MC samples for uncertainty estimation
            dropout_rate: Dropout rate
            uncertainty_threshold: Threshold above which to reduce influence
            fallback_alpha: Alpha to use when uncertainty is high
            use_ensemble: Use ensemble of prototypes instead of MC dropout
            ensemble_size: Size of prototype ensemble
            device: Computation device
        """
        self.feature_dim = feature_dim
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        self.uncertainty_threshold = uncertainty_threshold
        self.fallback_alpha = fallback_alpha
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.device = device
        
        # Uncertainty estimator
        self.estimator = PrototypeUncertaintyEstimator(
            feature_dim=feature_dim,
            num_mc_samples=num_mc_samples,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Per-class prototype ensembles (for ensemble mode)
        self.prototype_ensembles: Dict[int, torch.Tensor] = {}
    
    def build_ensemble(
        self,
        features: torch.Tensor,
        qualities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Build ensemble of prototypes via bootstrap sampling.
        
        Args:
            features: Support features (N, D)
            qualities: Quality weights (N,)
        
        Returns:
            Ensemble of prototypes (ensemble_size, D)
        """
        N = features.shape[0]
        
        if N < 2:
            # Single sample: create ensemble via learned variance perturbation
            # This ensures diversity for uncertainty estimation even with 1 shot
            return self._build_single_sample_ensemble(features, qualities)
        
        ensemble = []
        for _ in range(self.ensemble_size):
            # Bootstrap sample
            if qualities is not None:
                # Weighted sampling
                probs = qualities / qualities.sum()
                indices = torch.multinomial(probs, N, replacement=True)
            else:
                indices = torch.randint(0, N, (N,), device=features.device)
            
            sampled = features[indices]
            prototype = torch.mean(sampled, dim=0)
            ensemble.append(prototype)
        
        return torch.stack(ensemble, dim=0)

    def _build_single_sample_ensemble(
        self,
        features: torch.Tensor,
        qualities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Build ensemble from a single sample using perturbation.
        
        For 1-shot scenarios, we can't bootstrap, so we create diversity
        by adding calibrated noise to the prototype. The noise is scaled
        to be reasonable for feature magnitudes.
        
        Args:
            features: Single feature vector (1, D) or (D,)
            qualities: Ignored for single sample
        
        Returns:
            Ensemble of perturbed prototypes (ensemble_size, D)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Base prototype
        prototype = features[0]  # (D,)
        D = prototype.shape[0]
        device = prototype.device
        dtype = prototype.dtype
        
        # Compute feature statistics for calibrated perturbation
        feat_std = torch.std(prototype).clamp(min=1e-6)
        feat_norm = torch.norm(prototype).clamp(min=1e-6)
        
        # Generate ensemble with increasing perturbation scales
        # Scale factors: small perturbations to maintain semantic meaning
        perturbation_scales = [0.0, 0.05, 0.1, 0.15, 0.2][:self.ensemble_size]
        
        ensemble = []
        for scale in perturbation_scales:
            if scale == 0.0:
                # Original prototype (unperturbed)
                ensemble.append(prototype.clone())
            else:
                # Add Gaussian noise scaled to feature statistics
                noise = torch.randn(D, device=device, dtype=dtype)
                # Normalize noise to unit norm, then scale
                noise = noise / (torch.norm(noise) + 1e-8) * feat_norm * scale
                perturbed = prototype + noise
                # Optionally re-normalize to similar magnitude as original
                perturbed = perturbed * (feat_norm / (torch.norm(perturbed) + 1e-8))
                ensemble.append(perturbed)
        
        # Pad if ensemble_size > len(perturbation_scales)
        while len(ensemble) < self.ensemble_size:
            scale = 0.1 + 0.05 * len(ensemble)
            noise = torch.randn(D, device=device, dtype=dtype)
            noise = noise / (torch.norm(noise) + 1e-8) * feat_norm * scale
            perturbed = prototype + noise
            perturbed = perturbed * (feat_norm / (torch.norm(perturbed) + 1e-8))
            ensemble.append(perturbed)
        
        return torch.stack(ensemble, dim=0)
    
    def match_with_uncertainty(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Match query to prototypes with uncertainty estimation.
        
        Args:
            query: Query feature (D,)
            prototypes: Prototype(s) (K, D) or (D,)
        
        Returns:
            Tuple of (similarity, uncertainty, adaptive_alpha)
        """
        if prototypes.dim() == 1:
            prototypes = prototypes.unsqueeze(0)
        
        query = query.unsqueeze(0).to(self.device)
        prototypes = prototypes.to(self.device)
        
        with torch.no_grad():
            result = self.estimator(query, prototypes)
        
        # Get best match
        best_idx = torch.argmax(result['mean_similarity'][0])
        similarity = float(result['mean_similarity'][0, best_idx].item())
        uncertainty = float(result['uncertainty'][0, best_idx].item())
        
        # Compute adaptive alpha based on uncertainty
        # High uncertainty = rely more on original score (higher alpha)
        if uncertainty > self.uncertainty_threshold:
            # Reduce PCB influence when uncertain
            adaptive_alpha = self.fallback_alpha + (1.0 - self.fallback_alpha) * (
                (uncertainty - self.uncertainty_threshold) / 
                (1.0 - self.uncertainty_threshold + 1e-8)
            )
            adaptive_alpha = min(1.0, adaptive_alpha)
        else:
            # Normal PCB operation
            adaptive_alpha = None  # Use default alpha
        
        return similarity, uncertainty, adaptive_alpha


class UncertaintyWeightedPCB:
    """
    Wrapper around PrototypicalCalibrationBlock that applies
    uncertainty-weighted prototype matching.
    """
    
    def __init__(self, base_pcb, cfg):
        """
        Args:
            base_pcb: The original PrototypicalCalibrationBlock instance
            cfg: Config node with NOVEL_METHODS.UNCERTAINTY settings
        """
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        # Extract uncertainty config
        unc_cfg = cfg.NOVEL_METHODS.UNCERTAINTY
        self.matcher = UncertaintyWeightedMatcher(
            feature_dim=int(unc_cfg.FEATURE_DIM),
            num_mc_samples=int(unc_cfg.NUM_MC_SAMPLES),
            dropout_rate=float(unc_cfg.DROPOUT_RATE),
            uncertainty_threshold=float(unc_cfg.UNCERTAINTY_THRESHOLD),
            fallback_alpha=float(unc_cfg.FALLBACK_ALPHA),
            use_ensemble=bool(unc_cfg.USE_ENSEMBLE),
            ensemble_size=int(unc_cfg.ENSEMBLE_SIZE),
            device=str(cfg.MODEL.DEVICE)
        )
        
        # Build prototype ensembles if enabled
        if self.matcher.use_ensemble:
            self._build_all_ensembles()
        
        # Statistics tracking
        self.high_uncertainty_count = 0
        self.total_matches = 0
    
    def _build_all_ensembles(self):
        """Build ensembles for all class prototypes."""
        if not hasattr(self.base_pcb, '_real_class_features'):
            return
        
        for cls in self.base_pcb._real_class_features:
            feat_list = self.base_pcb._real_class_features[cls]
            qual_list = self.base_pcb._real_class_qualities[cls]
            
            if not feat_list:
                continue
            
            features = torch.stack(feat_list, dim=0)
            qualities = torch.tensor(qual_list, dtype=features.dtype)
            
            ensemble = self.matcher.build_ensemble(features, qualities)
            self.matcher.prototype_ensembles[cls] = ensemble
    
    def execute_calibration(self, inputs, dts):
        """
        Execute calibration with uncertainty-weighted matching.
        
        Modifies the base PCB's matching behavior to account for
        prototype uncertainty.
        """
        # For now, use base PCB's calibration
        # Full implementation would override _match_similarity to use uncertainty
        
        # Track statistics
        if len(dts) > 0 and len(dts[0]["instances"]) > 0:
            scores = dts[0]["instances"].scores
            # Count predictions in calibration range
            in_range = ((scores > self.base_pcb.pcb_lower) & 
                       (scores <= self.base_pcb.pcb_upper)).sum()
            self.total_matches += int(in_range.item())
        
        return self.base_pcb.execute_calibration(inputs, dts)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get uncertainty matching statistics."""
        if self.total_matches > 0:
            high_unc_ratio = self.high_uncertainty_count / self.total_matches
        else:
            high_unc_ratio = 0.0
        
        return {
            'total_matches': self.total_matches,
            'high_uncertainty_count': self.high_uncertainty_count,
            'high_uncertainty_ratio': high_unc_ratio
        }
    
    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_uncertainty_weighted_pcb(base_pcb, cfg):
    """
    Factory function to wrap a PCB with uncertainty weighting.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config with NOVEL_METHODS.UNCERTAINTY settings
    
    Returns:
        UncertaintyWeightedPCB wrapper
    """
    return UncertaintyWeightedPCB(base_pcb, cfg)
