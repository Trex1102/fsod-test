"""
Frequency-Domain Prototype Augmentation for Few-Shot Object Detection.

This module implements DCT-based frequency decomposition and augmentation
of support prototypes to increase prototype diversity without additional
training data.

Key idea: Decompose prototypes into low-frequency (semantic content) and
high-frequency (fine-grained details) components. Create augmented prototypes
by mixing frequency bands across instances within the same class.

Reference: Novel approach inspired by frequency-domain style transfer methods.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def dct_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Discrete Cosine Transform (Type-II) along the last dimension.
    
    Args:
        x: Input tensor of shape (..., N)
    
    Returns:
        DCT coefficients of shape (..., N)
    """
    N = x.shape[-1]
    # Create DCT-II matrix
    n = torch.arange(N, dtype=x.dtype, device=x.device)
    k = torch.arange(N, dtype=x.dtype, device=x.device)
    # DCT-II formula: X_k = sum_{n=0}^{N-1} x_n * cos(pi/N * (n + 0.5) * k)
    dct_matrix = torch.cos(math.pi / N * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1))
    # Normalization factor
    dct_matrix[0, :] *= 1.0 / math.sqrt(N)
    dct_matrix[1:, :] *= math.sqrt(2.0 / N)
    
    # Apply DCT via matrix multiplication
    return torch.matmul(x, dct_matrix.T)


def idct_1d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Inverse Discrete Cosine Transform (Type-II) along the last dimension.
    
    Args:
        X: DCT coefficients of shape (..., N)
    
    Returns:
        Reconstructed signal of shape (..., N)
    """
    N = X.shape[-1]
    n = torch.arange(N, dtype=X.dtype, device=X.device)
    k = torch.arange(N, dtype=X.dtype, device=X.device)
    # IDCT-II formula
    idct_matrix = torch.cos(math.pi / N * (n.unsqueeze(1) + 0.5) * k.unsqueeze(0))
    idct_matrix[:, 0] *= 1.0 / math.sqrt(N)
    idct_matrix[:, 1:] *= math.sqrt(2.0 / N)
    
    return torch.matmul(X, idct_matrix.T)


class FrequencyAugmentor:
    """
    Augments prototypes by decomposing them into frequency bands and
    performing controlled mixing across instances.
    """
    
    def __init__(
        self,
        low_freq_ratio: float = 0.3,
        high_freq_ratio: float = 0.3,
        num_augmented: int = 3,
        mix_alpha: float = 0.5,
        preserve_norm: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            low_freq_ratio: Fraction of DCT coefficients considered low-frequency (0.0-1.0)
            high_freq_ratio: Fraction of DCT coefficients considered high-frequency (0.0-1.0)
            num_augmented: Number of augmented prototypes to generate per original
            mix_alpha: Interpolation factor for frequency mixing (0.0-1.0)
            preserve_norm: Whether to preserve the L2 norm of original prototypes
            device: Device to run computations on
        """
        self.low_freq_ratio = low_freq_ratio
        self.high_freq_ratio = high_freq_ratio
        self.num_augmented = num_augmented
        self.mix_alpha = mix_alpha
        self.preserve_norm = preserve_norm
        self.device = device
    
    def decompose_frequency(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose features into low, mid, and high frequency components.
        
        Args:
            features: Feature tensor of shape (N, D)
        
        Returns:
            Tuple of (low_freq, mid_freq, high_freq) each of shape (N, D)
        """
        N, D = features.shape
        
        # Compute DCT
        dct_coeffs = dct_1d(features)
        
        # Determine frequency band boundaries
        low_cutoff = int(D * self.low_freq_ratio)
        high_cutoff = int(D * (1.0 - self.high_freq_ratio))
        
        # Create frequency masks
        low_mask = torch.zeros(D, device=features.device, dtype=features.dtype)
        mid_mask = torch.zeros(D, device=features.device, dtype=features.dtype)
        high_mask = torch.zeros(D, device=features.device, dtype=features.dtype)
        
        low_mask[:low_cutoff] = 1.0
        mid_mask[low_cutoff:high_cutoff] = 1.0
        high_mask[high_cutoff:] = 1.0
        
        # Apply masks
        low_freq = dct_coeffs * low_mask.unsqueeze(0)
        mid_freq = dct_coeffs * mid_mask.unsqueeze(0)
        high_freq = dct_coeffs * high_mask.unsqueeze(0)
        
        return low_freq, mid_freq, high_freq
    
    def reconstruct_from_frequency(
        self,
        low_freq: torch.Tensor,
        mid_freq: torch.Tensor,
        high_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct features from frequency components.
        
        Args:
            low_freq, mid_freq, high_freq: Frequency components
        
        Returns:
            Reconstructed features
        """
        combined_dct = low_freq + mid_freq + high_freq
        return idct_1d(combined_dct)
    
    def augment_prototypes(
        self,
        features: torch.Tensor,
        qualities: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate augmented prototypes by frequency mixing.
        
        Args:
            features: Original support features of shape (N, D)
            qualities: Optional quality weights of shape (N,)
        
        Returns:
            Tuple of (augmented_features, augmented_qualities)
        """
        N, D = features.shape
        features = features.to(self.device)
        
        if N < 2:
            # Cannot augment with single sample, return original
            aug_qualities = qualities if qualities is not None else torch.ones(N, device=self.device)
            return features, aug_qualities
        
        # Decompose all features
        low_freq, mid_freq, high_freq = self.decompose_frequency(features)
        
        augmented_list = [features]  # Include originals
        quality_list = [qualities if qualities is not None else torch.ones(N, device=self.device)]
        
        # Generate augmented samples
        for aug_idx in range(self.num_augmented):
            # Random permutation for mixing
            perm = torch.randperm(N, device=self.device)
            
            # Mix low-frequency from original with high-frequency from permuted
            # This preserves semantic content while adding variation
            mixed_low = low_freq
            mixed_high = self.mix_alpha * high_freq + (1 - self.mix_alpha) * high_freq[perm]
            
            # Reconstruct
            augmented = self.reconstruct_from_frequency(mixed_low, mid_freq, mixed_high)
            
            # Optionally preserve norm
            if self.preserve_norm:
                orig_norm = torch.norm(features, dim=1, keepdim=True).clamp(min=1e-8)
                aug_norm = torch.norm(augmented, dim=1, keepdim=True).clamp(min=1e-8)
                augmented = augmented * (orig_norm / aug_norm)
            
            augmented_list.append(augmented)
            
            # Augmented samples get reduced quality weight
            if qualities is not None:
                aug_qual = qualities * (0.5 + 0.5 * (1 - aug_idx / self.num_augmented))
            else:
                aug_qual = torch.ones(N, device=self.device) * (0.5 + 0.5 * (1 - aug_idx / self.num_augmented))
            quality_list.append(aug_qual)
        
        all_features = torch.cat(augmented_list, dim=0)
        all_qualities = torch.cat(quality_list, dim=0)
        
        return all_features, all_qualities


class FrequencyAugmentedPCB:
    """
    Wrapper around PrototypicalCalibrationBlock that applies frequency
    augmentation to support features before building prototypes.
    """
    
    def __init__(self, base_pcb, cfg):
        """
        Args:
            base_pcb: The original PrototypicalCalibrationBlock instance
            cfg: Config node with NOVEL_METHODS.FREQ_AUG settings
        """
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        # Extract frequency augmentation config
        freq_cfg = cfg.NOVEL_METHODS.FREQ_AUG
        self.augmentor = FrequencyAugmentor(
            low_freq_ratio=float(freq_cfg.LOW_FREQ_RATIO),
            high_freq_ratio=float(freq_cfg.HIGH_FREQ_RATIO),
            num_augmented=int(freq_cfg.NUM_AUGMENTED),
            mix_alpha=float(freq_cfg.MIX_ALPHA),
            preserve_norm=bool(freq_cfg.PRESERVE_NORM),
            device=str(cfg.MODEL.DEVICE)
        )
        
        # Rebuild prototypes with augmentation
        self._rebuild_prototypes_with_augmentation()
    
    def _rebuild_prototypes_with_augmentation(self):
        """
        Rebuild the prototype bank using frequency-augmented features.
        """
        # Access the raw support data from base PCB
        if not hasattr(self.base_pcb, '_real_class_features'):
            return
        
        new_prototypes = {}
        
        for cls in self.base_pcb._real_class_features:
            feat_list = self.base_pcb._real_class_features[cls]
            qual_list = self.base_pcb._real_class_qualities[cls]
            area_list = self.base_pcb._real_class_areas.get(cls, [])
            
            if not feat_list:
                continue
            
            features = torch.stack(feat_list, dim=0)
            qualities = torch.tensor(qual_list, dtype=features.dtype, device=features.device)
            
            # Apply frequency augmentation
            aug_features, aug_qualities = self.augmentor.augment_prototypes(features, qualities)
            
            # Build prototype bank using base PCB method
            class_entry = {
                "global": self.base_pcb._build_proto_bank(aug_features.cpu(), aug_qualities.cpu()),
                "scale": {}
            }
            
            # Handle scale-aware if enabled
            if self.base_pcb.enable_scale_aware and area_list:
                areas = torch.tensor(area_list, dtype=features.dtype)
                # Repeat areas for augmented samples
                num_aug = aug_features.shape[0] // features.shape[0]
                aug_areas = areas.repeat(num_aug)
                bin_ids = torch.tensor(
                    [self.base_pcb._area_bin(float(a.item())) for a in aug_areas],
                    dtype=torch.long
                )
                for bid in [0, 1, 2]:
                    idx = torch.nonzero(bin_ids == bid, as_tuple=False).flatten()
                    if idx.numel() == 0:
                        continue
                    class_entry["scale"][bid] = self.base_pcb._build_proto_bank(
                        aug_features.cpu()[idx], aug_qualities.cpu()[idx]
                    )
            
            new_prototypes[cls] = class_entry
        
        # Replace base PCB prototypes
        self.base_pcb.prototypes = new_prototypes
    
    def execute_calibration(self, inputs, dts):
        """
        Delegate to base PCB's execute_calibration.
        The augmented prototypes are already in place.
        """
        return self.base_pcb.execute_calibration(inputs, dts)
    
    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_frequency_augmented_pcb(base_pcb, cfg):
    """
    Factory function to wrap a PCB with frequency augmentation.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config with NOVEL_METHODS.FREQ_AUG settings
    
    Returns:
        FrequencyAugmentedPCB wrapper
    """
    return FrequencyAugmentedPCB(base_pcb, cfg)
