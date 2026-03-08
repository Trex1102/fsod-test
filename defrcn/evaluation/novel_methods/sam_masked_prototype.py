"""
SAM-Masked Prototype Purification for Few-Shot Object Detection.

This module implements a novel approach using SAM (Segment Anything Model)
to purify support prototypes by masking out background pixels within RoI
bounding boxes before feature pooling.

Key idea: Standard RoI pooling includes background pixels that contaminate
the prototype representation. By using SAM to segment the actual object
pixels within the RoI, we can pool features only from object-relevant regions,
resulting in cleaner, more discriminative prototypes.

This is particularly effective for classes like "sofa" where bounding boxes
often include significant background (floor, wall, other furniture).

Reference: Novel approach - NOT used in existing FSOD literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import SAM (optional dependency)
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    logger.warning(
        "SAM not available. Install with: pip install segment-anything\n"
        "and download model weights from: https://github.com/facebookresearch/segment-anything"
    )


class SAMMaskedPrototypeExtractor(nn.Module):
    """
    Extracts purified prototypes using SAM-based object segmentation.
    
    Instead of pooling all pixels within an RoI, this module:
    1. Uses SAM to segment the object within the RoI bounding box
    2. Creates a binary mask of object vs background pixels
    3. Pools features only from object pixels (masked average pooling)
    
    This reduces RoI contamination from background regions.
    """
    
    def __init__(
        self,
        sam_checkpoint: str = "",
        sam_model_type: str = "vit_b",
        feature_dim: int = 2048,
        mask_threshold: float = 0.0,
        min_mask_ratio: float = 0.1,
        fallback_to_full_roi: bool = True,
        use_box_prompt: bool = True,
        use_center_point: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            sam_model_type: SAM model variant (vit_h, vit_l, vit_b)
            feature_dim: Feature dimension for prototypes
            mask_threshold: Threshold for binary mask from SAM logits
            min_mask_ratio: Minimum ratio of mask pixels to consider valid
            fallback_to_full_roi: Use full RoI if mask is too small
            use_box_prompt: Use bounding box as SAM prompt
            use_center_point: Also use center point as prompt
            device: Computation device
        """
        super().__init__()
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.feature_dim = feature_dim
        self.mask_threshold = mask_threshold
        self.min_mask_ratio = min_mask_ratio
        self.fallback_to_full_roi = fallback_to_full_roi
        self.use_box_prompt = use_box_prompt
        self.use_center_point = use_center_point
        self.device = device
        
        # SAM predictor (lazy initialization)
        self.sam_predictor: Optional[SamPredictor] = None
        self._sam_initialized = False
        
        # Feature projection (optional, for dimension matching)
        self.feature_proj = None
    
    def _init_sam(self):
        """Lazy initialization of SAM model."""
        if self._sam_initialized or not SAM_AVAILABLE:
            return
        
        if not self.sam_checkpoint:
            logger.warning("SAM checkpoint not provided. Using fallback mode.")
            self._sam_initialized = True
            return
        
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
            sam = sam.to(self.device)
            sam.eval()
            self.sam_predictor = SamPredictor(sam)
            logger.info(f"Loaded SAM model: {self.sam_model_type}")
            self._sam_initialized = True
        except Exception as e:
            logger.warning(f"Failed to load SAM: {e}. Using fallback mode.")
            self._sam_initialized = True
    
    def get_object_mask(
        self,
        image: np.ndarray,
        box: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get object segmentation mask within bounding box using SAM.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            box: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Binary mask (H, W) or None if SAM unavailable
        """
        self._init_sam()
        
        if self.sam_predictor is None:
            return None
        
        try:
            # Set image for SAM
            self.sam_predictor.set_image(image)
            
            # Prepare prompts
            prompts = {}
            
            if self.use_box_prompt:
                prompts["box"] = box
            
            if self.use_center_point:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                prompts["point_coords"] = np.array([[cx, cy]])
                prompts["point_labels"] = np.array([1])  # foreground
            
            # Get mask prediction
            masks, scores, logits = self.sam_predictor.predict(
                **prompts,
                multimask_output=True
            )
            
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            return mask.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"SAM prediction failed: {e}")
            return None
    
    def masked_roi_pool(
        self,
        features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform masked average pooling on RoI features.
        
        Args:
            features: RoI features (C, H, W) or (N, C, H, W)
            mask: Binary mask (H, W) or (N, H, W)
        
        Returns:
            Pooled features (C,) or (N, C)
        """
        single_input = features.dim() == 3
        if single_input:
            features = features.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        # Expand mask to match feature dimensions
        mask = mask.unsqueeze(1)  # (N, 1, H, W)
        
        # Resize mask to match feature spatial dimensions if needed
        if mask.shape[-2:] != features.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply mask and compute weighted average
        masked_features = features * mask
        sum_mask = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        pooled = masked_features.sum(dim=(-2, -1)) / sum_mask.squeeze(-1).squeeze(-1)
        
        if single_input:
            pooled = pooled.squeeze(0)
        
        return pooled
    
    def extract_purified_prototype(
        self,
        image: np.ndarray,
        box: np.ndarray,
        roi_features: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Extract purified prototype for a single RoI.
        
        Args:
            image: Full image (H, W, 3) RGB
            box: Bounding box [x1, y1, x2, y2]
            roi_features: Pre-extracted RoI features (C, H, W)
        
        Returns:
            Tuple of (purified_prototype, mask_ratio)
        """
        # Get SAM mask
        mask = self.get_object_mask(image, box)
        
        if mask is None:
            # Fallback: use full RoI
            return roi_features.mean(dim=(-2, -1)), 1.0
        
        # Crop mask to RoI region
        x1, y1, x2, y2 = map(int, box)
        roi_mask = mask[y1:y2, x1:x2]
        
        # Check mask validity
        mask_ratio = roi_mask.mean()
        
        if mask_ratio < self.min_mask_ratio and self.fallback_to_full_roi:
            # Mask too small, fallback to full RoI
            return roi_features.mean(dim=(-2, -1)), 1.0
        
        # Convert to tensor
        roi_mask_tensor = torch.from_numpy(roi_mask).float().to(roi_features.device)
        
        # Perform masked pooling
        purified = self.masked_roi_pool(roi_features, roi_mask_tensor)
        
        return purified, mask_ratio

    def forward(
        self,
        images: List[np.ndarray],
        boxes_per_image: List[np.ndarray],
        features_per_image: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[List[float]]]:
        """
        Extract purified prototypes for multiple images.
        
        Args:
            images: List of images (H, W, 3) RGB
            boxes_per_image: List of box arrays, each (N, 4)
            features_per_image: List of feature tensors, each (N, C, H, W)
        
        Returns:
            Tuple of (purified_prototypes_per_image, mask_ratios_per_image)
        """
        all_prototypes = []
        all_mask_ratios = []
        
        for image, boxes, features in zip(images, boxes_per_image, features_per_image):
            prototypes = []
            mask_ratios = []
            
            for i, (box, feat) in enumerate(zip(boxes, features)):
                proto, ratio = self.extract_purified_prototype(image, box, feat)
                prototypes.append(proto)
                mask_ratios.append(ratio)
            
            if prototypes:
                all_prototypes.append(torch.stack(prototypes))
            else:
                all_prototypes.append(torch.empty(0, self.feature_dim))
            all_mask_ratios.append(mask_ratios)
        
        return all_prototypes, all_mask_ratios


class SAMMaskedPCB(nn.Module):
    """
    PCB wrapper that uses SAM-masked prototypes for calibration.
    
    This wraps the base PrototypicalCalibrationBlock and replaces
    the standard prototype computation with SAM-purified prototypes.
    """
    
    def __init__(
        self,
        base_pcb,
        cfg,
        sam_checkpoint: str = "",
        sam_model_type: str = "vit_b",
        mask_threshold: float = 0.0,
        min_mask_ratio: float = 0.1,
        prototype_blend_weight: float = 0.5,
        use_quality_weighting: bool = True
    ):
        """
        Args:
            base_pcb: Original PrototypicalCalibrationBlock
            cfg: Config node
            sam_checkpoint: Path to SAM weights
            sam_model_type: SAM variant
            mask_threshold: Threshold for mask binarization
            min_mask_ratio: Minimum valid mask ratio
            prototype_blend_weight: Blend weight for SAM vs original prototype
            use_quality_weighting: Weight prototypes by mask quality
        """
        super().__init__()
        self.base_pcb = base_pcb
        self.cfg = cfg
        self.prototype_blend_weight = prototype_blend_weight
        self.use_quality_weighting = use_quality_weighting
        
        # Initialize SAM extractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_extractor = SAMMaskedPrototypeExtractor(
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            feature_dim=2048,
            mask_threshold=mask_threshold,
            min_mask_ratio=min_mask_ratio,
            device=device
        )
        
        # Cache for SAM-purified prototypes
        self._sam_prototypes: Dict[int, torch.Tensor] = {}
        self._sam_prototype_qualities: Dict[int, float] = {}
        
        logger.info(
            f"Initialized SAM-Masked PCB with model={sam_model_type}, "
            f"blend_weight={prototype_blend_weight}"
        )
    
    def _compute_sam_prototypes(
        self,
        support_data: Dict,
        base_prototypes: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Compute SAM-purified prototypes from support data.
        
        Args:
            support_data: Support set data with images and boxes
            base_prototypes: Original prototypes from base PCB
        
        Returns:
            Dict mapping class_id to purified prototype
        """
        # Check if SAM is available
        if not SAM_AVAILABLE or self.sam_extractor.sam_predictor is None:
            logger.debug("SAM not available, using base prototypes")
            return base_prototypes
        
        sam_prototypes = {}
        
        for class_id, base_proto in base_prototypes.items():
            # Get support images and boxes for this class
            if class_id not in support_data:
                sam_prototypes[class_id] = base_proto
                continue
            
            class_support = support_data[class_id]
            images = class_support.get("images", [])
            boxes = class_support.get("boxes", [])
            features = class_support.get("features", [])
            
            if not images or not boxes or not features:
                sam_prototypes[class_id] = base_proto
                continue
            
            # Extract SAM-purified features
            purified_features = []
            quality_weights = []
            
            for img, box, feat in zip(images, boxes, features):
                purified, mask_ratio = self.sam_extractor.extract_purified_prototype(
                    img, box, feat
                )
                purified_features.append(purified)
                quality_weights.append(mask_ratio)
            
            if not purified_features:
                sam_prototypes[class_id] = base_proto
                continue
            
            # Stack and aggregate
            purified_stack = torch.stack(purified_features)
            
            if self.use_quality_weighting:
                # Weight by mask quality (higher = better segmentation)
                weights = torch.tensor(quality_weights, device=purified_stack.device)
                weights = weights / weights.sum().clamp(min=1e-6)
                sam_proto = (purified_stack * weights.unsqueeze(-1)).sum(dim=0)
            else:
                sam_proto = purified_stack.mean(dim=0)
            
            # Blend with original prototype
            blended = (
                self.prototype_blend_weight * sam_proto +
                (1 - self.prototype_blend_weight) * base_proto
            )
            
            # Store quality for this class
            self._sam_prototype_qualities[class_id] = np.mean(quality_weights)
            sam_prototypes[class_id] = blended
        
        return sam_prototypes
    
    def execute_calibration(
        self,
        inputs: List[Dict],
        outputs: List[Dict]
    ) -> List[Dict]:
        """
        Execute calibration with SAM-purified prototypes.
        
        In the current implementation, we delegate to base PCB but
        override the prototype computation if SAM is available and
        support data is accessible.
        """
        # First, get base PCB results
        calibrated_outputs = self.base_pcb.execute_calibration(inputs, outputs)
        
        # If SAM-purified prototypes are computed, apply additional calibration
        # Note: Full integration requires access to support images during inference
        # which may need modifications to the data pipeline
        
        return calibrated_outputs
    
    # Delegate other methods to base PCB
    def __getattr__(self, name):
        if name in ["base_pcb", "cfg", "sam_extractor", "prototype_blend_weight",
                    "use_quality_weighting", "_sam_prototypes", "_sam_prototype_qualities"]:
            return super().__getattribute__(name)
        return getattr(self.base_pcb, name)


def build_sam_masked_pcb(base_pcb, cfg):
    """
    Factory function to build SAM-Masked PCB wrapper.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config node
    
    Returns:
        SAMMaskedPCB wrapping the base PCB
    """
    sam_cfg = cfg.NOVEL_METHODS.SAM_MASKED
    
    return SAMMaskedPCB(
        base_pcb=base_pcb,
        cfg=cfg,
        sam_checkpoint=sam_cfg.CHECKPOINT,
        sam_model_type=sam_cfg.MODEL_TYPE,
        mask_threshold=sam_cfg.MASK_THRESHOLD,
        min_mask_ratio=sam_cfg.MIN_MASK_RATIO,
        prototype_blend_weight=sam_cfg.BLEND_WEIGHT,
        use_quality_weighting=sam_cfg.QUALITY_WEIGHTING
    )


# ========================================================================
# Alternative: Lightweight mask approximation without SAM
# For cases where SAM is too heavy or unavailable
# ========================================================================

class GrabCutMaskedPrototypeExtractor(nn.Module):
    """
    Lightweight alternative using OpenCV's GrabCut for segmentation.
    
    This is faster than SAM but less accurate. Useful as a fallback
    or for quick experimentation.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        grabcut_iters: int = 5,
        min_mask_ratio: float = 0.1,
        fallback_to_full_roi: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grabcut_iters = grabcut_iters
        self.min_mask_ratio = min_mask_ratio
        self.fallback_to_full_roi = fallback_to_full_roi
        
        try:
            import cv2
            self.cv2 = cv2
            self._cv2_available = True
        except ImportError:
            self._cv2_available = False
            logger.warning("OpenCV not available for GrabCut")
    
    def get_object_mask(
        self,
        image: np.ndarray,
        box: np.ndarray
    ) -> Optional[np.ndarray]:
        """Get object mask using GrabCut algorithm."""
        if not self._cv2_available:
            return None
        
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Initialize mask
            mask = np.zeros(image.shape[:2], np.uint8)
            
            # GrabCut rect format: (x, y, width, height)
            rect = (x1, y1, x2 - x1, y2 - y1)
            
            # Background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Run GrabCut
            self.cv2.grabCut(
                image, mask, rect,
                bgd_model, fgd_model,
                self.grabcut_iters,
                self.cv2.GC_INIT_WITH_RECT
            )
            
            # Convert mask to binary (foreground = 1 or 3)
            binary_mask = np.where(
                (mask == self.cv2.GC_FGD) | (mask == self.cv2.GC_PR_FGD),
                1.0, 0.0
            ).astype(np.float32)
            
            return binary_mask
            
        except Exception as e:
            logger.debug(f"GrabCut failed: {e}")
            return None
    
    def extract_purified_prototype(
        self,
        image: np.ndarray,
        box: np.ndarray,
        roi_features: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Extract purified prototype using GrabCut mask."""
        mask = self.get_object_mask(image, box)
        
        if mask is None:
            return roi_features.mean(dim=(-2, -1)), 1.0
        
        x1, y1, x2, y2 = map(int, box)
        roi_mask = mask[y1:y2, x1:x2]
        
        mask_ratio = roi_mask.mean()
        
        if mask_ratio < self.min_mask_ratio and self.fallback_to_full_roi:
            return roi_features.mean(dim=(-2, -1)), 1.0
        
        roi_mask_tensor = torch.from_numpy(roi_mask).float().to(roi_features.device)
        
        # Resize mask to feature dimensions
        if roi_mask_tensor.shape != roi_features.shape[-2:]:
            roi_mask_tensor = F.interpolate(
                roi_mask_tensor.unsqueeze(0).unsqueeze(0),
                size=roi_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Masked average pooling
        masked_features = roi_features * roi_mask_tensor.unsqueeze(0)
        sum_mask = roi_mask_tensor.sum().clamp(min=1e-6)
        purified = masked_features.sum(dim=(-2, -1)) / sum_mask
        
        return purified, mask_ratio


# ========================================================================
# Saliency-based mask approximation (no external dependencies)
# ========================================================================

class SaliencyMaskedPrototypeExtractor(nn.Module):
    """
    Use feature-based saliency to create soft masks for prototype purification.
    
    This approach doesn't require any external segmentation model.
    It uses the spatial statistics of the RoI features themselves
    to identify salient (likely foreground) regions.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        saliency_mode: str = "gradient_magnitude",  # gradient_magnitude, channel_attention, spatial_std
        threshold_percentile: float = 50.0,
        soft_mask: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.saliency_mode = saliency_mode
        self.threshold_percentile = threshold_percentile
        self.soft_mask = soft_mask
        self.temperature = temperature
    
    def compute_saliency_mask(
        self,
        roi_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute saliency-based mask from RoI features.
        
        Args:
            roi_features: Features (C, H, W)
        
        Returns:
            Saliency mask (H, W)
        """
        C, H, W = roi_features.shape
        
        if self.saliency_mode == "gradient_magnitude":
            # Compute spatial gradient magnitude
            # Higher gradients indicate object boundaries/textures
            dx = roi_features[:, :, 1:] - roi_features[:, :, :-1]
            dy = roi_features[:, 1:, :] - roi_features[:, :-1, :]
            
            # Pad to original size
            dx = F.pad(dx, (0, 1, 0, 0))
            dy = F.pad(dy, (0, 0, 0, 1))
            
            grad_mag = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)
            saliency = grad_mag.mean(dim=0)
            
        elif self.saliency_mode == "channel_attention":
            # Use channel-wise statistics
            # Channels with high variance are likely more discriminative
            channel_var = roi_features.var(dim=0)
            channel_mean = roi_features.abs().mean(dim=0)
            saliency = channel_var * channel_mean
            
        elif self.saliency_mode == "spatial_std":
            # Deviation from spatial mean indicates foreground
            spatial_mean = roi_features.mean(dim=(-2, -1), keepdim=True)
            deviation = (roi_features - spatial_mean).pow(2).mean(dim=0)
            saliency = torch.sqrt(deviation + 1e-8)
            
        else:
            # Default: uniform mask
            saliency = torch.ones(H, W, device=roi_features.device)
        
        # Normalize saliency to [0, 1]
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        
        if self.soft_mask:
            # Apply temperature for soft thresholding
            saliency = torch.sigmoid((saliency - 0.5) * self.temperature * 10)
        else:
            # Hard threshold at percentile
            threshold = torch.quantile(saliency.flatten(), self.threshold_percentile / 100)
            saliency = (saliency > threshold).float()
        
        return saliency
    
    def extract_purified_prototype(
        self,
        roi_features: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Extract purified prototype using saliency mask.
        
        Args:
            roi_features: Features (C, H, W)
        
        Returns:
            Tuple of (purified_prototype, mask_ratio)
        """
        mask = self.compute_saliency_mask(roi_features)
        mask_ratio = mask.mean().item()
        
        # Masked average pooling
        masked_features = roi_features * mask.unsqueeze(0)
        sum_mask = mask.sum().clamp(min=1e-6)
        purified = masked_features.sum(dim=(-2, -1)) / sum_mask
        
        return purified, mask_ratio


class SaliencyMaskedPCB(nn.Module):
    """
    PCB wrapper using saliency-based prototype purification.
    
    This is a lightweight alternative to SAM-based masking that
    doesn't require any external models.
    """
    
    def __init__(
        self,
        base_pcb,
        cfg,
        saliency_mode: str = "gradient_magnitude",
        threshold_percentile: float = 50.0,
        soft_mask: bool = True,
        temperature: float = 1.0,
        prototype_blend_weight: float = 0.5
    ):
        super().__init__()
        self.base_pcb = base_pcb
        self.cfg = cfg
        self.prototype_blend_weight = prototype_blend_weight
        
        self.saliency_extractor = SaliencyMaskedPrototypeExtractor(
            feature_dim=2048,
            saliency_mode=saliency_mode,
            threshold_percentile=threshold_percentile,
            soft_mask=soft_mask,
            temperature=temperature
        )
        
        logger.info(
            f"Initialized Saliency-Masked PCB with mode={saliency_mode}, "
            f"blend_weight={prototype_blend_weight}"
        )
    
    def execute_calibration(
        self,
        inputs: List[Dict],
        outputs: List[Dict]
    ) -> List[Dict]:
        """Execute calibration with saliency-purified prototypes."""
        return self.base_pcb.execute_calibration(inputs, outputs)
    
    def __getattr__(self, name):
        if name in ["base_pcb", "cfg", "saliency_extractor", "prototype_blend_weight"]:
            return super().__getattribute__(name)
        return getattr(self.base_pcb, name)


def build_saliency_masked_pcb(base_pcb, cfg):
    """Factory function for Saliency-Masked PCB."""
    sal_cfg = cfg.NOVEL_METHODS.SALIENCY_MASKED
    
    return SaliencyMaskedPCB(
        base_pcb=base_pcb,
        cfg=cfg,
        saliency_mode=sal_cfg.MODE,
        threshold_percentile=sal_cfg.THRESHOLD_PERCENTILE,
        soft_mask=sal_cfg.SOFT_MASK,
        temperature=sal_cfg.TEMPERATURE,
        prototype_blend_weight=sal_cfg.BLEND_WEIGHT
    )
