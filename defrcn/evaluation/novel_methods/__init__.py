"""
Novel Methods for Few-Shot Object Detection.

This package contains modular implementations of novel prototype
enhancement and matching techniques for improving few-shot detection.

Methods:
1. Frequency Augmentation - DCT-based prototype diversity
2. Contrastive Anchoring - InfoNCE prototype separation
3. Self-Distillation - Test-time pseudo-label refinement
4. Uncertainty Weighting - MC dropout confidence estimation
5. Part-Graph Reasoning - Compositional part-based matching
6. CLIP Grounding - Vision-language semantic alignment
7. SAM-Masked Prototype - Segment-based RoI purification (NOVEL)
8. Base-Weight Interpolation - Semantic weight initialization (NOVEL)
"""

from .frequency_augmentation import (
    FrequencyAugmentor,
    FrequencyAugmentedPCB,
    build_frequency_augmented_pcb,
    dct_1d,
    idct_1d
)

from .contrastive_anchoring import (
    ContrastivePrototypeAnchor,
    ContrastiveAnchoredPCB,
    build_contrastive_anchored_pcb
)

from .self_distillation import (
    SelfDistillationModule,
    SelfDistilledPCB,
    build_self_distilled_pcb
)

from .uncertainty_weighting import (
    PrototypeUncertaintyEstimator,
    UncertaintyWeightedMatcher,
    UncertaintyWeightedPCB,
    build_uncertainty_weighted_pcb
)

from .part_graph_reasoning import (
    PartDecomposer,
    PartGraphConv,
    PartGraphNetwork,
    PartGraphMatcher,
    PartGraphPCB,
    build_part_graph_pcb
)

from .clip_grounding import (
    CLIPGrounder,
    VisionLanguagePrototypeMatcher,
    CLIPGroundedPCB,
    build_clip_grounded_pcb,
    VOC_CLASS_NAMES,
    VOC_CLASS_DESCRIPTIONS,
    COCO_CLASS_NAMES
)

from .sam_masked_prototype import (
    SAMMaskedPrototypeExtractor,
    SAMMaskedPCB,
    build_sam_masked_pcb,
    SaliencyMaskedPrototypeExtractor,
    SaliencyMaskedPCB,
    build_saliency_masked_pcb,
    GrabCutMaskedPrototypeExtractor
)

from .base_weight_interpolation import (
    CLIPSemanticSimilarity,
    BaseWeightInterpolator,
    BaseWeightInterpolatedPCB,
    build_base_weight_interpolated_pcb,
    analyze_class_similarity,
    VOC_BASE_CLASSES_SPLIT1,
    VOC_NOVEL_CLASSES_SPLIT1,
    VOC_BASE_CLASSES_SPLIT2,
    VOC_NOVEL_CLASSES_SPLIT2,
    VOC_BASE_CLASSES_SPLIT3,
    VOC_NOVEL_CLASSES_SPLIT3
)

__all__ = [
    # Frequency Augmentation
    "FrequencyAugmentor",
    "FrequencyAugmentedPCB",
    "build_frequency_augmented_pcb",
    "dct_1d",
    "idct_1d",
    # Contrastive Anchoring
    "ContrastivePrototypeAnchor",
    "ContrastiveAnchoredPCB",
    "build_contrastive_anchored_pcb",
    # Self-Distillation
    "SelfDistillationModule",
    "SelfDistilledPCB",
    "build_self_distilled_pcb",
    # Uncertainty Weighting
    "PrototypeUncertaintyEstimator",
    "UncertaintyWeightedMatcher",
    "UncertaintyWeightedPCB",
    "build_uncertainty_weighted_pcb",
    # Part-Graph Reasoning
    "PartDecomposer",
    "PartGraphConv",
    "PartGraphNetwork",
    "PartGraphMatcher",
    "PartGraphPCB",
    "build_part_graph_pcb",
    # CLIP Grounding
    "CLIPGrounder",
    "VisionLanguagePrototypeMatcher",
    "CLIPGroundedPCB",
    "build_clip_grounded_pcb",
    "VOC_CLASS_NAMES",
    "VOC_CLASS_DESCRIPTIONS",
    "COCO_CLASS_NAMES",
    # SAM-Masked Prototype (NOVEL)
    "SAMMaskedPrototypeExtractor",
    "SAMMaskedPCB",
    "build_sam_masked_pcb",
    "SaliencyMaskedPrototypeExtractor",
    "SaliencyMaskedPCB",
    "build_saliency_masked_pcb",
    "GrabCutMaskedPrototypeExtractor",
    # Base-Weight Interpolation (NOVEL)
    "CLIPSemanticSimilarity",
    "BaseWeightInterpolator",
    "BaseWeightInterpolatedPCB",
    "build_base_weight_interpolated_pcb",
    "analyze_class_similarity",
    "VOC_BASE_CLASSES_SPLIT1",
    "VOC_NOVEL_CLASSES_SPLIT1",
    "VOC_BASE_CLASSES_SPLIT2",
    "VOC_NOVEL_CLASSES_SPLIT2",
    "VOC_BASE_CLASSES_SPLIT3",
    "VOC_NOVEL_CLASSES_SPLIT3",
]

def build_novel_method_pcb(base_pcb, cfg, method_name: str):
    """
    Factory function to build any novel method PCB wrapper.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config node
        method_name: Name of method to apply
    
    Returns:
        Wrapped PCB with novel method
    """
    method_builders = {
        "freq_aug": build_frequency_augmented_pcb,
        "frequency_augmentation": build_frequency_augmented_pcb,
        "contrastive": build_contrastive_anchored_pcb,
        "contrastive_anchoring": build_contrastive_anchored_pcb,
        "self_distill": build_self_distilled_pcb,
        "self_distillation": build_self_distilled_pcb,
        "uncertainty": build_uncertainty_weighted_pcb,
        "uncertainty_weighting": build_uncertainty_weighted_pcb,
        "part_graph": build_part_graph_pcb,
        "part_graph_reasoning": build_part_graph_pcb,
        "clip": build_clip_grounded_pcb,
        "clip_grounding": build_clip_grounded_pcb,
        "sam_masked": build_sam_masked_pcb,
        "sam": build_sam_masked_pcb,
        "saliency_masked": build_saliency_masked_pcb,
        "saliency": build_saliency_masked_pcb,
        "base_weight_interp": build_base_weight_interpolated_pcb,
        "base_weight": build_base_weight_interpolated_pcb,
        "bwi": build_base_weight_interpolated_pcb,
    }
    
    method_name_lower = method_name.lower()
    if method_name_lower not in method_builders:
        raise ValueError(
            f"Unknown novel method: {method_name}. "
            f"Available: {list(method_builders.keys())}"
        )
    
    return method_builders[method_name_lower](base_pcb, cfg)