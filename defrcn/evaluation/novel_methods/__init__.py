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
9. PCB-FMA - Foundation Model Alignment for PCB prototypes (NOVEL)
10. Meta-PCB - Meta-learned non-linear calibration (NOVEL)
11. UPR-TTA - Uncertainty-guided Prototype Refinement + TTA (NOVEL)
12. PCB-FMA-Patch - Patch-level local FM matching (NOVEL)
13. Negative Prototype Guard - Base-class false positive suppression (NOVEL)
14. Cycle-Consistent Correspondence - Multi-support cycle consistency (Direction 3)
15. Counterfactual Transport - Foreground vs background transport gap (Direction 11)
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

from .pcb_fma import (
    FoundationModelFeatureExtractor,
    PCBFMA,
    build_pcb_fma,
)

from .pcb_fma_patch import (
    PatchFeatureExtractor,
    PCBFMAPatch,
    build_pcb_fma_patch,
)

from .negative_proto_guard import (
    NegativeProtoGuard,
    build_neg_proto_guard,
    build_pcb_fma_patch_neg,
    build_pcb_fma_enhanced_neg,
)

from .pcb_fma_enhanced import (
    PCBFMAEnhanced,
    build_pcb_fma_enhanced,
)

from .meta_calibration import (
    MetaCalibrationNet,
    MetaPCB,
    build_meta_pcb,
)

from .upr_tta import (
    MCDropoutFeatureEstimator,
    UncertaintyGuidedPseudoLabeler,
    UPRTTA,
    build_upr_tta,
)

from .cycle_consistent_correspondence import (
    CycleCorrespondenceExtractor,
    CycleConsistentCorrespondence,
    build_cycle_consistent_pcb,
)

from .counterfactual_transport import (
    CounterfactualTransportExtractor,
    CounterfactualTransport,
    build_counterfactual_transport_pcb,
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
    # PCB-FMA (NOVEL)
    "FoundationModelFeatureExtractor",
    "PCBFMA",
    "build_pcb_fma",
    # PCB-FMA-Patch (NOVEL)
    "PatchFeatureExtractor",
    "PCBFMAPatch",
    "build_pcb_fma_patch",
    # Negative Prototype Guard (NOVEL)
    "NegativeProtoGuard",
    "build_neg_proto_guard",
    "build_pcb_fma_patch_neg",
    "build_pcb_fma_enhanced_neg",
    # PCB-FMA Enhanced (NOVEL)
    "PCBFMAEnhanced",
    "build_pcb_fma_enhanced",
    # Meta-PCB (NOVEL)
    "MetaCalibrationNet",
    "MetaPCB",
    "build_meta_pcb",
    # UPR-TTA (NOVEL)
    "MCDropoutFeatureEstimator",
    "UncertaintyGuidedPseudoLabeler",
    "UPRTTA",
    "build_upr_tta",
    # Cycle-Consistent Correspondence (Direction 3)
    "CycleCorrespondenceExtractor",
    "CycleConsistentCorrespondence",
    "build_cycle_consistent_pcb",
    # Counterfactual Transport (Direction 11)
    "CounterfactualTransportExtractor",
    "CounterfactualTransport",
    "build_counterfactual_transport_pcb",
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
        "pcb_fma": build_pcb_fma,
        "fma": build_pcb_fma,
        "foundation_model": build_pcb_fma,
        "pcb_fma_enhanced": build_pcb_fma_enhanced,
        "fma_enhanced": build_pcb_fma_enhanced,
        "enhanced_fma": build_pcb_fma_enhanced,
        "pcb_fma_patch": build_pcb_fma_patch,
        "fma_patch": build_pcb_fma_patch,
        "patch_local": build_pcb_fma_patch,
        "neg_proto_guard": build_neg_proto_guard,
        "npg": build_neg_proto_guard,
        "negative_guard": build_neg_proto_guard,
        "pcb_fma_patch_neg": build_pcb_fma_patch_neg,
        "fma_patch_neg": build_pcb_fma_patch_neg,
        "pcb_fma_enhanced_neg": build_pcb_fma_enhanced_neg,
        "fma_enhanced_neg": build_pcb_fma_enhanced_neg,
        "enhanced_fma_neg": build_pcb_fma_enhanced_neg,
        "meta_pcb": build_meta_pcb,
        "meta_calibration": build_meta_pcb,
        "meta": build_meta_pcb,
        "upr_tta": build_upr_tta,
        "upr": build_upr_tta,
        "uncertainty_refinement": build_upr_tta,
        "cycle_consistency": build_cycle_consistent_pcb,
        "cycle_consistent": build_cycle_consistent_pcb,
        "cycle_corr": build_cycle_consistent_pcb,
        "counterfactual_transport": build_counterfactual_transport_pcb,
        "counterfactual": build_counterfactual_transport_pcb,
        "ct": build_counterfactual_transport_pcb,
    }
    
    method_name_lower = method_name.lower()
    if method_name_lower not in method_builders:
        raise ValueError(
            f"Unknown novel method: {method_name}. "
            f"Available: {list(method_builders.keys())}"
        )
    
    return method_builders[method_name_lower](base_pcb, cfg)