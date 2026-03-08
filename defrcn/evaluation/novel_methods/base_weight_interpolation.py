"""
Base-Weight Interpolation Initialization for Few-Shot Object Detection.

This module implements a novel approach that initializes novel class classifier
weights as a semantic mixture of nearest base class weights using CLIP similarity.

Key idea: Instead of random or zero initialization for novel class weights,
compute semantic similarity between novel and base classes using CLIP, then
initialize novel weights as a weighted combination of the most similar base
class weights.

Example: sofa_weights = 0.45 * chair_weights + 0.30 * bed_weights + 0.25 * couch_weights

This provides a better starting point for few-shot fine-tuning by leveraging
semantic knowledge about class relationships.

Reference: Novel approach - NOT used in existing FSOD literature.
The key insight is that semantically similar classes share visual features,
so their classifier weights should be related.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import CLIP
CLIP_AVAILABLE = False
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    logger.warning(
        "CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git"
    )


# VOC class definitions
VOC_BASE_CLASSES_SPLIT1 = [
    "aeroplane", "bicycle", "boat", "bottle", "car",
    "cat", "chair", "diningtable", "dog", "horse",
    "person", "pottedplant", "sheep", "train", "tvmonitor"
]
VOC_NOVEL_CLASSES_SPLIT1 = ["bird", "bus", "cow", "motorbike", "sofa"]

VOC_BASE_CLASSES_SPLIT2 = [
    "bicycle", "bird", "boat", "bus", "car",
    "cat", "chair", "diningtable", "dog", "motorbike",
    "person", "pottedplant", "sheep", "train", "tvmonitor"
]
VOC_NOVEL_CLASSES_SPLIT2 = ["aeroplane", "bottle", "cow", "horse", "sofa"]

VOC_BASE_CLASSES_SPLIT3 = [
    "aeroplane", "bicycle", "bird", "bottle", "bus",
    "car", "cat", "cow", "diningtable", "dog",
    "horse", "person", "pottedplant", "train", "tvmonitor"
]
VOC_NOVEL_CLASSES_SPLIT3 = ["boat", "chair", "motorbike", "sheep", "sofa"]

# Class descriptions for richer semantic matching
CLASS_DESCRIPTIONS = {
    "aeroplane": "an aircraft airplane with wings that flies in the sky",
    "bicycle": "a two-wheeled vehicle powered by pedaling cycling",
    "bird": "a feathered flying animal with wings",
    "boat": "a watercraft vessel for traveling on water",
    "bottle": "a glass or plastic container for liquids",
    "bus": "a large passenger vehicle for public transportation",
    "car": "a four-wheeled automobile motor vehicle",
    "cat": "a small domesticated feline pet animal",
    "chair": "a piece of furniture with legs for sitting",
    "cow": "a large farm animal bovine that produces milk",
    "diningtable": "a table furniture used for eating meals",
    "dog": "a domesticated canine pet animal",
    "horse": "a large four-legged animal for riding equine",
    "motorbike": "a two-wheeled motor vehicle motorcycle",
    "person": "a human being man woman",
    "pottedplant": "a plant growing in a pot container houseplant",
    "sheep": "a woolly farm animal livestock",
    "sofa": "a long upholstered couch settee for sitting living room furniture",
    "train": "a rail vehicle locomotive for transportation",
    "tvmonitor": "a television display screen monitor"
}


class CLIPSemanticSimilarity(nn.Module):
    """
    Computes semantic similarity between classes using CLIP text embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        use_descriptions: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.model_name = model_name
        self.use_descriptions = use_descriptions
        self.device = device
        
        self.clip_model = None
        self._text_features_cache: Dict[str, torch.Tensor] = {}
        
        if CLIP_AVAILABLE:
            try:
                self.clip_model, _ = clip.load(model_name, device=device)
                self.clip_model.eval()
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                logger.info(f"Loaded CLIP model for semantic similarity: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")
    
    def get_text_embedding(self, class_name: str) -> torch.Tensor:
        """Get CLIP text embedding for a class name."""
        if class_name in self._text_features_cache:
            return self._text_features_cache[class_name]
        
        if self.clip_model is None:
            # Return random embedding as fallback
            return torch.randn(512, device=self.device)
        
        # Prepare text
        if self.use_descriptions and class_name in CLASS_DESCRIPTIONS:
            text = f"a photo of {CLASS_DESCRIPTIONS[class_name]}"
        else:
            text = f"a photo of a {class_name}"
        
        # Encode
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            embedding = self.clip_model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.squeeze(0)
        
        self._text_features_cache[class_name] = embedding
        return embedding
    
    def compute_similarity_matrix(
        self,
        classes_a: List[str],
        classes_b: List[str]
    ) -> torch.Tensor:
        """
        Compute pairwise semantic similarity between two sets of classes.
        
        Args:
            classes_a: First set of class names
            classes_b: Second set of class names
        
        Returns:
            Similarity matrix (len(classes_a), len(classes_b))
        """
        embeddings_a = torch.stack([
            self.get_text_embedding(c) for c in classes_a
        ])
        embeddings_b = torch.stack([
            self.get_text_embedding(c) for c in classes_b
        ])
        
        # Cosine similarity
        similarity = torch.mm(embeddings_a, embeddings_b.t())
        
        return similarity
    
    def get_nearest_base_classes(
        self,
        novel_class: str,
        base_classes: List[str],
        top_k: int = 3,
        temperature: float = 1.0
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Find the most semantically similar base classes for a novel class.
        
        Args:
            novel_class: Novel class name
            base_classes: List of base class names
            top_k: Number of nearest classes to return
            temperature: Temperature for softmax weights
        
        Returns:
            Tuple of (nearest_class_names, softmax_weights)
        """
        similarity = self.compute_similarity_matrix([novel_class], base_classes)
        similarity = similarity.squeeze(0)  # (num_base,)
        
        # Get top-k
        k = min(top_k, len(base_classes))
        top_values, top_indices = torch.topk(similarity, k)
        
        # Convert to softmax weights
        weights = F.softmax(top_values / temperature, dim=0)
        
        nearest_classes = [base_classes[i] for i in top_indices.cpu().tolist()]
        
        return nearest_classes, weights


class BaseWeightInterpolator(nn.Module):
    """
    Interpolates novel class weights from base class weights using semantic similarity.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        clip_model: str = "ViT-B/32",
        top_k: int = 3,
        temperature: float = 0.5,
        normalize_weights: bool = True,
        blend_with_random: float = 0.0,
        use_descriptions: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            feature_dim: Classifier weight dimension
            clip_model: CLIP model variant for similarity
            top_k: Number of base classes to interpolate from
            temperature: Temperature for similarity softmax
            normalize_weights: L2 normalize output weights
            blend_with_random: Blend ratio with random initialization
            use_descriptions: Use class descriptions for richer semantics
            device: Computation device
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.temperature = temperature
        self.normalize_weights = normalize_weights
        self.blend_with_random = blend_with_random
        self.device = device
        
        # Initialize CLIP similarity module
        self.semantic_sim = CLIPSemanticSimilarity(
            model_name=clip_model,
            use_descriptions=use_descriptions,
            device=device
        )
        
        logger.info(
            f"Initialized BaseWeightInterpolator: top_k={top_k}, "
            f"temperature={temperature}, blend_random={blend_with_random}"
        )
    
    def interpolate_weights(
        self,
        novel_class: str,
        base_classes: List[str],
        base_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interpolated weights for a novel class.
        
        Args:
            novel_class: Name of the novel class
            base_classes: Names of base classes (ordered to match weights)
            base_weights: Base classifier weights (num_base, feature_dim)
        
        Returns:
            Interpolated weights for novel class (feature_dim,)
        """
        # Get nearest base classes and similarity weights
        nearest_classes, sim_weights = self.semantic_sim.get_nearest_base_classes(
            novel_class=novel_class,
            base_classes=base_classes,
            top_k=self.top_k,
            temperature=self.temperature
        )
        
        # Get indices of nearest classes
        class_to_idx = {c: i for i, c in enumerate(base_classes)}
        nearest_indices = [class_to_idx[c] for c in nearest_classes]
        
        # Extract corresponding base weights
        nearest_weights = base_weights[nearest_indices]  # (top_k, feature_dim)
        
        # Weighted interpolation
        sim_weights = sim_weights.to(nearest_weights.device).unsqueeze(-1)  # (top_k, 1)
        interpolated = (nearest_weights * sim_weights).sum(dim=0)  # (feature_dim,)
        
        # Optional: blend with random initialization
        if self.blend_with_random > 0:
            random_init = torch.randn_like(interpolated)
            if self.normalize_weights:
                random_init = F.normalize(random_init, dim=0)
            interpolated = (
                (1 - self.blend_with_random) * interpolated +
                self.blend_with_random * random_init
            )
        
        # Optional: normalize
        if self.normalize_weights:
            interpolated = F.normalize(interpolated, dim=0)
        
        # Log interpolation details
        weight_strs = [
            f"{c}:{w:.2f}" for c, w in zip(nearest_classes, sim_weights.squeeze().tolist())
        ]
        logger.debug(f"Interpolated {novel_class} from: {', '.join(weight_strs)}")
        
        return interpolated
    
    def interpolate_all_novel_weights(
        self,
        novel_classes: List[str],
        base_classes: List[str],
        base_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interpolated weights for all novel classes.
        
        Args:
            novel_classes: List of novel class names
            base_classes: List of base class names
            base_weights: Base classifier weights (num_base, feature_dim)
        
        Returns:
            Novel classifier weights (num_novel, feature_dim)
        """
        novel_weights = []
        
        for novel_class in novel_classes:
            weight = self.interpolate_weights(
                novel_class=novel_class,
                base_classes=base_classes,
                base_weights=base_weights
            )
            novel_weights.append(weight)
        
        return torch.stack(novel_weights)


class BaseWeightInterpolatedPCB(nn.Module):
    """
    PCB wrapper that uses base-weight interpolation for novel class initialization.
    
    This approach provides better initial prototypes for novel classes by
    leveraging semantic similarity to base classes.
    """
    
    def __init__(
        self,
        base_pcb,
        cfg,
        top_k: int = 3,
        temperature: float = 0.5,
        blend_weight: float = 0.3,
        use_descriptions: bool = True,
        apply_to_prototypes: bool = True
    ):
        """
        Args:
            base_pcb: Original PrototypicalCalibrationBlock
            cfg: Config node
            top_k: Number of base classes to interpolate from
            temperature: Temperature for similarity softmax
            blend_weight: How much to blend interpolated weights with support prototypes
            use_descriptions: Use class descriptions for semantics
            apply_to_prototypes: Apply interpolation to prototypes (vs raw weights)
        """
        super().__init__()
        self.base_pcb = base_pcb
        self.cfg = cfg
        self.blend_weight = blend_weight
        self.apply_to_prototypes = apply_to_prototypes
        
        # Determine dataset and split
        self.base_classes, self.novel_classes = self._get_class_split(cfg)
        
        # Initialize interpolator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.interpolator = BaseWeightInterpolator(
            feature_dim=2048,
            clip_model="ViT-B/32",
            top_k=top_k,
            temperature=temperature,
            normalize_weights=True,
            use_descriptions=use_descriptions,
            device=device
        )
        
        # Cache for interpolated prototypes
        self._interpolated_prototypes: Dict[str, torch.Tensor] = {}
        self._prototype_computed = False
        
        # Apply prototype blending immediately after construction
        self._apply_prototype_blending()
        
        logger.info(
            f"Initialized BaseWeightInterpolatedPCB: top_k={top_k}, "
            f"temperature={temperature}, blend_weight={blend_weight}"
        )
    
    def _get_class_split(self, cfg) -> Tuple[List[str], List[str]]:
        """Determine base and novel classes from config."""
        # Try to infer from config
        dataset_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else ""
        
        if "voc" in dataset_name.lower():
            # Determine split from config or dataset name
            if "split1" in dataset_name.lower() or "_1" in dataset_name:
                return VOC_BASE_CLASSES_SPLIT1, VOC_NOVEL_CLASSES_SPLIT1
            elif "split2" in dataset_name.lower() or "_2" in dataset_name:
                return VOC_BASE_CLASSES_SPLIT2, VOC_NOVEL_CLASSES_SPLIT2
            elif "split3" in dataset_name.lower() or "_3" in dataset_name:
                return VOC_BASE_CLASSES_SPLIT3, VOC_NOVEL_CLASSES_SPLIT3
            else:
                # Default to split1
                return VOC_BASE_CLASSES_SPLIT1, VOC_NOVEL_CLASSES_SPLIT1
        else:
            # COCO or other - return empty for now
            logger.warning("COCO class split not yet implemented, using empty lists")
            return [], []
    
    def _apply_prototype_blending(self):
        """
        Apply prototype blending to base_pcb.prototypes for novel classes.
        
        This is the KEY fix: actually modify the prototypes in base_pcb
        so that execute_calibration uses blended prototypes.
        """
        if self._prototype_computed or not self.novel_classes:
            return
        
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available, cannot compute semantic similarity")
            return
        
        # Get base prototypes from base_pcb
        base_protos = self.base_pcb.prototypes
        if not base_protos:
            logger.warning("No base prototypes available")
            return
        
        # Collect base class prototype vectors
        # Base classes have IDs 0 to len(base_classes)-1
        # Novel classes have IDs len(base_classes) to len(base_classes)+len(novel_classes)-1
        num_base = len(self.base_classes)
        num_novel = len(self.novel_classes)
        
        base_proto_vectors = []
        available_base_classes = []
        for i in range(num_base):
            if i in base_protos:
                # Get the global prototype (first proto if multiple)
                proto_bank = base_protos[i].get("global", {})
                protos = proto_bank.get("protos", None)
                if protos is not None and protos.numel() > 0:
                    # Use mean of multi-protos
                    base_proto_vectors.append(protos.mean(dim=0))
                    if i < len(self.base_classes):
                        available_base_classes.append(self.base_classes[i])
        
        if not base_proto_vectors:
            logger.warning("No valid base prototypes found for interpolation")
            return
        
        base_proto_tensor = torch.stack(base_proto_vectors)  # (num_base, 2048)
        device = base_proto_tensor.device
        
        logger.info(f"Applying base-weight interpolation to {num_novel} novel classes")
        logger.info(f"Using {len(available_base_classes)} base classes: {available_base_classes}")
        
        # For each novel class, compute interpolated prototype and blend
        for j, novel_class in enumerate(self.novel_classes):
            novel_cls_id = num_base + j  # Novel class IDs come after base
            
            if novel_cls_id not in base_protos:
                logger.debug(f"Novel class {novel_class} (id={novel_cls_id}) not in prototypes, skipping")
                continue
            
            # Get the original support prototype
            novel_proto_bank = base_protos[novel_cls_id].get("global", {})
            original_protos = novel_proto_bank.get("protos", None)
            
            if original_protos is None or original_protos.numel() == 0:
                logger.debug(f"No prototype for {novel_class}, skipping")
                continue
            
            # Compute interpolated prototype from base classes
            try:
                interpolated = self.interpolator.interpolate_weights(
                    novel_class=novel_class,
                    base_classes=available_base_classes,
                    base_weights=base_proto_tensor.to(self.interpolator.device)
                )
                interpolated = interpolated.to(device)
            except Exception as e:
                logger.warning(f"Failed to interpolate for {novel_class}: {e}")
                continue
            
            # Blend each prototype in the bank
            blended_protos = []
            for proto_idx in range(original_protos.shape[0]):
                original = original_protos[proto_idx]
                blended = (
                    (1 - self.blend_weight) * original +
                    self.blend_weight * interpolated
                )
                blended = F.normalize(blended, dim=-1)
                blended_protos.append(blended)
            
            # Update the prototype bank
            base_protos[novel_cls_id]["global"]["protos"] = torch.stack(blended_protos)
            
            # Log the blending
            nearest, weights = self.interpolator.semantic_sim.get_nearest_base_classes(
                novel_class, available_base_classes, self.interpolator.top_k
            )
            weight_strs = [f"{c}:{w:.2f}" for c, w in zip(nearest, weights.tolist())]
            logger.info(f"  {novel_class}: blended with {', '.join(weight_strs)}")
        
        self._prototype_computed = True
        logger.info("Prototype blending completed")
    
    def _compute_interpolated_prototypes(self, base_prototypes: Dict[int, torch.Tensor]):
        """
        Compute interpolated prototypes for novel classes.
        
        Args:
            base_prototypes: Dict mapping class_id to prototype tensor
        """
        if self._prototype_computed or not self.base_classes:
            return
        
        # Get base class weights/prototypes
        # Note: This requires mapping class IDs to class names
        # For now, we'll use a simpler approach with the support prototypes
        
        # Stack base prototypes (assuming contiguous IDs for base classes)
        base_proto_list = []
        for i, class_name in enumerate(self.base_classes):
            if i in base_prototypes:
                base_proto_list.append(base_prototypes[i])
        
        if not base_proto_list:
            logger.warning("No base prototypes available for interpolation")
            return
        
        base_proto_tensor = torch.stack(base_proto_list)
        
        # Compute interpolated prototypes for each novel class
        for i, novel_class in enumerate(self.novel_classes):
            interpolated = self.interpolator.interpolate_weights(
                novel_class=novel_class,
                base_classes=self.base_classes[:len(base_proto_list)],
                base_weights=base_proto_tensor
            )
            self._interpolated_prototypes[novel_class] = interpolated
            
            # Log the interpolation
            logger.info(f"Interpolated prototype for {novel_class}")
        
        self._prototype_computed = True
    
    def blend_prototypes(
        self,
        support_prototype: torch.Tensor,
        novel_class_name: str
    ) -> torch.Tensor:
        """
        Blend support prototype with interpolated prototype.
        
        Args:
            support_prototype: Prototype from support set
            novel_class_name: Name of the novel class
        
        Returns:
            Blended prototype
        """
        if novel_class_name not in self._interpolated_prototypes:
            return support_prototype
        
        interpolated = self._interpolated_prototypes[novel_class_name]
        interpolated = interpolated.to(support_prototype.device)
        
        # Blend
        blended = (
            (1 - self.blend_weight) * support_prototype +
            self.blend_weight * interpolated
        )
        
        # Normalize
        blended = F.normalize(blended, dim=-1)
        
        return blended
    
    def execute_calibration(
        self,
        inputs: List[Dict],
        outputs: List[Dict]
    ) -> List[Dict]:
        """
        Execute calibration with base-weight interpolated prototypes.
        
        The interpolated prototypes provide a better semantic prior,
        especially for novel classes with limited support samples.
        """
        # Delegate to base PCB
        calibrated_outputs = self.base_pcb.execute_calibration(inputs, outputs)
        
        # Note: To fully integrate, we would need to modify the base PCB
        # to use our blended prototypes. For now, this serves as a wrapper
        # that can be extended.
        
        return calibrated_outputs
    
    def __getattr__(self, name):
        if name in ["base_pcb", "cfg", "interpolator", "blend_weight",
                    "apply_to_prototypes", "base_classes", "novel_classes",
                    "_interpolated_prototypes", "_prototype_computed"]:
            return super().__getattribute__(name)
        return getattr(self.base_pcb, name)


def build_base_weight_interpolated_pcb(base_pcb, cfg):
    """
    Factory function to build Base-Weight Interpolated PCB wrapper.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config node
    
    Returns:
        BaseWeightInterpolatedPCB wrapping the base PCB
    """
    bwi_cfg = cfg.NOVEL_METHODS.BASE_WEIGHT_INTERP
    
    return BaseWeightInterpolatedPCB(
        base_pcb=base_pcb,
        cfg=cfg,
        top_k=bwi_cfg.TOP_K,
        temperature=bwi_cfg.TEMPERATURE,
        blend_weight=bwi_cfg.BLEND_WEIGHT,
        use_descriptions=bwi_cfg.USE_DESCRIPTIONS,
        apply_to_prototypes=bwi_cfg.APPLY_TO_PROTOTYPES
    )


# ========================================================================
# Standalone weight initialization tool
# Can be used independently of PCB wrapper
# ========================================================================

def initialize_novel_classifier_weights(
    model_weights_path: str,
    output_path: str,
    dataset: str = "voc",
    split: int = 1,
    top_k: int = 3,
    temperature: float = 0.5,
    device: str = "cuda"
) -> None:
    """
    Standalone tool to initialize novel classifier weights.
    
    This can be run after base training to prepare weights for
    few-shot fine-tuning.
    
    Args:
        model_weights_path: Path to trained base model
        output_path: Path to save modified weights
        dataset: Dataset name (voc/coco)
        split: Split ID for VOC
        top_k: Number of base classes to interpolate from
        temperature: Temperature for similarity
        device: Computation device
    """
    # Load model weights
    state_dict = torch.load(model_weights_path, map_location=device)
    
    # Get classifier weight key
    cls_weight_key = None
    for key in state_dict.keys():
        if "cls_score" in key and "weight" in key:
            cls_weight_key = key
            break
    
    if cls_weight_key is None:
        logger.error("Could not find classifier weights in state dict")
        return
    
    # Get class split
    if dataset == "voc":
        if split == 1:
            base_classes, novel_classes = VOC_BASE_CLASSES_SPLIT1, VOC_NOVEL_CLASSES_SPLIT1
        elif split == 2:
            base_classes, novel_classes = VOC_BASE_CLASSES_SPLIT2, VOC_NOVEL_CLASSES_SPLIT2
        else:
            base_classes, novel_classes = VOC_BASE_CLASSES_SPLIT3, VOC_NOVEL_CLASSES_SPLIT3
    else:
        logger.error(f"Dataset {dataset} not supported")
        return
    
    # Get base weights (first N classes are base)
    cls_weights = state_dict[cls_weight_key]
    num_base = len(base_classes)
    base_weights = cls_weights[:num_base]
    
    # Initialize interpolator
    interpolator = BaseWeightInterpolator(
        feature_dim=cls_weights.shape[1],
        top_k=top_k,
        temperature=temperature,
        device=device
    )
    
    # Interpolate novel weights
    novel_weights = interpolator.interpolate_all_novel_weights(
        novel_classes=novel_classes,
        base_classes=base_classes,
        base_weights=base_weights
    )
    
    # Update state dict
    # Novel classes start at index num_base
    new_cls_weights = cls_weights.clone()
    for i, novel_weight in enumerate(novel_weights):
        new_cls_weights[num_base + i] = novel_weight
    
    state_dict[cls_weight_key] = new_cls_weights
    
    # Save modified weights
    torch.save(state_dict, output_path)
    logger.info(f"Saved interpolated weights to {output_path}")
    
    # Log interpolation summary
    logger.info(f"Interpolated {len(novel_classes)} novel classes from {num_base} base classes")
    for novel_class in novel_classes:
        nearest, weights = interpolator.semantic_sim.get_nearest_base_classes(
            novel_class, base_classes, top_k
        )
        weight_strs = [f"{c}:{w:.2f}" for c, w in zip(nearest, weights.tolist())]
        logger.info(f"  {novel_class}: {', '.join(weight_strs)}")


# ========================================================================
# Semantic similarity analysis tool
# ========================================================================

def analyze_class_similarity(
    base_classes: List[str],
    novel_classes: List[str],
    output_path: Optional[str] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze semantic similarity between novel and base classes.
    
    Useful for understanding why certain classes might perform poorly
    (low similarity to any base class).
    
    Args:
        base_classes: List of base class names
        novel_classes: List of novel class names
        output_path: Optional path to save analysis
    
    Returns:
        Dict mapping novel class to list of (base_class, similarity) tuples
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_module = CLIPSemanticSimilarity(device=device)
    
    # Compute full similarity matrix
    similarity_matrix = sim_module.compute_similarity_matrix(novel_classes, base_classes)
    
    # Analyze each novel class
    analysis = {}
    
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("="*60)
    
    for i, novel_class in enumerate(novel_classes):
        sims = similarity_matrix[i].cpu().numpy()
        sorted_indices = np.argsort(sims)[::-1]
        
        analysis[novel_class] = [
            (base_classes[j], float(sims[j]))
            for j in sorted_indices
        ]
        
        print(f"\n{novel_class.upper()}:")
        print("-" * 40)
        for j in sorted_indices[:5]:  # Top 5
            print(f"  {base_classes[j]:15s}: {sims[j]:.3f}")
        
        # Flag if max similarity is low
        max_sim = sims.max()
        if max_sim < 0.7:
            print(f"  ⚠️  Low max similarity ({max_sim:.3f}) - may struggle!")
    
    print("\n" + "="*60)
    
    # Save if path provided
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to {output_path}")
    
    return analysis


if __name__ == "__main__":
    # Quick test
    print("Testing Base-Weight Interpolation...")
    
    # Analyze VOC split1 similarity
    analysis = analyze_class_similarity(
        base_classes=VOC_BASE_CLASSES_SPLIT1,
        novel_classes=VOC_NOVEL_CLASSES_SPLIT1
    )
