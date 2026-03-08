"""
Cross-Modal Vision-Language Grounding for Few-Shot Object Detection.

This module implements CLIP-based semantic grounding where prototype
matching is augmented with vision-language alignment scores.

Key idea: Use pre-trained vision-language models (CLIP) to provide
semantic priors for class matching. Combine visual similarity with
text-based class description similarity.

Reference: Novel approach inspired by zero-shot detection and VLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Try to import CLIP (optional dependency)
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


# VOC class names and descriptions for text grounding
VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_CLASS_DESCRIPTIONS = {
    "aeroplane": "an aircraft with wings that flies in the sky",
    "bicycle": "a two-wheeled vehicle powered by pedaling",
    "bird": "a feathered animal with wings that can fly",
    "boat": "a watercraft for traveling on water",
    "bottle": "a container with a narrow neck for holding liquids",
    "bus": "a large vehicle for transporting many passengers",
    "car": "a four-wheeled motor vehicle for transportation",
    "cat": "a small domesticated feline pet",
    "chair": "a piece of furniture for sitting",
    "cow": "a large domesticated bovine animal",
    "diningtable": "a table used for eating meals",
    "dog": "a domesticated canine pet",
    "horse": "a large four-legged animal for riding",
    "motorbike": "a two-wheeled motor vehicle",
    "person": "a human being",
    "pottedplant": "a plant growing in a pot or container",
    "sheep": "a woolly domesticated farm animal",
    "sofa": "a long upholstered seat for multiple people",
    "train": "a rail vehicle for transporting passengers or cargo",
    "tvmonitor": "a display screen for watching television or using a computer"
}

# COCO class names (subset - novel classes in different splits)
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class CLIPGrounder(nn.Module):
    """
    CLIP-based semantic grounding for prototype matching.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        feature_dim: int = 2048,
        projection_dim: int = 512,
        temperature: float = 0.07,
        use_descriptions: bool = True,
        cache_text_features: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: CLIP model variant
            feature_dim: Input visual feature dimension (from detector)
            projection_dim: CLIP embedding dimension
            temperature: Temperature for similarity
            use_descriptions: Use class descriptions instead of just names
            cache_text_features: Cache text embeddings for efficiency
            device: Computation device
        """
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_descriptions = use_descriptions
        self.cache_text_features = cache_text_features
        self.device = device
        
        # Visual feature projector (from detector features to CLIP space)
        self.visual_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
        # Load CLIP if available
        self.clip_model = None
        self.clip_preprocess = None
        self.text_features_cache: Dict[str, torch.Tensor] = {}
        
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
                self.clip_model.eval()
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                logger.info(f"Loaded CLIP model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")
    
    def encode_text(self, class_names: List[str], descriptions: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """
        Encode class names/descriptions to text features.
        
        Args:
            class_names: List of class names
            descriptions: Optional dict mapping names to descriptions
        
        Returns:
            Text features (num_classes, projection_dim)
        """
        if self.clip_model is None:
            # Return random embeddings if CLIP not available
            return torch.randn(len(class_names), self.projection_dim, device=self.device)
        
        # Generate prompts
        prompts = []
        for name in class_names:
            if self.use_descriptions and descriptions and name in descriptions:
                prompt = f"a photo of {descriptions[name]}"
            else:
                prompt = f"a photo of a {name}"
            prompts.append(prompt)
        
        # Check cache
        cache_key = ";".join(prompts)
        if self.cache_text_features and cache_key in self.text_features_cache:
            return self.text_features_cache[cache_key].float()  # Ensure float32 for cached too
        # Encode with CLIP
        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=1)
        
        # Cache
        if self.cache_text_features:
            self.text_features_cache[cache_key] = text_features
        
        return text_features.float()
    
    def project_visual_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project detector features to CLIP space.
        
        Args:
            features: Visual features (N, feature_dim)
        
        Returns:
            Projected features (N, projection_dim)
        """
        projected = self.visual_projector(features)
        return F.normalize(projected, dim=1)
    
    def compute_text_similarity(
        self,
        visual_features: torch.Tensor,
        class_names: List[str],
        descriptions: Optional[Dict[str, str]] = None
    ) -> torch.Tensor:
        """
        Compute similarity between visual features and class text embeddings.
        
        Args:
            visual_features: Visual features (N, feature_dim)
            class_names: List of class names
            descriptions: Optional class descriptions
        
        Returns:
            Similarity scores (N, num_classes)
        """
        # Project visual features
        projected = self.project_visual_features(visual_features)
        
        # Get text features
        text_features = self.encode_text(class_names, descriptions)
        
        # Compute similarity
        similarity = torch.mm(projected, text_features.t()) / self.temperature
        
        return similarity
    
    def forward(
        self,
        visual_features: torch.Tensor,
        class_names: List[str],
        descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing text-visual alignment.
        
        Args:
            visual_features: Visual features (N, feature_dim)
            class_names: Class names for text encoding
            descriptions: Optional class descriptions
        
        Returns:
            Dict with 'text_similarity', 'projected_features', 'text_features'
        """
        projected = self.project_visual_features(visual_features)
        text_features = self.encode_text(class_names, descriptions)
        similarity = torch.mm(projected, text_features.t()) / self.temperature
        
        return {
            'text_similarity': similarity,
            'projected_features': projected,
            'text_features': text_features
        }


class VisionLanguagePrototypeMatcher:
    """
    Combines visual prototype matching with language grounding.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        clip_model: str = "ViT-B/32",
        visual_weight: float = 0.7,
        text_weight: float = 0.3,
        use_descriptions: bool = True,
        dataset: str = "voc",
        device: str = "cuda"
    ):
        """
        Args:
            feature_dim: Visual feature dimension
            clip_model: CLIP model name
            visual_weight: Weight for visual similarity
            text_weight: Weight for text similarity
            use_descriptions: Use detailed class descriptions
            dataset: Dataset name for class info
            device: Computation device
        """
        self.feature_dim = feature_dim
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        self.dataset = dataset
        self.device = device
        
        # Get class names and descriptions
        if dataset.lower() == "voc":
            self.class_names = VOC_CLASS_NAMES
            self.descriptions = VOC_CLASS_DESCRIPTIONS if use_descriptions else None
        else:
            self.class_names = COCO_CLASS_NAMES
            self.descriptions = None  # Add COCO descriptions if needed
        
        # Initialize CLIP grounder
        self.grounder = CLIPGrounder(
            model_name=clip_model,
            feature_dim=feature_dim,
            use_descriptions=use_descriptions,
            device=device
        ).to(device)
        
        self.grounder.eval()
    
    def compute_combined_similarity(
        self,
        query_feature: torch.Tensor,
        prototype_features: torch.Tensor,
        class_idx: int
    ) -> Tuple[float, Dict]:
        """
        Compute combined visual and text similarity.
        
        Args:
            query_feature: Query feature (D,)
            prototype_features: Prototype features (K, D)
            class_idx: Target class index
        
        Returns:
            Tuple of (combined_similarity, breakdown)
        """
        with torch.no_grad():
            # Visual similarity (cosine)
            q_norm = F.normalize(query_feature.unsqueeze(0), dim=1)
            p_norm = F.normalize(prototype_features, dim=1)
            visual_sims = torch.mm(q_norm, p_norm.t()).squeeze(0)
            visual_sim = visual_sims.max().item()
            
            # Text similarity
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
                text_out = self.grounder(
                    query_feature.unsqueeze(0).to(self.device),
                    [class_name],
                    {class_name: self.descriptions[class_name]} if self.descriptions else None
                )
                text_sim = text_out['text_similarity'][0, 0].item()
            else:
                text_sim = 0.0
            
            # Combine
            combined = (
                self.visual_weight * visual_sim +
                self.text_weight * text_sim
            )
            
            return combined, {
                'visual_similarity': visual_sim,
                'text_similarity': text_sim,
                'combined_similarity': combined
            }
    
    def get_class_text_scores(
        self,
        features: torch.Tensor,
        class_indices: List[int]
    ) -> torch.Tensor:
        """
        Get text-based class scores for multiple features.
        
        Args:
            features: Visual features (N, D)
            class_indices: List of target class indices
        
        Returns:
            Text similarity scores (N,)
        """
        if not class_indices:
            return torch.zeros(features.shape[0], device=features.device)
        
        # Get unique class names
        unique_classes = list(set(class_indices))
        class_names = [self.class_names[c] for c in unique_classes if c < len(self.class_names)]
        
        if not class_names:
            return torch.zeros(features.shape[0], device=features.device)
        
        with torch.no_grad():
            text_out = self.grounder(
                features.to(self.device),
                class_names,
                self.descriptions
            )
            
            # Map back to original class indices
            scores = torch.zeros(features.shape[0], device=features.device)
            for i, cls_idx in enumerate(class_indices):
                if cls_idx < len(self.class_names):
                    class_name = self.class_names[cls_idx]
                    if class_name in class_names:
                        name_idx = class_names.index(class_name)
                        scores[i] = text_out['text_similarity'][i, name_idx]
            
            return scores


class CLIPGroundedPCB:
    """
    Wrapper around PrototypicalCalibrationBlock that incorporates
    CLIP-based vision-language grounding.
    """
    
    def __init__(self, base_pcb, cfg):
        """
        Args:
            base_pcb: The original PrototypicalCalibrationBlock instance
            cfg: Config node with NOVEL_METHODS.CLIP_GROUND settings
        """
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        # Extract CLIP grounding config
        clip_cfg = cfg.NOVEL_METHODS.CLIP_GROUND
        
        # Determine dataset
        dataset = "voc" if "voc" in cfg.DATASETS.TEST[0].lower() else "coco"
        
        self.matcher = VisionLanguagePrototypeMatcher(
            feature_dim=int(clip_cfg.FEATURE_DIM),
            clip_model=str(clip_cfg.CLIP_MODEL),
            visual_weight=float(clip_cfg.VISUAL_WEIGHT),
            text_weight=float(clip_cfg.TEXT_WEIGHT),
            use_descriptions=bool(clip_cfg.USE_DESCRIPTIONS),
            dataset=dataset,
            device=str(cfg.MODEL.DEVICE)
        )
        
        # Pre-compute text embeddings for all classes
        self._precompute_text_embeddings()
    
    def _precompute_text_embeddings(self):
        """Pre-compute and cache text embeddings for all classes."""
        with torch.no_grad():
            _ = self.matcher.grounder.encode_text(
                self.matcher.class_names,
                self.matcher.descriptions
            )
        logger.info(f"Cached text embeddings for {len(self.matcher.class_names)} classes")
    
    def execute_calibration(self, inputs, dts):
        """
        Execute calibration with CLIP-grounded similarity.
        
        This overrides the base PCB's pure visual matching with a combined
        visual + text-based similarity score.
        """
        import cv2
        
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts
        
        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts
        img_h, img_w = img.shape[0], img.shape[1]
        
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
            proto_bank = self.base_pcb._select_proto_bank(cls, float(area_norm[q_idx].item()))
            if proto_bank is None:
                continue
            
            # Use CLIP-grounded combined similarity instead of pure visual
            query_feat = features[q_idx]
            # proto_bank is a dict with 'protos' key containing the tensor
            proto_features = proto_bank["protos"].to(query_feat.device)
            if proto_features.shape[0] == 0:
                continue
            combined_sim, breakdown = self.matcher.compute_combined_similarity(
                query_feat, proto_features, cls
            )
            
            # The combined similarity already blends visual + text
            sim = combined_sim
            
            alpha = self.base_pcb._effective_alpha(cls, sim)
            
            old_score = float(scores[i].item())
            fused = old_score * alpha + sim * (1.0 - alpha)
            fused = self.base_pcb._normalize_score(cls, fused)
            
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)
        
        return dts
    
    def get_text_priors(self, class_indices: List[int]) -> torch.Tensor:
        """
        Get text-based class priors that can be used for reranking.
        
        Args:
            class_indices: List of predicted class indices
        
        Returns:
            Prior weights based on text alignment
        """
        # This would be called after initial predictions to provide
        # text-based confidence adjustments
        pass
    
    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_clip_grounded_pcb(base_pcb, cfg):
    """
    Factory function to wrap a PCB with CLIP grounding.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config with NOVEL_METHODS.CLIP_GROUND settings
    
    Returns:
        CLIPGroundedPCB wrapper
    """
    return CLIPGroundedPCB(base_pcb, cfg)
