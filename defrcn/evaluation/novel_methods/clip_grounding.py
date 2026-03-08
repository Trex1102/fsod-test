"""
CLIP-Based Vision-Language Grounding for Few-Shot Object Detection.

This module implements a training-free CLIP-based approach that combines:
1. Visual prototype similarity (from PCB's K-shot support features)
2. CLIP text embedding similarity (from class names/descriptions)

Key insight from Proto-CLIP (IROS 2024): Combine visual and text prototype
probabilities with weighted interpolation:
    P(y|x) = α·P_visual(y|x) + (1-α)·P_text(y|x)

CRITICAL DESIGN DECISION:
- We use CLIP's visual encoder to encode cropped RoI patches directly
- This is TRAINING-FREE - no projection layer needed
- CLIP's vision-language alignment is preserved

Reference: Proto-CLIP (https://github.com/IRVLUTD/Proto-CLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Try to import CLIP (optional dependency)
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


# VOC class names (index matches class ID 0-19)
VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Rich semantic descriptions for better CLIP alignment
VOC_CLASS_DESCRIPTIONS = {
    "aeroplane": "an aircraft airplane flying in the sky",
    "bicycle": "a bicycle with two wheels for cycling",
    "bird": "a bird with feathers and wings",
    "boat": "a boat or ship on water",
    "bottle": "a bottle container for liquids",
    "bus": "a large bus vehicle for passengers",
    "car": "an automobile car vehicle",
    "cat": "a cat feline pet animal",
    "chair": "a chair furniture for sitting",
    "cow": "a cow bovine farm animal",
    "diningtable": "a dining table furniture",
    "dog": "a dog canine pet animal",
    "horse": "a horse equine animal",
    "motorbike": "a motorcycle motorbike vehicle",
    "person": "a person human being",
    "pottedplant": "a potted plant in a container",
    "sheep": "a sheep woolly farm animal",
    "sofa": "a sofa couch furniture for sitting",
    "train": "a train locomotive on railway tracks",
    "tvmonitor": "a television monitor screen display"
}

# Multiple prompt templates for ensembling (improves robustness)
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a photograph of {}",
    "an image of {}",
    "{} in the scene",
    "a clear photo of {}",
]

# COCO class names (80 classes)
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


class CLIPTextEncoder:
    """
    Encodes class names to CLIP text embeddings with prompt ensembling.
    
    This is the "text prototype" generator - produces semantic embeddings
    for each class that can be compared with CLIP visual embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        use_descriptions: bool = True,
        use_prompt_ensemble: bool = True,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.use_descriptions = use_descriptions
        self.use_prompt_ensemble = use_prompt_ensemble
        self.device = device
        
        self.clip_model = None
        self.clip_preprocess = None
        self._text_cache: Dict[str, torch.Tensor] = {}
        
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
                self.clip_model.eval()
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                logger.info(f"Loaded CLIP model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
    
    def encode_classes(
        self,
        class_names: List[str],
        descriptions: Optional[Dict[str, str]] = None
    ) -> torch.Tensor:
        """
        Encode class names to text embeddings.
        
        Args:
            class_names: List of class names
            descriptions: Optional dict of class descriptions
            
        Returns:
            Text embeddings [num_classes, clip_dim] (L2 normalized)
        """
        if self.clip_model is None:
            # Return zeros if CLIP not available
            return torch.zeros(len(class_names), 512, device=self.device)
        
        cache_key = ";".join(class_names) + str(self.use_descriptions)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]
        
        embeddings = []
        
        with torch.no_grad():
            for name in class_names:
                # Get text to encode
                if self.use_descriptions and descriptions and name in descriptions:
                    base_text = descriptions[name]
                else:
                    base_text = name
                
                # Generate prompts
                if self.use_prompt_ensemble:
                    prompts = [t.format(base_text) for t in PROMPT_TEMPLATES]
                else:
                    prompts = [f"a photo of {base_text}"]
                
                # Encode all prompts
                tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.clip_model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)
                
                # Average across prompts (ensemble)
                class_embedding = text_features.mean(dim=0)
                class_embedding = F.normalize(class_embedding, dim=-1)
                
                embeddings.append(class_embedding)
        
        result = torch.stack(embeddings, dim=0).float()  # Ensure float32
        self._text_cache[cache_key] = result
        
        return result
    
    def encode_image_patches(
        self,
        image: np.ndarray,
        boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode cropped image patches using CLIP visual encoder.
        
        This is the TRAINING-FREE way to get CLIP-space visual features:
        - Crop each box from the image
        - Resize and preprocess for CLIP
        - Encode with CLIP visual encoder
        
        Args:
            image: BGR image array (H, W, 3)
            boxes: Box tensor [N, 4] in xyxy format
            
        Returns:
            Visual embeddings [N, clip_dim] (L2 normalized)
        """
        if self.clip_model is None or len(boxes) == 0:
            return torch.zeros(len(boxes), 512, device=self.device)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]
        
        patches = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                # Invalid box - use a small region
                patch = image_rgb[:32, :32]
            else:
                patch = image_rgb[y1:y2, x1:x2]
            
            # Convert to PIL and preprocess
            pil_patch = Image.fromarray(patch)
            processed = self.clip_preprocess(pil_patch)
            patches.append(processed)
        
        # Stack and encode
        patches_tensor = torch.stack(patches, dim=0).to(self.device)
        
        with torch.no_grad():
            visual_features = self.clip_model.encode_image(patches_tensor)
            visual_features = F.normalize(visual_features, dim=-1)
        
        return visual_features.float()


class CLIPGrounder(nn.Module):
    """
    Legacy class for backward compatibility.
    Wraps CLIPTextEncoder functionality.
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
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.projection_dim = projection_dim
        
        self.encoder = CLIPTextEncoder(
            model_name=model_name,
            use_descriptions=use_descriptions,
            device=device
        )
        
        # Legacy projector (not used in training-free mode)
        self.visual_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
        self.clip_model = self.encoder.clip_model
        self.clip_preprocess = self.encoder.clip_preprocess
        self.text_features_cache = self.encoder._text_cache
    
    def encode_text(self, class_names: List[str], descriptions: Optional[Dict[str, str]] = None) -> torch.Tensor:
        return self.encoder.encode_classes(class_names, descriptions)
    
    def project_visual_features(self, features: torch.Tensor) -> torch.Tensor:
        """Legacy method - projects detector features (not recommended)."""
        projected = self.visual_projector(features)
        return F.normalize(projected, dim=1)
    
    def compute_text_similarity(
        self,
        visual_features: torch.Tensor,
        class_names: List[str],
        descriptions: Optional[Dict[str, str]] = None
    ) -> torch.Tensor:
        """Compute similarity using projected features (legacy)."""
        projected = self.project_visual_features(visual_features)
        text_features = self.encode_text(class_names, descriptions)
        similarity = torch.mm(projected, text_features.t()) / self.temperature
        return similarity
    
    def forward(
        self,
        visual_features: torch.Tensor,
        class_names: List[str],
        descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Legacy forward pass."""
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
    Combines visual prototype matching with CLIP text embedding similarity.
    
    This implements the Proto-CLIP approach:
        P(y|x) = α·P_visual(y|x) + (1-α)·P_text(y|x)
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
            self.descriptions = None
        
        # Initialize CLIP encoder
        self.clip_encoder = CLIPTextEncoder(
            model_name=clip_model,
            use_descriptions=use_descriptions,
            device=device
        )
        
        # Pre-compute text embeddings for all classes
        self._text_embeddings = self.clip_encoder.encode_classes(
            self.class_names, self.descriptions
        )
        logger.info(f"Pre-computed text embeddings for {len(self.class_names)} classes")
        
        # Legacy grounder for compatibility
        self.grounder = CLIPGrounder(
            model_name=clip_model,
            feature_dim=feature_dim,
            use_descriptions=use_descriptions,
            device=device
        )
    
    def compute_combined_similarity(
        self,
        query_feature: torch.Tensor,
        prototype_features: torch.Tensor,
        class_idx: int
    ) -> Tuple[float, Dict]:
        """
        Compute combined visual and text similarity.
        
        NOTE: This uses the detector's RoI features directly for visual similarity,
        and falls back to projected features for text similarity.
        For best results, use compute_clip_similarity with actual image patches.
        """
        with torch.no_grad():
            # Visual similarity (cosine with prototypes)
            q_norm = F.normalize(query_feature.unsqueeze(0), dim=1)
            p_norm = F.normalize(prototype_features, dim=1)
            visual_sims = torch.mm(q_norm, p_norm.t()).squeeze(0)
            visual_sim = visual_sims.max().item()
            
            # Text similarity (needs CLIP visual features - approximate with projection)
            if class_idx < len(self.class_names):
                text_emb = self._text_embeddings[class_idx].unsqueeze(0)
                # Project query to CLIP space (approximate)
                projected = self.grounder.project_visual_features(
                    query_feature.unsqueeze(0).to(self.device)
                )
                text_sim = torch.mm(projected, text_emb.t()).squeeze().item()
            else:
                text_sim = 0.0
            
            # Combine with weights
            combined = self.visual_weight * visual_sim + self.text_weight * text_sim
            
            return combined, {
                'visual_similarity': visual_sim,
                'text_similarity': text_sim,
                'combined_similarity': combined
            }
    
    def compute_clip_text_similarity(
        self,
        clip_visual_features: torch.Tensor,
        class_indices: List[int]
    ) -> torch.Tensor:
        """
        Compute text similarity using proper CLIP visual features.
        
        Args:
            clip_visual_features: CLIP-encoded image patches [N, clip_dim]
            class_indices: Target class index for each detection
            
        Returns:
            Text similarity scores [N]
        """
        if len(class_indices) == 0:
            return torch.tensor([], device=self.device)
        
        scores = torch.zeros(len(class_indices), device=self.device)
        
        with torch.no_grad():
            for i, cls_idx in enumerate(class_indices):
                if cls_idx < len(self._text_embeddings):
                    text_emb = self._text_embeddings[cls_idx]
                    vis_emb = clip_visual_features[i]
                    # Cosine similarity (both are L2 normalized)
                    scores[i] = torch.dot(vis_emb, text_emb)
        
        return scores
    
    def get_class_text_scores(
        self,
        features: torch.Tensor,
        class_indices: List[int]
    ) -> torch.Tensor:
        """Legacy method using projected features."""
        if not class_indices:
            return torch.zeros(features.shape[0], device=features.device)
        
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
    PCB wrapper that incorporates CLIP-based vision-language grounding.
    
    This implementation uses CLIP's visual encoder to encode cropped RoI
    patches, then computes text similarity against class text embeddings.
    This is TRAINING-FREE and properly uses CLIP's aligned VL space.
    
    The final score combines:
    - Visual prototype similarity (from PCB's K-shot prototypes)
    - CLIP text similarity (from CLIP visual encoding of RoI + text embedding)
    
    Formula:
        fused_score = old_score * α + combined_sim * (1 - α)
        combined_sim = visual_weight * proto_sim + text_weight * text_sim
    """
    
    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        # Extract CLIP grounding config
        clip_cfg = cfg.NOVEL_METHODS.CLIP_GROUND
        
        # Determine dataset
        dataset = "voc" if "voc" in cfg.DATASETS.TEST[0].lower() else "coco"
        
        self.visual_weight = float(clip_cfg.VISUAL_WEIGHT)
        self.text_weight = float(clip_cfg.TEXT_WEIGHT)
        self.use_descriptions = bool(clip_cfg.USE_DESCRIPTIONS)
        self.use_clip_visual = bool(getattr(clip_cfg, 'USE_CLIP_VISUAL', True))
        self.use_prompt_ensemble = bool(getattr(clip_cfg, 'USE_PROMPT_ENSEMBLE', True))
        self.clip_model_name = str(clip_cfg.CLIP_MODEL)
        self.device = str(cfg.MODEL.DEVICE)
        
        # Get class info
        if dataset.lower() == "voc":
            self.class_names = VOC_CLASS_NAMES
            self.descriptions = VOC_CLASS_DESCRIPTIONS if self.use_descriptions else None
        else:
            self.class_names = COCO_CLASS_NAMES
            self.descriptions = None
        
        # Initialize CLIP encoder
        self.clip_encoder = CLIPTextEncoder(
            model_name=self.clip_model_name,
            use_descriptions=self.use_descriptions,
            use_prompt_ensemble=self.use_prompt_ensemble,
            device=self.device
        )
        
        # Pre-compute text embeddings
        self._text_embeddings = self.clip_encoder.encode_classes(
            self.class_names, self.descriptions
        )
        
        # Also keep matcher for backward compatibility
        self.matcher = VisionLanguagePrototypeMatcher(
            feature_dim=int(clip_cfg.FEATURE_DIM),
            clip_model=self.clip_model_name,
            visual_weight=self.visual_weight,
            text_weight=self.text_weight,
            use_descriptions=self.use_descriptions,
            dataset=dataset,
            device=self.device
        )
        
        logger.info(
            f"CLIPGroundedPCB initialized: visual_weight={self.visual_weight}, "
            f"text_weight={self.text_weight}, model={self.clip_model_name}, "
            f"use_clip_visual={self.use_clip_visual}"
        )
    
    def execute_calibration(self, inputs, dts):
        """
        Execute calibration with CLIP-grounded similarity.
        
        This uses CLIP's visual encoder to encode cropped RoI patches,
        providing true CLIP visual features for text similarity computation.
        """
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
        
        # Get RoI features from detector (for visual prototype matching)
        boxes = [pred_boxes.to(self.base_pcb.device)]
        features = self.base_pcb.extract_roi_features(img, boxes)
        
        # Get CLIP visual features by encoding cropped patches (if enabled)
        box_tensor = pred_boxes.tensor
        if self.use_clip_visual:
            clip_visual_features = self.clip_encoder.encode_image_patches(img, box_tensor)
        else:
            clip_visual_features = None

        pred_classes = dts[0]["instances"].pred_classes
        score_device = scores.device
        score_dtype = scores.dtype
        
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
            
            # 1. Visual prototype similarity (standard PCB matching)
            query_feat = features[q_idx]
            proto_features = proto_bank["protos"].to(query_feat.device)
            if proto_features.shape[0] == 0:
                continue
            
            visual_sim = self.base_pcb._match_similarity(query_feat, proto_bank)
            
            # 2. CLIP text similarity (using CLIP visual features if enabled)
            if cls < len(self._text_embeddings) and clip_visual_features is not None:
                clip_vis = clip_visual_features[q_idx]
                text_emb = self._text_embeddings[cls]
                text_sim = torch.dot(clip_vis, text_emb).item()
            elif cls < len(self._text_embeddings):
                # Fallback: use projected detector features (less accurate)
                projected = self.matcher.grounder.project_visual_features(
                    query_feat.unsqueeze(0).to(self.device)
                )
                text_emb = self._text_embeddings[cls]
                text_sim = torch.mm(projected, text_emb.unsqueeze(1)).squeeze().item()
            else:
                text_sim = 0.0
            
            # 3. Combine visual and text similarities
            combined_sim = (
                self.visual_weight * visual_sim +
                self.text_weight * text_sim
            )
            
            # 4. Apply PCB's alpha blending
            alpha = self.base_pcb._effective_alpha(cls, combined_sim)
            
            old_score = float(scores[i].item())
            fused = old_score * alpha + combined_sim * (1.0 - alpha)
            fused = self.base_pcb._normalize_score(cls, fused)
            
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)
        
        return dts
    
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
