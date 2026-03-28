#!/usr/bin/env python3
"""
Comprehensive Foundation Model Comparison Analysis.

Computes prototype-space statistics across multiple foundation models to
support the analysis in the paper explaining why DINOv2 outperforms CLIP
and ImageNet features for FSOD calibration.

Metrics computed:
  1. Within-class variance: stability of support features around prototype
  2. Nearest-negative margin: separation between class prototypes  
  3. NN purity: fraction of queries matched to correct class prototype
  4. Crop stability: variance of features across different crop views
  5. Augmentation invariance: similarity between original and augmented views

Usage:
    python3 tools/analyze_fm_comparison.py \
        --config-file configs/voc/defrcn_det_r101_base1.yaml \
        --fm-models imagenet dinov1 clip dinov2 \
        --output results/fm_comparison.json

Output: JSON with per-FM statistics for paper Table 5.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defrcn.config import get_cfg, set_global_cfg

logger = logging.getLogger(__name__)


@dataclass
class FMMetrics:
    """Metrics for a single foundation model."""
    model_name: str
    within_class_variance: float
    nearest_negative_margin: float
    nn_purity: float
    crop_stability: float
    augmentation_invariance: float
    n_classes: int
    n_samples: int
    feature_dim: int


def get_fm_extractor(model_name: str, device: str = "cuda"):
    """Load a foundation model feature extractor.
    
    Supports:
      - imagenet: ResNet-101 pretrained on ImageNet
      - dinov1: DINOv1 ViT-B/16
      - clip: CLIP ViT-B/16
      - dinov2: DINOv2 ViT-B/14
      - dinov2_s: DINOv2 ViT-S/14
      - dinov2_l: DINOv2 ViT-L/14
    """
    from defrcn.evaluation.novel_methods.pcb_fma import FoundationModelFeatureExtractor
    
    model_configs = {
        "imagenet": {
            "model_name": "resnet101",
            "feat_dim": 2048,
            "roi_size": 224,
        },
        "dinov1": {
            "model_name": "dino_vitb16",
            "feat_dim": 768,
            "roi_size": 224,
        },
        "clip": {
            "model_name": "clip_vitb16",
            "feat_dim": 512,
            "roi_size": 224,
        },
        "dinov2": {
            "model_name": "dinov2_vitb14",
            "feat_dim": 768,
            "roi_size": 224,
        },
        "dinov2_s": {
            "model_name": "dinov2_vits14",
            "feat_dim": 384,
            "roi_size": 224,
        },
        "dinov2_l": {
            "model_name": "dinov2_vitl14",
            "feat_dim": 1024,
            "roi_size": 224,
        },
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")
    
    cfg = model_configs[model_name]
    
    if model_name == "imagenet":
        # Use ImageNet ResNet-101 (same as original PCB)
        return ImageNetFeatureExtractor(device=device)
    else:
        return FoundationModelFeatureExtractor(
            model_name=cfg["model_name"],
            feat_dim=cfg["feat_dim"],
            roi_size=cfg["roi_size"],
            device=device,
        )


class ImageNetFeatureExtractor:
    """ImageNet ResNet-101 feature extractor (baseline)."""
    
    def __init__(self, device: str = "cuda"):
        import torchvision.models as models
        import torchvision.transforms as T
        
        self.device = torch.device(device)
        self.model = models.resnet101(pretrained=True)
        # Remove final FC layer, keep avgpool
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feat_dim = 2048
        self.roi_size = 224
    
    @property
    def available(self) -> bool:
        return self.model is not None
    
    def extract_roi_features(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features for each RoI box."""
        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.feat_dim), device=self.device)
        
        boxes = boxes_tensor.cpu().numpy()
        img_h, img_w = img.shape[:2]
        
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crop = img[y1:y1 + 1, x1:x1 + 1]
            else:
                crop = img[y1:y2, x1:x2]
            # BGR -> RGB
            crop = crop[:, :, ::-1].copy()
            crops.append(self.transform(crop))
        
        batch = torch.stack(crops, dim=0).to(self.device)
        with torch.no_grad():
            features = self.model(batch).squeeze(-1).squeeze(-1)
        
        return features


def build_support_features_for_fm(cfg, fm_extractor) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, torch.Tensor]]:
    """Build per-class FM features from the support set.
    
    Returns:
        class_features: dict[class_id -> list of (feat_dim,) tensors]
        prototypes: dict[class_id -> (feat_dim,) L2-normalized tensor]
    """
    from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
    
    # Build base PCB to get the support dataloader
    pcb = PrototypicalCalibrationBlock(cfg)
    
    class_features: Dict[int, List[torch.Tensor]] = {}
    
    dataloader = pcb.dataloader
    for index in range(len(dataloader.dataset)):
        inputs = [dataloader.dataset[index]]
        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            continue
        
        inst = inputs[0]["instances"]
        if len(inst) == 0:
            continue
        
        img_h, img_w = img.shape[:2]
        ratio = img_h / float(inst.image_size[0])
        gt_boxes = inst.gt_boxes.tensor.clone() * ratio
        labels = inst.gt_classes.cpu()
        
        fm_feats = fm_extractor.extract_roi_features(img, gt_boxes)
        if fm_feats is None:
            continue
        fm_feats = fm_feats.detach().cpu()
        
        for i in range(len(labels)):
            cls = int(labels[i].item())
            if cls not in class_features:
                class_features[cls] = []
            class_features[cls].append(F.normalize(fm_feats[i], dim=0))
    
    # Compute prototypes (mean, L2-normalized)
    prototypes = {}
    for cls, feats in class_features.items():
        proto = torch.stack(feats, dim=0).mean(dim=0)
        prototypes[cls] = F.normalize(proto, dim=0)
    
    return class_features, prototypes


def compute_within_class_variance(class_features: Dict, prototypes: Dict) -> float:
    """Average L2 distance of support features to their class prototype."""
    variances = []
    for cls in sorted(prototypes.keys()):
        feats = class_features.get(cls, [])
        if not feats:
            continue
        proto = prototypes[cls]
        dists = [torch.norm(f - proto).item() for f in feats]
        variances.append(np.mean(dists))
    return float(np.mean(variances)) if variances else 0.0


def compute_nearest_negative_margin(prototypes: Dict) -> float:
    """Average min cosine distance between each prototype and its nearest different-class prototype."""
    cls_ids = sorted(prototypes.keys())
    margins = []
    
    for cls in cls_ids:
        proto = prototypes[cls]
        min_dist = float("inf")
        for other_cls in cls_ids:
            if other_cls == cls:
                continue
            other_proto = prototypes[other_cls]
            cos_sim = torch.dot(proto, other_proto).item()
            cos_dist = 1.0 - cos_sim
            if cos_dist < min_dist:
                min_dist = cos_dist
        if min_dist != float("inf"):
            margins.append(min_dist)
    
    return float(np.mean(margins)) if margins else 0.0


def compute_nn_purity(class_features: Dict, prototypes: Dict) -> float:
    """Fraction of features whose nearest prototype matches ground truth class."""
    cls_ids = sorted(prototypes.keys())
    correct = 0
    total = 0
    
    for cls in cls_ids:
        feats = class_features.get(cls, [])
        for f in feats:
            # Find nearest prototype
            best_cls = None
            best_sim = float("-inf")
            for other_cls in cls_ids:
                sim = torch.dot(f, prototypes[other_cls]).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cls = other_cls
            if best_cls == cls:
                correct += 1
            total += 1
    
    return float(correct / total) if total > 0 else 0.0


def compute_crop_stability(cfg, fm_extractor, n_samples: int = 50) -> float:
    """Measure feature variance across different crop views of the same object.
    
    Lower variance = more stable features across crop boundaries.
    """
    from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
    
    pcb = PrototypicalCalibrationBlock(cfg)
    dataloader = pcb.dataloader
    
    all_variances = []
    count = 0
    
    for index in range(len(dataloader.dataset)):
        if count >= n_samples:
            break
            
        inputs = [dataloader.dataset[index]]
        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            continue
        
        inst = inputs[0]["instances"]
        if len(inst) == 0:
            continue
        
        img_h, img_w = img.shape[:2]
        ratio = img_h / float(inst.image_size[0])
        gt_boxes = inst.gt_boxes.tensor.clone() * ratio
        
        for box_idx in range(min(len(gt_boxes), 3)):  # Max 3 boxes per image
            box = gt_boxes[box_idx:box_idx+1]
            x1, y1, x2, y2 = box[0].tolist()
            
            # Generate crop variants
            crop_views = []
            
            # Original
            crop_views.append(box.clone())
            
            # Slightly expanded (1.1x)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            for scale in [0.9, 1.0, 1.1]:
                new_w, new_h = w * scale, h * scale
                new_box = torch.tensor([[
                    max(0, cx - new_w/2),
                    max(0, cy - new_h/2),
                    min(img_w, cx + new_w/2),
                    min(img_h, cy + new_h/2),
                ]])
                crop_views.append(new_box)
            
            # Extract features for all views
            all_views_features = []
            for view_box in crop_views:
                feat = fm_extractor.extract_roi_features(img, view_box)
                if feat is not None and feat.numel() > 0:
                    all_views_features.append(F.normalize(feat[0], dim=0).cpu())
            
            if len(all_views_features) >= 2:
                # Compute variance of features across views
                stacked = torch.stack(all_views_features, dim=0)
                var = stacked.var(dim=0).mean().item()
                all_variances.append(var)
                count += 1
    
    return float(np.mean(all_variances)) if all_variances else 0.0


def compute_augmentation_invariance(cfg, fm_extractor, n_samples: int = 50) -> float:
    """Measure cosine similarity between original and horizontally flipped views.
    
    Higher similarity = more invariant features.
    """
    from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
    
    pcb = PrototypicalCalibrationBlock(cfg)
    dataloader = pcb.dataloader
    
    similarities = []
    count = 0
    
    for index in range(len(dataloader.dataset)):
        if count >= n_samples:
            break
            
        inputs = [dataloader.dataset[index]]
        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            continue
        
        inst = inputs[0]["instances"]
        if len(inst) == 0:
            continue
        
        img_h, img_w = img.shape[:2]
        ratio = img_h / float(inst.image_size[0])
        gt_boxes = inst.gt_boxes.tensor.clone() * ratio
        
        # Flip the image
        img_flipped = cv2.flip(img, 1)  # Horizontal flip
        
        for box_idx in range(min(len(gt_boxes), 3)):
            box = gt_boxes[box_idx:box_idx+1]
            x1, y1, x2, y2 = box[0].tolist()
            
            # Compute flipped box coordinates
            flipped_box = torch.tensor([[
                img_w - x2, y1, img_w - x1, y2
            ]])
            
            # Extract features
            feat_orig = fm_extractor.extract_roi_features(img, box)
            feat_flip = fm_extractor.extract_roi_features(img_flipped, flipped_box)
            
            if feat_orig is not None and feat_flip is not None:
                if feat_orig.numel() > 0 and feat_flip.numel() > 0:
                    f1 = F.normalize(feat_orig[0], dim=0)
                    f2 = F.normalize(feat_flip[0], dim=0)
                    sim = torch.dot(f1, f2).item()
                    similarities.append(sim)
                    count += 1
    
    return float(np.mean(similarities)) if similarities else 0.0


def analyze_fm(cfg, model_name: str, device: str = "cuda") -> FMMetrics:
    """Run full analysis for a single foundation model."""
    logger.info(f"Analyzing {model_name}...")
    
    fm_extractor = get_fm_extractor(model_name, device)
    
    if not fm_extractor.available:
        logger.warning(f"FM {model_name} not available, skipping")
        return None
    
    # Build features and prototypes
    logger.info(f"  Building support features...")
    class_features, prototypes = build_support_features_for_fm(cfg, fm_extractor)
    
    if not prototypes:
        logger.warning(f"  No prototypes built for {model_name}")
        return None
    
    n_samples = sum(len(f) for f in class_features.values())
    logger.info(f"  Built {len(prototypes)} class prototypes from {n_samples} samples")
    
    # Compute metrics
    logger.info(f"  Computing within-class variance...")
    variance = compute_within_class_variance(class_features, prototypes)
    
    logger.info(f"  Computing nearest-negative margin...")
    margin = compute_nearest_negative_margin(prototypes)
    
    logger.info(f"  Computing NN purity...")
    purity = compute_nn_purity(class_features, prototypes)
    
    logger.info(f"  Computing crop stability...")
    stability = compute_crop_stability(cfg, fm_extractor, n_samples=30)
    
    logger.info(f"  Computing augmentation invariance...")
    aug_inv = compute_augmentation_invariance(cfg, fm_extractor, n_samples=30)
    
    metrics = FMMetrics(
        model_name=model_name,
        within_class_variance=variance,
        nearest_negative_margin=margin,
        nn_purity=purity,
        crop_stability=stability,
        augmentation_invariance=aug_inv,
        n_classes=len(prototypes),
        n_samples=n_samples,
        feature_dim=fm_extractor.feat_dim,
    )
    
    logger.info(f"  {model_name}: Var={variance:.4f}, Margin={margin:.4f}, "
                f"Purity={purity:.4f}, Stability={stability:.6f}, AugInv={aug_inv:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Comprehensive FM comparison analysis")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--fm-models", nargs="+", 
                        default=["imagenet", "dinov1", "clip", "dinov2"],
                        help="Foundation models to compare")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Config overrides")
    args = parser.parse_args()
    
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Run analysis for each FM
    results = {}
    for model_name in args.fm_models:
        try:
            metrics = analyze_fm(cfg, model_name, args.device)
            if metrics is not None:
                results[model_name] = asdict(metrics)
        except Exception as e:
            logger.error(f"Failed to analyze {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Add metadata
    output = {
        "dataset": cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else cfg.DATASETS.TRAIN[0],
        "models": results,
    }
    
    # Save results
    output_path = args.output or os.path.join(cfg.OUTPUT_DIR, "fm_comparison.json")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Foundation Model Comparison Analysis")
    print("=" * 80)
    print(f"{'Model':<15} {'Variance↓':<12} {'Margin↑':<12} {'Purity↑':<12} {'Stability↓':<12} {'AugInv↑':<12}")
    print("-" * 80)
    for name in args.fm_models:
        if name in results:
            r = results[name]
            print(f"{name:<15} {r['within_class_variance']:<12.4f} {r['nearest_negative_margin']:<12.4f} "
                  f"{r['nn_purity']:<12.4f} {r['crop_stability']:<12.6f} {r['augmentation_invariance']:<12.4f}")
    print("=" * 80)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
