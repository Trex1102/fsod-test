#!/usr/bin/env python3
"""
Train Feature Hallucinator on base class RoI features and generate
synthetic features for novel classes.

Usage:
    python tools/train_feature_hallucination.py \
        --config-file configs/voc/defrcn_det_r101_base1.yaml \
        --weights checkpoints/voc/defrcn/defrcn_det_r101_base1/model_final.pth \
        --novel-config configs/voc/defrcn_fsod_r101_novel1_1shot_seed0.yaml \
        --novel-weights checkpoints/voc/defrcn/defrcn_det_r101_base1/model_reset_remove.pth \
        --output checkpoints/voc/feature_hallucination/halluc_bank_1shot_seed0.pth \
        --num-gen-per-class 30
"""

import argparse
import os
import sys

import torch
from detectron2.structures import Boxes
from detectron2.utils.logger import setup_logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defrcn.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg
from defrcn.dataloader import build_detection_test_loader
from defrcn.modeling import build_model
from defrcn.modeling.feature_hallucination import (
    FeatureHallucinator,
    build_hallucinated_feature_bank,
)
import defrcn.data  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser("Feature Hallucination for FSOD")
    parser.add_argument(
        "--config-file", required=True, help="Base training config"
    )
    parser.add_argument(
        "--weights", required=True, help="Base model weights"
    )
    parser.add_argument(
        "--novel-config", required=True, help="Novel fine-tuning config"
    )
    parser.add_argument(
        "--novel-weights", default="", help="Novel model weights (for extracting novel prototypes)"
    )
    parser.add_argument("--output", required=True, help="Output path for feature bank")
    parser.add_argument("--num-gen-per-class", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-base-samples", type=int, default=50000)
    parser.add_argument("--mode", default="gaussian", choices=["gaussian", "delta"])
    parser.add_argument("--variance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def setup_cfg(config_file, weights, opts):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = weights
    cfg.freeze()
    return cfg


@torch.no_grad()
def collect_base_features(model, cfg, device, max_samples, logger):
    """Collect RoI features from base class training data."""
    dataset_name = cfg.DATASETS.TRAIN[0]
    loader = build_detection_test_loader(cfg, dataset_name)
    
    model.eval()
    roi_feature_name = cfg.MODEL.ROI_HEADS.IN_FEATURES[0]
    
    feat_chunks = []
    label_chunks = []
    total = 0
    
    logger.info("Collecting base class RoI features from %s...", dataset_name)
    
    for batch in loader:
        if total >= max_samples:
            break
        
        images = model.preprocess_image(batch)
        backbone_feats = model.backbone(images.tensor)
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
    
    logger.info("Collected %d base RoI features.", features.shape[0])
    return features, labels


@torch.no_grad()
def collect_novel_prototypes(model, cfg, device, logger):
    """Collect prototype features for novel classes (mean of few-shot samples)."""
    dataset_name = cfg.DATASETS.TRAIN[0]
    loader = build_detection_test_loader(cfg, dataset_name)
    
    model.eval()
    roi_feature_name = cfg.MODEL.ROI_HEADS.IN_FEATURES[0]
    
    # Collect features per class
    class_features = {}
    
    logger.info("Collecting novel class prototypes from %s...", dataset_name)
    
    for batch in loader:
        images = model.preprocess_image(batch)
        backbone_feats = model.backbone(images.tensor)
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
            
            for j in range(pooled.shape[0]):
                cls = gt_classes[j].item()
                if cls not in class_features:
                    class_features[cls] = []
                class_features[cls].append(pooled[j])
    
    # Compute prototypes (mean per class)
    prototypes = []
    labels = []
    for cls in sorted(class_features.keys()):
        feats = torch.stack(class_features[cls], dim=0)
        proto = feats.mean(dim=0)
        prototypes.append(proto)
        labels.append(cls)
        logger.info("  Class %d: %d samples -> prototype", cls, len(class_features[cls]))
    
    prototypes = torch.stack(prototypes, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return prototypes, labels


def main():
    args = parse_args()
    logger = setup_logger(name="feature_hallucination")
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Step 1: Load base model and collect base features
    logger.info("Loading base model...")
    base_cfg = setup_cfg(args.config_file, args.weights, args.opts)
    base_model = build_model(base_cfg)
    DetectionCheckpointer(base_model).load(base_cfg.MODEL.WEIGHTS)
    base_model.to(device)
    base_model.eval()
    
    base_features, base_labels = collect_base_features(
        base_model, base_cfg, device, args.max_base_samples, logger
    )
    
    # Step 2: Fit hallucinator on base features
    logger.info("Fitting hallucinator on base features...")
    hallucinator = FeatureHallucinator(
        feature_dim=base_features.shape[1],
        mode=args.mode,
        variance_scale=args.variance_scale,
    )
    hallucinator.fit_from_base_features(base_features, base_labels)
    
    # Step 3: Load novel config and collect novel prototypes
    logger.info("Loading novel config...")
    novel_weights = args.novel_weights if args.novel_weights else args.weights
    novel_cfg = setup_cfg(args.novel_config, novel_weights, args.opts)
    
    # Use base model to extract novel features (same backbone)
    novel_prototypes, novel_labels = collect_novel_prototypes(
        base_model, novel_cfg, device, logger
    )
    
    # Step 4: Generate hallucinated features
    logger.info("Generating hallucinated features...")
    hallucinator.to(device)
    novel_prototypes = novel_prototypes.to(device)
    novel_labels = novel_labels.to(device)
    
    bank = build_hallucinated_feature_bank(
        novel_prototypes=novel_prototypes,
        novel_labels=novel_labels,
        hallucinator=hallucinator,
        num_gen_per_class=args.num_gen_per_class,
        temperature=args.temperature,
        include_originals=True,
    )
    
    # Add metadata
    bank["base_features_count"] = base_features.shape[0]
    bank["base_classes"] = base_labels.unique().tolist()
    bank["novel_classes"] = novel_labels.unique().tolist()
    bank["mode"] = args.mode
    bank["variance_scale"] = args.variance_scale
    
    # Save hallucinator state for potential reuse
    bank["hallucinator_state"] = {
        "global_mean": hallucinator.global_mean.cpu(),
        "global_std": hallucinator.global_std.cpu(),
        "cov_U": hallucinator.cov_U.cpu(),
        "cov_D": hallucinator.cov_D.cpu(),
    }
    
    # Move features to CPU for saving
    bank["features"] = bank["features"].cpu()
    bank["labels"] = bank["labels"].cpu()
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    torch.save(bank, args.output)
    
    logger.info(
        "Saved hallucinated feature bank: %s (N=%d, C=%d)",
        args.output,
        bank["features"].shape[0],
        bank["features"].shape[1],
    )
    
    # Print statistics
    logger.info("Feature statistics:")
    logger.info("  Mean: %.4f", bank["features"].mean().item())
    logger.info("  Std: %.4f", bank["features"].std().item())
    logger.info("  Norm: %.2f", bank["features"].norm(dim=1).mean().item())


if __name__ == "__main__":
    main()
