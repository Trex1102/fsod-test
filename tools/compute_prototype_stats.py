"""
Compute prototype-space statistics for Table 5: Variance, Margin, Purity.

Loads the FM prototypes (and individual support features) from the PCB-FMA
pipeline and computes:
  - Within-class variance: average L2 distance of support features to prototype
  - Nearest-negative margin: min distance between a prototype and its
    closest different-class prototype
  - Prototype purity: silhouette-like score measuring cluster quality

Usage:
    python3 tools/compute_prototype_stats.py \
        --config-file configs/voc/... \
        --opts MODEL.WEIGHTS path/to/model.pth \
        TEST.PCB_MODELPATH path/to/resnet101.pth

Output: JSON with per-class and aggregate statistics.
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
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defrcn.config import get_cfg, set_global_cfg
from defrcn.data import register_all_coco, register_all_voc  # noqa: F401

logger = logging.getLogger(__name__)


def build_support_features(cfg):
    """Build per-class FM features from the support set.

    Returns:
        class_features: dict[class_id -> list of (feat_dim,) tensors]
        prototypes: dict[class_id -> (feat_dim,) L2-normalized tensor]
    """
    from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
    from defrcn.evaluation.novel_methods.pcb_fma import FoundationModelFeatureExtractor

    # Build base PCB to get the support dataloader
    pcb = PrototypicalCalibrationBlock(cfg)

    # Determine FM config
    method = cfg.NOVEL_METHODS.METHOD if cfg.NOVEL_METHODS.ENABLE else ""
    if "enhanced" in method:
        fm_cfg = cfg.NOVEL_METHODS.PCB_FMA_ENHANCED
    elif "patch" in method:
        fm_cfg = cfg.NOVEL_METHODS.PCB_FMA_PATCH
    elif "pcb_fma" in method:
        fm_cfg = cfg.NOVEL_METHODS.PCB_FMA
    else:
        fm_cfg = cfg.NOVEL_METHODS.PCB_FMA_ENHANCED  # default

    fm_extractor = FoundationModelFeatureExtractor(
        model_name=str(fm_cfg.FM_MODEL_NAME),
        model_path=str(fm_cfg.FM_MODEL_PATH),
        feat_dim=int(fm_cfg.FM_FEAT_DIM),
        roi_size=int(fm_cfg.ROI_SIZE),
        batch_size=int(fm_cfg.BATCH_SIZE),
        device=str(cfg.MODEL.DEVICE),
    )

    if not fm_extractor.available:
        logger.error("FM extractor not available")
        return {}, {}

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

    logger.info("Built features for %d classes (%s samples each)",
                len(class_features),
                {c: len(f) for c, f in sorted(class_features.items())})

    return class_features, prototypes


def compute_within_class_variance(class_features, prototypes):
    """Average L2 distance of support features to their class prototype."""
    per_class = {}
    for cls in sorted(prototypes.keys()):
        feats = class_features.get(cls, [])
        if not feats:
            per_class[cls] = 0.0
            continue
        proto = prototypes[cls]
        dists = [torch.norm(f - proto).item() for f in feats]
        per_class[cls] = float(np.mean(dists))
    return per_class


def compute_nearest_negative_margin(prototypes):
    """Min cosine distance between each prototype and its nearest different-class prototype."""
    cls_ids = sorted(prototypes.keys())
    per_class = {}

    for cls in cls_ids:
        proto = prototypes[cls]
        min_dist = float("inf")
        nearest_cls = None
        for other_cls in cls_ids:
            if other_cls == cls:
                continue
            other_proto = prototypes[other_cls]
            # Cosine distance = 1 - cosine_similarity
            cos_sim = torch.dot(proto, other_proto).item()
            cos_dist = 1.0 - cos_sim
            if cos_dist < min_dist:
                min_dist = cos_dist
                nearest_cls = other_cls
        per_class[cls] = {
            "margin": float(min_dist) if min_dist != float("inf") else 0.0,
            "nearest_class": nearest_cls,
        }

    return per_class


def compute_prototype_purity(class_features, prototypes):
    """Silhouette-like score: (b - a) / max(a, b) per class.

    a = mean distance to own prototype
    b = mean distance to nearest other prototype
    """
    cls_ids = sorted(prototypes.keys())
    per_class = {}

    for cls in cls_ids:
        feats = class_features.get(cls, [])
        if not feats:
            per_class[cls] = 0.0
            continue

        proto = prototypes[cls]
        # a: mean distance to own prototype
        a = np.mean([torch.norm(f - proto).item() for f in feats])

        # b: mean distance to nearest other prototype
        b_values = []
        for f in feats:
            min_other_dist = float("inf")
            for other_cls in cls_ids:
                if other_cls == cls:
                    continue
                d = torch.norm(f - prototypes[other_cls]).item()
                if d < min_other_dist:
                    min_other_dist = d
            if min_other_dist < float("inf"):
                b_values.append(min_other_dist)

        if not b_values:
            per_class[cls] = 0.0
            continue

        b = np.mean(b_values)
        purity = (b - a) / max(a, b, 1e-8)
        per_class[cls] = float(purity)

    return per_class


def main():
    parser = argparse.ArgumentParser(description="Compute prototype-space statistics")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Config overrides")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)

    logging.basicConfig(level=logging.INFO)

    logger.info("Building support features...")
    class_features, prototypes = build_support_features(cfg)

    if not prototypes:
        logger.error("No prototypes built. Check config and support set.")
        return

    # Compute metrics
    variance = compute_within_class_variance(class_features, prototypes)
    margin = compute_nearest_negative_margin(prototypes)
    purity = compute_prototype_purity(class_features, prototypes)

    # Aggregate
    avg_var = np.mean(list(variance.values()))
    avg_margin = np.mean([m["margin"] for m in margin.values()])
    avg_purity = np.mean(list(purity.values()))

    # Per-class table
    per_class = {}
    for cls in sorted(prototypes.keys()):
        per_class[int(cls)] = {
            "variance": variance.get(cls, 0),
            "margin": margin.get(cls, {}).get("margin", 0),
            "nearest_class": margin.get(cls, {}).get("nearest_class"),
            "purity": purity.get(cls, 0),
            "n_support": len(class_features.get(cls, [])),
        }

    results = {
        "aggregate": {
            "avg_variance": float(avg_var),
            "avg_margin": float(avg_margin),
            "avg_purity": float(avg_purity),
            "n_classes": len(prototypes),
        },
        "per_class": per_class,
        "config": {
            "dataset": cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else cfg.DATASETS.TRAIN[0],
            "method": cfg.NOVEL_METHODS.METHOD if cfg.NOVEL_METHODS.ENABLE else "vanilla_pcb",
            "fm_model": str(getattr(cfg.NOVEL_METHODS.PCB_FMA_ENHANCED, "FM_MODEL_NAME", "unknown")),
        },
    }

    output_path = args.output or os.path.join(cfg.OUTPUT_DIR, "prototype_stats.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Prototype-Space Statistics")
    print("=" * 60)
    print(f"  Avg Within-Class Variance: {avg_var:.4f}")
    print(f"  Avg Nearest-Neg Margin:    {avg_margin:.4f}")
    print(f"  Avg Prototype Purity:      {avg_purity:.4f}")
    print(f"  Number of classes:         {len(prototypes)}")
    print()
    for cls in sorted(per_class.keys()):
        c = per_class[cls]
        print(f"  Class {cls:2d}: Var={c['variance']:.4f}  Margin={c['margin']:.4f}  "
              f"Purity={c['purity']:.4f}  (K={c['n_support']})")
    print(f"\n  Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
