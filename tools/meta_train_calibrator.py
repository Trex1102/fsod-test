#!/usr/bin/env python3
"""
Meta-train the calibration network for Meta-PCB.

Trains a small MLP to learn non-linear score calibration by simulating
few-shot episodes from base classes. The trained calibrator replaces
the fixed linear alpha in PCB.

Usage:
    python3 tools/meta_train_calibrator.py \
        --config-file configs/voc/defrcn_det_r101_base1.yaml \
        --base-model checkpoints/voc/vanilla_defrcn/defrcn_det_r101_base1/model_final.pth \
        --output calibrators/meta_pcb_split1.pth \
        --episodes 10000

The trained calibrator is then used by setting:
    NOVEL_METHODS.META_PCB.CALIBRATOR_PATH: "calibrators/meta_pcb_split1.pth"
"""

import os
import sys
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectron2.config import get_cfg
from defrcn.config import defaults  # noqa: register custom config
import defrcn.data  # noqa: register VOC/COCO datasets
from defrcn.evaluation.novel_methods.meta_calibration import MetaCalibrationNet
from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-train PCB calibrator")
    parser.add_argument("--config-file", required=True, help="Base training config")
    parser.add_argument("--base-model", required=True, help="Path to trained base model")
    parser.add_argument("--output", default="calibrators/meta_pcb.pth", help="Output path")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of meta-training episodes")
    parser.add_argument("--n-way", type=int, default=5, help="Classes per episode")
    parser.add_argument("--k-shot", type=int, default=1, help="Support shots per class")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Calibrator hidden dim")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pcb-modelpath", default="",
        help="Path to ImageNet pretrained model for PCB (e.g., resnet101-5d3b4d8f.pth)",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.base_model
    if args.pcb_modelpath:
        cfg.TEST.PCB_MODELPATH = args.pcb_modelpath
    cfg.freeze()
    return cfg


def collect_base_class_features(cfg):
    """Collect per-class prototype features from support set using PCB pipeline."""
    logger.info("Collecting base class features for meta-training...")

    # Build PCB to extract features (reuses the same pipeline)
    pcb = PrototypicalCalibrationBlock(cfg)

    class_features = {}
    for cls in pcb._real_class_features:
        feats = pcb._real_class_features[cls]
        if feats:
            class_features[cls] = torch.stack(feats, dim=0)

    logger.info("Collected features for %d classes", len(class_features))
    return class_features, pcb


def simulate_episode(class_features, pcb, n_way, k_shot):
    """Simulate one few-shot episode from base classes.

    Returns list of (det_score, cos_sim, support_stats, roi_stats, label) tuples.
    """
    available_classes = [c for c in class_features if len(class_features[c]) >= k_shot + 1]
    if len(available_classes) < n_way:
        n_way = len(available_classes)
    if n_way < 2:
        return []

    selected = random.sample(available_classes, n_way)

    # Build support prototypes
    support_protos = {}
    support_stats = {}
    query_data = []

    for cls in selected:
        feats = class_features[cls]
        n_available = len(feats)

        # Randomly split into support and query
        indices = list(range(n_available))
        random.shuffle(indices)
        support_idx = indices[:k_shot]
        query_idx = indices[k_shot : k_shot + 5]  # up to 5 query

        support_feats = feats[support_idx]
        proto = support_feats.mean(dim=0)
        support_protos[cls] = proto

        # Support statistics
        mean_norm = float(proto.norm().item())
        std_norm = float(support_feats.std(dim=0).norm().item()) if k_shot > 1 else 0.0
        count_norm = min(float(k_shot) / 10.0, 1.0)

        if k_shot > 1:
            proto_normed = F.normalize(proto.unsqueeze(0), dim=1)
            feat_normed = F.normalize(support_feats, dim=1)
            cos_sims = torch.mm(feat_normed, proto_normed.t()).squeeze(1)
            dispersion = float((1.0 - cos_sims.mean()).item())
        else:
            dispersion = 0.5

        support_stats[cls] = {
            "mean_norm": min(mean_norm / 100.0, 1.0),
            "std_norm": min(std_norm / 50.0, 1.0),
            "count_norm": count_norm,
            "dispersion": dispersion,
        }

        # Query samples
        for qi in query_idx:
            query_data.append((feats[qi], cls))

    if not query_data:
        return []

    # Simulate detection + calibration for each query
    examples = []
    for query_feat, true_cls in query_data:
        # Cosine similarity to all prototypes
        query_normed = F.normalize(query_feat.unsqueeze(0), dim=1)
        sims = {}
        for cls, proto in support_protos.items():
            proto_normed = F.normalize(proto.unsqueeze(0), dim=1)
            sim = float(torch.mm(query_normed, proto_normed.t()).item())
            sims[cls] = sim

        # Simulate detector score: higher for correct class
        # Use softmax over similarities as proxy
        sim_values = torch.tensor([sims[c] for c in selected])
        det_probs = F.softmax(sim_values / 0.1, dim=0)
        cls_idx = selected.index(true_cls)
        det_score = float(det_probs[cls_idx].item())

        # Add noise to simulate real detector scores
        det_score = max(0.0, min(1.0, det_score + random.gauss(0, 0.1)))

        cos_sim = (sims[true_cls] + 1.0) / 2.0  # map to [0,1]
        roi_norm = min(float(query_feat.norm().item()) / 100.0, 1.0)

        # Score entropy
        s = max(1e-8, min(1 - 1e-8, det_score))
        score_entropy = -(s * np.log(s) + (1 - s) * np.log(1 - s))

        stats = support_stats[true_cls]

        # Ground truth: 1.0 if correct class and high similarity, 0.0 otherwise
        # Use smooth label based on actual similarity
        gt_label = max(0.0, min(1.0, (sims[true_cls] + 1.0) / 2.0))

        examples.append({
            "det_score": det_score,
            "cos_sim": cos_sim,
            "mean_norm": stats["mean_norm"],
            "std_norm": stats["std_norm"],
            "count_norm": stats["count_norm"],
            "dispersion": stats["dispersion"],
            "roi_norm": roi_norm,
            "score_entropy": score_entropy,
            "label": gt_label,
        })

    return examples


def train(args, cfg):
    """Meta-train the calibration network."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect base class features
    class_features, pcb = collect_base_class_features(cfg)

    # Move features to device
    class_features = {cls: feats.to(device) for cls, feats in class_features.items()}

    # Build calibrator
    calibrator = MetaCalibrationNet(
        input_dim=8,
        hidden_dim=args.hidden_dim,
        residual_scale=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    logger.info("Starting meta-training for %d episodes...", args.episodes)

    running_loss = 0.0
    for episode in range(args.episodes):
        examples = simulate_episode(
            class_features, pcb, args.n_way, args.k_shot
        )
        if not examples:
            continue

        # Batch all examples (must be float32 to match model weights)
        det_scores = torch.tensor([e["det_score"] for e in examples], device=device, dtype=torch.float32)
        cos_sims = torch.tensor([e["cos_sim"] for e in examples], device=device, dtype=torch.float32)
        mean_norms = torch.tensor([e["mean_norm"] for e in examples], device=device, dtype=torch.float32)
        std_norms = torch.tensor([e["std_norm"] for e in examples], device=device, dtype=torch.float32)
        count_norms = torch.tensor([e["count_norm"] for e in examples], device=device, dtype=torch.float32)
        dispersions = torch.tensor([e["dispersion"] for e in examples], device=device, dtype=torch.float32)
        roi_norms = torch.tensor([e["roi_norm"] for e in examples], device=device, dtype=torch.float32)
        entropies = torch.tensor([e["score_entropy"] for e in examples], device=device, dtype=torch.float32)
        labels = torch.tensor([e["label"] for e in examples], device=device, dtype=torch.float32)

        calibrated = calibrator(
            det_scores, cos_sims,
            mean_norms, std_norms,
            count_norms, dispersions,
            roi_norms, entropies,
        )

        loss = F.binary_cross_entropy(calibrated, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (episode + 1) % 1000 == 0:
            avg_loss = running_loss / 1000
            logger.info(
                "Episode %d/%d, Loss: %.4f, LR: %.6f",
                episode + 1, args.episodes, avg_loss, scheduler.get_last_lr()[0],
            )
            running_loss = 0.0

    # Save calibrator
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(calibrator.state_dict(), args.output)
    logger.info("Calibrator saved to %s", args.output)

    # Verify
    calibrator.eval()
    with torch.no_grad():
        test_out = calibrator(
            torch.tensor(0.8, device=device),
            torch.tensor(0.6, device=device),
            torch.tensor(0.5, device=device),
            torch.tensor(0.1, device=device),
            torch.tensor(0.1, device=device),
            torch.tensor(0.3, device=device),
            torch.tensor(0.5, device=device),
            torch.tensor(0.3, device=device),
        )
        logger.info(
            "Verification: input(det=0.8, cos=0.6) -> calibrated=%.4f "
            "(vanilla PCB would give %.4f)",
            test_out.item(),
            0.5 * 0.8 + 0.5 * 0.6,
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()
    cfg = setup_cfg(args)
    train(args, cfg)


if __name__ == "__main__":
    main()
