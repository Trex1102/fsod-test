"""
Compute feature-space geometry metrics for Appendix E.

Analyzes the FM feature space of novel-class prototypes and support features:
  - Inter-class / intra-class distance ratio (Fisher discriminant ratio)
  - Silhouette score
  - Hubness analysis (how often features are nearest neighbors of others)
  - Cosine similarity distribution statistics
  - Prototype angular spread

Usage:
    python3 tools/compute_feature_geometry.py \
        --config-file configs/voc/... \
        --output figures/feature_geometry/ \
        --opts MODEL.WEIGHTS path/to/model.pth

Output: JSON metrics + visualization plots.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defrcn.config import get_cfg, set_global_cfg

logger = logging.getLogger(__name__)


def compute_fisher_ratio(class_features, prototypes):
    """Fisher discriminant ratio: inter-class variance / intra-class variance."""
    cls_ids = sorted(prototypes.keys())
    if len(cls_ids) < 2:
        return {"fisher_ratio": 0.0}

    # Global mean
    all_protos = torch.stack([prototypes[c] for c in cls_ids], dim=0)
    global_mean = all_protos.mean(dim=0)

    # Inter-class scatter: variance of prototypes around global mean
    inter_var = torch.mean(torch.sum((all_protos - global_mean) ** 2, dim=1)).item()

    # Intra-class scatter: average within-class variance
    intra_vars = []
    for cls in cls_ids:
        feats = class_features.get(cls, [])
        if len(feats) < 2:
            continue
        feat_stack = torch.stack(feats, dim=0)
        proto = prototypes[cls]
        var = torch.mean(torch.sum((feat_stack - proto) ** 2, dim=1)).item()
        intra_vars.append(var)

    intra_var = np.mean(intra_vars) if intra_vars else 1e-8

    return {
        "fisher_ratio": inter_var / max(intra_var, 1e-8),
        "inter_class_variance": inter_var,
        "intra_class_variance": intra_var,
    }


def compute_silhouette(class_features, prototypes):
    """Compute silhouette score for the prototype feature space."""
    cls_ids = sorted(prototypes.keys())
    if len(cls_ids) < 2:
        return {"avg_silhouette": 0.0, "per_class": {}}

    per_class = {}
    all_scores = []

    for cls in cls_ids:
        feats = class_features.get(cls, [])
        if not feats:
            per_class[cls] = 0.0
            continue

        scores = []
        for f in feats:
            # a: mean distance to same-class samples
            a_dists = [torch.norm(f - other).item() for other in feats if not torch.equal(f, other)]
            a = np.mean(a_dists) if a_dists else 0.0

            # b: min mean distance to any other class
            b = float("inf")
            for other_cls in cls_ids:
                if other_cls == cls:
                    continue
                other_feats = class_features.get(other_cls, [])
                if not other_feats:
                    continue
                b_dists = [torch.norm(f - of).item() for of in other_feats]
                b = min(b, np.mean(b_dists))

            if b == float("inf"):
                b = 0
            s = (b - a) / max(a, b, 1e-8)
            scores.append(s)

        per_class[cls] = float(np.mean(scores)) if scores else 0.0
        all_scores.extend(scores)

    return {
        "avg_silhouette": float(np.mean(all_scores)) if all_scores else 0.0,
        "per_class": {int(k): v for k, v in per_class.items()},
    }


def compute_cosine_similarity_matrix(prototypes):
    """Compute pairwise cosine similarity matrix between prototypes."""
    cls_ids = sorted(prototypes.keys())
    n = len(cls_ids)
    sim_matrix = np.zeros((n, n))

    for i, ci in enumerate(cls_ids):
        for j, cj in enumerate(cls_ids):
            sim_matrix[i, j] = torch.dot(prototypes[ci], prototypes[cj]).item()

    return {
        "class_ids": cls_ids,
        "similarity_matrix": sim_matrix.tolist(),
        "off_diagonal_mean": float(np.mean(sim_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
        "off_diagonal_std": float(np.std(sim_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
        "min_pairwise_sim": float(np.min(sim_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
        "max_pairwise_sim": float(np.max(sim_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0.0,
    }


def compute_hubness(class_features, prototypes, k=5):
    """Compute hubness: how many times each prototype is a k-NN of support features."""
    cls_ids = sorted(prototypes.keys())
    all_feats = []
    all_labels = []
    for cls in cls_ids:
        for f in class_features.get(cls, []):
            all_feats.append(f)
            all_labels.append(cls)

    if len(all_feats) < k + 1:
        return {"skewness": 0.0, "per_class_hubness": {}}

    feat_stack = torch.stack(all_feats, dim=0)
    proto_stack = torch.stack([prototypes[c] for c in cls_ids], dim=0)

    # Compute distances from each feature to each prototype
    # cosine similarity
    sims = torch.mm(feat_stack, proto_stack.T)  # (N, C)

    # Count how many times each prototype is in top-k
    _, topk_indices = sims.topk(min(k, len(cls_ids)), dim=1)
    hub_counts = np.zeros(len(cls_ids))
    for row in topk_indices:
        for idx in row:
            hub_counts[idx.item()] += 1

    # Hubness skewness
    from scipy.stats import skew
    skewness = float(skew(hub_counts))

    return {
        "skewness": skewness,
        "hub_counts": {int(cls_ids[i]): int(hub_counts[i]) for i in range(len(cls_ids))},
    }


def plot_feature_geometry(results, output_dir):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Cosine similarity heatmap
    sim_data = results["cosine_similarity"]
    cls_ids = sim_data["class_ids"]
    sim_matrix = np.array(sim_data["similarity_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cls_ids)))
    ax.set_yticks(range(len(cls_ids)))
    ax.set_xticklabels([str(c) for c in cls_ids], fontsize=8)
    ax.set_yticklabels([str(c) for c in cls_ids], fontsize=8)
    ax.set_title("Prototype Cosine Similarity", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(len(cls_ids)):
        for j in range(len(cls_ids)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(sim_matrix[i, j]) > 0.5 else "black")

    fig.savefig(os.path.join(output_dir, "cosine_similarity_heatmap.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Silhouette per-class bar chart
    sil = results["silhouette"]
    if sil["per_class"]:
        cls_labels = sorted(sil["per_class"].keys())
        sil_vals = [sil["per_class"][c] for c in cls_labels]

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in sil_vals]
        ax.bar(range(len(cls_labels)), sil_vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(cls_labels)))
        ax.set_xticklabels([str(c) for c in cls_labels], fontsize=9)
        ax.set_xlabel("Class ID", fontsize=10)
        ax.set_ylabel("Silhouette Score", fontsize=10)
        ax.set_title(f"Per-Class Silhouette (avg={sil['avg_silhouette']:.3f})", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        fig.savefig(os.path.join(output_dir, "silhouette_scores.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved feature geometry plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute feature-space geometry")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--output", default="figures/feature_geometry/", help="Output directory")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Config overrides")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)

    logging.basicConfig(level=logging.INFO)

    # Build support features
    from tools.compute_prototype_stats import build_support_features
    class_features, prototypes = build_support_features(cfg)

    if not prototypes:
        logger.error("No prototypes built.")
        return

    logger.info("Computing feature-space geometry metrics...")

    fisher = compute_fisher_ratio(class_features, prototypes)
    silhouette = compute_silhouette(class_features, prototypes)
    cosine_sim = compute_cosine_similarity_matrix(prototypes)

    try:
        hubness = compute_hubness(class_features, prototypes)
    except ImportError:
        logger.warning("scipy not available, skipping hubness computation")
        hubness = {"skewness": 0.0, "hub_counts": {}}

    results = {
        "fisher": fisher,
        "silhouette": silhouette,
        "cosine_similarity": cosine_sim,
        "hubness": hubness,
        "config": {
            "dataset": cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else cfg.DATASETS.TRAIN[0],
            "method": cfg.NOVEL_METHODS.METHOD if cfg.NOVEL_METHODS.ENABLE else "vanilla_pcb",
        },
    }

    # Save JSON
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "feature_geometry.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate plots
    plot_feature_geometry(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("Feature-Space Geometry Summary")
    print("=" * 60)
    print(f"  Fisher Ratio:        {fisher['fisher_ratio']:.4f}")
    print(f"  Inter-class Var:     {fisher['inter_class_variance']:.4f}")
    print(f"  Intra-class Var:     {fisher['intra_class_variance']:.4f}")
    print(f"  Avg Silhouette:      {silhouette['avg_silhouette']:.4f}")
    print(f"  Pairwise Cos (mean): {cosine_sim['off_diagonal_mean']:.4f}")
    print(f"  Pairwise Cos (std):  {cosine_sim['off_diagonal_std']:.4f}")
    print(f"  Hubness Skewness:    {hubness['skewness']:.4f}")
    print(f"\n  Saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
