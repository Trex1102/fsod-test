"""
Generate reliability diagrams and confidence histograms for Appendix D.

Reads calibration metrics JSON (from compute_calibration_metrics.py) or
runs inference directly to produce publication-quality reliability plots.

Usage:
    # From pre-computed metrics:
    python3 tools/generate_reliability_plots.py \
        --input calibration_metrics.json \
        --output figures/reliability_diagram.pdf

    # Or run inference directly:
    python3 tools/generate_reliability_plots.py \
        --config-file configs/voc/... \
        --output figures/reliability_diagram.pdf \
        --opts MODEL.WEIGHTS path/to/model.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def load_or_compute_calibration_data(args):
    """Load calibration data from JSON or compute from config."""
    input_path = args.input
    if isinstance(input_path, (list, tuple)):
        input_path = input_path[0] if len(input_path) == 1 else None

    if input_path and os.path.exists(input_path):
        with open(input_path) as f:
            return json.load(f)

    if not args.config_file:
        raise ValueError("Either --input (JSON) or --config-file must be provided")

    # Run calibration metrics computation
    from tools.compute_calibration_metrics import (
        get_cfg, set_global_cfg, build_pcb, run_inference_and_collect,
        compute_ece, compute_brier, get_class_ids,
    )
    from detectron2.checkpoint import DetectionCheckpointer
    from defrcn.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    pcb = build_pcb(cfg)
    all_matches = run_inference_and_collect(cfg, model, pcb)

    scores = np.array([m["score"] for m in all_matches])
    correct = np.array([m["is_correct"] for m in all_matches])

    n_bins = args.n_bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 else (scores >= lo) & (scores <= hi)
        count = int(mask.sum())
        if count > 0:
            bin_data.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "avg_confidence": float(scores[mask].mean()),
                "avg_accuracy": float(correct[mask].mean()),
                "count": count,
            })

    return {
        "ece": compute_ece(scores, correct, n_bins),
        "reliability_bins": bin_data,
        "n_total_detections": len(all_matches),
    }


def plot_reliability_diagram(data, output_path, title="Reliability Diagram"):
    """Plot reliability diagram with confidence histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    bins = data["reliability_bins"]
    if not bins:
        logger.warning("No bins to plot.")
        return

    ece = data.get("ece", 0)

    fig = plt.figure(figsize=(5, 5.5))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # --- Top: Reliability diagram ---
    ax1 = fig.add_subplot(gs[0])

    confidences = [b["avg_confidence"] for b in bins]
    accuracies = [b["avg_accuracy"] for b in bins]
    bin_centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    bar_width = bins[0]["bin_hi"] - bins[0]["bin_lo"]

    # Gap bars
    gap_colors = []
    for c, a in zip(confidences, accuracies):
        gap_colors.append("#e74c3c" if c > a else "#2ecc71")

    ax1.bar(bin_centers, accuracies, width=bar_width * 0.85, color="#3498db",
            edgecolor="white", linewidth=0.5, label="Accuracy", alpha=0.85, zorder=2)

    # Gap overlay
    for i, (bc, c, a) in enumerate(zip(bin_centers, confidences, accuracies)):
        if c > a:
            ax1.bar(bc, c - a, width=bar_width * 0.85, bottom=a,
                    color="#e74c3c", alpha=0.35, zorder=3)

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", zorder=1)
    ax1.scatter(confidences, accuracies, color="#2c3e50", s=20, zorder=4)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(f"{title}\nECE = {ece:.4f}", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.3)

    # --- Bottom: Confidence histogram ---
    ax2 = fig.add_subplot(gs[1])

    counts = [b["count"] for b in bins]
    ax2.bar(bin_centers, counts, width=bar_width * 0.85, color="#95a5a6",
            edgecolor="white", linewidth=0.5)

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram to: {output_path}")


def plot_comparison_reliability(data_list, labels, output_path):
    """Plot side-by-side reliability diagrams for multiple methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, data, label in zip(axes, data_list, labels):
        bins = data["reliability_bins"]
        ece = data.get("ece", 0)

        confidences = [b["avg_confidence"] for b in bins]
        accuracies = [b["avg_accuracy"] for b in bins]
        bin_centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        bar_width = bins[0]["bin_hi"] - bins[0]["bin_lo"] if bins else 0.067

        ax.bar(bin_centers, accuracies, width=bar_width * 0.85, color="#3498db",
               edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.scatter(confidences, accuracies, color="#2c3e50", s=15, zorder=4)

        for bc, c, a in zip(bin_centers, confidences, accuracies):
            if c > a:
                ax.bar(bc, c - a, width=bar_width * 0.85, bottom=a,
                       color="#e74c3c", alpha=0.35)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_title(f"{label}\nECE = {ece:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Accuracy", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison reliability diagram to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate reliability diagrams")
    parser.add_argument("--input", nargs="+", default=[], help="Pre-computed calibration JSON(s)")
    parser.add_argument("--labels", nargs="+", default=[], help="Labels for comparison plot")
    parser.add_argument("--config-file", default="", help="Config file (if computing from scratch)")
    parser.add_argument("--output", default="figures/reliability_diagram.pdf", help="Output plot path")
    parser.add_argument("--title", default="Reliability Diagram", help="Plot title")
    parser.add_argument("--n-bins", type=int, default=15, help="Number of bins")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Config overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if len(args.input) > 1:
        # Comparison mode: multiple JSONs
        data_list = []
        for inp in args.input:
            with open(inp) as f:
                data_list.append(json.load(f))
        labels = args.labels if args.labels else [os.path.basename(p).replace(".json", "") for p in args.input]
        plot_comparison_reliability(data_list, labels, args.output)
    elif len(args.input) == 1:
        data = load_or_compute_calibration_data(args)
        plot_reliability_diagram(data, args.output, title=args.title)
    elif args.config_file:
        data = load_or_compute_calibration_data(args)
        plot_reliability_diagram(data, args.output, title=args.title)
    else:
        parser.error("Provide --input JSON(s) or --config-file")


if __name__ == "__main__":
    main()
