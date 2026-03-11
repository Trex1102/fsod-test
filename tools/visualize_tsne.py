"""
t-SNE Feature Visualization for PCB-FMA.

Extracts foundation model (DINOv2 / DINOv1 / CLIP) CLS features from
support and test images, runs t-SNE, and produces publication-quality
scatter plots.

Modes:
  support_test  - Support prototypes (stars) + test GT features (dots)
  support_only  - Support features only (colored by class)
  compare_fm    - Side-by-side: ImageNet ResNet vs DINOv2 features

Usage:
    # Support + test features
    python tools/visualize_tsne.py --split 1 --shot 5 --seed 0 \
        --output figures/tsne_split1_5shot.pdf

    # Compare ResNet vs DINOv2
    python tools/visualize_tsne.py --split 1 --shot 5 --seed 0 \
        --mode compare_fm --output figures/tsne_compare.pdf

    # Include base-class features
    python tools/visualize_tsne.py --split 1 --shot 5 --seed 0 \
        --include-base --max-base-per-class 20 --output figures/tsne_base.pdf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Register datasets on import
from defrcn.data.builtin import register_all_voc  # noqa: F401
from detectron2.data import DatasetCatalog

# ---- VOC class definitions ----
ALL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

NOVEL_CLASSES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

BASE_CLASSES = {
    1: ["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor"],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor"],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
        "horse", "person", "pottedplant", "train", "tvmonitor"],
}


# ---- Model loading ----

def load_fm_model(model_name, device="cuda"):
    """Load foundation model and return (model, norm_mean, norm_std)."""
    norm_mean = [0.485, 0.456, 0.406]  # ImageNet default
    norm_std = [0.229, 0.224, 0.225]

    if model_name.startswith("dinov2_"):
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    elif model_name.startswith("dino_"):
        model = torch.hub.load("facebookresearch/dino:main", model_name)
    elif model_name.startswith("clip_"):
        norm_mean = [0.48145466, 0.4578275, 0.40821073]
        norm_std = [0.26862954, 0.26130258, 0.27577711]
        clip_map = {
            "clip_vitb16": "ViT-B-16", "clip_vitb32": "ViT-B-32", "clip_vitl14": "ViT-L-14",
        }
        open_clip_name = clip_map.get(model_name, "ViT-B-16")
        try:
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms(open_clip_name, pretrained="openai")
        except ImportError:
            import clip
            openai_name = open_clip_name.replace("-", "/")
            clip_model, _ = clip.load(openai_name, device="cpu")

        class CLIPVisual(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m.encode_image(x).float()

        model = CLIPVisual(clip_model)
    elif model_name == "resnet101":
        import torchvision.models as models
        resnet = models.resnet101(pretrained=True)
        # Remove final FC, keep avgpool output (2048-d)
        model = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, norm_mean, norm_std


def preprocess_crop(crop, norm_mean, norm_std, roi_size=224, device="cuda"):
    """Preprocess a BGR crop for the FM."""
    mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)

    resized = cv2.resize(crop, (roi_size, roi_size))
    t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0).to(device)
    return (t - mean) / std


def batch_preprocess(crops, norm_mean, norm_std, roi_size=224, device="cuda"):
    """Preprocess a list of BGR crops into a batch tensor."""
    mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)

    tensors = []
    for crop in crops:
        resized = cv2.resize(crop, (roi_size, roi_size))
        t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors, dim=0).to(device)
    return (batch - mean) / std


# ---- Feature extraction from datasets ----

def extract_features_from_dataset(
    fm_model, norm_mean, norm_std, dataset_dicts, class_filter=None,
    max_per_class=None, batch_size=32, device="cuda"
):
    """Extract FM features from dataset GT boxes.

    Args:
        fm_model: Foundation model
        norm_mean/norm_std: Normalization constants
        dataset_dicts: List of detectron2 dataset dicts
        class_filter: Optional set of class names to include
        max_per_class: Max features per class (None = unlimited)
        batch_size: Batch size for FM forward pass

    Returns:
        features: np.array (N, dim)
        labels: list of class name strings
    """
    features = []
    labels = []
    class_counts = defaultdict(int)
    pending_crops = []
    pending_labels = []

    def flush():
        nonlocal pending_crops, pending_labels
        if not pending_crops:
            return
        batch = batch_preprocess(pending_crops, norm_mean, norm_std, device=device)
        with torch.no_grad():
            feats = fm_model(batch)
        for i, feat in enumerate(feats):
            features.append(F.normalize(feat, dim=0).cpu().numpy())
            labels.append(pending_labels[i])
        pending_crops = []
        pending_labels = []

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        for ann in d.get("annotations", []):
            cat_id = ann["category_id"]
            cls_name = ALL_CLASSES[cat_id] if cat_id < len(ALL_CLASSES) else str(cat_id)

            if class_filter and cls_name not in class_filter:
                continue
            if max_per_class and class_counts[cls_name] >= max_per_class:
                continue

            bbox = ann["bbox"]
            # Detectron2 VOC uses XYXY_ABS (BoxMode 0)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            pending_crops.append(crop)
            pending_labels.append(cls_name)
            class_counts[cls_name] += 1

            if len(pending_crops) >= batch_size:
                flush()

    flush()

    if features:
        return np.array(features), labels
    return np.empty((0, 768)), []


# ---- Plotting ----

# Colorblind-friendly palette (Tab10 extended)
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def plot_tsne(
    embeddings, labels, class_names, output_path, title="",
    support_mask=None, figsize=(7, 6), dpi=300
):
    """Plot t-SNE scatter with class-colored points."""
    color_map = {}
    for i, cls in enumerate(class_names):
        color_map[cls] = CLASS_COLORS[i % len(CLASS_COLORS)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for cls in class_names:
        mask = np.array([l == cls for l in labels])
        if not mask.any():
            continue

        if support_mask is not None:
            # Test points: small, semi-transparent
            test_mask = mask & ~support_mask
            if test_mask.any():
                ax.scatter(
                    embeddings[test_mask, 0], embeddings[test_mask, 1],
                    c=[color_map[cls]], alpha=0.35, s=15, edgecolors="none",
                )
            # Support points: large stars with border
            sup_mask = mask & support_mask
            if sup_mask.any():
                ax.scatter(
                    embeddings[sup_mask, 0], embeddings[sup_mask, 1],
                    c=[color_map[cls]], alpha=0.95, s=120, edgecolors="black",
                    linewidths=1.2, marker="*", zorder=5,
                )
        else:
            ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=[color_map[cls]], alpha=0.6, s=25, edgecolors="none",
            )

    # Legend
    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in class_names if cls in set(labels)]
    if support_mask is not None:
        handles.append(plt.Line2D([], [], marker="*", color="gray", markersize=10,
                                  label="Support", linestyle="None"))
        handles.append(plt.Line2D([], [], marker="o", color="gray", markersize=4,
                                  alpha=0.4, label="Test", linestyle="None"))
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.8)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_tsne_side_by_side(
    emb_left, labels_left, emb_right, labels_right, class_names,
    output_path, title_left="ResNet-101", title_right="DINOv2 ViT-B/14",
    support_mask_left=None, support_mask_right=None, figsize=(14, 6), dpi=300
):
    """Side-by-side t-SNE comparison."""
    color_map = {}
    for i, cls in enumerate(class_names):
        color_map[cls] = CLASS_COLORS[i % len(CLASS_COLORS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, emb, lbls, smask, title in [
        (ax1, emb_left, labels_left, support_mask_left, title_left),
        (ax2, emb_right, labels_right, support_mask_right, title_right),
    ]:
        for cls in class_names:
            mask = np.array([l == cls for l in lbls])
            if not mask.any():
                continue

            if smask is not None:
                test_mask = mask & ~smask
                if test_mask.any():
                    ax.scatter(emb[test_mask, 0], emb[test_mask, 1],
                               c=[color_map[cls]], alpha=0.35, s=15, edgecolors="none")
                sup_mask = mask & smask
                if sup_mask.any():
                    ax.scatter(emb[sup_mask, 0], emb[sup_mask, 1],
                               c=[color_map[cls]], alpha=0.95, s=120, edgecolors="black",
                               linewidths=1.2, marker="*", zorder=5)
            else:
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=[color_map[cls]], alpha=0.6, s=25, edgecolors="none")

        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in class_names]
    if support_mask_left is not None:
        handles.append(plt.Line2D([], [], marker="*", color="gray", markersize=10,
                                  label="Support", linestyle="None"))
        handles.append(plt.Line2D([], [], marker="o", color="gray", markersize=4,
                                  alpha=0.4, label="Test", linestyle="None"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(class_names) + 2, 7),
               fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="t-SNE feature visualization for PCB-FMA")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3],
                        help="VOC novel split")
    parser.add_argument("--shot", type=int, default=5,
                        help="Number of shots (1, 2, 3, 5, 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for few-shot data")
    parser.add_argument("--fm-model", type=str, default="dinov2_vitb14",
                        help="FM model: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, "
                             "dino_vitb16, clip_vitb16, resnet101")
    parser.add_argument("--output", type=str, default="figures/tsne.pdf",
                        help="Output file path (.pdf or .png)")
    parser.add_argument("--mode", choices=["support_test", "support_only", "compare_fm"],
                        default="support_test",
                        help="Visualization mode")
    parser.add_argument("--max-test-per-class", type=int, default=50,
                        help="Max test samples per class")
    parser.add_argument("--include-base", action="store_true",
                        help="Include base-class features (from test set)")
    parser.add_argument("--max-base-per-class", type=int, default=20,
                        help="Max base-class samples per class")
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity")
    parser.add_argument("--n-iter", type=int, default=1000,
                        help="t-SNE iterations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    novel_cls = NOVEL_CLASSES[args.split]
    base_cls = BASE_CLASSES[args.split]

    # Dataset names
    support_name = f"voc_2007_trainval_novel{args.split}_{args.shot}shot_seed{args.seed}"
    test_novel_name = f"voc_2007_test_novel{args.split}"
    test_base_name = f"voc_2007_test_base{args.split}"

    print(f"Split {args.split}: novel={novel_cls}")
    print(f"Support: {support_name}")
    print(f"Test novel: {test_novel_name}")

    support_data = DatasetCatalog.get(support_name)
    test_novel_data = DatasetCatalog.get(test_novel_name)
    print(f"  Support entries: {len(support_data)}, Test novel entries: {len(test_novel_data)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.mode == "compare_fm":
        # ---- Side-by-side: ResNet-101 vs DINOv2 ----
        print("Loading ResNet-101...")
        resnet_model, rn_mean, rn_std = load_fm_model("resnet101", device=args.device)
        print("Loading DINOv2...")
        dino_model, dn_mean, dn_std = load_fm_model(args.fm_model, device=args.device)

        print("Extracting ResNet features (support)...")
        rn_sup_feats, rn_sup_labels = extract_features_from_dataset(
            resnet_model, rn_mean, rn_std, support_data, class_filter=set(novel_cls),
            device=args.device)
        print("Extracting ResNet features (test)...")
        rn_test_feats, rn_test_labels = extract_features_from_dataset(
            resnet_model, rn_mean, rn_std, test_novel_data, class_filter=set(novel_cls),
            max_per_class=args.max_test_per_class, device=args.device)

        print("Extracting DINOv2 features (support)...")
        dn_sup_feats, dn_sup_labels = extract_features_from_dataset(
            dino_model, dn_mean, dn_std, support_data, class_filter=set(novel_cls),
            device=args.device)
        print("Extracting DINOv2 features (test)...")
        dn_test_feats, dn_test_labels = extract_features_from_dataset(
            dino_model, dn_mean, dn_std, test_novel_data, class_filter=set(novel_cls),
            max_per_class=args.max_test_per_class, device=args.device)

        # Combine per model and run separate t-SNE
        rn_all = np.concatenate([rn_sup_feats, rn_test_feats])
        rn_labels = rn_sup_labels + rn_test_labels
        rn_smask = np.array([True] * len(rn_sup_feats) + [False] * len(rn_test_feats))

        dn_all = np.concatenate([dn_sup_feats, dn_test_feats])
        dn_labels = dn_sup_labels + dn_test_labels
        dn_smask = np.array([True] * len(dn_sup_feats) + [False] * len(dn_test_feats))

        print(f"t-SNE on ResNet features ({len(rn_all)} points)...")
        rn_emb = TSNE(n_components=2, perplexity=min(args.perplexity, max(5, len(rn_all) - 1)),
                       random_state=42, n_iter=args.n_iter).fit_transform(rn_all)
        print(f"t-SNE on DINOv2 features ({len(dn_all)} points)...")
        dn_emb = TSNE(n_components=2, perplexity=min(args.perplexity, max(5, len(dn_all) - 1)),
                       random_state=42, n_iter=args.n_iter).fit_transform(dn_all)

        fm_label = args.fm_model.replace("_", " ").replace("dinov2", "DINOv2").replace("vitb14", "ViT-B/14")
        plot_tsne_side_by_side(
            rn_emb, rn_labels, dn_emb, dn_labels, novel_cls,
            args.output, title_left="ResNet-101 (ImageNet)",
            title_right=fm_label,
            support_mask_left=rn_smask, support_mask_right=dn_smask,
        )
        return

    # ---- Standard modes: support_test or support_only ----
    print(f"Loading FM: {args.fm_model}...")
    fm_model, norm_mean, norm_std = load_fm_model(args.fm_model, device=args.device)

    print("Extracting support features...")
    sup_feats, sup_labels = extract_features_from_dataset(
        fm_model, norm_mean, norm_std, support_data, class_filter=set(novel_cls),
        device=args.device)
    print(f"  Support: {len(sup_feats)} features from {len(set(sup_labels))} classes")

    if args.mode == "support_only":
        all_feats = sup_feats
        all_labels = sup_labels
        support_mask = None
        all_class_names = novel_cls
    else:
        # support_test
        print("Extracting test (novel) features...")
        test_feats, test_labels = extract_features_from_dataset(
            fm_model, norm_mean, norm_std, test_novel_data, class_filter=set(novel_cls),
            max_per_class=args.max_test_per_class, device=args.device)
        print(f"  Test novel: {len(test_feats)} features")

        all_feats = np.concatenate([sup_feats, test_feats])
        all_labels = sup_labels + test_labels
        support_mask = np.array([True] * len(sup_feats) + [False] * len(test_feats))
        all_class_names = novel_cls

        # Optionally include base-class features
        if args.include_base:
            print("Extracting test (base) features...")
            test_base_data = DatasetCatalog.get(test_base_name)
            base_feats, base_labels = extract_features_from_dataset(
                fm_model, norm_mean, norm_std, test_base_data, class_filter=set(base_cls),
                max_per_class=args.max_base_per_class, device=args.device)
            print(f"  Test base: {len(base_feats)} features")

            all_feats = np.concatenate([all_feats, base_feats])
            all_labels = all_labels + base_labels
            support_mask = np.concatenate([support_mask, np.array([False] * len(base_feats))])
            all_class_names = novel_cls + [c for c in base_cls if c in set(base_labels)]

    if len(all_feats) < 2:
        print("ERROR: Not enough features for t-SNE. Check dataset paths.")
        sys.exit(1)

    perp = min(args.perplexity, max(5, len(all_feats) - 1))
    print(f"Running t-SNE ({len(all_feats)} points, perplexity={perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=args.n_iter)
    embeddings = tsne.fit_transform(all_feats)

    fm_label = args.fm_model.replace("dinov2_vitb14", "DINOv2 ViT-B/14") \
                             .replace("dinov2_vits14", "DINOv2 ViT-S/14") \
                             .replace("dinov2_vitl14", "DINOv2 ViT-L/14") \
                             .replace("dino_vitb16", "DINOv1 ViT-B/16") \
                             .replace("clip_vitb16", "CLIP ViT-B/16")
    title = f"{fm_label} Features: Split {args.split}, {args.shot}-shot"
    plot_tsne(embeddings, all_labels, all_class_names, args.output,
              title=title, support_mask=support_mask)


if __name__ == "__main__":
    main()
