#!/usr/bin/env python3
"""
Qualitative Visualization: Side-by-side comparison of PCB vs PCB-FMA detections.

Generates publication-quality figures showing how PCB-FMA improves detection
through foundation model aligned calibration. Produces side-by-side images
with bounding boxes color-coded by correctness (TP/FP).

Usage:
    # Compare vanilla PCB vs PCB-FMA on a specific split/shot/seed
    python tools/visualize_detections.py \
        --pcb-dir checkpoints/voc/vanilla_defrcn/split1/1shot_seed0 \
        --fma-dir checkpoints/voc/voc_novel_methods/novelMethodsPretrainedNovelEval/pcb_fma/split1/1shot_seed0 \
        --dataset voc_2007_test_novel1 \
        --output-dir paper/figures/qualitative \
        --num-images 10 \
        --score-thresh 0.3

    # Or compare different FM model variants
    python tools/visualize_detections.py \
        --pcb-dir checkpoints/voc/vanilla_defrcn/split1/1shot_seed0 \
        --fma-dir checkpoints/voc/voc_novel_methods/fm_ablation/split1/dinov2_vitb14/1shot_seed0 \
        --dataset voc_2007_test_novel1 \
        --output-dir paper/figures/qualitative
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# VOC class names (all 20 classes)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Novel class splits for VOC
VOC_NOVEL_SPLITS = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

# Colors for visualization
TP_COLOR = (0, 200, 0)       # Green for true positives
FP_COLOR = (0, 0, 220)       # Red for false positives
GT_COLOR = (220, 180, 0)     # Cyan for ground truth
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_predictions(result_dir):
    """Load predictions from instances_predictions.pth or coco_instances_results.json."""
    pth_path = os.path.join(result_dir, "inference", "instances_predictions.pth")
    json_path = os.path.join(result_dir, "inference", "coco_instances_results.json")

    if os.path.exists(pth_path):
        preds = torch.load(pth_path, map_location="cpu")
        # Convert to per-image dict
        per_image = {}
        for item in preds:
            img_id = item["image_id"]
            instances = item.get("instances", [])
            per_image[img_id] = instances
        return per_image
    elif os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        # Group by image_id
        per_image = {}
        for det in results:
            img_id = det["image_id"]
            if img_id not in per_image:
                per_image[img_id] = []
            per_image[img_id].append(det)
        return per_image
    else:
        print(f"No predictions found in {result_dir}")
        print(f"  Checked: {pth_path}")
        print(f"  Checked: {json_path}")
        return {}


def load_gt_annotations(dataset_name):
    """Load ground truth annotations for a VOC dataset."""
    try:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        import defrcn.data  # noqa: F401 — trigger VOC registration

        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        return dataset_dicts, metadata
    except Exception as e:
        print(f"Warning: Could not load dataset '{dataset_name}': {e}")
        print("Ground truth overlay will be disabled.")
        return None, None


def compute_iou(box1, box2):
    """Compute IoU between two boxes in XYXY format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


def xywh_to_xyxy(box):
    """Convert XYWH box to XYXY."""
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def classify_detections(detections, gt_boxes, gt_classes, iou_thresh=0.5):
    """Classify detections as TP or FP based on IoU with ground truth."""
    matched_gt = set()
    results = []

    # Sort by score (descending)
    sorted_dets = sorted(detections, key=lambda d: d.get("score", 0), reverse=True)

    for det in sorted_dets:
        bbox = det["bbox"]
        if len(bbox) == 4 and bbox[2] < bbox[0] + 100:  # Likely XYWH
            bbox = xywh_to_xyxy(bbox)
        cat_id = det["category_id"]
        score = det["score"]

        is_tp = False
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_idx in matched_gt:
                continue
            if gt_cls != cat_id:
                continue

            iou = compute_iou(bbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            is_tp = True
            matched_gt.add(best_gt_idx)

        results.append({
            "bbox": bbox,
            "category_id": cat_id,
            "score": score,
            "is_tp": is_tp,
            "iou": best_iou,
        })

    return results


def draw_detections(img, detections, class_names, score_thresh=0.3, title="",
                    show_gt=False, gt_boxes=None, gt_classes=None):
    """Draw detections on an image with TP/FP color coding."""
    vis = img.copy()
    h, w = vis.shape[:2]

    # Draw GT boxes first (thin, dashed-like)
    if show_gt and gt_boxes is not None:
        for gt_box, gt_cls in zip(gt_boxes, gt_classes):
            x1, y1, x2, y2 = [int(v) for v in gt_box]
            # Draw dotted rectangle
            for i in range(x1, x2, 8):
                cv2.line(vis, (i, y1), (min(i + 4, x2), y1), GT_COLOR, 1)
                cv2.line(vis, (i, y2), (min(i + 4, x2), y2), GT_COLOR, 1)
            for i in range(y1, y2, 8):
                cv2.line(vis, (x1, i), (x1, min(i + 4, y2)), GT_COLOR, 1)
                cv2.line(vis, (x2, i), (x2, min(i + 4, y2)), GT_COLOR, 1)

    # Draw detections
    for det in detections:
        if det["score"] < score_thresh:
            continue

        bbox = det["bbox"]
        cat_id = det["category_id"]
        score = det["score"]
        is_tp = det.get("is_tp", True)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = TP_COLOR if is_tp else FP_COLOR
        thickness = 2

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        cls_name = class_names[cat_id] if cat_id < len(class_names) else f"cls{cat_id}"
        label = f"{cls_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, FONT, 0.45, 1)
        label_y = max(y1 - 4, label_size[1] + 4)

        # Label background
        cv2.rectangle(
            vis,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0] + 4, label_y + 2),
            color, -1,
        )
        cv2.putText(vis, label, (x1 + 2, label_y - 2), FONT, 0.45, (255, 255, 255), 1)

    # Title bar
    if title:
        bar_h = 28
        cv2.rectangle(vis, (0, 0), (w, bar_h), (40, 40, 40), -1)
        cv2.putText(vis, title, (8, 20), FONT, 0.55, (255, 255, 255), 1)

    return vis


def create_side_by_side(img_path, pcb_dets, fma_dets, gt_boxes, gt_classes,
                        class_names, score_thresh=0.3, split_id=1):
    """Create a side-by-side comparison image."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Classify detections
    pcb_classified = classify_detections(pcb_dets, gt_boxes, gt_classes)
    fma_classified = classify_detections(fma_dets, gt_boxes, gt_classes)

    # Count TPs
    pcb_tp = sum(1 for d in pcb_classified if d["is_tp"] and d["score"] >= score_thresh)
    fma_tp = sum(1 for d in fma_classified if d["is_tp"] and d["score"] >= score_thresh)
    pcb_total = sum(1 for d in pcb_classified if d["score"] >= score_thresh)
    fma_total = sum(1 for d in fma_classified if d["score"] >= score_thresh)

    # Draw both versions
    pcb_vis = draw_detections(
        img, pcb_classified, class_names, score_thresh,
        title=f"DeFRCN + PCB  ({pcb_tp}/{pcb_total} TP)",
        show_gt=True, gt_boxes=gt_boxes, gt_classes=gt_classes,
    )
    fma_vis = draw_detections(
        img, fma_classified, class_names, score_thresh,
        title=f"DeFRCN + PCB-FMA  ({fma_tp}/{fma_total} TP)",
        show_gt=True, gt_boxes=gt_boxes, gt_classes=gt_classes,
    )

    # Concatenate side by side
    combined = np.concatenate([pcb_vis, fma_vis], axis=1)

    return combined, pcb_tp, fma_tp, len(gt_boxes)


def find_interesting_images(pcb_preds, fma_preds, gt_data, class_names,
                            score_thresh=0.3, novel_class_ids=None):
    """Find images where PCB-FMA does significantly better than PCB."""
    interesting = []

    for gt_item in gt_data:
        img_id = gt_item["image_id"]
        file_name = gt_item["file_name"]

        # Get GT boxes for novel classes only
        gt_boxes = []
        gt_classes = []
        for ann in gt_item.get("annotations", []):
            cat_id = ann["category_id"]
            if novel_class_ids and cat_id not in novel_class_ids:
                continue
            bbox = ann["bbox"]
            if ann.get("bbox_mode", 0) == 1:  # XYWH_ABS
                bbox = xywh_to_xyxy(bbox)
            gt_boxes.append(bbox)
            gt_classes.append(cat_id)

        if not gt_boxes:
            continue

        # Get predictions
        pcb_dets = pcb_preds.get(img_id, [])
        fma_dets = fma_preds.get(img_id, [])

        # Filter to novel classes and threshold
        pcb_novel = [d for d in pcb_dets
                     if d["score"] >= score_thresh
                     and (novel_class_ids is None or d["category_id"] in novel_class_ids)]
        fma_novel = [d for d in fma_dets
                     if d["score"] >= score_thresh
                     and (novel_class_ids is None or d["category_id"] in novel_class_ids)]

        # Classify
        pcb_classified = classify_detections(pcb_novel, gt_boxes, gt_classes)
        fma_classified = classify_detections(fma_novel, gt_boxes, gt_classes)

        pcb_tp = sum(1 for d in pcb_classified if d["is_tp"])
        fma_tp = sum(1 for d in fma_classified if d["is_tp"])
        pcb_fp = sum(1 for d in pcb_classified if not d["is_tp"])
        fma_fp = sum(1 for d in fma_classified if not d["is_tp"])

        # Score: prefer images where FMA finds more TPs or has fewer FPs
        improvement = (fma_tp - pcb_tp) + 0.5 * (pcb_fp - fma_fp)

        if improvement > 0 or (fma_tp > 0 and pcb_tp == 0):
            interesting.append({
                "img_id": img_id,
                "file_name": file_name,
                "gt_boxes": gt_boxes,
                "gt_classes": gt_classes,
                "pcb_dets": pcb_novel,
                "fma_dets": fma_novel,
                "improvement": improvement,
                "pcb_tp": pcb_tp,
                "fma_tp": fma_tp,
                "n_gt": len(gt_boxes),
            })

    # Sort by improvement (descending)
    interesting.sort(key=lambda x: x["improvement"], reverse=True)
    return interesting


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative comparison: PCB vs PCB-FMA detections"
    )
    parser.add_argument("--pcb-dir", required=True,
                        help="Output directory of vanilla PCB evaluation")
    parser.add_argument("--fma-dir", required=True,
                        help="Output directory of PCB-FMA evaluation")
    parser.add_argument("--dataset", default="voc_2007_test_novel1",
                        help="Dataset name for GT annotations")
    parser.add_argument("--split", type=int, default=1,
                        help="VOC novel split (1, 2, or 3)")
    parser.add_argument("--output-dir", default="paper/figures/qualitative",
                        help="Directory to save visualization images")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to visualize")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="Score threshold for displaying detections")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for TP/FP classification")
    parser.add_argument("--novel-only", action="store_true", default=True,
                        help="Only show novel class detections")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions
    print(f"Loading PCB predictions from: {args.pcb_dir}")
    pcb_preds = load_predictions(args.pcb_dir)
    print(f"  Found predictions for {len(pcb_preds)} images")

    print(f"Loading PCB-FMA predictions from: {args.fma_dir}")
    fma_preds = load_predictions(args.fma_dir)
    print(f"  Found predictions for {len(fma_preds)} images")

    # Load GT
    print(f"Loading GT from dataset: {args.dataset}")
    gt_data, metadata = load_gt_annotations(args.dataset)

    if gt_data is None:
        print("Cannot proceed without ground truth annotations.")
        return

    class_names = VOC_CLASSES
    if metadata and hasattr(metadata, "thing_classes"):
        class_names = metadata.thing_classes

    # Get novel class IDs
    novel_class_ids = None
    if args.novel_only and args.split in VOC_NOVEL_SPLITS:
        novel_names = VOC_NOVEL_SPLITS[args.split]
        novel_class_ids = set()
        for name in novel_names:
            if name in class_names:
                novel_class_ids.add(class_names.index(name))
        print(f"Novel classes (split {args.split}): {novel_names}")
        print(f"Novel class IDs: {novel_class_ids}")

    # Find interesting images
    print("Finding images with significant PCB-FMA improvement...")
    interesting = find_interesting_images(
        pcb_preds, fma_preds, gt_data, class_names,
        score_thresh=args.score_thresh,
        novel_class_ids=novel_class_ids,
    )
    print(f"  Found {len(interesting)} images with improvement")

    # Generate visualizations
    n_vis = min(args.num_images, len(interesting))
    print(f"\nGenerating {n_vis} side-by-side comparison images...")

    for idx in range(n_vis):
        item = interesting[idx]
        img_path = item["file_name"]

        result = create_side_by_side(
            img_path,
            item["pcb_dets"],
            item["fma_dets"],
            item["gt_boxes"],
            item["gt_classes"],
            class_names,
            score_thresh=args.score_thresh,
            split_id=args.split,
        )
        if result is None:
            continue

        combined, pcb_tp, fma_tp, n_gt = result

        # Save
        out_name = f"comparison_{idx:02d}_pcb{pcb_tp}_fma{fma_tp}_gt{n_gt}.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [{idx+1}/{n_vis}] {os.path.basename(img_path)}: "
              f"PCB {pcb_tp}/{n_gt} TP, FMA {fma_tp}/{n_gt} TP -> {out_path}")

    # Generate a summary grid (2x3 or 2x4)
    if n_vis >= 4:
        print("\nGenerating summary grid...")
        grid_images = []
        for idx in range(min(6, n_vis)):
            item = interesting[idx]
            result = create_side_by_side(
                item["file_name"],
                item["pcb_dets"],
                item["fma_dets"],
                item["gt_boxes"],
                item["gt_classes"],
                class_names,
                score_thresh=args.score_thresh,
            )
            if result is not None:
                combined, _, _, _ = result
                # Resize for grid
                target_w = 800
                scale = target_w / combined.shape[1]
                resized = cv2.resize(combined, None, fx=scale, fy=scale)
                grid_images.append(resized)

        if len(grid_images) >= 2:
            # Make all same height
            max_h = max(img.shape[0] for img in grid_images)
            max_w = max(img.shape[1] for img in grid_images)
            padded = []
            for img in grid_images:
                pad_h = max_h - img.shape[0]
                pad_w = max_w - img.shape[1]
                padded_img = cv2.copyMakeBorder(
                    img, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
                padded.append(padded_img)

            # Arrange in rows of 2
            rows = []
            for i in range(0, len(padded), 1):
                rows.append(padded[i])

            grid = np.concatenate(rows, axis=0)

            grid_path = os.path.join(args.output_dir, "qualitative_grid.jpg")
            cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Grid saved to: {grid_path}")

    # Print summary
    print(f"\nDone! {n_vis} comparison images saved to: {args.output_dir}")
    print("\nColor coding:")
    print("  Green boxes = True Positives (correct detections)")
    print("  Red boxes   = False Positives (incorrect detections)")
    print("  Cyan dashed = Ground Truth boxes")


if __name__ == "__main__":
    main()
