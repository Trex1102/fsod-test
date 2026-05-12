#!/usr/bin/env python3
"""
False Positive Error Analysis for PCB-based Few-Shot Detectors.

Runs the model on the VOC test set, applies PCB calibration, then categorises
every prediction above a score threshold into one of four error types:

  TP           - correct class, IoU >= iou_thresh with an unmatched GT box
  Background   - max IoU with ANY GT box < 0.3  (pure background proposals)
  Localisation - max IoU with correct-class GT in [0.3, iou_thresh)
  Class-conf   - IoU >= iou_thresh with a GT box of a DIFFERENT class
  Duplicate    - correct class, IoU >= iou_thresh but GT already claimed

Output:
  - Global counts and percentages for each category (novel classes only)
  - Per-class AP50 breakdown (from the standard VOC evaluator)
  - Saves a JSON summary to the output dir

Usage:
    python tools/analyze_fp_types.py \\
        --config-file configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel1_5shot_seed0_pcb_fma_enhanced.yaml \\
        --weights checkpoints/voc/vanilla_defrcn/split1/5shot_seed0/model_final.pth \\
        --output-dir /tmp/fp_analysis_fma_enhanced_5shot \\
        --score-thresh 0.05 \\
        --iou-thresh 0.5
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import defrcn.data  # triggers register_all_voc() at module level

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.modeling import build_model

from defrcn.config import get_cfg as defrcn_get_cfg
from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
from defrcn.evaluation.evaluator import inference_on_dataset
from defrcn.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------

def iou_single(box, gt_boxes):
    """Compute IoU between one box [x1,y1,x2,y2] and each row of gt_boxes."""
    if len(gt_boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    areas_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_b + areas_g - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# -----------------------------------------------------------------------
# GT annotation loader
# -----------------------------------------------------------------------

def load_gt(anno_path, class_names):
    """Load GT boxes and class ids from a VOC XML annotation file."""
    if not os.path.exists(anno_path):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

    tree = ET.parse(anno_path)
    boxes, classes = [], []
    for obj in tree.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_names:
            continue
        cls_id = class_names.index(name)
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text) - 1
        y1 = float(bb.find("ymin").text) - 1
        x2 = float(bb.find("xmax").text) - 1
        y2 = float(bb.find("ymax").text) - 1
        boxes.append([x1, y1, x2, y2])
        classes.append(cls_id)

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)


# -----------------------------------------------------------------------
# Custom evaluator that captures per-prediction details
# -----------------------------------------------------------------------

class FPAnalysisEvaluator:
    """Wraps PascalVOCDetectionEvaluator and captures raw predictions for FP analysis."""

    def __init__(self, dataset_name, anno_template, class_names, novel_class_ids,
                 score_thresh, iou_thresh):
        self._voc_eval = PascalVOCDetectionEvaluator(dataset_name)
        self._anno_template = anno_template
        self._class_names = class_names
        self._novel_ids = set(novel_class_ids)
        self._score_thresh = score_thresh
        self._iou_thresh = iou_thresh
        self._cpu = torch.device("cpu")

        # Counters (novel classes only)
        self.counts = defaultdict(int)        # global
        self.per_class = defaultdict(lambda: defaultdict(int))  # cls -> error_type -> count

    def reset(self):
        self._voc_eval.reset()
        self.counts.clear()
        self.per_class.clear()

    def process(self, inputs, outputs):
        # Standard VOC evaluation bookkeeping
        self._voc_eval.process(inputs, outputs)

        # FP analysis
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]
            anno_path = self._anno_template.format(image_id)
            gt_boxes, gt_classes = load_gt(anno_path, self._class_names)

            instances = out["instances"].to(self._cpu)
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()

            # Keep only predictions above threshold AND in novel classes
            keep = (pred_scores >= self._score_thresh) & \
                   np.isin(pred_classes, list(self._novel_ids))
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_classes = pred_classes[keep]

            if len(pred_boxes) == 0:
                continue

            # Sort by score descending (greedy TP matching)
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_classes = pred_classes[order]

            # Only consider novel GT boxes for matching
            if len(gt_boxes) > 0:
                novel_gt_mask = np.isin(gt_classes, list(self._novel_ids))
                gt_boxes_novel = gt_boxes[novel_gt_mask]
                gt_classes_novel = gt_classes[novel_gt_mask]
            else:
                gt_boxes_novel = np.zeros((0, 4), dtype=np.float32)
                gt_classes_novel = np.zeros(0, dtype=np.int32)

            matched_gt = set()

            for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
                cls_name = self._class_names[cls]

                ious_all = iou_single(box, gt_boxes_novel)

                # Correct-class GT
                same_cls_mask = gt_classes_novel == cls
                ious_same = ious_all[same_cls_mask]
                max_iou_same = float(ious_same.max()) if len(ious_same) > 0 else 0.0

                # Any-class GT
                max_iou_any = float(ious_all.max()) if len(ious_all) > 0 else 0.0

                # Different-class GT
                diff_cls_mask = gt_classes_novel != cls
                ious_diff = ious_all[diff_cls_mask]
                max_iou_diff = float(ious_diff.max()) if len(ious_diff) > 0 else 0.0

                if max_iou_same >= self._iou_thresh:
                    gt_idx = int(np.where(same_cls_mask)[0][np.argmax(ious_same)])
                    if gt_idx not in matched_gt:
                        label = "TP"
                        matched_gt.add(gt_idx)
                    else:
                        label = "Duplicate"
                elif max_iou_diff >= self._iou_thresh:
                    label = "ClassConfusion"
                elif max_iou_any >= 0.3:
                    label = "Localisation"
                else:
                    label = "Background"

                self.counts[label] += 1
                self.per_class[cls_name][label] += 1

    def evaluate(self):
        return self._voc_eval.evaluate()

    def print_fp_summary(self):
        total = sum(self.counts.values())
        if total == 0:
            print("No predictions found above threshold.")
            return

        print("\n" + "=" * 60)
        print("FALSE POSITIVE ANALYSIS (novel classes only)")
        print("=" * 60)
        print(f"Total predictions: {total}")
        print(f"\n{'Type':<20} {'Count':>8} {'%':>8}")
        print("-" * 40)
        for label in ["TP", "Background", "Localisation", "ClassConfusion", "Duplicate"]:
            n = self.counts.get(label, 0)
            pct = 100.0 * n / total if total > 0 else 0
            print(f"{label:<20} {n:>8} {pct:>7.1f}%")

        print("\n--- Per-class FP breakdown ---")
        for cls_name in sorted(self.per_class.keys()):
            cd = self.per_class[cls_name]
            cls_total = sum(cd.values())
            bg = cd.get("Background", 0)
            cc = cd.get("ClassConfusion", 0)
            tp = cd.get("TP", 0)
            dup = cd.get("Duplicate", 0)
            loc = cd.get("Localisation", 0)
            print(f"  {cls_name:<15} total={cls_total:4d}  "
                  f"TP={tp:4d}  BG={bg:4d}  Loc={loc:4d}  "
                  f"ClsCon={cc:4d}  Dup={dup:4d}")
        print("=" * 60)

    def save_json(self, path):
        out = {
            "global": dict(self.counts),
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved FP analysis JSON to: {path}")


# -----------------------------------------------------------------------
# Config setup
# -----------------------------------------------------------------------

def setup_cfg(config_file, weights, pcb_modelpath=None):
    cfg = defrcn_get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    if pcb_modelpath:
        cfg.TEST.PCB_MODELPATH = pcb_modelpath
    cfg.freeze()
    return cfg


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FP type analysis for few-shot detectors")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="Min score to include a prediction in analysis")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for TP matching")
    parser.add_argument("--pcb-modelpath", default=None,
                        help="Path to PCB ResNet101 pretrain weights")
    args = parser.parse_args()

    cfg = setup_cfg(args.config_file, args.weights, args.pcb_modelpath)

    # Dataset metadata
    dataset_name = cfg.DATASETS.TEST[0]
    meta = MetadataCatalog.get(dataset_name)
    anno_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
    class_names = meta.thing_classes
    novel_class_ids = [i for i, n in enumerate(class_names) if n in meta.novel_classes]

    # Build model
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Data loader
    data_loader = build_detection_test_loader(cfg, dataset_name)

    # Build evaluator
    evaluator = FPAnalysisEvaluator(
        dataset_name=dataset_name,
        anno_template=anno_template,
        class_names=class_names,
        novel_class_ids=novel_class_ids,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    # PCB initialisation (same as inference_on_dataset)
    pcb = None
    if cfg.TEST.PCB_ENABLE:
        from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
        pcb = PrototypicalCalibrationBlock(cfg)

        if getattr(cfg.NOVEL_METHODS, "ENABLE", False):
            from defrcn.evaluation.novel_methods import build_novel_method_pcb
            method = cfg.NOVEL_METHODS.METHOD
            pcb = build_novel_method_pcb(pcb, cfg, method)

    # Run inference with manual calibration loop
    evaluator.reset()
    total = len(data_loader)
    for i, inputs in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(inputs)

        if pcb is not None:
            outputs = pcb.execute_calibration(inputs, outputs)

        evaluator.process(inputs, outputs)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] images processed...")

    # AP results
    results = evaluator.evaluate()
    if results:
        bbox = results.get("bbox", {})
        print(f"\nAP={bbox.get('AP',0):.2f}  "
              f"AP50={bbox.get('AP50',0):.2f}  "
              f"AP75={bbox.get('AP75',0):.2f}  "
              f"nAP50={bbox.get('nAP50',0):.2f}")

    # FP analysis
    evaluator.print_fp_summary()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluator.save_json(os.path.join(args.output_dir, "fp_analysis.json"))


if __name__ == "__main__":
    main()
