#!/usr/bin/env python3
"""All-GT false-positive analysis for isolated BACC runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from xml.etree import ElementTree as ET

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import defrcn.data  # noqa: F401

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model

from bacc import BackgroundAwareCompetitiveCalibrator, load_bacc_params
from defrcn.config import get_cfg as defrcn_get_cfg
from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
from defrcn.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator


def iou_single(box, gt_boxes):
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


def load_gt(anno_path, class_names):
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


class AllGTFPAnalysisEvaluator:
    def __init__(self, dataset_name, anno_template, class_names, novel_class_ids, score_thresh, iou_thresh):
        self._voc_eval = PascalVOCDetectionEvaluator(dataset_name)
        self._anno_template = anno_template
        self._class_names = class_names
        self._novel_ids = set(novel_class_ids)
        self._score_thresh = score_thresh
        self._iou_thresh = iou_thresh
        self._cpu = torch.device("cpu")
        self.counts = defaultdict(int)
        self.per_class = defaultdict(lambda: defaultdict(int))

    def reset(self):
        self._voc_eval.reset()
        self.counts.clear()
        self.per_class.clear()

    def process(self, inputs, outputs):
        self._voc_eval.process(inputs, outputs)
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]
            anno_path = self._anno_template.format(image_id)
            gt_boxes, gt_classes = load_gt(anno_path, self._class_names)

            instances = out["instances"].to(self._cpu)
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()

            keep = (pred_scores >= self._score_thresh) & np.isin(pred_classes, list(self._novel_ids))
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_classes = pred_classes[keep]
            if len(pred_boxes) == 0:
                continue

            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_classes = pred_classes[order]

            novel_gt_mask = np.isin(gt_classes, list(self._novel_ids))
            matched_gt = set()

            for box, cls in zip(pred_boxes, pred_classes):
                cls_name = self._class_names[cls]
                ious_all = iou_single(box, gt_boxes)
                max_iou_any = float(ious_all.max()) if len(ious_all) > 0 else 0.0

                same_cls_mask = gt_classes == cls
                ious_same = ious_all[same_cls_mask]
                max_iou_same = float(ious_same.max()) if len(ious_same) > 0 else 0.0

                if max_iou_same >= self._iou_thresh:
                    gt_idx = int(np.where(same_cls_mask)[0][np.argmax(ious_same)])
                    if gt_idx not in matched_gt:
                        label = "TP"
                        matched_gt.add(gt_idx)
                    else:
                        label = "Duplicate"
                else:
                    diff_mask = gt_classes != cls
                    diff_ious = ious_all[diff_mask]
                    if len(diff_ious) > 0 and float(diff_ious.max()) >= self._iou_thresh:
                        diff_idx = np.where(diff_mask)[0][np.argmax(diff_ious)]
                        gt_cls = int(gt_classes[diff_idx])
                        label = "NovelConfusion" if gt_cls in self._novel_ids else "BaseConfusion"
                    elif max_iou_any >= 0.3:
                        label = "Localisation"
                    else:
                        label = "Background"

                self.counts[label] += 1
                self.per_class[cls_name][label] += 1

    def evaluate(self):
        return self._voc_eval.evaluate()

    def save_json(self, path, extra=None):
        payload = {
            "global": dict(self.counts),
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
        }
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)


def setup_cfg(config_file: str, weights: str, pcb_modelpath: str | None):
    cfg = defrcn_get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    if pcb_modelpath:
        cfg.TEST.PCB_MODELPATH = pcb_modelpath
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="All-GT FP analysis for BACC")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--pcb-modelpath", default=None)
    parser.add_argument("--max-images", type=int, default=None, help="Benchmark-only cap")
    args = parser.parse_args()

    wall_start = time.time()
    cfg = setup_cfg(args.config_file, args.weights, args.pcb_modelpath)
    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    anno_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
    class_names = meta.thing_classes
    novel_ids = [i for i, n in enumerate(class_names) if n in meta.novel_classes]

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    base_pcb = PrototypicalCalibrationBlock(cfg)
    bacc = BackgroundAwareCompetitiveCalibrator(base_pcb, cfg, load_bacc_params(args.params_file))

    evaluator = AllGTFPAnalysisEvaluator(
        cfg.DATASETS.TEST[0],
        anno_template,
        class_names,
        novel_ids,
        args.score_thresh,
        args.iou_thresh,
    )
    evaluator.reset()

    total = len(data_loader)
    start = time.time()
    processed = 0
    for idx, inputs in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(inputs)
        outputs = bacc.execute_calibration(inputs, outputs)
        evaluator.process(inputs, outputs)
        processed += len(inputs)
        if args.max_images is not None and processed >= args.max_images:
            break
        if (idx + 1) % 500 == 0:
            print(f"[{idx + 1}/{total}] images processed...")

    results = evaluator.evaluate()
    elapsed = time.time() - start
    wall_elapsed = time.time() - wall_start
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "bacc_allgt_eval.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    evaluator.save_json(
        os.path.join(args.output_dir, "fp_analysis_all_gt.json"),
        extra={
            "results": results,
            "elapsed_sec": elapsed,
            "wall_elapsed_sec": wall_elapsed,
            "processed_images": processed,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
