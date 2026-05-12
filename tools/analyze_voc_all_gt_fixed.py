#!/usr/bin/env python3
"""Corrected all-GT VOC FP analysis for pcb_fma_enhanced and residual veto."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import defrcn.data  # noqa: F401

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model

from defrcn.config import get_cfg as defrcn_get_cfg
from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
from defrcn.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from defrcn.evaluation.novel_methods.pcb_fma_enhanced import PCBFMAEnhanced

from rrv import ResidualVetoCalibrator, load_rrv_params
from rrv.common import (
    build_local_to_global_map,
    classify_prediction,
    get_voc_split_meta,
    infer_voc_split_id,
    load_voc_gt,
)


class FixedAllGTEvaluator:
    def __init__(self, dataset_name: str, score_thresh: float, iou_thresh: float):
        self._voc_eval = PascalVOCDetectionEvaluator(dataset_name)
        meta = MetadataCatalog.get(dataset_name)
        self._local_class_names = list(meta.thing_classes)
        self._split_id = infer_voc_split_id(dataset_name)
        split_meta = get_voc_split_meta(self._split_id)
        self._all_class_names = split_meta["all"]
        self._novel_global_ids = {self._all_class_names.index(name) for name in split_meta["novel"]}
        self._local_to_global = build_local_to_global_map(self._local_class_names, self._all_class_names)
        self._anno_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._score_thresh = score_thresh
        self._iou_thresh = iou_thresh
        self.counts = defaultdict(int)
        self.per_class = defaultdict(lambda: defaultdict(int))
        self._cpu = torch.device("cpu")

    def reset(self):
        self._voc_eval.reset()
        self.counts.clear()
        self.per_class.clear()

    def process(self, inputs, outputs):
        self._voc_eval.process(inputs, outputs)
        for inp, out in zip(inputs, outputs):
            gt_boxes, gt_classes = load_voc_gt(self._anno_template.format(inp["image_id"]), self._all_class_names)
            instances = out["instances"].to(self._cpu)
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()

            keep = pred_scores >= self._score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_classes = pred_classes[keep]
            if len(pred_boxes) == 0:
                continue

            order = pred_scores.argsort()[::-1]
            pred_boxes = pred_boxes[order]
            pred_classes = pred_classes[order]
            matched_gt = set()

            for pred_box, pred_local_cls in zip(pred_boxes, pred_classes):
                pred_class_name = self._local_class_names[int(pred_local_cls)]
                pred_global_cls = self._local_to_global[int(pred_local_cls)]
                label = classify_prediction(
                    pred_box=pred_box,
                    pred_global_cls=pred_global_cls,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    novel_global_ids=self._novel_global_ids,
                    matched_gt=matched_gt,
                    iou_thresh=self._iou_thresh,
                )
                self.counts[label] += 1
                self.per_class[pred_class_name][label] += 1

    def evaluate(self):
        return self._voc_eval.evaluate()

    def save_json(self, path: str, extra=None):
        payload = {
            "global": dict(self.counts),
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
        }
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)


def setup_cfg(config_file: str, weights: str, pcb_modelpath: str):
    cfg = defrcn_get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.TEST.PCB_MODELPATH = pcb_modelpath
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Corrected all-GT VOC FP analysis")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pcb-modelpath", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["baseline_enhanced", "rrv"], required=True)
    parser.add_argument("--params-file", default=None)
    parser.add_argument("--veto-model", default=None)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None, help="Benchmark-only cap")
    args = parser.parse_args()

    wall_start = time.time()
    cfg = setup_cfg(args.config_file, args.weights, args.pcb_modelpath)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    base_pcb = PrototypicalCalibrationBlock(cfg)
    if args.mode == "baseline_enhanced":
        calibrator = PCBFMAEnhanced(base_pcb, cfg)
    else:
        if not args.params_file or not args.veto_model:
            raise ValueError("--params-file and --veto-model are required in rrv mode")
        calibrator = ResidualVetoCalibrator(base_pcb, cfg, load_rrv_params(args.params_file), args.veto_model)

    evaluator = FixedAllGTEvaluator(cfg.DATASETS.TEST[0], args.score_thresh, args.iou_thresh)
    evaluator.reset()

    loop_start = time.time()
    processed = 0
    for idx, inputs in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(inputs)
        outputs = calibrator.execute_calibration(inputs, outputs)
        evaluator.process(inputs, outputs)
        processed += len(inputs)
        if args.max_images is not None and processed >= args.max_images:
            break
        if (idx + 1) % 500 == 0:
            print(f"[{idx + 1}/{len(data_loader)}] analysis images processed...")

    results = evaluator.evaluate()
    loop_elapsed = time.time() - loop_start
    wall_elapsed = time.time() - wall_start
    os.makedirs(args.output_dir, exist_ok=True)
    evaluator.save_json(
        os.path.join(args.output_dir, "fp_analysis_all_gt.json"),
        extra={
            "results": results,
            "processed_images": processed,
            "elapsed_sec": loop_elapsed,
            "wall_elapsed_sec": wall_elapsed,
            "mode": args.mode,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
