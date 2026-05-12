#!/usr/bin/env python3
"""Assignment-based oracle AP-impact analysis on corrected all-GT baseline outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import defrcn.data  # noqa: F401

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model

from defrcn.config import get_cfg as defrcn_get_cfg
from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
from defrcn.evaluation.novel_methods.pcb_fma_enhanced import PCBFMAEnhanced
from defrcn.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator

from ccr.common import (
    build_local_to_global_map,
    get_voc_split_meta,
    infer_voc_split_id,
    load_voc_gt,
    match_prediction_details,
)


SCENARIOS = (
    ("baseline", None, None),
    ("remove_baseconf_all", "BaseConfusion", None),
    ("remove_baseconf_ge_0p10", "BaseConfusion", 0.10),
    ("remove_baseconf_ge_0p30", "BaseConfusion", 0.30),
    ("remove_background_all", "Background", None),
    ("remove_localisation_all", "Localisation", None),
    ("remove_duplicate_all", "Duplicate", None),
    ("remove_novelconf_all", "NovelConfusion", None),
    ("remove_non_tp_all", "NON_TP", None),
)


def load_cfg(config_file: str, weights: str, pcb_modelpath: str):
    cfg = defrcn_get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.TEST.PCB_MODELPATH = pcb_modelpath
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Oracle AP-impact analysis")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pcb-modelpath", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    wall_start = time.time()
    cfg = load_cfg(args.config_file, args.weights, args.pcb_modelpath)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    calibrator = PCBFMAEnhanced(PrototypicalCalibrationBlock(cfg), cfg)

    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    local_class_names = list(meta.thing_classes)
    split_id = infer_voc_split_id(cfg.DATASETS.TEST[0])
    split_meta = get_voc_split_meta(split_id)
    all_class_names = split_meta["all"]
    local_to_global = build_local_to_global_map(local_class_names, all_class_names)
    novel_global_ids = {all_class_names.index(name) for name in split_meta["novel"]}
    anno_template = os.path.join(meta.dirname, "Annotations", "{}.xml")

    evaluators = {
        name: PascalVOCDetectionEvaluator(cfg.DATASETS.TEST[0]) for name, _, _ in SCENARIOS
    }
    for evaluator in evaluators.values():
        evaluator.reset()

    removed_counts = {name: defaultdict(int) for name, _, _ in SCENARIOS if name != "baseline"}
    processed = 0
    cpu = torch.device("cpu")
    loop_start = time.time()

    for batch_idx, inputs in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(inputs)
        outputs = calibrator.execute_calibration(inputs, outputs)

        for inp, out in zip(inputs, outputs):
            gt_boxes, gt_classes = load_voc_gt(anno_template.format(inp["image_id"]), all_class_names)
            instances_cpu = out["instances"].to(cpu)
            pred_boxes = instances_cpu.pred_boxes.tensor.numpy()
            pred_scores = instances_cpu.scores.numpy()
            pred_classes = instances_cpu.pred_classes.numpy()

            keep_mask = pred_scores >= float(args.score_thresh)
            keep_indices = np.where(keep_mask)[0]
            keep_scores = pred_scores[keep_mask]
            keep_classes = pred_classes[keep_mask]
            keep_boxes = pred_boxes[keep_mask]
            order = keep_scores.argsort()[::-1]
            sorted_indices = keep_indices[order]
            matched_gt = set()
            labels_by_index = {}

            for orig_idx, pred_box, pred_local_cls, pred_score in zip(
                sorted_indices,
                keep_boxes[order],
                keep_classes[order],
                keep_scores[order],
            ):
                pred_global_cls = local_to_global[int(pred_local_cls)]
                details = match_prediction_details(
                    pred_box=pred_box,
                    pred_global_cls=pred_global_cls,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                    novel_global_ids=novel_global_ids,
                    matched_gt=matched_gt,
                    iou_thresh=float(args.iou_thresh),
                )
                labels_by_index[int(orig_idx)] = {
                    "label": str(details["label"]),
                    "score": float(pred_score),
                }

            evaluators["baseline"].process([inp], [{"instances": out["instances"]}])
            for name, label_name, score_floor in SCENARIOS[1:]:
                keep = torch.ones(len(out["instances"]), dtype=torch.bool, device=out["instances"].scores.device)
                for idx, info in labels_by_index.items():
                    label = info["label"]
                    if label_name == "NON_TP":
                        should_remove = label != "TP"
                    else:
                        should_remove = label == label_name
                    if should_remove and (score_floor is None or float(info["score"]) >= float(score_floor)):
                        keep[idx] = False
                        removed_counts[name][label] += 1
                evaluators[name].process([inp], [{"instances": out["instances"][keep]}])

            processed += 1
            if args.max_images is not None and processed >= args.max_images:
                break

        if (batch_idx + 1) % 500 == 0:
            print(f"[{batch_idx + 1}/{len(data_loader)}] oracle images processed...")
        if args.max_images is not None and processed >= args.max_images:
            break

    scenario_results = {}
    baseline_bbox = evaluators["baseline"].evaluate().get("bbox", {})
    baseline_nap50 = float(baseline_bbox.get("nAP50", 0.0))
    scenario_results["baseline"] = {
        "bbox": baseline_bbox,
        "delta_nAP50": 0.0,
        "removed_counts": {},
    }
    for name, _, _ in SCENARIOS[1:]:
        bbox = evaluators[name].evaluate().get("bbox", {})
        scenario_results[name] = {
            "bbox": bbox,
            "delta_nAP50": float(bbox.get("nAP50", 0.0)) - baseline_nap50,
            "removed_counts": dict(removed_counts[name]),
        }

    payload = {
        "dataset": cfg.DATASETS.TEST[0],
        "processed_images": int(processed),
        "elapsed_sec": float(time.time() - loop_start),
        "wall_elapsed_sec": float(time.time() - wall_start),
        "score_thresh": float(args.score_thresh),
        "iou_thresh": float(args.iou_thresh),
        "scenarios": scenario_results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
