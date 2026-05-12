#!/usr/bin/env python3
"""Proposal-level open-world oracle analysis for OWCH."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import defrcn.data  # noqa: F401

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader

from defrcn.config import get_cfg as defrcn_get_cfg
from defrcn.evaluation.calibration_layer import PrototypicalCalibrationBlock
from defrcn.evaluation.novel_methods.pcb_fma_enhanced import PCBFMAEnhanced
from defrcn.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from defrcn.modeling import build_model

from owch import load_owch_params
from owch.common import get_voc_split_meta, infer_voc_split_id, load_voc_gt
from owch.proposals import (
    class_boxes_from_pred_boxes,
    extract_batch_proposal_outputs,
    novel_instances_from_joint_probs,
    postprocess_instances,
)
from rrv.common import iou_single


def load_cfg(config_file: str, weights: str, pcb_modelpath: str):
    cfg = defrcn_get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.TEST.PCB_MODELPATH = pcb_modelpath
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="OWCH proposal-level oracle analysis")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pcb-modelpath", required=True)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    params = load_owch_params(args.params_file)
    cfg = load_cfg(args.config_file, args.weights, args.pcb_modelpath)
    detector_model = build_model(cfg)
    detector_model.eval()
    DetectionCheckpointer(detector_model).load(cfg.MODEL.WEIGHTS)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    enhanced = PCBFMAEnhanced(PrototypicalCalibrationBlock(cfg), cfg)

    baseline_eval = PascalVOCDetectionEvaluator(cfg.DATASETS.TEST[0])
    oracle_eval = PascalVOCDetectionEvaluator(cfg.DATASETS.TEST[0])
    baseline_eval.reset()
    oracle_eval.reset()

    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    split_id = infer_voc_split_id(cfg.DATASETS.TEST[0])
    split_meta = get_voc_split_meta(split_id)
    all_class_names = list(split_meta["all"])
    novel_class_names = list(meta.thing_classes)
    anno_template = os.path.join(meta.dirname, "Annotations", "{}.xml")

    processed = 0
    loop_start = time.time()
    wall_start = time.time()

    for batch_idx, inputs in enumerate(data_loader):
        with torch.no_grad():
            baseline_outputs = detector_model(inputs)
        baseline_outputs = enhanced.execute_calibration(inputs, baseline_outputs)
        baseline_eval.process(inputs, baseline_outputs)

        proposal_batch = extract_batch_proposal_outputs(
            detector_model,
            inputs,
            proposals=None,
            max_props_per_image=int(params.max_props_per_image),
        )
        oracle_joint_probs = []
        for img_idx, inp in enumerate(inputs):
            gt_boxes, gt_classes = load_voc_gt(anno_template.format(inp["image_id"]), all_class_names)
            pred_boxes_img = proposal_batch.pred_boxes[img_idx].detach().cpu()
            probs = torch.zeros(
                (pred_boxes_img.shape[0], len(novel_class_names) + len(split_meta["base"]) + 1),
                dtype=torch.float32,
                device=detector_model.device,
            )
            probs[:, -1] = 1.0
            if pred_boxes_img.shape[0] > 0 and len(gt_boxes) > 0:
                for novel_idx, novel_name in enumerate(novel_class_names):
                    gt_global = all_class_names.index(novel_name)
                    gt_mask = gt_classes == gt_global
                    if int(gt_mask.sum()) == 0:
                        continue
                    boxes_k = class_boxes_from_pred_boxes(
                        pred_boxes_img,
                        class_index=novel_idx,
                        num_classes=len(novel_class_names),
                    ).numpy()
                    gt_boxes_k = gt_boxes[gt_mask]
                    for prop_idx in range(boxes_k.shape[0]):
                        max_iou = float(iou_single(boxes_k[prop_idx], gt_boxes_k).max()) if len(gt_boxes_k) > 0 else 0.0
                        if max_iou >= float(params.iou_thresh):
                            probs[prop_idx, :] = 0.0
                            probs[prop_idx, novel_idx] = 1.0
            oracle_joint_probs.append(probs)

        oracle_instances = novel_instances_from_joint_probs(
            proposal_batch=proposal_batch,
            joint_probs=oracle_joint_probs,
            num_novel_classes=len(novel_class_names),
            score_thresh=float(params.final_score_thresh),
            nms_thresh=float(params.final_nms_thresh),
            topk_per_image=int(params.final_topk_per_image),
        )
        oracle_outputs = postprocess_instances(oracle_instances, inputs)
        oracle_eval.process(inputs, oracle_outputs)

        processed += len(inputs)
        if args.max_images is not None and processed >= args.max_images:
            break
        if (batch_idx + 1) % 500 == 0:
            print(f"[{batch_idx + 1}/{len(data_loader)}] oracle images processed...")

    baseline_bbox = baseline_eval.evaluate().get("bbox", {})
    oracle_bbox = oracle_eval.evaluate().get("bbox", {})
    payload = {
        "dataset": cfg.DATASETS.TEST[0],
        "processed_images": int(processed),
        "elapsed_sec": float(time.time() - loop_start),
        "wall_elapsed_sec": float(time.time() - wall_start),
        "scenarios": {
            "baseline_enhanced": {"bbox": baseline_bbox},
            "proposal_oracle": {
                "bbox": oracle_bbox,
                "delta_nAP50": float(oracle_bbox.get("nAP50", 0.0)) - float(baseline_bbox.get("nAP50", 0.0)),
            },
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
