#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

import torch
from detectron2.structures import pairwise_iou
from detectron2.utils.logger import setup_logger

from defrcn.config import get_cfg
from defrcn.modeling import build_model
from defrcn.checkpoint import DetectionCheckpointer
from defrcn.dataloader import build_detection_test_loader, MetadataCatalog
from defrcn.modeling.meta_arch.gdl import decouple_layer
import defrcn.data  # noqa: F401 - registers datasets


def parse_csv_floats(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantify RPN proposal bottleneck gap for vanilla DeFRCN."
    )
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dataset", default="", help="Override cfg.DATASETS.TEST[0]")
    parser.add_argument("--weights", default="", help="Override MODEL.WEIGHTS")
    parser.add_argument("--device", default="", help="Override MODEL.DEVICE (e.g., cpu)")
    parser.add_argument("--topk", default="100,300,1000", help="Comma-separated top-k proposal list")
    parser.add_argument("--ious", default="0.5,0.75", help="Comma-separated IoU thresholds")
    parser.add_argument("--max-images", type=int, default=-1, help="Limit images for quick analysis")
    parser.add_argument(
        "--ap50",
        type=float,
        default=None,
        help="Optional observed AP50 to compare against proposal ceiling",
    )
    parser.add_argument("--output-json", default="", help="Optional path to save raw results json")
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.dataset:
        cfg.DATASETS.TEST = (args.dataset,)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    if args.device:
        cfg.MODEL.DEVICE = args.device
    cfg.freeze()
    return cfg


def _rpn_features_for_model(model, features):
    features_rpn = features
    if getattr(model, "dual_fusion", None) is not None:
        fused_rpn, _ = model.dual_fusion(features)
        features_rpn = dict(features)
        features_rpn[model.fusion_out_feature] = fused_rpn

    features_de_rpn = features_rpn
    if model.cfg.MODEL.RPN.ENABLE_DECOUPLE:
        scale = model.cfg.MODEL.RPN.BACKWARD_SCALE
        features_de_rpn = {}
        for k, v in features_rpn.items():
            if k in model.rpn_in_features:
                x = decouple_layer(v, scale)
                if k == model.rpn_affine_feature:
                    x = model.affine_rpn(x)
                features_de_rpn[k] = x
            else:
                features_de_rpn[k] = v
    return features_de_rpn


def area_bucket(area):
    # COCO-style area ranges
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


@torch.no_grad()
def main():
    args = parse_args()
    logger = setup_logger(name="proposal_gap")
    cfg = setup_cfg(args)

    if len(cfg.DATASETS.TEST) == 0:
        raise ValueError("cfg.DATASETS.TEST is empty.")
    dataset_name = cfg.DATASETS.TEST[0]
    topk_list = parse_csv_ints(args.topk)
    iou_list = parse_csv_floats(args.ious)
    topk_list = sorted(set(topk_list))
    iou_list = sorted(set(iou_list))

    logger.info("Dataset: %s", dataset_name)
    logger.info("Top-K: %s", topk_list)
    logger.info("IoUs: %s", iou_list)

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    loader = build_detection_test_loader(cfg, dataset_name)

    # Aggregate stats
    total_gt = 0
    total_images = 0
    avg_props = 0.0
    missing_instances_field = 0
    recall_hits = {(k, t): 0 for k in topk_list for t in iou_list}

    size_total = defaultdict(int)
    size_hits = defaultdict(int)  # key: (size_name, k, t)

    class_total = defaultdict(int)
    class_hits = defaultdict(int)  # key: (class_id, k, t)

    for batch in loader:
        if args.max_images > 0 and total_images >= args.max_images:
            break

        images = model.preprocess_image(batch)
        features = model.backbone(images.tensor)
        features_de_rpn = _rpn_features_for_model(model, features)
        proposals, _ = model.proposal_generator(images, features_de_rpn, gt_instances=None)

        for sample, props in zip(batch, proposals):
            total_images += 1
            num_props = len(props.proposal_boxes)
            avg_props += float(num_props)

            if "instances" not in sample:
                missing_instances_field += 1
                continue

            gt_inst = sample["instances"].to(model.device)
            if len(gt_inst) == 0:
                continue

            gt_boxes = gt_inst.gt_boxes
            gt_classes = gt_inst.gt_classes.tolist()
            gt_areas = gt_boxes.area().tolist()
            total_gt += len(gt_boxes)

            for cid, area in zip(gt_classes, gt_areas):
                class_total[int(cid)] += 1
                size_total[area_bucket(float(area))] += 1

            if num_props == 0:
                continue

            ious = pairwise_iou(gt_boxes, props.proposal_boxes)  # [G, P]
            for k in topk_list:
                k_eff = min(k, num_props)
                if k_eff <= 0:
                    max_iou = torch.zeros((len(gt_boxes),), device=ious.device)
                else:
                    max_iou = ious[:, :k_eff].max(dim=1).values

                for t in iou_list:
                    hit_mask = max_iou >= float(t)
                    num_hit = int(hit_mask.sum().item())
                    recall_hits[(k, t)] += num_hit

                    for idx, hit in enumerate(hit_mask.tolist()):
                        cid = int(gt_classes[idx])
                        size_name = area_bucket(float(gt_areas[idx]))
                        if hit:
                            class_hits[(cid, k, t)] += 1
                            size_hits[(size_name, k, t)] += 1

        if args.max_images > 0 and total_images >= args.max_images:
            break

    if total_images == 0:
        raise RuntimeError("No images were processed.")

    avg_props /= float(total_images)
    logger.info("Processed images: %d", total_images)
    logger.info("Average proposals/image: %.2f", avg_props)
    if missing_instances_field > 0:
        logger.warning(
            "Missing 'instances' in %d images from test loader. "
            "Recall stats may be underestimated.",
            missing_instances_field,
        )

    print("\n=== Proposal Recall / Bottleneck ===")
    print("Total GT boxes:", total_gt)
    print("Avg proposals/image:", round(avg_props, 2))
    print("")
    print("topK\tIoU\tRecall\tMissGap(1-Recall)\tAPCeiling")
    summary_rows = []
    for k in topk_list:
        for t in iou_list:
            recall = (recall_hits[(k, t)] / max(total_gt, 1.0))
            miss_gap = 1.0 - recall
            ap_ceiling = 100.0 * recall
            row = {
                "topk": k,
                "iou": t,
                "recall": recall,
                "miss_gap": miss_gap,
                "ap_ceiling": ap_ceiling,
            }
            if args.ap50 is not None and abs(t - 0.5) < 1e-9:
                row["headroom_vs_ap50"] = ap_ceiling - args.ap50
            summary_rows.append(row)
            print(
                f"{k}\t{t:.2f}\t{recall:.4f}\t{miss_gap:.4f}\t\t{ap_ceiling:.2f}"
                + (
                    f"\t(headroom vs AP50={args.ap50:.2f}: {ap_ceiling - args.ap50:.2f})"
                    if args.ap50 is not None and abs(t - 0.5) < 1e-9
                    else ""
                )
            )

    # Per-size report
    print("\n=== Size-wise Recall ===")
    print("size\ttopK\tIoU\tRecall")
    size_rows = []
    for size_name in ["small", "medium", "large"]:
        denom = size_total.get(size_name, 0)
        if denom == 0:
            continue
        for k in topk_list:
            for t in iou_list:
                rec = size_hits.get((size_name, k, t), 0) / float(denom)
                size_rows.append(
                    {"size": size_name, "topk": k, "iou": t, "recall": rec, "count": denom}
                )
                print(f"{size_name}\t{k}\t{t:.2f}\t{rec:.4f}")

    # Per-class report (best effort)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = getattr(metadata, "thing_classes", None)
    class_iou = 0.5 if 0.5 in iou_list else iou_list[0]
    print(f"\n=== Per-class Recall @ max(topK), IoU={class_iou:.2f} ===")
    print("class_id\tclass_name\trecall\tcount")
    class_rows = []
    max_k = max(topk_list)
    for cid, denom in sorted(class_total.items()):
        if denom <= 0:
            continue
        rec = class_hits.get((cid, max_k, class_iou), 0) / float(denom)
        cname = class_names[cid] if class_names is not None and cid < len(class_names) else str(cid)
        class_rows.append(
            {"class_id": cid, "class_name": cname, "recall": rec, "count": int(denom)}
        )
        print(f"{cid}\t{cname}\t{rec:.4f}\t{int(denom)}")

    output = {
        "dataset": dataset_name,
        "total_images": total_images,
        "total_gt": total_gt,
        "avg_proposals_per_image": avg_props,
        "summary": summary_rows,
        "size_rows": size_rows,
        "class_rows": class_rows,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved json: {args.output_json}")


if __name__ == "__main__":
    main()
