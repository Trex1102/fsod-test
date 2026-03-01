#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass

import torch
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.logger import setup_logger
from torch.utils.data import DataLoader, TensorDataset

from defrcn.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg
from defrcn.dataloader import MetadataCatalog, build_detection_test_loader
from defrcn.modeling import build_model
from defrcn.modeling.vae_fsod import (
    DEFAULT_QUALITY_KEYS,
    NormConditionalVAE,
    QualityConditionalVAE,
    build_text_semantic_embeddings,
    compute_quality_hardness,
    paper_default_norm_range,
    quality_consistency_loss,
)
from defrcn.modeling.vae_fsod.norm_vae import vae_loss
import defrcn.data  # noqa: F401


@dataclass
class SamplePack:
    features: torch.Tensor
    labels: torch.Tensor
    ious: torch.Tensor
    qualities: torch.Tensor
    quality_keys: tuple


def parse_args():
    parser = argparse.ArgumentParser("Train Norm-VAE / Quality-VAE for VAE-FSOD")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", default="", help="Override MODEL.WEIGHTS")
    parser.add_argument("--dataset", default="", help="Override cfg.DATASETS.TRAIN[0]")
    parser.add_argument("--output", required=True, help="Path to save VAE checkpoint (.pth)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    if args.dataset:
        cfg.DATASETS.TRAIN = (args.dataset,)
    cfg.freeze()
    return cfg


def _augment_box_xyxy(box, image_hw, scale_max):
    h, w = image_hw
    x1, y1, x2, y2 = box.tolist()
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    scale = 1.0 + random.uniform(0.0, float(scale_max))
    nw = bw * scale
    nh = bh * scale

    nx1 = max(0.0, cx - 0.5 * nw)
    ny1 = max(0.0, cy - 0.5 * nh)
    nx2 = min(float(w - 1), cx + 0.5 * nw)
    ny2 = min(float(h - 1), cy + 0.5 * nh)
    if nx2 <= nx1:
        nx2 = min(float(w - 1), nx1 + 1.0)
    if ny2 <= ny1:
        ny2 = min(float(h - 1), ny1 + 1.0)
    return torch.tensor([nx1, ny1, nx2, ny2], dtype=torch.float32)


def _compute_quality_vector(roi_box, gt_box, other_gt_boxes):
    eps = 1e-6

    ix1 = torch.maximum(roi_box[0], gt_box[0])
    iy1 = torch.maximum(roi_box[1], gt_box[1])
    ix2 = torch.minimum(roi_box[2], gt_box[2])
    iy2 = torch.minimum(roi_box[3], gt_box[3])
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    roi_w = (roi_box[2] - roi_box[0]).clamp(min=0.0)
    roi_h = (roi_box[3] - roi_box[1]).clamp(min=0.0)
    roi_area = roi_w * roi_h

    gt_w = (gt_box[2] - gt_box[0]).clamp(min=0.0)
    gt_h = (gt_box[3] - gt_box[1]).clamp(min=0.0)
    gt_area = gt_w * gt_h

    iou = pairwise_iou(Boxes(roi_box.view(1, 4)), Boxes(gt_box.view(1, 4)))[0, 0]
    fg_ratio = inter / (roi_area + eps)
    gt_coverage = inter / (gt_area + eps)

    roi_cx = 0.5 * (roi_box[0] + roi_box[2])
    roi_cy = 0.5 * (roi_box[1] + roi_box[3])
    gt_cx = 0.5 * (gt_box[0] + gt_box[2])
    gt_cy = 0.5 * (gt_box[1] + gt_box[3])
    center_dist = torch.sqrt((roi_cx - gt_cx).pow(2) + (roi_cy - gt_cy).pow(2))
    gt_diag = torch.sqrt(gt_w.pow(2) + gt_h.pow(2)).clamp(min=eps)
    center_offset = torch.clamp(center_dist / gt_diag, min=0.0, max=1.0)

    if other_gt_boxes.numel() > 0:
        crowding = pairwise_iou(Boxes(roi_box.view(1, 4)), Boxes(other_gt_boxes)).max()
    else:
        crowding = roi_box.new_tensor(0.0)

    q = torch.stack([iou, fg_ratio, gt_coverage, center_offset, crowding], dim=0)
    return torch.clamp(q, min=0.0, max=1.0)


@torch.no_grad()
def collect_training_samples(cfg, model, dataset_name, max_images, logger):
    if not hasattr(model.roi_heads, "_shared_roi_transform"):
        raise RuntimeError(
            "VAE-FSOD trainer currently supports Res5ROIHeads (C4) only."
        )

    loader = build_detection_test_loader(cfg, dataset_name)
    roi_feature_name = cfg.MODEL.ROI_HEADS.IN_FEATURES[0]
    aug_per_box = int(cfg.MODEL.VAE_FSOD.AUG_PER_BOX)
    scale_max = float(cfg.MODEL.VAE_FSOD.AUG_BOX_SCALE_MAX)
    max_rois = int(cfg.MODEL.VAE_FSOD.MAX_ROIS)

    feat_chunks = []
    label_chunks = []
    iou_chunks = []
    quality_chunks = []
    num_imgs = 0
    num_rois = 0

    for batch in loader:
        if max_images > 0 and num_imgs >= max_images:
            break
        images = model.preprocess_image(batch)
        backbone_feats = model.backbone(images.tensor)
        feat_map = backbone_feats[roi_feature_name]

        for i, sample in enumerate(batch):
            num_imgs += 1
            if max_images > 0 and num_imgs > max_images:
                break
            if "instances" not in sample:
                continue
            inst = sample["instances"].to(model.device)
            if len(inst) == 0:
                continue

            gt_boxes = inst.gt_boxes.tensor
            gt_classes = inst.gt_classes
            image_hw = inst.image_size

            all_boxes = []
            all_labels = []
            all_ious = []
            all_qualities = []
            for g in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[g]
                if gt_boxes.shape[0] > 1:
                    other_gt = torch.cat([gt_boxes[:g], gt_boxes[g + 1 :]], dim=0)
                else:
                    other_gt = gt_boxes.new_zeros((0, 4))

                q_gt = _compute_quality_vector(gt_box, gt_box, other_gt)
                all_boxes.append(gt_box)
                all_labels.append(gt_classes[g].item())
                all_ious.append(float(q_gt[0].item()))
                all_qualities.append(q_gt)

                for _ in range(aug_per_box):
                    aug_box = _augment_box_xyxy(gt_box, image_hw, scale_max).to(model.device)
                    q_aug = _compute_quality_vector(aug_box, gt_box, other_gt)
                    all_boxes.append(aug_box)
                    all_labels.append(gt_classes[g].item())
                    all_ious.append(float(q_aug[0].item()))
                    all_qualities.append(q_aug)

            boxes_tensor = torch.stack(all_boxes, dim=0)
            pooled = model.roi_heads._shared_roi_transform(
                [feat_map[i : i + 1]],
                [Boxes(boxes_tensor)],
            )
            pooled = pooled.mean(dim=[2, 3]).detach().cpu()

            feat_chunks.append(pooled)
            label_chunks.append(torch.tensor(all_labels, dtype=torch.long))
            iou_chunks.append(torch.tensor(all_ious, dtype=torch.float32))
            quality_chunks.append(torch.stack(all_qualities, dim=0).detach().cpu())
            num_rois += pooled.shape[0]
            if num_rois >= max_rois:
                break

        if num_rois >= max_rois:
            break

    if not feat_chunks:
        raise RuntimeError("No RoI features collected for VAE training.")

    features = torch.cat(feat_chunks, dim=0)[:max_rois]
    labels = torch.cat(label_chunks, dim=0)[:max_rois]
    ious = torch.cat(iou_chunks, dim=0)[:max_rois]
    qualities = torch.cat(quality_chunks, dim=0)[:max_rois]

    logger.info("Collected %d training RoI samples from %d images.", features.shape[0], num_imgs)
    return SamplePack(
        features=features,
        labels=labels,
        ious=ious,
        qualities=qualities,
        quality_keys=tuple(DEFAULT_QUALITY_KEYS),
    )


def _select_quality_subset(samples, cfg):
    if not bool(cfg.MODEL.VAE_FSOD.QUALITY.ENABLE):
        return None, None, None

    requested_keys = list(cfg.MODEL.VAE_FSOD.QUALITY.KEYS)
    if len(requested_keys) == 0:
        raise ValueError("MODEL.VAE_FSOD.QUALITY.KEYS must be non-empty when QUALITY.ENABLE=True.")
    if "iou" not in requested_keys:
        raise ValueError("MODEL.VAE_FSOD.QUALITY.KEYS must include 'iou'.")

    available = list(samples.quality_keys)
    index_list = []
    for key in requested_keys:
        if key not in available:
            raise ValueError("Requested quality key '{}' not available in collected samples.".format(key))
        index_list.append(available.index(key))

    selected_qualities = samples.qualities[:, index_list].contiguous()
    iou_index = requested_keys.index("iou")
    return selected_qualities, requested_keys, iou_index


def _subsample_tensor(tensor, max_count, generator):
    n = int(tensor.shape[0])
    max_count = int(max_count)
    if max_count <= 0 or n <= max_count:
        return tensor.detach().cpu()
    idx = torch.randperm(n, generator=generator)[:max_count]
    return tensor[idx].detach().cpu()


def _quantiles_compat(x, q_values):
    if x.ndim != 1:
        raise ValueError("Expected rank-1 tensor for quantile computation, got shape={}.".format(tuple(x.shape)))
    if x.numel() == 0:
        raise ValueError("Cannot compute quantiles of an empty tensor.")

    q_tensor = x.new_tensor([float(q) for q in q_values], dtype=x.dtype, device=x.device)
    if hasattr(torch, "quantile"):
        return torch.quantile(x, q_tensor)

    # Backward-compatible linear interpolation quantile.
    xs, _ = torch.sort(x)
    n = int(xs.numel())
    out = []
    for q in q_tensor:
        pos = float(q.item()) * float(max(n - 1, 0))
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        w = pos - float(lo)
        out.append((1.0 - w) * xs[lo] + w * xs[hi])
    return torch.stack(out, dim=0)


def _build_quality_generation_bank(samples, selected_qualities, quality_keys, num_classes, cfg, seed, logger):
    weights = list(cfg.MODEL.VAE_FSOD.QUALITY.HARDNESS_WEIGHTS)
    if len(weights) != len(quality_keys):
        raise ValueError(
            "QUALITY.HARDNESS_WEIGHTS length ({}) must match QUALITY.KEYS length ({}).".format(
                len(weights), len(quality_keys)
            )
        )

    qtiles = list(cfg.MODEL.VAE_FSOD.QUALITY.BIN_QUANTILES)
    if len(qtiles) != 2:
        raise ValueError("QUALITY.BIN_QUANTILES must have exactly two values.")
    q_low, q_high = float(qtiles[0]), float(qtiles[1])
    if not (0.0 < q_low < q_high < 1.0):
        raise ValueError("QUALITY.BIN_QUANTILES must satisfy 0 < q1 < q2 < 1.")

    hardness = compute_quality_hardness(selected_qualities, quality_keys, weights)
    quantile_vals = _quantiles_compat(hardness, [q_low, q_high])
    bin_ids = torch.bucketize(hardness, boundaries=quantile_vals, right=False)

    max_per_bin = int(cfg.MODEL.VAE_FSOD.QUALITY.MAX_BANK_PER_CLASS_BIN)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    class_bins = {}
    class_all = {}
    for cid in range(int(num_classes)):
        cls_mask = samples.labels == int(cid)
        cls_q = selected_qualities[cls_mask]
        cls_bin = bin_ids[cls_mask]
        class_all[int(cid)] = _subsample_tensor(cls_q, max_per_bin, generator)
        per_bin = []
        for bid in range(3):
            bin_q = cls_q[cls_bin == bid]
            per_bin.append(_subsample_tensor(bin_q, max_per_bin, generator))
        class_bins[int(cid)] = per_bin

    global_bins = []
    for bid in range(3):
        global_bins.append(_subsample_tensor(selected_qualities[bin_ids == bid], max_per_bin, generator))

    bank = {
        "quality_keys": list(quality_keys),
        "hardness_weights": [float(x) for x in weights],
        "bin_quantiles": [q_low, q_high],
        "bin_values": [float(quantile_vals[0].item()), float(quantile_vals[1].item())],
        "class_bins": class_bins,
        "class_all": class_all,
        "global_bins": global_bins,
        "global_all": _subsample_tensor(selected_qualities, max_per_bin, generator),
    }

    logger.info(
        "Built quality generation bank: keys=%s, quantile_values=[%.4f, %.4f]",
        quality_keys,
        bank["bin_values"][0],
        bank["bin_values"][1],
    )
    return bank


def train_vae(cfg, samples, class_names, device, seed, logger):
    torch.manual_seed(seed)
    random.seed(seed)

    sem_emb = build_text_semantic_embeddings(
        class_names,
        source=cfg.MODEL.VAE_FSOD.SEMANTIC_SOURCE,
        device=device,
    )
    if sem_emb.shape[1] != int(cfg.MODEL.VAE_FSOD.SEMANTIC_DIM):
        raise ValueError(
            "Semantic embedding dim mismatch: got {}, expected {}.".format(
                sem_emb.shape[1], int(cfg.MODEL.VAE_FSOD.SEMANTIC_DIM)
            )
        )

    selected_qualities, quality_keys, quality_iou_index = _select_quality_subset(samples, cfg)
    quality_enabled = selected_qualities is not None

    if quality_enabled:
        vae = QualityConditionalVAE(
            feature_dim=int(cfg.MODEL.VAE_FSOD.FEATURE_DIM),
            semantic_dim=int(cfg.MODEL.VAE_FSOD.SEMANTIC_DIM),
            quality_dim=int(selected_qualities.shape[1]),
            latent_dim=int(cfg.MODEL.VAE_FSOD.LATENT_DIM),
            encoder_hidden=int(cfg.MODEL.VAE_FSOD.ENCODER_HIDDEN),
            decoder_hidden=int(cfg.MODEL.VAE_FSOD.DECODER_HIDDEN),
            iou_index=int(quality_iou_index),
        ).to(device)
        dataset = TensorDataset(samples.features, samples.labels, selected_qualities)
    else:
        vae = NormConditionalVAE(
            feature_dim=int(cfg.MODEL.VAE_FSOD.FEATURE_DIM),
            semantic_dim=int(cfg.MODEL.VAE_FSOD.SEMANTIC_DIM),
            latent_dim=int(cfg.MODEL.VAE_FSOD.LATENT_DIM),
            encoder_hidden=int(cfg.MODEL.VAE_FSOD.ENCODER_HIDDEN),
            decoder_hidden=int(cfg.MODEL.VAE_FSOD.DECODER_HIDDEN),
        ).to(device)
        dataset = TensorDataset(samples.features, samples.labels, samples.ious)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.MODEL.VAE_FSOD.TRAIN_BATCH_SIZE),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    optim = torch.optim.Adam(
        vae.parameters(),
        lr=float(cfg.MODEL.VAE_FSOD.TRAIN_LR),
        weight_decay=float(cfg.MODEL.VAE_FSOD.TRAIN_WEIGHT_DECAY),
    )

    base_norm_min, base_norm_max = paper_default_norm_range(int(cfg.MODEL.VAE_FSOD.LATENT_DIM))
    norm_min = float(cfg.MODEL.VAE_FSOD.NORM_MIN_FACTOR) * base_norm_min
    norm_max = float(cfg.MODEL.VAE_FSOD.NORM_MAX_FACTOR) * base_norm_max
    iou_min = float(cfg.MODEL.VAE_FSOD.IOU_MIN)
    iou_max = float(cfg.MODEL.VAE_FSOD.IOU_MAX)

    q_aux_weight = float(cfg.MODEL.VAE_FSOD.QUALITY.AUX_LOSS_WEIGHT)
    q_aux_type = str(cfg.MODEL.VAE_FSOD.QUALITY.AUX_LOSS_TYPE)

    for epoch in range(int(cfg.MODEL.VAE_FSOD.TRAIN_EPOCHS)):
        vae.train()
        running = 0.0
        running_q = 0.0
        for batch in loader:
            if quality_enabled:
                feats, labels, qualities = batch
                feats = feats.to(device)
                labels = labels.to(device)
                qualities = qualities.to(device)
                semantics = sem_emb[labels]

                recon, mu, logvar, q_pred = vae(
                    feats,
                    semantics,
                    qualities,
                    iou_min=iou_min,
                    iou_max=iou_max,
                    norm_min=norm_min,
                    norm_max=norm_max,
                )
                base_loss, recon_loss, kl = vae_loss(
                    recon,
                    feats,
                    mu,
                    logvar,
                    recon_weight=float(cfg.MODEL.VAE_FSOD.RECON_LOSS_WEIGHT),
                    kl_weight=float(cfg.MODEL.VAE_FSOD.KL_LOSS_WEIGHT),
                )
                q_loss = quality_consistency_loss(q_pred, qualities, loss_type=q_aux_type)
                loss = base_loss + q_aux_weight * q_loss
                running_q += float(q_loss.item())
            else:
                feats, labels, ious = batch
                feats = feats.to(device)
                labels = labels.to(device)
                ious = ious.to(device)
                semantics = sem_emb[labels]

                recon, mu, logvar = vae(
                    feats,
                    semantics,
                    ious,
                    iou_min=iou_min,
                    iou_max=iou_max,
                    norm_min=norm_min,
                    norm_max=norm_max,
                )
                loss, recon_loss, kl = vae_loss(
                    recon,
                    feats,
                    mu,
                    logvar,
                    recon_weight=float(cfg.MODEL.VAE_FSOD.RECON_LOSS_WEIGHT),
                    kl_weight=float(cfg.MODEL.VAE_FSOD.KL_LOSS_WEIGHT),
                )

            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())

        if quality_enabled:
            logger.info(
                "Epoch %d/%d loss=%.6f q_loss=%.6f",
                epoch + 1,
                int(cfg.MODEL.VAE_FSOD.TRAIN_EPOCHS),
                running / max(len(loader), 1),
                running_q / max(len(loader), 1),
            )
        else:
            logger.info(
                "Epoch %d/%d loss=%.6f",
                epoch + 1,
                int(cfg.MODEL.VAE_FSOD.TRAIN_EPOCHS),
                running / max(len(loader), 1),
            )

    quality_bank = None
    if quality_enabled:
        quality_bank = _build_quality_generation_bank(
            samples=samples,
            selected_qualities=selected_qualities,
            quality_keys=quality_keys,
            num_classes=len(class_names),
            cfg=cfg,
            seed=seed,
            logger=logger,
        )
        quality_bank["class_names"] = list(class_names)

    return vae, sem_emb, quality_enabled, quality_keys, quality_iou_index, quality_bank


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(name="train_vae_fsod")
    cfg = setup_cfg(args)

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    dataset_name = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(dataset_name)
    class_names = list(getattr(metadata, "thing_classes"))
    if len(class_names) == 0:
        raise RuntimeError("Dataset has empty thing_classes: {}".format(dataset_name))

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    device = torch.device(cfg.MODEL.DEVICE)

    samples = collect_training_samples(cfg, model, dataset_name, args.max_images, logger)
    (
        vae,
        sem_emb,
        quality_enabled,
        quality_keys,
        quality_iou_index,
        quality_bank,
    ) = train_vae(cfg, samples, class_names, device, args.seed, logger)

    model_type = "quality_conditional" if quality_enabled else "norm_conditional"
    save_obj = {
        "model_state": vae.state_dict(),
        "model_type": model_type,
        "class_names": class_names,
        "semantic_source": cfg.MODEL.VAE_FSOD.SEMANTIC_SOURCE,
        "semantic_embeddings": sem_emb.detach().cpu(),
        "feature_dim": int(cfg.MODEL.VAE_FSOD.FEATURE_DIM),
        "semantic_dim": int(cfg.MODEL.VAE_FSOD.SEMANTIC_DIM),
        "latent_dim": int(cfg.MODEL.VAE_FSOD.LATENT_DIM),
        "encoder_hidden": int(cfg.MODEL.VAE_FSOD.ENCODER_HIDDEN),
        "decoder_hidden": int(cfg.MODEL.VAE_FSOD.DECODER_HIDDEN),
        "iou_min": float(cfg.MODEL.VAE_FSOD.IOU_MIN),
        "iou_max": float(cfg.MODEL.VAE_FSOD.IOU_MAX),
        "norm_min_factor": float(cfg.MODEL.VAE_FSOD.NORM_MIN_FACTOR),
        "norm_max_factor": float(cfg.MODEL.VAE_FSOD.NORM_MAX_FACTOR),
        "beta_interval": float(cfg.MODEL.VAE_FSOD.BETA_INTERVAL),
        "num_gen_per_class": int(cfg.MODEL.VAE_FSOD.NUM_GEN_PER_CLASS),
        "train_dataset": dataset_name,
        "train_samples": int(samples.features.shape[0]),
        "quality_enabled": bool(quality_enabled),
    }

    if quality_enabled:
        save_obj.update(
            {
                "quality_keys": list(quality_keys),
                "quality_iou_index": int(quality_iou_index),
                "quality_aux_loss_weight": float(cfg.MODEL.VAE_FSOD.QUALITY.AUX_LOSS_WEIGHT),
                "quality_aux_loss_type": str(cfg.MODEL.VAE_FSOD.QUALITY.AUX_LOSS_TYPE),
                "quality_generation_bank": quality_bank,
            }
        )

    torch.save(save_obj, args.output)
    logger.info("Saved VAE checkpoint: %s", args.output)

    meta_path = os.path.splitext(args.output)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "train_dataset": dataset_name,
                "train_samples": int(samples.features.shape[0]),
                "class_names": class_names,
                "model_type": model_type,
                "quality_enabled": bool(quality_enabled),
                "quality_keys": list(quality_keys) if quality_enabled else [],
            },
            f,
            indent=2,
        )
    logger.info("Saved metadata: %s", meta_path)
