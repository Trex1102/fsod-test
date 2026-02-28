#!/usr/bin/env python3
import argparse
import json
import math
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
    NormConditionalVAE,
    build_text_semantic_embeddings,
    paper_default_norm_range,
)
from defrcn.modeling.vae_fsod.norm_vae import vae_loss
import defrcn.data  # noqa: F401


@dataclass
class SamplePack:
    features: torch.Tensor
    labels: torch.Tensor
    ious: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser("Train Norm-VAE for VAE-FSOD")
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
            for g in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[g]
                all_boxes.append(gt_box)
                all_labels.append(gt_classes[g].item())
                all_ious.append(1.0)
                for _ in range(aug_per_box):
                    aug_box = _augment_box_xyxy(gt_box, image_hw, scale_max).to(model.device)
                    iou = pairwise_iou(Boxes(aug_box.view(1, 4)), Boxes(gt_box.view(1, 4))).item()
                    all_boxes.append(aug_box)
                    all_labels.append(gt_classes[g].item())
                    all_ious.append(float(iou))

            boxes_tensor = torch.stack(all_boxes, dim=0)
            pooled = model.roi_heads._shared_roi_transform(
                [feat_map[i : i + 1]],
                [Boxes(boxes_tensor)],
            )
            pooled = pooled.mean(dim=[2, 3]).detach().cpu()

            feat_chunks.append(pooled)
            label_chunks.append(torch.tensor(all_labels, dtype=torch.long))
            iou_chunks.append(torch.tensor(all_ious, dtype=torch.float32))
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
    logger.info("Collected %d training RoI samples from %d images.", features.shape[0], num_imgs)
    return SamplePack(features=features, labels=labels, ious=ious)


def train_vae(cfg, samples, class_names, device, seed, logger):
    torch.manual_seed(seed)
    random.seed(seed)

    vae = NormConditionalVAE(
        feature_dim=int(cfg.MODEL.VAE_FSOD.FEATURE_DIM),
        semantic_dim=int(cfg.MODEL.VAE_FSOD.SEMANTIC_DIM),
        latent_dim=int(cfg.MODEL.VAE_FSOD.LATENT_DIM),
        encoder_hidden=int(cfg.MODEL.VAE_FSOD.ENCODER_HIDDEN),
        decoder_hidden=int(cfg.MODEL.VAE_FSOD.DECODER_HIDDEN),
    ).to(device)

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

    for epoch in range(int(cfg.MODEL.VAE_FSOD.TRAIN_EPOCHS)):
        vae.train()
        running = 0.0
        for feats, labels, ious in loader:
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
        logger.info(
            "Epoch %d/%d loss=%.6f",
            epoch + 1,
            int(cfg.MODEL.VAE_FSOD.TRAIN_EPOCHS),
            running / max(len(loader), 1),
        )

    return vae, sem_emb


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
    vae, sem_emb = train_vae(cfg, samples, class_names, device, args.seed, logger)

    save_obj = {
        "model_state": vae.state_dict(),
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
    }
    torch.save(save_obj, args.output)
    logger.info("Saved VAE checkpoint: %s", args.output)

    meta_path = os.path.splitext(args.output)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "train_dataset": dataset_name,
                "train_samples": int(samples.features.shape[0]),
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    logger.info("Saved metadata: %s", meta_path)
