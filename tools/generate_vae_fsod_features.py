#!/usr/bin/env python3
import argparse
import math
import os

import torch
from detectron2.utils.logger import setup_logger

from defrcn.config import get_cfg
from defrcn.dataloader import MetadataCatalog
from defrcn.modeling.vae_fsod import NormConditionalVAE, build_text_semantic_embeddings
from defrcn.modeling.vae_fsod.norm_vae import paper_default_norm_range
import defrcn.data  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser("Generate VAE-FSOD synthetic feature bank")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--vae-ckpt", required=True, help="Path to trained VAE checkpoint")
    parser.add_argument("--output", required=True, help="Output path for feature bank (.pth)")
    parser.add_argument("--dataset", default="", help="Override cfg.DATASETS.TRAIN[0]")
    parser.add_argument("--num-gen-per-class", type=int, default=-1, help="Override cfg")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.dataset:
        cfg.DATASETS.TRAIN = (args.dataset,)
    cfg.freeze()
    return cfg


@torch.no_grad()
def main():
    args = parse_args()
    logger = setup_logger(name="generate_vae_fsod_features")
    cfg = setup_cfg(args)
    device = torch.device(cfg.MODEL.DEVICE)

    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae = NormConditionalVAE(
        feature_dim=int(ckpt["feature_dim"]),
        semantic_dim=int(ckpt["semantic_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        encoder_hidden=int(ckpt["encoder_hidden"]),
        decoder_hidden=int(ckpt["decoder_hidden"]),
    ).to(device)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()

    dataset_name = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(dataset_name)
    class_names = list(getattr(metadata, "thing_classes"))
    if len(class_names) == 0:
        raise RuntimeError("Dataset has empty thing_classes: {}".format(dataset_name))

    sem = build_text_semantic_embeddings(
        class_names,
        source=cfg.MODEL.VAE_FSOD.SEMANTIC_SOURCE,
        device=device,
    )
    if sem.shape[1] != int(ckpt["semantic_dim"]):
        raise ValueError(
            "Semantic embedding dim mismatch: got {}, expected {}.".format(
                sem.shape[1], int(ckpt["semantic_dim"])
            )
        )

    num_gen = (
        int(args.num_gen_per_class)
        if args.num_gen_per_class > 0
        else int(cfg.MODEL.VAE_FSOD.NUM_GEN_PER_CLASS)
    )
    beta_interval = float(cfg.MODEL.VAE_FSOD.BETA_INTERVAL)

    base_norm_min, base_norm_max = paper_default_norm_range(int(ckpt["latent_dim"]))
    norm_min = float(cfg.MODEL.VAE_FSOD.NORM_MIN_FACTOR) * base_norm_min
    norm_max = float(cfg.MODEL.VAE_FSOD.NORM_MAX_FACTOR) * base_norm_max

    feat_chunks = []
    label_chunks = []
    for cid in range(len(class_names)):
        semantics = sem[cid].unsqueeze(0).expand(num_gen, -1).contiguous()
        betas = norm_min + beta_interval * torch.arange(num_gen, device=device, dtype=torch.float32)
        betas = torch.clamp(betas, min=norm_min, max=norm_max)
        gen_feats = vae.generate(semantics, betas)
        feat_chunks.append(gen_feats.cpu())
        label_chunks.append(torch.full((num_gen,), int(cid), dtype=torch.long))

    features = torch.cat(feat_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    save_obj = {
        "features": features,
        "labels": labels,
        "class_names": class_names,
        "num_gen_per_class": num_gen,
        "beta_interval": beta_interval,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "vae_ckpt": args.vae_ckpt,
        "dataset": dataset_name,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    torch.save(save_obj, args.output)
    logger.info(
        "Saved generated feature bank: %s (N=%d, C=%d)",
        args.output,
        int(features.shape[0]),
        int(features.shape[1]),
    )


if __name__ == "__main__":
    main()
