#!/usr/bin/env python3
import argparse
import os

import torch
from detectron2.utils.logger import setup_logger

from defrcn.config import get_cfg
from defrcn.dataloader import MetadataCatalog
from defrcn.modeling.vae_fsod import (
    DEFAULT_QUALITY_KEYS,
    NormConditionalVAE,
    QualityConditionalVAE,
    build_text_semantic_embeddings,
    normalize_quality_ratios,
)
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


def _counts_from_ratios(total, ratios):
    raw = [float(r) * float(total) for r in ratios]
    counts = [int(x) for x in raw]
    remain = int(total) - sum(counts)
    frac_order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i]), reverse=True)
    for i in range(remain):
        counts[frac_order[i % len(frac_order)]] += 1
    return counts


def _lookup_with_int_or_str(mapping, key):
    if key in mapping:
        return mapping[key]
    skey = str(key)
    if skey in mapping:
        return mapping[skey]
    return None


def _draw_from_source(source, count, device):
    if source is None or count <= 0:
        return None
    if not isinstance(source, torch.Tensor):
        return None
    n = int(source.shape[0])
    if n <= 0:
        return None
    idx = torch.randint(0, n, (int(count),), device=source.device)
    return source[idx].to(device=device, non_blocking=True)


def _sample_quality_for_class(class_id, num_gen, bank, ratios, quality_dim, device, use_class_specific=True):
    class_bins_map = bank.get("class_bins", {})
    class_all_map = bank.get("class_all", {})
    global_bins = bank.get("global_bins", [])
    global_all = bank.get("global_all", None)

    class_bins = None
    class_all = None
    if use_class_specific:
        class_bins = _lookup_with_int_or_str(class_bins_map, class_id)
        class_all = _lookup_with_int_or_str(class_all_map, class_id)

    if class_bins is None:
        class_bins = [None, None, None]
    if len(class_bins) != 3:
        class_bins = [None, None, None]

    counts = _counts_from_ratios(num_gen, ratios)
    draws = []
    for bid, cnt in enumerate(counts):
        src_candidates = [
            class_bins[bid],
            global_bins[bid] if bid < len(global_bins) else None,
            class_all,
            global_all,
        ]
        sampled = None
        for src in src_candidates:
            sampled = _draw_from_source(src, cnt, device)
            if sampled is not None:
                break
        if sampled is None:
            raise RuntimeError(
                "Unable to sample quality vectors for class {} bin {} (count={}).".format(
                    class_id, bid, cnt
                )
            )
        draws.append(sampled)

    q = torch.cat(draws, dim=0)
    if q.shape[0] < int(num_gen):
        extra = _draw_from_source(class_all, int(num_gen) - int(q.shape[0]), device)
        if extra is None:
            extra = _draw_from_source(global_all, int(num_gen) - int(q.shape[0]), device)
        if extra is None:
            raise RuntimeError("Not enough quality samples to fill generation batch.")
        q = torch.cat([q, extra], dim=0)

    if q.shape[0] > int(num_gen):
        q = q[: int(num_gen)]

    if q.shape[1] != int(quality_dim):
        raise ValueError(
            "Quality dim mismatch in sampled vectors: got {}, expected {}.".format(
                q.shape[1], quality_dim
            )
        )

    perm = torch.randperm(q.shape[0], device=q.device)
    return q[perm]


@torch.no_grad()
def main():
    args = parse_args()
    logger = setup_logger(name="generate_vae_fsod_features")
    cfg = setup_cfg(args)
    device = torch.device(cfg.MODEL.DEVICE)

    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    model_type = str(ckpt.get("model_type", "norm_conditional"))
    quality_enabled = bool(ckpt.get("quality_enabled", False) or model_type == "quality_conditional")

    if quality_enabled:
        quality_keys = list(ckpt.get("quality_keys", DEFAULT_QUALITY_KEYS))
        quality_iou_index = int(ckpt.get("quality_iou_index", quality_keys.index("iou") if "iou" in quality_keys else 0))
        vae = QualityConditionalVAE(
            feature_dim=int(ckpt["feature_dim"]),
            semantic_dim=int(ckpt["semantic_dim"]),
            quality_dim=len(quality_keys),
            latent_dim=int(ckpt["latent_dim"]),
            encoder_hidden=int(ckpt["encoder_hidden"]),
            decoder_hidden=int(ckpt["decoder_hidden"]),
            iou_index=quality_iou_index,
        ).to(device)
    else:
        quality_keys = []
        quality_iou_index = -1
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
    iou_min = float(cfg.MODEL.VAE_FSOD.IOU_MIN)
    iou_max = float(cfg.MODEL.VAE_FSOD.IOU_MAX)

    feat_chunks = []
    label_chunks = []
    quality_chunks = []

    if quality_enabled:
        if "quality_generation_bank" not in ckpt:
            raise RuntimeError("Quality-conditioned VAE ckpt missing 'quality_generation_bank'.")
        quality_bank = ckpt["quality_generation_bank"]
        ratios = normalize_quality_ratios(list(cfg.MODEL.VAE_FSOD.QUALITY.GEN_BIN_RATIOS))
        train_bank_class_names = list(quality_bank.get("class_names", []))
        use_class_specific = train_bank_class_names == class_names
        if not use_class_specific:
            logger.info(
                "Quality bank class names differ from target dataset classes; using global quality bins."
            )

        for cid in range(len(class_names)):
            semantics = sem[cid].unsqueeze(0).expand(num_gen, -1).contiguous()
            qualities = _sample_quality_for_class(
                class_id=cid,
                num_gen=num_gen,
                bank=quality_bank,
                ratios=ratios,
                quality_dim=len(quality_keys),
                device=device,
                use_class_specific=use_class_specific,
            )
            gen_feats = vae.generate(
                semantics,
                qualities,
                iou_min=iou_min,
                iou_max=iou_max,
                norm_min=norm_min,
                norm_max=norm_max,
            )
            feat_chunks.append(gen_feats.cpu())
            label_chunks.append(torch.full((num_gen,), int(cid), dtype=torch.long))
            quality_chunks.append(qualities.cpu())
    else:
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
        "quality_enabled": quality_enabled,
        "model_type": model_type,
    }

    if quality_enabled:
        save_obj.update(
            {
                "qualities": torch.cat(quality_chunks, dim=0),
                "quality_keys": quality_keys,
                "quality_iou_index": quality_iou_index,
                "quality_gen_bin_ratios": list(cfg.MODEL.VAE_FSOD.QUALITY.GEN_BIN_RATIOS),
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    torch.save(save_obj, args.output)
    logger.info(
        "Saved generated feature bank: %s (N=%d, C=%d, quality=%s)",
        args.output,
        int(features.shape[0]),
        int(features.shape[1]),
        str(quality_enabled),
    )


if __name__ == "__main__":
    main()
