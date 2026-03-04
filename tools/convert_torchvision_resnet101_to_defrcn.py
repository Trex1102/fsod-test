#!/usr/bin/env python3
import argparse
import os

import torch

from defrcn.config import get_cfg
from defrcn.modeling import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert torchvision ResNet-101 weights to a DeFRCN-aligned checkpoint."
    )
    parser.add_argument("--config-file", required=True, help="Target model config file")
    parser.add_argument("--src", required=True, help="Source torchvision .pth path")
    parser.add_argument("--dst", required=True, help="Output checkpoint path")
    parser.add_argument(
        "--copy-roi-res5",
        action="store_true",
        help="Also initialize roi_heads.res5.* from torchvision layer4.*",
    )
    return parser.parse_args()


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_torchvision_state(path):
    ckpt = safe_torch_load(path)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format at {}".format(path))

    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def assign_tensor(target_sd, mapped_sd, dst_key, src_sd, src_key, stats):
    if dst_key not in target_sd:
        return
    if src_key not in src_sd:
        stats["missing_src"].append((dst_key, src_key))
        return

    src_t = src_sd[src_key]
    dst_t = target_sd[dst_key]
    if tuple(src_t.shape) != tuple(dst_t.shape):
        stats["shape_mismatch"].append((dst_key, tuple(dst_t.shape), src_key, tuple(src_t.shape)))
        return

    mapped_sd[dst_key] = src_t.detach().clone().cpu()
    stats["loaded"].add(dst_key)


def map_bn(target_sd, mapped_sd, dst_prefix, src_prefix, src_sd, stats):
    for n in ["weight", "bias", "running_mean", "running_var"]:
        assign_tensor(
            target_sd,
            mapped_sd,
            "{}.{}".format(dst_prefix, n),
            src_sd,
            "{}.{}".format(src_prefix, n),
            stats,
        )


def map_stage(target_sd, mapped_sd, src_sd, stage_idx, dst_stage_name, dst_prefix, stats):
    # torchvision: layer{1..4}.{block}.{conv/bn/downsample}
    # defrcn: {prefix}.res{2..5}.{block}.{conv/norm/shortcut}
    for block in range(32):
        src_block = "layer{}.{}".format(stage_idx, block)
        if "{}.conv1.weight".format(src_block) not in src_sd:
            if block == 0:
                raise ValueError("No source keys found for {}".format(src_block))
            break

        dst_block = "{}.{}.{}".format(dst_prefix, dst_stage_name, block)

        assign_tensor(
            target_sd,
            mapped_sd,
            "{}.conv1.weight".format(dst_block),
            src_sd,
            "{}.conv1.weight".format(src_block),
            stats,
        )
        map_bn(
            target_sd,
            mapped_sd,
            "{}.conv1.norm".format(dst_block),
            "{}.bn1".format(src_block),
            src_sd,
            stats,
        )

        assign_tensor(
            target_sd,
            mapped_sd,
            "{}.conv2.weight".format(dst_block),
            src_sd,
            "{}.conv2.weight".format(src_block),
            stats,
        )
        map_bn(
            target_sd,
            mapped_sd,
            "{}.conv2.norm".format(dst_block),
            "{}.bn2".format(src_block),
            src_sd,
            stats,
        )

        assign_tensor(
            target_sd,
            mapped_sd,
            "{}.conv3.weight".format(dst_block),
            src_sd,
            "{}.conv3.weight".format(src_block),
            stats,
        )
        map_bn(
            target_sd,
            mapped_sd,
            "{}.conv3.norm".format(dst_block),
            "{}.bn3".format(src_block),
            src_sd,
            stats,
        )

        if block == 0:
            assign_tensor(
                target_sd,
                mapped_sd,
                "{}.shortcut.weight".format(dst_block),
                src_sd,
                "{}.downsample.0.weight".format(src_block),
                stats,
            )
            map_bn(
                target_sd,
                mapped_sd,
                "{}.shortcut.norm".format(dst_block),
                "{}.downsample.1".format(src_block),
                src_sd,
                stats,
            )


def main():
    args = parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    model = build_model(cfg)
    target_sd = model.state_dict()
    mapped_sd = {}

    src_sd = load_torchvision_state(args.src)

    stats = {
        "loaded": set(),
        "missing_src": [],
        "shape_mismatch": [],
    }

    assign_tensor(target_sd, mapped_sd, "backbone.stem.conv1.weight", src_sd, "conv1.weight", stats)
    map_bn(target_sd, mapped_sd, "backbone.stem.conv1.norm", "bn1", src_sd, stats)

    # backbone res2..res5 from layer1..layer4
    for stage_idx, dst_stage in [(1, "res2"), (2, "res3"), (3, "res4"), (4, "res5")]:
        map_stage(target_sd, mapped_sd, src_sd, stage_idx, dst_stage, "backbone", stats)

    # Optional: initialize roi_heads.res5 from layer4.
    if args.copy_roi_res5:
        map_stage(target_sd, mapped_sd, src_sd, 4, "res5", "roi_heads", stats)

    dst_dir = os.path.dirname(args.dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    out = {
        "model": mapped_sd,
        "__author__": "torchvision_resnet101_explicit_mapper",
        "matching_heuristics": False,
        "meta": {
            "source": args.src,
            "config": args.config_file,
            "copy_roi_res5": bool(args.copy_roi_res5),
            "loaded_tensors": len(stats["loaded"]),
            "target_tensors": len(target_sd),
            "missing_src": len(stats["missing_src"]),
            "shape_mismatch": len(stats["shape_mismatch"]),
        },
    }
    torch.save(out, args.dst)

    print("Saved:", args.dst)
    print("Loaded tensors:", len(stats["loaded"]), "/", len(target_sd))
    print("Checkpoint tensors:", len(mapped_sd))
    print("Missing source tensors:", len(stats["missing_src"]))
    print("Shape mismatch tensors:", len(stats["shape_mismatch"]))
    if stats["shape_mismatch"]:
        for item in stats["shape_mismatch"][:10]:
            print("  mismatch:", item)


if __name__ == "__main__":
    main()
