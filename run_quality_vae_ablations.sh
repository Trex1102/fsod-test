#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_quality_vae_ablations.sh \
    --dataset {coco|voc} \
    --exp-name EXP_NAME \
    --imagenet-pretrain /path/to/R-101.pkl \
    --imagenet-pretrain-torch /path/to/resnet101-5d3b4d8f.pth \
    [--split-id {1|2|3}] \
    [--num-gpus N] \
    [--shots "1 2 3 5 10 30"] \
    [--seeds "0"] \
    [--ablations "baseline iou_only no_qaux no_crowding center_heavy hard_bias easy_bias wide_bins"] \
    [--skip-base]

Notes:
  - This script runs quality-VAE ablations sequentially by calling run_vae_fsod.sh.
  - Each ablation writes results to:
    checkpoints/<dataset>/<exp-name>/qualityVaeFsod/<ablation>/
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET=""
EXP_NAME=""
IMAGENET_PRETRAIN=""
IMAGENET_PRETRAIN_TORCH=""
SPLIT_ID="1"
NUM_GPUS="8"
SHOTS=""
SEEDS="0"
ABLATIONS="baseline iou_only no_qaux no_crowding center_heavy hard_bias easy_bias wide_bins"
SKIP_BASE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --imagenet-pretrain) IMAGENET_PRETRAIN="$2"; shift 2 ;;
    --imagenet-pretrain-torch) IMAGENET_PRETRAIN_TORCH="$2"; shift 2 ;;
    --split-id) SPLIT_ID="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --ablations) ABLATIONS="$2"; shift 2 ;;
    --skip-base) SKIP_BASE="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET}" || -z "${EXP_NAME}" || -z "${IMAGENET_PRETRAIN}" || -z "${IMAGENET_PRETRAIN_TORCH}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi
if [[ "${DATASET}" != "coco" && "${DATASET}" != "voc" ]]; then
  echo "--dataset must be coco or voc" >&2
  exit 1
fi
if [[ "${DATASET}" == "voc" && "${SPLIT_ID}" != "1" && "${SPLIT_ID}" != "2" && "${SPLIT_ID}" != "3" ]]; then
  echo "--split-id must be 1, 2, or 3 for VOC" >&2
  exit 1
fi
if [[ -z "${SHOTS}" ]]; then
  if [[ "${DATASET}" == "coco" ]]; then
    SHOTS="1 2 3 5 10 30"
  else
    SHOTS="1 2 3 5 10"
  fi
fi

for abl in ${ABLATIONS}; do
  echo "[Quality-VAE Ablation] ${abl}"

  cmd=(
    bash run_vae_fsod.sh
    --dataset "${DATASET}"
    --variant quality
    --ablation "${abl}"
    --exp-name "${EXP_NAME}"
    --imagenet-pretrain "${IMAGENET_PRETRAIN}"
    --imagenet-pretrain-torch "${IMAGENET_PRETRAIN_TORCH}"
    --num-gpus "${NUM_GPUS}"
    --shots "${SHOTS}"
    --seeds "${SEEDS}"
  )

  if [[ "${DATASET}" == "voc" ]]; then
    cmd+=(--split-id "${SPLIT_ID}")
  fi

  if [[ "${SKIP_BASE}" == "1" ]]; then
    cmd+=(--skip-base)
  fi

  "${cmd[@]}"
done

echo "All quality-VAE ablation runs completed."
