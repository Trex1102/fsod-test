#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_vae_fsod.sh \
    --dataset {coco|voc} \
    --exp-name EXP_NAME \
    --imagenet-pretrain /path/to/R-101.pkl \
    --imagenet-pretrain-torch /path/to/resnet101-5d3b4d8f.pth \
    [--split-id {1|2|3}] \
    [--num-gpus N] \
    [--shots "1 2 3 5 10 30"] \
    [--seeds "0"] \
    [--skip-base]

Notes:
  - This pipeline matches VAE-FSOD (FSOD only):
    base pretrain -> model surgery(remove) -> train Norm-VAE on base features
    -> generate per-shot feature banks -> FSOD fine-tuning with VAE aux cls loss.
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
RUN_BASE="1"

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
    --skip-base) RUN_BASE="0"; shift 1 ;;
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

SAVE_DIR="checkpoints/${DATASET}/${EXP_NAME}/vaeFsod"
mkdir -p "${SAVE_DIR}"
TMP_CFG_ROOT="$(mktemp -d /tmp/vaefsod_cfgs.XXXXXX)"
cleanup() {
  rm -rf "${TMP_CFG_ROOT}"
}
trap cleanup EXIT

if [[ "${DATASET}" == "coco" ]]; then
  BASE_CFG="configs/coco/vaeFsod/defrcn_det_r101_base_vaefsod.yaml"
  BASE_OUT_DIR="${SAVE_DIR}/defrcn_det_r101_base"
  TEMPLATE_ROOT="configs/coco/vaeFsod"
  DATASET_FLAG="coco14"
  MODEL_SURGERY_DATASET="coco"
else
  BASE_CFG="configs/voc/vaeFsod/defrcn_det_r101_base${SPLIT_ID}_vaefsod.yaml"
  BASE_OUT_DIR="${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}"
  TEMPLATE_ROOT="configs/voc/vaeFsod"
  DATASET_FLAG="voc"
  MODEL_SURGERY_DATASET="voc"
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "Base config not found: ${BASE_CFG}" >&2
  exit 1
fi

run_base_train() {
  echo "[Base] Training with ${BASE_CFG}"
  python3 main.py --num-gpus "${NUM_GPUS}" --config-file "${BASE_CFG}" \
    --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}" OUTPUT_DIR "${BASE_OUT_DIR}"
}

prepare_surgery_weights() {
  local model_final="${BASE_OUT_DIR}/model_final.pth"
  if [[ ! -f "${model_final}" ]]; then
    echo "Base model checkpoint missing: ${model_final}" >&2
    exit 1
  fi
  python3 tools/model_surgery.py --dataset "${MODEL_SURGERY_DATASET}" --method remove \
    --src-path "${model_final}" --save-dir "${BASE_OUT_DIR}"
}

train_norm_vae() {
  local vae_out="${SAVE_DIR}/norm_vae/model_final.pth"
  mkdir -p "$(dirname "${vae_out}")"
  echo "[VAE] Training Norm-VAE on base dataset features"
  python3 tools/train_vae_fsod.py \
    --config-file "${BASE_CFG}" \
    --weights "${BASE_OUT_DIR}/model_final.pth" \
    --output "${vae_out}"
  echo "${vae_out}"
}

render_template_cfg() {
  local src="$1"
  local dst="$2"
  local seed="$3"
  local dataset="$4"
  local split_id="$5"
  python3 - "$src" "$dst" "$seed" "$dataset" "$split_id" <<'PY'
import sys
from pathlib import Path

src, dst, seed, dataset, split_id = sys.argv[1:]
text = Path(src).read_text()
text = text.replace("seedx", f"seed{seed}")
if dataset == "voc":
    text = text.replace("novelx", f"novel{split_id}")
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(text)
PY
}

generate_seed_cfg() {
  local shot="$1"
  local seed="$2"
  local extra_args=()
  local root_template
  local root_generated
  local vae_template
  local vae_generated

  if [[ "${DATASET}" == "coco" ]]; then
    root_template="defrcn_fsod_r101_novel_${shot}shot_seedx.yaml"
    root_generated="defrcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml"
    vae_template="defrcn_fsod_r101_novel_${shot}shot_seedx_vaefsod.yaml"
    vae_generated="defrcn_fsod_r101_novel_${shot}shot_seed${seed}_vaefsod.yaml"
  else
    root_template="defrcn_fsod_r101_novelx_${shot}shot_seedx.yaml"
    root_generated="defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml"
    vae_template="defrcn_fsod_r101_novelx_${shot}shot_seedx_vaefsod.yaml"
    vae_generated="defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_vaefsod.yaml"
    extra_args=(--split "${SPLIT_ID}")
  fi

  local tmp_dataset_root="${TMP_CFG_ROOT}/${DATASET}/fsod/${shot}shot_seed${seed}"
  mkdir -p "${tmp_dataset_root}"
  cp "configs/${DATASET}/${root_template}" "${tmp_dataset_root}/${root_template}"

  python3 tools/create_config.py \
    --dataset "${DATASET_FLAG}" \
    --config_root "${tmp_dataset_root}" \
    --shot "${shot}" \
    --seed "${seed}" \
    --setting "fsod" \
    "${extra_args[@]}"

  if [[ ! -f "${tmp_dataset_root}/${root_generated}" ]]; then
    echo "Failed to create seed config: ${tmp_dataset_root}/${root_generated}" >&2
    exit 1
  fi

  local vae_cfg="${tmp_dataset_root}/vaeFsod/${vae_generated}"
  render_template_cfg "${TEMPLATE_ROOT}/${vae_template}" "${vae_cfg}" "${seed}" "${DATASET}" "${SPLIT_ID}"
  echo "${vae_cfg}"
}

if [[ "${RUN_BASE}" == "1" ]]; then
  run_base_train
fi
prepare_surgery_weights
BASE_WEIGHT="${BASE_OUT_DIR}/model_reset_remove.pth"
if [[ ! -f "${BASE_WEIGHT}" ]]; then
  echo "Missing base FSOD surgery weight: ${BASE_WEIGHT}" >&2
  exit 1
fi

VAE_CKPT="$(train_norm_vae)"
RES_DIR="${SAVE_DIR}/fsod"
mkdir -p "${RES_DIR}"

for shot in ${SHOTS}; do
  for seed in ${SEEDS}; do
    echo "[Fine-tune] shot=${shot} seed=${seed}"
    cfg_path="$(generate_seed_cfg "${shot}" "${seed}")"
    bank_path="${SAVE_DIR}/feature_banks/fsod_${shot}shot_seed${seed}.pth"
    mkdir -p "$(dirname "${bank_path}")"

    python3 tools/generate_vae_fsod_features.py \
      --config-file "${cfg_path}" \
      --vae-ckpt "${VAE_CKPT}" \
      --output "${bank_path}"

    out_dir="${RES_DIR}/${shot}shot_seed${seed}"
    python3 main.py --num-gpus "${NUM_GPUS}" --config-file "${cfg_path}" \
      --opts MODEL.WEIGHTS "${BASE_WEIGHT}" OUTPUT_DIR "${out_dir}" \
             TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}" \
             MODEL.VAE_FSOD.FEATURE_BANK_PATH "${bank_path}"
  done
done

python3 tools/extract_results.py --res-dir "${RES_DIR}" --shot-list ${SHOTS}
echo "All VAE-FSOD runs completed. Results stored in ${SAVE_DIR}"
