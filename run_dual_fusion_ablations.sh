#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_dual_fusion_ablations.sh \
    --dataset {coco|voc} \
    --exp-name EXP_NAME \
    --ablation ABLATION_NAME \
    --imagenet-pretrain /path/to/R-101.pkl \
    --imagenet-pretrain-torch /path/to/resnet101-5d3b4d8f.pth \
    [--setting {fsod|gfsod|both}] \
    [--split-id {1|2|3}] \
    [--num-gpus N] \
    [--shots "1 2 3 5 10 30"] \
    [--seeds "0 1 2 3 4 5 6 7 8 9"] \
    [--skip-base]

Examples:
  bash run_dual_fusion_ablations.sh \
    --dataset coco --exp-name df_run1 --ablation baseline \
    --imagenet-pretrain /data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl \
    --imagenet-pretrain-torch /data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth \
    --setting both --seeds "0" --shots "1 2 3 5 10 30"

  bash run_dual_fusion_ablations.sh \
    --dataset voc --split-id 1 --exp-name df_voc_s1 --ablation lr_0p005 \
    --imagenet-pretrain /data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl \
    --imagenet-pretrain-torch /data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth \
    --setting fsod --seeds "0 1 2 3 4 5 6 7 8 9" --shots "1 2 3 5 10"
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET=""
EXP_NAME=""
ABLATION=""
IMAGENET_PRETRAIN=""
IMAGENET_PRETRAIN_TORCH=""
SETTING="both"
SPLIT_ID="1"
NUM_GPUS="8"
SHOTS=""
SEEDS="0"
RUN_BASE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --exp-name)
      EXP_NAME="$2"; shift 2 ;;
    --ablation)
      ABLATION="$2"; shift 2 ;;
    --imagenet-pretrain)
      IMAGENET_PRETRAIN="$2"; shift 2 ;;
    --imagenet-pretrain-torch)
      IMAGENET_PRETRAIN_TORCH="$2"; shift 2 ;;
    --setting)
      SETTING="$2"; shift 2 ;;
    --split-id)
      SPLIT_ID="$2"; shift 2 ;;
    --num-gpus)
      NUM_GPUS="$2"; shift 2 ;;
    --shots)
      SHOTS="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --skip-base)
      RUN_BASE="0"; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "${DATASET}" || -z "${EXP_NAME}" || -z "${ABLATION}" || -z "${IMAGENET_PRETRAIN}" || -z "${IMAGENET_PRETRAIN_TORCH}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "${DATASET}" != "coco" && "${DATASET}" != "voc" ]]; then
  echo "--dataset must be coco or voc" >&2
  exit 1
fi
if [[ "${SETTING}" != "fsod" && "${SETTING}" != "gfsod" && "${SETTING}" != "both" ]]; then
  echo "--setting must be fsod, gfsod, or both" >&2
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

CFG_ROOT="configs/${DATASET}/dualFusionAblations"
ABLATION_ROOT="${CFG_ROOT}/${ABLATION}"
BASELINE_ROOT="${CFG_ROOT}/baseline"

if [[ ! -d "${ABLATION_ROOT}" ]]; then
  echo "Ablation folder not found: ${ABLATION_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${BASELINE_ROOT}" ]]; then
  echo "Baseline folder not found: ${BASELINE_ROOT}" >&2
  exit 1
fi

SAVE_DIR="checkpoints/${DATASET}/${EXP_NAME}/${ABLATION}"
mkdir -p "${SAVE_DIR}"

if [[ "${DATASET}" == "coco" ]]; then
  BASE_CFG="${ABLATION_ROOT}/defrcn_det_r101_base_dualfusion.yaml"
  MODEL_SURGERY_DATASET="coco"
  DATASET_FLAG="coco14"
  BASE_OUT_DIR="${SAVE_DIR}/defrcn_det_r101_base"
else
  BASE_CFG="${ABLATION_ROOT}/defrcn_det_r101_base${SPLIT_ID}_dualfusion.yaml"
  MODEL_SURGERY_DATASET="voc"
  DATASET_FLAG="voc"
  BASE_OUT_DIR="${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}"
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "Base config not found: ${BASE_CFG}" >&2
  exit 1
fi

TMP_CFG_ROOT="$(mktemp -d /tmp/dual_fusion_ablation_cfgs.XXXXXX)"
cleanup() {
  rm -rf "${TMP_CFG_ROOT}"
}
trap cleanup EXIT

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
text = text.replace('seedx', f'seed{seed}')
if dataset == 'voc':
    text = text.replace('novelx', f'novel{split_id}')
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(text)
PY
}

generate_seed_cfg() {
  local setting="$1"
  local shot="$2"
  local seed="$3"
  local create_cfg_extra=()

  local root_template
  local dual_template
  local root_generated
  local dual_generated

  if [[ "${DATASET}" == "coco" ]]; then
    root_template="defrcn_${setting}_r101_novel_${shot}shot_seedx.yaml"
    dual_template="defrcn_${setting}_r101_novel_${shot}shot_seedx_dualfusion.yaml"
    root_generated="defrcn_${setting}_r101_novel_${shot}shot_seed${seed}.yaml"
    dual_generated="defrcn_${setting}_r101_novel_${shot}shot_seed${seed}_dualfusion.yaml"
  else
    root_template="defrcn_${setting}_r101_novelx_${shot}shot_seedx.yaml"
    dual_template="defrcn_${setting}_r101_novelx_${shot}shot_seedx_dualfusion.yaml"
    root_generated="defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml"
    dual_generated="defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_dualfusion.yaml"
  fi

  local tmp_dataset_root="${TMP_CFG_ROOT}/${DATASET}/${setting}/${shot}shot_seed${seed}"
  mkdir -p "${tmp_dataset_root}"

  cp "configs/${DATASET}/${root_template}" "${tmp_dataset_root}/${root_template}"

  if [[ "${DATASET}" == "voc" ]]; then
    create_cfg_extra=(--split "${SPLIT_ID}")
  fi

  python3 tools/create_config.py \
    --dataset "${DATASET_FLAG}" \
    --config_root "${tmp_dataset_root}" \
    --shot "${shot}" \
    --seed "${seed}" \
    --setting "${setting}" \
    "${create_cfg_extra[@]}"

  if [[ ! -f "${tmp_dataset_root}/${root_generated}" ]]; then
    echo "Failed to generate seed config: ${tmp_dataset_root}/${root_generated}" >&2
    return 1
  fi

  local tmp_baseline_cfg="${tmp_dataset_root}/dualFusionAblations/baseline/${dual_generated}"
  local tmp_ablation_cfg="${tmp_dataset_root}/dualFusionAblations/${ABLATION}/${dual_generated}"

  render_template_cfg "${BASELINE_ROOT}/${dual_template}" "${tmp_baseline_cfg}" "${seed}" "${DATASET}" "${SPLIT_ID}"

  if [[ "${ABLATION}" == "baseline" ]]; then
    echo "${tmp_baseline_cfg}"
    return 0
  fi

  render_template_cfg "${ABLATION_ROOT}/${dual_template}" "${tmp_ablation_cfg}" "${seed}" "${DATASET}" "${SPLIT_ID}"
  echo "${tmp_ablation_cfg}"
}

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

  if [[ "${SETTING}" == "fsod" || "${SETTING}" == "both" ]]; then
    python3 tools/model_surgery.py --dataset "${MODEL_SURGERY_DATASET}" --method remove \
      --src-path "${model_final}" --save-dir "${BASE_OUT_DIR}"
  fi

  if [[ "${SETTING}" == "gfsod" || "${SETTING}" == "both" ]]; then
    python3 tools/model_surgery.py --dataset "${MODEL_SURGERY_DATASET}" --method randinit \
      --src-path "${model_final}" --save-dir "${BASE_OUT_DIR}"
  fi
}

run_finetune_setting() {
  local setting="$1"
  local base_weight="$2"

  if [[ ! -f "${base_weight}" ]]; then
    echo "Missing base weight for ${setting}: ${base_weight}" >&2
    exit 1
  fi

  local res_dir="${SAVE_DIR}/${setting}"
  mkdir -p "${res_dir}"

  for shot in ${SHOTS}; do
    for seed in ${SEEDS}; do
      echo "[Fine-tune] setting=${setting} shot=${shot} seed=${seed}"
      cfg_path="$(generate_seed_cfg "${setting}" "${shot}" "${seed}")"
      out_dir="${res_dir}/${shot}shot_seed${seed}"
      python3 main.py --num-gpus "${NUM_GPUS}" --config-file "${cfg_path}" \
        --opts MODEL.WEIGHTS "${base_weight}" OUTPUT_DIR "${out_dir}" \
               TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
    done
  done

  python3 tools/extract_results.py --res-dir "${res_dir}" --shot-list ${SHOTS}
}

if [[ "${RUN_BASE}" == "1" ]]; then
  run_base_train
fi

prepare_surgery_weights

if [[ "${SETTING}" == "fsod" || "${SETTING}" == "both" ]]; then
  run_finetune_setting "fsod" "${BASE_OUT_DIR}/model_reset_remove.pth"
fi

if [[ "${SETTING}" == "gfsod" || "${SETTING}" == "both" ]]; then
  run_finetune_setting "gfsod" "${BASE_OUT_DIR}/model_reset_surgery.pth"
fi

echo "All runs completed. Results stored in ${SAVE_DIR}"
