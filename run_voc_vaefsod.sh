#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_voc_vaefsod.sh <EXP_NAME> <SPLIT_ID>
# Example:
#   bash run_voc_vaefsod.sh voc_vaefsod_split1 1

EXP_NAME=${1:-}
SPLIT_ID=${2:-}

if [ -z "${EXP_NAME}" ] || [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_vaefsod.sh <EXP_NAME> <SPLIT_ID>"
    exit 1
fi

SAVE_DIR=checkpoints/voc/${EXP_NAME}

# You may override these with env vars before running the script.
IMAGENET_PRETRAIN=${IMAGENET_PRETRAIN:-.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}

# Keep defaults small for a single-GPU workflow.
NUM_GPUS=${NUM_GPUS:-1}
SHOTS=${SHOTS:-"1 2 3"}
SEEDS=${SEEDS:-"0"}

# Stage toggles (1=run, 0=skip).
RUN_BASE=${RUN_BASE:-1}
RUN_VAE_TRAIN=${RUN_VAE_TRAIN:-1}
RUN_FSOD=${RUN_FSOD:-1}

# --------------------------- Batch-size-safe defaults ---------------------------
# Original VOC base config is tuned around IMS_PER_BATCH=16.
# For IMS_PER_BATCH=8, use linear LR scaling + doubled iterations/steps.
BASE_IMS_PER_BATCH=${BASE_IMS_PER_BATCH:-8}
BASE_LR=${BASE_LR:-0.01}
BASE_MAX_ITER=${BASE_MAX_ITER:-30000}
BASE_STEPS=${BASE_STEPS:-"(20000,26600)"}
BASE_WARMUP_ITERS=${BASE_WARMUP_ITERS:-100}

# FSOD stage defaults for batch size 8 (scaled from 16-batch templates).
FSOD_IMS_PER_BATCH=${FSOD_IMS_PER_BATCH:-8}
FSOD_BASE_LR=${FSOD_BASE_LR:-0.005}

# VAE training/generation knobs.
VAE_TRAIN_BATCH_SIZE=${VAE_TRAIN_BATCH_SIZE:-256}
VAE_TRAIN_EPOCHS=${VAE_TRAIN_EPOCHS:-20}
VAE_NUM_GEN_PER_CLASS=${VAE_NUM_GEN_PER_CLASS:-30}
VAE_AUX_BATCH_SIZE=${VAE_AUX_BATCH_SIZE:-64}

if [ ! -f "${IMAGENET_PRETRAIN}" ]; then
    echo "Missing IMAGENET_PRETRAIN: ${IMAGENET_PRETRAIN}"
    exit 1
fi
if [ ! -f "${IMAGENET_PRETRAIN_TORCH}" ]; then
    echo "Missing IMAGENET_PRETRAIN_TORCH: ${IMAGENET_PRETRAIN_TORCH}"
    exit 1
fi

BASE_CFG=configs/voc/vaeFsod/defrcn_det_r101_base${SPLIT_ID}_vaefsod.yaml
if [ ! -f "${BASE_CFG}" ]; then
    echo "Missing base VAE config: ${BASE_CFG}"
    exit 1
fi

BASE_STAGE_DIR=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
VAE_DIR=${SAVE_DIR}/vae_model
VAE_CKPT=${VAE_DIR}/model_final.pth
FEATURE_BANK_DIR=${SAVE_DIR}/feature_banks
FSOD_SAVE_DIR=${SAVE_DIR}/fsod

mkdir -p "${BASE_STAGE_DIR}" "${VAE_DIR}" "${FEATURE_BANK_DIR}" "${FSOD_SAVE_DIR}"

get_fsod_schedule() {
    # Returns: <max_iter> <step_iter>
    local shot="$1"
    if [ "${FSOD_IMS_PER_BATCH}" = "8" ]; then
        case "${shot}" in
            1) echo "1600 1280" ;;
            2) echo "2400 1920" ;;
            3) echo "3200 2560" ;;
            5) echo "4000 3200" ;;
            10) echo "8000 6400" ;;
            *)
                echo "Unsupported shot for VOC FSOD: ${shot}"
                return 1
                ;;
        esac
    elif [ "${FSOD_IMS_PER_BATCH}" = "16" ]; then
        case "${shot}" in
            1) echo "800 640" ;;
            2) echo "1200 960" ;;
            3) echo "1600 1280" ;;
            5) echo "2000 1600" ;;
            10) echo "4000 3200" ;;
            *)
                echo "Unsupported shot for VOC FSOD: ${shot}"
                return 1
                ;;
        esac
    else
        echo "FSOD_IMS_PER_BATCH must be 8 or 16 for this script. Got: ${FSOD_IMS_PER_BATCH}"
        return 1
    fi
}

echo "[1/5] Base pre-train (optional)"
if [ "${RUN_BASE}" = "1" ]; then
    python3 main.py --num-gpus "${NUM_GPUS}" --config-file "${BASE_CFG}" \
        --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}" \
               OUTPUT_DIR "${BASE_STAGE_DIR}" \
               SOLVER.IMS_PER_BATCH "${BASE_IMS_PER_BATCH}" \
               SOLVER.BASE_LR "${BASE_LR}" \
               SOLVER.MAX_ITER "${BASE_MAX_ITER}" \
               SOLVER.STEPS "${BASE_STEPS}" \
               SOLVER.WARMUP_ITERS "${BASE_WARMUP_ITERS}"
fi

if [ ! -f "${BASE_STAGE_DIR}/model_final.pth" ]; then
    echo "Missing base checkpoint: ${BASE_STAGE_DIR}/model_final.pth"
    echo "Run with RUN_BASE=1 or provide existing checkpoint."
    exit 1
fi

echo "[2/5] Model surgery"
python3 tools/model_surgery.py --dataset voc --method remove \
    --src-path "${BASE_STAGE_DIR}/model_final.pth" \
    --save-dir "${BASE_STAGE_DIR}"
BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_remove.pth

echo "[3/5] Train Norm-VAE (optional)"
if [ "${RUN_VAE_TRAIN}" = "1" ]; then
    python3 tools/train_vae_fsod.py \
        --config-file "${BASE_CFG}" \
        --weights "${BASE_STAGE_DIR}/model_final.pth" \
        --output "${VAE_CKPT}" \
        --opts MODEL.VAE_FSOD.TRAIN_BATCH_SIZE "${VAE_TRAIN_BATCH_SIZE}" \
               MODEL.VAE_FSOD.TRAIN_EPOCHS "${VAE_TRAIN_EPOCHS}"
fi

if [ ! -f "${VAE_CKPT}" ]; then
    echo "Missing VAE checkpoint: ${VAE_CKPT}"
    echo "Run with RUN_VAE_TRAIN=1 or provide existing VAE ckpt."
    exit 1
fi

echo "[4/5] FSOD with VAE feature generation (optional)"
if [ "${RUN_FSOD}" = "1" ]; then
    for shot in ${SHOTS}; do
        read -r fsod_max_iter fsod_step_iter < <(get_fsod_schedule "${shot}")

        for seed in ${SEEDS}; do
            echo "---- shot=${shot}, seed=${seed} ----"

            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot "${shot}" --seed "${seed}" --setting fsod --split "${SPLIT_ID}"

            BASE_CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml

            VAE_TEMPLATE=configs/voc/vaeFsod/defrcn_fsod_r101_novelx_${shot}shot_seedx_vaefsod.yaml
            if [ ! -f "${VAE_TEMPLATE}" ]; then
                echo "Missing VAE template: ${VAE_TEMPLATE}"
                exit 1
            fi

            VAE_CONFIG_PATH=configs/voc/vaeFsod/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_vaefsod.yaml
            cp "${VAE_TEMPLATE}" "${VAE_CONFIG_PATH}"
            sed -i "s/novelx/novel${SPLIT_ID}/g" "${VAE_CONFIG_PATH}"
            sed -i "s/seedx/seed${seed}/g" "${VAE_CONFIG_PATH}"

            BANK_PATH=${FEATURE_BANK_DIR}/fsod_${shot}shot_seed${seed}.pth
            OUTPUT_DIR=${FSOD_SAVE_DIR}/${shot}shot_seed${seed}

            python3 tools/generate_vae_fsod_features.py \
                --config-file "${VAE_CONFIG_PATH}" \
                --vae-ckpt "${VAE_CKPT}" \
                --output "${BANK_PATH}" \
                --opts MODEL.VAE_FSOD.NUM_GEN_PER_CLASS "${VAE_NUM_GEN_PER_CLASS}"

            python3 main.py --num-gpus "${NUM_GPUS}" --config-file "${VAE_CONFIG_PATH}" \
                --opts MODEL.WEIGHTS "${BASE_WEIGHT}" \
                       OUTPUT_DIR "${OUTPUT_DIR}" \
                       TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}" \
                       MODEL.VAE_FSOD.FEATURE_BANK_PATH "${BANK_PATH}" \
                       MODEL.VAE_FSOD.AUX_BATCH_SIZE "${VAE_AUX_BATCH_SIZE}" \
                       SOLVER.IMS_PER_BATCH "${FSOD_IMS_PER_BATCH}" \
                       SOLVER.BASE_LR "${FSOD_BASE_LR}" \
                       SOLVER.MAX_ITER "${fsod_max_iter}" \
                       SOLVER.STEPS "(${fsod_step_iter},)"

            rm -f "${VAE_CONFIG_PATH}" "${BASE_CONFIG_PATH}" "${OUTPUT_DIR}/model_final.pth"
        done
    done
fi

echo "[5/5] Summarize"
python3 tools/extract_results.py --res-dir "${FSOD_SAVE_DIR}" --shot-list ${SHOTS}

echo "Done. Results: ${FSOD_SAVE_DIR}/results.txt"
