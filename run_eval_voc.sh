#!/usr/bin/env bash

set -euo pipefail

SPLIT_ID=${1:-}
SHOTS=${2:-"1 2 3 5 10"}
SEEDS=${3:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_eval_voc.sh <split_id> [shots] [seeds]"
    echo "Example: bash run_eval_voc.sh 1 \"1 2 3 5 10\" \"0\""
    exit 1
fi

EXP_NAME=vanilla_defrcn
SAVE_DIR=checkpoints/voc/${EXP_NAME}
EVAL_DIR=${SAVE_DIR}/eval/split${SPLIT_ID}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-${SAVE_DIR}}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
NUM_GPUS=${NUM_GPUS:-1}

mkdir -p "${EVAL_DIR}"

echo "=============================================="
echo "VOC eval-only runner"
echo "=============================================="
echo "Split: ${SPLIT_ID}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Checkpoint root: ${PRETRAINED_NOVEL_ROOT}"
echo "Eval output dir: ${EVAL_DIR}"
echo "=============================================="

for SHOT in ${SHOTS}; do
    for SEED in ${SEEDS}; do
        MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth
        if [ ! -f "${MODEL_WEIGHT}" ]; then
            echo "Error: missing checkpoint ${MODEL_WEIGHT}"
            echo "Set PRETRAINED_NOVEL_ROOT or provide an existing split/shot/seed."
            exit 1
        fi

        python3 tools/create_config.py --dataset voc --config_root configs/voc \
            --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

        CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
        OUTPUT_DIR=${EVAL_DIR}/${SHOT}shot_seed${SEED}
        mkdir -p "${OUTPUT_DIR}"

        python3 main.py --num-gpus ${NUM_GPUS} --eval-only --config-file "${CONFIG_PATH}" \
            --opts MODEL.WEIGHTS "${MODEL_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"

        rm -f "${CONFIG_PATH}"
    done
done

python3 tools/extract_results.py --res-dir "${EVAL_DIR}" --shot-list ${SHOTS}

echo "Eval summary written to ${EVAL_DIR}/results.txt"
