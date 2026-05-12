#!/usr/bin/env bash
set -euo pipefail

EXP_NAME=${1:-vanilla_defrcn}

SAVE_DIR=checkpoints/coco/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

SHOTS="1 2 3 5 10 30"
SEED=0
SKIP_BASE_TRAIN=${SKIP_BASE_TRAIN:-1}
BASE_NUM_GPUS=${BASE_NUM_GPUS:-1}
NOVEL_NUM_GPUS=${NOVEL_NUM_GPUS:-1}

BASE_DIR=${SAVE_DIR}/defrcn_det_r101_base
BASE_MODEL=${BASE_DIR}/model_final.pth
BASE_WEIGHT=${BASE_DIR}/model_reset_remove.pth


# ------------------------------- Base Pre-train ---------------------------------- #
if [[ "${SKIP_BASE_TRAIN}" != "1" ]]; then
    python3 main.py --num-gpus "${BASE_NUM_GPUS}" --config-file configs/coco/defrcn_det_r101_base.yaml \
        --opts MODEL.WEIGHTS "${IMAGENET_PRETRAIN}"                                     \
               OUTPUT_DIR "${BASE_DIR}"
elif [[ ! -f "${BASE_MODEL}" ]]; then
    echo "Missing base checkpoint: ${BASE_MODEL}" >&2
    echo "Run the base stage once or set SKIP_BASE_TRAIN=0." >&2
    exit 1
else
    echo "Skipping base pre-train and reusing ${BASE_MODEL}"
fi


# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset coco --method remove                     \
    --src-path "${BASE_MODEL}"                                                \
    --save-dir "${BASE_DIR}"


# ------------------------------ Novel Fine-tuning -------------------------------- #
for SHOT in ${SHOTS}
do
    python3 tools/create_config.py --dataset coco14 --config_root configs/coco \
        --shot "${SHOT}" --seed "${SEED}" --setting fsod

    CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${SHOT}shot_seed${SEED}.yaml
    OUTPUT_DIR=${SAVE_DIR}/defrcn_fsod_r101_novel/${SHOT}shot_seed${SEED}

    python3 main.py --num-gpus "${NOVEL_NUM_GPUS}" --config-file "${CONFIG_PATH}"        \
        --opts MODEL.WEIGHTS "${BASE_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}"                 \
               TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"

    rm -f "${CONFIG_PATH}"
done

mkdir -p "${SAVE_DIR}/defrcn_fsod_r101_novel"
python3 tools/extract_results.py --res-dir "${SAVE_DIR}/defrcn_fsod_r101_novel" --shot-list ${SHOTS}
