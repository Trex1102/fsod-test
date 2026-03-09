#!/usr/bin/env bash
# Run DP-GDL (Dynamic Gradient Decoupled Layer) on VOC
#
# This method replaces fixed GDL with learned gating, requiring retraining.
# Pipeline: base_train -> model_surgery -> novel_finetune -> PCB_eval
#
# Usage:
#   bash run_voc_dpgdl.sh <split_id> [stage]
#
# Stages:
#   base    - Train base model with dynamic GDL
#   novel   - Fine-tune on novel classes (1-shot, 10-shot)
#   all     - Run full pipeline
#
# Examples:
#   bash run_voc_dpgdl.sh 1 all
#   bash run_voc_dpgdl.sh 1 base
#   bash run_voc_dpgdl.sh 1 novel

set -euo pipefail

SPLIT_ID=${1:-}
STAGE=${2:-all}
SHOTS=${SHOTS:-"1 10"}
SEEDS=${SEEDS:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_dpgdl.sh <split_id> [stage]"
    echo "  stage: base, novel, all (default: all)"
    exit 1
fi

EXP_NAME=voc_dpgdl
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN}

BASE_DIR=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

echo "=============================================="
echo "DP-GDL: Dynamic Gradient Decoupled Layer"
echo "Split: ${SPLIT_ID}, Stage: ${STAGE}"
echo "Shots: ${SHOTS}, Seeds: ${SEEDS}"
echo "=============================================="

# Stage 1: Base Training with Dynamic GDL
if [ "${STAGE}" = "base" ] || [ "${STAGE}" = "all" ]; then
    echo ""
    echo ">>> Stage 1: Base training with dynamic GDL"

    python3 main.py --num-gpus 1 \
        --config-file configs/voc/dpgdl/defrcn_det_r101_base${SPLIT_ID}_dpgdl.yaml \
        --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
               OUTPUT_DIR ${BASE_DIR}

    # Model surgery for FSOD
    python3 tools/model_surgery.py --dataset voc --method remove \
        --src-path ${BASE_DIR}/model_final.pth \
        --save-dir ${BASE_DIR}

    echo "Base training complete. Weights saved to ${BASE_DIR}"
fi

# Stage 2: Novel Fine-tuning (gate is frozen)
if [ "${STAGE}" = "novel" ] || [ "${STAGE}" = "all" ]; then
    echo ""
    echo ">>> Stage 2: Novel fine-tuning (gate frozen)"

    BASE_WEIGHT=${BASE_DIR}/model_reset_remove.pth
    if [ ! -f "${BASE_WEIGHT}" ]; then
        echo "Error: Base weight not found at ${BASE_WEIGHT}"
        echo "Run base stage first: bash run_voc_dpgdl.sh ${SPLIT_ID} base"
        exit 1
    fi

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  Processing: ${shot}-shot, seed ${seed}"

            # Generate base config
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

            # Create DP-GDL novel config from template
            TEMPLATE=configs/voc/dpgdl/defrcn_fsod_r101_novelx_${shot}shot_seedx_dpgdl.yaml
            if [ ! -f "${TEMPLATE}" ]; then
                if [ "${shot}" -le 5 ]; then
                    TEMPLATE=configs/voc/dpgdl/defrcn_fsod_r101_novelx_1shot_seedx_dpgdl.yaml
                else
                    TEMPLATE=configs/voc/dpgdl/defrcn_fsod_r101_novelx_10shot_seedx_dpgdl.yaml
                fi
            fi

            CONFIG_PATH=configs/voc/dpgdl/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_dpgdl.yaml
            cp ${TEMPLATE} ${CONFIG_PATH}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG_PATH}
            sed -i "s/seedx/seed${seed}/g" ${CONFIG_PATH}

            OUTPUT_DIR=${SAVE_DIR}/split${SPLIT_ID}/${shot}shot_seed${seed}

            python3 main.py --num-gpus 1 \
                --config-file ${CONFIG_PATH} \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} \
                       OUTPUT_DIR ${OUTPUT_DIR} \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

            # Cleanup
            rm -f ${CONFIG_PATH}
            rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml

            echo "  Completed: ${shot}-shot, seed ${seed}"
        done
    done

    # Extract results
    python3 tools/extract_results.py --res-dir ${SAVE_DIR}/split${SPLIT_ID} --shot-list ${SHOTS}
fi

echo ""
echo "=============================================="
echo "DP-GDL pipeline complete!"
echo "Results: ${SAVE_DIR}"
echo "=============================================="
