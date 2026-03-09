#!/usr/bin/env bash
# Foundation Model Ablation for PCB-FMA
# Runs PCB-FMA evaluation with different DINOv2 model variants (ViT-S/14, ViT-B/14, ViT-L/14)
# to study how foundation model capacity affects calibration quality.
#
# Usage:
#   bash run_fm_ablation.sh <split_id> [shots] [seeds]
#
# Examples:
#   bash run_fm_ablation.sh 1                          # Split 1, 1-shot, seed 0
#   bash run_fm_ablation.sh 1 "1 2 3 5 10" "0"        # All shots
#   bash run_fm_ablation.sh 1 "1" "0 1 2 3 4"         # Multiple seeds

set -e

SPLIT_ID=$1
SHOTS=${2:-"1"}
SEEDS=${3:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_fm_ablation.sh <split_id> [shots] [seeds]"
    exit 1
fi

# Paths
SAVE_DIR=checkpoints/voc/voc_novel_methods/fm_ablation
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}

# DINOv2 model variants: name, feature_dim
declare -a FM_MODELS=(
    "dinov2_vits14:384"
    "dinov2_vitb14:768"
    "dinov2_vitl14:1024"
)

echo "=============================================="
echo "Foundation Model Ablation - PCB-FMA"
echo "=============================================="
echo "Split: ${SPLIT_ID}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Models: ${FM_MODELS[*]}"
echo "=============================================="

for model_spec in "${FM_MODELS[@]}"; do
    FM_NAME="${model_spec%%:*}"
    FM_DIM="${model_spec##*:}"

    echo ""
    echo ">>> Model: ${FM_NAME} (dim=${FM_DIM})"
    echo "--------------------------------------------"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  Processing: ${shot}-shot, seed ${seed}, model ${FM_NAME}"

            # Generate seed-specific base config
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

            # Use the pcb_fma template config
            TEMPLATE_CONFIG=configs/voc/novelMethods/pcb_fma/defrcn_fsod_r101_novelx_${shot}shot_seedx_pcb_fma.yaml
            CONFIG_PATH=configs/voc/novelMethods/pcb_fma/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_pcb_fma_${FM_NAME}.yaml

            cp ${TEMPLATE_CONFIG} ${CONFIG_PATH}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG_PATH}
            sed -i "s/seedx/seed${seed}/g" ${CONFIG_PATH}

            # Output directory includes model name
            OUTPUT_DIR=${SAVE_DIR}/split${SPLIT_ID}/${FM_NAME}/${shot}shot_seed${seed}

            # Pretrained novel model weights
            MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth

            if [ ! -f "${MODEL_WEIGHT}" ]; then
                echo "  Warning: Model weight not found at ${MODEL_WEIGHT}, skipping."
                rm -f ${CONFIG_PATH}
                continue
            fi

            python3 main.py \
                --num-gpus 1 \
                --eval-only \
                --config-file ${CONFIG_PATH} \
                --opts \
                MODEL.WEIGHTS ${MODEL_WEIGHT} \
                OUTPUT_DIR ${OUTPUT_DIR} \
                TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                NOVEL_METHODS.PCB_FMA.FM_MODEL_NAME ${FM_NAME} \
                NOVEL_METHODS.PCB_FMA.FM_FEAT_DIM ${FM_DIM}

            # Cleanup temporary config
            rm -f ${CONFIG_PATH}
            BASE_CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            rm -f ${BASE_CONFIG_PATH}

            echo "  Completed: ${FM_NAME}, ${shot}-shot, seed ${seed}"
        done
    done

    echo "  Extracting results for ${FM_NAME}..."
    python3 tools/extract_results.py \
        --res-dir ${SAVE_DIR}/split${SPLIT_ID}/${FM_NAME} \
        --shot-list ${SHOTS} 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "FM Ablation Summary"
echo "=============================================="
echo "Results saved in: ${SAVE_DIR}/split${SPLIT_ID}/"
echo ""
echo "Model directories:"
for model_spec in "${FM_MODELS[@]}"; do
    FM_NAME="${model_spec%%:*}"
    echo "  ${FM_NAME}: ${SAVE_DIR}/split${SPLIT_ID}/${FM_NAME}/"
done
echo "=============================================="
