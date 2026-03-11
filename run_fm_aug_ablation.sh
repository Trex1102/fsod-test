#!/usr/bin/env bash
# =============================================================================
# FM + Augmentation Interaction Ablation
# =============================================================================
# Tests the interaction between FM architecture choice and augmentation strategy.
# Produces a grid: FM_model × Aug_strategy → nAP50
#
# This helps answer: does augmentation benefit all FMs equally, or is it
# more important for certain architectures?
#
# Usage:
#   bash run_fm_aug_ablation.sh <split_id> [shots] [seeds]
#
# Examples:
#   bash run_fm_aug_ablation.sh 1                      # Default: 1 5 10 shot, seed 0
#   bash run_fm_aug_ablation.sh 1 "1 2 3 5 10" "0"    # All shots
#   bash run_fm_aug_ablation.sh 1 "5" "0 1 2"          # 5-shot, 3 seeds
# =============================================================================

set -e

SPLIT_ID=$1
SHOTS=${2:-"1 5 10"}
SEEDS=${3:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_fm_aug_ablation.sh <split_id> [shots] [seeds]"
    echo ""
    echo "Runs a grid of FM models × augmentation strategies."
    echo ""
    echo "FM models:    dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dino_vitb16, clip_vitb16"
    echo "Aug configs:  no_aug, flip_only, multicrop_only, flip_multicrop"
    exit 1
fi

# ---- Paths ----
SAVE_DIR=checkpoints/voc/ablations/fm_aug_grid
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}

# FM models: name:dim
declare -a FM_MODELS=(
    "dinov2_vits14:384"
    "dinov2_vitb14:768"
    "dinov2_vitl14:1024"
    "dino_vitb16:768"
    "clip_vitb16:512"
)

# Aug configs: label:flip:multicrop
declare -a AUG_CONFIGS=(
    "no_aug:False:False"
    "flip_only:True:False"
    "multicrop_only:False:True"
    "flip_multicrop:True:True"
)

echo "=============================================="
echo "FM × Augmentation Interaction Ablation"
echo "=============================================="
echo "Split: ${SPLIT_ID}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "FM models: ${#FM_MODELS[@]}"
echo "Aug configs: ${#AUG_CONFIGS[@]}"
echo "Total configs: $(( ${#FM_MODELS[@]} * ${#AUG_CONFIGS[@]} ))"
echo "=============================================="

# ---- Helper: run a single eval ----
run_eval() {
    local LABEL=$1
    local SHOT=$2
    local SEED=$3
    local FM_NAME=$4
    local FM_DIM=$5
    local AUG_FLIP=$6
    local AUG_MC=$7

    # Generate seed-specific base config
    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

    TEMPLATE=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma_enhanced.yaml
    CONFIG=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_fm_aug_ablation.yaml

    cp ${TEMPLATE} ${CONFIG}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG}
    sed -i "s/seedx/seed${SEED}/g" ${CONFIG}

    OUTPUT_DIR=${SAVE_DIR}/split${SPLIT_ID}/${LABEL}/${SHOT}shot_seed${SEED}

    MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth
    if [ ! -f "${MODEL_WEIGHT}" ]; then
        echo "      Warning: ${MODEL_WEIGHT} not found, skipping."
        rm -f ${CONFIG}
        return
    fi

    python3 main.py \
        --num-gpus 1 \
        --eval-only \
        --config-file ${CONFIG} \
        --opts \
        MODEL.WEIGHTS ${MODEL_WEIGHT} \
        OUTPUT_DIR ${OUTPUT_DIR} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME ${FM_NAME} \
        NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM ${FM_DIM} \
        NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP ${AUG_FLIP} \
        NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP ${AUG_MC}

    rm -f ${CONFIG}
    rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
}

# ---- Main loop: FM × Aug grid ----
for model_spec in "${FM_MODELS[@]}"; do
    FM_NAME="${model_spec%%:*}"
    FM_DIM="${model_spec##*:}"

    echo ""
    echo ">>> FM: ${FM_NAME} (dim=${FM_DIM})"
    echo "--------------------------------------------"

    for aug_spec in "${AUG_CONFIGS[@]}"; do
        IFS=':' read -r AUG_LABEL AUG_FLIP AUG_MC <<< "${aug_spec}"

        echo "  Aug: ${AUG_LABEL} (flip=${AUG_FLIP}, mc=${AUG_MC})"

        for shot in ${SHOTS}; do
            for seed in ${SEEDS}; do
                echo "    [${shot}-shot, seed ${seed}]"
                run_eval "${FM_NAME}/${AUG_LABEL}" ${shot} ${seed} \
                    ${FM_NAME} ${FM_DIM} ${AUG_FLIP} ${AUG_MC}
            done
        done
    done
done

# ---- Summary ----
echo ""
echo "=============================================="
echo "FM × Aug Ablation Summary"
echo "=============================================="

for model_spec in "${FM_MODELS[@]}"; do
    FM_NAME="${model_spec%%:*}"
    echo ""
    echo "=== ${FM_NAME} ==="
    for aug_spec in "${AUG_CONFIGS[@]}"; do
        AUG_LABEL="${aug_spec%%:*}"
        echo "  ${AUG_LABEL}:"
        python3 tools/extract_results.py \
            --res-dir ${SAVE_DIR}/split${SPLIT_ID}/${FM_NAME}/${AUG_LABEL} \
            --shot-list ${SHOTS} 2>/dev/null || true
    done
done

echo ""
echo "=============================================="
echo "Results saved in: ${SAVE_DIR}/split${SPLIT_ID}/"
echo "=============================================="
