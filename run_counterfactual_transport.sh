#!/usr/bin/env bash
# Direction 11: Foreground by Counterfactual Transport Runner
# =============================================================
# Runs counterfactual transport scoring (inference-only, no base training needed).
#
# Scores proposals by the gap between foreground and background transport costs.
# True objects: support foreground explains the proposal well (low cost) while
# support context/background explains it poorly (high cost). The detector is
# a counterfactual evidence ratio, not a similarity score.
#
# This is eval-only and uses pretrained vanilla DeFRCN novel checkpoints.
#
# Usage:
#   bash run_counterfactual_transport.sh <split_id> [shots] [seeds]
#
# Examples:
#   bash run_counterfactual_transport.sh 1                           # All shots, seed 0
#   bash run_counterfactual_transport.sh 1 "1 2 3 5 10" "0"         # All shots, seed 0
#   bash run_counterfactual_transport.sh 1 "1 10" "0 1 2"           # 1+10 shot, 3 seeds

set -e

SPLIT_ID=$1
SHOTS=${2:-"1 2 3 5 10"}
SEEDS=${3:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_counterfactual_transport.sh <split_id> [shots] [seeds]"
    echo ""
    echo "Examples:"
    echo "  bash run_counterfactual_transport.sh 1"
    echo "  bash run_counterfactual_transport.sh 1 \"1 2 3 5 10\" \"0\""
    echo "  bash run_counterfactual_transport.sh 1 \"1 10\" \"0 1 2\""
    exit 1
fi

# ======================================================================
# Paths
# ======================================================================
EXP_NAME=voc_counterfactual_transport
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}

echo "=============================================="
echo "Direction 11: Foreground by Counterfactual Transport"
echo "=============================================="
echo "Split:   ${SPLIT_ID}"
echo "Shots:   ${SHOTS}"
echo "Seeds:   ${SEEDS}"
echo "Save:    ${SAVE_DIR}"
echo "Novel weights: ${PRETRAINED_NOVEL_ROOT}"
echo "=============================================="

METHOD_SAVE_DIR=${SAVE_DIR}/split${SPLIT_ID}
mkdir -p ${METHOD_SAVE_DIR}

for shot in ${SHOTS}; do
    for seed in ${SEEDS}; do
        echo "  [counterfactual_transport] ${shot}-shot, seed ${seed}"

        # Generate seed-specific base config
        python3 tools/create_config.py --dataset voc --config_root configs/voc \
            --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

        # Template config
        TEMPLATE_CONFIG=configs/voc/novelMethods/counterfactual_transport/defrcn_fsod_r101_novelx_${shot}shot_seedx_counterfactual_transport.yaml
        CONFIG_PATH=configs/voc/novelMethods/counterfactual_transport/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_counterfactual_transport.yaml

        cp ${TEMPLATE_CONFIG} ${CONFIG_PATH}
        sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG_PATH}
        sed -i "s/seedx/seed${seed}/g" ${CONFIG_PATH}

        OUTPUT_DIR=${METHOD_SAVE_DIR}/${shot}shot_seed${seed}

        # Check pretrained novel weights exist
        MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth
        if [ ! -f "${MODEL_WEIGHT}" ]; then
            echo "  Warning: Model weight not found at ${MODEL_WEIGHT}, skipping."
            rm -f ${CONFIG_PATH}
            continue
        fi

        # Run eval-only inference
        python3 main.py \
            --num-gpus 1 \
            --eval-only \
            --config-file ${CONFIG_PATH} \
            --opts \
            MODEL.WEIGHTS ${MODEL_WEIGHT} \
            OUTPUT_DIR ${OUTPUT_DIR} \
            TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

        # Cleanup temporary configs
        rm -f ${CONFIG_PATH}
        BASE_CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        rm -f ${BASE_CONFIG_PATH}

        echo "  Completed: ${shot}-shot, seed ${seed}"
    done
done

# Extract results
echo ""
echo "  Results:"
python3 tools/extract_results.py --res-dir ${METHOD_SAVE_DIR} --shot-list ${SHOTS} 2>/dev/null || true

echo ""
echo "=============================================="
echo "Counterfactual Transport completed!"
echo "=============================================="
echo "Results saved in: ${SAVE_DIR}/"
echo ""
echo "Compare against baselines:"
echo "  Vanilla DeFRCN:  checkpoints/voc/vanilla_defrcn/"
echo "  PCB-FMA (CLS):   checkpoints/voc/voc_novel_methods/"
echo "=============================================="
