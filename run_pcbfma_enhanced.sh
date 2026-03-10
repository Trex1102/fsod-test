#!/usr/bin/env bash
# PCB-FMA Enhanced Methods Runner
# ================================
# Runs three enhanced PCB-FMA methods (inference-only, no base training needed):
#
#   pcb_fma_patch     - Patch-level local matching (DN4-style) replacing CLS-only
#   neg_proto_guard   - Negative prototype guard (base-class false positive suppression)
#   pcb_fma_patch_neg - Combined: patch matching + negative guard
#
# All methods are eval-only and use pretrained vanilla DeFRCN novel checkpoints.
#
# Usage:
#   bash run_pcbfma_enhanced.sh <split_id> [method] [shots] [seeds]
#
# method:
#   pcb_fma_patch     - Patch-level local matching only
#   neg_proto_guard   - Negative prototype guard only
#   pcb_fma_patch_neg - Combined (patch + guard)
#   all               - Run all three (default)
#
# Examples:
#   bash run_pcbfma_enhanced.sh 1                                   # All methods, all shots, seed 0
#   bash run_pcbfma_enhanced.sh 1 pcb_fma_patch                     # Patch matching only
#   bash run_pcbfma_enhanced.sh 1 pcb_fma_patch_neg "1 10" "0"      # Combined, 1+10 shot
#   bash run_pcbfma_enhanced.sh 1 all "1 2 3 5 10" "0 1 2"          # All methods, all shots, 3 seeds

set -e

SPLIT_ID=$1
METHOD=${2:-"all"}
SHOTS=${3:-"1 2 3 5 10"}
SEEDS=${4:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_pcbfma_enhanced.sh <split_id> [method] [shots] [seeds]"
    echo ""
    echo "Methods:"
    echo "  pcb_fma_patch      Patch-level local FM matching (replaces CLS-only)"
    echo "  neg_proto_guard    Negative prototype guard (base-class FP suppression)"
    echo "  pcb_fma_patch_neg  Combined: patch matching + negative guard"
    echo "  all                Run all three (default)"
    echo ""
    echo "Examples:"
    echo "  bash run_pcbfma_enhanced.sh 1"
    echo "  bash run_pcbfma_enhanced.sh 1 pcb_fma_patch"
    echo "  bash run_pcbfma_enhanced.sh 1 pcb_fma_patch_neg \"1 10\" \"0\""
    echo "  bash run_pcbfma_enhanced.sh 1 all \"1 2 3 5 10\" \"0 1 2\""
    exit 1
fi

# ======================================================================
# Paths
# ======================================================================
EXP_NAME=voc_pcbfma_enhanced
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}

# ======================================================================
# Method registry
# ======================================================================
declare -A METHOD_DIRS
METHOD_DIRS["pcb_fma_patch"]="pcb_fma_patch"
METHOD_DIRS["neg_proto_guard"]="neg_proto_guard"
METHOD_DIRS["pcb_fma_patch_neg"]="pcb_fma_patch_neg"

declare -A METHOD_SUFFIXES
METHOD_SUFFIXES["pcb_fma_patch"]="pcb_fma_patch"
METHOD_SUFFIXES["neg_proto_guard"]="neg_proto_guard"
METHOD_SUFFIXES["pcb_fma_patch_neg"]="pcb_fma_patch_neg"

declare -A METHOD_DESCS
METHOD_DESCS["pcb_fma_patch"]="Patch-Level Local FM Matching"
METHOD_DESCS["neg_proto_guard"]="Negative Prototype Guard"
METHOD_DESCS["pcb_fma_patch_neg"]="Patch Matching + Negative Guard (Combined)"

# Determine which methods to run
if [ "${METHOD}" = "all" ]; then
    METHODS="pcb_fma_patch neg_proto_guard pcb_fma_patch_neg"
else
    if [ -z "${METHOD_DIRS[$METHOD]+_}" ]; then
        echo "Unknown method: ${METHOD}"
        echo "Available: pcb_fma_patch, neg_proto_guard, pcb_fma_patch_neg, all"
        exit 1
    fi
    METHODS="${METHOD}"
fi

echo "=============================================="
echo "PCB-FMA Enhanced Methods Runner"
echo "=============================================="
echo "Split:   ${SPLIT_ID}"
echo "Methods: ${METHODS}"
echo "Shots:   ${SHOTS}"
echo "Seeds:   ${SEEDS}"
echo "Save:    ${SAVE_DIR}"
echo "Novel weights: ${PRETRAINED_NOVEL_ROOT}"
echo "=============================================="

# ======================================================================
# Main loop
# ======================================================================
for method in ${METHODS}; do
    METHOD_DIR=${METHOD_DIRS[$method]}
    METHOD_SUFFIX=${METHOD_SUFFIXES[$method]}
    METHOD_DESC=${METHOD_DESCS[$method]}

    echo ""
    echo ">>> ${METHOD_DESC} (${method})"
    echo "---------------------------------------------------"

    METHOD_SAVE_DIR=${SAVE_DIR}/${METHOD_DIR}/split${SPLIT_ID}
    mkdir -p ${METHOD_SAVE_DIR}

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  [${method}] ${shot}-shot, seed ${seed}"

            # Generate seed-specific base config
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

            # Template config
            TEMPLATE_CONFIG=configs/voc/novelMethods/${METHOD_DIR}/defrcn_fsod_r101_novelx_${shot}shot_seedx_${METHOD_SUFFIX}.yaml
            CONFIG_PATH=configs/voc/novelMethods/${METHOD_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_${METHOD_SUFFIX}.yaml

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

    # Extract results for this method
    echo ""
    echo "  Results for ${method}:"
    python3 tools/extract_results.py --res-dir ${METHOD_SAVE_DIR} --shot-list ${SHOTS} 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "All enhanced methods completed!"
echo "=============================================="
echo "Results saved in: ${SAVE_DIR}/"
echo ""
echo "Directory structure:"
echo "  ${EXP_NAME}/"
for method in ${METHODS}; do
    echo "    ${METHOD_DIRS[$method]}/split${SPLIT_ID}/    <- ${METHOD_DESCS[$method]}"
done
echo ""
echo "Compare against baselines:"
echo "  Vanilla DeFRCN:  checkpoints/voc/vanilla_defrcn/"
echo "  PCB-FMA (CLS):   checkpoints/voc/voc_novel_methods/"
echo "=============================================="
