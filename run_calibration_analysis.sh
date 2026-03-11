#!/usr/bin/env bash
# =============================================================================
# Calibration Analysis Runner (Tables 4 & 5)
# =============================================================================
# Computes calibration metrics (ECE, Brier, Base→Novel FP) and prototype stats
# (Variance, Margin, Purity) for different methods.
#
# Usage:
#   bash run_calibration_analysis.sh <split_id> [shot] [seed] [methods]
#
# Examples:
#   bash run_calibration_analysis.sh 1 5 0
#   bash run_calibration_analysis.sh 1 5 0 "vanilla_pcb pcb_fma pcb_fma_enhanced_neg"
# =============================================================================

set -e

SPLIT_ID=$1
SHOT=${2:-5}
SEED=${3:-0}
METHODS=${4:-"vanilla_pcb pcb_fma_enhanced_neg"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_calibration_analysis.sh <split_id> [shot] [seed] [methods]"
    exit 1
fi

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}
MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth

SAVE_DIR=checkpoints/voc/calibration_analysis/split${SPLIT_ID}/${SHOT}shot_seed${SEED}
mkdir -p ${SAVE_DIR}

echo "=============================================="
echo "Calibration Analysis (Tables 4 & 5)"
echo "=============================================="
echo "Split: ${SPLIT_ID}, Shot: ${SHOT}, Seed: ${SEED}"
echo "Methods: ${METHODS}"
echo "Output: ${SAVE_DIR}/"
echo "=============================================="

for method in ${METHODS}; do
    echo ""
    echo ">>> Processing method: ${method}"
    METHOD_DIR=${SAVE_DIR}/${method}
    mkdir -p ${METHOD_DIR}

    # Determine config and overrides
    case ${method} in
        vanilla_pcb)
            # Generate config
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}
            CONFIG=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
            EXTRA_OPTS=""
            ;;
        pcb_fma)
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}
            TEMPLATE=configs/voc/novelMethods/pcb_fma/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma.yaml
            CONFIG=configs/voc/novelMethods/pcb_fma/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_pcb_fma.yaml
            cp ${TEMPLATE} ${CONFIG}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG}
            sed -i "s/seedx/seed${SEED}/g" ${CONFIG}
            EXTRA_OPTS=""
            ;;
        pcb_fma_enhanced)
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}
            TEMPLATE=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma_enhanced.yaml
            CONFIG=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_pcb_fma_enhanced.yaml
            cp ${TEMPLATE} ${CONFIG}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG}
            sed -i "s/seedx/seed${SEED}/g" ${CONFIG}
            EXTRA_OPTS=""
            ;;
        pcb_fma_enhanced_neg)
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}
            TEMPLATE=configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma_enhanced_neg.yaml
            CONFIG=configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_pcb_fma_enhanced_neg.yaml
            cp ${TEMPLATE} ${CONFIG}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG}
            sed -i "s/seedx/seed${SEED}/g" ${CONFIG}
            EXTRA_OPTS=""
            ;;
        *)
            echo "Unknown method: ${method}"
            continue
            ;;
    esac

    # --- Table 4: Calibration Metrics ---
    echo "  Computing calibration metrics (Table 4)..."
    python3 tools/compute_calibration_metrics.py \
        --config-file ${CONFIG} \
        --output ${METHOD_DIR}/calibration_metrics.json \
        --opts \
        MODEL.WEIGHTS ${MODEL_WEIGHT} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        OUTPUT_DIR ${METHOD_DIR}

    # --- Table 5: Prototype Statistics ---
    echo "  Computing prototype statistics (Table 5)..."
    python3 tools/compute_prototype_stats.py \
        --config-file ${CONFIG} \
        --output ${METHOD_DIR}/prototype_stats.json \
        --opts \
        MODEL.WEIGHTS ${MODEL_WEIGHT} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        OUTPUT_DIR ${METHOD_DIR}

    # Cleanup temp configs
    rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
    if [[ "${method}" != "vanilla_pcb" ]]; then
        rm -f ${CONFIG}
    fi

    echo "  Done: ${method}"
done

# Print summary table
echo ""
echo "=============================================="
echo "Results Summary"
echo "=============================================="
for method in ${METHODS}; do
    JSON=${SAVE_DIR}/${method}/calibration_metrics.json
    if [ -f "${JSON}" ]; then
        echo ""
        echo "--- ${method} ---"
        python3 -c "
import json
d = json.load(open('${JSON}'))
print(f\"  ECE:           {d['ece']:.4f}\")
print(f\"  Brier:         {d['brier']:.4f}\")
print(f\"  Base→Novel FP: {d['base_novel_fp_rate']:.4f}\")
"
    fi

    JSON=${SAVE_DIR}/${method}/prototype_stats.json
    if [ -f "${JSON}" ]; then
        python3 -c "
import json
d = json.load(open('${JSON}'))
a = d['aggregate']
print(f\"  Avg Variance:  {a['avg_variance']:.4f}\")
print(f\"  Avg Margin:    {a['avg_margin']:.4f}\")
print(f\"  Avg Purity:    {a['avg_purity']:.4f}\")
"
    fi
done
echo "=============================================="
