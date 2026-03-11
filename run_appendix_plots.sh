#!/usr/bin/env bash
# =============================================================================
# Appendix Visualization Runner (Appendices D, E, F)
# =============================================================================
# Generates all plots for the NeurIPS paper appendix:
#   D - Calibration and Reliability Plots
#   E - Feature-Space Geometry
#   F - Qualitative Visualization (t-SNE)
#
# Usage:
#   bash run_appendix_plots.sh <split_id> [shot] [seed] [section]
#
# Sections: D, E, F, all (default)
#
# Examples:
#   bash run_appendix_plots.sh 1 5 0          # All appendix sections
#   bash run_appendix_plots.sh 1 5 0 D        # Only reliability plots
#   bash run_appendix_plots.sh 1 5 0 F        # Only t-SNE
# =============================================================================

set -e

SPLIT_ID=$1
SHOT=${2:-5}
SEED=${3:-0}
SECTION=${4:-"all"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_appendix_plots.sh <split_id> [shot] [seed] [section]"
    echo "Sections: D (reliability), E (geometry), F (tsne), all"
    exit 1
fi

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}
MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth

OUTDIR=figures/appendix
mkdir -p ${OUTDIR}

echo "=============================================="
echo "Appendix Visualization Generator"
echo "=============================================="
echo "Split: ${SPLIT_ID}, Shot: ${SHOT}, Seed: ${SEED}"
echo "Section: ${SECTION}"
echo "Output: ${OUTDIR}/"
echo "=============================================="

# Helper: generate seed-specific config for a method
make_config() {
    local method=$1
    local suffix=$2
    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

    if [ "${method}" = "vanilla_pcb" ]; then
        echo "configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml"
        return
    fi

    local template="configs/voc/novelMethods/${method}/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_${suffix}.yaml"
    local config="configs/voc/novelMethods/${method}/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_${suffix}.yaml"
    cp ${template} ${config}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${config}
    sed -i "s/seedx/seed${SEED}/g" ${config}
    echo "${config}"
}

# ===== APPENDIX D: Calibration and Reliability Plots =====
if [[ "${SECTION}" == "D" || "${SECTION}" == "all" ]]; then
    echo ""
    echo ">>> Appendix D: Calibration and Reliability Plots"
    DDIR=${OUTDIR}/appendix_d
    mkdir -p ${DDIR}

    CALIB_DIR=checkpoints/voc/calibration_analysis/split${SPLIT_ID}/${SHOT}shot_seed${SEED}

    # Check if calibration metrics already exist (from run_calibration_analysis.sh)
    METHODS_D="vanilla_pcb pcb_fma_enhanced_neg"
    JSONS=""
    LABELS=""

    for method in ${METHODS_D}; do
        JSON=${CALIB_DIR}/${method}/calibration_metrics.json
        if [ ! -f "${JSON}" ]; then
            echo "  Calibration metrics not found for ${method}. Run run_calibration_analysis.sh first."
            echo "  Or computing now..."

            CONFIG=$(make_config ${method} ${method})
            mkdir -p ${CALIB_DIR}/${method}

            python3 tools/compute_calibration_metrics.py \
                --config-file ${CONFIG} \
                --output ${JSON} \
                --opts \
                MODEL.WEIGHTS ${MODEL_WEIGHT} \
                TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                OUTPUT_DIR ${CALIB_DIR}/${method}
        fi

        if [ -f "${JSON}" ]; then
            JSONS="${JSONS} ${JSON}"
            LABELS="${LABELS} ${method}"
        fi
    done

    if [ -n "${JSONS}" ]; then
        # Single plots
        for method in ${METHODS_D}; do
            JSON=${CALIB_DIR}/${method}/calibration_metrics.json
            if [ -f "${JSON}" ]; then
                python3 tools/generate_reliability_plots.py \
                    --input ${JSON} \
                    --output ${DDIR}/reliability_${method}_split${SPLIT_ID}_${SHOT}shot.pdf \
                    --title "${method} (Split ${SPLIT_ID}, ${SHOT}-shot)"
            fi
        done

        # Comparison plot (side by side)
        python3 tools/generate_reliability_plots.py \
            --input ${JSONS} \
            --labels ${LABELS} \
            --output ${DDIR}/reliability_comparison_split${SPLIT_ID}_${SHOT}shot.pdf
    fi

    echo "  Appendix D complete."
fi

# ===== APPENDIX E: Feature-Space Geometry =====
if [[ "${SECTION}" == "E" || "${SECTION}" == "all" ]]; then
    echo ""
    echo ">>> Appendix E: Feature-Space Geometry"
    EDIR=${OUTDIR}/appendix_e
    mkdir -p ${EDIR}

    CONFIG=$(make_config pcb_fma_enhanced_neg pcb_fma_enhanced_neg)

    python3 tools/compute_feature_geometry.py \
        --config-file ${CONFIG} \
        --output ${EDIR}/split${SPLIT_ID}_${SHOT}shot/ \
        --opts \
        MODEL.WEIGHTS ${MODEL_WEIGHT} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        OUTPUT_DIR ${EDIR}

    echo "  Appendix E complete."
fi

# ===== APPENDIX F: Qualitative Visualization (t-SNE) =====
if [[ "${SECTION}" == "F" || "${SECTION}" == "all" ]]; then
    echo ""
    echo ">>> Appendix F: t-SNE Feature Visualization"
    FDIR=${OUTDIR}/appendix_f
    mkdir -p ${FDIR}

    # Support + test features
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode support_test \
        --max-test-per-class 50 \
        --output ${FDIR}/tsne_novel_split${SPLIT_ID}_${SHOT}shot.pdf

    # ResNet-101 vs DINOv2 comparison
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode compare_fm \
        --max-test-per-class 50 \
        --output ${FDIR}/tsne_compare_split${SPLIT_ID}_${SHOT}shot.pdf

    # With base classes included
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode support_test \
        --include-base --max-base-per-class 20 \
        --max-test-per-class 30 \
        --output ${FDIR}/tsne_novel_base_split${SPLIT_ID}_${SHOT}shot.pdf

    echo "  Appendix F complete."
fi

echo ""
echo "=============================================="
echo "Appendix visualization complete!"
echo "Output files in: ${OUTDIR}/"
echo "=============================================="
ls -la ${OUTDIR}/appendix_*/ 2>/dev/null || echo "  (check subdirectories)"
echo "=============================================="
