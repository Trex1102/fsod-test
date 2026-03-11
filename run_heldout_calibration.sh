#!/usr/bin/env bash
# =============================================================================
# Held-Out Hyperparameter Calibration Runner
# =============================================================================
# Selects all PCB-FMA calibration hyperparameters on held-out base classes,
# then freezes them for all novel-class evaluations.
#
# Protocol:
#   1. Take 15 base classes → 3 folds of 5 pseudo-novel each
#   2. Grid search over (τ, w_d, w_φ, w_ψ, NPG margin, NPG suppression)
#   3. Select the config maximizing mean nAP50 across folds
#   4. Save selected params to JSON for use in all real evaluations
#
# Usage:
#   # Step 1: Generate evaluation configs
#   bash run_heldout_calibration.sh setup <split_id> [shot] [seed]
#
#   # Step 2: Run all evaluations (can be parallelized)
#   bash run_heldout_calibration.sh run <split_id> [shot] [seed]
#
#   # Step 3: Aggregate results and select best
#   bash run_heldout_calibration.sh aggregate <split_id> [shot] [seed]
#
#   # Or run everything:
#   bash run_heldout_calibration.sh all <split_id> [shot] [seed]
#
# Example:
#   bash run_heldout_calibration.sh all 1 5 0
# =============================================================================

set -e

ACTION=${1:-"all"}
SPLIT_ID=${2:-1}
SHOT=${3:-5}
SEED=${4:-0}
FAST=${5:-""}  # pass "fast" for reduced grid

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}
MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth

HELDOUT_DIR=checkpoints/voc/heldout_calibration
OUTPUT_JSON=${HELDOUT_DIR}/heldout_params_split${SPLIT_ID}.json
mkdir -p ${HELDOUT_DIR}

echo "=============================================="
echo "Held-Out Hyperparameter Calibration"
echo "=============================================="
echo "Action: ${ACTION}"
echo "Split: ${SPLIT_ID}, Shot: ${SHOT}, Seed: ${SEED}"
echo "=============================================="

case "${ACTION}" in
    setup)
        echo ">>> Generating held-out calibration configs..."
        FAST_FLAG=""
        if [ "${FAST}" = "fast" ]; then
            FAST_FLAG="--fast"
        fi

        python3 tools/heldout_calibration.py \
            --config-file configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma_enhanced_neg.yaml \
            --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
            --output ${OUTPUT_JSON} \
            --model-weights ${MODEL_WEIGHT} \
            --pcb-modelpath ${IMAGENET_PRETRAIN_TORCH} \
            ${FAST_FLAG}

        echo "Setup complete. See ${OUTPUT_JSON}"
        ;;

    run)
        echo ">>> Running held-out evaluations..."
        EVAL_SCRIPT=${OUTPUT_JSON%.json}_eval.sh
        if [ ! -f "${EVAL_SCRIPT}" ]; then
            echo "Error: Evaluation script not found at ${EVAL_SCRIPT}"
            echo "Run 'setup' first."
            exit 1
        fi
        bash ${EVAL_SCRIPT}
        echo "Evaluations complete."
        ;;

    aggregate)
        echo ">>> Aggregating results..."
        RESULTS_DIR=${HELDOUT_DIR}/split${SPLIT_ID}
        if [ ! -d "${RESULTS_DIR}" ]; then
            echo "Error: Results directory not found at ${RESULTS_DIR}"
            echo "Run 'run' first."
            exit 1
        fi

        python3 tools/heldout_calibration.py \
            --aggregate \
            --results-dir ${RESULTS_DIR} \
            --output ${HELDOUT_DIR}/heldout_params_final_split${SPLIT_ID}.json

        echo ""
        echo "Final selected parameters saved to:"
        echo "  ${HELDOUT_DIR}/heldout_params_final_split${SPLIT_ID}.json"
        ;;

    all)
        echo ">>> Running full held-out calibration pipeline..."
        bash $0 setup ${SPLIT_ID} ${SHOT} ${SEED} ${FAST}
        bash $0 run ${SPLIT_ID} ${SHOT} ${SEED}
        bash $0 aggregate ${SPLIT_ID} ${SHOT} ${SEED}
        ;;

    *)
        echo "Unknown action: ${ACTION}"
        echo "Available: setup, run, aggregate, all"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="

# Print usage instructions
if [[ "${ACTION}" == "all" || "${ACTION}" == "aggregate" ]]; then
    FINAL=${HELDOUT_DIR}/heldout_params_final_split${SPLIT_ID}.json
    if [ -f "${FINAL}" ]; then
        echo ""
        echo "To use the selected hyperparameters in evaluation:"
        echo ""
        echo "  # Read the selected params:"
        echo "  cat ${FINAL}"
        echo ""
        echo "  # Apply to evaluation (example):"
        echo "  python3 main.py --num-gpus 1 --eval-only \\"
        echo "      --config-file configs/voc/novelMethods/pcb_fma_enhanced_neg/... \\"
        echo "      --opts MODEL.WEIGHTS ... \\"
        echo "      NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE <selected_τ> \\"
        echo "      NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT <selected_w_d> \\"
        echo "      NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT <selected_w_φ> \\"
        echo "      NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT <selected_w_ψ> \\"
        echo "      NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN <selected_margin> \\"
        echo "      NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR <selected_supp>"
    fi
fi
