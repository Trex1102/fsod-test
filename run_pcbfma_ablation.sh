#!/usr/bin/env bash
# PCB-FMA Ablation Studies
# 1) Component ablation: FM-only (no original PCB branch) vs tri-modal
# 2) Weight sensitivity sweep: different (w_d, w_phi, w_psi) combinations
#
# Usage:
#   bash run_pcbfma_ablation.sh <split_id> [shots] [seeds] [ablation_type]
#
# ablation_type:
#   component  - Run FM-only (bi-modal) ablation (Table III row 2)
#   weights    - Run weight sensitivity sweep (Table IV)
#   all        - Run both (default)
#
# Examples:
#   bash run_pcbfma_ablation.sh 1                          # All ablations, 1-shot, seed 0
#   bash run_pcbfma_ablation.sh 1 "1" "0" component        # FM-only ablation
#   bash run_pcbfma_ablation.sh 1 "1" "0" weights           # Weight sweep
#   bash run_pcbfma_ablation.sh 1 "1 2 3 5 10" "0" all     # All shots

set -e

SPLIT_ID=$1
SHOTS=${2:-"1"}
SEEDS=${3:-"0"}
ABLATION=${4:-"all"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_pcbfma_ablation.sh <split_id> [shots] [seeds] [ablation_type]"
    echo ""
    echo "ablation_type: component | weights | all (default)"
    echo ""
    echo "Component ablation runs:"
    echo "  - det + FM only (USE_ORIGINAL_PCB=False)"
    echo "  This fills Table III row 2 in the paper."
    echo ""
    echo "Weight sweep runs:"
    echo "  - Multiple (w_d, w_phi, w_psi) combos with USE_ORIGINAL_PCB=True"
    echo "  This fills Table IV in the paper."
    exit 1
fi

# Paths
SAVE_DIR=checkpoints/voc/voc_novel_methods/pcbfma_ablation
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}

echo "=============================================="
echo "PCB-FMA Ablation Studies"
echo "=============================================="
echo "Split: ${SPLIT_ID}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Ablation: ${ABLATION}"
echo "=============================================="

# Helper function to run a single evaluation with specific PCB-FMA config overrides
run_eval() {
    local LABEL=$1     # descriptive label for output dir
    local DET_W=$2     # DET_WEIGHT
    local FM_W=$3      # FM_WEIGHT
    local USE_PCB=$4   # USE_ORIGINAL_PCB (True/False)
    local PCB_W=$5     # ORIGINAL_PCB_WEIGHT
    local SHOT=$6
    local SEED=$7

    echo "  [${LABEL}] w_d=${DET_W}, w_phi=${FM_W}, use_pcb=${USE_PCB}, w_psi=${PCB_W}"

    # Generate seed-specific base config
    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

    # Template config
    TEMPLATE_CONFIG=configs/voc/novelMethods/pcb_fma/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma.yaml
    CONFIG_PATH=configs/voc/novelMethods/pcb_fma/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_pcb_fma_ablation.yaml

    cp ${TEMPLATE_CONFIG} ${CONFIG_PATH}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG_PATH}
    sed -i "s/seedx/seed${SEED}/g" ${CONFIG_PATH}

    OUTPUT_DIR=${SAVE_DIR}/split${SPLIT_ID}/${LABEL}/${SHOT}shot_seed${SEED}

    MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth
    if [ ! -f "${MODEL_WEIGHT}" ]; then
        echo "  Warning: Model weight not found at ${MODEL_WEIGHT}, skipping."
        rm -f ${CONFIG_PATH}
        return
    fi

    python3 main.py \
        --num-gpus 1 \
        --eval-only \
        --config-file ${CONFIG_PATH} \
        --opts \
        MODEL.WEIGHTS ${MODEL_WEIGHT} \
        OUTPUT_DIR ${OUTPUT_DIR} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        NOVEL_METHODS.PCB_FMA.DET_WEIGHT ${DET_W} \
        NOVEL_METHODS.PCB_FMA.FM_WEIGHT ${FM_W} \
        NOVEL_METHODS.PCB_FMA.USE_ORIGINAL_PCB ${USE_PCB} \
        NOVEL_METHODS.PCB_FMA.ORIGINAL_PCB_WEIGHT ${PCB_W}

    rm -f ${CONFIG_PATH}
    BASE_CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
    rm -f ${BASE_CONFIG_PATH}
}

# ======================================================================
# 1) COMPONENT ABLATION (Table III)
# ======================================================================
# Paper Table III has these rows:
#   det only              → vanilla PCB baseline (no PCB-FMA, already have this)
#   det + FM              → USE_ORIGINAL_PCB=False
#   det + PCB             → vanilla PCB (already have this as baseline = 49.54)
#   det + FM + PCB        → full tri-modal (already have this = 56.52)
#
# We only need to run: det + FM (bi-modal, no original PCB)

if [ "${ABLATION}" = "component" ] || [ "${ABLATION}" = "all" ]; then
    echo ""
    echo "=== Component Ablation ==="
    echo "Running: det + FM only (USE_ORIGINAL_PCB=False)"
    echo ""

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            # det + FM only (bi-modal): w_d=0.5, w_phi=0.5, no PCB
            run_eval "bimodal_det_fm" 0.5 0.5 False 0.0 ${shot} ${seed}
        done
    done

    echo ""
    echo "Component ablation results:"
    python3 tools/extract_results.py \
        --res-dir ${SAVE_DIR}/split${SPLIT_ID}/bimodal_det_fm \
        --shot-list ${SHOTS} 2>/dev/null || true
fi

# ======================================================================
# 2) WEIGHT SENSITIVITY SWEEP (Table IV)
# ======================================================================
# Paper Table IV: fix w_psi=0.2, vary w_d/w_phi
# Config values get normalized automatically, so we can pass raw values.
#
# Sweep configurations (DET_W, FM_W, PCB_W):
#   (0.7, 0.1, 0.2) → normalized to w_d=0.70, w_phi=0.10, w_psi=0.20
#   (0.6, 0.2, 0.2) → normalized to w_d=0.60, w_phi=0.20, w_psi=0.20
#   (0.5, 0.3, 0.2) → normalized to w_d=0.50, w_phi=0.30, w_psi=0.20  (default)
#   (0.4, 0.4, 0.2) → normalized to w_d=0.40, w_phi=0.40, w_psi=0.20
#   (0.3, 0.5, 0.2) → normalized to w_d=0.30, w_phi=0.50, w_psi=0.20
#
# Additional sweep: vary w_psi too
#   (0.5, 0.5, 0.0) → bi-modal (same as component ablation above)
#   (0.5, 0.4, 0.1)
#   (0.5, 0.3, 0.2) → default
#   (0.5, 0.2, 0.3)
#   (0.4, 0.3, 0.3)

if [ "${ABLATION}" = "weights" ] || [ "${ABLATION}" = "all" ]; then
    echo ""
    echo "=== Weight Sensitivity Sweep ==="
    echo ""

    # Weight configurations: label DET_W FM_W USE_PCB PCB_W
    declare -a WEIGHT_CONFIGS=(
        "wd07_wf01_wp02:0.7:0.1:True:0.2"
        "wd06_wf02_wp02:0.6:0.2:True:0.2"
        "wd05_wf03_wp02:0.5:0.3:True:0.2"
        "wd04_wf04_wp02:0.4:0.4:True:0.2"
        "wd03_wf05_wp02:0.3:0.5:True:0.2"
        "wd05_wf04_wp01:0.5:0.4:True:0.1"
        "wd05_wf02_wp03:0.5:0.2:True:0.3"
        "wd04_wf03_wp03:0.4:0.3:True:0.3"
        "wd03_wf04_wp03:0.3:0.4:True:0.3"
    )

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            for config_spec in "${WEIGHT_CONFIGS[@]}"; do
                IFS=':' read -r LABEL DET_W FM_W USE_PCB PCB_W <<< "${config_spec}"
                run_eval "weights/${LABEL}" ${DET_W} ${FM_W} ${USE_PCB} ${PCB_W} ${shot} ${seed}
            done
        done
    done

    echo ""
    echo "Weight sweep results:"
    for config_spec in "${WEIGHT_CONFIGS[@]}"; do
        IFS=':' read -r LABEL DET_W FM_W USE_PCB PCB_W <<< "${config_spec}"
        echo "--- ${LABEL} (w_d=${DET_W}, w_phi=${FM_W}, w_psi=${PCB_W}) ---"
        python3 tools/extract_results.py \
            --res-dir ${SAVE_DIR}/split${SPLIT_ID}/weights/${LABEL} \
            --shot-list ${SHOTS} 2>/dev/null || true
    done
fi

echo ""
echo "=============================================="
echo "Ablation complete!"
echo "=============================================="
echo "Results saved in: ${SAVE_DIR}/split${SPLIT_ID}/"
echo ""
echo "Directory structure:"
echo "  pcbfma_ablation/split${SPLIT_ID}/"
if [ "${ABLATION}" = "component" ] || [ "${ABLATION}" = "all" ]; then
    echo "    bimodal_det_fm/           <- det + FM only (no PCB)"
fi
if [ "${ABLATION}" = "weights" ] || [ "${ABLATION}" = "all" ]; then
    echo "    weights/"
    echo "      wd07_wf01_wp02/         <- w_d=0.7, w_phi=0.1, w_psi=0.2"
    echo "      wd06_wf02_wp02/         <- w_d=0.6, w_phi=0.2, w_psi=0.2"
    echo "      wd05_wf03_wp02/         <- w_d=0.5, w_phi=0.3, w_psi=0.2 (default)"
    echo "      wd04_wf04_wp02/         <- w_d=0.4, w_phi=0.4, w_psi=0.2"
    echo "      wd03_wf05_wp02/         <- w_d=0.3, w_phi=0.5, w_psi=0.2"
    echo "      wd05_wf04_wp01/         <- w_d=0.5, w_phi=0.4, w_psi=0.1"
    echo "      wd05_wf02_wp03/         <- w_d=0.5, w_phi=0.2, w_psi=0.3"
    echo "      wd04_wf03_wp03/         <- w_d=0.4, w_phi=0.3, w_psi=0.3"
    echo "      wd03_wf04_wp03/         <- w_d=0.3, w_phi=0.4, w_psi=0.3"
fi
echo "=============================================="
