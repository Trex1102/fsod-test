#!/usr/bin/env bash
# =============================================================================
# Master Ablation Script for PCB-FMA Enhanced (NeurIPS 2025)
# =============================================================================
# Runs all ablation experiments for the paper:
#   component    - Progressive component addition       (Table: tab:component)
#   temperature  - Temperature sensitivity τ sweep      (Table: tab:temperature)
#   fm_arch      - FM architecture comparison           (Table: tab:fm_choice)
#   augmentation - Augmentation strategy                (Table: tab:augmentation)
#   weights      - Fusion weight sensitivity            (Table: tab:weights)
#
# All ablations use eval-only mode with pretrained vanilla DeFRCN checkpoints.
# PCB-FMA Enhanced is inference-only, so we just vary config overrides.
#
# Usage:
#   bash run_all_ablations.sh <split_id> [ablation] [shots] [seeds]
#
# Examples:
#   bash run_all_ablations.sh 1                              # All ablations, all shots
#   bash run_all_ablations.sh 1 temperature                  # Temperature sweep only
#   bash run_all_ablations.sh 1 component "1 2 3 5 10" "0"  # Component ablation
#   bash run_all_ablations.sh 1 all "1 5 10" "0"            # All ablations, 3 shots
# =============================================================================

set -e

SPLIT_ID=$1
ABLATION=${2:-"all"}
SEEDS=${4:-"0"}

# Per-ablation default shot lists (matching paper tables)
case "${ABLATION}" in
    fm_arch|weights)
        SHOTS=${3:-"1 5 10"}
        ;;
    *)
        SHOTS=${3:-"1 2 3 5 10"}
        ;;
esac

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_all_ablations.sh <split_id> [ablation] [shots] [seeds]"
    echo ""
    echo "Ablation types:"
    echo "  component    - Progressive component addition (FM → Aug → Comp → Guard)"
    echo "  temperature  - Temperature τ sweep (1.0, 0.5, 0.2, 0.1, 0.05, 0.01, raw)"
    echo "  fm_arch      - FM model comparison (DINOv2 S/B/L, DINOv1, CLIP)"
    echo "  augmentation - Aug strategy (none, flip, multicrop, flip+multicrop)"
    echo "  weights      - Fusion weight (w_d, w_φ, w_ψ) sweep"
    echo "  all          - Run everything (default)"
    exit 1
fi

# ---- Paths ----
SAVE_DIR=checkpoints/voc/ablations
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}

echo "=============================================="
echo "PCB-FMA Enhanced Ablation Studies"
echo "=============================================="
echo "Split:    ${SPLIT_ID}"
echo "Ablation: ${ABLATION}"
echo "Shots:    ${SHOTS}"
echo "Seeds:    ${SEEDS}"
echo "Save dir: ${SAVE_DIR}"
echo "=============================================="

# =========================================================================
# Helper: run eval with pcb_fma_enhanced config + arbitrary overrides
# =========================================================================
run_enhanced_eval() {
    local LABEL=$1; shift
    local SHOT=$1; shift
    local SEED=$1; shift
    local EXTRA_OPTS=("$@")

    # Generate seed-specific base config
    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

    # Copy template and fill in split/seed
    TEMPLATE=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma_enhanced.yaml
    CONFIG=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_pcb_fma_enhanced_ablation.yaml

    cp ${TEMPLATE} ${CONFIG}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG}
    sed -i "s/seedx/seed${SEED}/g" ${CONFIG}

    OUTPUT_DIR=${SAVE_DIR}/${LABEL}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}

    MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth
    if [ ! -f "${MODEL_WEIGHT}" ]; then
        echo "    Warning: ${MODEL_WEIGHT} not found, skipping."
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
        "${EXTRA_OPTS[@]}"

    # Cleanup temp configs
    rm -f ${CONFIG}
    rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
}

# =========================================================================
# Helper: run eval with pcb_fma_enhanced_neg config
# =========================================================================
run_enhanced_neg_eval() {
    local LABEL=$1; shift
    local SHOT=$1; shift
    local SEED=$1; shift
    local EXTRA_OPTS=("$@")

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

    TEMPLATE=configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novelx_${SHOT}shot_seedx_pcb_fma_enhanced_neg.yaml
    CONFIG=configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}_pcb_fma_enhanced_neg_ablation.yaml

    cp ${TEMPLATE} ${CONFIG}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG}
    sed -i "s/seedx/seed${SEED}/g" ${CONFIG}

    OUTPUT_DIR=${SAVE_DIR}/${LABEL}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}

    MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth
    if [ ! -f "${MODEL_WEIGHT}" ]; then
        echo "    Warning: ${MODEL_WEIGHT} not found, skipping."
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
        "${EXTRA_OPTS[@]}"

    rm -f ${CONFIG}
    rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
}

# =========================================================================
# Helper: run vanilla PCB eval (no novel methods)
# =========================================================================
run_vanilla_pcb_eval() {
    local LABEL=$1
    local SHOT=$2
    local SEED=$3

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

    CONFIG=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
    OUTPUT_DIR=${SAVE_DIR}/${LABEL}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}

    MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${SHOT}shot_seed${SEED}/model_final.pth
    if [ ! -f "${MODEL_WEIGHT}" ]; then
        echo "    Warning: ${MODEL_WEIGHT} not found, skipping."
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
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

    rm -f ${CONFIG}
}

# =========================================================================
# Helper: extract and print results
# =========================================================================
extract_results() {
    local DIR=$1
    local SHOTS_LIST=$2
    python3 tools/extract_results.py \
        --res-dir ${DIR} \
        --shot-list ${SHOTS_LIST} 2>/dev/null || true
}

# #########################################################################
# 1) COMPONENT ABLATION  (Table: tab:component)
# #########################################################################
# Row 1: Vanilla PCB (ResNet-101)           — no FM, no Aug, no Comp, no Guard
# Row 2: + FM (DINOv2 naive replacement)    — FM only, no aug, independent cosine
# Row 3: + Augmented prototypes             — FM + Aug, independent cosine
# Row 4: + Class-competitive scoring        — FM + Aug + Comp (= default Enhanced)
# Row 5: + Base confusion guard             — FM + Aug + Comp + Guard (= Enhanced+NPG)

if [[ "${ABLATION}" == "component" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo "########## COMPONENT ABLATION ##########"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  [${shot}-shot, seed ${seed}]"

            # Row 1: Vanilla PCB
            echo "    Row 1: Vanilla PCB"
            run_vanilla_pcb_eval "component/vanilla_pcb" ${shot} ${seed}

            # Row 2: +FM (no aug, raw cosine)
            echo "    Row 2: +FM"
            run_enhanced_eval "component/fm_only" ${shot} ${seed} \
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False \
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False \
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE raw

            # Row 3: +FM+Aug (aug enabled, raw cosine)
            echo "    Row 3: +FM+Aug"
            run_enhanced_eval "component/fm_aug" ${shot} ${seed} \
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE raw

            # Row 4: +FM+Aug+Comp (full Enhanced, default config)
            echo "    Row 4: +FM+Aug+Comp"
            run_enhanced_eval "component/fm_aug_comp" ${shot} ${seed}

            # Row 5: +FM+Aug+Comp+Guard (Enhanced + NPG)
            echo "    Row 5: +FM+Aug+Comp+Guard"
            run_enhanced_neg_eval "component/fm_aug_comp_guard" ${shot} ${seed}
        done
    done

    echo ""
    echo "--- Component Ablation Results ---"
    for label in vanilla_pcb fm_only fm_aug fm_aug_comp fm_aug_comp_guard; do
        echo "  ${label}:"
        extract_results "${SAVE_DIR}/component/${label}/split${SPLIT_ID}" "${SHOTS}"
    done
fi

# #########################################################################
# 2) TEMPERATURE SENSITIVITY  (Table: tab:temperature)
# #########################################################################
# τ = 1.0, 0.5, 0.2, 0.1 (default), 0.05, 0.01
# + Independent cosine baseline (COMPETITIVE_MODE=raw)

if [[ "${ABLATION}" == "temperature" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo "########## TEMPERATURE SENSITIVITY ##########"

    TEMPERATURES="1.0 0.5 0.2 0.1 0.05 0.01"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  [${shot}-shot, seed ${seed}]"

            # Independent cosine baseline
            echo "    τ=independent (raw cosine)"
            run_enhanced_eval "temperature/raw_cosine" ${shot} ${seed} \
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE raw

            # Temperature sweep
            for tau in ${TEMPERATURES}; do
                echo "    τ=${tau}"
                run_enhanced_eval "temperature/tau_${tau}" ${shot} ${seed} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE ${tau}
            done
        done
    done

    echo ""
    echo "--- Temperature Sensitivity Results ---"
    echo "  Independent cosine:"
    extract_results "${SAVE_DIR}/temperature/raw_cosine/split${SPLIT_ID}" "${SHOTS}"
    for tau in ${TEMPERATURES}; do
        echo "  τ=${tau}:"
        extract_results "${SAVE_DIR}/temperature/tau_${tau}/split${SPLIT_ID}" "${SHOTS}"
    done
fi

# #########################################################################
# 3) FM ARCHITECTURE  (Table: tab:fm_choice)
# #########################################################################
# ResNet-101 (ImageNet) = vanilla PCB baseline (already covered in component)
# DINOv2 ViT-S/14 (21M, 384d)
# DINOv2 ViT-B/14 (86M, 768d) — default
# DINOv2 ViT-L/14 (304M, 1024d)
# DINOv1 ViT-B/16 (86M, 768d)
# CLIP ViT-B/16 (86M, 512d) — requires open_clip or clip installed

if [[ "${ABLATION}" == "fm_arch" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo "########## FM ARCHITECTURE COMPARISON ##########"

    # model_name:feat_dim
    declare -a FM_MODELS=(
        "dinov2_vits14:384"
        "dinov2_vitb14:768"
        "dinov2_vitl14:1024"
        "dino_vitb16:768"
        "clip_vitb16:512"
    )

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  [${shot}-shot, seed ${seed}]"

            for model_spec in "${FM_MODELS[@]}"; do
                FM_NAME="${model_spec%%:*}"
                FM_DIM="${model_spec##*:}"
                echo "    ${FM_NAME} (dim=${FM_DIM})"

                run_enhanced_eval "fm_arch/${FM_NAME}" ${shot} ${seed} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME ${FM_NAME} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM ${FM_DIM}
            done
        done
    done

    echo ""
    echo "--- FM Architecture Results ---"
    for model_spec in "${FM_MODELS[@]}"; do
        FM_NAME="${model_spec%%:*}"
        echo "  ${FM_NAME}:"
        extract_results "${SAVE_DIR}/fm_arch/${FM_NAME}/split${SPLIT_ID}" "${SHOTS}"
    done
fi

# #########################################################################
# 4) AUGMENTATION STRATEGY  (Table: tab:augmentation)
# #########################################################################
# No augmentation       — AUG_FLIP=False, AUG_MULTICROP=False  (~1 view)
# Flip only             — AUG_FLIP=True,  AUG_MULTICROP=False  (~2 views)
# Multicrop only        — AUG_FLIP=False, AUG_MULTICROP=True   (~5 views)
# Flip + Multicrop      — AUG_FLIP=True,  AUG_MULTICROP=True   (~6 views, default)

if [[ "${ABLATION}" == "augmentation" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo "########## AUGMENTATION STRATEGY ##########"

    # label:flip:multicrop
    declare -a AUG_CONFIGS=(
        "no_aug:False:False"
        "flip_only:True:False"
        "multicrop_only:False:True"
        "flip_multicrop:True:True"
    )

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  [${shot}-shot, seed ${seed}]"

            for aug_spec in "${AUG_CONFIGS[@]}"; do
                IFS=':' read -r AUG_LABEL AUG_FLIP AUG_MC <<< "${aug_spec}"
                echo "    ${AUG_LABEL} (flip=${AUG_FLIP}, mc=${AUG_MC})"

                run_enhanced_eval "augmentation/${AUG_LABEL}" ${shot} ${seed} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP ${AUG_FLIP} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP ${AUG_MC}
            done
        done
    done

    echo ""
    echo "--- Augmentation Strategy Results ---"
    for aug_spec in "${AUG_CONFIGS[@]}"; do
        AUG_LABEL="${aug_spec%%:*}"
        echo "  ${AUG_LABEL}:"
        extract_results "${SAVE_DIR}/augmentation/${AUG_LABEL}/split${SPLIT_ID}" "${SHOTS}"
    done
fi

# #########################################################################
# 5) FUSION WEIGHT SENSITIVITY  (Table: tab:weights)
# #########################################################################
# (w_d, w_φ, w_ψ) — raw values, auto-normalized to sum=1
# Default: (0.4, 0.6, 0.3) → normalized (0.308, 0.462, 0.231)

if [[ "${ABLATION}" == "weights" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo "########## FUSION WEIGHT SENSITIVITY ##########"

    # label:det_w:fm_w:pcb_w
    declare -a WEIGHT_CONFIGS=(
        "wd03_wf05_wp02:0.3:0.5:0.2"
        "wd03_wf06_wp03:0.3:0.6:0.3"
        "wd04_wf06_wp03:0.4:0.6:0.3"
        "wd05_wf05_wp02:0.5:0.5:0.2"
        "wd05_wf03_wp02:0.5:0.3:0.2"
        "wd06_wf04_wp02:0.6:0.4:0.2"
    )

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  [${shot}-shot, seed ${seed}]"

            for wt_spec in "${WEIGHT_CONFIGS[@]}"; do
                IFS=':' read -r WT_LABEL DET_W FM_W PCB_W <<< "${wt_spec}"
                echo "    ${WT_LABEL} (w_d=${DET_W}, w_φ=${FM_W}, w_ψ=${PCB_W})"

                run_enhanced_eval "weights/${WT_LABEL}" ${shot} ${seed} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT ${DET_W} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT ${FM_W} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT ${PCB_W}
            done
        done
    done

    echo ""
    echo "--- Fusion Weight Results ---"
    for wt_spec in "${WEIGHT_CONFIGS[@]}"; do
        IFS=':' read -r WT_LABEL DET_W FM_W PCB_W <<< "${wt_spec}"
        echo "  (${DET_W}, ${FM_W}, ${PCB_W}):"
        extract_results "${SAVE_DIR}/weights/${WT_LABEL}/split${SPLIT_ID}" "${SHOTS}"
    done
fi

# =========================================================================
echo ""
echo "=============================================="
echo "Ablation complete!"
echo "=============================================="
echo "Results saved in: ${SAVE_DIR}/"
echo ""
echo "Directory structure:"
echo "  ablations/"
echo "    component/{vanilla_pcb,fm_only,fm_aug,fm_aug_comp,fm_aug_comp_guard}/"
echo "    temperature/{raw_cosine,tau_1.0,...,tau_0.01}/"
echo "    fm_arch/{dinov2_vits14,dinov2_vitb14,dinov2_vitl14,dino_vitb16,clip_vitb16}/"
echo "    augmentation/{no_aug,flip_only,multicrop_only,flip_multicrop}/"
echo "    weights/{wd03_wf05_wp02,...,wd06_wf04_wp02}/"
echo "=============================================="
