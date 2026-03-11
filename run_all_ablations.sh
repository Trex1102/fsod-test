#!/usr/bin/env bash
# =============================================================================
# Master Ablation Runner (Tables 6-10)
# =============================================================================
# Runs all ablation studies for the NeurIPS paper.
#
# Usage:
#   bash run_all_ablations.sh <split_id> [ablation] [shots] [seeds]
#
# Ablation types:
#   component   - Table 6:  Component ablation (vanilla→+FM→+Aug→+Comp→+Guard)
#   temperature - Table 7:  Temperature sensitivity (τ sweep)
#   fm_arch     - Table 8:  FM architecture comparison
#   augmentation - Table 9: Augmentation strategy
#   weights     - Table 10: Fusion weights
#   all         - Run all ablations
#
# Examples:
#   bash run_all_ablations.sh 1 all "5" "0"
#   bash run_all_ablations.sh 1 temperature "1 5 10" "0"
#   bash run_all_ablations.sh 1 component "5" "0 1 2"
# =============================================================================

set -euo pipefail

SPLIT_ID=$1
ABLATION=${2:-"all"}
SHOTS=${3:-"5"}
SEEDS=${4:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_all_ablations.sh <split_id> [ablation] [shots] [seeds]"
    exit 1
fi

# Validate ablation type
case "${ABLATION}" in
    component|temperature|fm_arch|augmentation|weights|all) ;;
    *)
        echo "Unknown ablation: ${ABLATION}"
        echo "Available: component, temperature, fm_arch, augmentation, weights, all"
        exit 1
        ;;
esac

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}
SAVE_DIR=checkpoints/voc/ablations
mkdir -p "${SAVE_DIR}"

echo "=============================================="
echo "Ablation Runner: ${ABLATION}"
echo "=============================================="
echo "Split: ${SPLIT_ID}, Shots: ${SHOTS}, Seeds: ${SEEDS}"
echo "Output: ${SAVE_DIR}/"
echo "=============================================="

# ---- Helper: run evaluation with PCB-FMA Enhanced ----
run_enhanced_eval() {
    local shot=$1 seed=$2 tag=$3 extra_opts=$4
    local model_weight=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth
    local out_dir=${SAVE_DIR}/${tag}/split${SPLIT_ID}/${shot}shot_seed${seed}
    mkdir -p "${out_dir}"

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

    local template=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novelx_${shot}shot_seedx_pcb_fma_enhanced.yaml
    local config=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_pcb_fma_enhanced.yaml
    cp ${template} ${config}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${config}
    sed -i "s/seedx/seed${seed}/g" ${config}

    python3 main.py --num-gpus 1 --eval-only \
        --config-file ${config} \
        --opts \
        MODEL.WEIGHTS ${model_weight} \
        OUTPUT_DIR ${out_dir} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        ${extra_opts}

    rm -f ${config}
    rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
}

# ---- Helper: run evaluation with PCB-FMA Enhanced + NPG ----
run_enhanced_neg_eval() {
    local shot=$1 seed=$2 tag=$3 extra_opts=$4
    local model_weight=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth
    local out_dir=${SAVE_DIR}/${tag}/split${SPLIT_ID}/${shot}shot_seed${seed}
    mkdir -p "${out_dir}"

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

    local template=configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novelx_${shot}shot_seedx_pcb_fma_enhanced_neg.yaml
    local config=configs/voc/novelMethods/pcb_fma_enhanced_neg/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_pcb_fma_enhanced_neg.yaml
    cp ${template} ${config}
    sed -i "s/novelx/novel${SPLIT_ID}/g" ${config}
    sed -i "s/seedx/seed${seed}/g" ${config}

    python3 main.py --num-gpus 1 --eval-only \
        --config-file ${config} \
        --opts \
        MODEL.WEIGHTS ${model_weight} \
        OUTPUT_DIR ${out_dir} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        ${extra_opts}

    rm -f ${config}
    rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
}

# ---- Helper: run vanilla PCB evaluation ----
run_vanilla_pcb_eval() {
    local shot=$1 seed=$2 tag=$3
    local model_weight=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth
    local out_dir=${SAVE_DIR}/${tag}/split${SPLIT_ID}/${shot}shot_seed${seed}
    mkdir -p "${out_dir}"

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

    local config=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml

    python3 main.py --num-gpus 1 --eval-only \
        --config-file ${config} \
        --opts \
        MODEL.WEIGHTS ${model_weight} \
        OUTPUT_DIR ${out_dir} \
        TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
        TEST.PCB_ENABLE True

    rm -f ${config}
}

# ---- Helper: extract results ----
extract_results() {
    local tag=$1
    local res_dir=${SAVE_DIR}/${tag}/split${SPLIT_ID}
    if [ -d "${res_dir}" ]; then
        python3 tools/extract_results.py --res-dir ${res_dir} --shot-list ${SHOTS}
    fi
}

# =====================================================================
# TABLE 6: Component Ablation
# =====================================================================
if [[ "${ABLATION}" == "component" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo ">>> Table 6: Component Ablation"
    echo "  Row 1: Vanilla PCB (ResNet-101)"
    echo "  Row 2: +FM (DINOv2 alignment)"
    echo "  Row 3: +Aug (support augmentation)"
    echo "  Row 4: +Comp (class-competitive similarity)"
    echo "  Row 5: +Guard (negative proto guard)"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo "  ${shot}-shot seed ${seed}..."

            # Row 1: Vanilla PCB
            run_vanilla_pcb_eval ${shot} ${seed} "component/1_vanilla_pcb"

            # Row 2: +FM only (no aug, raw cosine)
            run_enhanced_eval ${shot} ${seed} "component/2_plus_fm" \
                "NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE raw"

            # Row 3: +FM +Aug (augmentation, but raw cosine)
            run_enhanced_eval ${shot} ${seed} "component/3_plus_aug" \
                "NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE raw"

            # Row 4: +FM +Aug +Competitive (full enhanced, no guard)
            run_enhanced_eval ${shot} ${seed} "component/4_plus_comp" ""

            # Row 5: Full (enhanced + guard)
            run_enhanced_neg_eval ${shot} ${seed} "component/5_full" ""
        done
    done

    for row in 1_vanilla_pcb 2_plus_fm 3_plus_aug 4_plus_comp 5_full; do
        extract_results "component/${row}"
    done
fi

# =====================================================================
# TABLE 7: Temperature Sensitivity
# =====================================================================
if [[ "${ABLATION}" == "temperature" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo ">>> Table 7: Temperature Sensitivity"

    TEMPS="1.0 0.5 0.2 0.1 0.05 0.01"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            # Baseline: raw cosine (no temperature scaling)
            run_enhanced_eval ${shot} ${seed} "temperature/raw_cosine" \
                "NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE raw"

            for temp in ${TEMPS}; do
                tag="temperature/tau_${temp}"
                run_enhanced_eval ${shot} ${seed} "${tag}" \
                    "NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE ${temp}"
            done
        done
    done

    extract_results "temperature/raw_cosine"
    for temp in ${TEMPS}; do
        extract_results "temperature/tau_${temp}"
    done
fi

# =====================================================================
# TABLE 8: FM Architecture Comparison
# =====================================================================
if [[ "${ABLATION}" == "fm_arch" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo ">>> Table 8: FM Architecture Comparison"

    declare -A FM_DIMS
    FM_DIMS["dinov2_vits14"]=384
    FM_DIMS["dinov2_vitb14"]=768
    FM_DIMS["dinov2_vitl14"]=1024
    FM_DIMS["dino_vitb16"]=768
    FM_DIMS["clip_vitb16"]=512

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            for fm_model in dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dino_vitb16 clip_vitb16; do
                dim=${FM_DIMS[$fm_model]}
                tag="fm_arch/${fm_model}"
                run_enhanced_neg_eval ${shot} ${seed} "${tag}" \
                    "NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME ${fm_model} NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM ${dim} NOVEL_METHODS.NEG_PROTO_GUARD.FM_MODEL_NAME ${fm_model} NOVEL_METHODS.NEG_PROTO_GUARD.FM_FEAT_DIM ${dim}"
            done
        done
    done

    for fm_model in dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dino_vitb16 clip_vitb16; do
        extract_results "fm_arch/${fm_model}"
    done
fi

# =====================================================================
# TABLE 9: Augmentation Strategy
# =====================================================================
if [[ "${ABLATION}" == "augmentation" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo ">>> Table 9: Augmentation Strategy"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            # No augmentation
            run_enhanced_eval ${shot} ${seed} "augmentation/no_aug" \
                "NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False"

            # Flip only
            run_enhanced_eval ${shot} ${seed} "augmentation/flip_only" \
                "NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False"

            # Multicrop only
            run_enhanced_eval ${shot} ${seed} "augmentation/multicrop_only" \
                "NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True"

            # Flip + multicrop (default)
            run_enhanced_eval ${shot} ${seed} "augmentation/flip_multicrop" ""
        done
    done

    for aug in no_aug flip_only multicrop_only flip_multicrop; do
        extract_results "augmentation/${aug}"
    done
fi

# =====================================================================
# TABLE 10: Fusion Weights
# =====================================================================
if [[ "${ABLATION}" == "weights" || "${ABLATION}" == "all" ]]; then
    echo ""
    echo ">>> Table 10: Fusion Weights"

    # (w_d, w_phi, w_psi) tuples - unnormalized, will be normalized by the code
    WEIGHT_CONFIGS=(
        "0.5 0.5 0.0"    # det + FM only (no PCB)
        "0.4 0.6 0.0"    # det + FM only (heavier FM)
        "0.4 0.6 0.3"    # default tri-modal
        "0.3 0.5 0.3"    # balanced
        "0.3 0.7 0.3"    # FM-heavy
        "0.5 0.5 0.5"    # equal weights
    )

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            for wconfig in "${WEIGHT_CONFIGS[@]}"; do
                read -r wd wf wp <<< "${wconfig}"
                tag="weights/wd${wd}_wf${wf}_wp${wp}"

                use_pcb="True"
                if [ "${wp}" = "0.0" ]; then
                    use_pcb="False"
                fi

                run_enhanced_eval ${shot} ${seed} "${tag}" \
                    "NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT ${wd} NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT ${wf} NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT ${wp} NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB ${use_pcb}"
            done
        done
    done

    for wconfig in "${WEIGHT_CONFIGS[@]}"; do
        read -r wd wf wp <<< "${wconfig}"
        extract_results "weights/wd${wd}_wf${wf}_wp${wp}"
    done
fi

echo ""
echo "=============================================="
echo "All ablations complete!"
echo "Results in: ${SAVE_DIR}/"
echo "=============================================="
