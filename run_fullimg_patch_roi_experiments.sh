#!/usr/bin/env bash
set -euo pipefail

# Full-image DINOv2 patch-token RoI experiments stacked on base methods.
# Usage: bash run_fullimg_patch_roi_experiments.sh [split] [shots] [seeds]

source /home/bio/anaconda3/etc/profile.d/conda.sh
conda activate detectron2_03

SPLIT_ID=${1:-1}
SHOTS=${2:-"1 2 3 5"}
SEEDS=${3:-"0"}
EXP_ROOT=${EXP_ROOT:-checkpoints/voc/fullimg_patch_roi_onbase_experiments}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
FORCE=${FORCE:-0}

RUN_ROOT="${EXP_ROOT}/split${SPLIT_ID}"
QUEUE_LOG="${RUN_ROOT}/queue.log"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

run_one() {
    local label=$1
    local shot=$2
    local seed=$3
    shift 3
    local opts=("$@")

    local output_dir="${RUN_ROOT}/${label}/${shot}shot_seed${seed}"
    local log_file="${output_dir}/log.txt"
    if [ "${FORCE}" != "1" ] && [ -f "${log_file}" ] && grep -q "copypaste: AP,AP50" "${log_file}"; then
        echo "[$(date "+%F %T %z")] SKIP completed ${label} ${shot}shot seed${seed}"
        return 0
    fi

    echo "[$(date "+%F %T %z")] START ${label} ${shot}shot seed${seed}"
    mkdir -p "${output_dir}"

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot "${shot}" --seed "${seed}" --setting fsod --split "${SPLIT_ID}"

    local config_path="configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml"
    local model_weight="${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth"
    if [ ! -f "${model_weight}" ]; then
        echo "Missing model weight: ${model_weight}" >&2
        exit 1
    fi

    python3 main.py \
        --num-gpus 1 \
        --eval-only \
        --config-file "${config_path}" \
        --opts \
        MODEL.WEIGHTS "${model_weight}" \
        OUTPUT_DIR "${output_dir}" \
        TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}" \
        "${opts[@]}"

    rm -f "${config_path}"
    echo "[$(date "+%F %T %z")] DONE ${label} ${shot}shot seed${seed}"
}

run_label() {
    local label=$1
    shift
    local opts=("$@")
    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            run_one "${label}" "${shot}" "${seed}" "${opts[@]}"
        done
    done
    python3 tools/extract_results.py --res-dir "${RUN_ROOT}/${label}" --shot-list ${SHOTS} || true
}

run_variant() {
    local base_label=$1
    local variant_label=$2
    local method=$3
    local fm_only=$4
    local run_inner_first=$5
    local feature_mode=$6
    local pool_mode=$7
    shift 7
    local extra_opts=("$@")

    run_label "${base_label}/${variant_label}" \
        TEST.PCB_ENABLE True \
        NOVEL_METHODS.ENABLE True \
        NOVEL_METHODS.METHOD "${method}" \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.ENABLE True \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.FM_ONLY "${fm_only}" \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.RUN_INNER_FIRST "${run_inner_first}" \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.USE_ORIGINAL_PCB False \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.ORIGINAL_PCB_WEIGHT 0.0 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.DET_WEIGHT 0.4 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.FM_WEIGHT 0.6 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.FEATURE_MODE "${feature_mode}" \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.POOL_MODE "${pool_mode}" \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.CROP_CLS_WEIGHT 0.5 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.IMAGE_SIZE 518 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.CROP_SIZE 224 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.TEMPERATURE 0.1 \
        NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH.COMPETITIVE_MODE softmax \
        "${extra_opts[@]}"
}

enhanced_noaug_opts=(
    NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
    NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False
    NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False
)

npg_opts=(
    NOVEL_METHODS.NEG_PROTO_GUARD.ENABLE True
)

echo "============================================================"
echo "Full-image patch RoI experiments stacked on base methods"
echo "Started: $(date "+%F %T %z")"
echo "Split: ${SPLIT_ID}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Output: ${RUN_ROOT}"
echo "Matrix: 3 base methods x 2 feature modes x 5 pool modes x 2 guard settings"
echo "No standalone baseline-only rows are run."
echo "============================================================"

for pool in average roi_align center_weighted max attention; do
    for feature in patch hybrid; do
        variant="fullimg_${feature}_${pool}"

        # Base 1: vanilla_defrcn -> full-image patch/hybrid branch.
        run_variant "vanilla_defrcn" "${variant}" \
            pcb_fma_fullimg_patch True False "${feature}" "${pool}"
        run_variant "vanilla_defrcn" "${variant}_guard" \
            pcb_fma_fullimg_patch_neg True False "${feature}" "${pool}" \
            "${npg_opts[@]}"

        # Base 2: pcb_fma_no_aug -> full-image patch/hybrid branch.
        run_variant "pcb_fma_no_aug" "${variant}" \
            pcb_fma_enhanced_fullimg_patch False True "${feature}" "${pool}" \
            "${enhanced_noaug_opts[@]}"
        run_variant "pcb_fma_no_aug" "${variant}_guard" \
            pcb_fma_enhanced_fullimg_patch_neg False True "${feature}" "${pool}" \
            "${enhanced_noaug_opts[@]}" "${npg_opts[@]}"

        # Base 3: pcb_fma_neg_proto_guard_no_aug -> full-image patch/hybrid branch.
        run_variant "pcb_fma_neg_proto_guard_no_aug" "${variant}" \
            pcb_fma_enhanced_neg_fullimg_patch False True "${feature}" "${pool}" \
            "${enhanced_noaug_opts[@]}" "${npg_opts[@]}"
        run_variant "pcb_fma_neg_proto_guard_no_aug" "${variant}_guard" \
            pcb_fma_enhanced_neg_fullimg_patch_neg False True "${feature}" "${pool}" \
            "${enhanced_noaug_opts[@]}" "${npg_opts[@]}"
    done
done

echo "============================================================"
echo "Finished: $(date "+%F %T %z")"
echo "Results saved in ${RUN_ROOT}"
echo "============================================================"
