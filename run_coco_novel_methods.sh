#!/usr/bin/env bash
# Run inference-only novel methods on saved COCO DeFRCN novel checkpoints.
#
# Usage:
#   bash run_coco_novel_methods.sh [methods] [shots] [seeds]
#
# Examples:
#   bash run_coco_novel_methods.sh
#   bash run_coco_novel_methods.sh all
#   bash run_coco_novel_methods.sh "pcb_fma_enhanced pcb_fma_enhanced_neg" "10 30" "0"

set -euo pipefail

METHOD=${1:-"pcb_resnet101 without_pcb"}
SHOTS=${2:-"30"}
SEEDS=${3:-"0"}

show_usage() {
    echo "Usage: bash run_coco_novel_methods.sh [methods] [shots] [seeds]"
    echo ""
    echo "Arguments:"
    echo "  methods : Method name(s) or 'all'"
    echo "            Available: without_pcb, pcb_resnet101, pcb_fma, pcb_fma_dino_only, pcb_fma_enhanced, pcb_fma_enhanced_noaug, pcb_fma_enhanced_dino_only, pcb_fma_enhanced_neg, pcb_fma_enhanced_neg_noaug, pcb_fma_enhanced_neg_dino_only"
    echo "  shots   : Shot settings (default: \"10 30\")"
    echo "  seeds   : Random seeds (default: \"0\")"
    echo ""
    echo "Environment variables:"
    echo "  EXP_NAME                 Output experiment root under checkpoints/coco/"
    echo "  SAVE_DIR                 Full output directory override"
    echo "  PRETRAINED_NOVEL_ROOT    Root containing saved COCO novel checkpoints"
    echo "  IMAGENET_PRETRAIN_TORCH  PCB backbone path"
    echo "  NUM_GPUS                 GPUs passed to main.py (default: 1)"
    echo "  SKIP_DONE                Skip runs with inference/res_final.json (default: 1)"
    echo ""
    echo "Examples:"
    echo "  bash run_coco_novel_methods.sh"
    echo "  bash run_coco_novel_methods.sh all"
    echo "  PRETRAINED_NOVEL_ROOT=checkpoints/coco/vanilla_defrcn/defrcn_fsod_r101_novel \\"
    echo "    bash run_coco_novel_methods.sh 'pcb_fma_enhanced_neg_dino_only' '10 30' '0'"
}

if [ "${METHOD}" = "-h" ] || [ "${METHOD}" = "--help" ]; then
    show_usage
    exit 0
fi

if [ "${METHOD}" = "all" ]; then
    METHODS="without_pcb pcb_fma pcb_fma_dino_only pcb_fma_enhanced pcb_fma_enhanced_noaug pcb_fma_enhanced_dino_only pcb_fma_enhanced_neg pcb_fma_enhanced_neg_noaug pcb_fma_enhanced_neg_dino_only"
else
    METHODS="${METHOD}"
fi

EXP_NAME=${EXP_NAME:-coco_novel_methods}
SAVE_DIR=${SAVE_DIR:-checkpoints/coco/${EXP_NAME}/pretrainedNovelEval}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/coco/vanilla_defrcn/defrcn_fsod_r101_novel}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
if [ ! -f "${IMAGENET_PRETRAIN_TORCH}" ] && [ -f "${IMAGENET_PRETRAIN_TORCH%.pth}" ]; then
    IMAGENET_PRETRAIN_TORCH="${IMAGENET_PRETRAIN_TORCH%.pth}"
fi
NUM_GPUS=${NUM_GPUS:-1}
SKIP_DONE=${SKIP_DONE:-1}

validate_method() {
    case "$1" in
        without_pcb|pcb_resnet101|pcb_fma|pcb_fma_dino_only|pcb_fma_enhanced|pcb_fma_enhanced_noaug|pcb_fma_enhanced_dino_only|pcb_fma_enhanced_neg|pcb_fma_enhanced_neg_noaug|pcb_fma_enhanced_neg_dino_only)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_model_weight() {
    local ckpt_dir="$1"
    local last_ckpt=""
    local fallback=""

    if [ ! -d "${ckpt_dir}" ]; then
        return 1
    fi

    if [ -f "${ckpt_dir}/model_final.pth" ]; then
        echo "${ckpt_dir}/model_final.pth"
        return 0
    fi

    if [ -f "${ckpt_dir}/last_checkpoint" ]; then
        last_ckpt=$(tr -d '\r\n' < "${ckpt_dir}/last_checkpoint")
        if [ -n "${last_ckpt}" ] && [ -f "${ckpt_dir}/${last_ckpt}" ]; then
            echo "${ckpt_dir}/${last_ckpt}"
            return 0
        fi
        if [ -n "${last_ckpt}" ] && [ -f "${last_ckpt}" ]; then
            echo "${last_ckpt}"
            return 0
        fi
    fi

    fallback=$(find "${ckpt_dir}" -maxdepth 1 -type f -name 'model_*.pth' | sort | tail -n 1 || true)
    if [ -n "${fallback}" ]; then
        echo "${fallback}"
        return 0
    fi

    return 1
}

build_method_opts() {
    local method="$1"

    METHOD_OPTS=()

    case "${method}" in
        without_pcb)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE False
                NOVEL_METHODS.ENABLE False
            )
            ;;
        pcb_resnet101)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                NOVEL_METHODS.ENABLE False
            )
            ;;
        pcb_fma)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma
                NOVEL_METHODS.PCB_FMA.ENABLE True
                NOVEL_METHODS.PCB_FMA.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA.FM_ONLY False
            )
            ;;
        pcb_fma_dino_only)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma
                NOVEL_METHODS.PCB_FMA.ENABLE True
                NOVEL_METHODS.PCB_FMA.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA.USE_ORIGINAL_PCB False
                NOVEL_METHODS.PCB_FMA.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA.FM_ONLY True
            )
            ;;
        pcb_fma_enhanced)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma_enhanced
                NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_ONLY False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_SCALES "[0.8,0.9]"
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_NUM 2
                NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
            )
            ;;
        pcb_fma_enhanced_noaug)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma_enhanced
                NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_ONLY False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_SCALES "[0.8,0.9]"
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_NUM 2
                NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
            )
            ;;
        pcb_fma_enhanced_dino_only)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma_enhanced
                NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB False
                NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_ONLY True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_SCALES "[0.8,0.9]"
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_NUM 2
                NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
            )
            ;;
        pcb_fma_enhanced_neg)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma_enhanced_neg
                NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_ONLY False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_SCALES "[0.8,0.9]"
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_NUM 2
                NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
                NOVEL_METHODS.NEG_PROTO_GUARD.ENABLE True
                NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05
                NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3
                NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
            )
            ;;
        pcb_fma_enhanced_neg_noaug)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma_enhanced_neg
                NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_ONLY False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_SCALES "[0.8,0.9]"
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_NUM 2
                NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
                NOVEL_METHODS.NEG_PROTO_GUARD.ENABLE True
                NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05
                NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3
                NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
            )
            ;;
        pcb_fma_enhanced_neg_dino_only)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE True
                NOVEL_METHODS.ENABLE True
                NOVEL_METHODS.METHOD pcb_fma_enhanced_neg
                NOVEL_METHODS.PCB_FMA_ENHANCED.ENABLE True
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_ENHANCED.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_ENHANCED.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_ENHANCED.USE_ORIGINAL_PCB False
                NOVEL_METHODS.PCB_FMA_ENHANCED.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_ENHANCED.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_ENHANCED.FM_ONLY True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_SCALES "[0.8,0.9]"
                NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP_NUM 2
                NOVEL_METHODS.PCB_FMA_ENHANCED.TEMPERATURE 0.1
                NOVEL_METHODS.PCB_FMA_ENHANCED.COMPETITIVE_MODE softmax
                NOVEL_METHODS.NEG_PROTO_GUARD.ENABLE True
                NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05
                NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3
                NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
            )
            ;;
        *)
            echo "Unknown method: ${method}" >&2
            return 1
            ;;
    esac
}

echo "=============================================="
echo "COCO Novel Methods Runner"
echo "=============================================="
echo "Methods: ${METHODS}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Save dir: ${SAVE_DIR}"
echo "Pretrained novel root: ${PRETRAINED_NOVEL_ROOT}"
echo "=============================================="

mkdir -p "${SAVE_DIR}"

for method in ${METHODS}; do
    if ! validate_method "${method}"; then
        echo "Unsupported method: ${method}" >&2
        exit 1
    fi
done

for shot in ${SHOTS}; do
    for seed in ${SEEDS}; do
        echo ""
        echo ">>> Preparing ${shot}-shot, seed ${seed}"

        python3 tools/create_config.py --dataset coco14 --config_root configs/coco \
            --shot "${shot}" --seed "${seed}" --setting fsod

        CONFIG_PATH="configs/coco/defrcn_fsod_r101_novel_${shot}shot_seed${seed}.yaml"
        CKPT_DIR="${PRETRAINED_NOVEL_ROOT}/${shot}shot_seed${seed}"

        if ! MODEL_WEIGHT=$(resolve_model_weight "${CKPT_DIR}"); then
            echo "[WARN] Missing checkpoint under ${CKPT_DIR}; skipping ${shot}-shot seed ${seed}."
            rm -f "${CONFIG_PATH}"
            continue
        fi

        for method in ${METHODS}; do
            OUTPUT_DIR="${SAVE_DIR}/${method}/${shot}shot_seed${seed}"

            if [ "${SKIP_DONE}" = "1" ] && [ -f "${OUTPUT_DIR}/inference/res_final.json" ]; then
                echo "[INFO] Skip ${method} ${shot}-shot seed ${seed} (already complete)"
                continue
            fi

            build_method_opts "${method}"

            echo "[INFO] Running ${method} for ${shot}-shot seed ${seed}"
            python3 main.py \
                --num-gpus "${NUM_GPUS}" \
                --eval-only \
                --config-file "${CONFIG_PATH}" \
                --opts \
                MODEL.WEIGHTS "${MODEL_WEIGHT}" \
                OUTPUT_DIR "${OUTPUT_DIR}" \
                "${METHOD_OPTS[@]}"
        done

        rm -f "${CONFIG_PATH}"
    done
done

for method in ${METHODS}; do
    if [ -d "${SAVE_DIR}/${method}" ]; then
        echo "[INFO] Extracting results for ${method}"
        python3 tools/extract_results.py --res-dir "${SAVE_DIR}/${method}" --shot-list ${SHOTS} 2>/dev/null || true
    fi
done

echo ""
echo "=============================================="
echo "Done. Results saved in ${SAVE_DIR}/"
echo "=============================================="
