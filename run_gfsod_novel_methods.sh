#!/usr/bin/env bash
# Run inference-only novel methods on saved VOC GFSOD DeFRCN checkpoints.
#
# Usage:
#   bash run_gfsod_novel_methods.sh <split_id> [methods] [shots] [seeds] [run_mode]
#
# Examples:
#   bash run_gfsod_novel_methods.sh 1
#   bash run_gfsod_novel_methods.sh 1 all "1 2 3 5 10" "0 1"
#   bash run_gfsod_novel_methods.sh 1 "without_pcb pcb_fma_enhanced_dino_only" "5" "0 1"
#   DRY_RUN=1 bash run_gfsod_novel_methods.sh 1 "pcb_resnet101 pcb_fma_enhanced_noaug" "5" "0 1"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

ALL_METHODS="without_pcb pcb_resnet101 pcb_fma pcb_fma_dino_only pcb_fma_enhanced pcb_fma_enhanced_noaug pcb_fma_enhanced_dino_only pcb_fma_enhanced_neg pcb_fma_enhanced_neg_noaug pcb_fma_enhanced_neg_dino_only"

show_usage() {
    echo "Usage: bash run_gfsod_novel_methods.sh <split_id> [methods] [shots] [seeds] [run_mode]"
    echo ""
    echo "Arguments:"
    echo "  split_id  : VOC split (1, 2, or 3)"
    echo "  methods   : Method name(s) or 'all'"
    echo "              Available: ${ALL_METHODS}"
    echo "  shots     : Shot settings (default: \"1 2 3 5 10\")"
    echo "  seeds     : Random seeds (default: \"0 1\")"
    echo "  run_mode  : infer_pretrained_gfsod | eval_pretrained_gfsod | pretrained_gfsod"
    echo ""
    echo "Environment variables:"
    echo "  EXP_NAME                 Output experiment root under checkpoints/voc/"
    echo "  SAVE_DIR                 Full output directory override"
    echo "  PRETRAINED_GFSOD_ROOT    Root containing saved GFSOD checkpoints"
    echo "                           Default: checkpoints/voc/vanilla_defrcn/defrcn_gfsod_r101_novel<split>"
    echo "  IMAGENET_PRETRAIN_TORCH  PCB backbone path"
    echo "  BASE_WEIGHT_DIR          Base model dir for meta_pcb calibrator training"
    echo "  META_PCB_EPISODES        Meta-training episodes (default: 10000)"
    echo "  META_PCB_N_WAY           N-way episodes (default: 5)"
    echo "  META_PCB_K_SHOT          K-shot episodes (default: 1)"
    echo "  NUM_GPUS                 GPUs passed to main.py (default: 1)"
    echo "  SKIP_DONE                Skip completed runs with inference/res_final.json (default: 1)"
    echo "  DRY_RUN                  Print commands without running main.py (default: 0)"
    echo ""
    echo "Examples:"
    echo "  bash run_gfsod_novel_methods.sh 1"
    echo "  bash run_gfsod_novel_methods.sh 1 all \"1 2 3 5 10\" \"0 1\""
    echo "  bash run_gfsod_novel_methods.sh 1 \"pcb_resnet101 pcb_fma_enhanced_dino_only\" \"5\" \"0 1\""
    echo "  DRY_RUN=1 bash run_gfsod_novel_methods.sh 1 \"without_pcb pcb_resnet101\" \"5\" \"0 1\""
}

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_usage
    exit 0
fi

SPLIT_ID=$1
METHOD=${2:-all}
SHOTS=${3:-"1 2 3 5 10"}
SEEDS=${4:-"2 3"}
RUN_MODE=${5:-infer_pretrained_gfsod}

case "${SPLIT_ID}" in
    1|2|3)
        ;;
    *)
        echo "Invalid split_id: ${SPLIT_ID}" >&2
        show_usage
        exit 1
        ;;
esac

case "${RUN_MODE}" in
    infer_pretrained_gfsod|eval_pretrained_gfsod|pretrained_gfsod)
        RUN_MODE="infer_pretrained_gfsod"
        ;;
    *)
        echo "Unsupported run_mode: ${RUN_MODE}" >&2
        echo "Only eval-only GFSOD checkpoint inference is supported here." >&2
        exit 1
        ;;
esac

if [ "${METHOD}" = "all" ]; then
    METHODS="${ALL_METHODS}"
else
    METHODS="${METHOD}"
fi

EXP_NAME=${EXP_NAME:-voc_gfsod_novel_methods}
SAVE_DIR=${SAVE_DIR:-checkpoints/voc/${EXP_NAME}/pretrainedGfsodEval}
DEFAULT_GFSOD_ROOT="checkpoints/voc/vanilla_defrcn/defrcn_gfsod_r101_novel${SPLIT_ID}"
PRETRAINED_GFSOD_ROOT=${PRETRAINED_GFSOD_ROOT:-${DEFAULT_GFSOD_ROOT}}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
BASE_WEIGHT_DIR=${BASE_WEIGHT_DIR:-checkpoints/voc/vanilla_defrcn/defrcn_det_r101_base${SPLIT_ID}}
NUM_GPUS=${NUM_GPUS:-1}
SKIP_DONE=${SKIP_DONE:-1}
DRY_RUN=${DRY_RUN:-0}

if [ ! -f "${IMAGENET_PRETRAIN_TORCH}" ] && [ -f "${IMAGENET_PRETRAIN_TORCH%.pth}" ]; then
    IMAGENET_PRETRAIN_TORCH="${IMAGENET_PRETRAIN_TORCH%.pth}"
fi

validate_method() {
    case "$1" in
        without_pcb|freq_aug|contrastive|self_distill|uncertainty|part_graph|clip|pcb_resnet101|pcb_fma|pcb_fma_dino_only|pcb_fma_patch|neg_proto_guard|pcb_fma_patch_neg|pcb_fma_enhanced|pcb_fma_enhanced_noaug|pcb_fma_enhanced_dino_only|pcb_fma_enhanced_neg|pcb_fma_enhanced_neg_noaug|pcb_fma_enhanced_neg_dino_only|meta_pcb|upr_tta)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

append_resnet_pcb_opts() {
    METHOD_OPTS+=(
        TEST.PCB_ENABLE True
        TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
    )
}

append_common_method_header() {
    local method_name="$1"
    METHOD_OPTS+=(
        NOVEL_METHODS.ENABLE True
        NOVEL_METHODS.METHOD "${method_name}"
    )
}

build_method_opts() {
    local method="$1"
    local calibrator_path="${2:-}"

    METHOD_OPTS=()

    case "${method}" in
        without_pcb)
            METHOD_OPTS+=(
                TEST.PCB_ENABLE False
                NOVEL_METHODS.ENABLE False
            )
            ;;
        freq_aug)
            append_resnet_pcb_opts
            append_common_method_header freq_aug
            METHOD_OPTS+=(
                NOVEL_METHODS.FREQ_AUG.ENABLE True
                NOVEL_METHODS.FREQ_AUG.LOW_FREQ_RATIO 0.3
                NOVEL_METHODS.FREQ_AUG.HIGH_FREQ_RATIO 0.3
                NOVEL_METHODS.FREQ_AUG.NUM_AUGMENTED 3
                NOVEL_METHODS.FREQ_AUG.MIX_ALPHA 0.5
                NOVEL_METHODS.FREQ_AUG.PRESERVE_NORM True
            )
            ;;
        contrastive)
            append_resnet_pcb_opts
            append_common_method_header contrastive
            METHOD_OPTS+=(
                NOVEL_METHODS.CONTRASTIVE.ENABLE True
                NOVEL_METHODS.CONTRASTIVE.FEATURE_DIM 2048
                NOVEL_METHODS.CONTRASTIVE.TEMPERATURE 0.07
                NOVEL_METHODS.CONTRASTIVE.MARGIN 0.5
                NOVEL_METHODS.CONTRASTIVE.PROTO_MOMENTUM 0.99
                NOVEL_METHODS.CONTRASTIVE.USE_HARD_NEGATIVES True
                NOVEL_METHODS.CONTRASTIVE.NUM_HARD_NEGATIVES 3
                NOVEL_METHODS.CONTRASTIVE.LOSS_WEIGHT 0.1
            )
            ;;
        self_distill)
            append_resnet_pcb_opts
            append_common_method_header self_distill
            METHOD_OPTS+=(
                NOVEL_METHODS.SELF_DISTILL.ENABLE True
                NOVEL_METHODS.SELF_DISTILL.CONFIDENCE_THRESHOLD 0.9
                NOVEL_METHODS.SELF_DISTILL.MAX_PSEUDO_PER_CLASS 20
                NOVEL_METHODS.SELF_DISTILL.PSEUDO_WEIGHT 0.3
                NOVEL_METHODS.SELF_DISTILL.EMA_MOMENTUM 0.99
                NOVEL_METHODS.SELF_DISTILL.MIN_SAMPLES_FOR_UPDATE 3
                NOVEL_METHODS.SELF_DISTILL.TEMPERATURE 1.0
                NOVEL_METHODS.SELF_DISTILL.USE_SOFT_LABELS True
                NOVEL_METHODS.SELF_DISTILL.ENTROPY_THRESHOLD 0.5
                NOVEL_METHODS.SELF_DISTILL.TWO_PASS_MODE True
                NOVEL_METHODS.SELF_DISTILL.UPDATE_INTERVAL 100
            )
            ;;
        uncertainty)
            append_resnet_pcb_opts
            append_common_method_header uncertainty
            METHOD_OPTS+=(
                NOVEL_METHODS.UNCERTAINTY.ENABLE True
                NOVEL_METHODS.UNCERTAINTY.FEATURE_DIM 2048
                NOVEL_METHODS.UNCERTAINTY.NUM_MC_SAMPLES 10
                NOVEL_METHODS.UNCERTAINTY.DROPOUT_RATE 0.1
                NOVEL_METHODS.UNCERTAINTY.UNCERTAINTY_THRESHOLD 0.3
                NOVEL_METHODS.UNCERTAINTY.FALLBACK_ALPHA 0.7
                NOVEL_METHODS.UNCERTAINTY.USE_ENSEMBLE False
                NOVEL_METHODS.UNCERTAINTY.ENSEMBLE_SIZE 5
            )
            ;;
        part_graph)
            append_resnet_pcb_opts
            append_common_method_header part_graph
            METHOD_OPTS+=(
                NOVEL_METHODS.PART_GRAPH.ENABLE True
                NOVEL_METHODS.PART_GRAPH.FEATURE_DIM 2048
                NOVEL_METHODS.PART_GRAPH.NUM_PARTS 4
                NOVEL_METHODS.PART_GRAPH.PART_DIM 256
                NOVEL_METHODS.PART_GRAPH.NUM_LAYERS 2
                NOVEL_METHODS.PART_GRAPH.SIMILARITY_MODE combined
            )
            ;;
        clip)
            append_resnet_pcb_opts
            append_common_method_header clip
            METHOD_OPTS+=(
                NOVEL_METHODS.CLIP_GROUND.ENABLE True
                NOVEL_METHODS.CLIP_GROUND.FEATURE_DIM 2048
                NOVEL_METHODS.CLIP_GROUND.CLIP_MODEL ViT-B/32
                NOVEL_METHODS.CLIP_GROUND.VISUAL_WEIGHT 0.7
                NOVEL_METHODS.CLIP_GROUND.TEXT_WEIGHT 0.3
                NOVEL_METHODS.CLIP_GROUND.USE_DESCRIPTIONS True
            )
            ;;
        pcb_resnet101)
            append_resnet_pcb_opts
            METHOD_OPTS+=(
                NOVEL_METHODS.ENABLE False
            )
            ;;
        pcb_fma)
            append_resnet_pcb_opts
            append_common_method_header pcb_fma
            METHOD_OPTS+=(
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
        pcb_fma_patch)
            append_resnet_pcb_opts
            append_common_method_header pcb_fma_patch
            METHOD_OPTS+=(
                NOVEL_METHODS.PCB_FMA_PATCH.ENABLE True
                NOVEL_METHODS.PCB_FMA_PATCH.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_PATCH.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_PATCH.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_PATCH.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_PATCH.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_PATCH.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA_PATCH.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_PATCH.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_PATCH.TOP_K_PATCHES 0
                NOVEL_METHODS.PCB_FMA_PATCH.BIDIRECTIONAL False
                NOVEL_METHODS.PCB_FMA_PATCH.CLS_WEIGHT 0.0
            )
            ;;
        neg_proto_guard)
            append_resnet_pcb_opts
            append_common_method_header neg_proto_guard
            METHOD_OPTS+=(
                NOVEL_METHODS.NEG_PROTO_GUARD.ENABLE True
                NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05
                NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3
                NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
                NOVEL_METHODS.NEG_PROTO_GUARD.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.NEG_PROTO_GUARD.FM_FEAT_DIM 768
                NOVEL_METHODS.NEG_PROTO_GUARD.ROI_SIZE 224
                NOVEL_METHODS.NEG_PROTO_GUARD.BATCH_SIZE 32
            )
            ;;
        pcb_fma_patch_neg)
            append_resnet_pcb_opts
            append_common_method_header pcb_fma_patch_neg
            METHOD_OPTS+=(
                NOVEL_METHODS.PCB_FMA_PATCH.ENABLE True
                NOVEL_METHODS.PCB_FMA_PATCH.FM_MODEL_NAME dinov2_vitb14
                NOVEL_METHODS.PCB_FMA_PATCH.FM_FEAT_DIM 768
                NOVEL_METHODS.PCB_FMA_PATCH.ROI_SIZE 224
                NOVEL_METHODS.PCB_FMA_PATCH.DET_WEIGHT 0.4
                NOVEL_METHODS.PCB_FMA_PATCH.FM_WEIGHT 0.6
                NOVEL_METHODS.PCB_FMA_PATCH.USE_ORIGINAL_PCB True
                NOVEL_METHODS.PCB_FMA_PATCH.ORIGINAL_PCB_WEIGHT 0.3
                NOVEL_METHODS.PCB_FMA_PATCH.BATCH_SIZE 32
                NOVEL_METHODS.PCB_FMA_PATCH.TOP_K_PATCHES 0
                NOVEL_METHODS.PCB_FMA_PATCH.BIDIRECTIONAL False
                NOVEL_METHODS.PCB_FMA_PATCH.CLS_WEIGHT 0.0
                NOVEL_METHODS.NEG_PROTO_GUARD.ENABLE True
                NOVEL_METHODS.NEG_PROTO_GUARD.MARGIN 0.05
                NOVEL_METHODS.NEG_PROTO_GUARD.SUPPRESSION_FACTOR 0.3
                NOVEL_METHODS.NEG_PROTO_GUARD.MAX_BASE_SAMPLES_PER_CLASS 20
            )
            ;;
        pcb_fma_enhanced)
            append_resnet_pcb_opts
            append_common_method_header pcb_fma_enhanced
            METHOD_OPTS+=(
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
            append_resnet_pcb_opts
            append_common_method_header pcb_fma_enhanced
            METHOD_OPTS+=(
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
            append_resnet_pcb_opts
            append_common_method_header pcb_fma_enhanced_neg
            METHOD_OPTS+=(
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
            append_resnet_pcb_opts
            append_common_method_header pcb_fma_enhanced_neg
            METHOD_OPTS+=(
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
        meta_pcb)
            append_resnet_pcb_opts
            append_common_method_header meta_pcb
            METHOD_OPTS+=(
                NOVEL_METHODS.META_PCB.ENABLE True
                NOVEL_METHODS.META_PCB.INPUT_DIM 8
                NOVEL_METHODS.META_PCB.HIDDEN_DIM 64
                NOVEL_METHODS.META_PCB.RESIDUAL_SCALE 0.1
                NOVEL_METHODS.META_PCB.FALLBACK_ALPHA 0.50
            )
            if [ -n "${calibrator_path}" ]; then
                METHOD_OPTS+=(
                    NOVEL_METHODS.META_PCB.CALIBRATOR_PATH "${calibrator_path}"
                )
            fi
            ;;
        upr_tta)
            append_resnet_pcb_opts
            append_common_method_header upr_tta
            METHOD_OPTS+=(
                NOVEL_METHODS.UPR_TTA.ENABLE True
                NOVEL_METHODS.UPR_TTA.NUM_MC_PASSES 10
                NOVEL_METHODS.UPR_TTA.DROPOUT_RATE 0.1
                NOVEL_METHODS.UPR_TTA.SCORE_THRESH 0.5
                NOVEL_METHODS.UPR_TTA.UNCERTAINTY_THRESH 0.05
                NOVEL_METHODS.UPR_TTA.MAX_PSEUDO_PER_CLASS 10
                NOVEL_METHODS.UPR_TTA.PSEUDO_WEIGHT 0.3
                NOVEL_METHODS.UPR_TTA.ALPHA_BASE 0.5
                NOVEL_METHODS.UPR_TTA.UNC_NORM 0.1
                NOVEL_METHODS.UPR_TTA.ENABLE_TTA False
            )
            ;;
        *)
            echo "Unsupported method in build_method_opts: ${method}" >&2
            return 1
            ;;
    esac
}

resolve_model_weight_from_dir() {
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

build_root_candidates() {
    local input_root="$1"
    local alt_root=""
    ROOT_CANDIDATES=()

    ROOT_CANDIDATES+=("${input_root}")

    if [[ "${input_root}" == *"r0101"* ]]; then
        alt_root="${input_root/r0101/r101}"
        ROOT_CANDIDATES+=("${alt_root}")
    elif [[ "${input_root}" == *"r101"* ]]; then
        alt_root="${input_root/r101/r0101}"
        ROOT_CANDIDATES+=("${alt_root}")
    fi

    if [[ "${input_root}" == "checkpoints/voc/vanilla_defrcn" ]]; then
        ROOT_CANDIDATES+=(
            "checkpoints/voc/vanilla_defrcn/defrcn_gfsod_r101_novel${SPLIT_ID}"
            "checkpoints/voc/vanilla_defrcn/defrcn_gfsod_r0101_novel${SPLIT_ID}"
        )
    fi
}

resolve_model_weight() {
    local root="$1"
    local shot="$2"
    local seed="$3"
    local root_candidate=""
    local ckpt_dir=""
    local base_name=""

    build_root_candidates "${root}"

    for root_candidate in "${ROOT_CANDIDATES[@]}"; do
        base_name=$(basename "${root_candidate}")

        if [ "${base_name}" = "${shot}shot_seed${seed}" ]; then
            if resolve_model_weight_from_dir "${root_candidate}"; then
                return 0
            fi
            continue
        fi

        if [ "${base_name}" = "tfa-like" ]; then
            ckpt_dir="${root_candidate}/${shot}shot_seed${seed}"
            if resolve_model_weight_from_dir "${ckpt_dir}"; then
                return 0
            fi
            continue
        fi

        ckpt_dir="${root_candidate}/tfa-like/${shot}shot_seed${seed}"
        if resolve_model_weight_from_dir "${ckpt_dir}"; then
            return 0
        fi

        ckpt_dir="${root_candidate}/${shot}shot_seed${seed}"
        if resolve_model_weight_from_dir "${ckpt_dir}"; then
            return 0
        fi
    done

    return 1
}

run_main() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf '[DRY_RUN] '
        printf '%q ' python3 main.py "$@"
        printf '\n'
        return 0
    fi

    python3 main.py "$@"
}

prepare_meta_calibrator() {
    local split_id="$1"
    local calibrator_file="calibrators/meta_pcb_split${split_id}.pth"
    local base_config="configs/voc/defrcn_det_r101_base${split_id}.yaml"
    local base_model="${BASE_WEIGHT_DIR}/model_final.pth"

    META_CALIBRATOR_PATH=""

    if [ -f "${calibrator_file}" ]; then
        META_CALIBRATOR_PATH="${calibrator_file}"
        return 0
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi

    mkdir -p calibrators

    if [ ! -f "${base_model}" ]; then
        echo "[WARN] Base model not found at ${base_model}; meta_pcb will use fallback alpha." >&2
        return 0
    fi

    python3 tools/meta_train_calibrator.py \
        --config-file "${base_config}" \
        --base-model "${base_model}" \
        --pcb-modelpath "${IMAGENET_PRETRAIN_TORCH}" \
        --output "${calibrator_file}" \
        --episodes "${META_PCB_EPISODES:-10000}" \
        --n-way "${META_PCB_N_WAY:-5}" \
        --k-shot "${META_PCB_K_SHOT:-1}" \
        --hidden-dim 64 \
        --lr 1e-3

    if [ -f "${calibrator_file}" ]; then
        META_CALIBRATOR_PATH="${calibrator_file}"
    fi
}

echo "=================================================="
echo "VOC GFSOD Novel Methods Runner"
echo "=================================================="
echo "Split: ${SPLIT_ID}"
echo "Methods: ${METHODS}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Run mode: ${RUN_MODE}"
echo "Save dir: ${SAVE_DIR}"
echo "Pretrained GFSOD root: ${PRETRAINED_GFSOD_ROOT}"
echo "ImageNet PCB backbone: ${IMAGENET_PRETRAIN_TORCH}"
echo "DRY_RUN: ${DRY_RUN}"
echo "=================================================="

mkdir -p "${SAVE_DIR}"

for method in ${METHODS}; do
    if ! validate_method "${method}"; then
        echo "Unsupported method: ${method}" >&2
        exit 1
    fi
done

META_CALIBRATOR_PATH=""
if [[ " ${METHODS} " == *" meta_pcb "* ]]; then
    prepare_meta_calibrator "${SPLIT_ID}"
fi

for shot in ${SHOTS}; do
    for seed in ${SEEDS}; do
        echo ""
        echo ">>> Preparing split ${SPLIT_ID}, ${shot}-shot, seed ${seed}"

        python3 tools/create_config.py --dataset voc --config_root configs/voc \
            --shot "${shot}" --seed "${seed}" --setting gfsod --split "${SPLIT_ID}"

        CONFIG_PATH="configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml"
        if [ ! -f "${CONFIG_PATH}" ]; then
            echo "Failed to create config: ${CONFIG_PATH}" >&2
            exit 1
        fi

        if ! MODEL_WEIGHT=$(resolve_model_weight "${PRETRAINED_GFSOD_ROOT}" "${shot}" "${seed}"); then
            echo "[WARN] Missing GFSOD checkpoint for split ${SPLIT_ID}, ${shot}-shot seed ${seed}; skipping." >&2
            rm -f "${CONFIG_PATH}"
            continue
        fi

        echo "[INFO] Using checkpoint: ${MODEL_WEIGHT}"

        for method in ${METHODS}; do
            METHOD_SAVE_DIR="${SAVE_DIR}/${method}/split${SPLIT_ID}"
            OUTPUT_DIR="${METHOD_SAVE_DIR}/${shot}shot_seed${seed}"
            mkdir -p "${METHOD_SAVE_DIR}"

            if [ "${SKIP_DONE}" = "1" ] && [ -f "${OUTPUT_DIR}/inference/res_final.json" ]; then
                echo "[INFO] Skip ${method} ${shot}-shot seed ${seed} (already complete)"
                continue
            fi

            build_method_opts "${method}" "${META_CALIBRATOR_PATH}"

            echo "[INFO] Running ${method} for split ${SPLIT_ID}, ${shot}-shot seed ${seed}"
            run_main \
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
    METHOD_SAVE_DIR="${SAVE_DIR}/${method}/split${SPLIT_ID}"
    if [ ! -d "${METHOD_SAVE_DIR}" ]; then
        continue
    fi

    HAS_LOG=$(find "${METHOD_SAVE_DIR}" -type f -name 'log.txt' | head -n 1 || true)
    if [ -n "${HAS_LOG}" ]; then
        echo "[INFO] Extracting results for ${method}"
        python3 tools/extract_results.py --res-dir "${METHOD_SAVE_DIR}" --shot-list ${SHOTS} 2>/dev/null || true
    fi
done

echo ""
echo "=================================================="
echo "Done. Results saved in ${SAVE_DIR}/"
echo "=================================================="
