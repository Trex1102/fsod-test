#!/usr/bin/env bash
# Run Novel Methods for Few-Shot Object Detection on VOC
# Usage: bash run_novel_methods.sh <split_id> [method] [shots] [seeds]
#
# Examples:
#   bash run_novel_methods.sh 1                           # Run all methods, all shots, seed 0
#   bash run_novel_methods.sh 1 freq_aug                  # Run only frequency augmentation
#   bash run_novel_methods.sh 1 all "1 2 3 5 10" "0 1 2"  # Custom shots and seeds

set -e

SPLIT_ID=$1
METHOD=${2:-"all"}  # all, freq_aug, contrastive, self_distill, uncertainty, part_graph, clip
SHOTS=${3:-"1 2 3 5 10"}
SEEDS=${4:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_novel_methods.sh <split_id> [method] [shots] [seeds]"
    echo ""
    echo "Arguments:"
    echo "  split_id  : VOC split (1, 2, or 3)"
    echo "  method    : Method to run (all, freq_aug, contrastive, self_distill, uncertainty, part_graph, clip)"
    echo "  shots     : Shot settings (default: \"1 2 3 5 10\")"
    echo "  seeds     : Random seeds (default: \"0\")"
    echo ""
    echo "Examples:"
    echo "  bash run_novel_methods.sh 1"
    echo "  bash run_novel_methods.sh 1 freq_aug"
    echo "  bash run_novel_methods.sh 1 all \"1 2 3 5 10\" \"0 1 2 3 4\""
    exit 1
fi

# Paths - adjust these to your setup
EXP_NAME=voc_novel_methods
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}

# Base model weights (from vanilla DeFRCN base training)
BASE_WEIGHT_DIR=${BASE_WEIGHT_DIR:-checkpoints/voc/defrcn/defrcn_det_r101_base${SPLIT_ID}}
BASE_WEIGHT=${BASE_WEIGHT_DIR}/model_reset_remove.pth

if [ ! -f "${BASE_WEIGHT}" ]; then
    echo "Warning: Base weight not found at ${BASE_WEIGHT}"
    echo "Please set BASE_WEIGHT_DIR to your trained base model directory."
    echo "Continuing anyway (will fail if weight is actually needed)..."
fi

# Method configurations
declare -A METHOD_NAMES
METHOD_NAMES["freq_aug"]="frequency_augmentation"
METHOD_NAMES["contrastive"]="contrastive_anchoring"
METHOD_NAMES["self_distill"]="self_distillation"
METHOD_NAMES["uncertainty"]="uncertainty_weighting"
METHOD_NAMES["part_graph"]="part_graph_reasoning"
METHOD_NAMES["clip"]="clip_grounding"

declare -A METHOD_SUFFIXES
METHOD_SUFFIXES["freq_aug"]="freq_aug"
METHOD_SUFFIXES["contrastive"]="contrastive"
METHOD_SUFFIXES["self_distill"]="self_distill"
METHOD_SUFFIXES["uncertainty"]="uncertainty"
METHOD_SUFFIXES["part_graph"]="part_graph"
METHOD_SUFFIXES["clip"]="clip"

# Determine which methods to run
if [ "${METHOD}" = "all" ]; then
    METHODS="freq_aug contrastive self_distill uncertainty part_graph clip"
else
    METHODS="${METHOD}"
fi

echo "=============================================="
echo "Novel Methods Runner for VOC Split ${SPLIT_ID}"
echo "=============================================="
echo "Methods: ${METHODS}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Save dir: ${SAVE_DIR}"
echo "Base weight: ${BASE_WEIGHT}"
echo "=============================================="

for method in ${METHODS}; do
    METHOD_DIR=${METHOD_NAMES[$method]}
    METHOD_SUFFIX=${METHOD_SUFFIXES[$method]}
    
    if [ -z "${METHOD_DIR}" ]; then
        echo "Unknown method: ${method}"
        echo "Available: freq_aug, contrastive, self_distill, uncertainty, part_graph, clip"
        exit 1
    fi
    
    echo ""
    echo ">>> Running method: ${method} (${METHOD_DIR})"
    
    METHOD_SAVE_DIR=${SAVE_DIR}/novelMethods/${METHOD_DIR}/split${SPLIT_ID}
    mkdir -p ${METHOD_SAVE_DIR}
    
    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo ""
            echo "  Processing: ${shot}-shot, seed ${seed}"
            
            # Generate seed-specific base config using create_config.py
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}
            
            # Template config path
            TEMPLATE_CONFIG=configs/voc/novelMethods/${METHOD_DIR}/defrcn_fsod_r101_novelx_${shot}shot_seedx_${METHOD_SUFFIX}.yaml
            
            # Create seed-specific config
            CONFIG_PATH=configs/voc/novelMethods/${METHOD_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_${METHOD_SUFFIX}.yaml
            
            cp ${TEMPLATE_CONFIG} ${CONFIG_PATH}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG_PATH}
            sed -i "s/seedx/seed${seed}/g" ${CONFIG_PATH}
            
            OUTPUT_DIR=${METHOD_SAVE_DIR}/${shot}shot_seed${seed}
            
            # Run evaluation with novel method
            python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH} \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} \
                       OUTPUT_DIR ${OUTPUT_DIR} \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
            
            # Cleanup temporary configs
            rm -f ${CONFIG_PATH}
            BASE_CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            rm -f ${BASE_CONFIG_PATH}
            
            # Remove model checkpoint to save space (keep only results)
            rm -f ${OUTPUT_DIR}/model_final.pth
            
            echo "  Completed: ${shot}-shot, seed ${seed}"
        done
    done
    
    # Extract results for this method
    echo ""
    echo "  Extracting results for ${method}..."
    python3 tools/extract_results.py --res-dir ${METHOD_SAVE_DIR} --shot-list ${SHOTS}
done

echo ""
echo "=============================================="
echo "All methods completed!"
echo "Results saved in: ${SAVE_DIR}/novelMethods/"
echo "=============================================="
