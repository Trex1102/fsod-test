#!/usr/bin/env bash
# =============================================================================
# FM × Augmentation Interaction Grid Ablation
# =============================================================================
# Tests the interaction between foundation model architecture and augmentation
# strategy in a full grid: 5 FM models × 4 aug strategies = 20 configs.
#
# Usage:
#   bash run_fm_aug_ablation.sh <split_id> [shots] [seeds]
#
# Examples:
#   bash run_fm_aug_ablation.sh 1 "5" "0"
#   bash run_fm_aug_ablation.sh 1 "1 5 10" "0 1 2"
# =============================================================================

set -euo pipefail

SPLIT_ID=$1
SHOTS=${2:-"5"}
SEEDS=${3:-"0"}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_fm_aug_ablation.sh <split_id> [shots] [seeds]"
    exit 1
fi

IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/vanilla_defrcn}
SAVE_DIR=checkpoints/voc/ablations/fm_aug_grid
mkdir -p "${SAVE_DIR}"

echo "=============================================="
echo "FM × Augmentation Interaction Grid"
echo "=============================================="
echo "Split: ${SPLIT_ID}, Shots: ${SHOTS}, Seeds: ${SEEDS}"
echo "=============================================="

# FM models and their feature dimensions
declare -A FM_DIMS
FM_DIMS["dinov2_vits14"]=384
FM_DIMS["dinov2_vitb14"]=768
FM_DIMS["dinov2_vitl14"]=1024
FM_DIMS["dino_vitb16"]=768
FM_DIMS["clip_vitb16"]=512

# Augmentation strategies
declare -A AUG_CONFIGS
AUG_CONFIGS["no_aug"]="NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False"
AUG_CONFIGS["flip_only"]="NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP False"
AUG_CONFIGS["multicrop_only"]="NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP False NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True"
AUG_CONFIGS["flip_multicrop"]="NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_FLIP True NOVEL_METHODS.PCB_FMA_ENHANCED.AUG_MULTICROP True"

total_configs=0
for fm_model in dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dino_vitb16 clip_vitb16; do
    for aug in no_aug flip_only multicrop_only flip_multicrop; do
        total_configs=$((total_configs + 1))
    done
done
echo "Total configurations: ${total_configs}"
echo "=============================================="

config_num=0
for fm_model in dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dino_vitb16 clip_vitb16; do
    dim=${FM_DIMS[$fm_model]}

    for aug in no_aug flip_only multicrop_only flip_multicrop; do
        config_num=$((config_num + 1))
        aug_opts=${AUG_CONFIGS[$aug]}
        tag="${fm_model}_${aug}"

        echo ""
        echo ">>> [${config_num}/${total_configs}] FM=${fm_model}, Aug=${aug}"

        for shot in ${SHOTS}; do
            for seed in ${SEEDS}; do
                model_weight=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth
                out_dir=${SAVE_DIR}/${tag}/split${SPLIT_ID}/${shot}shot_seed${seed}
                mkdir -p "${out_dir}"

                python3 tools/create_config.py --dataset voc --config_root configs/voc \
                    --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

                template=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novelx_${shot}shot_seedx_pcb_fma_enhanced.yaml
                config=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_pcb_fma_enhanced.yaml
                cp ${template} ${config}
                sed -i "s/novelx/novel${SPLIT_ID}/g" ${config}
                sed -i "s/seedx/seed${seed}/g" ${config}

                python3 main.py --num-gpus 1 --eval-only \
                    --config-file ${config} \
                    --opts \
                    MODEL.WEIGHTS ${model_weight} \
                    OUTPUT_DIR ${out_dir} \
                    TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_MODEL_NAME ${fm_model} \
                    NOVEL_METHODS.PCB_FMA_ENHANCED.FM_FEAT_DIM ${dim} \
                    ${aug_opts}

                rm -f ${config}
                rm -f configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            done
        done

        # Extract results
        res_dir=${SAVE_DIR}/${tag}/split${SPLIT_ID}
        if [ -d "${res_dir}" ]; then
            python3 tools/extract_results.py --res-dir ${res_dir} --shot-list ${SHOTS}
        fi
    done
done

echo ""
echo "=============================================="
echo "FM × Augmentation grid complete!"
echo "Results in: ${SAVE_DIR}/"
echo "=============================================="

# Print summary grid
echo ""
echo "Results Grid:"
echo "=============================================="
printf "%-20s" "FM \\ Aug"
for aug in no_aug flip_only multicrop_only flip_multicrop; do
    printf "%-15s" "${aug}"
done
echo ""
echo "----------------------------------------------------------------------"
for fm_model in dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dino_vitb16 clip_vitb16; do
    printf "%-20s" "${fm_model}"
    for aug in no_aug flip_only multicrop_only flip_multicrop; do
        results_file=${SAVE_DIR}/${fm_model}_${aug}/split${SPLIT_ID}/results.txt
        if [ -f "${results_file}" ]; then
            # Extract mean AP from results
            mean_ap=$(grep "μ" ${results_file} 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
            printf "%-15s" "${mean_ap}"
        else
            printf "%-15s" "pending"
        fi
    done
    echo ""
done
echo "=============================================="
