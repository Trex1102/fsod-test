#!/usr/bin/env bash
# FP Error Analysis for pcb_fma_enhanced across shot counts.
#
# Usage:
#   bash run_fp_analysis.sh <shot> [seed] [split]
#
# Examples:
#   bash run_fp_analysis.sh 1
#   bash run_fp_analysis.sh 5 0 1
#   bash run_fp_analysis.sh 10 0 1

set -e
source /home/bio/anaconda3/etc/profile.d/conda.sh
conda activate detectron2_03
cd /home/bio/Tanvir_Saikat/fsod-test

shot=${1:?Usage: bash run_fp_analysis.sh <shot> [seed] [split]}
seed=${2:-0}
split=${3:-1}

echo "=== FP Analysis: pcb_fma_enhanced ${shot}-shot seed${seed} split${split} ==="

python3 tools/create_config.py --dataset voc --config_root configs/voc \
    --shot ${shot} --seed ${seed} --setting fsod --split ${split}

CFG=configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novel${split}_${shot}shot_seed${seed}_pcb_fma_enhanced.yaml
cp configs/voc/novelMethods/pcb_fma_enhanced/defrcn_fsod_r101_novelx_${shot}shot_seedx_pcb_fma_enhanced.yaml ${CFG}
sed -i "s/novelx/novel${split}/g; s/seedx/seed${seed}/g" ${CFG}

OUT_DIR=checkpoints/voc/voc_fma_enhanced_analysis/split${split}/${shot}shot_seed${seed}
mkdir -p ${OUT_DIR}

python3 tools/analyze_fp_types.py \
    --config-file ${CFG} \
    --weights checkpoints/voc/vanilla_defrcn/split${split}/${shot}shot_seed${seed}/model_final.pth \
    --pcb-modelpath .pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth \
    --output-dir ${OUT_DIR} \
    --score-thresh 0.05 \
    --iou-thresh 0.5 \
    2>&1 | tee ${OUT_DIR}/fp_analysis_log.txt

rm -f ${CFG} configs/voc/defrcn_fsod_r101_novel${split}_${shot}shot_seed${seed}.yaml
echo "=== DONE — results in ${OUT_DIR}/ ==="
