#!/usr/bin/env bash

EXP_NAME=vanilla_defrcn
SPLIT_ID=$1

SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path

SHOT=1
SEED=0


# ------------------------------- Base Pre-train ---------------------------------- #
# python3 main.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#            OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}


# ------------------------------ Model Preparation -------------------------------- #
# python3 tools/model_surgery.py --dataset voc --method remove                                    \
#     --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
#     --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=checkpoints/voc/voc_qualityvaefsod_ablations/defrcn_det_r101_base${SPLIT_ID}/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning -------------------------------- #
# Rerun only DeFRCN FSOD 1-shot in the same setting as run_voc_quality_vae_ablations.sh
python3 tools/create_config.py --dataset voc --config_root configs/voc \
    --shot ${SHOT} --seed ${SEED} --setting fsod --split ${SPLIT_ID}

CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${SHOT}shot_seed${SEED}.yaml
OUTPUT_DIR=${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}/${SHOT}shot_seed${SEED}

python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                             \
    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                      \
           TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

rm ${CONFIG_PATH}
rm ${OUTPUT_DIR}/model_final.pth

python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID} --shot-list ${SHOT}
