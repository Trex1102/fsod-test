#!/usr/bin/env bash

EXP_NAME=${1:-vanilla_defrcn}

SAVE_DIR=checkpoints/coco/${EXP_NAME}
IMAGENET_PRETRAIN=/data/.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=/data/.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path

SHOTS="1 2 3 5 10 30"
SEED=0


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus 8 --config-file configs/coco/defrcn_det_r101_base.yaml \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                      \
           OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base


# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset coco --method remove                      \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base/model_final.pth                    \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning -------------------------------- #
for SHOT in ${SHOTS}
do
    python3 tools/create_config.py --dataset coco14 --config_root configs/coco \
        --shot ${SHOT} --seed ${SEED} --setting fsod

    CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${SHOT}shot_seed${SEED}.yaml
    OUTPUT_DIR=${SAVE_DIR}/defrcn_fsod_r101_novel/${SHOT}shot_seed${SEED}

    python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                             \
        --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                      \
               TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

    rm -f ${CONFIG_PATH}
    rm -f ${OUTPUT_DIR}/model_final.pth
done

python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_fsod_r101_novel --shot-list ${SHOTS}
