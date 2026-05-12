#!/usr/bin/env bash

EXP_NAME=vanilla_defrcn
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path
SPLIT_ID=$1


# ------------------------------- Base Pre-train ---------------------------------- #
# python3 main.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#            OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}


# # ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                    \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth


# # ------------------------------ Novel Fine-tuning ------------------------------- #
# # --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
# for seed in 0 1 2 3 4 5 6 7 8 9
for seed in 2 3 
do
    for shot in 1 2 3 5 10 
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        # rm ${OUTPUT_DIR}/model_final.pth
    done
done
# python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10  # surmarize all results

