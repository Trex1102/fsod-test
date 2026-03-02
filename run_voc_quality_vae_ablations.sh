#!/usr/bin/env bash

EXP_NAME=voc_qualityvaefsod_ablations
SPLIT_ID=$1

SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path

# ABLATIONS="baseline iou_only no_qaux no_crowding center_heavy hard_bias easy_bias wide_bins"
ABLATIONS="baseline"
# SHOTS="1 2 3 5 10"
SHOTS="1"
SEEDS="0"


# ------------------------------- Base Pre-train ---------------------------------- #
# python3 main.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#            OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}


# ------------------------------ Model Preparation -------------------------------- #
# python3 tools/model_surgery.py --dataset voc --method remove                                    \
#     --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
#     --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_remove.pth


# ---------------------------- qualityVaeFsod Ablations --------------------------- #
for ablation in ${ABLATIONS}
do
    ABLATION_SAVE_DIR=${SAVE_DIR}/qualityVaeFsod/${ablation}
    BASE_CFG=configs/voc/qualityVaeFsod/${ablation}/defrcn_det_r101_base${SPLIT_ID}_qualityvaefsod.yaml
    VAE_CKPT=${ABLATION_SAVE_DIR}/vae_model/model_final.pth

    mkdir -p ${ABLATION_SAVE_DIR}/vae_model
    mkdir -p ${ABLATION_SAVE_DIR}/feature_banks
    mkdir -p ${ABLATION_SAVE_DIR}/fsod

    # ------------------------------ VAE Training --------------------------------- #
    # python3 tools/train_vae_fsod.py \
    #     --config-file ${BASE_CFG} \
    #     --weights ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth \
    #     --output ${VAE_CKPT}

    # ---------------------------- Novel Fine-tuning ------------------------------ #
    for shot in ${SHOTS}
    do
        for seed in ${SEEDS}
        do
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting fsod --split ${SPLIT_ID}

            BASE_CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            Q_TEMPLATE=configs/voc/qualityVaeFsod/${ablation}/defrcn_fsod_r101_novelx_${shot}shot_seedx_qualityvaefsod.yaml
            Q_CONFIG_PATH=configs/voc/qualityVaeFsod/${ablation}/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_qualityvaefsod.yaml

            cp ${Q_TEMPLATE} ${Q_CONFIG_PATH}
            sed -i "s/novelx/novel${SPLIT_ID}/g" ${Q_CONFIG_PATH}
            sed -i "s/seedx/seed${seed}/g" ${Q_CONFIG_PATH}

            BANK_PATH=${ABLATION_SAVE_DIR}/feature_banks/fsod_${shot}shot_seed${seed}.pth
            OUTPUT_DIR=${ABLATION_SAVE_DIR}/fsod/${shot}shot_seed${seed}

            python3 tools/generate_vae_fsod_features.py \
                --config-file ${Q_CONFIG_PATH} \
                --vae-ckpt ${VAE_CKPT} \
                --output ${BANK_PATH}

            python3 main.py --num-gpus 1 --config-file ${Q_CONFIG_PATH}                         \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                    \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}                             \
                       MODEL.VAE_FSOD.FEATURE_BANK_PATH ${BANK_PATH}

            rm ${Q_CONFIG_PATH}
            rm ${BASE_CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        done
    done

    python3 tools/extract_results.py --res-dir ${ABLATION_SAVE_DIR}/fsod --shot-list ${SHOTS}
done
