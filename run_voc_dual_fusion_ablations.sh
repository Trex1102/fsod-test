#!/usr/bin/env bash
set -e

EXP_NAME=voc_dualfusion_ablations
SPLIT_ID=$1

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_dual_fusion_ablations.sh <split_id>"
    exit 1
fi

SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # optional legacy C2 pretrain
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to your path
BASE_PRETRAIN=${BASE_PRETRAIN:-${IMAGENET_PRETRAIN_TORCH}}  # dual-fusion base should use .pth to avoid C2 key ambiguity

# ABLATIONS="baseline norefine roi_res5 origweights rpn_no_res5 roi_res4_only light512 lr_0p005 lr_0p0025"
ABLATIONS="baseline"
# SHOTS="1 2 3 5 10"
SHOTS="1"
SEEDS="0"
# SETTINGS="fsod gfsod"
SETTINGS="fsod"


for ablation in ${ABLATIONS}
do
    ABLATION_SAVE_DIR=${SAVE_DIR}/dualFusionAblations/${ablation}
    BASE_CFG=configs/voc/dualFusionAblations/${ablation}/defrcn_det_r101_base${SPLIT_ID}_dualfusion.yaml
    BASE_STAGE_DIR=${ABLATION_SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

    mkdir -p ${ABLATION_SAVE_DIR}

    # ------------------------------- Base Pre-train ---------------------------------- #
    python3 main.py --num-gpus 1 --config-file ${BASE_CFG}                               \
        --opts MODEL.WEIGHTS ${BASE_PRETRAIN}                                            \
               OUTPUT_DIR ${BASE_STAGE_DIR}

    for setting in ${SETTINGS}
    do
        if [ "${setting}" = "fsod" ]; then
            SURGERY_METHOD=remove
            BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_remove.pth
        elif [ "${setting}" = "gfsod" ]; then
            SURGERY_METHOD=randinit
            BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_surgery.pth
        else
            echo "Unsupported setting: ${setting}. Use fsod, gfsod, or both via SETTINGS=\"fsod gfsod\"."
            exit 1
        fi

        # ------------------------------ Model Preparation -------------------------------- #
        python3 tools/model_surgery.py --dataset voc --method ${SURGERY_METHOD}                 \
            --src-path ${BASE_STAGE_DIR}/model_final.pth                                         \
            --save-dir ${BASE_STAGE_DIR}


        # ------------------------------ Novel Fine-tuning -------------------------------- #
        for shot in ${SHOTS}
        do
            for seed in ${SEEDS}
            do
                python3 tools/create_config.py --dataset voc --config_root configs/voc           \
                    --shot ${shot} --seed ${seed} --setting ${setting} --split ${SPLIT_ID}

                BASE_CONFIG_PATH=configs/voc/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
                DF_TEMPLATE=configs/voc/dualFusionAblations/${ablation}/defrcn_${setting}_r101_novelx_${shot}shot_seedx_dualfusion.yaml
                DF_CONFIG_PATH=configs/voc/dualFusionAblations/${ablation}/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_dualfusion.yaml
                OUTPUT_DIR=${ABLATION_SAVE_DIR}/${setting}/${shot}shot_seed${seed}

                cp ${DF_TEMPLATE} ${DF_CONFIG_PATH}
                sed -i "s/novelx/novel${SPLIT_ID}/g" ${DF_CONFIG_PATH}
                sed -i "s/seedx/seed${seed}/g" ${DF_CONFIG_PATH}

                python3 main.py --num-gpus 1 --config-file ${DF_CONFIG_PATH}                     \
                    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                 \
                           TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

                rm ${DF_CONFIG_PATH}
                rm ${BASE_CONFIG_PATH}
                rm ${OUTPUT_DIR}/model_final.pth
            done
        done

        python3 tools/extract_results.py --res-dir ${ABLATION_SAVE_DIR}/${setting} --shot-list ${SHOTS}
    done
done
