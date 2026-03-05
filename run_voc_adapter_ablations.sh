#!/usr/bin/env bash
set -e

EXP_NAME=voc_adapter_ablations
SPLIT_ID=$1

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_adapter_ablations.sh <split_id>"
    exit 1
fi

SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=.pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                            # optional legacy C2 pretrain
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to your path
BASE_PRETRAIN=${BASE_PRETRAIN:-${IMAGENET_PRETRAIN}}

ABLATIONS="heavy light shared"
# SHOTS="1 2 3 5 10"
SHOTS="1"
SEEDS="0"
# SETTINGS="fsod gfsod"
SETTINGS="fsod"


for ablation in ${ABLATIONS}
do
    ABLATION_SAVE_DIR=${SAVE_DIR}/adapterAblations/${ablation}
    BASE_CFG=configs/voc/adapterAblations/${ablation}/defrcn_det_r101_base${SPLIT_ID}_adapter.yaml
    BASE_STAGE_DIR=${ABLATION_SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

    mkdir -p ${ABLATION_SAVE_DIR}

    # ------------------------------- Base Pre-train ---------------------------------- #
    # Reuse existing base-stage checkpoint; do not rerun this stage.
    # python3 main.py --num-gpus 1 --config-file ${BASE_CFG}                               \
    #     --opts MODEL.WEIGHTS ${BASE_PRETRAIN}                                            \
    #            OUTPUT_DIR ${BASE_STAGE_DIR}

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

        SETTING_SAVE_DIR=${ABLATION_SAVE_DIR}/${setting}/split${SPLIT_ID}
        mkdir -p ${SETTING_SAVE_DIR}

        # ------------------------------ Model Preparation -------------------------------- #
        # Reuse existing surgery output; do not rerun this stage.
        # python3 tools/model_surgery.py --dataset voc --method ${SURGERY_METHOD}                 \
        #     --src-path ${BASE_STAGE_DIR}/model_final.pth                                         \
        #     --save-dir ${BASE_STAGE_DIR}


        # ------------------------------ Novel Fine-tuning -------------------------------- #
        for shot in ${SHOTS}
        do
            for seed in ${SEEDS}
            do
                python3 tools/create_config.py --dataset voc --config_root configs/voc           \
                    --shot ${shot} --seed ${seed} --setting ${setting} --split ${SPLIT_ID}

                BASE_CONFIG_PATH=configs/voc/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml

                # For non-baseline ablations, generate the split/seed-specific baseline adapter config
                # because template _BASE_ paths are rewritten from novelx/seedx to concrete names.
                BASELINE_TEMPLATE=configs/voc/adapterAblations/baseline/defrcn_${setting}_r101_novelx_${shot}shot_seedx_adapter.yaml
                BASELINE_CONFIG_PATH=configs/voc/adapterAblations/baseline/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_adapter.yaml
                if [ "${ablation}" != "baseline" ]; then
                    cp ${BASELINE_TEMPLATE} ${BASELINE_CONFIG_PATH}
                    sed -i "s/novelx/novel${SPLIT_ID}/g" ${BASELINE_CONFIG_PATH}
                    sed -i "s/seedx/seed${seed}/g" ${BASELINE_CONFIG_PATH}
                fi

                ADAPTER_TEMPLATE=configs/voc/adapterAblations/${ablation}/defrcn_${setting}_r101_novelx_${shot}shot_seedx_adapter.yaml
                ADAPTER_CONFIG_PATH=configs/voc/adapterAblations/${ablation}/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_adapter.yaml
                OUTPUT_DIR=${SETTING_SAVE_DIR}/${shot}shot_seed${seed}

                cp ${ADAPTER_TEMPLATE} ${ADAPTER_CONFIG_PATH}
                sed -i "s/novelx/novel${SPLIT_ID}/g" ${ADAPTER_CONFIG_PATH}
                sed -i "s/seedx/seed${seed}/g" ${ADAPTER_CONFIG_PATH}

                python3 main.py --num-gpus 1 --config-file ${ADAPTER_CONFIG_PATH}                \
                    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                 \
                           TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

                rm -f ${ADAPTER_CONFIG_PATH}
                rm -f ${BASE_CONFIG_PATH}
                rm -f ${OUTPUT_DIR}/model_final.pth
                if [ "${ablation}" != "baseline" ]; then
                    rm -f ${BASELINE_CONFIG_PATH}
                fi
            done
        done

        python3 tools/extract_results.py --res-dir ${SETTING_SAVE_DIR} --shot-list ${SHOTS}
    done
done
