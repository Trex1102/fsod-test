#!/usr/bin/env bash
set -e

EXP_NAME=voc_proto_init
BASE_EXP_NAME=vanilla_defrcn
SPLIT_ID=$1

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_proto_init.sh <split_id>"
    exit 1
fi

SAVE_DIR=checkpoints/voc/${EXP_NAME}
BASE_SAVE_DIR=checkpoints/voc/${BASE_EXP_NAME}
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to your path

ABLATIONS=${ABLATIONS:-"baseline"}
SHOTS=${SHOTS:-"1"}
SEEDS=${SEEDS:-"0"}
SETTINGS=${SETTINGS:-"fsod"}

BASE_STAGE_DIR=${BASE_SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

if [ ! -d "${BASE_STAGE_DIR}" ]; then
    echo "Missing base-stage directory: ${BASE_STAGE_DIR}"
    echo "Available vanilla_defrcn base directories:"
    find "${BASE_SAVE_DIR}" -maxdepth 1 -mindepth 1 -type d | sort
    exit 1
fi

for setting in ${SETTINGS}
do
    if [ "${setting}" = "fsod" ]; then
        BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_remove.pth
    elif [ "${setting}" = "gfsod" ]; then
        BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_surgery.pth
    else
        echo "Unsupported setting: ${setting}. Use fsod, gfsod, or both via SETTINGS=\"fsod gfsod\"."
        exit 1
    fi

    if [ ! -f "${BASE_WEIGHT}" ]; then
        echo "Missing base weight for ${setting}: ${BASE_WEIGHT}"
        exit 1
    fi

    for ablation in ${ABLATIONS}
    do
        ABLATION_TEMPLATE_DIR=configs/voc/protoInit/${ablation}
        if [ ! -d "${ABLATION_TEMPLATE_DIR}" ]; then
            echo "Missing ablation config directory: ${ABLATION_TEMPLATE_DIR}"
            exit 1
        fi

        SETTING_SAVE_DIR=${SAVE_DIR}/protoInit/${ablation}/${setting}/split${SPLIT_ID}
        mkdir -p ${SETTING_SAVE_DIR}

        for shot in ${SHOTS}
        do
            for seed in ${SEEDS}
            do
                python3 tools/create_config.py --dataset voc --config_root configs/voc           \
                    --shot ${shot} --seed ${seed} --setting ${setting} --split ${SPLIT_ID}

                BASE_CONFIG_PATH=configs/voc/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml

                BASELINE_TEMPLATE=configs/voc/protoInit/baseline/defrcn_${setting}_r101_novelx_${shot}shot_seedx_res5adapter.yaml
                BASELINE_CONFIG_PATH=configs/voc/protoInit/baseline/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_res5adapter.yaml
                cp ${BASELINE_TEMPLATE} ${BASELINE_CONFIG_PATH}
                sed -i "s/novelx/novel${SPLIT_ID}/g" ${BASELINE_CONFIG_PATH}
                sed -i "s/seedx/seed${seed}/g" ${BASELINE_CONFIG_PATH}

                if [ "${ablation}" = "baseline" ]; then
                    CONFIG_PATH=${BASELINE_CONFIG_PATH}
                else
                    ABLATION_TEMPLATE=${ABLATION_TEMPLATE_DIR}/defrcn_${setting}_r101_novelx_${shot}shot_seedx_res5adapter.yaml
                    CONFIG_PATH=${ABLATION_TEMPLATE_DIR}/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_res5adapter.yaml
                    cp ${ABLATION_TEMPLATE} ${CONFIG_PATH}
                    sed -i "s/novelx/novel${SPLIT_ID}/g" ${CONFIG_PATH}
                    sed -i "s/seedx/seed${seed}/g" ${CONFIG_PATH}
                fi

                OUTPUT_DIR=${SETTING_SAVE_DIR}/${shot}shot_seed${seed}

                python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                        \
                    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                 \
                           TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}

                if [ "${ablation}" != "baseline" ]; then
                    rm -f ${CONFIG_PATH}
                fi
                rm -f ${BASELINE_CONFIG_PATH}
                rm -f ${BASE_CONFIG_PATH}
                rm -f ${OUTPUT_DIR}/model_final.pth
            done
        done

        python3 tools/extract_results.py --res-dir ${SETTING_SAVE_DIR} --shot-list ${SHOTS}
    done
done
