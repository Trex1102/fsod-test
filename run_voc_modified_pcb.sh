#!/usr/bin/env bash
set -euo pipefail

EXP_NAME=voc_modified_pcb
BASE_EXP_NAME=vanilla_defrcn
SPLIT_ID=${1:-}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_modified_pcb.sh <split_id>"
    exit 1
fi

SAVE_DIR=checkpoints/voc/${EXP_NAME}
BASE_SAVE_DIR=${BASE_SAVE_DIR:-checkpoints/voc/${BASE_EXP_NAME}}
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

# If 1, skip shot runs that already have inference/res_final.json.
SKIP_DONE=${SKIP_DONE:-1}

# GPU health / retry controls.
GPU_WAIT_RETRIES=${GPU_WAIT_RETRIES:-0}
GPU_WAIT_SEC=${GPU_WAIT_SEC:-15}
TRAIN_RETRIES=${TRAIN_RETRIES:-1}

MODS=${MODS:-"multi_prototype scale_aware adaptive_alpha robust_aggregation class_conditional_gate score_normalization transductive"}
SHOTS=${SHOTS:-"10"}
SEEDS=${SEEDS:-"0"}
SETTINGS=${SETTINGS:-"fsod"}  # fsod | gfsod | "fsod gfsod"

check_gpu_ready() {
    local out
    out=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>&1 || true)
    if echo "${out}" | grep -qiE "No devices were found|Unable to determine the device handle|Unknown Error|NVIDIA-SMI has failed"; then
        return 1
    fi
    if [ -z "$(echo "${out}" | tr -d [:space:])" ]; then
        return 1
    fi
    return 0
}

wait_for_gpu() {
    local attempt=0
    while true; do
        if check_gpu_ready; then
            return 0
        fi

        if [ "${attempt}" -ge "${GPU_WAIT_RETRIES}" ]; then
            echo "[ERROR] GPU is not available (driver/NVML issue)."
            echo "        nvidia-smi output:"
            nvidia-smi 2>&1 || true
            echo "        Increase GPU_WAIT_RETRIES or recover the driver, then rerun."
            return 1
        fi

        attempt=$((attempt + 1))
        echo "[WARN] GPU unavailable. retry=${attempt}/${GPU_WAIT_RETRIES} in ${GPU_WAIT_SEC}s ..."
        sleep "${GPU_WAIT_SEC}"
    done
}

run_main_train() {
    local cfg_path="$1"
    local model_weights="$2"
    local output_dir="$3"

    local attempt=1
    while [ "${attempt}" -le "${TRAIN_RETRIES}" ]; do
        wait_for_gpu
        if python3 main.py --num-gpus 1 --config-file "${cfg_path}" \
            --opts MODEL.WEIGHTS "${model_weights}" OUTPUT_DIR "${output_dir}" \
                   TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"; then
            return 0
        fi

        if [ "${attempt}" -ge "${TRAIN_RETRIES}" ]; then
            echo "[ERROR] Training failed after ${TRAIN_RETRIES} attempt(s)."
            return 1
        fi

        attempt=$((attempt + 1))
        echo "[WARN] Training failed, retrying attempt ${attempt}/${TRAIN_RETRIES} ..."
        sleep 5
    done
}

setting_is_done() {
    local setting_dir="$1"
    local shot
    local seed
    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            if [ ! -f "${setting_dir}/${shot}shot_seed${seed}/inference/res_final.json" ]; then
                return 1
            fi
        done
    done
    return 0
}

for mod in ${MODS}
do
    MOD_SAVE_DIR=${SAVE_DIR}/${mod}
    BASE_STAGE_DIR=${BASE_SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

    mkdir -p "${MOD_SAVE_DIR}"

    if [ ! -d "${BASE_STAGE_DIR}" ]; then
        echo "Missing vanilla base-stage directory: ${BASE_STAGE_DIR}"
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
            echo "Missing vanilla base weight for setting=${setting}: ${BASE_WEIGHT}"
            exit 1
        fi

        SETTING_SAVE_DIR=${MOD_SAVE_DIR}/${setting}/split${SPLIT_ID}
        mkdir -p "${SETTING_SAVE_DIR}"

        if [ "${SKIP_DONE}" = "1" ] && setting_is_done "${SETTING_SAVE_DIR}"; then
            echo "[INFO] Skip setting (already complete): ${SETTING_SAVE_DIR}"
            continue
        fi

        for shot in ${SHOTS}
        do
            for seed in ${SEEDS}
            do
                OUTPUT_DIR=${SETTING_SAVE_DIR}/${shot}shot_seed${seed}
                if [ "${SKIP_DONE}" = "1" ] && [ -f "${OUTPUT_DIR}/inference/res_final.json" ]; then
                    echo "[INFO] Skip shot (already complete): ${OUTPUT_DIR}"
                    continue
                fi

                python3 tools/create_config.py --dataset voc --config_root configs/voc \
                    --shot "${shot}" --seed "${seed}" --setting "${setting}" --split "${SPLIT_ID}"

                BASE_CONFIG_PATH=configs/voc/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
                MOD_TEMPLATE=configs/voc/modifiedPCB/${mod}/defrcn_${setting}_r101_novelx_${shot}shot_seedx_modifiedpcb.yaml
                MOD_CONFIG_PATH=configs/voc/modifiedPCB/${mod}/defrcn_${setting}_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}_modifiedpcb.yaml

                if [ ! -f "${MOD_TEMPLATE}" ]; then
                    echo "Missing template: ${MOD_TEMPLATE}"
                    exit 1
                fi

                cp "${MOD_TEMPLATE}" "${MOD_CONFIG_PATH}"
                sed -i "s/novelx/novel${SPLIT_ID}/g" "${MOD_CONFIG_PATH}"
                sed -i "s/seedx/seed${seed}/g" "${MOD_CONFIG_PATH}"

                run_main_train "${MOD_CONFIG_PATH}" "${BASE_WEIGHT}" "${OUTPUT_DIR}"

                rm -f "${MOD_CONFIG_PATH}" "${BASE_CONFIG_PATH}" "${OUTPUT_DIR}/model_final.pth"
            done
        done

        python3 tools/extract_results.py --res-dir "${SETTING_SAVE_DIR}" --shot-list ${SHOTS}
    done
done
