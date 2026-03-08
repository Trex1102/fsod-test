#!/usr/bin/env bash
set -euo pipefail

EXP_NAME=${EXP_NAME:-voc_modified_pcb_new}
BASE_EXP_NAME=${BASE_EXP_NAME:-vanilla_defrcn}
SPLIT_ID=${1:-}
RUN_MODE=${2:-${RUN_MODE:-"infer_pretrained_novel"}}

if [ -z "${SPLIT_ID}" ]; then
    echo "Usage: bash run_voc_modified_pcb.sh <split_id> [run_mode]"
    echo ""
    echo "Run modes:"
    echo "  finetune                Fine-tune modified PCB models from vanilla base weights"
    echo "  infer_pretrained_novel  Eval-only using pretrained vanilla novel checkpoints"
    echo ""
    echo "Examples:"
    echo "  bash run_voc_modified_pcb.sh 1"
    echo "  bash run_voc_modified_pcb.sh 1 infer_pretrained_novel"
    exit 1
fi

SAVE_DIR=checkpoints/voc/${EXP_NAME}
BASE_SAVE_DIR=${BASE_SAVE_DIR:-checkpoints/voc/${BASE_EXP_NAME}}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-${BASE_SAVE_DIR}}
IMAGENET_PRETRAIN_TORCH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

# If 1, skip shot runs that already have inference/res_final.json.
SKIP_DONE=${SKIP_DONE:-1}

# GPU health / retry controls.
GPU_WAIT_RETRIES=${GPU_WAIT_RETRIES:-0}
GPU_WAIT_SEC=${GPU_WAIT_SEC:-15}
TRAIN_RETRIES=${TRAIN_RETRIES:-1}

MODS=${MODS:-"transductive_multi_prototype transductive_quality_weighted transductive_scale_aware"}
SHOTS=${SHOTS:-"1 10"}
SEEDS=${SEEDS:-"0"}
SETTINGS=${SETTINGS:-"fsod"}  # fsod | gfsod | "fsod gfsod"
EXTRA_OPTS=${EXTRA_OPTS:-}

case "${RUN_MODE}" in
    finetune|train|base)
        RUN_MODE="finetune"
        RUN_MODE_DESC="fine-tune from vanilla base weights"
        ROOT_SAVE_DIR=${SAVE_DIR}
        ;;
    infer_pretrained_novel|pretrained_novel|eval_pretrained_novel)
        RUN_MODE="infer_pretrained_novel"
        RUN_MODE_DESC="eval-only from pretrained vanilla novel checkpoints"
        ROOT_SAVE_DIR=${SAVE_DIR}/pretrainedNovelEval
        ;;
    *)
        echo "Unknown run_mode: ${RUN_MODE}"
        echo "Available run modes: finetune, infer_pretrained_novel"
        exit 1
        ;;
esac

echo "[INFO] split=${SPLIT_ID} run_mode=${RUN_MODE} (${RUN_MODE_DESC})"
echo "[INFO] mods=${MODS} settings=${SETTINGS} shots=${SHOTS} seeds=${SEEDS}"
if [ -n "${EXTRA_OPTS}" ]; then
    echo "[INFO] extra_opts=${EXTRA_OPTS}"
fi
if [ "${RUN_MODE}" = "infer_pretrained_novel" ]; then
    echo "[INFO] pretrained_novel_root=${PRETRAINED_NOVEL_ROOT}"
else
    echo "[INFO] base_save_dir=${BASE_SAVE_DIR}"
fi

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

run_main_job() {
    local cfg_path="$1"
    local model_weights="$2"
    local output_dir="$3"

    local attempt=1
    local main_args=(--num-gpus 1)
    if [ "${RUN_MODE}" = "infer_pretrained_novel" ]; then
        main_args+=(--eval-only)
    fi
    main_args+=(
        --config-file "${cfg_path}"
        --opts
        MODEL.WEIGHTS "${model_weights}"
        OUTPUT_DIR "${output_dir}"
        TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
    )
    if [ -n "${EXTRA_OPTS}" ]; then
        local extra_opts=()
        read -r -a extra_opts <<< "${EXTRA_OPTS}"
        main_args+=("${extra_opts[@]}")
    fi

    while [ "${attempt}" -le "${TRAIN_RETRIES}" ]; do
        wait_for_gpu
        if python3 main.py "${main_args[@]}"; then
            return 0
        fi

        if [ "${attempt}" -ge "${TRAIN_RETRIES}" ]; then
            echo "[ERROR] main.py failed after ${TRAIN_RETRIES} attempt(s)."
            return 1
        fi

        attempt=$((attempt + 1))
        echo "[WARN] ${RUN_MODE} run failed, retrying attempt ${attempt}/${TRAIN_RETRIES} ..."
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
    MOD_SAVE_DIR=${ROOT_SAVE_DIR}/${mod}
    BASE_STAGE_DIR=${BASE_SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

    mkdir -p "${MOD_SAVE_DIR}"

    if [ "${RUN_MODE}" = "finetune" ] && [ ! -d "${BASE_STAGE_DIR}" ]; then
        echo "Missing vanilla base-stage directory: ${BASE_STAGE_DIR}"
        echo "Available vanilla_defrcn base directories:"
        find "${BASE_SAVE_DIR}" -maxdepth 1 -mindepth 1 -type d | sort
        exit 1
    fi

    for setting in ${SETTINGS}
    do
        if [ "${setting}" != "fsod" ] && [ "${setting}" != "gfsod" ]; then
            echo "Unsupported setting: ${setting}. Use fsod, gfsod, or both via SETTINGS=\"fsod gfsod\"."
            exit 1
        fi

        if [ "${RUN_MODE}" = "finetune" ]; then
            if [ "${setting}" = "fsod" ]; then
                BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_remove.pth
            else
                BASE_WEIGHT=${BASE_STAGE_DIR}/model_reset_surgery.pth
            fi

            if [ ! -f "${BASE_WEIGHT}" ]; then
                echo "Missing vanilla base weight for setting=${setting}: ${BASE_WEIGHT}"
                exit 1
            fi
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

                if [ "${RUN_MODE}" = "infer_pretrained_novel" ]; then
                    MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/split${SPLIT_ID}/${shot}shot_seed${seed}/model_final.pth
                    if [ ! -f "${MODEL_WEIGHT}" ]; then
                        echo "Missing pretrained novel weight: ${MODEL_WEIGHT}"
                        echo "Set PRETRAINED_NOVEL_ROOT or add the checkpoint for split=${SPLIT_ID}, shot=${shot}, seed=${seed}."
                        exit 1
                    fi
                else
                    MODEL_WEIGHT=${BASE_WEIGHT}
                fi

                run_main_job "${MOD_CONFIG_PATH}" "${MODEL_WEIGHT}" "${OUTPUT_DIR}"

                rm -f "${MOD_CONFIG_PATH}" "${BASE_CONFIG_PATH}" "${OUTPUT_DIR}/model_final.pth"
            done
        done

        python3 tools/extract_results.py --res-dir "${SETTING_SAVE_DIR}" --shot-list ${SHOTS}
    done
done
