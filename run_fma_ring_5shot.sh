#!/usr/bin/env bash
set -e
source /home/bio/anaconda3/etc/profile.d/conda.sh
conda activate detectron2_03
cd /home/bio/Tanvir_Saikat/fsod-test

shot=5
echo "=== PCB-FMA-Ring ${shot}-shot seed0 split1 ==="

python3 tools/create_config.py --dataset voc --config_root configs/voc \
    --shot ${shot} --seed 0 --setting fsod --split 1

CFG=configs/voc/novelMethods/pcb_fma_ring/defrcn_fsod_r101_novel1_${shot}shot_seed0_pcb_fma_ring.yaml
cp configs/voc/novelMethods/pcb_fma_ring/defrcn_fsod_r101_novelx_${shot}shot_seedx_pcb_fma_ring.yaml ${CFG}
sed -i 's/novelx/novel1/g; s/seedx/seed0/g' ${CFG}

WEIGHTS=checkpoints/voc/vanilla_defrcn/split1/${shot}shot_seed0/model_final.pth
OUT_DIR=checkpoints/voc/voc_fma_ring/split1/${shot}shot_seed0
PCB_PATH=.pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

mkdir -p ${OUT_DIR}

python3 main.py --num-gpus 1 --eval-only \
    --config-file ${CFG} \
    --opts \
    MODEL.WEIGHTS ${WEIGHTS} \
    OUTPUT_DIR ${OUT_DIR} \
    TEST.PCB_MODELPATH ${PCB_PATH} \
    2>&1 | tee ${OUT_DIR}/eval_log.txt

rm -f ${CFG} configs/voc/defrcn_fsod_r101_novel1_${shot}shot_seed0.yaml
echo "=== DONE — results in ${OUT_DIR}/ ==="
