#!/usr/bin/env bash
set -e
cd /home/bio/Tanvir_Saikat/fsod-test

for shot in 1 5; do
    echo "=== CT scene-bg ${shot}-shot seed0 ==="

    python3 tools/create_config.py --dataset voc --config_root configs/voc \
        --shot ${shot} --seed 0 --setting fsod --split 1

    CFG=configs/voc/novelMethods/counterfactual_transport/defrcn_fsod_r101_novel1_${shot}shot_seed0_ct_scenebg.yaml
    cp configs/voc/novelMethods/counterfactual_transport/defrcn_fsod_r101_novelx_${shot}shot_seedx_counterfactual_transport.yaml ${CFG}
    sed -i 's/novelx/novel1/g; s/seedx/seed0/g' ${CFG}

    python3 main.py --num-gpus 1 --eval-only \
        --config-file ${CFG} \
        --opts \
        MODEL.WEIGHTS checkpoints/voc/vanilla_defrcn/split1/${shot}shot_seed0/model_final.pth \
        OUTPUT_DIR checkpoints/voc/voc_ct_scenebg/split1/${shot}shot_seed0 \
        TEST.PCB_MODELPATH .pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth \
        2>&1 | tee /tmp/ct_scenebg_${shot}shot.log

    rm -f ${CFG} configs/voc/defrcn_fsod_r101_novel1_${shot}shot_seed0.yaml
    echo "=== DONE ${shot}-shot ==="
done
