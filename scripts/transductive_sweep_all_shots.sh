#!/bin/bash
# Transductive PCB sweep for 2, 3, 5, 10 shots with pw=0.3 (best from 1-shot)

set -e
cd /home/bio/Tanvir_Saikat/fsod-test
source /home/bio/anaconda3/etc/profile.d/conda.sh
conda activate detectron2_03
export PYTHONPATH=/home/bio/Tanvir_Saikat/fsod-test:$PYTHONPATH

# Best params from 1-shot sweep
PW=0.3
MIN_SCORE=0.6
MAX_PER_CLASS=30

# Create shot-specific configs by copying template
echo "Creating shot-specific configs..."
for SHOT in 2 3 5 10; do
    TEMPLATE="configs/voc/defrcn_fsod_r101_novelx_${SHOT}shot_seedx.yaml"
    TARGET="configs/voc/defrcn_fsod_r101_novel1_${SHOT}shot_seed0.yaml"
    
    if [ ! -f "$TARGET" ]; then
        if [ -f "$TEMPLATE" ]; then
            sed "s/novelx/novel1/g; s/seedx/seed0/g" "$TEMPLATE" > "$TARGET"
            echo "Created: $TARGET"
        else
            echo "Warning: Template $TEMPLATE not found"
        fi
    else
        echo "Config exists: $TARGET"
    fi
done

# Run transductive sweep
for SHOT in 2 3 5 10; do
    echo ""
    echo "================================================"
    echo "Running Transductive PCB: ${SHOT}-shot, split1, pw=${PW}"
    echo "================================================"
    
    OUTPUT_DIR="checkpoints/voc/pcb_transductive_sweep/pw${PW}/split1/${SHOT}shot_seed0"
    mkdir -p $OUTPUT_DIR
    
    CONFIG="configs/voc/defrcn_fsod_r101_novel1_${SHOT}shot_seed0.yaml"
    WEIGHTS="checkpoints/voc/vanilla_defrcn/split1/${SHOT}shot_seed0/model_final.pth"
    
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config $CONFIG not found, skipping"
        continue
    fi
    
    if [ ! -f "$WEIGHTS" ]; then
        echo "Warning: $WEIGHTS not found, skipping"
        continue
    fi
    
    python3 main.py --num-gpus 1 --eval-only \
        --config-file $CONFIG \
        --opts MODEL.WEIGHTS $WEIGHTS \
               TEST.PCB_ENABLE True \
               TEST.PCB_MODELPATH .pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth \
               TEST.PCB_TRANSDUCTIVE True \
               TEST.PCB_TRANS_MIN_SCORE $MIN_SCORE \
               TEST.PCB_TRANS_MAX_PER_CLASS $MAX_PER_CLASS \
               TEST.PCB_TRANS_PSEUDO_WEIGHT $PW \
               OUTPUT_DIR $OUTPUT_DIR \
        2>&1 | tee $OUTPUT_DIR/log.txt
    
    echo "Completed: ${SHOT}-shot"
done

echo ""
echo "================================================"
echo "All transductive sweeps complete!"
echo "================================================"

# Extract all results
echo ""
echo "=== RESULTS SUMMARY ==="
for SHOT in 1 2 3 5 10; do
    LOG="checkpoints/voc/pcb_transductive_sweep/pw${PW}/split1/${SHOT}shot_seed0/log.txt"
    if [ -f "$LOG" ]; then
        RESULT=$(grep -E 'copypaste:.*[0-9]+\.[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -5 | tr '\n' ' ')
        echo "${SHOT}-shot: nAP50=$(echo $RESULT | awk '{print $5}')"
    else
        echo "${SHOT}-shot: No results"
    fi
done
