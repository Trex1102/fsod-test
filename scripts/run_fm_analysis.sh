#!/bin/bash
# Foundation Model Comparison Analysis Script
# Runs analyze_fm_comparison.py across all foundation models to generate
# statistics for the paper tables.
#
# Usage:
#   ./scripts/run_fm_analysis.sh [split] [shot]
#   ./scripts/run_fm_analysis.sh 1 1      # Split 1, 1-shot
#   ./scripts/run_fm_analysis.sh 1 5      # Split 1, 5-shot
#   ./scripts/run_fm_analysis.sh          # Default: Split 1, 1-shot

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Parse arguments
SPLIT=${1:-1}
SHOT=${2:-1}

echo "=============================================="
echo "Foundation Model Comparison Analysis"
echo "=============================================="
echo "Split: $SPLIT"
echo "Shot:  $SHOT"
echo "=============================================="

# Activate conda environment if available
if [ -f "/home/bio/anaconda3/etc/profile.d/conda.sh" ]; then
    source /home/bio/anaconda3/etc/profile.d/conda.sh
    conda activate detectron2_03
fi

export PYTHONPATH="${ROOT_DIR}:$PYTHONPATH"

# Config and weights paths
CONFIG="configs/voc/defrcn_fsod_r101_novel${SPLIT}_${SHOT}shot_seed0.yaml"
WEIGHTS="checkpoints/voc/vanilla_defrcn/split${SPLIT}/${SHOT}shot_seed0/model_final.pth"
OUTPUT_DIR="results/fm_analysis/split${SPLIT}_${SHOT}shot"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Warning: Config $CONFIG not found"
    echo "Trying to generate from template..."
    TEMPLATE="configs/voc/defrcn_fsod_r101_novelx_${SHOT}shot_seedx.yaml"
    if [ -f "$TEMPLATE" ]; then
        mkdir -p "$(dirname "$CONFIG")"
        sed "s/novelx/novel${SPLIT}/g; s/seedx/seed0/g" "$TEMPLATE" > "$CONFIG"
        echo "Created: $CONFIG"
    else
        echo "Error: Template $TEMPLATE also not found"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running FM comparison analysis..."
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR/fm_comparison.json"
echo ""

# Foundation models to analyze
FM_MODELS="imagenet dinov1 clip dinov2 dinov2_s dinov2_l"

# Run the analysis
python3 tools/analyze_fm_comparison.py \
    --config-file "$CONFIG" \
    --fm-models $FM_MODELS \
    --output "$OUTPUT_DIR/fm_comparison.json" \
    --device cuda \
    --opts MODEL.WEIGHTS "$WEIGHTS" \
           TEST.PCB_ENABLE True \
           TEST.PCB_MODELPATH .pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth \
    2>&1 | tee "$OUTPUT_DIR/analysis_log.txt"

echo ""
echo "=============================================="
echo "Analysis Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/fm_comparison.json"
echo "Log saved to:     $OUTPUT_DIR/analysis_log.txt"
echo ""

# Pretty print summary if jq is available
if command -v jq &> /dev/null; then
    echo "Summary:"
    jq -r '.models | to_entries[] | "\(.key): Var=\(.value.within_class_variance | tostring[0:6]) Margin=\(.value.nearest_negative_margin | tostring[0:6]) Purity=\(.value.nn_purity | tostring[0:6])"' \
        "$OUTPUT_DIR/fm_comparison.json" 2>/dev/null || true
fi
