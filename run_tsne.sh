#!/usr/bin/env bash
# =============================================================================
# t-SNE Feature Visualization Runner
# =============================================================================
# Generates t-SNE plots for the NeurIPS paper (Appendix: Feature Visualization).
#
# Usage:
#   bash run_tsne.sh [split_id] [shot] [seed] [mode]
#
# Modes:
#   support_test  - Support (stars) + test GT features (dots) [default]
#   support_only  - Support features only
#   compare_fm    - Side-by-side: ResNet-101 vs DINOv2
#   all           - Generate all visualizations
#
# Examples:
#   bash run_tsne.sh                         # Split 1, 5-shot, all modes
#   bash run_tsne.sh 1 5 0 compare_fm       # Just the comparison plot
#   bash run_tsne.sh 1 1 0 all              # All modes for 1-shot
# =============================================================================

set -e

SPLIT_ID=${1:-1}
SHOT=${2:-5}
SEED=${3:-0}
MODE=${4:-"all"}

OUTDIR=figures/tsne
mkdir -p ${OUTDIR}

echo "=============================================="
echo "t-SNE Feature Visualization"
echo "=============================================="
echo "Split: ${SPLIT_ID}, Shot: ${SHOT}, Seed: ${SEED}"
echo "Mode:  ${MODE}"
echo "Output: ${OUTDIR}/"
echo "=============================================="

# ---- support_test: novel class features ----
if [[ "${MODE}" == "support_test" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> support_test: Novel class support + test features"
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode support_test \
        --max-test-per-class 50 \
        --output ${OUTDIR}/tsne_novel_split${SPLIT_ID}_${SHOT}shot.pdf
fi

# ---- support_test with base classes ----
if [[ "${MODE}" == "support_test_base" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> support_test + base classes"
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode support_test \
        --include-base --max-base-per-class 20 \
        --max-test-per-class 30 \
        --output ${OUTDIR}/tsne_novel_base_split${SPLIT_ID}_${SHOT}shot.pdf
fi

# ---- compare_fm: ResNet-101 vs DINOv2 side-by-side ----
if [[ "${MODE}" == "compare_fm" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> compare_fm: ResNet-101 vs DINOv2 ViT-B/14"
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode compare_fm \
        --max-test-per-class 50 \
        --output ${OUTDIR}/tsne_compare_resnet_dinov2_split${SPLIT_ID}_${SHOT}shot.pdf
fi

# ---- support_only: just support features ----
if [[ "${MODE}" == "support_only" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> support_only: Support features only"
    python3 tools/visualize_tsne.py \
        --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
        --fm-model dinov2_vitb14 \
        --mode support_only \
        --output ${OUTDIR}/tsne_support_split${SPLIT_ID}_${SHOT}shot.pdf
fi

# ---- Multi-shot comparison (1-shot vs 5-shot vs 10-shot) ----
if [[ "${MODE}" == "multi_shot" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> Multi-shot comparison (1, 5, 10-shot)"
    for s in 1 5 10; do
        echo "  ${s}-shot..."
        python3 tools/visualize_tsne.py \
            --split ${SPLIT_ID} --shot ${s} --seed ${SEED} \
            --fm-model dinov2_vitb14 \
            --mode support_test \
            --max-test-per-class 50 \
            --output ${OUTDIR}/tsne_novel_split${SPLIT_ID}_${s}shot.pdf
    done
fi

# ---- FM model comparison (DINOv2 S/B/L) ----
if [[ "${MODE}" == "fm_compare" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> FM model comparison (DINOv2 ViT-S/B/L)"
    for fm in dinov2_vits14 dinov2_vitb14 dinov2_vitl14; do
        echo "  ${fm}..."
        python3 tools/visualize_tsne.py \
            --split ${SPLIT_ID} --shot ${SHOT} --seed ${SEED} \
            --fm-model ${fm} \
            --mode support_test \
            --max-test-per-class 50 \
            --output ${OUTDIR}/tsne_${fm}_split${SPLIT_ID}_${SHOT}shot.pdf
    done
fi

echo ""
echo "=============================================="
echo "t-SNE visualization complete!"
echo "Output files in: ${OUTDIR}/"
echo "=============================================="
ls -la ${OUTDIR}/tsne_*split${SPLIT_ID}* 2>/dev/null || echo "  (no output files yet)"
echo "=============================================="
