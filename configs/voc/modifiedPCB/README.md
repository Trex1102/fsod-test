# VOC Modified PCB Ablations

Each subfolder enables one inference-only PCB modification on top of
`configs/voc/adapterAblations/baseline/*_adapter.yaml`.

Methods:
- `quality_weighted`: quality-weighted prototype aggregation.
- `multi_prototype`: multiple prototypes per class with max-sim matching.
- `scale_aware`: scale-bin-aware prototype bank selection.
- `adaptive_alpha`: per-detection adaptive alpha for score/prototype fusion.
- `robust_aggregation`: trimmed-mean robust prototype aggregation.
- `class_conditional_gate`: weaken/skip PCB for unstable class support.
- `score_normalization`: per-class post-calibration temperature normalization.
- `transductive`: two-pass pseudo-support prototype rebuild.
- `transductive_quality_weighted`: transductive pseudo-support plus quality-weighted aggregation.
- `transductive_multi_prototype`: transductive pseudo-support plus multi-prototype matching.
- `transductive_scale_aware`: transductive pseudo-support plus scale-aware prototype selection.
