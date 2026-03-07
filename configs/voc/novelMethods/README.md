# VOC Novel Methods Configs

Each subfolder enables one novel prototype enhancement method for few-shot detection.
These methods wrap the base PCB (Prototypical Calibration Block) and are inference-only
modifications that do not require retraining the base model.

## Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `frequency_augmentation` | DCT-based prototype augmentation for diversity | `LOW_FREQ_RATIO`, `HIGH_FREQ_RATIO`, `NUM_AUGMENTED` |
| `contrastive_anchoring` | InfoNCE-based prototype separation | `TEMPERATURE`, `MARGIN`, `PROTO_MOMENTUM` |
| `self_distillation` | Test-time pseudo-label prototype refinement | `CONFIDENCE_THRESHOLD`, `MAX_PSEUDO_PER_CLASS` |
| `uncertainty_weighting` | MC dropout confidence estimation | `NUM_MC_SAMPLES`, `DROPOUT_RATE`, `UNCERTAINTY_THRESHOLD` |
| `part_graph_reasoning` | Graph neural network part-based matching | `NUM_PARTS`, `PART_DIM`, `NUM_LAYERS` |
| `clip_grounding` | CLIP vision-language semantic alignment | `CLIP_MODEL`, `VISUAL_WEIGHT`, `TEXT_WEIGHT` |

## Usage

These configs inherit from vanilla VOC FSOD configs and add novel method parameters.

Example:
```bash
python3 main.py --num-gpus 1 \
    --config-file configs/voc/novelMethods/frequency_augmentation/defrcn_fsod_r101_novel1_1shot_seed0_freq_aug.yaml \
    --opts MODEL.WEIGHTS /path/to/model_reset_remove.pth \
           TEST.PCB_MODELPATH /path/to/resnet101.pth \
           OUTPUT_DIR checkpoints/voc/novel_methods/freq_aug/1shot_seed0
```

Use `run_novel_methods.sh` for automated runs across all methods, shots, and seeds.

## Config Structure

Each config file:
1. Inherits from base VOC FSOD config (`_BASE_`)
2. Enables PCB (`TEST.PCB_ENABLE: True`)
3. Enables novel method (`NOVEL_METHODS.ENABLE: True`)
4. Sets method name (`NOVEL_METHODS.METHOD`)
5. Configures method-specific parameters

## Notes

- These methods use the same base-trained model (no base retraining needed)
- All methods are inference-only modifications to PCB
- Results can be compared directly with vanilla PCB baselines
- See `docs/novel_fsod_solutions.pdf` for detailed mathematical descriptions
