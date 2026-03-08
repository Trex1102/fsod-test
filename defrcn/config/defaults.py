from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C

# ----------- Backbone ----------- #
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_AT = 3

# ------------- RPN -------------- #
_CC.MODEL.RPN.FREEZE = False
_CC.MODEL.RPN.ENABLE_DECOUPLE = False
_CC.MODEL.RPN.BACKWARD_SCALE = 1.0

# ------------- ROI -------------- #
_CC.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
_CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
_CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
_CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.8
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster

# -------- Dual Fusion Neck ------ #
_CC.MODEL.DUAL_FUSION = CN()
_CC.MODEL.DUAL_FUSION.ENABLE = False
_CC.MODEL.DUAL_FUSION.IN_FEATURES = ["res3", "res4", "res5"]
_CC.MODEL.DUAL_FUSION.OUT_FEATURE = "res4"
_CC.MODEL.DUAL_FUSION.ALIGN_CHANNELS = 1024
_CC.MODEL.DUAL_FUSION.USE_REFINE = True
# Strong res3/res4, moderate res5
_CC.MODEL.DUAL_FUSION.RPN_INIT_LOGITS = [2.0, 1.5, 0.5]
# Strong res3/res4, near-zero res5
_CC.MODEL.DUAL_FUSION.ROI_INIT_LOGITS = [2.0, 1.5, -3.0]
_CC.MODEL.DUAL_FUSION.RPN_LEVELS = ["res3", "res4", "res5"]
_CC.MODEL.DUAL_FUSION.ROI_LEVELS = ["res3", "res4"]

# ------ Branch Adapter Neck ----- #
_CC.MODEL.BRANCH_ADAPTER = CN()
_CC.MODEL.BRANCH_ADAPTER.ENABLE = False
_CC.MODEL.BRANCH_ADAPTER.RPN_FEATURES = ["res4"]
_CC.MODEL.BRANCH_ADAPTER.ROI_FEATURES = ["res4"]
_CC.MODEL.BRANCH_ADAPTER.SHARED = False
_CC.MODEL.BRANCH_ADAPTER.BOTTLENECK_RATIO = 0.25
_CC.MODEL.BRANCH_ADAPTER.MIN_CHANNELS = 64
_CC.MODEL.BRANCH_ADAPTER.NORM = "GN"  # GN, LN, NONE
_CC.MODEL.BRANCH_ADAPTER.GN_NUM_GROUPS = 32
_CC.MODEL.BRANCH_ADAPTER.USE_GATE = True
_CC.MODEL.BRANCH_ADAPTER.GATE_INIT = 0.0
_CC.MODEL.BRANCH_ADAPTER.ZERO_INIT_LAST = True

# -------- Res5 Adapter ---------- #
_CC.MODEL.ROI_HEADS.RES5_ADAPTER = CN()
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.ENABLE = False
# Block indices (0,1,2) to insert an adapter after. Start with [2] (last block).
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.AFTER_BLOCKS = [2]
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.BOTTLENECK_RATIO = 0.25
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.MIN_CHANNELS = 64
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.NORM = "GN"
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.GN_NUM_GROUPS = 32
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.USE_GATE = True
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.GATE_INIT = 0.0
_CC.MODEL.ROI_HEADS.RES5_ADAPTER.ZERO_INIT_LAST = True

# ----------- VAE-FSOD ----------- #
_CC.MODEL.VAE_FSOD = CN()
_CC.MODEL.VAE_FSOD.ENABLE = False
_CC.MODEL.VAE_FSOD.FEATURE_BANK_PATH = ""
_CC.MODEL.VAE_FSOD.AUX_LOSS_WEIGHT = 1.0
_CC.MODEL.VAE_FSOD.AUX_BATCH_SIZE = 64

# Paper-aligned architecture/settings
_CC.MODEL.VAE_FSOD.FEATURE_DIM = 2048
_CC.MODEL.VAE_FSOD.SEMANTIC_SOURCE = "clip"
_CC.MODEL.VAE_FSOD.SEMANTIC_DIM = 512
_CC.MODEL.VAE_FSOD.LATENT_DIM = 512
_CC.MODEL.VAE_FSOD.ENCODER_HIDDEN = 4096
_CC.MODEL.VAE_FSOD.DECODER_HIDDEN = 4096

# IoU->norm mapping from paper
_CC.MODEL.VAE_FSOD.IOU_MIN = 0.5
_CC.MODEL.VAE_FSOD.IOU_MAX = 1.0
_CC.MODEL.VAE_FSOD.NORM_MIN_FACTOR = 1.0   # sqrt(latent_dim) * factor
_CC.MODEL.VAE_FSOD.NORM_MAX_FACTOR = 5.0   # sqrt(latent_dim) * factor
_CC.MODEL.VAE_FSOD.BETA_INTERVAL = 0.75
_CC.MODEL.VAE_FSOD.NUM_GEN_PER_CLASS = 30

# Quality-conditioned VAE extension (modular, optional).
_CC.MODEL.VAE_FSOD.QUALITY = CN()
_CC.MODEL.VAE_FSOD.QUALITY.ENABLE = False
_CC.MODEL.VAE_FSOD.QUALITY.KEYS = ["iou", "fg_ratio", "gt_coverage", "center_offset", "crowding"]
_CC.MODEL.VAE_FSOD.QUALITY.AUX_LOSS_WEIGHT = 0.2
_CC.MODEL.VAE_FSOD.QUALITY.AUX_LOSS_TYPE = "smooth_l1"  # smooth_l1, mse, l1
_CC.MODEL.VAE_FSOD.QUALITY.HARDNESS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.5]
_CC.MODEL.VAE_FSOD.QUALITY.BIN_QUANTILES = [0.33, 0.66]
_CC.MODEL.VAE_FSOD.QUALITY.GEN_BIN_RATIOS = [0.34, 0.33, 0.33]  # easy, medium, hard
_CC.MODEL.VAE_FSOD.QUALITY.MAX_BANK_PER_CLASS_BIN = 512

# VAE training settings
_CC.MODEL.VAE_FSOD.TRAIN_BATCH_SIZE = 256
_CC.MODEL.VAE_FSOD.TRAIN_EPOCHS = 20
_CC.MODEL.VAE_FSOD.TRAIN_LR = 1e-4
_CC.MODEL.VAE_FSOD.TRAIN_WEIGHT_DECAY = 1e-5
_CC.MODEL.VAE_FSOD.RECON_LOSS_WEIGHT = 1.0
_CC.MODEL.VAE_FSOD.KL_LOSS_WEIGHT = 1.0
_CC.MODEL.VAE_FSOD.AUG_PER_BOX = 10
_CC.MODEL.VAE_FSOD.AUG_BOX_SCALE_MAX = 0.3
_CC.MODEL.VAE_FSOD.MAX_ROIS = 200000

# ------- Prototype Init --------- #
_CC.MODEL.PROTO_INIT = CN()
_CC.MODEL.PROTO_INIT.ENABLE = False
# L2-normalize each class prototype before writing to cls_score.weight
_CC.MODEL.PROTO_INIT.NORMALIZE = True

# ------------- TEST ------------- #
_CC.TEST.PCB_ENABLE = False
_CC.TEST.PCB_MODELTYPE = "resnet"             # res-like
_CC.TEST.PCB_MODELPATH = ""
_CC.TEST.PCB_ALPHA = 0.50
_CC.TEST.PCB_UPPER = 1.0
_CC.TEST.PCB_LOWER = 0.05

# PCB inference-only modular options
# 1) Quality-weighted prototypes
_CC.TEST.PCB_QUALITY_WEIGHTED = False
_CC.TEST.PCB_QUALITY_POWER = 1.0
_CC.TEST.PCB_QUALITY_MIN_WEIGHT = 0.05
_CC.TEST.PCB_TINY_AREA_THRESH = 0.01
_CC.TEST.PCB_AREA_POWER = 0.5
_CC.TEST.PCB_CROWD_PENALTY = 0.15

# 2) Multi-prototype per class
_CC.TEST.PCB_MULTIPROTO = False
_CC.TEST.PCB_MULTIPROTO_K = 2
_CC.TEST.PCB_MULTIPROTO_ITERS = 8
_CC.TEST.PCB_MULTIPROTO_MATCH = "max"  # max, softmax
_CC.TEST.PCB_MULTIPROTO_TEMP = 0.07

# 3) Scale-aware matching
_CC.TEST.PCB_SCALE_AWARE = False
_CC.TEST.PCB_SCALE_THRESH = [0.01, 0.05]  # normalized box area thresholds

# 4) Adaptive alpha fusion
_CC.TEST.PCB_ADAPTIVE_ALPHA = False
_CC.TEST.PCB_ALPHA_MIN = 0.30
_CC.TEST.PCB_ALPHA_MAX = 0.98
_CC.TEST.PCB_ALPHA_RELIABILITY_POWER = 1.0
_CC.TEST.PCB_ALPHA_SIM_POWER = 1.0
_CC.TEST.PCB_ALPHA_USE_SIMILARITY = True

# 5) Outlier-robust aggregation
_CC.TEST.PCB_ROBUST_AGG = False
_CC.TEST.PCB_ROBUST_MODE = "trimmed_mean"  # trimmed_mean, medoid
_CC.TEST.PCB_TRIM_RATIO = 0.2

# 6) Class-conditional calibration gating
_CC.TEST.PCB_CLASS_GATE = False
_CC.TEST.PCB_CLASS_GATE_MODE = "weaken"  # weaken, skip
_CC.TEST.PCB_CLASS_GATE_TINY_RATIO = 0.60
_CC.TEST.PCB_CLASS_GATE_MIN_QUALITY = 0.20
_CC.TEST.PCB_CLASS_GATE_WEAKEN = 0.35
_CC.TEST.PCB_CLASS_GATE_MIN_SAMPLES = 2

# 7) Post-calibration score normalization
_CC.TEST.PCB_SCORE_NORM = False
_CC.TEST.PCB_SCORE_NORM_BASE_TEMP = 1.0
_CC.TEST.PCB_SCORE_NORM_MAX_TEMP = 2.5
_CC.TEST.PCB_SCORE_NORM_POWER = 1.0
_CC.TEST.PCB_SCORE_CLAMP_EPS = 1e-4

# 8) Transductive inference (test-time prototype expansion)
_CC.TEST.PCB_TRANSDUCTIVE = False
_CC.TEST.PCB_TRANS_MIN_SCORE = 0.80    # confidence threshold for pseudo-labels
_CC.TEST.PCB_TRANS_MAX_PER_CLASS = 10  # max pseudo samples per class
_CC.TEST.PCB_TRANS_PSEUDO_WEIGHT = 0.30  # pseudo quality scale factor vs real support
_CC.TEST.PCB_TRANS_ONLINE = False      # True=online (incremental), False=two-pass

# ------------ Other ------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5
_CC.MUTE_HEADER = True

# ========== NOVEL METHODS ========== #
# Novel prototype enhancement methods for few-shot detection.
# These are modular extensions that wrap the base PCB.

_CC.NOVEL_METHODS = CN()
_CC.NOVEL_METHODS.ENABLE = False
_CC.NOVEL_METHODS.METHOD = ""  # freq_aug, contrastive, self_distill, uncertainty, part_graph, clip

# ----- 1) Frequency-Domain Prototype Augmentation ----- #
_CC.NOVEL_METHODS.FREQ_AUG = CN()
_CC.NOVEL_METHODS.FREQ_AUG.ENABLE = False
_CC.NOVEL_METHODS.FREQ_AUG.LOW_FREQ_RATIO = 0.3      # Fraction of DCT coeffs as low-freq
_CC.NOVEL_METHODS.FREQ_AUG.HIGH_FREQ_RATIO = 0.3     # Fraction of DCT coeffs as high-freq
_CC.NOVEL_METHODS.FREQ_AUG.NUM_AUGMENTED = 3         # Augmented prototypes per original
_CC.NOVEL_METHODS.FREQ_AUG.MIX_ALPHA = 0.5           # Interpolation factor for mixing
_CC.NOVEL_METHODS.FREQ_AUG.PRESERVE_NORM = True      # Preserve L2 norm after augmentation

# ----- 2) Contrastive Prototype Anchoring ----- #
_CC.NOVEL_METHODS.CONTRASTIVE = CN()
_CC.NOVEL_METHODS.CONTRASTIVE.ENABLE = False
_CC.NOVEL_METHODS.CONTRASTIVE.FEATURE_DIM = 2048    # Input feature dimension
_CC.NOVEL_METHODS.CONTRASTIVE.TEMPERATURE = 0.07    # Contrastive softmax temperature
_CC.NOVEL_METHODS.CONTRASTIVE.MARGIN = 0.5          # Prototype separation margin
_CC.NOVEL_METHODS.CONTRASTIVE.PROTO_MOMENTUM = 0.99 # EMA momentum for prototype updates
_CC.NOVEL_METHODS.CONTRASTIVE.USE_HARD_NEGATIVES = True
_CC.NOVEL_METHODS.CONTRASTIVE.NUM_HARD_NEGATIVES = 3
_CC.NOVEL_METHODS.CONTRASTIVE.LOSS_WEIGHT = 0.1     # Weight for contrastive loss

# ----- 3) Self-Distillation from Test-Time Predictions ----- #
_CC.NOVEL_METHODS.SELF_DISTILL = CN()
_CC.NOVEL_METHODS.SELF_DISTILL.ENABLE = False
_CC.NOVEL_METHODS.SELF_DISTILL.CONFIDENCE_THRESHOLD = 0.9  # Min confidence for pseudo-labels
_CC.NOVEL_METHODS.SELF_DISTILL.MAX_PSEUDO_PER_CLASS = 20   # Max pseudo-samples per class
_CC.NOVEL_METHODS.SELF_DISTILL.PSEUDO_WEIGHT = 0.3         # Weight of pseudo vs real support
_CC.NOVEL_METHODS.SELF_DISTILL.EMA_MOMENTUM = 0.99         # EMA for prototype updates
_CC.NOVEL_METHODS.SELF_DISTILL.MIN_SAMPLES_FOR_UPDATE = 3  # Min pseudo-samples before update
_CC.NOVEL_METHODS.SELF_DISTILL.TEMPERATURE = 1.0           # Temperature for soft labels
_CC.NOVEL_METHODS.SELF_DISTILL.USE_SOFT_LABELS = True      # Use probability distribution
_CC.NOVEL_METHODS.SELF_DISTILL.ENTROPY_THRESHOLD = 0.5     # Max entropy for accepting pseudo
_CC.NOVEL_METHODS.SELF_DISTILL.TWO_PASS_MODE = True        # Two-pass vs online mode
_CC.NOVEL_METHODS.SELF_DISTILL.UPDATE_INTERVAL = 100       # Images between prototype updates

# ----- 4) Uncertainty-Weighted Prototype Matching ----- #
_CC.NOVEL_METHODS.UNCERTAINTY = CN()
_CC.NOVEL_METHODS.UNCERTAINTY.ENABLE = False
_CC.NOVEL_METHODS.UNCERTAINTY.FEATURE_DIM = 2048           # Feature dimension
_CC.NOVEL_METHODS.UNCERTAINTY.NUM_MC_SAMPLES = 10          # MC dropout samples
_CC.NOVEL_METHODS.UNCERTAINTY.DROPOUT_RATE = 0.1           # Dropout rate
_CC.NOVEL_METHODS.UNCERTAINTY.UNCERTAINTY_THRESHOLD = 0.3  # High uncertainty threshold
_CC.NOVEL_METHODS.UNCERTAINTY.FALLBACK_ALPHA = 0.7         # Alpha when uncertain
_CC.NOVEL_METHODS.UNCERTAINTY.USE_ENSEMBLE = False         # Use bootstrap ensemble
_CC.NOVEL_METHODS.UNCERTAINTY.ENSEMBLE_SIZE = 5            # Ensemble size

# ----- 5) Compositional Part-Graph Reasoning ----- #
_CC.NOVEL_METHODS.PART_GRAPH = CN()
_CC.NOVEL_METHODS.PART_GRAPH.ENABLE = False
_CC.NOVEL_METHODS.PART_GRAPH.FEATURE_DIM = 2048     # Input feature dimension
_CC.NOVEL_METHODS.PART_GRAPH.NUM_PARTS = 4          # Number of object parts
_CC.NOVEL_METHODS.PART_GRAPH.PART_DIM = 256         # Part representation dimension
_CC.NOVEL_METHODS.PART_GRAPH.NUM_LAYERS = 2         # Graph convolution layers
_CC.NOVEL_METHODS.PART_GRAPH.SIMILARITY_MODE = "combined"  # graph, parts, combined

# ----- 6) Cross-Modal Vision-Language Grounding (CLIP) ----- #
_CC.NOVEL_METHODS.CLIP_GROUND = CN()
_CC.NOVEL_METHODS.CLIP_GROUND.ENABLE = False
_CC.NOVEL_METHODS.CLIP_GROUND.FEATURE_DIM = 2048    # Detector feature dimension
_CC.NOVEL_METHODS.CLIP_GROUND.CLIP_MODEL = "ViT-B/32"  # CLIP model variant
_CC.NOVEL_METHODS.CLIP_GROUND.VISUAL_WEIGHT = 0.7   # Weight for visual similarity
_CC.NOVEL_METHODS.CLIP_GROUND.TEXT_WEIGHT = 0.3     # Weight for text similarity
_CC.NOVEL_METHODS.CLIP_GROUND.USE_DESCRIPTIONS = True  # Use class descriptions

# ----- 7) SAM-Masked Prototype Purification ----- #
# Uses Segment Anything Model to mask out background pixels in RoI
# before prototype pooling, reducing RoI contamination.
_CC.NOVEL_METHODS.SAM_MASKED = CN()
_CC.NOVEL_METHODS.SAM_MASKED.ENABLE = False
_CC.NOVEL_METHODS.SAM_MASKED.CHECKPOINT = ""           # Path to SAM checkpoint
_CC.NOVEL_METHODS.SAM_MASKED.MODEL_TYPE = "vit_b"      # vit_h, vit_l, vit_b
_CC.NOVEL_METHODS.SAM_MASKED.MASK_THRESHOLD = 0.0      # Threshold for binary mask
_CC.NOVEL_METHODS.SAM_MASKED.MIN_MASK_RATIO = 0.1      # Min valid mask ratio
_CC.NOVEL_METHODS.SAM_MASKED.BLEND_WEIGHT = 0.5        # Blend SAM vs original prototype
_CC.NOVEL_METHODS.SAM_MASKED.QUALITY_WEIGHTING = True  # Weight by mask quality

# ----- 7b) Saliency-Masked Prototype (lightweight alternative) ----- #
# Uses feature-based saliency instead of SAM (no external model needed)
_CC.NOVEL_METHODS.SALIENCY_MASKED = CN()
_CC.NOVEL_METHODS.SALIENCY_MASKED.ENABLE = False
_CC.NOVEL_METHODS.SALIENCY_MASKED.MODE = "gradient_magnitude"  # gradient_magnitude, channel_attention, spatial_std
_CC.NOVEL_METHODS.SALIENCY_MASKED.THRESHOLD_PERCENTILE = 50.0
_CC.NOVEL_METHODS.SALIENCY_MASKED.SOFT_MASK = True
_CC.NOVEL_METHODS.SALIENCY_MASKED.TEMPERATURE = 1.0
_CC.NOVEL_METHODS.SALIENCY_MASKED.BLEND_WEIGHT = 0.5

# ----- 8) Base-Weight Interpolation Initialization ----- #
# Initializes novel class weights as weighted combination of base class
# weights using CLIP semantic similarity. Provides better starting point.
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP = CN()
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP.ENABLE = False
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP.TOP_K = 3           # Number of base classes to interpolate from
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP.TEMPERATURE = 0.5   # Temperature for similarity softmax
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP.BLEND_WEIGHT = 0.3  # Blend with support prototypes
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP.USE_DESCRIPTIONS = True  # Use class descriptions
_CC.NOVEL_METHODS.BASE_WEIGHT_INTERP.APPLY_TO_PROTOTYPES = True  # Apply to prototypes vs raw weights
