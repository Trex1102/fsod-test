"""
Prototype initialization for cls_score weights.

Runs one pass over the K-shot support set (training dataset) before iteration 0,
computes the mean res5 feature vector for each class using GT boxes, and writes
those vectors as the initial rows of box_predictor.cls_score.weight.

This directly targets the worst single failure mode in DeFRCN novel fine-tuning:
the cls_score layer is randomly initialized (Normal(0, 0.01)) on only K samples,
requiring the short fine-tuning schedule to recover from random init.
"""

import logging
import torch
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


@torch.no_grad()
def init_cls_score_from_prototypes(model, cfg):
    """
    Initialize cls_score weight rows from mean res5 features of the K-shot support set.

    For each GT box in the training dataset:
      backbone(image) → roi_pool(GT box) → res5 → spatial mean-pool → feature vector

    The per-class mean of those vectors is written into
    box_predictor.cls_score.weight[class_id].

    Args:
        model: GeneralizedRCNN (possibly DDP-wrapped). Must already have base
               weights loaded (backbone and res5 should be frozen / pre-trained).
        cfg:   CfgNode. Reads MODEL.PROTO_INIT.NORMALIZE and MODEL.ROI_HEADS.NUM_CLASSES.
    """
    from detectron2.utils import comm
    from defrcn.dataloader import build_detection_test_loader
    from defrcn.dataloader.dataset_mapper import DatasetMapper

    raw = model.module if isinstance(model, DistributedDataParallel) else model

    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    normalize = cfg.MODEL.PROTO_INIT.NORMALIZE
    feat_dim = raw.roi_heads.box_predictor.cls_score.weight.shape[1]

    accum = torch.zeros(num_classes, feat_dim)
    counts = torch.zeros(num_classes, dtype=torch.long)

    # Single-pass loader: sequential, batch=1, no shuffling.
    # DatasetMapper(is_train=True) preserves GT annotations in the output dict.
    mapper = DatasetMapper(cfg, is_train=True)
    support_loader = build_detection_test_loader(
        cfg, cfg.DATASETS.TRAIN[0], mapper=mapper
    )

    was_training = model.training
    model.eval()

    for batch in support_loader:
        for d in batch:
            if "instances" not in d or len(d["instances"]) == 0:
                continue

            # 1. Normalize image and build ImageList (same as GeneralizedRCNN.preprocess_image)
            images = raw.preprocess_image([d])

            # 2. Backbone → res4 feature map
            features = raw.backbone(images.tensor)

            # 3. ROI-pool each GT box (IoU=1.0 crop, no proposal noise)
            gt_boxes = d["instances"].gt_boxes.to(raw.device)
            gt_classes = d["instances"].gt_classes  # CPU tensor

            roi_feats = raw.roi_heads.pooler(
                [features[f] for f in raw.roi_heads.in_features],
                [gt_boxes],
            )  # (N_gt, C, 7, 7)

            # 4. Frozen res5 + global average pool → (N_gt, feat_dim)
            roi_feats = raw.roi_heads.res5(roi_feats)
            pooled = roi_feats.mean(dim=[2, 3]).cpu()  # (N_gt, feat_dim)

            # 5. Accumulate per class
            for feat, cls_id in zip(pooled, gt_classes):
                c = int(cls_id.item())
                if 0 <= c < num_classes:
                    accum[c] += feat
                    counts[c] += 1

    # Reduce partial accumulators across DDP workers so every rank writes
    # the same prototype (all_reduce sums, then we divide by total counts).
    if comm.get_world_size() > 1:
        accum = accum.to(raw.device)
        counts = counts.to(raw.device)
        torch.distributed.all_reduce(accum)
        torch.distributed.all_reduce(counts)
        accum = accum.cpu()
        counts = counts.cpu()

    valid = counts > 0
    if not valid.any():
        logger.warning("PrototypeInit: no valid GT boxes found — skipping cls_score init.")
        if was_training:
            model.train()
        return

    # Compute mean; L2-normalize if requested
    protos = torch.zeros_like(accum)
    protos[valid] = accum[valid] / counts[valid].float().unsqueeze(1)
    if normalize:
        norms = protos[valid].norm(dim=1, keepdim=True).clamp(min=1e-8)
        protos[valid] = protos[valid] / norms

    # Write into cls_score weight rows for classes that have support examples
    weight = raw.roi_heads.box_predictor.cls_score.weight.data
    weight[valid] = protos[valid].to(weight.device)

    logger.info(
        "PrototypeInit: initialized cls_score for %d / %d classes "
        "(counts min=%d max=%d, normalize=%s)",
        int(valid.sum()),
        num_classes,
        int(counts[valid].min()),
        int(counts[valid].max()),
        normalize,
    )

    if was_training:
        model.train()
