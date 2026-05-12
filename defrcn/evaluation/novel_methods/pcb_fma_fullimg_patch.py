"""
PCB-FMA full-image patch-token RoI pooling.

This inference-time wrapper keeps the detector fixed, runs DINOv2 once on the
full image, pools contextualized patch tokens inside each detector box, and
uses the resulting RoI feature as prototype evidence. It supports a patch-only
mode and a hybrid mode that blends full-image patch RoI evidence with the
standard crop-CLS evidence.
"""

import cv2
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FullImagePatchFeatureExtractor:
    """Extract full-image DINOv2 patch-grid features and pool them over boxes."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        model_path: str = "",
        feat_dim: int = 768,
        image_size: int = 518,
        crop_size: int = 224,
        batch_size: int = 32,
        patch_size: int = 14,
        pool_mode: str = "average",
        attention_temperature: float = 0.2,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.feat_dim = feat_dim
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.pool_mode = str(pool_mode).lower()
        self.attention_temperature = float(attention_temperature)
        self.device = torch.device(device)
        self.model = None

        if self.image_size % self.patch_size != 0:
            self.image_size = int(math.ceil(self.image_size / self.patch_size) * self.patch_size)
            logger.warning("Full-image DINO size rounded up to %d", self.image_size)

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self._load_model(model_name, model_path)

    @staticmethod
    def _patch_dinov2_for_py38():
        import glob
        import os
        import sys

        if sys.version_info >= (3, 10):
            return
        hub_dir = torch.hub.get_dir()
        for d in glob.glob(os.path.join(hub_dir, "facebookresearch_dinov2*")):
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as f:
                            content = f.read()
                        if "float | None" in content or "int | None" in content or "str | None" in content:
                            if "from __future__ import annotations" not in content:
                                content = "from __future__ import annotations\n" + content
                                with open(fpath, "w") as f:
                                    f.write(content)
                    except Exception:
                        pass

    def _load_model(self, model_name: str, model_path: str):
        try:
            if model_path:
                logger.info("Loading full-image patch FM from local path: %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("Loading full-image patch FM via torch.hub: %s", model_name)
                self._patch_dinov2_for_py38()
                try:
                    self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
                except TypeError as e:
                    if "unsupported operand type" not in str(e):
                        raise
                    self._patch_dinov2_for_py38()
                    import sys as _sys
                    for key in [k for k in _sys.modules if "dinov2" in k]:
                        del _sys.modules[key]
                    self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)

            self.model = self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

            with torch.no_grad():
                dummy = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
                patch_tokens, cls_token = self.model.get_intermediate_layers(
                    dummy, n=1, return_class_token=True,
                )[0]
                self.feat_dim = int(cls_token.shape[-1])
                self.grid_h = self.grid_w = int(round(math.sqrt(patch_tokens.shape[1])))
                if self.grid_h * self.grid_w != patch_tokens.shape[1]:
                    raise ValueError("DINO patch token count is not square for full-image pooling")

            logger.info(
                "Full-image patch FM loaded: %s dim=%d grid=%dx%d pool=%s",
                model_name, self.feat_dim, self.grid_h, self.grid_w, self.pool_mode,
            )
        except Exception as e:
            logger.warning("Failed to load full-image patch FM '%s': %s", model_name, e)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _preprocess_image(self, img: np.ndarray, size: int) -> torch.Tensor:
        resized = cv2.resize(img, (size, size))
        t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.norm_mean).view(3, 1, 1)
        std = torch.tensor(self.norm_std).view(3, 1, 1)
        return ((t - mean) / std).unsqueeze(0).to(self.device)

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        mean = torch.tensor(self.norm_mean).view(3, 1, 1)
        std = torch.tensor(self.norm_std).view(3, 1, 1)
        for crop in crops:
            resized = cv2.resize(crop, (self.crop_size, self.crop_size))
            t = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tensors.append((t - mean) / std)
        return torch.stack(tensors, dim=0).to(self.device)

    def _crop_rois(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> List[np.ndarray]:
        boxes = boxes_tensor.detach().cpu().numpy()
        img_h, img_w = img.shape[:2]
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(img[max(0, y1):max(0, y1) + 1, max(0, x1):max(0, x1) + 1])
            else:
                crops.append(img[y1:y2, x1:x2])
        return crops

    def _full_image_tokens(self, img: np.ndarray) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        with torch.no_grad():
            tensor = self._preprocess_image(img, self.image_size)
            patch_tokens, _ = self.model.get_intermediate_layers(
                tensor, n=1, return_class_token=True,
            )[0]
        grid_h = grid_w = int(round(math.sqrt(patch_tokens.shape[1])))
        return patch_tokens[0].view(grid_h, grid_w, -1)

    def _scaled_boxes(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> torch.Tensor:
        img_h, img_w = img.shape[:2]
        boxes = boxes_tensor.detach().to(self.device).float().clone()
        boxes[:, [0, 2]] *= float(self.image_size) / max(float(img_w), 1.0)
        boxes[:, [1, 3]] *= float(self.image_size) / max(float(img_h), 1.0)
        boxes[:, 0::2].clamp_(0.0, float(self.image_size - 1))
        boxes[:, 1::2].clamp_(0.0, float(self.image_size - 1))
        return boxes

    def _roi_align_pool(self, patch_grid: torch.Tensor, scaled_boxes: torch.Tensor) -> torch.Tensor:
        try:
            from torchvision.ops import roi_align
        except Exception as e:
            logger.warning("ROIAlign unavailable (%s); falling back to average pooling", e)
            return self._manual_pool(patch_grid, scaled_boxes, mode="average")

        feat = patch_grid.permute(2, 0, 1).unsqueeze(0).contiguous()
        batch_idx = torch.zeros((scaled_boxes.shape[0], 1), device=self.device, dtype=scaled_boxes.dtype)
        rois = torch.cat([batch_idx, scaled_boxes], dim=1)
        spatial_scale = float(patch_grid.shape[1]) / float(self.image_size)
        pooled = roi_align(
            feat, rois, output_size=(1, 1), spatial_scale=spatial_scale,
            sampling_ratio=-1, aligned=True,
        ).flatten(1)
        return F.normalize(pooled, dim=1)

    def _manual_pool(self, patch_grid: torch.Tensor, scaled_boxes: torch.Tensor, mode: str) -> torch.Tensor:
        grid_h, grid_w, feat_dim = patch_grid.shape
        flat = patch_grid.view(grid_h * grid_w, feat_dim)
        ys = (torch.arange(grid_h, device=self.device).float() + 0.5) * (self.image_size / float(grid_h))
        xs = (torch.arange(grid_w, device=self.device).float() + 0.5) * (self.image_size / float(grid_w))
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

        pooled = []
        for box in scaled_boxes:
            x1, y1, x2, y2 = box.tolist()
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            mask = (
                (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
                (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
            )
            if not bool(mask.any()):
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                dist = (centers[:, 0] - cx).pow(2) + (centers[:, 1] - cy).pow(2)
                mask[dist.argmin()] = True

            tokens = F.normalize(flat[mask], dim=1)
            token_centers = centers[mask]
            if mode == "max":
                vec = tokens.max(dim=0).values
            elif mode == "center_weighted":
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                sx = max((x2 - x1) * 0.5, self.image_size / float(grid_w))
                sy = max((y2 - y1) * 0.5, self.image_size / float(grid_h))
                dist = ((token_centers[:, 0] - cx) / sx).pow(2) + ((token_centers[:, 1] - cy) / sy).pow(2)
                weights = torch.exp(-0.5 * dist)
                weights = weights / weights.sum().clamp_min(1e-6)
                vec = (tokens * weights[:, None]).sum(dim=0)
            elif mode == "attention":
                anchor = F.normalize(tokens.mean(dim=0), dim=0)
                temp = max(self.attention_temperature, 1e-6)
                weights = F.softmax((tokens @ anchor) / temp, dim=0)
                vec = (tokens * weights[:, None]).sum(dim=0)
            else:
                vec = tokens.mean(dim=0)
            pooled.append(F.normalize(vec, dim=0))
        return torch.stack(pooled, dim=0)

    def extract_patch_roi_features(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.feat_dim), device=self.device)
        patch_grid = self._full_image_tokens(img)
        if patch_grid is None:
            return None
        patch_grid = patch_grid.to(self.device)
        scaled_boxes = self._scaled_boxes(img, boxes_tensor)
        if self.pool_mode == "roi_align":
            return self._roi_align_pool(patch_grid, scaled_boxes)
        return self._manual_pool(patch_grid, scaled_boxes, self.pool_mode)

    def extract_crop_cls_features(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.feat_dim), device=self.device)
        crops = self._crop_rois(img, boxes_tensor)
        all_cls = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = self._preprocess_crops(crops[start:start + self.batch_size])
                _, cls_token = self.model.get_intermediate_layers(
                    batch, n=1, return_class_token=True,
                )[0]
                all_cls.append(cls_token)
        return torch.cat(all_cls, dim=0)

    def extract_roi_features(self, img: np.ndarray, boxes_tensor: torch.Tensor):
        patch_feats = self.extract_patch_roi_features(img, boxes_tensor)
        cls_feats = self.extract_crop_cls_features(img, boxes_tensor)
        return patch_feats, cls_feats


class PCBFMAFullImagePatch:
    """Full-image DINO patch RoI prototype rescoring wrapper."""

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        full_cfg = cfg.NOVEL_METHODS.PCB_FMA_FULLIMG_PATCH

        self.det_weight = float(full_cfg.DET_WEIGHT)
        self.fm_weight = float(full_cfg.FM_WEIGHT)
        self.use_original_pcb = bool(full_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(full_cfg.ORIGINAL_PCB_WEIGHT)
        self.feature_mode = str(full_cfg.FEATURE_MODE).lower()
        self.crop_cls_weight = float(full_cfg.CROP_CLS_WEIGHT)
        self.temperature = float(full_cfg.TEMPERATURE)
        self.competitive_mode = str(full_cfg.COMPETITIVE_MODE).lower()
        self.run_inner_first = bool(full_cfg.RUN_INNER_FIRST)

        total = self.det_weight + self.fm_weight + (self.original_pcb_weight if self.use_original_pcb else 0.0)
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        self.fm_extractor = FullImagePatchFeatureExtractor(
            model_name=str(full_cfg.FM_MODEL_NAME),
            model_path=str(full_cfg.FM_MODEL_PATH),
            feat_dim=int(full_cfg.FM_FEAT_DIM),
            image_size=int(full_cfg.IMAGE_SIZE),
            crop_size=int(full_cfg.CROP_SIZE),
            batch_size=int(full_cfg.BATCH_SIZE),
            patch_size=int(full_cfg.PATCH_SIZE),
            pool_mode=str(full_cfg.POOL_MODE),
            attention_temperature=float(full_cfg.ATTENTION_TEMPERATURE),
            device=str(cfg.MODEL.DEVICE),
        )

        self.patch_prototypes: Dict[int, torch.Tensor] = {}
        self.cls_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_prototypes()
        else:
            logger.warning("PCB-FMA-FullImgPatch: FM unavailable, falling back to base PCB.")

    def _build_prototypes(self):
        logger.info("PCB-FMA-FullImgPatch: Building support prototypes.")
        class_patch: Dict[int, List[torch.Tensor]] = {}
        class_cls: Dict[int, List[torch.Tensor]] = {}
        dataloader = self.base_pcb.dataloader
        for index in range(len(dataloader.dataset)):
            inputs = [dataloader.dataset[index]]
            img = cv2.imread(inputs[0]["file_name"])
            if img is None:
                continue
            inst = inputs[0]["instances"]
            if len(inst) == 0:
                continue
            img_h = img.shape[0]
            ratio = img_h / float(inst.image_size[0])
            gt_boxes = inst.gt_boxes.tensor.clone() * ratio
            labels = inst.gt_classes.cpu()

            patch_feats = self.fm_extractor.extract_patch_roi_features(img, gt_boxes)
            cls_feats = self.fm_extractor.extract_crop_cls_features(img, gt_boxes)
            if patch_feats is None:
                continue
            patch_feats = patch_feats.detach().cpu()
            cls_feats = cls_feats.detach().cpu() if cls_feats is not None else None
            for i in range(len(labels)):
                cls = int(labels[i].item())
                class_patch.setdefault(cls, []).append(patch_feats[i])
                if cls_feats is not None:
                    class_cls.setdefault(cls, []).append(cls_feats[i])

        for cls, feats in class_patch.items():
            self.patch_prototypes[cls] = F.normalize(torch.stack(feats, dim=0).mean(dim=0), dim=0)
        for cls, feats in class_cls.items():
            self.cls_prototypes[cls] = F.normalize(torch.stack(feats, dim=0).mean(dim=0), dim=0)
        logger.info(
            "PCB-FMA-FullImgPatch: Built %d patch prototypes and %d crop-CLS prototypes.",
            len(self.patch_prototypes), len(self.cls_prototypes),
        )

    def _competitive_score(self, query_feat: torch.Tensor, prototypes: Dict[int, torch.Tensor], cls: int) -> float:
        if cls not in prototypes:
            return 0.0
        all_cls_ids = sorted(prototypes.keys())
        if len(all_cls_ids) <= 1:
            sim = float(torch.dot(query_feat, prototypes[cls].to(query_feat.device)).item())
            return (sim + 1.0) / 2.0

        sims = []
        target_idx = -1
        for idx, c in enumerate(all_cls_ids):
            sims.append(torch.dot(query_feat, prototypes[c].to(query_feat.device)))
            if c == cls:
                target_idx = idx
        if target_idx < 0:
            return 0.0
        sims_tensor = torch.stack(sims, dim=0)
        if self.competitive_mode == "softmax":
            scores = F.softmax(sims_tensor / max(self.temperature, 1e-6), dim=0)
            return float(scores[target_idx].item())
        return (float(sims_tensor[target_idx].item()) + 1.0) / 2.0

    def _fm_score(self, patch_feat, cls_feat, cls: int) -> float:
        patch_score = 0.0
        if patch_feat is not None and cls in self.patch_prototypes:
            patch_score = self._competitive_score(F.normalize(patch_feat, dim=0), self.patch_prototypes, cls)
        if self.feature_mode == "patch":
            return patch_score

        cls_score = 0.0
        if cls_feat is not None and cls in self.cls_prototypes:
            cls_score = self._competitive_score(F.normalize(cls_feat, dim=0), self.cls_prototypes, cls)
        w = min(max(self.crop_cls_weight, 0.0), 1.0)
        return (1.0 - w) * patch_score + w * cls_score

    def execute_calibration(self, inputs, dts):
        if self.run_inner_first:
            dts = self.base_pcb.execute_calibration(inputs, dts)
        if not self.fm_extractor.available or not self.patch_prototypes:
            if self.run_inner_first:
                return dts
            return self.base_pcb.execute_calibration(inputs, dts)
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts

        instances = dts[0]["instances"]
        scores = instances.scores
        ileft = int((scores > self.base_pcb.pcb_upper).sum().item())
        iright = int((scores > self.base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = instances.pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        pred_classes = instances.pred_classes
        box_tensor = pred_boxes.tensor
        patch_feats = self.fm_extractor.extract_patch_roi_features(img, box_tensor)
        cls_feats = None
        if self.feature_mode == "hybrid":
            cls_feats = self.fm_extractor.extract_crop_cls_features(img, box_tensor)

        if self.use_original_pcb:
            img_h, img_w = img.shape[:2]
            boxes = [pred_boxes.to(self.base_pcb.device)]
            pcb_features = self.base_pcb.extract_roi_features(img, boxes)
            area_norm = self.base_pcb._normalized_area(box_tensor, img_h, img_w)

        score_device = scores.device
        score_dtype = scores.dtype
        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb.exclude_cls:
                continue
            q_idx = i - ileft
            det_score = float(scores[i].item())
            patch_feat = patch_feats[q_idx] if patch_feats is not None else None
            cls_feat = cls_feats[q_idx] if cls_feats is not None else None
            fm_sim = self._fm_score(patch_feat, cls_feat, cls)

            pcb_sim = 0.0
            if self.use_original_pcb and cls in self.base_pcb.prototypes:
                proto_bank = self.base_pcb._select_proto_bank(cls, float(area_norm[q_idx].item()))
                if proto_bank is not None:
                    pcb_sim = self.base_pcb._match_similarity(pcb_features[q_idx], proto_bank)
                    pcb_sim = (pcb_sim + 1.0) / 2.0

            if self.use_original_pcb:
                fused = self.det_weight * det_score + self.fm_weight * fm_sim + self.original_pcb_weight * pcb_sim
            else:
                fused = self.det_weight * det_score + self.fm_weight * fm_sim
            scores[i] = torch.tensor(max(0.0, min(1.0, fused)), device=score_device, dtype=score_dtype)
        return dts

    def __getattr__(self, name):
        return getattr(self.base_pcb, name)


def build_pcb_fma_fullimg_patch(base_pcb, cfg):
    if base_pcb is None:
        from .pcb_fma import build_fm_only_support
        base_pcb = build_fm_only_support(cfg)
    return PCBFMAFullImagePatch(base_pcb, cfg)



def build_pcb_fma_enhanced_fullimg_patch(base_pcb, cfg):
    """PCB-FMA no/aug configuration followed by full-image patch RoI rescoring."""
    from .pcb_fma import build_fm_only_support
    from .pcb_fma_enhanced import PCBFMAEnhanced

    if base_pcb is None:
        base_pcb = build_fm_only_support(cfg)
    enhanced_pcb = PCBFMAEnhanced(base_pcb, cfg)
    fullimg_pcb = PCBFMAFullImagePatch(enhanced_pcb, cfg)
    fullimg_pcb.run_inner_first = True
    return fullimg_pcb
