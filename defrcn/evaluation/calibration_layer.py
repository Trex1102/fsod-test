import cv2
import math
import torch
import logging
from detectron2.structures import Boxes, ImageList
from detectron2.modeling.poolers import ROIPooler
from defrcn.dataloader import build_detection_test_loader
from defrcn.evaluation.archs import resnet101

logger = logging.getLogger(__name__)


class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Base PCB knobs
        self.alpha = float(cfg.TEST.PCB_ALPHA)
        self.pcb_upper = float(cfg.TEST.PCB_UPPER)
        self.pcb_lower = float(cfg.TEST.PCB_LOWER)

        # 1) Quality-weighted prototypes
        self.enable_quality_weighted = bool(cfg.TEST.PCB_QUALITY_WEIGHTED)
        self.quality_power = float(cfg.TEST.PCB_QUALITY_POWER)
        self.quality_min_weight = float(cfg.TEST.PCB_QUALITY_MIN_WEIGHT)
        self.tiny_area_thresh = float(cfg.TEST.PCB_TINY_AREA_THRESH)
        self.area_power = float(cfg.TEST.PCB_AREA_POWER)
        self.crowd_penalty = float(cfg.TEST.PCB_CROWD_PENALTY)

        # 2) Multi-prototype
        self.enable_multiproto = bool(cfg.TEST.PCB_MULTIPROTO)
        self.multiproto_k = int(cfg.TEST.PCB_MULTIPROTO_K)
        self.multiproto_iters = int(cfg.TEST.PCB_MULTIPROTO_ITERS)
        self.multiproto_match = str(cfg.TEST.PCB_MULTIPROTO_MATCH).lower()
        self.multiproto_temp = float(cfg.TEST.PCB_MULTIPROTO_TEMP)

        # 3) Scale-aware matching
        self.enable_scale_aware = bool(cfg.TEST.PCB_SCALE_AWARE)
        scale_thresh = list(cfg.TEST.PCB_SCALE_THRESH)
        if len(scale_thresh) != 2:
            raise ValueError("TEST.PCB_SCALE_THRESH must contain exactly 2 values.")
        self.scale_thresh = sorted([float(scale_thresh[0]), float(scale_thresh[1])])

        # 4) Adaptive alpha
        self.enable_adaptive_alpha = bool(cfg.TEST.PCB_ADAPTIVE_ALPHA)
        self.alpha_min = float(cfg.TEST.PCB_ALPHA_MIN)
        self.alpha_max = float(cfg.TEST.PCB_ALPHA_MAX)
        self.alpha_rel_power = float(cfg.TEST.PCB_ALPHA_RELIABILITY_POWER)
        self.alpha_sim_power = float(cfg.TEST.PCB_ALPHA_SIM_POWER)
        self.alpha_use_similarity = bool(cfg.TEST.PCB_ALPHA_USE_SIMILARITY)

        # 5) Robust aggregation
        self.enable_robust_agg = bool(cfg.TEST.PCB_ROBUST_AGG)
        self.robust_mode = str(cfg.TEST.PCB_ROBUST_MODE).lower()
        self.trim_ratio = float(cfg.TEST.PCB_TRIM_RATIO)

        # 6) Class-conditional gating
        self.enable_class_gate = bool(cfg.TEST.PCB_CLASS_GATE)
        self.class_gate_mode = str(cfg.TEST.PCB_CLASS_GATE_MODE).lower()
        self.class_gate_tiny_ratio = float(cfg.TEST.PCB_CLASS_GATE_TINY_RATIO)
        self.class_gate_min_quality = float(cfg.TEST.PCB_CLASS_GATE_MIN_QUALITY)
        self.class_gate_weaken = float(cfg.TEST.PCB_CLASS_GATE_WEAKEN)
        self.class_gate_min_samples = int(cfg.TEST.PCB_CLASS_GATE_MIN_SAMPLES)

        # 7) Post-calibration score normalization
        self.enable_score_norm = bool(cfg.TEST.PCB_SCORE_NORM)
        self.score_norm_base_temp = float(cfg.TEST.PCB_SCORE_NORM_BASE_TEMP)
        self.score_norm_max_temp = float(cfg.TEST.PCB_SCORE_NORM_MAX_TEMP)
        self.score_norm_power = float(cfg.TEST.PCB_SCORE_NORM_POWER)
        self.score_clamp_eps = float(cfg.TEST.PCB_SCORE_CLAMP_EPS)

        if self.multiproto_match not in {"max", "softmax"}:
            raise ValueError("TEST.PCB_MULTIPROTO_MATCH must be one of: max, softmax")
        if self.robust_mode not in {"trimmed_mean", "medoid"}:
            raise ValueError("TEST.PCB_ROBUST_MODE must be one of: trimmed_mean, medoid")
        if self.class_gate_mode not in {"weaken", "skip"}:
            raise ValueError("TEST.PCB_CLASS_GATE_MODE must be one of: weaken, skip")

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=0, pooler_type="ROIAlignV2")

        self.class_support_stats = {}
        self.class_reliability = {}
        self.class_gate_factor = {}
        self.class_temperature = {}

        self.prototypes = self.build_prototypes()
        self.exclude_cls = self.clsid_filter()

        logger.info(
            "PCB options | quality_weighted=%s multiproto=%s scale_aware=%s adaptive_alpha=%s "
            "robust_agg=%s class_gate=%s score_norm=%s",
            self.enable_quality_weighted,
            self.enable_multiproto,
            self.enable_scale_aware,
            self.enable_adaptive_alpha,
            self.enable_robust_agg,
            self.enable_class_gate,
            self.enable_score_norm,
        )

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from %s", self.cfg.TEST.PCB_MODELPATH)
        if self.cfg.TEST.PCB_MODELTYPE == "resnet":
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def _normalized_area(self, boxes, img_h, img_w):
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        area = widths * heights
        denom = max(float(img_h * img_w), 1.0)
        return area / denom

    def _compute_quality(self, area_norm, num_boxes):
        # Small/crowded boxes get downweighted.
        area_score = torch.pow(area_norm / (area_norm + self.tiny_area_thresh), self.area_power)
        crowd_score = 1.0 / (1.0 + self.crowd_penalty * max(int(num_boxes) - 1, 0))
        quality = area_score * crowd_score
        quality = quality.clamp(min=self.quality_min_weight)
        return quality

    def _safe_weight_norm(self, weights):
        total = torch.sum(weights)
        if float(total) <= 0:
            return torch.ones_like(weights) / max(int(weights.numel()), 1)
        return weights / total

    def _aggregate_one_proto(self, features, qualities):
        feats = features
        quals = qualities
        n = int(feats.shape[0])

        if n == 1:
            return feats[0]

        if self.enable_robust_agg:
            if self.robust_mode == "trimmed_mean" and n > 2:
                keep_n = max(1, int(math.ceil((1.0 - self.trim_ratio) * n)))
                mean0 = torch.mean(feats, dim=0, keepdim=True)
                sims = self._cosine_to_protos(feats, mean0).squeeze(1)
                _, idx = torch.topk(sims, k=keep_n, largest=True)
                feats = feats[idx]
                quals = quals[idx]
            elif self.robust_mode == "medoid" and n > 1:
                norm_feats = torch.nn.functional.normalize(feats, dim=1)
                sim_matrix = torch.mm(norm_feats, norm_feats.t())
                medoid_idx = torch.argmax(torch.sum(sim_matrix, dim=1))
                return feats[int(medoid_idx)]

        if self.enable_quality_weighted:
            weights = torch.pow(quals.clamp(min=1e-8), self.quality_power)
            weights = self._safe_weight_norm(weights).view(-1, 1)
            return torch.sum(feats * weights, dim=0)

        return torch.mean(feats, dim=0)

    def _init_kmeans_centers(self, features, k):
        # Deterministic farthest-point init to keep runs stable.
        n = int(features.shape[0])
        if k >= n:
            return features.clone()
        centers = [features[0]]
        min_dist = torch.full((n,), float("inf"), device=features.device)
        for _ in range(1, k):
            last = centers[-1].unsqueeze(0)
            dist = 1.0 - self._cosine_to_protos(features, last).squeeze(1)
            min_dist = torch.minimum(min_dist, dist)
            next_idx = int(torch.argmax(min_dist))
            centers.append(features[next_idx])
        return torch.stack(centers, dim=0)

    def _kmeans(self, features, k, num_iters):
        n = int(features.shape[0])
        if n == 0:
            return torch.empty((0, features.shape[1]), device=features.device), torch.empty((0,), dtype=torch.long, device=features.device)
        k = max(1, min(int(k), n))
        centers = self._init_kmeans_centers(features, k)
        assign = torch.zeros((n,), dtype=torch.long, device=features.device)

        for _ in range(max(int(num_iters), 1)):
            sims = self._cosine_to_protos(features, centers)
            assign = torch.argmax(sims, dim=1)
            new_centers = []
            for cid in range(k):
                idx = torch.nonzero(assign == cid, as_tuple=False).flatten()
                if idx.numel() == 0:
                    new_centers.append(centers[cid])
                else:
                    new_centers.append(torch.mean(features[idx], dim=0))
            centers = torch.stack(new_centers, dim=0)

        return centers, assign

    def _build_proto_bank(self, features, qualities):
        if int(features.shape[0]) == 0:
            return {
                "protos": torch.empty((0, features.shape[1]), device=features.device),
                "weights": torch.empty((0,), device=features.device),
            }

        if self.enable_multiproto and int(features.shape[0]) > 1:
            _, assign = self._kmeans(features, self.multiproto_k, self.multiproto_iters)
            protos = []
            weights = []
            for cid in range(int(torch.max(assign).item()) + 1):
                idx = torch.nonzero(assign == cid, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                cluster_feats = features[idx]
                cluster_quals = qualities[idx]
                proto = self._aggregate_one_proto(cluster_feats, cluster_quals)
                protos.append(proto)
                weights.append(float(idx.numel()))
            if len(protos) == 0:
                proto = self._aggregate_one_proto(features, qualities)
                return {"protos": proto.unsqueeze(0), "weights": torch.tensor([1.0], device=features.device)}
            proto_tensor = torch.stack(protos, dim=0)
            weight_tensor = torch.tensor(weights, dtype=proto_tensor.dtype, device=proto_tensor.device)
            weight_tensor = self._safe_weight_norm(weight_tensor)
            return {"protos": proto_tensor, "weights": weight_tensor}

        proto = self._aggregate_one_proto(features, qualities)
        return {"protos": proto.unsqueeze(0), "weights": torch.tensor([1.0], device=features.device)}

    def _area_bin(self, area_norm):
        t_small, t_medium = self.scale_thresh
        if area_norm < t_small:
            return 0
        if area_norm < t_medium:
            return 1
        return 2

    def _compute_class_stats(self, features, qualities, areas):
        n = int(features.shape[0])
        tiny_ratio = float(torch.mean((areas < self.tiny_area_thresh).float()).item()) if n > 0 else 1.0
        quality_mean = float(torch.mean(qualities).item()) if n > 0 else 0.0

        if n > 1:
            proto = self._aggregate_one_proto(features, qualities).unsqueeze(0)
            sims = self._cosine_to_protos(features, proto).squeeze(1)
            dispersion = float((1.0 - torch.clamp(sims, min=-1.0, max=1.0).mean()).item() / 2.0)
            dispersion = float(max(0.0, min(1.0, dispersion)))
        else:
            dispersion = 0.5

        sample_term = min(float(n) / 5.0, 1.0)
        reliability = (quality_mean + (1.0 - tiny_ratio) + (1.0 - dispersion) + sample_term) / 4.0
        reliability = float(max(0.0, min(1.0, reliability)))

        return {
            "n": n,
            "tiny_ratio": tiny_ratio,
            "quality_mean": quality_mean,
            "dispersion": dispersion,
            "reliability": reliability,
        }

    def _build_gate_factor(self, stats):
        if not self.enable_class_gate:
            return 1.0

        unstable = (
            stats["n"] < self.class_gate_min_samples
            or stats["tiny_ratio"] > self.class_gate_tiny_ratio
            or stats["quality_mean"] < self.class_gate_min_quality
        )

        if not unstable:
            return 1.0

        if self.class_gate_mode == "skip":
            return 0.0

        return float(max(0.0, min(1.0, self.class_gate_weaken)))

    def _build_temperature(self, stats):
        if not self.enable_score_norm:
            return 1.0
        disp = max(0.0, min(1.0, float(stats["dispersion"])))
        temp = self.score_norm_base_temp + (self.score_norm_max_temp - self.score_norm_base_temp) * (disp ** self.score_norm_power)
        return float(max(self.score_norm_base_temp, min(self.score_norm_max_temp, temp)))

    def build_prototypes(self):
        class_features = {}
        class_qualities = {}
        class_areas = {}

        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1

            img = cv2.imread(inputs[0]["file_name"])
            if img is None:
                continue
            img_h, img_w = img.shape[0], img.shape[1]

            inst = inputs[0]["instances"]
            if len(inst) == 0:
                continue

            ratio = img_h / float(inst.image_size[0])
            gt_boxes = inst.gt_boxes.tensor.clone() * ratio
            boxes = [Boxes(gt_boxes).to(self.device)]

            features = self.extract_roi_features(img, boxes).detach().cpu()
            labels = inst.gt_classes.cpu()
            areas = self._normalized_area(gt_boxes, img_h, img_w).cpu()
            qualities = self._compute_quality(areas, int(labels.shape[0])).cpu()

            for i in range(int(labels.shape[0])):
                cls = int(labels[i].item())
                if cls not in class_features:
                    class_features[cls] = []
                    class_qualities[cls] = []
                    class_areas[cls] = []
                class_features[cls].append(features[i])
                class_qualities[cls].append(float(qualities[i].item()))
                class_areas[cls].append(float(areas[i].item()))

        prototypes_dict = {}
        self.class_support_stats = {}
        self.class_reliability = {}
        self.class_gate_factor = {}
        self.class_temperature = {}

        for cls in class_features:
            feats = torch.stack(class_features[cls], dim=0)
            quals = torch.tensor(class_qualities[cls], dtype=feats.dtype)
            areas = torch.tensor(class_areas[cls], dtype=feats.dtype)

            stats = self._compute_class_stats(feats, quals, areas)
            self.class_support_stats[cls] = stats
            self.class_reliability[cls] = float(stats["reliability"])
            self.class_gate_factor[cls] = self._build_gate_factor(stats)
            self.class_temperature[cls] = self._build_temperature(stats)

            class_entry = {"global": self._build_proto_bank(feats, quals), "scale": {}}
            if self.enable_scale_aware:
                bin_ids = torch.tensor([self._area_bin(float(a.item())) for a in areas], dtype=torch.long)
                for bid in [0, 1, 2]:
                    idx = torch.nonzero(bin_ids == bid, as_tuple=False).flatten()
                    if idx.numel() == 0:
                        continue
                    class_entry["scale"][bid] = self._build_proto_bank(feats[idx], quals[idx])
            prototypes_dict[cls] = class_entry

        return prototypes_dict

    def extract_roi_features(self, img, boxes):
        mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255.0 - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # BxCxHxW

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
        activation_vectors = self.imagenet_model.fc(box_features)
        return activation_vectors

    def _cosine_to_protos(self, feature_vectors, proto_vectors):
        fv = torch.nn.functional.normalize(feature_vectors, dim=1)
        pv = torch.nn.functional.normalize(proto_vectors, dim=1)
        return torch.mm(fv, pv.t())

    def _select_proto_bank(self, cls, area_norm):
        class_entry = self.prototypes.get(cls)
        if class_entry is None:
            return None
        if self.enable_scale_aware:
            bid = self._area_bin(area_norm)
            if bid in class_entry["scale"]:
                return class_entry["scale"][bid]
        return class_entry["global"]

    def _match_similarity(self, query_feature, proto_bank):
        protos = proto_bank["protos"].to(query_feature.device)
        if int(protos.shape[0]) == 0:
            return 0.0

        sims = self._cosine_to_protos(query_feature.unsqueeze(0), protos).squeeze(0)
        if int(protos.shape[0]) == 1 or not self.enable_multiproto:
            return float(sims[0].item())

        if self.multiproto_match == "max":
            return float(torch.max(sims).item())

        # softmax similarity pooling
        temp = max(self.multiproto_temp, 1e-6)
        weights = torch.softmax(sims / temp, dim=0)
        sim = torch.sum(weights * sims)
        return float(sim.item())

    def _effective_alpha(self, cls, sim):
        gate = float(self.class_gate_factor.get(cls, 1.0))
        base_alpha = float(self.alpha)

        if self.enable_adaptive_alpha:
            rel = float(max(0.0, min(1.0, self.class_reliability.get(cls, 0.5))))
            rel = rel ** self.alpha_rel_power
            if self.alpha_use_similarity:
                sim01 = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                sim_term = sim01 ** self.alpha_sim_power
            else:
                sim_term = 1.0

            pcb_strength = (1.0 - base_alpha) * rel * sim_term * gate
            alpha = 1.0 - pcb_strength
            alpha = max(self.alpha_min, min(self.alpha_max, alpha))
            return float(alpha)

        # Class gate can still weaken static-alpha PCB.
        if self.enable_class_gate and gate < 1.0:
            return float(1.0 - (1.0 - base_alpha) * gate)

        return base_alpha

    def _normalize_score(self, cls, score):
        if not self.enable_score_norm:
            return score

        eps = max(self.score_clamp_eps, 1e-8)
        s = max(eps, min(1.0 - eps, score))
        temp = float(self.class_temperature.get(cls, self.score_norm_base_temp))
        temp = max(temp, 1e-6)
        logit = math.log(s / (1.0 - s))
        normed = 1.0 / (1.0 + math.exp(-(logit / temp)))
        return float(normed)

    def execute_calibration(self, inputs, dts):
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts
        img_h, img_w = img.shape[0], img.shape[1]

        scores = dts[0]["instances"].scores
        ileft = int((scores > self.pcb_upper).sum().item())
        iright = int((scores > self.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = dts[0]["instances"].pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        boxes = [pred_boxes.to(self.device)]
        features = self.extract_roi_features(img, boxes)

        pred_classes = dts[0]["instances"].pred_classes
        score_device = scores.device
        score_dtype = scores.dtype

        box_tensor = pred_boxes.tensor
        area_norm = self._normalized_area(box_tensor, img_h, img_w)

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.exclude_cls:
                continue
            if cls not in self.prototypes:
                continue

            q_idx = i - ileft
            proto_bank = self._select_proto_bank(cls, float(area_norm[q_idx].item()))
            if proto_bank is None:
                continue

            sim = self._match_similarity(features[q_idx], proto_bank)
            alpha = self._effective_alpha(cls, sim)

            old_score = float(scores[i].item())
            fused = old_score * alpha + sim * (1.0 - alpha)
            fused = self._normalize_score(cls, fused)

            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if "test_all" in dsname:
            if "coco" in dsname:
                exclude_ids = [
                    7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                ]
            elif "voc" in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
