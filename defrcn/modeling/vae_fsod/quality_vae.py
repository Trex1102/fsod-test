from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .norm_vae import latent_norm_rescale, linear_iou_to_norm


DEFAULT_QUALITY_KEYS = ("iou", "fg_ratio", "gt_coverage", "center_offset", "crowding")

# Lower value = easier for these keys, higher value = harder for the others.
_INVERT_FOR_HARDNESS = {"iou", "fg_ratio", "gt_coverage"}


def quality_consistency_loss(pred, target, loss_type="smooth_l1"):
    loss_name = str(loss_type).lower()
    if loss_name == "mse":
        return F.mse_loss(pred, target, reduction="mean")
    if loss_name == "l1":
        return F.l1_loss(pred, target, reduction="mean")
    return F.smooth_l1_loss(pred, target, reduction="mean")


def compute_quality_hardness(qualities, quality_keys, weights):
    if qualities.ndim != 2:
        raise ValueError("qualities must be rank-2 [N, D], got shape={}.".format(tuple(qualities.shape)))
    if len(quality_keys) != qualities.shape[1]:
        raise ValueError(
            "quality_keys size ({}) does not match quality dim ({}).".format(
                len(quality_keys), qualities.shape[1]
            )
        )
    if len(weights) != qualities.shape[1]:
        raise ValueError(
            "weights size ({}) does not match quality dim ({}).".format(len(weights), qualities.shape[1])
        )
    w = qualities.new_tensor(weights).view(1, -1)
    if float(torch.sum(w).item()) <= 0.0:
        raise ValueError("Sum of hardness weights must be > 0.")

    comp = qualities.clone()
    for idx, key in enumerate(quality_keys):
        if key in _INVERT_FOR_HARDNESS:
            comp[:, idx] = 1.0 - comp[:, idx]

    hardness = torch.sum(comp * w, dim=1) / torch.sum(w)
    return torch.clamp(hardness, min=0.0, max=1.0)


class QualityConditionalVAE(nn.Module):
    """
    Conditional VAE with additional proposal-quality conditioning and
    a small quality-prediction head for consistency regularization.
    """

    def __init__(
        self,
        feature_dim=2048,
        semantic_dim=512,
        quality_dim=5,
        latent_dim=512,
        encoder_hidden=4096,
        decoder_hidden=4096,
        quality_head_hidden=512,
        iou_index=0,
    ):
        super().__init__()
        cond_dim = int(semantic_dim + quality_dim)
        in_dim = int(feature_dim + cond_dim)

        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.quality_dim = int(quality_dim)
        self.latent_dim = int(latent_dim)
        self.iou_index = int(iou_index)

        self.enc_fc1 = nn.Linear(in_dim, encoder_hidden)
        self.enc_fc2 = nn.Linear(encoder_hidden, encoder_hidden)
        self.enc_fc3 = nn.Linear(encoder_hidden, encoder_hidden)
        self.mu_head = nn.Linear(encoder_hidden, latent_dim)
        self.logvar_head = nn.Linear(encoder_hidden, latent_dim)

        dec_in_dim = int(latent_dim + cond_dim)
        self.dec_fc1 = nn.Linear(dec_in_dim, decoder_hidden)
        self.dec_fc2 = nn.Linear(decoder_hidden, decoder_hidden)
        self.out_head = nn.Linear(decoder_hidden, feature_dim)

        self.q_head1 = nn.Linear(feature_dim, quality_head_hidden)
        self.q_head2 = nn.Linear(quality_head_hidden, self.quality_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def _build_cond(self, semantics, qualities):
        return torch.cat([semantics, qualities], dim=1)

    def encode(self, features, semantics, qualities):
        cond = self._build_cond(semantics, qualities)
        x = torch.cat([features, cond], dim=1)
        x = self.leaky_relu(self.enc_fc1(x))
        x = self.leaky_relu(self.enc_fc2(x))
        x = self.leaky_relu(self.enc_fc3(x))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, semantics, qualities):
        cond = self._build_cond(semantics, qualities)
        x = torch.cat([z, cond], dim=1)
        x = self.leaky_relu(self.dec_fc1(x))
        x = self.leaky_relu(self.dec_fc2(x))
        x = self.relu(self.out_head(x))
        return x

    def predict_quality(self, features):
        x = self.leaky_relu(self.q_head1(features))
        return torch.sigmoid(self.q_head2(x))

    def forward(self, features, semantics, qualities, iou_min, iou_max, norm_min, norm_max):
        if qualities.shape[1] <= self.iou_index:
            raise ValueError(
                "IoU index {} out of range for quality shape {}.".format(
                    self.iou_index, tuple(qualities.shape)
                )
            )
        mu, logvar = self.encode(features, semantics, qualities)
        z = self.reparameterize(mu, logvar)
        iou_scores = qualities[:, self.iou_index]
        target_norm = linear_iou_to_norm(iou_scores, iou_min, iou_max, norm_min, norm_max)
        z_tilde = latent_norm_rescale(z, target_norm)
        recon = self.decode(z_tilde, semantics, qualities)
        q_pred = self.predict_quality(recon)
        return recon, mu, logvar, q_pred

    @torch.no_grad()
    def generate(self, semantics, qualities, iou_min, iou_max, norm_min, norm_max):
        if qualities.shape[1] <= self.iou_index:
            raise ValueError(
                "IoU index {} out of range for quality shape {}.".format(
                    self.iou_index, tuple(qualities.shape)
                )
            )
        z = torch.randn((semantics.shape[0], self.latent_dim), device=semantics.device)
        iou_scores = qualities[:, self.iou_index]
        target_norm = linear_iou_to_norm(iou_scores, iou_min, iou_max, norm_min, norm_max)
        z_tilde = latent_norm_rescale(z, target_norm)
        return self.decode(z_tilde, semantics, qualities)


def normalize_quality_ratios(ratios: Sequence[float]):
    if len(ratios) != 3:
        raise ValueError("GEN_BIN_RATIOS must have exactly 3 elements (easy/medium/hard).")
    vals = [max(float(x), 0.0) for x in ratios]
    s = sum(vals)
    if s <= 0.0:
        raise ValueError("GEN_BIN_RATIOS must sum to a positive value.")
    return [x / s for x in vals]
