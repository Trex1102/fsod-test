import math
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


def linear_iou_to_norm(iou, iou_min, iou_max, norm_min, norm_max):
    # Clamp for stability and map linearly from IoU range to target norm range.
    x = torch.clamp(iou, min=float(iou_min), max=float(iou_max))
    alpha = (x - float(iou_min)) / max(float(iou_max - iou_min), 1e-6)
    return float(norm_min) + alpha * float(norm_max - norm_min)


def latent_norm_rescale(z, target_norm):
    z_norm = z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
    return (z / z_norm) * target_norm.view(-1, 1)


def _try_clip_openai_embeddings(class_names: List[str], device: torch.device):
    import clip  # type: ignore

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(class_names).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=1)
    return text_features


def _try_open_clip_embeddings(class_names: List[str], device: torch.device):
    import open_clip  # type: ignore

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(class_names).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=1)
    return text_features


def _try_hf_clip_embeddings(class_names: List[str], device: torch.device):
    from transformers import CLIPModel, CLIPTokenizer  # type: ignore

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(class_names, padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**tokens).float()
        text_features = F.normalize(text_features, dim=1)
    return text_features


def build_text_semantic_embeddings(
    class_names: Iterable[str],
    source: str = "clip",
    device: str = "cpu",
    normalize: bool = True,
):
    class_names = list(class_names)
    device_t = torch.device(device)
    src = source.lower()
    if src != "clip":
        raise ValueError("Only source='clip' is supported for paper-matching setup.")

    last_err = None
    for fn in (
        _try_clip_openai_embeddings,
        _try_open_clip_embeddings,
        _try_hf_clip_embeddings,
    ):
        try:
            emb = fn(class_names, device_t)
            if normalize:
                emb = F.normalize(emb, dim=1)
            return emb
        except Exception as exc:  # pragma: no cover
            last_err = exc

    raise RuntimeError(
        "Failed to build CLIP semantic embeddings. Install one of: "
        "`clip`, `open_clip_torch`, or `transformers`."
    ) from last_err


class NormConditionalVAE(nn.Module):
    """
    Conditional VAE used in VAE-FSOD.
    Encoder: 3 FC layers (hidden dim), LeakyReLU.
    Decoder: 2 FC layers (hidden dim), ReLU output.
    """

    def __init__(
        self,
        feature_dim=2048,
        semantic_dim=512,
        latent_dim=512,
        encoder_hidden=4096,
        decoder_hidden=4096,
    ):
        super().__init__()
        in_dim = int(feature_dim + semantic_dim)

        self.feature_dim = int(feature_dim)
        self.semantic_dim = int(semantic_dim)
        self.latent_dim = int(latent_dim)

        self.enc_fc1 = nn.Linear(in_dim, encoder_hidden)
        self.enc_fc2 = nn.Linear(encoder_hidden, encoder_hidden)
        self.enc_fc3 = nn.Linear(encoder_hidden, encoder_hidden)
        self.mu_head = nn.Linear(encoder_hidden, latent_dim)
        self.logvar_head = nn.Linear(encoder_hidden, latent_dim)

        dec_in_dim = int(latent_dim + semantic_dim)
        self.dec_fc1 = nn.Linear(dec_in_dim, decoder_hidden)
        self.dec_fc2 = nn.Linear(decoder_hidden, decoder_hidden)
        self.out_head = nn.Linear(decoder_hidden, feature_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, features, semantics):
        x = torch.cat([features, semantics], dim=1)
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

    def decode(self, z, semantics):
        x = torch.cat([z, semantics], dim=1)
        x = self.leaky_relu(self.dec_fc1(x))
        x = self.leaky_relu(self.dec_fc2(x))
        # Paper uses ReLU as output activation.
        x = self.relu(self.out_head(x))
        return x

    def forward(self, features, semantics, iou_scores, iou_min, iou_max, norm_min, norm_max):
        mu, logvar = self.encode(features, semantics)
        z = self.reparameterize(mu, logvar)
        target_norm = linear_iou_to_norm(iou_scores, iou_min, iou_max, norm_min, norm_max)
        z_tilde = latent_norm_rescale(z, target_norm)
        recon = self.decode(z_tilde, semantics)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, semantics, betas):
        z = torch.randn((semantics.shape[0], self.latent_dim), device=semantics.device)
        z = latent_norm_rescale(z, betas)
        feats = self.decode(z, semantics)
        return feats


def vae_loss(recon, target, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = float(recon_weight) * recon_loss + float(kl_weight) * kl
    return total, recon_loss, kl


def paper_default_norm_range(latent_dim):
    # Paper maps to [sqrt(d), 5 * sqrt(d)] for d=latent_dim.
    base = math.sqrt(float(latent_dim))
    return base, 5.0 * base
