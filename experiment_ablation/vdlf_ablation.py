"""带消融开关的 VDLF-Net（Mini-ImageNet / experiment_framework 路线）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ablation_spec import AblationSpec, GatingMode


def postprocess_gating_weights(weights: torch.Tensor, mode: GatingMode) -> torch.Tensor:
    if mode == "learned":
        return weights
    k = weights.size(-1)
    if mode == "uniform":
        return torch.full_like(weights, 1.0 / k)
    if mode == "coarse_only":
        w = torch.zeros_like(weights)
        w[..., 0] = 1.0
        return w
    if mode == "fine_only":
        w = torch.zeros_like(weights)
        w[..., -1] = 1.0
        return w
    raise ValueError(f"Unknown gating_mode: {mode}")


class VDLFNetAblation(nn.Module):
    """包装 experiment_framework.VDLFNet：门控 softmax 后按 spec 替换融合权重。"""

    def __init__(self, config, spec: AblationSpec):
        super().__init__()
        _ensure_framework()
        from experiment_framework import VDLFNet

        self.spec = spec
        self.core = VDLFNet(config)

    @property
    def config(self):
        return self.core.config

    def forward(self, x, num_samples: int = 1, is_support: bool = False):
        c = self.core
        batch_size = x.size(0)
        multiscale_features = c.extract_multiscale_features(x)
        initial_fused = c.compute_initial_fused_feature(multiscale_features)
        mu, logvar = c.vae_encoder(initial_fused)

        if is_support and num_samples > 1:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(
                batch_size, num_samples, c.config.latent_dim, device=x.device, dtype=mu.dtype
            )
            z = mu.unsqueeze(1) + std.unsqueeze(1) * eps
            z = z.view(-1, c.config.latent_dim)
            weights = c.gating_network(z)
            weights = postprocess_gating_weights(weights, self.spec.gating_mode)

            multiscale_tensor = torch.stack(multiscale_features, dim=1)
            multiscale_tensor = multiscale_tensor.unsqueeze(1).expand(
                batch_size, num_samples, c.config.num_scales, -1
            )
            multiscale_tensor = multiscale_tensor.contiguous().view(
                batch_size * num_samples, c.config.num_scales, -1
            )
            fused_features = (multiscale_tensor * weights.unsqueeze(-1)).sum(dim=1)
            fused_features = c.normalize_features(fused_features)
            return fused_features, mu, logvar

        z = mu
        weights = c.gating_network(z)
        weights = postprocess_gating_weights(weights, self.spec.gating_mode)
        multiscale_tensor = torch.stack(multiscale_features, dim=1)
        fused_features = (multiscale_tensor * weights.unsqueeze(-1)).sum(dim=1)
        fused_features = c.normalize_features(fused_features)
        return fused_features, mu, logvar

    def extract_multiscale_features(self, x):
        return self.core.extract_multiscale_features(x)

    def compute_initial_fused_feature(self, m):
        return self.core.compute_initial_fused_feature(m)

    def decode(self, z):
        return self.core.decode(z)


def _ensure_framework():
    try:
        import experiment_framework  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "请将 experiment_2 加入 sys.path，例如在 notebook 中：\n"
            "  sys.path.insert(0, str(Path('..') / 'experiment_2'))"
        ) from e


def compute_vdlf_loss_ablation(model, support_images, support_labels, query_images, query_labels, config, spec: AblationSpec):
    n_way = config.n_way
    k_shot = config.k_shot
    num_samples = config.num_samples

    support_features, support_mu, support_logvar = model(
        support_images, num_samples=num_samples, is_support=True
    )
    query_features, query_mu, query_logvar = model(
        query_images, num_samples=1, is_support=False
    )

    support_features_reshaped = support_features.view(n_way, k_shot, num_samples, -1)
    prototypes = support_features_reshaped.mean(dim=(1, 2))

    cosine_sim = (query_features.unsqueeze(1) * prototypes.unsqueeze(0)).sum(dim=-1)
    logits = cosine_sim * config.temperature
    ce_loss = F.cross_entropy(logits, query_labels)

    all_images = torch.cat([support_images, query_images], dim=0)
    all_mu = torch.cat([support_mu, query_mu], dim=0)
    all_logvar = torch.cat([support_logvar, query_logvar], dim=0)

    all_features_initial = model.compute_initial_fused_feature(model.extract_multiscale_features(all_images))
    std = torch.exp(0.5 * all_logvar)
    eps = torch.randn_like(std)
    z = all_mu + std * eps
    recon_features = model.decode(z)
    recon_loss = F.mse_loss(recon_features, all_features_initial)
    kl_loss = -0.5 * torch.sum(1 + all_logvar - all_mu.pow(2) - all_logvar.exp()) / all_images.size(0)

    reg = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)
    if not spec.drop_recon:
        reg = reg + recon_loss
    if not spec.drop_kl:
        reg = reg + kl_loss
    total_loss = ce_loss + config.alpha * reg
    return total_loss, ce_loss, recon_loss, kl_loss
