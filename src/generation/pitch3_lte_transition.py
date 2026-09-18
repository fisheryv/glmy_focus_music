"""Linear-time, masked high-resolution transition features for LTE V3.9-A."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedTemporalResidual(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        valid = mask.unsqueeze(-1)
        hidden = self.norm(values).masked_fill(~valid, 0)
        hidden = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = F.silu(hidden).masked_fill(~valid, 0)
        hidden = self.pointwise(hidden.transpose(1, 2)).transpose(1, 2)
        return (values + F.silu(hidden)).masked_fill(~valid, 0)


class HighResolutionTransitionBranch(nn.Module):
    """Keep ordered frame pairs before stride-4; no attention or random dropout.

    Four validity bits accompany 256 pooled features. Only the final projection
    is zero-initialized, so the new path starts as an exact additive identity.
    """

    lags = (1, 2, 4, 8)
    channels = 64
    pair_channels = 32

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.normalized_projection = nn.Linear(latent_dim, self.channels)
        self.raw_projection = nn.Linear(latent_dim, self.channels, bias=False)
        self.blocks = nn.ModuleList([MaskedTemporalResidual(self.channels, d) for d in (1, 2, 4)])
        self.pair_mlp = nn.Sequential(nn.Linear(self.channels * 4, self.pair_channels), nn.SiLU())
        self.summary = nn.Sequential(
            nn.Linear(len(self.lags) * (2 * self.pair_channels + 1), output_dim), nn.SiLU()
        )
        self.output = nn.Linear(output_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, latent: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if latent.ndim != 3 or mask.shape != latent.shape[:2]:
            raise ValueError("Transition branch requires [B,T,C] and matching mask")
        mask = mask.to(device=latent.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Transition branch requires a valid frame per sample")
        valid = mask.unsqueeze(-1)
        clean = latent.masked_fill(~valid, 0)
        count = mask.sum(1).to(latent.dtype) * latent.shape[-1]
        rms = (clean.square().sum((1, 2)) / count).clamp_min(1e-16).sqrt()
        hidden = self.normalized_projection(self.norm(clean))
        hidden = (hidden + self.raw_projection(clean / rms[:, None, None])).masked_fill(~valid, 0)
        for block in self.blocks:
            hidden = block(hidden, mask)
        summaries, flags = [], []
        for lag in self.lags:
            if lag >= hidden.shape[1]:
                summaries.append(hidden.new_zeros((len(hidden), self.pair_channels * 2)))
                flags.append(hidden.new_zeros((len(hidden), 1)))
                continue
            left, right = hidden[:, :-lag], hidden[:, lag:]
            pair_mask = mask[:, :-lag] & mask[:, lag:]
            values = self.pair_mlp(torch.cat((left, right, right - left, left * right), -1))
            weights = pair_mask.unsqueeze(-1).to(values.dtype)
            pairs = weights.sum(1)
            available = (pairs > 0).to(values.dtype)
            mean = (values * weights).sum(1) / pairs.clamp_min(1)
            variance = ((values - mean[:, None]).square() * weights).sum(1) / pairs.clamp_min(1)
            std = variance.clamp_min(1e-8).sqrt() * available
            summaries.append(torch.cat((mean, std), -1))
            flags.append(available)
        validity = torch.cat(flags, -1)
        return torch.cat((*summaries, validity), -1), validity

    def forward(self, latent: Tensor, mask: Tensor) -> Tensor:
        pooled, _ = self.features(latent, mask)
        return self.output(self.summary(pooled))
