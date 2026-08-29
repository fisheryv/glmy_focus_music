"""Differentiable 18-D latent-to-Path-Homology surrogate network."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ltsn_contract import FINGERPRINT_DIMENSIONS, FingerprintContract


class LTSNOutput(NamedTuple):
    """Surrogate coordinate distribution, safety score, and frozen Focus readout."""

    coordinate_mean: Tensor
    coordinate_logvar: Tensor
    ood_logit: Tensor
    focus_logit: Tensor


@dataclass(frozen=True, slots=True)
class LTSNConfig:
    """Architecture settings frozen together with each LTSN checkpoint."""

    latent_dim: int = 64
    condition_dim: int = 128
    stem_channels: int = 128
    local_channels: int = 192
    global_channels: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 2
    dropout: float = 0.1
    logvar_min: float = -8.0
    logvar_max: float = 4.0


def _resize_mask(mask: Tensor, length: int) -> Tensor:
    resized = F.adaptive_max_pool1d(mask.float().unsqueeze(1), length)
    return resized.squeeze(1) > 0.5


def _sinusoidal_positions(length: int, channels: int, reference: Tensor) -> Tensor:
    half = channels // 2
    positions = torch.arange(length, device=reference.device, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(half, device=reference.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    ).unsqueeze(0)
    angles = positions * scales
    encoding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return encoding[:, :channels].to(dtype=reference.dtype)


class FourierStepEmbedding(nn.Module):
    """Encode continuous noise time and discrete step number into 128 dimensions."""

    def __init__(self, output_dim: int, frequencies: int = 32) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(0, 3, frequencies), persistent=False)
        input_dim = frequencies * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, timestep: Tensor, step_number: Tensor) -> Tensor:
        """Return a time condition for batch-aligned scalar inputs."""

        values = (timestep.float(), step_number.float() / 8.0)
        features = []
        for value in values:
            angles = 2.0 * math.pi * value.unsqueeze(-1) * self.frequencies.unsqueeze(0)
            features.extend((angles.sin(), angles.cos()))
        return self.mlp(torch.cat(features, dim=-1))


class FiLMResidualBlock(nn.Module):
    """Depthwise-separable temporal residual block with time conditioning."""

    def __init__(self, channels: int, condition_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        """Apply one conditioned residual update to ``[B,C,T]`` features."""

        scale, shift = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.norm(inputs) * (1.0 + scale) + shift
        hidden = self.depthwise(F.silu(hidden))
        hidden = self.dropout(self.pointwise(F.silu(hidden)))
        return inputs + hidden


class ConditionedTransformerBlock(nn.Module):
    """Pre-norm global self-attention block with AdaLN-style time modulation."""

    def __init__(self, channels: int, heads: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(channels)
        self.norm_ffn = nn.LayerNorm(channels)
        self.attention_condition = nn.Linear(condition_dim, channels * 2)
        self.ffn_condition = nn.Linear(condition_dim, channels * 2)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _modulate(hidden: Tensor, parameters: Tensor) -> Tensor:
        scale, shift = parameters.unsqueeze(1).chunk(2, dim=-1)
        return hidden * (1.0 + scale) + shift

    def forward(self, inputs: Tensor, condition: Tensor, valid_mask: Tensor) -> Tensor:
        """Apply conditioned attention while excluding padded global tokens."""

        query = self._modulate(self.norm_attention(inputs), self.attention_condition(condition))
        attended, _ = self.attention(
            query,
            query,
            query,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        hidden = inputs + attended
        feedforward = self._modulate(self.norm_ffn(hidden), self.ffn_condition(condition))
        return hidden + self.ffn(feedforward)


class MaskedAttentionStatsPool(nn.Module):
    """Concatenate masked attention, mean, and standard-deviation pooling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Linear(channels, 1)

    def forward(self, sequence: Tensor, valid_mask: Tensor) -> Tensor:
        """Pool a ``[B,T,C]`` sequence without using padded frames."""

        weights = self.attention(sequence).squeeze(-1).masked_fill(~valid_mask, -torch.inf)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        attended = torch.sum(sequence * weights, dim=1)
        mask = valid_mask.unsqueeze(-1).to(sequence.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = torch.sum(sequence * mask, dim=1) / count
        variance = torch.sum((sequence - mean.unsqueeze(1)).square() * mask, dim=1) / count
        return torch.cat((attended, mean, torch.sqrt(variance.clamp_min(1e-8))), dim=-1)


class PathHomologySurrogate(nn.Module):
    """Predict the frozen Pitch-plus-dual-phase 18-D fingerprint from ACE latents."""

    def __init__(self, contract: FingerprintContract, config: LTSNConfig | None = None) -> None:
        super().__init__()
        self.config = config or LTSNConfig()
        cfg = self.config
        self.input_norm = nn.LayerNorm(cfg.latent_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.latent_dim, cfg.stem_channels, kernel_size=9, stride=5, padding=4),
            nn.GroupNorm(8, cfg.stem_channels),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.time_embedding = FourierStepEmbedding(cfg.condition_dim)
        self.local_projection = nn.Conv1d(cfg.stem_channels, cfg.local_channels, kernel_size=1)
        self.local_blocks = nn.ModuleList(
            FiLMResidualBlock(cfg.local_channels, cfg.condition_dim, dilation, cfg.dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.global_downsample = nn.Conv1d(
            cfg.local_channels, cfg.global_channels, kernel_size=7, stride=4, padding=3
        )
        self.global_blocks = nn.ModuleList(
            FiLMResidualBlock(cfg.global_channels, cfg.condition_dim, dilation, cfg.dropout)
            for dilation in (1, 4, 16, 64)
        )
        self.transformer = nn.ModuleList(
            ConditionedTransformerBlock(
                cfg.global_channels, cfg.transformer_heads, cfg.condition_dim, cfg.dropout
            )
            for _ in range(cfg.transformer_layers)
        )
        self.local_pool = MaskedAttentionStatsPool(cfg.local_channels)
        self.global_pool = MaskedAttentionStatsPool(cfg.global_channels)
        fusion_dim = cfg.local_channels * 3 + cfg.global_channels * 3 + cfg.condition_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        self.coordinate_mean_head = nn.Linear(256, FINGERPRINT_DIMENSIONS)
        self.coordinate_logvar_head = nn.Linear(256, FINGERPRINT_DIMENSIONS)
        self.ood_head = nn.Linear(256, 1)
        self.register_buffer(
            "focus_coef", torch.tensor(contract.classifier_coef, dtype=torch.float32)
        )
        self.register_buffer(
            "focus_intercept", torch.tensor(contract.classifier_intercept, dtype=torch.float32)
        )

    @staticmethod
    def _batch_scalar(value: Tensor | float | int, batch: int, reference: Tensor) -> Tensor:
        tensor = torch.as_tensor(value, device=reference.device, dtype=torch.float32).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch)
        if tensor.numel() != batch:
            raise ValueError("timestep and step_number must be scalar or batch-aligned")
        return tensor

    def forward(
        self,
        latent: Tensor,
        timestep: Tensor | float,
        step_number: Tensor | int,
        attention_mask: Tensor | None = None,
    ) -> LTSNOutput:
        """Predict 18-D coordinates for a clean-latent estimate shaped ``[B,T,64]``."""

        if latent.ndim != 3 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError(f"latent must have shape [B,T,{self.config.latent_dim}]")
        batch, frames, _ = latent.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch, frames, dtype=torch.bool, device=latent.device)
        else:
            attention_mask = attention_mask.to(device=latent.device, dtype=torch.bool)
        if attention_mask.shape != (batch, frames) or not attention_mask.any(dim=1).all():
            raise ValueError(
                "attention_mask must be [B,T] with at least one valid frame per sample"
            )
        time = self._batch_scalar(timestep, batch, latent)
        step = self._batch_scalar(step_number, batch, latent)
        condition = self.time_embedding(time, step).to(dtype=latent.dtype)

        stem = self.stem(self.input_norm(latent).transpose(1, 2))
        local_mask = _resize_mask(attention_mask, stem.shape[-1])
        local = self.local_projection(stem)
        for block in self.local_blocks:
            local = block(local, condition)

        global_features = self.global_downsample(local)
        global_mask = _resize_mask(local_mask, global_features.shape[-1])
        for block in self.global_blocks:
            global_features = block(global_features, condition)
        global_sequence = global_features.transpose(1, 2)
        global_sequence = global_sequence + _sinusoidal_positions(
            global_sequence.shape[1], global_sequence.shape[2], global_sequence
        ).unsqueeze(0)
        for block in self.transformer:
            global_sequence = block(global_sequence, condition, global_mask)

        pooled = torch.cat(
            (
                self.local_pool(local.transpose(1, 2), local_mask),
                self.global_pool(global_sequence, global_mask),
                condition,
            ),
            dim=-1,
        )
        shared = self.fusion(pooled)
        coordinate_mean = self.coordinate_mean_head(shared)
        coordinate_logvar = self.coordinate_logvar_head(shared).clamp(
            self.config.logvar_min, self.config.logvar_max
        )
        ood_logit = self.ood_head(shared).squeeze(-1)
        focus_logit = coordinate_mean.float() @ self.focus_coef + self.focus_intercept
        return LTSNOutput(coordinate_mean, coordinate_logvar, ood_logit, focus_logit)
