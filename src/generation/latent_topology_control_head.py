"""Lightweight trajectory-conditioned latent control head for Pitch-3 topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .path_homology_surrogate import (
    FourierStepEmbedding,
    MaskedAttentionStatsPool,
    _resize_mask,
    _sinusoidal_positions,
)
from .pitch3_contract import PITCH3_DIMENSIONS, Pitch3Contract


class Pitch3ControlOutput(NamedTuple):
    coordinate_mean: Tensor
    coordinate_logvar: Tensor
    ood_logit: Tensor
    focus_logit: Tensor


@dataclass(frozen=True, slots=True)
class Pitch3ControlHeadConfig:
    """Small global-control architecture for 180-second ACE latent sequences."""

    latent_dim: int = 64
    model_dim: int = 128
    condition_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 3
    feedforward_dim: int = 512
    temporal_stride: int = 4
    dropout: float = 0.1
    logvar_min: float = -8.0
    logvar_max: float = 4.0


class LatentTopologyControlHead(nn.Module):
    """Map predicted-clean ACE latents to the frozen three-coordinate teacher.

    The head is deliberately global: the exact topology controls summarize a
    complete 180-second trajectory rather than a frame-aligned control curve.
    Time conditioning makes backward-simulated trajectory snapshots usable at
    the same sampling steps where guidance will later be evaluated.
    """

    def __init__(
        self,
        contract: Pitch3Contract,
        config: Pitch3ControlHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Pitch3ControlHeadConfig()
        cfg = self.config
        if cfg.model_dim % cfg.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")
        if cfg.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")
        self.input_norm = nn.LayerNorm(cfg.latent_dim)
        self.input_projection = nn.Linear(cfg.latent_dim, cfg.model_dim)
        self.temporal_downsample = nn.Conv1d(
            cfg.model_dim,
            cfg.model_dim,
            kernel_size=2 * cfg.temporal_stride + 1,
            stride=cfg.temporal_stride,
            padding=cfg.temporal_stride,
            groups=cfg.model_dim,
        )
        self.time_embedding = FourierStepEmbedding(cfg.condition_dim)
        self.condition_projection = nn.Linear(cfg.condition_dim, cfg.model_dim * 2)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.transformer_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.pool = MaskedAttentionStatsPool(cfg.model_dim)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.model_dim * 3 + cfg.condition_dim, cfg.model_dim * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.SiLU(),
        )
        self.coordinate_mean_head = nn.Linear(cfg.model_dim, PITCH3_DIMENSIONS)
        self.coordinate_logvar_head = nn.Linear(cfg.model_dim, PITCH3_DIMENSIONS)
        self.ood_head = nn.Linear(cfg.model_dim, 1)
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

    def encode(
        self,
        latent: Tensor,
        timestep: Tensor | float,
        step_number: Tensor | int,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError(f"latent must have shape [B,T,{self.config.latent_dim}]")
        batch, frames, _ = latent.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch, frames, dtype=torch.bool, device=latent.device)
        else:
            attention_mask = attention_mask.to(device=latent.device, dtype=torch.bool)
        if attention_mask.shape != (batch, frames) or not attention_mask.any(dim=1).all():
            raise ValueError("attention_mask must be [B,T] with a valid frame in every sample")
        time = self._batch_scalar(timestep, batch, latent)
        step = self._batch_scalar(step_number, batch, latent)
        condition = self.time_embedding(time, step).to(dtype=latent.dtype)
        sequence = self.input_projection(self.input_norm(latent))
        sequence = self.temporal_downsample(sequence.transpose(1, 2)).transpose(1, 2)
        mask = _resize_mask(attention_mask, sequence.shape[1])
        scale, shift = self.condition_projection(condition).unsqueeze(1).chunk(2, dim=-1)
        sequence = sequence * (1.0 + scale) + shift
        sequence = sequence + _sinusoidal_positions(
            sequence.shape[1], sequence.shape[2], sequence
        ).unsqueeze(0)
        sequence = self.transformer(sequence, src_key_padding_mask=~mask)
        pooled = self.pool(self.output_norm(sequence), mask)
        return self.fusion(torch.cat((pooled, condition), dim=-1))

    def readout(self, shared: Tensor) -> Pitch3ControlOutput:
        if shared.ndim != 2 or shared.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"shared representation must have shape [B,{self.config.model_dim}]"
            )
        mean = self.coordinate_mean_head(shared)
        logvar = self.coordinate_logvar_head(shared).clamp(
            self.config.logvar_min, self.config.logvar_max
        )
        ood = self.ood_head(shared).squeeze(-1)
        focus = mean.float() @ self.focus_coef + self.focus_intercept
        return Pitch3ControlOutput(mean, logvar, ood, focus)

    def forward(
        self,
        latent: Tensor,
        timestep: Tensor | float,
        step_number: Tensor | int,
        attention_mask: Tensor | None = None,
    ) -> Pitch3ControlOutput:
        return self.readout(self.encode(latent, timestep, step_number, attention_mask))

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
