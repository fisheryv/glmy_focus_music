"""Prompt-conditioned local topology energy model and Step-4 guidance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ltsn_contract import LTSNContractError
from .path_homology_surrogate import MaskedAttentionStatsPool, _resize_mask, _sinusoidal_positions


class Pitch3LTEOutput(NamedTuple):
    """Direct scalar topology energy; lower is closer to the frozen Pitch-3 band."""

    energy: Tensor


class Pitch3LTEPotentialComponents(NamedTuple):
    """Conservative V3.3 potential decomposition; ``energy`` is always the sum."""

    energy: Tensor
    global_energy: Tensor
    local_energy: Tensor
    coordinates: Tensor | None


@dataclass(frozen=True, slots=True)
class Pitch3LTEConfig:
    latent_dim: int = 64
    text_dim: int = 1024
    model_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 3
    feedforward_dim: int = 512
    temporal_stride: int = 4
    dropout: float = 0.1
    fusion_mode: str = "joint_v3"
    prompt_residual_scale: float = 0.25
    latent_stem_mode: str = "normalized_v3"
    potential_mode: str = "single_v3"
    local_residual_scale: float = 1.0
    coordinate_lower: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coordinate_upper: tuple[float, float, float] = (1.0, 1.0, 1.0)
    coordinate_center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    coordinate_distance_weights: tuple[float, float, float] = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )
    prompt_film_fraction: float = 0.25

    def validate(self) -> None:
        if self.model_dim % self.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")
        if min(self.latent_dim, self.text_dim, self.model_dim, self.temporal_stride) < 1:
            raise ValueError("V3-LTE dimensions and temporal stride must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("V3-LTE dropout must lie in [0,1)")
        if self.fusion_mode not in {"joint_v3", "latent_primary_residual_v31"}:
            raise ValueError("unknown V3-LTE fusion mode")
        if self.prompt_residual_scale <= 0:
            raise ValueError("V3-LTE prompt residual scale must be positive")
        if self.latent_stem_mode not in {"normalized_v3", "dual_rms_v32"}:
            raise ValueError("unknown V3-LTE latent stem mode")
        if self.potential_mode not in {
            "single_v3",
            "global_local_v33",
            "structured_coordinate_v34",
            "structured_anchored_v35",
        }:
            raise ValueError("unknown V3-LTE potential mode")
        if self.potential_mode in {
            "global_local_v33",
            "structured_coordinate_v34",
            "structured_anchored_v35",
        } and self.fusion_mode != "latent_primary_residual_v31":
            raise ValueError("decomposed LTE potentials require latent-primary fusion")
        if self.local_residual_scale <= 0:
            raise ValueError("V3-LTE local residual scale must be positive")
        coordinate_vectors = (
            self.coordinate_lower,
            self.coordinate_upper,
            self.coordinate_center,
            self.coordinate_distance_weights,
        )
        if any(len(values) != 3 for values in coordinate_vectors):
            raise ValueError("V3.4 coordinate contract must contain three values")
        if self.potential_mode in {
            "structured_coordinate_v34",
            "structured_anchored_v35",
        }:
            if any(
                lower >= upper
                for lower, upper in zip(
                    self.coordinate_lower, self.coordinate_upper, strict=True
                )
            ):
                raise ValueError("V3.4 coordinate bounds are invalid")
            if any(value <= 0 for value in self.coordinate_distance_weights) or not math.isclose(
                sum(self.coordinate_distance_weights), 1.0, abs_tol=1e-8
            ):
                raise ValueError("V3.4 coordinate weights must be positive and sum to one")
            if not 0.0 < self.prompt_film_fraction < 1.0:
                raise ValueError("V3.4 prompt FiLM fraction must lie in (0,1)")


class PromptConditionedTopologyEnergy(nn.Module):
    """Predict ``log1p(exact Pitch-3 target-band distance)`` from ``(z,c)``.

    The latent branch keeps the proven 180-second LTCH temporal encoder.  The
    prompt branch consumes frozen ACE text-token states, pools them without a
    prompt-ID table, and interacts with the latent representation through
    product and absolute-difference features before the scalar readout.
    """

    def __init__(self, config: Pitch3LTEConfig | None = None) -> None:
        super().__init__()
        self.config = config or Pitch3LTEConfig()
        self.config.validate()
        cfg = self.config
        self.latent_norm = nn.LayerNorm(cfg.latent_dim)
        self.latent_projection = nn.Linear(cfg.latent_dim, cfg.model_dim)
        if cfg.latent_stem_mode == "dual_rms_v32":
            self.raw_latent_projection = nn.Linear(cfg.latent_dim, cfg.model_dim, bias=False)
            nn.init.zeros_(self.raw_latent_projection.weight)
        self.temporal_downsample = nn.Conv1d(
            cfg.model_dim,
            cfg.model_dim,
            kernel_size=2 * cfg.temporal_stride + 1,
            stride=cfg.temporal_stride,
            padding=cfg.temporal_stride,
            groups=cfg.model_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.transformer_layers, enable_nested_tensor=False
        )
        self.latent_output_norm = nn.LayerNorm(cfg.model_dim)
        self.latent_pool = MaskedAttentionStatsPool(cfg.model_dim)
        self.latent_fusion = nn.Sequential(
            nn.Linear(cfg.model_dim * 3, cfg.model_dim * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.SiLU(),
        )

        # ACE hidden states are frozen.  Mean and standard deviation retain
        # prompt content and token-distribution information with few trainable
        # parameters, which is important for the 320-prompt low-resource set.
        self.text_input_norm = nn.LayerNorm(cfg.text_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(cfg.text_dim * 2, cfg.model_dim * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.SiLU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(cfg.model_dim * 4, cfg.model_dim * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.SiLU(),
        )
        if cfg.potential_mode in {
            "structured_coordinate_v34",
            "structured_anchored_v35",
        }:
            self.coordinate_head = nn.Sequential(
                nn.Linear(cfg.model_dim, cfg.model_dim),
                nn.SiLU(),
                nn.Linear(cfg.model_dim, 3),
            )
            self.coordinate_prompt_film = nn.Linear(cfg.model_dim, 6)
            nn.init.zeros_(self.coordinate_prompt_film.weight)
            nn.init.zeros_(self.coordinate_prompt_film.bias)
            self.register_buffer(
                "coordinate_lower",
                torch.tensor(cfg.coordinate_lower, dtype=torch.float32),
            )
            self.register_buffer(
                "coordinate_upper",
                torch.tensor(cfg.coordinate_upper, dtype=torch.float32),
            )
            self.register_buffer(
                "coordinate_center",
                torch.tensor(cfg.coordinate_center, dtype=torch.float32),
            )
            self.register_buffer(
                "coordinate_distance_weights",
                torch.tensor(cfg.coordinate_distance_weights, dtype=torch.float32),
            )
        elif cfg.fusion_mode == "joint_v3":
            self.energy_head = nn.Linear(cfg.model_dim, 1)
        else:
            self.latent_energy_head = nn.Sequential(
                nn.Linear(cfg.model_dim, cfg.model_dim),
                nn.SiLU(),
                nn.Linear(cfg.model_dim, 1),
            )
            self.interaction_energy_head = nn.Linear(cfg.model_dim, 1)
            nn.init.zeros_(self.interaction_energy_head.weight)
            nn.init.zeros_(self.interaction_energy_head.bias)
        if cfg.potential_mode in {
            "global_local_v33",
            "structured_coordinate_v34",
            "structured_anchored_v35",
        }:
            self.local_energy_head = nn.Sequential(
                nn.Linear(cfg.model_dim, cfg.model_dim),
                nn.SiLU(),
                nn.Linear(cfg.model_dim, 1),
            )
            nn.init.zeros_(self.local_energy_head[-1].weight)
            nn.init.zeros_(self.local_energy_head[-1].bias)

    @staticmethod
    def _validate_mask(values: Tensor, mask: Tensor, name: str) -> Tensor:
        checked = mask.to(device=values.device, dtype=torch.bool)
        if checked.shape != values.shape[:2] or not checked.any(dim=1).all():
            raise ValueError(f"{name} must match [B,T] and retain at least one token")
        return checked

    def encode_latent(self, latent: Tensor, attention_mask: Tensor) -> Tensor:
        cfg = self.config
        if latent.ndim != 3 or latent.shape[-1] != cfg.latent_dim:
            raise ValueError(f"latent must have shape [B,T,{cfg.latent_dim}]")
        mask = self._validate_mask(latent, attention_mask, "attention_mask")
        sequence = self.latent_projection(self.latent_norm(latent))
        if self.config.latent_stem_mode == "dual_rms_v32":
            weights = mask.to(latent.dtype).unsqueeze(-1)
            count = weights.sum(dim=(1, 2)).clamp_min(1.0) * latent.shape[-1]
            rms = torch.sqrt((latent.square() * weights).sum(dim=(1, 2)) / count)
            raw = latent / rms.clamp_min(1e-8)[:, None, None]
            sequence = sequence + self.raw_latent_projection(raw)
        sequence = self.temporal_downsample(sequence.transpose(1, 2)).transpose(1, 2)
        mask = _resize_mask(mask, sequence.shape[1])
        sequence = sequence + _sinusoidal_positions(
            sequence.shape[1], sequence.shape[2], sequence
        ).unsqueeze(0)
        sequence = self.latent_transformer(sequence, src_key_padding_mask=~mask)
        pooled = self.latent_pool(self.latent_output_norm(sequence), mask)
        return self.latent_fusion(pooled)

    def encode_prompt(self, text_hidden: Tensor, text_mask: Tensor) -> Tensor:
        cfg = self.config
        if text_hidden.ndim != 3 or text_hidden.shape[-1] != cfg.text_dim:
            raise ValueError(f"ACE text_hidden must have shape [B,L,{cfg.text_dim}]")
        mask = self._validate_mask(text_hidden, text_mask, "text_mask")
        values = self.text_input_norm(text_hidden)
        weights = mask.to(values.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (values * weights).sum(dim=1) / count
        variance = ((values - mean[:, None, :]).square() * weights).sum(dim=1) / count
        return self.text_projection(torch.cat((mean, variance.clamp_min(0).sqrt()), dim=1))

    def forward(
        self,
        latent: Tensor,
        attention_mask: Tensor,
        text_hidden: Tensor,
        text_mask: Tensor,
        anchor_latent: Tensor | None = None,
        anchor_attention_mask: Tensor | None = None,
    ) -> Pitch3LTEOutput:
        return Pitch3LTEOutput(
            self.potential_components(
                latent,
                attention_mask,
                text_hidden,
                text_mask,
                anchor_latent=anchor_latent,
                anchor_attention_mask=anchor_attention_mask,
            ).energy
        )

    def potential_components(
        self,
        latent: Tensor,
        attention_mask: Tensor,
        text_hidden: Tensor,
        text_mask: Tensor,
        *,
        anchor_latent: Tensor | None = None,
        anchor_attention_mask: Tensor | None = None,
    ) -> Pitch3LTEPotentialComponents:
        latent_state = self.encode_latent(latent, attention_mask)
        prompt_state = self.encode_prompt(text_hidden, text_mask)
        anchor_state: Tensor | None = None
        if self.config.potential_mode == "structured_anchored_v35":
            if anchor_latent is None:
                anchor_state = latent_state.detach()
            else:
                if anchor_attention_mask is None:
                    raise ValueError("V3.5 anchor mask is required with anchor latent")
                anchor_state = self.encode_latent(
                    anchor_latent.detach(), anchor_attention_mask
                ).detach()
        return self.potential_components_from_states(
            latent_state,
            prompt_state,
            anchor_latent_state=anchor_state,
        )

    def energy_from_states(
        self,
        latent_state: Tensor,
        prompt_state: Tensor,
        anchor_latent_state: Tensor | None = None,
    ) -> Pitch3LTEOutput:
        """Read energy from reusable latent/prompt states.

        V3.1 training uses this boundary for prompt dropout and shuffled-prompt
        consistency without recomputing the expensive temporal transformer.
        """

        components = self.potential_components_from_states(
            latent_state,
            prompt_state,
            anchor_latent_state=anchor_latent_state,
        )
        return Pitch3LTEOutput(components.energy)

    def potential_components_from_states(
        self,
        latent_state: Tensor,
        prompt_state: Tensor,
        anchor_latent_state: Tensor | None = None,
    ) -> Pitch3LTEPotentialComponents:
        """Return the global and conservative local-residual scalar potentials."""

        if prompt_state.shape[0] == 1 and latent_state.shape[0] != 1:
            prompt_state = prompt_state.expand(latent_state.shape[0], -1)
        if prompt_state.shape != latent_state.shape:
            raise ValueError("latent and ACE prompt batches must align")
        fused = self.joint(
            torch.cat(
                (
                    latent_state,
                    prompt_state,
                    latent_state * prompt_state,
                    (latent_state - prompt_state).abs(),
                ),
                dim=1,
            )
        )
        coordinates: Tensor | None = None
        if self.config.potential_mode in {
            "structured_coordinate_v34",
            "structured_anchored_v35",
        }:
            raw_coordinates = self.coordinate_head(latent_state)
            film_scale, film_offset = self.coordinate_prompt_film(prompt_state).chunk(2, dim=1)
            fraction = self.config.prompt_film_fraction
            scale = 1.0 + fraction * torch.tanh(film_scale)
            width = self.coordinate_upper - self.coordinate_lower
            offset = fraction * torch.tanh(film_offset) * width
            coordinates = self.coordinate_center + scale * (
                raw_coordinates - self.coordinate_center
            ) + offset
            below = torch.relu(self.coordinate_lower - coordinates)
            above = torch.relu(coordinates - self.coordinate_upper)
            band = ((below.square() + above.square()) * self.coordinate_distance_weights).sum(
                dim=1
            )
            global_energy = torch.log1p(band)
        elif self.config.fusion_mode == "joint_v3":
            global_energy = self.energy_head(fused).squeeze(-1)
        else:
            latent_energy = self.latent_energy_head(latent_state).squeeze(-1)
            interaction = self.config.prompt_residual_scale * torch.tanh(
                self.interaction_energy_head(fused).squeeze(-1)
            )
            global_energy = latent_energy + interaction
        if self.config.potential_mode == "structured_anchored_v35":
            if anchor_latent_state is None:
                raise ValueError("V3.5 anchored potential requires an anchor latent state")
            if anchor_latent_state.shape != latent_state.shape:
                raise ValueError("V3.5 anchor and current latent states must align")
            anchor_fused = self.joint(
                torch.cat(
                    (
                        anchor_latent_state,
                        prompt_state,
                        anchor_latent_state * prompt_state,
                        (anchor_latent_state - prompt_state).abs(),
                    ),
                    dim=1,
                )
            )
            current_residual = self.local_energy_head(fused).squeeze(-1)
            anchor_residual = self.local_energy_head(anchor_fused).squeeze(-1)
            local_energy = self.config.local_residual_scale * (
                current_residual - anchor_residual
            )
        elif self.config.potential_mode in {"global_local_v33", "structured_coordinate_v34"}:
            local_energy = self.config.local_residual_scale * self.local_energy_head(
                fused
            ).squeeze(-1)
        else:
            local_energy = torch.zeros_like(global_energy)
        energy = global_energy + local_energy
        return Pitch3LTEPotentialComponents(
            energy.float(),
            global_energy.float(),
            local_energy.float(),
            coordinates.float() if coordinates is not None else None,
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class Pitch3LTEEnergyEnsemble(nn.Module):
    """Equal-contract average of scalar energies; its latent gradient is averaged too."""

    def __init__(
        self,
        models: list[PromptConditionedTopologyEnergy],
        weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("V3-LTE ensemble requires at least one model")
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = torch.ones(len(models), dtype=torch.float32)
        checked = torch.as_tensor(weights, dtype=torch.float32)
        if checked.shape != (len(models),) or not torch.isfinite(checked).all():
            raise ValueError("V3-LTE ensemble weights must be finite and match members")
        if (checked <= 0).any():
            raise ValueError("V3-LTE ensemble weights must be positive")
        self.register_buffer("weights", checked / checked.sum())

    def forward(
        self,
        latent: Tensor,
        attention_mask: Tensor,
        text_hidden: Tensor,
        text_mask: Tensor,
        anchor_latent: Tensor | None = None,
        anchor_attention_mask: Tensor | None = None,
    ) -> Pitch3LTEOutput:
        return Pitch3LTEOutput(
            self.potential_components(
                latent,
                attention_mask,
                text_hidden,
                text_mask,
                anchor_latent=anchor_latent,
                anchor_attention_mask=anchor_attention_mask,
            ).energy
        )

    def potential_components(
        self,
        latent: Tensor,
        attention_mask: Tensor,
        text_hidden: Tensor,
        text_mask: Tensor,
        *,
        anchor_latent: Tensor | None = None,
        anchor_attention_mask: Tensor | None = None,
    ) -> Pitch3LTEPotentialComponents:
        components = [
            model.potential_components(
                latent,
                attention_mask,
                text_hidden,
                text_mask,
                anchor_latent=anchor_latent,
                anchor_attention_mask=anchor_attention_mask,
            )
            for model in self.models
        ]
        energies = torch.stack([item.energy for item in components], dim=0)
        global_energies = torch.stack(
            [item.global_energy for item in components], dim=0
        )
        local_energies = torch.stack([item.local_energy for item in components], dim=0)
        weights = self.weights.to(device=energies.device, dtype=energies.dtype)
        coordinate_values = [item.coordinates for item in components]
        coordinates = None
        if all(value is not None for value in coordinate_values):
            stacked = torch.stack(
                [value for value in coordinate_values if value is not None], dim=0
            )
            coordinates = (stacked * weights[:, None, None]).sum(dim=0).float()
        return Pitch3LTEPotentialComponents(
            (energies * weights[:, None]).sum(dim=0).float(),
            (global_energies * weights[:, None]).sum(dim=0).float(),
            (local_energies * weights[:, None]).sum(dim=0).float(),
            coordinates,
        )


@dataclass(frozen=True, slots=True)
class Pitch3LTEGuidanceConfig:
    enabled: bool = False
    authorization_scope: str = "development_only"
    qualification_passed: bool = False
    correction_step: int = 4
    training_radius_ratio: float = 0.05
    maximum_update_fraction: float = 0.5
    low_pass_kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0)
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.authorization_scope not in {"development_only", "qualified"}:
            raise LTSNContractError("unknown V3-LTE authorization scope")
        if (
            self.authorization_scope == "qualified"
            and self.enabled
            and not self.qualification_passed
        ):
            raise LTSNContractError("V3-LTE production guidance requires passed qualification")
        if self.authorization_scope == "development_only" and self.qualification_passed:
            raise LTSNContractError("development-only V3-LTE cannot claim qualification")
        if self.correction_step != 4:
            raise LTSNContractError("V3-LTE is frozen to exactly one Step-4 correction")
        if not math.isclose(self.training_radius_ratio, 0.05, abs_tol=1e-12):
            raise LTSNContractError("V3-LTE training radius must remain 5% latent RMS")
        if not math.isclose(self.maximum_update_fraction, 0.5, abs_tol=1e-12):
            raise LTSNContractError("V3-LTE update cap must remain half the training radius")


class Pitch3LTEGuidanceDiagnostics(NamedTuple):
    applied: Tensor
    energy_before: Tensor
    energy_after: Tensor
    gradient_rms: Tensor
    clean_update_rms: Tensor
    mapped_update_rms: Tensor
    backtracked: Tensor
    no_op_reason_code: Tensor


class Pitch3LTECorrector:
    """One-shot, trust-region, energy-decreasing Step-4 ACE correction."""

    def __init__(
        self,
        model: PromptConditionedTopologyEnergy,
        prompt_hidden: Tensor,
        prompt_mask: Tensor,
        checkpoint_metadata: Mapping[str, object],
        config: Pitch3LTEGuidanceConfig | None = None,
    ) -> None:
        self.config = config or Pitch3LTEGuidanceConfig()
        self.config.validate()
        if checkpoint_metadata.get("model_family") != "pitch3_prompt_conditioned_local_energy_v3":
            raise LTSNContractError("checkpoint is not a V3-LTE scalar-energy model")
        if checkpoint_metadata.get("guidance_steps") != [4]:
            raise LTSNContractError("V3-LTE checkpoint changed the frozen guidance step")
        self.model = model.eval().requires_grad_(False)
        self.prompt_hidden = prompt_hidden.detach()
        self.prompt_mask = prompt_mask.detach().bool()
        self._telemetry: list[dict[str, object]] = []

    @staticmethod
    def _masked_rms(values: Tensor, mask: Tensor) -> Tensor:
        expanded = mask.unsqueeze(-1).expand_as(values).to(values.dtype)
        count = expanded.sum(dim=(1, 2)).clamp_min(1.0)
        return torch.sqrt((values.square() * expanded).sum(dim=(1, 2)) / count)

    def _low_pass(self, gradient: Tensor) -> Tensor:
        kernel = torch.tensor(
            self.config.low_pass_kernel, device=gradient.device, dtype=gradient.dtype
        )
        kernel = kernel / kernel.sum()
        weights = kernel.reshape(1, 1, -1).expand(gradient.shape[-1], 1, -1)
        padding = kernel.numel() // 2
        values = F.pad(gradient.transpose(1, 2), (padding, padding), mode="replicate")
        return F.conv1d(values, weights, groups=gradient.shape[-1]).transpose(1, 2)

    def propose_clean_update(
        self, clean: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Pitch3LTEGuidanceDiagnostics]:
        mask = attention_mask.to(device=clean.device, dtype=torch.bool)
        if mask.shape != clean.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("V3-LTE attention mask must align with clean latent")
        prompt_hidden = self.prompt_hidden.to(device=clean.device, dtype=clean.dtype)
        prompt_mask = self.prompt_mask.to(device=clean.device)
        if prompt_hidden.shape[0] == 1 and clean.shape[0] != 1:
            prompt_hidden = prompt_hidden.expand(clean.shape[0], -1, -1)
            prompt_mask = prompt_mask.expand(clean.shape[0], -1)
        with torch.inference_mode(False), torch.enable_grad():
            variable = clean.detach().float().clone().requires_grad_(True)
            anchor = clean.detach().float().clone()
            before = self.model(
                variable,
                mask,
                prompt_hidden.float(),
                prompt_mask,
                anchor_latent=anchor,
                anchor_attention_mask=mask,
            ).energy
            gradient = torch.autograd.grad(before.sum(), variable, allow_unused=False)[0]
            gradient = self._low_pass(gradient)
            finite = (
                torch.isfinite(before)
                & torch.isfinite(variable).all(dim=(1, 2))
                & torch.isfinite(gradient).all(dim=(1, 2))
            )
            gradient_rms = self._masked_rms(gradient, mask)
            clean_rms = self._masked_rms(variable, mask)
            maximum = (
                self.config.training_radius_ratio * self.config.maximum_update_fraction * clean_rms
            )
            valid = finite & (gradient_rms > self.config.epsilon) & (maximum > 0)
            scale = maximum / gradient_rms.clamp_min(self.config.epsilon)
            update = -gradient * scale[:, None, None]
            update = update * mask[:, :, None].to(update.dtype)
            candidate = variable + update
            after = self.model(
                candidate,
                mask,
                prompt_hidden.float(),
                prompt_mask,
                anchor_latent=anchor,
                anchor_attention_mask=mask,
            ).energy
            needs_backtrack = valid & (~torch.isfinite(after) | (after > before))
            update = torch.where(needs_backtrack[:, None, None], update * 0.5, update)
            candidate = variable + update
            after = self.model(
                candidate,
                mask,
                prompt_hidden.float(),
                prompt_mask,
                anchor_latent=anchor,
                anchor_attention_mask=mask,
            ).energy
            energy_decreased = torch.isfinite(after) & (after < before - self.config.epsilon)
            applied = (
                valid & energy_decreased & (self._masked_rms(update, mask) > self.config.epsilon)
            )
            update = torch.where(applied[:, None, None], update, torch.zeros_like(update))
            corrected = variable + update
            reason = torch.zeros(clean.shape[0], dtype=torch.int64, device=clean.device)
            reason |= (~finite).to(torch.int64) * 1
            reason |= (gradient_rms <= self.config.epsilon).to(torch.int64) * 2
            reason |= (maximum <= 0).to(torch.int64) * 4
            reason |= (~energy_decreased).to(torch.int64) * 8
            reason = torch.where(applied, torch.zeros_like(reason), reason)
            update_rms = self._masked_rms(update, mask)
        diagnostics = Pitch3LTEGuidanceDiagnostics(
            applied.detach(),
            before.detach(),
            after.detach(),
            gradient_rms.detach(),
            update_rms.detach(),
            torch.zeros_like(update_rms),
            needs_backtrack.detach(),
            reason.detach(),
        )
        return corrected.detach().to(dtype=clean.dtype), diagnostics

    def apply_with_diagnostics(
        self,
        *,
        xt_next: Tensor,
        xt_before_step: Tensor,
        velocity: Tensor,
        timestep: float,
        next_timestep: float,
        step_index: int,
        attention_mask: Tensor,
        repaint_mask: Tensor | None = None,
    ) -> tuple[Tensor, Pitch3LTEGuidanceDiagnostics | None]:
        step_number = step_index + 1
        if not self.config.enabled or step_number != self.config.correction_step:
            return xt_next, None
        if xt_next.shape != xt_before_step.shape or velocity.shape != xt_before_step.shape:
            raise ValueError("V3-LTE sampler tensors must have identical shapes")
        mask = attention_mask.to(device=xt_next.device, dtype=torch.bool)
        if repaint_mask is not None:
            mask = mask & repaint_mask.to(device=xt_next.device, dtype=torch.bool)
        clean = xt_before_step.float() - float(timestep) * velocity.float()
        corrected_clean, diagnostics = self.propose_clean_update(clean, mask)
        clean_update = corrected_clean.float() - clean
        mapped = (1.0 - float(next_timestep)) * clean_update
        mapped_rms = self._masked_rms(mapped, mask)
        corrected = xt_next + mapped.to(dtype=xt_next.dtype)
        corrected = torch.where(diagnostics.applied[:, None, None], corrected, xt_next)
        diagnostics = diagnostics._replace(mapped_update_rms=mapped_rms.detach())
        return corrected, diagnostics

    def __call__(self, **kwargs: object) -> Tensor:
        corrected, diagnostics = self.apply_with_diagnostics(**kwargs)  # type: ignore[arg-type]
        if diagnostics is not None:
            self._telemetry.append(
                {
                    "step_number": int(kwargs["step_index"]) + 1,
                    "applied": diagnostics.applied.cpu().tolist(),
                    "energy_before": diagnostics.energy_before.cpu().tolist(),
                    "energy_after": diagnostics.energy_after.cpu().tolist(),
                    "gradient_rms": diagnostics.gradient_rms.cpu().tolist(),
                    "clean_update_rms": diagnostics.clean_update_rms.cpu().tolist(),
                    "mapped_update_rms": diagnostics.mapped_update_rms.cpu().tolist(),
                    "backtracked": diagnostics.backtracked.cpu().tolist(),
                    "no_op_reason_code": diagnostics.no_op_reason_code.cpu().tolist(),
                }
            )
        return corrected

    def drain_telemetry(self) -> list[dict[str, object]]:
        values, self._telemetry = self._telemetry, []
        return values
