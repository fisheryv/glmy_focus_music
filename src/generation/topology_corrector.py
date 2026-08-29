"""Safety-gated mid-step topology correction for ACE-Step 1.5 Turbo."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ltsn_contract import (
    FingerprintContract,
    LTSNContractError,
    validate_checkpoint_metadata,
)

_ALLOWED_RMS_RATIOS = (0.0025, 0.005, 0.01)


@dataclass(frozen=True, slots=True)
class TopologyCorrectorConfig:
    """Frozen sampling controls and calibrated no-op thresholds."""

    enabled: bool = False
    qualification_passed: bool = False
    guidance_scale: float = 1.0
    rms_clip_ratio: float = 0.005
    step_weights: Mapping[int, float] = field(
        default_factory=lambda: {4: 0.5, 5: 1.0, 6: 0.5}
    )
    ood_probability_threshold: float | None = None
    max_aleatoric_variance: float | None = None
    max_epistemic_variance: float | None = None
    max_interval_width: float | None = None
    variance_scale: tuple[float, ...] = (1.0,) * 18
    low_pass_kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0)
    epsilon: float = 1e-8

    def validate(self) -> None:
        """Reject activation without qualification and calibrated thresholds."""

        if self.rms_clip_ratio not in _ALLOWED_RMS_RATIOS:
            raise LTSNContractError("rms_clip_ratio must be 0.25%, 0.5%, or 1.0%")
        if self.guidance_scale < 0 or not math.isfinite(self.guidance_scale):
            raise LTSNContractError("guidance_scale must be finite and non-negative")
        if len(self.variance_scale) != 18 or any(
            value <= 0 or not math.isfinite(value) for value in self.variance_scale
        ):
            raise LTSNContractError("variance_scale must contain 18 finite positive values")
        if not self.enabled:
            return
        if not self.qualification_passed:
            raise LTSNContractError("sampling guidance requires passed LTSN qualification")
        thresholds = (
            self.ood_probability_threshold,
            self.max_aleatoric_variance,
            self.max_epistemic_variance,
            self.max_interval_width,
        )
        if any(value is None or value < 0 or not math.isfinite(value) for value in thresholds):
            raise LTSNContractError("enabled guidance requires frozen finite safety thresholds")


@dataclass(frozen=True, slots=True)
class TopologyCorrectionDiagnostics:
    """Per-sample audit values produced by one corrector invocation."""

    applied: Tensor
    focus_logit: Tensor
    ood_probability: Tensor
    aleatoric_variance: Tensor
    epistemic_variance: Tensor
    interval_width: Tensor


class TopologyCorrector:
    """Differentiate only through a frozen LTSN ensemble and RMS-clip its update."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        contract: FingerprintContract,
        checkpoint_metadata: Sequence[Mapping[str, object]],
        config: TopologyCorrectorConfig | None = None,
    ) -> None:
        if not models or len(models) != len(checkpoint_metadata):
            raise LTSNContractError("models and checkpoint metadata must be non-empty and aligned")
        self.config = config or TopologyCorrectorConfig()
        self.config.validate()
        for metadata in checkpoint_metadata:
            validate_checkpoint_metadata(metadata, contract)
        self.models = tuple(models)
        self.contract = contract
        for model in self.models:
            model.eval()
            model.requires_grad_(False)

    @property
    def is_active(self) -> bool:
        """Return whether the corrector is qualified and configured to change samples."""

        return self.config.enabled and self.config.guidance_scale > 0

    @staticmethod
    def _mask(
        attention_mask: Tensor,
        repaint_mask: Tensor | None,
        latent: Tensor,
    ) -> Tensor:
        mask = attention_mask.to(device=latent.device, dtype=torch.bool)
        if mask.shape != latent.shape[:2]:
            raise ValueError("attention_mask must match latent [B,T]")
        if repaint_mask is not None:
            repaint = repaint_mask.to(device=latent.device, dtype=torch.bool)
            if repaint.shape != mask.shape:
                raise ValueError("repaint_mask must match latent [B,T]")
            mask = mask & repaint
        return mask

    def _low_pass(self, gradient: Tensor) -> Tensor:
        kernel = torch.tensor(
            self.config.low_pass_kernel, device=gradient.device, dtype=gradient.dtype
        )
        kernel = kernel / kernel.sum()
        channels = gradient.shape[-1]
        weights = kernel.reshape(1, 1, -1).expand(channels, 1, -1)
        channel_first = gradient.transpose(1, 2)
        padding = kernel.numel() // 2
        padded = F.pad(channel_first, (padding, padding), mode="replicate")
        return F.conv1d(padded, weights, groups=channels).transpose(1, 2)

    def _rms_clip(self, update: Tensor, clean: Tensor, mask: Tensor) -> Tensor:
        expanded = mask.unsqueeze(-1).expand_as(update).to(update.dtype)
        count = expanded.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        clean_rms = torch.sqrt((clean.square() * expanded).sum((1, 2), keepdim=True) / count)
        update_rms = torch.sqrt((update.square() * expanded).sum((1, 2), keepdim=True) / count)
        maximum = self.config.rms_clip_ratio * clean_rms
        scale = torch.minimum(
            torch.ones_like(update_rms), maximum / (update_rms + self.config.epsilon)
        )
        return update * scale * expanded

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
    ) -> tuple[Tensor, TopologyCorrectionDiagnostics | None]:
        """Apply one qualified correction; all unsafe samples remain bitwise unchanged."""

        step_number = step_index + 1
        step_weight = float(self.config.step_weights.get(step_number, 0.0))
        if not self.is_active or step_weight == 0.0:
            return xt_next, None
        if xt_next.shape != xt_before_step.shape or velocity.shape != xt_before_step.shape:
            raise ValueError("xt_next, xt_before_step, and velocity shapes must match")
        valid_mask = self._mask(attention_mask, repaint_mask, xt_before_step)
        if not valid_mask.any():
            return xt_next, None

        with torch.inference_mode(False), torch.enable_grad():
            clean = (xt_before_step.float() - float(timestep) * velocity.float()).detach().clone()
            clean.requires_grad_(True)
            outputs = [
                model(clean, float(timestep), step_number, valid_mask) for model in self.models
            ]
            means = torch.stack([output.coordinate_mean.float() for output in outputs])
            logvars = torch.stack([output.coordinate_logvar.float() for output in outputs])
            scores = torch.stack([output.focus_logit.float() for output in outputs])
            ood_logits = torch.stack([output.ood_logit.float() for output in outputs])
            mean_score = scores.mean(dim=0)
            scale = torch.tensor(
                self.config.variance_scale, device=clean.device, dtype=torch.float32
            ).unsqueeze(0)
            aleatoric_coordinates = torch.exp(logvars).mean(dim=0) * scale
            epistemic_coordinates = means.var(dim=0, unbiased=False) * scale
            aleatoric = aleatoric_coordinates.mean(dim=1)
            epistemic = epistemic_coordinates.mean(dim=1)
            total_variance = aleatoric_coordinates + epistemic_coordinates
            interval_width = (2.0 * 1.645 * torch.sqrt(total_variance.clamp_min(0))).mean(dim=1)
            ood_probability = torch.sigmoid(ood_logits).amax(dim=0)
            finite = torch.stack(
                (
                    torch.isfinite(means).all(dim=(0, 2)),
                    torch.isfinite(logvars).all(dim=(0, 2)),
                    torch.isfinite(scores).all(dim=0),
                    torch.isfinite(ood_logits).all(dim=0),
                    torch.isfinite(clean).all(dim=(1, 2)),
                )
            ).all(dim=0)
            safe = (
                finite
                & valid_mask.any(dim=1)
                & (ood_probability <= float(self.config.ood_probability_threshold))
                & (aleatoric <= float(self.config.max_aleatoric_variance))
                & (epistemic <= float(self.config.max_epistemic_variance))
                & (interval_width <= float(self.config.max_interval_width))
            )
            band_energy = F.relu(self.contract.focus_band_threshold - mean_score).square()
            energy = (band_energy * safe.to(band_energy.dtype)).sum()
            gradient = torch.autograd.grad(energy, clean, allow_unused=False)[0]
            safe = safe & torch.isfinite(gradient).all(dim=(1, 2))
            gradient = torch.where(safe[:, None, None], gradient, torch.zeros_like(gradient))
            update = -self.config.guidance_scale * self._low_pass(gradient)
            update = self._rms_clip(update, clean, valid_mask)
            mapped = (1.0 - float(next_timestep)) * step_weight * update
            corrected = xt_next + mapped.to(dtype=xt_next.dtype)
            corrected = torch.where(safe[:, None, None], corrected, xt_next)

        diagnostics = TopologyCorrectionDiagnostics(
            applied=safe.detach(),
            focus_logit=mean_score.detach(),
            ood_probability=ood_probability.detach(),
            aleatoric_variance=aleatoric.detach(),
            epistemic_variance=epistemic.detach(),
            interval_width=interval_width.detach(),
        )
        return corrected, diagnostics

    def __call__(self, **kwargs: object) -> Tensor:
        """Return only the corrected latent for the ACE-Step sampler protocol."""

        corrected, _ = self.apply_with_diagnostics(**kwargs)  # type: ignore[arg-type]
        return corrected
