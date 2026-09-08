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
    authorization_scope: str = "qualified"
    guidance_scale: float = 1.0
    rms_clip_ratio: float = 0.005
    step_weights: Mapping[int, float] = field(default_factory=lambda: {4: 0.5, 5: 1.0, 6: 0.5})
    ood_probability_threshold: float | None = None
    max_aleatoric_variance: float | None = None
    max_epistemic_variance: float | None = None
    max_interval_width: float | None = None
    variance_scale: tuple[float, ...] = (1.0,) * 18
    low_pass_kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0)
    require_all_members_out_of_band: bool = False
    require_all_member_improvement: bool = False
    minimum_member_gradient_cosine: float = -1.0
    epsilon: float = 1e-8

    def validate(self) -> None:
        """Reject activation without qualification and calibrated thresholds."""

        if self.authorization_scope not in {"qualified", "development_only"}:
            raise LTSNContractError("unknown topology-corrector authorization scope")
        if self.rms_clip_ratio not in _ALLOWED_RMS_RATIOS:
            raise LTSNContractError("rms_clip_ratio must be 0.25%, 0.5%, or 1.0%")
        if self.guidance_scale < 0 or not math.isfinite(self.guidance_scale):
            raise LTSNContractError("guidance_scale must be finite and non-negative")
        if not -1.0 <= self.minimum_member_gradient_cosine <= 1.0:
            raise LTSNContractError("minimum_member_gradient_cosine must be in [-1,1]")
        if len(self.variance_scale) != 18 or any(
            value <= 0 or not math.isfinite(value) for value in self.variance_scale
        ):
            raise LTSNContractError("variance_scale must contain 18 finite positive values")
        if self.authorization_scope == "development_only" and self.qualification_passed:
            raise LTSNContractError(
                "development-only correction must not claim passed qualification"
            )
        if not self.enabled:
            return
        if self.authorization_scope == "qualified" and not self.qualification_passed:
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
    member_focus_logit: Tensor
    member_gradient_cosine: Tensor
    member_predicted_improvement: Tensor
    predicted_improvement: Tensor
    proposed_rms: Tensor
    applied_rms: Tensor
    no_op_reason_code: Tensor


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
        self._telemetry: list[dict[str, object]] = []
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

    @staticmethod
    def _masked_rms(values: Tensor, mask: Tensor) -> Tensor:
        expanded = mask.unsqueeze(-1).expand_as(values).to(values.dtype)
        count = expanded.sum(dim=(1, 2)).clamp_min(1.0)
        return torch.sqrt((values.square() * expanded).sum(dim=(1, 2)) / count)

    @staticmethod
    def _minimum_member_cosine(gradients: Tensor, mask: Tensor, epsilon: float) -> Tensor:
        if gradients.shape[0] < 2:
            return torch.ones(gradients.shape[1], device=gradients.device)
        expanded = mask[None, :, :, None].expand_as(gradients).to(gradients.dtype)
        masked = gradients * expanded
        values = []
        for left in range(gradients.shape[0]):
            for right in range(left + 1, gradients.shape[0]):
                dot = (masked[left] * masked[right]).sum(dim=(1, 2))
                left_norm = torch.sqrt(masked[left].square().sum(dim=(1, 2)))
                right_norm = torch.sqrt(masked[right].square().sum(dim=(1, 2)))
                denominator = left_norm * right_norm
                cosine = torch.where(
                    denominator > epsilon,
                    dot / denominator.clamp_min(epsilon),
                    torch.full_like(dot, -1.0),
                )
                values.append(cosine)
        return torch.stack(values).amin(dim=0)

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
            member_band_energy = F.relu(self.contract.focus_band_threshold - scores).square()
            member_gradients = torch.stack(
                [
                    torch.autograd.grad(
                        member_band_energy[index].sum(),
                        clean,
                        retain_graph=True,
                        allow_unused=False,
                    )[0]
                    for index in range(member_band_energy.shape[0])
                ]
            )
            member_gradient_cosine = self._minimum_member_cosine(
                member_gradients, valid_mask, self.config.epsilon
            )
            member_gradient_finite = torch.isfinite(member_gradients).all(dim=(0, 2, 3))
            band_energy = F.relu(self.contract.focus_band_threshold - mean_score).square()
            energy = (band_energy * safe.to(band_energy.dtype)).sum()
            gradient = torch.autograd.grad(energy, clean, allow_unused=False)[0]
            gradient_finite = torch.isfinite(gradient).all(dim=(1, 2))
            safe = safe & gradient_finite
            gradient = torch.where(safe[:, None, None], gradient, torch.zeros_like(gradient))
            proposed = -self.config.guidance_scale * self._low_pass(gradient)
            proposed_rms = self._masked_rms(proposed, valid_mask)
            update = self._rms_clip(proposed, clean, valid_mask)
            mapped = (1.0 - float(next_timestep)) * step_weight * update
            applied_rms = self._masked_rms(mapped, valid_mask)
            valid_count = valid_mask.sum(dim=1).clamp_min(1).to(mapped.dtype) * mapped.shape[2]
            member_predicted_improvement = -(member_gradients * mapped.unsqueeze(0)).sum(
                dim=(2, 3)
            ) / valid_count.unsqueeze(0)
            predicted_improvement = -(gradient * mapped).sum(dim=(1, 2)) / valid_count
            all_members_out_of_band = (scores < float(self.contract.focus_band_threshold)).all(
                dim=0
            )
            all_members_improve = (member_predicted_improvement > 0).all(dim=0)
            applied = (
                safe & (band_energy > self.config.epsilon) & (applied_rms > self.config.epsilon)
            )
            if self.config.require_all_members_out_of_band:
                applied = applied & all_members_out_of_band
            if self.config.require_all_member_improvement:
                applied = applied & all_members_improve
            applied = (
                applied
                & (member_gradient_cosine >= self.config.minimum_member_gradient_cosine)
                & member_gradient_finite
            )
            corrected = xt_next + mapped.to(dtype=xt_next.dtype)
            corrected = torch.where(applied[:, None, None], corrected, xt_next)

            reason = torch.zeros(mean_score.shape, device=mean_score.device, dtype=torch.int64)
            reason |= (~finite).to(torch.int64) * 1
            reason |= (ood_probability > float(self.config.ood_probability_threshold)).to(
                torch.int64
            ) * 2
            reason |= (aleatoric > float(self.config.max_aleatoric_variance)).to(torch.int64) * 4
            reason |= (epistemic > float(self.config.max_epistemic_variance)).to(torch.int64) * 8
            reason |= (interval_width > float(self.config.max_interval_width)).to(torch.int64) * 16
            reason |= (band_energy <= self.config.epsilon).to(torch.int64) * 32
            reason |= (member_gradient_cosine < self.config.minimum_member_gradient_cosine).to(
                torch.int64
            ) * 64
            if self.config.require_all_members_out_of_band:
                reason |= (~all_members_out_of_band).to(torch.int64) * 128
            if self.config.require_all_member_improvement:
                reason |= (~all_members_improve).to(torch.int64) * 256
            reason |= (~gradient_finite).to(torch.int64) * 512
            reason |= (~member_gradient_finite).to(torch.int64) * 1024
            reason = torch.where(applied, torch.zeros_like(reason), reason)

        diagnostics = TopologyCorrectionDiagnostics(
            applied=applied.detach(),
            focus_logit=mean_score.detach(),
            ood_probability=ood_probability.detach(),
            aleatoric_variance=aleatoric.detach(),
            epistemic_variance=epistemic.detach(),
            interval_width=interval_width.detach(),
            member_focus_logit=scores.detach(),
            member_gradient_cosine=member_gradient_cosine.detach(),
            member_predicted_improvement=member_predicted_improvement.detach(),
            predicted_improvement=predicted_improvement.detach(),
            proposed_rms=proposed_rms.detach(),
            applied_rms=applied_rms.detach(),
            no_op_reason_code=reason.detach(),
        )
        return corrected, diagnostics

    def __call__(self, **kwargs: object) -> Tensor:
        """Return only the corrected latent for the ACE-Step sampler protocol."""

        corrected, diagnostics = self.apply_with_diagnostics(**kwargs)  # type: ignore[arg-type]
        if diagnostics is not None:
            self._telemetry.append(
                {
                    "step_number": int(kwargs["step_index"]) + 1,
                    "applied": diagnostics.applied.cpu().tolist(),
                    "focus_logit": diagnostics.focus_logit.cpu().tolist(),
                    "member_focus_logit": diagnostics.member_focus_logit.cpu().tolist(),
                    "ood_probability": diagnostics.ood_probability.cpu().tolist(),
                    "aleatoric_variance": diagnostics.aleatoric_variance.cpu().tolist(),
                    "epistemic_variance": diagnostics.epistemic_variance.cpu().tolist(),
                    "interval_width": diagnostics.interval_width.cpu().tolist(),
                    "member_gradient_cosine": (diagnostics.member_gradient_cosine.cpu().tolist()),
                    "member_predicted_improvement": (
                        diagnostics.member_predicted_improvement.cpu().tolist()
                    ),
                    "predicted_improvement": (diagnostics.predicted_improvement.cpu().tolist()),
                    "proposed_rms": diagnostics.proposed_rms.cpu().tolist(),
                    "applied_rms": diagnostics.applied_rms.cpu().tolist(),
                    "no_op_reason_code": diagnostics.no_op_reason_code.cpu().tolist(),
                }
            )
        return corrected

    def drain_telemetry(self) -> list[dict[str, object]]:
        """Return and clear auditable diagnostics accumulated by sampler calls."""

        values, self._telemetry = self._telemetry, []
        return values
