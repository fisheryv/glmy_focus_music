"""Frozen block-balanced training losses for the 18-D LTSN."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor
from torch.nn import functional as F

from .path_homology_surrogate import LTSNOutput


@dataclass(frozen=True, slots=True)
class LTSNLossWeights:
    """Development-start weights that must be frozen before qualification."""

    coordinate: float = 1.0
    nll: float = 0.25
    score: float = 0.5
    focus_band: float = 0.0
    ranking: float = 0.2
    trajectory_delta: float = 0.2
    phase_ranking: float = 0.0
    local_direction: float = 0.0
    ood: float = 0.1

    def validate(self) -> None:
        if any(
            not math.isfinite(float(getattr(self, item.name)))
            or float(getattr(self, item.name)) < 0
            for item in fields(self)
        ):
            raise ValueError("LTSN loss weights must be finite and non-negative")


class LTSNLossResult(dict[str, Tensor]):
    """Dictionary containing the total loss and each auditable component."""


def _coordinate_contract(
    values: Tensor,
    coordinate_scale: Tensor | None,
    active_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    scale = torch.ones(18, device=values.device, dtype=torch.float32)
    if coordinate_scale is not None:
        scale = coordinate_scale.to(device=values.device, dtype=torch.float32).reshape(-1)
    mask = torch.ones(18, device=values.device, dtype=torch.bool)
    if active_mask is not None:
        mask = active_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    if scale.shape != (18,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("coordinate_scale must contain 18 finite positive values")
    if mask.shape != (18,) or not mask[:16].any() or not mask[16:].all():
        raise ValueError("active_mask must retain Pitch and both phase coordinates")
    return scale, mask


def _block_reduce(values: Tensor, active_mask: Tensor | None = None) -> Tensor:
    if values.ndim != 2 or values.shape[1] != 18:
        raise ValueError("coordinate loss inputs must have shape [B,18]")
    _, mask = _coordinate_contract(values, None, active_mask)
    pitch_mask = mask[:16].to(values.dtype)
    pitch = (values[:, :16] * pitch_mask).sum(dim=1) / pitch_mask.sum()
    acoustic = values[:, 16]
    chroma = values[:, 17]
    return (0.5 * pitch + 0.25 * acoustic + 0.25 * chroma).mean()


def block_balanced_huber(
    prediction: Tensor,
    target: Tensor,
    *,
    delta: float = 1.0,
    coordinate_scale: Tensor | None = None,
    active_mask: Tensor | None = None,
) -> Tensor:
    """Apply Huber loss with Pitch/Acoustic/Chroma weights 1/2, 1/4, 1/4."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target coordinate shapes differ")
    scale, mask = _coordinate_contract(prediction, coordinate_scale, active_mask)
    residual = (prediction.float() - target.float()) / scale.unsqueeze(0)
    elementwise = F.huber_loss(
        residual, torch.zeros_like(residual), delta=delta, reduction="none"
    )
    return _block_reduce(elementwise, mask)


def block_balanced_nll(
    mean: Tensor,
    logvar: Tensor,
    target: Tensor,
    *,
    coordinate_scale: Tensor | None = None,
    active_mask: Tensor | None = None,
) -> Tensor:
    """Compute frozen block-balanced heteroscedastic Gaussian NLL."""

    if mean.shape != target.shape or logvar.shape != target.shape:
        raise ValueError("mean, logvar, and target coordinate shapes must match")
    mean32, logvar32, target32 = mean.float(), logvar.float(), target.float()
    scale, mask = _coordinate_contract(mean, coordinate_scale, active_mask)
    normalized_error = (target32 - mean32) / scale.unsqueeze(0)
    normalized_logvar = logvar32 - 2.0 * torch.log(scale).unsqueeze(0)
    elementwise = 0.5 * (
        torch.exp(-normalized_logvar) * normalized_error.square() + normalized_logvar
    )
    return _block_reduce(elementwise, mask)


def same_prompt_ranking_loss(
    predicted_score: Tensor,
    exact_score: Tensor,
    pair_indices: Tensor | None,
    *,
    exact_margin: float = 0.1,
    temperature: float = 1.0,
) -> Tensor:
    """Rank explicit same-prompt pairs whose exact-score gap exceeds a margin."""

    if pair_indices is None or pair_indices.numel() == 0:
        return predicted_score.sum() * 0.0
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape [P,2]")
    left, right = pair_indices[:, 0].long(), pair_indices[:, 1].long()
    exact_difference = exact_score[left] - exact_score[right]
    valid = exact_difference.abs() >= exact_margin
    if not valid.any():
        return predicted_score.sum() * 0.0
    direction = exact_difference[valid].sign()
    predicted_difference = predicted_score[left[valid]] - predicted_score[right[valid]]
    return F.softplus(-direction * predicted_difference / temperature).mean()


def trajectory_delta_loss(
    current_mean: Tensor,
    next_mean: Tensor | None,
    current_target: Tensor,
    next_target: Tensor | None,
    *,
    coordinate_scale: Tensor | None = None,
    active_mask: Tensor | None = None,
) -> Tensor:
    """Match exact coordinate increments without imposing artificial smoothness."""

    if next_mean is None or next_target is None:
        return current_mean.sum() * 0.0
    predicted_delta = next_mean - current_mean
    exact_delta = next_target - current_target
    return block_balanced_huber(
        predicted_delta,
        exact_delta,
        coordinate_scale=coordinate_scale,
        active_mask=active_mask,
    )


def paired_direction_loss(
    predicted_score: Tensor,
    exact_score: Tensor,
    pair_indices: Tensor | None,
    *,
    exact_margin: float = 0.02,
    temperature: float = 1.0,
) -> Tensor:
    """Match the exact local-improvement direction for anchor/perturbation pairs."""

    if pair_indices is None or pair_indices.numel() == 0:
        return predicted_score.sum() * 0.0
    anchor, perturbed = pair_indices[:, 0].long(), pair_indices[:, 1].long()
    exact_difference = exact_score[perturbed] - exact_score[anchor]
    valid = exact_difference.abs() >= exact_margin
    if not valid.any():
        return predicted_score.sum() * 0.0
    predicted_difference = predicted_score[perturbed[valid]] - predicted_score[anchor[valid]]
    return F.softplus(
        -exact_difference[valid].sign() * predicted_difference / temperature
    ).mean()


def focus_band_classification_loss(
    predicted_focus_logit: Tensor,
    exact_focus_logit: Tensor,
    in_distribution: Tensor,
    *,
    focus_band_threshold: float | None,
) -> Tensor:
    """Classify the frozen Focus band using the existing Focus-logit readout."""

    if focus_band_threshold is None or not in_distribution.any():
        return predicted_focus_logit.sum() * 0.0
    if not math.isfinite(focus_band_threshold):
        raise ValueError("focus_band_threshold must be finite")
    predicted_margin = predicted_focus_logit.float()[in_distribution] - float(
        focus_band_threshold
    )
    target = (
        exact_focus_logit.float()[in_distribution] >= float(focus_band_threshold)
    ).to(predicted_margin.dtype)
    return F.binary_cross_entropy_with_logits(predicted_margin, target)


def phase_pair_ranking_loss(
    predicted_coordinates: Tensor,
    exact_coordinates: Tensor,
    pair_indices: Tensor | None,
    *,
    coordinate_scale: Tensor | None = None,
    exact_margin: float = 0.05,
) -> Tensor:
    """Preserve same-prompt Acoustic/Chroma ordering without adding a new head."""

    if pair_indices is None or pair_indices.numel() == 0:
        return predicted_coordinates.sum() * 0.0
    scale, _ = _coordinate_contract(predicted_coordinates, coordinate_scale, None)
    left, right = pair_indices[:, 0].long(), pair_indices[:, 1].long()
    losses: list[Tensor] = []
    for coordinate in (16, 17):
        exact_difference = (
            exact_coordinates[left, coordinate] - exact_coordinates[right, coordinate]
        ) / scale[coordinate]
        valid = exact_difference.abs() >= exact_margin
        if valid.any():
            predicted_difference = (
                predicted_coordinates[left[valid], coordinate]
                - predicted_coordinates[right[valid], coordinate]
            ) / scale[coordinate]
            losses.append(
                F.softplus(-exact_difference[valid].sign() * predicted_difference).mean()
            )
    return predicted_coordinates.sum() * 0.0 if not losses else torch.stack(losses).mean()


def ltsn_loss(
    output: LTSNOutput,
    coordinate_target: Tensor,
    exact_focus_logit: Tensor,
    ood_target: Tensor,
    *,
    pair_indices: Tensor | None = None,
    local_pair_indices: Tensor | None = None,
    next_output: LTSNOutput | None = None,
    next_coordinate_target: Tensor | None = None,
    coordinate_scale: Tensor | None = None,
    active_mask: Tensor | None = None,
    ood_positive_weight: Tensor | None = None,
    focus_band_threshold: float | None = None,
    weights: LTSNLossWeights | None = None,
) -> LTSNLossResult:
    """Compute the complete development-start LTSN objective and components."""

    selected = weights or LTSNLossWeights()
    in_distribution = ood_target.float() < 0.5
    if not in_distribution.any():
        zero = output.coordinate_mean.sum() * 0.0
        coordinate = zero
        nll = zero
        score = zero
    else:
        coordinate = block_balanced_huber(
            output.coordinate_mean[in_distribution],
            coordinate_target[in_distribution],
            coordinate_scale=coordinate_scale,
            active_mask=active_mask,
        )
        nll = block_balanced_nll(
            output.coordinate_mean[in_distribution],
            output.coordinate_logvar[in_distribution],
            coordinate_target[in_distribution],
            coordinate_scale=coordinate_scale,
            active_mask=active_mask,
        )
        score = F.huber_loss(
            output.focus_logit.float()[in_distribution],
            exact_focus_logit.float()[in_distribution],
        )
    focus_band = focus_band_classification_loss(
        output.focus_logit,
        exact_focus_logit,
        in_distribution,
        focus_band_threshold=focus_band_threshold,
    )
    ranking = same_prompt_ranking_loss(
        output.focus_logit.float(), exact_focus_logit.float(), pair_indices
    )
    phase_ranking = phase_pair_ranking_loss(
        output.coordinate_mean,
        coordinate_target,
        pair_indices,
        coordinate_scale=coordinate_scale,
    )
    local_direction = paired_direction_loss(
        output.focus_logit.float(), exact_focus_logit.float(), local_pair_indices
    )
    delta = trajectory_delta_loss(
        output.coordinate_mean,
        None if next_output is None else next_output.coordinate_mean,
        coordinate_target,
        next_coordinate_target,
        coordinate_scale=coordinate_scale,
        active_mask=active_mask,
    )
    ood = F.binary_cross_entropy_with_logits(
        output.ood_logit.float(),
        ood_target.float(),
        pos_weight=ood_positive_weight,
    )
    total = (
        selected.coordinate * coordinate
        + selected.nll * nll
        + selected.score * score
        + selected.focus_band * focus_band
        + selected.ranking * ranking
        + selected.trajectory_delta * delta
        + selected.phase_ranking * phase_ranking
        + selected.local_direction * local_direction
        + selected.ood * ood
    )
    return LTSNLossResult(
        total=total,
        coordinate=coordinate,
        nll=nll,
        score=score,
        focus_band=focus_band,
        ranking=ranking,
        trajectory_delta=delta,
        phase_ranking=phase_ranking,
        local_direction=local_direction,
        ood=ood,
    )
