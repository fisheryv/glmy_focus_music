"""Frozen block-balanced training losses for the 18-D LTSN."""

from __future__ import annotations

from dataclasses import dataclass

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
    ranking: float = 0.2
    trajectory_delta: float = 0.2
    ood: float = 0.1


class LTSNLossResult(dict[str, Tensor]):
    """Dictionary containing the total loss and each auditable component."""


def _block_reduce(values: Tensor) -> Tensor:
    if values.ndim != 2 or values.shape[1] != 18:
        raise ValueError("coordinate loss inputs must have shape [B,18]")
    pitch = values[:, :16].mean(dim=1)
    acoustic = values[:, 16]
    chroma = values[:, 17]
    return (0.5 * pitch + 0.25 * acoustic + 0.25 * chroma).mean()


def block_balanced_huber(prediction: Tensor, target: Tensor, *, delta: float = 1.0) -> Tensor:
    """Apply Huber loss with Pitch/Acoustic/Chroma weights 1/2, 1/4, 1/4."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target coordinate shapes differ")
    elementwise = F.huber_loss(prediction.float(), target.float(), delta=delta, reduction="none")
    return _block_reduce(elementwise)


def block_balanced_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """Compute frozen block-balanced heteroscedastic Gaussian NLL."""

    if mean.shape != target.shape or logvar.shape != target.shape:
        raise ValueError("mean, logvar, and target coordinate shapes must match")
    mean32, logvar32, target32 = mean.float(), logvar.float(), target.float()
    elementwise = 0.5 * (torch.exp(-logvar32) * (target32 - mean32).square() + logvar32)
    return _block_reduce(elementwise)


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
) -> Tensor:
    """Match exact coordinate increments without imposing artificial smoothness."""

    if next_mean is None or next_target is None:
        return current_mean.sum() * 0.0
    predicted_delta = next_mean - current_mean
    exact_delta = next_target - current_target
    return block_balanced_huber(predicted_delta, exact_delta)


def ltsn_loss(
    output: LTSNOutput,
    coordinate_target: Tensor,
    exact_focus_logit: Tensor,
    ood_target: Tensor,
    *,
    pair_indices: Tensor | None = None,
    next_output: LTSNOutput | None = None,
    next_coordinate_target: Tensor | None = None,
    weights: LTSNLossWeights | None = None,
) -> LTSNLossResult:
    """Compute the complete development-start LTSN objective and components."""

    selected = weights or LTSNLossWeights()
    coordinate = block_balanced_huber(output.coordinate_mean, coordinate_target)
    nll = block_balanced_nll(output.coordinate_mean, output.coordinate_logvar, coordinate_target)
    score = F.huber_loss(output.focus_logit.float(), exact_focus_logit.float())
    ranking = same_prompt_ranking_loss(
        output.focus_logit.float(), exact_focus_logit.float(), pair_indices
    )
    delta = trajectory_delta_loss(
        output.coordinate_mean,
        None if next_output is None else next_output.coordinate_mean,
        coordinate_target,
        next_coordinate_target,
    )
    ood = F.binary_cross_entropy_with_logits(output.ood_logit.float(), ood_target.float())
    total = (
        selected.coordinate * coordinate
        + selected.nll * nll
        + selected.score * score
        + selected.ranking * ranking
        + selected.trajectory_delta * delta
        + selected.ood * ood
    )
    return LTSNLossResult(
        total=total,
        coordinate=coordinate,
        nll=nll,
        score=score,
        ranking=ranking,
        trajectory_delta=delta,
        ood=ood,
    )
