from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    topology: NDArray[np.float64]
    quality_penalty: float = 0.0
    payload: Any = None


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: Candidate
    topology_distance: float
    total_score: float


def topology_distance(
    descriptor: ArrayLike,
    target: ArrayLike,
    *,
    precision: ArrayLike | None = None,
) -> float:
    delta = np.asarray(descriptor, dtype=float).reshape(-1) - np.asarray(
        target, dtype=float
    ).reshape(-1)
    if precision is None:
        return float(np.linalg.norm(delta))
    precision_matrix = np.asarray(precision, dtype=float)
    if precision_matrix.shape != (delta.size, delta.size):
        raise ValueError("precision matrix shape does not match descriptor")
    squared = float(delta.T @ precision_matrix @ delta)
    return float(np.sqrt(max(squared, 0.0)))


def rerank_candidates(
    candidates: list[Candidate],
    target: ArrayLike,
    *,
    precision: ArrayLike | None = None,
    quality_weight: float = 1.0,
) -> list[RankedCandidate]:
    """Rank a fixed generation budget by target distance plus quality penalty."""

    if quality_weight < 0:
        raise ValueError("quality_weight cannot be negative")
    ranked = []
    for candidate in candidates:
        distance = topology_distance(candidate.topology, target, precision=precision)
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                topology_distance=distance,
                total_score=distance + quality_weight * candidate.quality_penalty,
            )
        )
    return sorted(ranked, key=lambda item: (item.total_score, item.candidate.candidate_id))
