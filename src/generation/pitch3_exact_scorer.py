"""Runtime exact scorer for the frozen three-coordinate Pitch fingerprint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from topology.metrics import TOPOLOGY_METRICS

from .pitch3_contract import Pitch3Contract, load_pitch3_contract


@dataclass(frozen=True, slots=True)
class Pitch3Score:
    coordinates: NDArray[np.float64]
    focus_logit: NDArray[np.float64]
    focus_probability: NDArray[np.float64]
    target_band_loss: NDArray[np.float64]
    target_center_distance: NDArray[np.float64]


class ExactPitch3Scorer:
    """Select, standardize, and score three exact Pitch descriptors."""

    def __init__(self, contract: Pitch3Contract) -> None:
        self.contract = contract
        self._indices = tuple(TOPOLOGY_METRICS.index(name) for name in contract.input_features)

    @classmethod
    def from_json(
        cls, path: Path, *, expected_sha256: str | None = None
    ) -> ExactPitch3Scorer:
        return cls(load_pitch3_contract(path, expected_sha256=expected_sha256))

    def score(self, pitch_descriptors: ArrayLike) -> Pitch3Score:
        values = np.asarray(pitch_descriptors, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(TOPOLOGY_METRICS):
            raise ValueError(
                f"pitch_descriptors must have shape [B,{len(TOPOLOGY_METRICS)}]"
            )
        selected = values[:, self._indices]
        if not np.isfinite(selected).all():
            raise ValueError("selected Pitch-3 descriptors contain NaN or Inf")
        center = np.asarray(self.contract.transform_center, dtype=float)
        scale = np.asarray(self.contract.transform_scale, dtype=float)
        coordinates = (selected - center) / scale
        coef = np.asarray(self.contract.classifier_coef, dtype=float)
        logit = coordinates @ coef + self.contract.classifier_intercept
        probability = np.exp(-np.logaddexp(0.0, -logit))
        lower = np.asarray(self.contract.target_lower, dtype=float)
        upper = np.asarray(self.contract.target_upper, dtype=float)
        below = np.maximum(lower - coordinates, 0.0)
        above = np.maximum(coordinates - upper, 0.0)
        weights = np.asarray(self.contract.distance_weights, dtype=float)
        band_loss = (np.square(below) + np.square(above)) @ weights
        target_center = np.asarray(self.contract.target_center, dtype=float)
        center_distance = np.sqrt(np.square(coordinates - target_center) @ weights)
        return Pitch3Score(
            coordinates=coordinates,
            focus_logit=logit,
            focus_probability=probability,
            target_band_loss=band_loss,
            target_center_distance=center_distance,
        )
