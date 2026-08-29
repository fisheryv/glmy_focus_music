"""Runtime application of the signed frozen 18-D exact-scoring artifact."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from topology.statistics import TOPOLOGY_METRICS

from .ltsn_contract import FingerprintContract, LTSNContractError, load_fingerprint_contract


@dataclass(frozen=True, slots=True)
class ExactPathHomologyScore:
    """Batch-aligned coordinates and deterministic frozen classifier readout."""

    coordinates: NDArray[np.float64]
    focus_logit: NDArray[np.float64]
    focus_probability: NDArray[np.float64]
    focus_band_loss: NDArray[np.float64]
    pitch_block_l2_norm: NDArray[np.float64]
    phase_block_l2_norm: NDArray[np.float64]


class ExactPathHomologyScorer:
    """Transform exact Pitch/phase descriptors with the signed discovery fit."""

    _BLOCKS = ("pitch", "path_acoustic_phase", "path_chroma_phase")

    def __init__(self, contract: FingerprintContract, payload: dict[str, Any]) -> None:
        self.contract = contract
        self.payload = payload
        transforms = payload.get("block_transforms")
        if not isinstance(transforms, dict) or set(transforms) != set(self._BLOCKS):
            raise LTSNContractError("exact scorer must contain only Pitch/Acoustic/Chroma blocks")
        self.transforms = transforms
        pitch_features = transforms["pitch"].get("input_features")
        if pitch_features != list(TOPOLOGY_METRICS):
            raise LTSNContractError("Pitch descriptor order differs from TOPOLOGY_METRICS")
        expected_scales = {
            "pitch": 1.0 / math.sqrt(2.0),
            "path_acoustic_phase": 0.5,
            "path_chroma_phase": 0.5,
        }
        for name, expected in expected_scales.items():
            actual = float(transforms[name].get("fusion_scale", float("nan")))
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise LTSNContractError(f"unexpected frozen fusion scale for {name}")

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> ExactPathHomologyScorer:
        """Load a scorer only after the frozen contract and optional hash pass."""

        contract = load_fingerprint_contract(path, expected_sha256=expected_sha256)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(contract, payload)

    @staticmethod
    def _matrix(values: ArrayLike, columns: int, name: str) -> NDArray[np.float64]:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != columns:
            raise ValueError(f"{name} must have shape [B,{columns}]")
        if np.isinf(matrix).any():
            raise ValueError(f"{name} contains Inf")
        return matrix

    @staticmethod
    def _transform(values: NDArray[np.float64], block: dict[str, Any]) -> NDArray[np.float64]:
        medians = np.asarray(block["imputer_median"], dtype=float)
        keep = np.asarray(block["keep_mask"], dtype=bool)
        mean = np.asarray(block["retained_mean"], dtype=float)
        whitening = np.asarray(block["whitening"], dtype=float)
        rank = int(block["effective_rank"])
        dimensions = int(block["output_dimensions"])
        scale = float(block["fusion_scale"])
        if medians.shape != (values.shape[1],) or keep.shape != medians.shape:
            raise LTSNContractError("malformed imputer or keep mask in exact scorer")
        if rank < 1 or mean.shape != (int(keep.sum()),):
            raise LTSNContractError("malformed retained mean or rank in exact scorer")
        if whitening.shape != (int(keep.sum()), dimensions):
            raise LTSNContractError("malformed whitening matrix in exact scorer")
        imputed = np.where(np.isnan(values), medians, values)
        transformed = ((imputed[:, keep] - mean) @ whitening) / math.sqrt(rank)
        transformed *= scale
        if not np.isfinite(transformed).all():
            raise ValueError("exact scorer produced NaN or Inf")
        return transformed

    def score(
        self,
        pitch_descriptors: ArrayLike,
        acoustic_loop_score: ArrayLike,
        chroma_loop_score: ArrayLike,
    ) -> ExactPathHomologyScore:
        """Score batch-aligned exact descriptors in the frozen 18-D space."""

        pitch = self._matrix(pitch_descriptors, len(TOPOLOGY_METRICS), "pitch_descriptors")
        acoustic = self._matrix(acoustic_loop_score, 1, "acoustic_loop_score")
        chroma = self._matrix(chroma_loop_score, 1, "chroma_loop_score")
        if len({pitch.shape[0], acoustic.shape[0], chroma.shape[0]}) != 1:
            raise ValueError("Pitch and phase batches must contain the same rows")
        pitch_coordinates = self._transform(pitch, self.transforms["pitch"])
        acoustic_coordinate = self._transform(
            acoustic, self.transforms["path_acoustic_phase"]
        )
        chroma_coordinate = self._transform(chroma, self.transforms["path_chroma_phase"])
        coordinates = np.concatenate(
            (pitch_coordinates, acoustic_coordinate, chroma_coordinate), axis=1
        )
        if coordinates.shape[1] != 18:
            raise LTSNContractError("exact scorer did not produce 18 coordinates")
        coefficient = np.asarray(self.contract.classifier_coef, dtype=float)
        logit = coordinates @ coefficient + self.contract.classifier_intercept
        probability = np.exp(-np.logaddexp(0.0, -logit))
        band_loss = np.maximum(0.0, self.contract.focus_band_threshold - logit) ** 2
        return ExactPathHomologyScore(
            coordinates=coordinates,
            focus_logit=logit,
            focus_probability=probability,
            focus_band_loss=band_loss,
            pitch_block_l2_norm=np.linalg.norm(pitch_coordinates, axis=1),
            phase_block_l2_norm=np.linalg.norm(
                np.concatenate((acoustic_coordinate, chroma_coordinate), axis=1), axis=1
            ),
        )
