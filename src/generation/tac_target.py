"""Frozen Stage-0 target geometry for Topology Action Critic experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TACTargetError(ValueError):
    """Raised when a TAC target artifact violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_center_scale(values: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return coordinate-wise median and deterministic robust non-zero scale."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise TACTargetError("reference values must have shape [N,D] with N >= 2")
    if not np.isfinite(matrix).all():
        raise TACTargetError("reference values must be finite")
    center = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.std(matrix, axis=0, ddof=1)
    tolerance = np.finfo(np.float64).eps
    scale = np.where(scale > tolerance, scale, iqr_scale)
    scale = np.where(scale > tolerance, scale, std_scale)
    if np.any(~np.isfinite(scale)) or np.any(scale <= tolerance):
        raise TACTargetError("reference block contains an unscalable coordinate")
    return center, scale


def fit_target_payload(
    coordinates: ArrayLike,
    *,
    active_pitch_indices: tuple[int, ...],
    acoustic_index: int,
    chroma_index: int,
    distance_weights: tuple[float, float, float],
    pitch_shrinkage: float,
    reference_quantiles: tuple[float, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Fit the fixed robust center and block geometry from Focus references."""

    matrix = np.asarray(coordinates, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 18 or matrix.shape[0] < 2:
        raise TACTargetError("coordinates must have shape [N,18] with N >= 2")
    if not np.isfinite(matrix).all():
        raise TACTargetError("coordinates must be finite")
    if not 0.0 <= pitch_shrinkage <= 1.0:
        raise TACTargetError("pitch_shrinkage must be in [0,1]")
    if len(active_pitch_indices) < 1 or len(set(active_pitch_indices)) != len(active_pitch_indices):
        raise TACTargetError("active Pitch indices must be non-empty and unique")
    if any(index < 0 or index >= 16 for index in active_pitch_indices):
        raise TACTargetError("active Pitch indices must lie in [0,15]")
    inactive = sorted(set(range(16)) - set(active_pitch_indices))
    if inactive and np.max(np.abs(matrix[:, inactive]), initial=0.0) > 1e-12:
        raise TACTargetError("inactive Pitch coordinates are not exact zero")
    if (acoustic_index, chroma_index) != (16, 17):
        raise TACTargetError("phase coordinates must retain frozen indices 16 and 17")
    if len(distance_weights) != 3 or not math.isclose(
        sum(distance_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise TACTargetError("three distance weights must sum to one")

    pitch = matrix[:, active_pitch_indices]
    pitch_center, pitch_scale = robust_center_scale(pitch)
    standardized = (pitch - pitch_center) / pitch_scale
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=1))
    shrunk = (1.0 - pitch_shrinkage) * covariance + pitch_shrinkage * np.eye(
        len(active_pitch_indices), dtype=np.float64
    )
    precision = np.linalg.inv(shrunk)
    minimum_eigenvalue = float(np.linalg.eigvalsh(shrunk)[0])
    if minimum_eigenvalue <= 0.0 or not np.isfinite(precision).all():
        raise TACTargetError("shrunk Pitch covariance is not positive definite")

    acoustic_center, acoustic_scale = robust_center_scale(matrix[:, [acoustic_index]])
    chroma_center, chroma_scale = robust_center_scale(matrix[:, [chroma_index]])
    payload: dict[str, Any] = {
        "schema_version": 1,
        **metadata,
        "dimensions": 18,
        "reference_count": int(matrix.shape[0]),
        "distance_definition": {
            "orientation": "lower_is_more_focus_like",
            "action_reward": "distance_before_minus_distance_after",
            "formula": "0.5*pitch_mahalanobis_sq/13 + 0.25*acoustic_z_sq + 0.25*chroma_z_sq",
            "weights": [float(value) for value in distance_weights],
        },
        "blocks": {
            "pitch": {
                "indices": list(active_pitch_indices),
                "inactive_indices": inactive,
                "center": pitch_center.tolist(),
                "scale": pitch_scale.tolist(),
                "covariance": shrunk.tolist(),
                "precision": precision.tolist(),
                "shrinkage": float(pitch_shrinkage),
                "normalization_dimensions": len(active_pitch_indices),
                "minimum_covariance_eigenvalue": minimum_eigenvalue,
            },
            "path_acoustic_phase": {
                "indices": [acoustic_index],
                "center": acoustic_center.tolist(),
                "scale": acoustic_scale.tolist(),
            },
            "path_chroma_phase": {
                "indices": [chroma_index],
                "center": chroma_center.tolist(),
                "scale": chroma_scale.tolist(),
            },
        },
    }
    target = TACTopologyTarget(payload, verify_sources=False)
    distances = target.distance(matrix)
    payload["reference_distance"] = {
        "minimum": float(np.min(distances)),
        "maximum": float(np.max(distances)),
        "mean": float(np.mean(distances)),
        "quantiles": {
            format(quantile, ".6g"): float(np.quantile(distances, quantile))
            for quantile in reference_quantiles
        },
    }
    return payload


@dataclass(frozen=True, slots=True)
class TACTopologyTarget:
    """Apply the frozen full-18D target distance to exact coordinates."""

    payload: dict[str, Any]

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        root: Path | None = None,
        verify_sources: bool = True,
    ) -> None:
        object.__setattr__(self, "payload", payload)
        self._validate(root=root, verify_sources=verify_sources)

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        root: Path | None = None,
        expected_sha256: str | None = None,
        verify_sources: bool = True,
    ) -> TACTopologyTarget:
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise TACTargetError("TAC target SHA256 mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload, root=root, verify_sources=verify_sources)

    def _validate(self, *, root: Path | None, verify_sources: bool) -> None:
        payload = self.payload
        if payload.get("schema_version") != 1:
            raise TACTargetError("unsupported TAC target schema")
        if payload.get("target_id") != "tac_topology_target_v1":
            raise TACTargetError("unexpected TAC target id")
        if payload.get("dimensions") != 18:
            raise TACTargetError("TAC target must contain 18 coordinates")
        evidence = payload.get("evidence", {})
        forbidden_true = (
            "scientific_evidence",
            "qualification_eligible",
            "guidance_promotion_eligible",
            "production_authorization",
        )
        if any(evidence.get(name) is not False for name in forbidden_true):
            raise TACTargetError("Stage-0 target must remain diagnostic and ineligible")
        definition = payload.get("distance_definition", {})
        weights = definition.get("weights")
        if weights != [0.5, 0.25, 0.25]:
            raise TACTargetError("TAC target distance weights changed")
        blocks = payload.get("blocks", {})
        if set(blocks) != {"pitch", "path_acoustic_phase", "path_chroma_phase"}:
            raise TACTargetError("TAC target blocks changed")
        pitch = blocks["pitch"]
        indices = tuple(int(value) for value in pitch.get("indices", []))
        dimensions = int(pitch.get("normalization_dimensions", 0))
        center = np.asarray(pitch.get("center"), dtype=np.float64)
        scale = np.asarray(pitch.get("scale"), dtype=np.float64)
        precision = np.asarray(pitch.get("precision"), dtype=np.float64)
        if dimensions != len(indices) or center.shape != (dimensions,):
            raise TACTargetError("malformed TAC Pitch center")
        if scale.shape != center.shape or np.any(scale <= 0.0):
            raise TACTargetError("malformed TAC Pitch scale")
        if precision.shape != (dimensions, dimensions) or not np.isfinite(precision).all():
            raise TACTargetError("malformed TAC Pitch precision")
        for name, expected_index in (
            ("path_acoustic_phase", 16),
            ("path_chroma_phase", 17),
        ):
            block = blocks[name]
            if block.get("indices") != [expected_index]:
                raise TACTargetError(f"malformed {name} index")
            if len(block.get("center", [])) != 1 or float(block["scale"][0]) <= 0.0:
                raise TACTargetError(f"malformed {name} center or scale")
        if verify_sources:
            if root is None:
                raise TACTargetError("root is required when source verification is enabled")
            for relative, expected in payload.get("source_sha256", {}).items():
                source = root / relative
                if not source.is_file() or sha256_file(source) != expected:
                    raise TACTargetError(f"source SHA256 mismatch: {relative}")

    @staticmethod
    def _coordinates(values: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != 18:
            raise TACTargetError("coordinates must have shape [B,18]")
        if not np.isfinite(matrix).all():
            raise TACTargetError("coordinates must be finite")
        return matrix

    def block_distances(self, values: ArrayLike) -> dict[str, NDArray[np.float64]]:
        matrix = self._coordinates(values)
        blocks = self.payload["blocks"]
        pitch = blocks["pitch"]
        indices = np.asarray(pitch["indices"], dtype=int)
        z_pitch = (matrix[:, indices] - np.asarray(pitch["center"])) / np.asarray(pitch["scale"])
        precision = np.asarray(pitch["precision"], dtype=np.float64)
        pitch_distance = np.einsum("bi,ij,bj->b", z_pitch, precision, z_pitch)
        pitch_distance /= int(pitch["normalization_dimensions"])
        result = {"pitch": pitch_distance}
        for name in ("path_acoustic_phase", "path_chroma_phase"):
            block = blocks[name]
            index = int(block["indices"][0])
            z = (matrix[:, index] - float(block["center"][0])) / float(block["scale"][0])
            result[name] = z * z
        return result

    def distance(self, values: ArrayLike) -> NDArray[np.float64]:
        blocks = self.block_distances(values)
        weights = self.payload["distance_definition"]["weights"]
        return (
            float(weights[0]) * blocks["pitch"]
            + float(weights[1]) * blocks["path_acoustic_phase"]
            + float(weights[2]) * blocks["path_chroma_phase"]
        )

    def reward(self, before: ArrayLike, after: ArrayLike) -> NDArray[np.float64]:
        before_distance = self.distance(before)
        after_distance = self.distance(after)
        if before_distance.shape != after_distance.shape:
            raise TACTargetError("before and after batches must have the same size")
        return before_distance - after_distance
