from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CORE_FEATURES = (
    "acoustic_novelty_delay__h0_max_persistence",
    "rhythm__h0_total_persistence",
    "path_acoustic_phase__loop_score",
    "path_rhythm_phase__loop_score",
)

CORE_DIRECTIONS = (
    "lower_than_controls_but_target_distribution",
    "lower_than_controls_but_target_distribution",
    "higher_than_controls_but_target_distribution",
    "higher_than_controls_but_target_distribution",
)

SUPPORT_FEATURES = (
    "pitch__h0_observed_persistence",
    "pitch__path_entropy",
    "rhythm__edge_density",
    "rhythm__reciprocity",
)

CHALLENGER_FEATURES = ("path_chroma_phase__loop_score",)


def _robust_location_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    iqr_scale = (q75 - q25) / 1.349
    mad_scale = np.median(np.abs(values - center), axis=0) * 1.4826
    std_scale = np.std(values, axis=0, ddof=1)
    scale = np.where(
        iqr_scale > 1e-8,
        iqr_scale,
        np.where(mad_scale > 1e-8, mad_scale, std_scale),
    )
    return center, np.where(scale > 1e-8, scale, 1.0)


@dataclass(frozen=True, slots=True)
class TopologyFingerprint:
    schema_version: int
    fingerprint_id: str
    reference_group: str
    reference_split: str
    reference_scale_seconds: float
    reference_sample_count: int
    reference_segment_ids: tuple[str, ...]
    core_feature_names: tuple[str, ...]
    core_directions: tuple[str, ...]
    core_center: tuple[float, ...]
    core_scale: tuple[float, ...]
    core_precision: tuple[tuple[float, ...], ...]
    core_radius_quantile: float
    core_radius: float
    core_quantiles: tuple[tuple[float, ...], ...]
    support_feature_names: tuple[str, ...]
    support_center: tuple[float, ...]
    support_scale: tuple[float, ...]
    support_lower_quantile: float
    support_upper_quantile: float
    support_lower: tuple[float, ...]
    support_upper: tuple[float, ...]
    challenger_feature_names: tuple[str, ...]
    challenger_quantiles: tuple[tuple[float, ...], ...]
    covariance_shrinkage: float
    source_sha256: dict[str, str]
    frozen_hashes: dict[str, str]
    evidence_role: str

    def standardized_core(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        return (matrix - np.asarray(self.core_center)) / np.asarray(self.core_scale)

    def squared_distance(self, values: np.ndarray) -> np.ndarray:
        standardized = self.standardized_core(values)
        precision = np.asarray(self.core_precision, dtype=float)
        return np.einsum("...i,ij,...j->...", standardized, precision, standardized)

    def distance(self, values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.squared_distance(values), 0.0))

    def core_shell_loss(self, values: np.ndarray) -> np.ndarray:
        excess = np.maximum(self.squared_distance(values) - self.core_radius**2, 0.0)
        return excess**2

    def support_band_loss(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        lower = np.asarray(self.support_lower)
        upper = np.asarray(self.support_upper)
        scale = np.asarray(self.support_scale)
        below = np.maximum(lower - matrix, 0.0) / scale
        above = np.maximum(matrix - upper, 0.0) / scale
        return np.sum(below**2 + above**2, axis=-1)


def fit_topology_fingerprint(
    core_matrix: np.ndarray,
    support_matrix: np.ndarray,
    challenger_matrix: np.ndarray,
    *,
    fingerprint_id: str,
    reference_segment_ids: tuple[str, ...],
    covariance_shrinkage: float = 0.2,
    core_radius_quantile: float = 0.9,
    support_lower_quantile: float = 0.1,
    support_upper_quantile: float = 0.9,
    source_sha256: dict[str, str] | None = None,
    frozen_hashes: dict[str, str] | None = None,
    evidence_role: str = "exploratory_current_open_focus_target",
) -> TopologyFingerprint:
    core = np.asarray(core_matrix, dtype=float)
    support = np.asarray(support_matrix, dtype=float)
    challenger = np.asarray(challenger_matrix, dtype=float)
    rows = core.shape[0]
    expected_shapes = (
        (core, len(CORE_FEATURES), "core"),
        (support, len(SUPPORT_FEATURES), "support"),
        (challenger, len(CHALLENGER_FEATURES), "challenger"),
    )
    for matrix, columns, name in expected_shapes:
        if matrix.ndim != 2 or matrix.shape != (rows, columns):
            raise ValueError(f"{name} matrix must have shape ({rows}, {columns})")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} matrix contains non-finite values")
    if rows < 20 or len(reference_segment_ids) != rows:
        raise ValueError("reference sample is too small or identities do not align")
    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be in [0, 1]")
    if not 0.5 < core_radius_quantile < 1.0:
        raise ValueError("core_radius_quantile must be in (0.5, 1)")
    if not 0.0 < support_lower_quantile < support_upper_quantile < 1.0:
        raise ValueError("support quantiles are invalid")

    core_center, core_scale = _robust_location_scale(core)
    standardized = (core - core_center) / core_scale
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1.0 - covariance_shrinkage) * covariance + covariance_shrinkage * diagonal
    precision = np.linalg.pinv(shrunk, hermitian=True)
    squared = np.einsum("...i,ij,...j->...", standardized, precision, standardized)
    radius = float(np.quantile(np.sqrt(np.maximum(squared, 0.0)), core_radius_quantile))

    support_center, support_scale = _robust_location_scale(support)
    quantile_levels = (0.1, 0.25, 0.5, 0.75, 0.9)
    core_quantiles = np.quantile(core, quantile_levels, axis=0).T
    challenger_quantiles = np.quantile(challenger, quantile_levels, axis=0).T
    support_lower = np.quantile(support, support_lower_quantile, axis=0)
    support_upper = np.quantile(support, support_upper_quantile, axis=0)

    return TopologyFingerprint(
        schema_version=1,
        fingerprint_id=fingerprint_id,
        reference_group="focus",
        reference_split="discovery",
        reference_scale_seconds=180.0,
        reference_sample_count=rows,
        reference_segment_ids=reference_segment_ids,
        core_feature_names=CORE_FEATURES,
        core_directions=CORE_DIRECTIONS,
        core_center=tuple(float(value) for value in core_center),
        core_scale=tuple(float(value) for value in core_scale),
        core_precision=tuple(tuple(float(value) for value in row) for row in precision),
        core_radius_quantile=core_radius_quantile,
        core_radius=radius,
        core_quantiles=tuple(
            tuple(float(value) for value in row) for row in core_quantiles
        ),
        support_feature_names=SUPPORT_FEATURES,
        support_center=tuple(float(value) for value in support_center),
        support_scale=tuple(float(value) for value in support_scale),
        support_lower_quantile=support_lower_quantile,
        support_upper_quantile=support_upper_quantile,
        support_lower=tuple(float(value) for value in support_lower),
        support_upper=tuple(float(value) for value in support_upper),
        challenger_feature_names=CHALLENGER_FEATURES,
        challenger_quantiles=tuple(
            tuple(float(value) for value in row) for row in challenger_quantiles
        ),
        covariance_shrinkage=covariance_shrinkage,
        source_sha256=source_sha256 or {},
        frozen_hashes=frozen_hashes or {},
        evidence_role=evidence_role,
    )


def write_topology_fingerprint(path: Path, fingerprint: TopologyFingerprint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(asdict(fingerprint), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_topology_fingerprint(path: Path) -> TopologyFingerprint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tuple_fields = (
        "reference_segment_ids",
        "core_feature_names",
        "core_directions",
        "core_center",
        "core_scale",
        "support_feature_names",
        "support_center",
        "support_scale",
        "support_lower",
        "support_upper",
        "challenger_feature_names",
    )
    for name in tuple_fields:
        payload[name] = tuple(payload[name])
    payload["core_precision"] = tuple(tuple(row) for row in payload["core_precision"])
    payload["core_quantiles"] = tuple(tuple(row) for row in payload["core_quantiles"])
    payload["challenger_quantiles"] = tuple(
        tuple(row) for row in payload["challenger_quantiles"]
    )
    return TopologyFingerprint(**payload)
