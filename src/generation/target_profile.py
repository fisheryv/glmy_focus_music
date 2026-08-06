from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CORE_FEATURES = (
    "acoustic_novelty_delay__h0_max_persistence",
    "rhythm__h0_total_persistence",
    "path_acoustic_phase__loop_score",
    "path_rhythm_phase__loop_score",
)


@dataclass(frozen=True, slots=True)
class TargetProfile:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    group: str
    split: str
    scale_seconds: float
    sample_count: int
    covariance_shrinkage: float
    source_sha256: dict[str, str]

    def standardized(self, descriptor: list[float] | tuple[float, ...]) -> np.ndarray:
        values = np.asarray(descriptor, dtype=float)
        return (values - np.asarray(self.center)) / np.asarray(self.scale)

    def distance(self, descriptor: list[float] | tuple[float, ...]) -> float:
        delta = self.standardized(descriptor)
        precision = np.asarray(self.precision, dtype=float)
        return float(np.sqrt(max(float(delta.T @ precision @ delta), 0.0)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fit_target_profile(
    root: Path,
    *,
    group: str = "focus",
    split: str = "discovery",
    scale_seconds: float = 180.0,
    covariance_shrinkage: float = 0.2,
) -> TargetProfile:
    tda_path = root / "metadata" / "tda_features.csv"
    repetition_path = root / "metadata" / "repetition_homology_features.csv"
    values: dict[str, dict[str, float]] = {}
    tda_mapping = {
        "acoustic_novelty_delay": ("h0_max_persistence", CORE_FEATURES[0]),
        "rhythm": ("h0_total_persistence", CORE_FEATURES[1]),
    }
    for row in _read_rows(tda_path):
        if (
            row["group"] == group
            and row["split"] == split
            and np.isclose(float(row["scale_seconds"]), scale_seconds)
            and row["representation"] in tda_mapping
        ):
            metric, name = tda_mapping[row["representation"]]
            values.setdefault(row["track_id"], {})[name] = float(row[metric])
    repetition_mapping = {
        "path_acoustic_phase": CORE_FEATURES[2],
        "path_rhythm_phase": CORE_FEATURES[3],
    }
    for row in _read_rows(repetition_path):
        if (
            row["group"] == group
            and row["split"] == split
            and np.isclose(float(row["scale_seconds"]), scale_seconds)
            and row["representation"] in repetition_mapping
        ):
            name = repetition_mapping[row["representation"]]
            values.setdefault(row["track_id"], {})[name] = float(row["loop_score"])
    matrix = np.asarray(
        [
            [track[name] for name in CORE_FEATURES]
            for track in values.values()
            if all(name in track for name in CORE_FEATURES)
        ],
        dtype=float,
    )
    if matrix.shape[0] < 20:
        raise ValueError("too few complete discovery Focus rows for a target profile")
    center = np.median(matrix, axis=0)
    iqr_scale = (np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)) / 1.349
    mad_scale = np.median(np.abs(matrix - center), axis=0) * 1.4826
    std_scale = np.std(matrix, axis=0, ddof=1)
    scale = np.where(iqr_scale > 1e-8, iqr_scale, np.where(mad_scale > 1e-8, mad_scale, std_scale))
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (matrix - center) / scale
    covariance = np.cov(standardized, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1.0 - covariance_shrinkage) * covariance + covariance_shrinkage * diagonal
    precision = np.linalg.pinv(shrunk, hermitian=True)
    return TargetProfile(
        feature_names=CORE_FEATURES,
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        precision=tuple(tuple(float(value) for value in row) for row in precision),
        group=group,
        split=split,
        scale_seconds=scale_seconds,
        sample_count=int(matrix.shape[0]),
        covariance_shrinkage=covariance_shrinkage,
        source_sha256={
            tda_path.relative_to(root).as_posix(): _sha256(tda_path),
            repetition_path.relative_to(root).as_posix(): _sha256(repetition_path),
        },
    )


def write_target_profile(path: Path, profile: TargetProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(profile)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_target_profile(path: Path) -> TargetProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feature_names"] = tuple(payload["feature_names"])
    payload["center"] = tuple(payload["center"])
    payload["scale"] = tuple(payload["scale"])
    payload["precision"] = tuple(tuple(row) for row in payload["precision"])
    return TargetProfile(**payload)
