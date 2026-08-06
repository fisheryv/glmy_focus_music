from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import tomllib
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import FeatureBundle, FeatureExtractionConfig
from .extractor import FeatureExtractionError, LibrosaFeatureExtractor
from .structure import structural_features

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

VIEW_NAMES = ("acoustic", "chroma", "rhythm", "modulation", "structure")
STATE_MODEL_VERSION = 2
MANIFEST_COLUMNS = (
    "segment_id",
    "track_id",
    "group",
    "split",
    "scale_seconds",
    "input_relative_path",
    "input_sha256",
    "acoustic_relative_path",
    "acoustic_sha256",
    "chroma_relative_path",
    "chroma_sha256",
    "rhythm_relative_path",
    "rhythm_sha256",
    "modulation_relative_path",
    "modulation_sha256",
    "structure_relative_path",
    "structure_sha256",
    "sidecar_relative_path",
    "config_sha256",
    "model_sha256",
    "acoustic_windows",
    "pitch_steps",
    "rhythm_windows",
    "modulation_windows",
    "structure_blocks",
    "structure_boundaries",
    "invalid_acoustic_windows",
    "uncertain_pitch_steps",
    "invalid_rhythm_values",
    "invalid_modulation_windows",
    "status",
    "processed_at",
    "error",
)


class FeatureBatchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SegmentJob:
    segment_id: str
    track_id: str
    group: str
    split: str
    scale_seconds: float
    input_relative_path: str
    input_sha256: str


@dataclass(frozen=True, slots=True)
class FeatureStateModel:
    rhythm_impute: np.ndarray
    rhythm_mean: np.ndarray
    rhythm_scale: np.ndarray
    rhythm_centers: np.ndarray
    modulation_edges: np.ndarray
    acoustic_mean: np.ndarray
    acoustic_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    acoustic_centers: np.ndarray
    structure_centers: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "rhythm_impute": self.rhythm_impute,
            "rhythm_mean": self.rhythm_mean,
            "rhythm_scale": self.rhythm_scale,
            "rhythm_centers": self.rhythm_centers,
            "modulation_edges": self.modulation_edges,
            "acoustic_mean": self.acoustic_mean,
            "acoustic_scale": self.acoustic_scale,
            "pca_mean": self.pca_mean,
            "pca_components": self.pca_components,
            "acoustic_centers": self.acoustic_centers,
            "structure_centers": self.structure_centers,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _config_hash(config: FeatureExtractionConfig) -> str:
    return _json_hash(
        {
            "extractor": LibrosaFeatureExtractor.name,
            "extractor_version": LibrosaFeatureExtractor.version,
            "config": asdict(config),
        }
    )


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 6) -> None:
    """Atomically replace a file, tolerating brief Windows reader locks."""

    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (2**attempt))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a deterministic compressed NPZ with fixed ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.ascontiguousarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compresslevel=6)
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FeatureBatchError(f"invalid feature archive for {path.name}: {exc}") from exc


def _seconds_token(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else str(seconds).replace(".", "p")


def _output_paths(root: Path, job: SegmentJob) -> tuple[dict[str, Path], Path]:
    token = f"{_seconds_token(job.scale_seconds)}s"
    suffix = Path(token) / job.group / job.split / f"{job.segment_id}.npz"
    paths = {view: root / "features" / view / suffix for view in VIEW_NAMES}
    sidecar = root / "features" / "manifests" / suffix.with_suffix(".json")
    return paths, sidecar


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _bundle_arrays(bundle: FeatureBundle) -> dict[str, dict[str, np.ndarray]]:
    return {
        "acoustic": {
            "times": bundle.acoustic.times,
            "log_mel": bundle.acoustic.log_mel,
            "mfcc": bundle.acoustic.mfcc,
            "chroma": bundle.acoustic.chroma,
            "spectral_contrast": bundle.acoustic.spectral_contrast,
            "tempogram": bundle.acoustic.tempogram,
            "vectors": bundle.acoustic.vectors,
            "valid": bundle.acoustic.valid,
        },
        "chroma": {
            "times": bundle.pitch.times,
            "chroma": bundle.pitch.chroma,
            "states": bundle.pitch.states,
            "valid": bundle.pitch.valid,
        },
        "rhythm": {
            "times": bundle.rhythm.times,
            "vectors": bundle.rhythm.vectors,
            "valid": bundle.rhythm.valid,
            "onset_times": bundle.rhythm.onset_times,
            "inter_onset_intervals": bundle.rhythm.inter_onset_intervals,
            "beat_times": bundle.rhythm.beat_times,
        },
        "modulation": {
            "times": bundle.modulation.times,
            "frequencies": bundle.modulation.frequencies,
            "spectrum": bundle.modulation.spectrum,
            "band_energies": bundle.modulation.band_energies,
            "key_band_energies": bundle.modulation.key_band_energies,
            "valid": bundle.modulation.valid,
        },
        "structure": {
            "times": bundle.structure.times,
            "boundary_times": bundle.structure.boundary_times,
            "boundary_indices": bundle.structure.boundary_indices,
            "self_similarity": bundle.structure.self_similarity,
            "novelty": bundle.structure.novelty,
            "block_vectors": bundle.structure.block_vectors,
            "valid": bundle.structure.valid,
        },
    }


def _quality_payload(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, int]:
    acoustic_valid = np.asarray(arrays["acoustic"]["valid"], dtype=bool)
    pitch_states = np.asarray(arrays["chroma"]["states"])
    rhythm_valid = np.asarray(arrays["rhythm"]["valid"], dtype=bool)
    modulation_valid = np.asarray(arrays["modulation"]["valid"], dtype=bool)
    structure_valid = np.asarray(arrays["structure"]["valid"], dtype=bool)
    structure_boundaries = np.asarray(arrays["structure"]["boundary_indices"])
    return {
        "acoustic_windows": int(acoustic_valid.size),
        "pitch_steps": int(pitch_states.size),
        "rhythm_windows": int(rhythm_valid.shape[0]),
        "modulation_windows": int(modulation_valid.size),
        "structure_blocks": int(structure_valid.size),
        "structure_boundaries": int(structure_boundaries.size),
        "invalid_acoustic_windows": int(np.count_nonzero(~acoustic_valid)),
        "uncertain_pitch_steps": int(np.count_nonzero(pitch_states == 12)),
        "invalid_rhythm_values": int(np.count_nonzero(~rhythm_valid)),
        "invalid_modulation_windows": int(np.count_nonzero(~modulation_valid)),
    }


def _row_from_sidecar(
    root: Path,
    job: SegmentJob,
    sidecar: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    outputs = sidecar["outputs"]
    quality = sidecar["quality"]
    return {
        "segment_id": job.segment_id,
        "track_id": job.track_id,
        "group": job.group,
        "split": job.split,
        "scale_seconds": job.scale_seconds,
        "input_relative_path": job.input_relative_path,
        "input_sha256": job.input_sha256,
        **{f"{view}_relative_path": outputs[view]["relative_path"] for view in VIEW_NAMES},
        **{f"{view}_sha256": outputs[view]["sha256"] for view in VIEW_NAMES},
        "sidecar_relative_path": _relative(root, root / Path(sidecar["sidecar_relative_path"])),
        "config_sha256": sidecar["config_sha256"],
        "model_sha256": sidecar.get("model_sha256", ""),
        **quality,
        "status": status,
        "processed_at": sidecar["processed_at"],
        "error": "",
    }


def _verified_sidecar(
    root: Path,
    job: SegmentJob,
    *,
    config_sha256: str,
    required_model_sha256: str | None = None,
) -> dict[str, Any] | None:
    paths, sidecar_path = _output_paths(root, job)
    if not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if sidecar.get("input_sha256") != job.input_sha256:
        return None
    if sidecar.get("config_sha256") != config_sha256:
        return None
    if required_model_sha256 is not None:
        if sidecar.get("model_sha256") != required_model_sha256:
            return None
    outputs = sidecar.get("outputs", {})
    for view, path in paths.items():
        payload = outputs.get(view, {})
        if payload.get("relative_path") != _relative(root, path) or not path.is_file():
            return None
        if payload.get("sha256") != _sha256(path):
            return None
    return sidecar


def _extract_job(
    job: SegmentJob,
    *,
    root: Path,
    config: FeatureExtractionConfig,
    config_sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite:
        existing = _verified_sidecar(root, job, config_sha256=config_sha256)
        if existing is not None:
            return _row_from_sidecar(root, job, existing, status="verified_existing")

    audio_path = root / Path(job.input_relative_path)
    if not audio_path.is_file():
        raise FeatureBatchError(f"missing preprocessed audio for {job.segment_id}")
    actual_input_hash = _sha256(audio_path)
    if actual_input_hash != job.input_sha256:
        raise FeatureBatchError(f"input hash mismatch for {job.segment_id}")
    bundle = LibrosaFeatureExtractor().extract(audio_path, track_id=job.track_id, config=config)
    arrays = _bundle_arrays(bundle)
    paths, sidecar_path = _output_paths(root, job)
    output_payload: dict[str, dict[str, Any]] = {}
    for view, path in paths.items():
        _write_npz_atomic(path, arrays[view])
        output_payload[view] = {
            "relative_path": _relative(root, path),
            "sha256": _sha256(path),
            "arrays": {name: list(np.asarray(value).shape) for name, value in arrays[view].items()},
        }
    quality = _quality_payload(arrays)
    sidecar = {
        "schema_version": 1,
        "segment_id": job.segment_id,
        "track_id": job.track_id,
        "group": job.group,
        "split": job.split,
        "scale_seconds": job.scale_seconds,
        "duration_seconds": bundle.duration_seconds,
        "input_relative_path": job.input_relative_path,
        "input_sha256": job.input_sha256,
        "config_sha256": config_sha256,
        "model_sha256": "",
        "extractor": LibrosaFeatureExtractor.name,
        "extractor_version": LibrosaFeatureExtractor.version,
        "outputs": output_payload,
        "quality": quality,
        "sidecar_relative_path": _relative(root, sidecar_path),
        "processed_at": date.today().isoformat(),
        "error": "",
    }
    _write_json_atomic(sidecar_path, sidecar)
    return _row_from_sidecar(root, job, sidecar, status="extracted")


def _failure_row(job: SegmentJob, exc: BaseException, config_sha256: str) -> dict[str, Any]:
    return {
        "segment_id": job.segment_id,
        "track_id": job.track_id,
        "group": job.group,
        "split": job.split,
        "scale_seconds": job.scale_seconds,
        "input_relative_path": job.input_relative_path,
        "input_sha256": job.input_sha256,
        "config_sha256": config_sha256,
        "status": "failed",
        "processed_at": date.today().isoformat(),
        "error": str(exc),
    }


def _write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("group", ""),
            row.get("track_id", ""),
            row.get("scale_seconds", 0),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    _replace_with_retry(temporary, path)


def _load_jobs(path: Path) -> list[SegmentJob]:
    if not path.is_file():
        raise FeatureBatchError(f"preprocessed manifest not found: {path}")
    jobs: list[SegmentJob] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "failed":
                continue
            if not row.get("sha256") or not row.get("output_relative_path"):
                raise FeatureBatchError(
                    f"preprocessed row {row.get('segment_id', '<unknown>')} is incomplete"
                )
            jobs.append(
                SegmentJob(
                    segment_id=row["segment_id"],
                    track_id=row["track_id"],
                    group=row["group"],
                    split=row["split"],
                    scale_seconds=float(row["scale_seconds"]),
                    input_relative_path=row["output_relative_path"],
                    input_sha256=row["sha256"],
                )
            )
    if len({job.segment_id for job in jobs}) != len(jobs):
        raise FeatureBatchError("preprocessed manifest contains duplicate segment IDs")
    return sorted(jobs, key=lambda job: (job.group, job.track_id, job.scale_seconds))


def _filter_jobs(
    jobs: Sequence[SegmentJob],
    *,
    groups: set[str] | None,
    scales: set[float] | None,
    track_ids: set[str] | None,
) -> list[SegmentJob]:
    selected = [
        job
        for job in jobs
        if (groups is None or job.group in groups)
        and (scales is None or job.scale_seconds in scales)
        and (track_ids is None or job.track_id in track_ids)
    ]
    if track_ids is not None:
        missing = track_ids - {job.track_id for job in selected}
        if missing:
            raise FeatureBatchError(f"unknown or excluded track IDs: {sorted(missing)}")
    return selected


def extract_batch(
    jobs: Sequence[SegmentJob],
    *,
    root: Path,
    config: FeatureExtractionConfig,
    workers: int,
    overwrite: bool,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise FeatureBatchError("workers must be positive")
    config_sha256 = _config_hash(config)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _extract_job,
                job,
                root=root,
                config=config,
                config_sha256=config_sha256,
                overwrite=overwrite,
            ): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep the full batch resumable and auditable
                row = _failure_row(job, exc, config_sha256)
            rows.append(row)
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                failures = sum(item["status"] == "failed" for item in rows)
                print(
                    f"features extract: {completed}/{len(futures)} complete; failures={failures}",
                    flush=True,
                )
    _write_manifest(manifest_path, rows)
    return rows


def _backfill_structure_job(
    job: SegmentJob,
    *,
    root: Path,
    config: FeatureExtractionConfig,
    config_sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Create the structure archive from an existing acoustic archive.

    This migration path avoids decoding and re-extracting the source audio when
    a pre-structure feature corpus already contains the frame-wise acoustic
    vectors required by the SSM boundary detector.
    """

    paths, sidecar_path = _output_paths(root, job)
    if not paths["acoustic"].is_file() or not sidecar_path.is_file():
        raise FeatureBatchError(f"existing acoustic features are missing for {job.segment_id}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    structure_path = paths["structure"]
    structure: dict[str, np.ndarray]
    if structure_path.is_file() and not overwrite:
        structure = _read_npz(structure_path)
        required = {
            "times",
            "boundary_times",
            "boundary_indices",
            "self_similarity",
            "novelty",
            "block_vectors",
            "valid",
        }
        if not required.issubset(structure):
            raise FeatureBatchError(
                f"existing structure archive has an incomplete schema for {job.segment_id}"
            )
    else:
        acoustic = _read_npz(paths["acoustic"])
        features = structural_features(
            acoustic["vectors"],
            acoustic["times"],
            acoustic["valid"],
            duration=job.scale_seconds,
            kernel_seconds=config.structure_kernel_seconds,
            min_segment_seconds=config.structure_min_segment_seconds,
            max_segment_seconds=config.structure_max_segment_seconds,
            novelty_threshold=config.structure_novelty_threshold,
        )
        structure = {
            "times": features.times,
            "boundary_times": features.boundary_times,
            "boundary_indices": features.boundary_indices,
            "self_similarity": features.self_similarity,
            "novelty": features.novelty,
            "block_vectors": features.block_vectors,
            "valid": features.valid,
        }
        _write_npz_atomic(structure_path, structure)

    sidecar.setdefault("outputs", {})["structure"] = {
        "relative_path": _relative(root, structure_path),
        "sha256": _sha256(structure_path),
        "arrays": {
            name: list(np.asarray(value).shape) for name, value in structure.items()
        },
    }
    sidecar.setdefault("quality", {})["structure_blocks"] = int(
        np.asarray(structure["valid"]).size
    )
    sidecar["quality"]["structure_boundaries"] = int(
        np.asarray(structure["boundary_indices"]).size
    )
    sidecar["config_sha256"] = config_sha256
    sidecar["sidecar_relative_path"] = _relative(root, sidecar_path)
    sidecar["processed_at"] = date.today().isoformat()
    _write_json_atomic(sidecar_path, sidecar)
    return _row_from_sidecar(root, job, sidecar, status="structure_backfilled")


def backfill_structure_batch(
    jobs: Sequence[SegmentJob],
    *,
    root: Path,
    config: FeatureExtractionConfig,
    workers: int,
    overwrite: bool,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Backfill SSM-derived macro-structure features for an existing corpus."""

    if workers <= 0:
        raise FeatureBatchError("workers must be positive")
    config_sha256 = _config_hash(config)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _backfill_structure_job,
                job,
                root=root,
                config=config,
                config_sha256=config_sha256,
                overwrite=overwrite,
            ): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _failure_row(job, exc, config_sha256)
            rows.append(row)
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                failures = sum(item["status"] == "failed" for item in rows)
                print(
                    f"structure backfill: {completed}/{len(futures)} complete; "
                    f"failures={failures}",
                    flush=True,
                )
    _write_manifest(manifest_path, rows)
    return rows


def _balanced_rows(
    values_by_group: dict[str, list[np.ndarray]],
    *,
    maximum_per_group: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    if not values_by_group:
        raise FeatureBatchError("no discovery features are available for state fitting")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    counts: dict[str, int] = {}
    available = {
        group: np.concatenate(group_values, axis=0)
        for group, group_values in values_by_group.items()
    }
    balanced_count = min(maximum_per_group, *(values.shape[0] for values in available.values()))
    if balanced_count < 1:
        raise FeatureBatchError("one or more groups have no discovery features for state fitting")
    for group in sorted(available):
        values = available[group]
        if values.shape[0] > balanced_count:
            indices = np.sort(rng.choice(values.shape[0], size=balanced_count, replace=False))
            values = values[indices]
        selected.append(values)
        counts[group] = int(values.shape[0])
    return np.concatenate(selected, axis=0), counts


def _training_hash(rows: Sequence[dict[str, Any]], config_sha256: str) -> str:
    payload = [
        {
            "segment_id": row["segment_id"],
            "input_sha256": row["input_sha256"],
        }
        for row in sorted(rows, key=lambda item: item["segment_id"])
    ]
    return _json_hash(
        {
            "state_model_version": STATE_MODEL_VERSION,
            "config_sha256": config_sha256,
            "segments": payload,
        }
    )


def _load_feature_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FeatureBatchError(f"feature manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)


def fit_state_model(
    rows: Sequence[dict[str, Any]],
    *,
    root: Path,
    config: FeatureExtractionConfig,
    model_path: Path,
    metadata_path: Path,
    overwrite: bool,
) -> tuple[FeatureStateModel, str, dict[str, Any]]:
    training_rows = [
        row
        for row in rows
        if row.get("status") != "failed"
        and row.get("split") == "discovery"
        and math.isclose(float(row.get("scale_seconds", 0)), 180.0, abs_tol=1e-6)
    ]
    if not training_rows:
        raise FeatureBatchError("state fitting requires discovery 180-second features")
    if any(row.get("split") != "discovery" for row in training_rows):
        raise FeatureBatchError("non-discovery data reached state fitting")
    config_sha256 = _config_hash(config)
    training_sha256 = _training_hash(training_rows, config_sha256)
    if not overwrite and model_path.is_file() and metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_hash = _sha256(model_path)
            if (
                previous.get("config_sha256") == config_sha256
                and previous.get("training_sha256") == training_sha256
                and previous.get("model_sha256") == model_hash
            ):
                return _load_state_model(model_path), model_hash, previous
        except OSError, ValueError, FeatureBatchError:
            pass

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise FeatureBatchError(
            "state modelling requires scikit-learn; install the project with .[stats]"
        ) from exc

    acoustic_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    rhythm_values_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    rhythm_masks_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    modulation_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    structure_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in training_rows:
        group = str(row["group"])
        acoustic = _read_npz(root / Path(row["acoustic_relative_path"]))
        rhythm = _read_npz(root / Path(row["rhythm_relative_path"]))
        modulation = _read_npz(root / Path(row["modulation_relative_path"]))
        acoustic_valid = np.asarray(acoustic["valid"], dtype=bool)
        acoustic_by_group[group].append(
            np.asarray(acoustic["vectors"], dtype=np.float32)[acoustic_valid]
        )
        rhythm_values_by_group[group].append(np.asarray(rhythm["vectors"], dtype=np.float32))
        rhythm_masks_by_group[group].append(np.asarray(rhythm["valid"], dtype=bool))
        modulation_valid = np.asarray(modulation["valid"], dtype=bool)
        modulation_by_group[group].append(
            np.asarray(modulation["key_band_energies"], dtype=np.float32)[modulation_valid]
        )
        structure_path = row.get("structure_relative_path")
        if structure_path:
            structure = _read_npz(root / Path(structure_path))
            structure_valid = np.asarray(structure["valid"], dtype=bool)
            structure_by_group[group].append(
                np.asarray(structure["block_vectors"], dtype=np.float32)[structure_valid]
            )
        else:
            # Compatibility with pre-structure manifests: acoustic windows still
            # provide a valid training matrix until features are regenerated.
            structure_by_group[group].append(
                np.asarray(acoustic["vectors"], dtype=np.float32)[acoustic_valid]
            )

    acoustic_values, acoustic_counts = _balanced_rows(
        acoustic_by_group,
        maximum_per_group=config.max_fit_windows_per_group,
        seed=config.random_seed,
    )
    rhythm_values, rhythm_counts = _balanced_rows(
        rhythm_values_by_group,
        maximum_per_group=config.max_fit_windows_per_group,
        seed=config.random_seed + 1,
    )
    rhythm_masks, _ = _balanced_rows(
        {
            group: [mask.astype(np.float32) for mask in masks]
            for group, masks in rhythm_masks_by_group.items()
        },
        maximum_per_group=config.max_fit_windows_per_group,
        seed=config.random_seed + 1,
    )
    rhythm_masks = rhythm_masks.astype(bool)
    modulation_values, modulation_counts = _balanced_rows(
        modulation_by_group,
        maximum_per_group=config.max_fit_windows_per_group,
        seed=config.random_seed + 2,
    )
    structure_values, structure_counts = _balanced_rows(
        structure_by_group,
        maximum_per_group=config.max_fit_windows_per_group,
        seed=config.random_seed + 3,
    )

    rhythm_impute = np.zeros(rhythm_values.shape[1], dtype=np.float64)
    for column in range(rhythm_values.shape[1]):
        observed = rhythm_values[rhythm_masks[:, column], column]
        rhythm_impute[column] = float(np.median(observed)) if observed.size else 0.0
    rhythm_filled = np.where(rhythm_masks, rhythm_values, rhythm_impute)
    rhythm_scaler = StandardScaler().fit(rhythm_filled)
    rhythm_scaled = rhythm_scaler.transform(rhythm_filled)
    rhythm_kmeans = MiniBatchKMeans(
        n_clusters=config.rhythm_clusters,
        random_state=config.random_seed,
        batch_size=4_096,
        n_init=10,
        max_iter=200,
    ).fit(rhythm_scaled)

    acoustic_scaler = StandardScaler().fit(acoustic_values)
    acoustic_scaled = acoustic_scaler.transform(acoustic_values)
    pca = PCA(
        n_components=config.acoustic_pca_components,
        svd_solver="randomized",
        random_state=config.random_seed,
    ).fit(acoustic_scaled)
    reduced = pca.transform(acoustic_scaled)
    acoustic_kmeans = MiniBatchKMeans(
        n_clusters=config.acoustic_clusters,
        random_state=config.random_seed,
        batch_size=4_096,
        n_init=10,
        max_iter=200,
    ).fit(reduced)
    structure_scaled = (structure_values - acoustic_scaler.mean_) / _safe_scale(
        acoustic_scaler.scale_
    )
    structure_reduced = pca.transform(structure_scaled)
    structure_kmeans = MiniBatchKMeans(
        n_clusters=config.structure_clusters,
        random_state=config.random_seed + 3,
        batch_size=4_096,
        n_init=10,
        max_iter=200,
    ).fit(structure_reduced)
    modulation_edges = np.quantile(modulation_values, [1.0 / 3.0, 2.0 / 3.0], axis=0).T

    model = FeatureStateModel(
        rhythm_impute=np.asarray(rhythm_impute, dtype=np.float32),
        rhythm_mean=np.asarray(rhythm_scaler.mean_, dtype=np.float32),
        rhythm_scale=np.asarray(_safe_scale(rhythm_scaler.scale_), dtype=np.float32),
        rhythm_centers=np.asarray(rhythm_kmeans.cluster_centers_, dtype=np.float32),
        modulation_edges=np.asarray(modulation_edges, dtype=np.float32),
        acoustic_mean=np.asarray(acoustic_scaler.mean_, dtype=np.float32),
        acoustic_scale=np.asarray(_safe_scale(acoustic_scaler.scale_), dtype=np.float32),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        acoustic_centers=np.asarray(acoustic_kmeans.cluster_centers_, dtype=np.float32),
        structure_centers=np.asarray(structure_kmeans.cluster_centers_, dtype=np.float32),
    )
    _write_npz_atomic(model_path, model.arrays())
    model_sha256 = _sha256(model_path)
    metadata = {
        "schema_version": 1,
        "state_model_version": STATE_MODEL_VERSION,
        "model_sha256": model_sha256,
        "config_sha256": config_sha256,
        "training_sha256": training_sha256,
        "random_seed": config.random_seed,
        "training_split": "discovery",
        "training_scale_seconds": 180.0,
        "training_segment_ids": sorted(row["segment_id"] for row in training_rows),
        "training_groups": dict(Counter(row["group"] for row in training_rows)),
        "sampled_windows": {
            "acoustic": acoustic_counts,
            "rhythm": rhythm_counts,
            "modulation": modulation_counts,
            "structure": structure_counts,
        },
        "dimensions": {
            "acoustic_input": int(acoustic_values.shape[1]),
            "acoustic_pca": config.acoustic_pca_components,
            "acoustic_clusters": config.acoustic_clusters,
            "rhythm_input": int(rhythm_values.shape[1]),
            "rhythm_clusters": config.rhythm_clusters,
            "modulation_key_bands": int(modulation_values.shape[1]),
            "structure_clusters": config.structure_clusters,
        },
        "generated_at": date.today().isoformat(),
    }
    _write_json_atomic(metadata_path, metadata)
    return model, model_sha256, metadata


def _load_state_model(path: Path) -> FeatureStateModel:
    arrays = _read_npz(path)
    expected = set(FeatureStateModel.__dataclass_fields__)
    if set(arrays) != expected:
        raise FeatureBatchError("state model archive has an unexpected schema")
    return FeatureStateModel(**{name: arrays[name] for name in expected})


def _nearest_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(values * values, axis=1, keepdims=True)
        - 2.0 * values @ centers.T
        + np.sum(centers * centers, axis=1)[None, :]
    )
    return np.argmin(distances, axis=1).astype(np.int16)


def _transform_job(
    job: SegmentJob,
    *,
    root: Path,
    config_sha256: str,
    model: FeatureStateModel,
    model_sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite:
        existing = _verified_sidecar(
            root,
            job,
            config_sha256=config_sha256,
            required_model_sha256=model_sha256,
        )
        if existing is not None:
            return _row_from_sidecar(root, job, existing, status="verified_existing")

    paths, sidecar_path = _output_paths(root, job)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    acoustic = _read_npz(paths["acoustic"])
    rhythm = _read_npz(paths["rhythm"])
    modulation = _read_npz(paths["modulation"])
    structure = _read_npz(paths["structure"])

    acoustic_values = np.asarray(acoustic["vectors"], dtype=np.float32)
    acoustic_scaled = (acoustic_values - model.acoustic_mean) / model.acoustic_scale
    acoustic_reduced = (acoustic_scaled - model.pca_mean) @ model.pca_components.T
    acoustic["prototype_states"] = _nearest_centers(acoustic_reduced, model.acoustic_centers)
    acoustic["prototype_states"][~np.asarray(acoustic["valid"], dtype=bool)] = -1

    rhythm_values = np.asarray(rhythm["vectors"], dtype=np.float32)
    rhythm_valid = np.asarray(rhythm["valid"], dtype=bool)
    rhythm_filled = np.where(rhythm_valid, rhythm_values, model.rhythm_impute)
    rhythm_scaled = (rhythm_filled - model.rhythm_mean) / model.rhythm_scale
    rhythm["states"] = _nearest_centers(rhythm_scaled, model.rhythm_centers)

    modulation_values = np.asarray(modulation["key_band_energies"], dtype=np.float32)
    modulation_valid = np.asarray(modulation["valid"], dtype=bool)
    modulation_states = np.column_stack(
        [
            np.digitize(modulation_values[:, index], model.modulation_edges[index])
            for index in range(modulation_values.shape[1])
        ]
    ).astype(np.int16)
    modulation_states[~modulation_valid] = -1
    modulation["states"] = modulation_states

    structure_values = np.asarray(structure["block_vectors"], dtype=np.float32)
    structure_valid = np.asarray(structure["valid"], dtype=bool)
    structure_scaled = (structure_values - model.acoustic_mean) / model.acoustic_scale
    structure_reduced = (structure_scaled - model.pca_mean) @ model.pca_components.T
    structure["states"] = _nearest_centers(structure_reduced, model.structure_centers)
    structure["states"][~structure_valid] = -1

    transformed = {
        "acoustic": acoustic,
        "chroma": _read_npz(paths["chroma"]),
        "rhythm": rhythm,
        "modulation": modulation,
        "structure": structure,
    }
    for view, path in paths.items():
        _write_npz_atomic(path, transformed[view])
        sidecar["outputs"][view] = {
            "relative_path": _relative(root, path),
            "sha256": _sha256(path),
            "arrays": {
                name: list(np.asarray(value).shape) for name, value in transformed[view].items()
            },
        }
    sidecar["model_sha256"] = model_sha256
    sidecar["processed_at"] = date.today().isoformat()
    _write_json_atomic(sidecar_path, sidecar)
    return _row_from_sidecar(root, job, sidecar, status="transformed")


def transform_batch(
    jobs: Sequence[SegmentJob],
    *,
    root: Path,
    config: FeatureExtractionConfig,
    model: FeatureStateModel,
    model_sha256: str,
    workers: int,
    overwrite: bool,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    config_sha256 = _config_hash(config)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _transform_job,
                job,
                root=root,
                config_sha256=config_sha256,
                model=model,
                model_sha256=model_sha256,
                overwrite=overwrite,
            ): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _failure_row(job, exc, config_sha256)
            rows.append(row)
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                failures = sum(item["status"] == "failed" for item in rows)
                print(
                    f"features transform: {completed}/{len(futures)} complete; failures={failures}",
                    flush=True,
                )
    _write_manifest(manifest_path, rows)
    return rows


def _summary(
    rows: Sequence[dict[str, Any]],
    *,
    root: Path,
    config: FeatureExtractionConfig,
    manifest_path: Path,
    model_sha256: str,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") != "failed"]
    output_paths = [
        root / Path(row[f"{view}_relative_path"]) for row in successful for view in VIEW_NAMES
    ]
    output_bytes = sum(path.stat().st_size for path in output_paths if path.is_file())
    state_coverage: dict[str, Any] = {}
    if model_sha256:
        acoustic_states: set[int] = set()
        pitch_states: set[int] = set()
        rhythm_states: set[int] = set()
        modulation_states: set[tuple[int, ...]] = set()
        structure_states: set[int] = set()
        for row in successful:
            acoustic = _read_npz(root / Path(row["acoustic_relative_path"]))
            chroma = _read_npz(root / Path(row["chroma_relative_path"]))
            rhythm = _read_npz(root / Path(row["rhythm_relative_path"]))
            modulation = _read_npz(root / Path(row["modulation_relative_path"]))
            structure = _read_npz(root / Path(row["structure_relative_path"]))
            acoustic_states.update(
                int(value)
                for value in acoustic.get("prototype_states", np.array([], dtype=int))
                if value >= 0
            )
            pitch_states.update(int(value) for value in chroma["states"])
            rhythm_states.update(
                int(value) for value in rhythm.get("states", np.array([], dtype=int)) if value >= 0
            )
            for state in modulation.get("states", np.empty((0, 3), dtype=int)):
                key = tuple(int(value) for value in state)
                if all(value >= 0 for value in key):
                    modulation_states.add(key)
            structure_states.update(
                int(value)
                for value in structure.get("states", np.array([], dtype=int))
                if value >= 0
            )
        state_coverage = {
            "acoustic_prototypes": sorted(acoustic_states),
            "pitch_states": sorted(pitch_states),
            "rhythm_states": sorted(rhythm_states),
            "modulation_states_observed": len(modulation_states),
            "modulation_states_possible": 27,
            "structure_states": sorted(structure_states),
        }
    return {
        "generated_at": date.today().isoformat(),
        "ok": len(successful) == len(rows),
        "manifest": str(manifest_path),
        "segments": len(rows),
        "tracks": len({row["track_id"] for row in rows}),
        "status_counts": dict(Counter(row.get("status", "") for row in rows)),
        "group_counts": dict(Counter(row["group"] for row in successful)),
        "split_counts": dict(Counter(row["split"] for row in successful)),
        "scale_counts": dict(
            Counter(f"{_seconds_token(float(row['scale_seconds']))}s" for row in successful)
        ),
        "quality": {
            key: sum(int(row.get(key, 0) or 0) for row in successful)
            for key in (
                "acoustic_windows",
                "pitch_steps",
                "rhythm_windows",
                "modulation_windows",
                "invalid_acoustic_windows",
                "uncertain_pitch_steps",
                "invalid_rhythm_values",
                "invalid_modulation_windows",
                "structure_blocks",
                "structure_boundaries",
            )
        },
        "state_coverage": state_coverage,
        "output_files": len(output_paths),
        "output_bytes": output_bytes,
        "output_gib": round(output_bytes / (1024**3), 3),
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "model_sha256": model_sha256,
        "manifest_sha256": _sha256(manifest_path),
    }


def _load_config(root: Path) -> FeatureExtractionConfig:
    config_path = root / "configs" / "pipeline.toml"
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    payload = raw.get("features", {})
    aliases = {
        "analysis_sample_rate": "sample_rate",
        "window_seconds": "analysis_window_seconds",
        "window_hop_seconds": "analysis_hop_seconds",
    }
    for source, target in aliases.items():
        if source in payload and target not in payload:
            payload[target] = payload.pop(source)
    known = set(FeatureExtractionConfig.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        raise FeatureBatchError(f"unknown feature configuration keys: {sorted(unknown)}")
    config = FeatureExtractionConfig(**payload)
    config.validate()
    return config


def _parse_csv_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {value.strip() for value in raw.split(",") if value.strip()}
    return values or None


def _parse_scales(raw: str | None) -> set[float] | None:
    if raw is None:
        return None
    try:
        values = {float(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise FeatureBatchError("scales must be comma-separated seconds") from exc
    if any(value <= 0 for value in values):
        raise FeatureBatchError("scales must be positive")
    return values or None


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--groups", default="focus,classical")
    parser.add_argument("--scales", default="180,300")
    parser.add_argument("--track-ids")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--input-manifest", type=Path, default=Path("metadata/preprocessed_segments.csv")
    )
    parser.add_argument(
        "--feature-manifest", type=Path, default=Path("metadata/feature_segments.csv")
    )
    parser.add_argument("--summary", type=Path, default=Path("metadata/feature_summary.json"))
    parser.add_argument("--model", type=Path, default=Path("features/models/state_model.npz"))
    parser.add_argument(
        "--model-metadata", type=Path, default=Path("features/models/state_model.json")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-features")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "extract",
        "backfill-structure",
        "fit-states",
        "transform-states",
        "run",
    ):
        child = subparsers.add_parser(command)
        _add_common_arguments(child)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = _load_config(root)
    input_manifest = _resolve(root, args.input_manifest)
    feature_manifest = _resolve(root, args.feature_manifest)
    summary_path = _resolve(root, args.summary)
    model_path = _resolve(root, args.model)
    model_metadata_path = _resolve(root, args.model_metadata)
    jobs = _filter_jobs(
        _load_jobs(input_manifest),
        groups=_parse_csv_set(args.groups),
        scales=_parse_scales(args.scales),
        track_ids=_parse_csv_set(args.track_ids),
    )
    if not jobs:
        raise FeatureBatchError("no segments matched the requested filters")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "segments": len(jobs),
                    "tracks": len({job.track_id for job in jobs}),
                    "groups": dict(Counter(job.group for job in jobs)),
                    "splits": dict(Counter(job.split for job in jobs)),
                    "scales": dict(
                        Counter(f"{_seconds_token(job.scale_seconds)}s" for job in jobs)
                    ),
                    "config_sha256": _config_hash(config),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "backfill-structure":
        rows = backfill_structure_batch(
            jobs,
            root=root,
            config=config,
            workers=args.workers,
            overwrite=args.overwrite,
            manifest_path=feature_manifest,
        )
        payload = _summary(
            rows,
            root=root,
            config=config,
            manifest_path=feature_manifest,
            model_sha256="",
        )
        _write_json_atomic(summary_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["ok"] else 1

    if args.command in {"extract", "run"}:
        rows = extract_batch(
            jobs,
            root=root,
            config=config,
            workers=args.workers,
            overwrite=args.overwrite,
            manifest_path=feature_manifest,
        )
        if any(row.get("status") == "failed" for row in rows):
            payload = _summary(
                rows,
                root=root,
                config=config,
                manifest_path=feature_manifest,
                model_sha256="",
            )
            _write_json_atomic(summary_path, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1
        if args.command == "extract":
            payload = _summary(
                rows,
                root=root,
                config=config,
                manifest_path=feature_manifest,
                model_sha256="",
            )
            _write_json_atomic(summary_path, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
    else:
        rows = _load_feature_manifest(feature_manifest)
        selected_ids = {job.segment_id for job in jobs}
        rows = [row for row in rows if row.get("segment_id") in selected_ids]

    model: FeatureStateModel
    model_sha256: str
    if args.command in {"fit-states", "run"}:
        model, model_sha256, metadata = fit_state_model(
            rows,
            root=root,
            config=config,
            model_path=model_path,
            metadata_path=model_metadata_path,
            overwrite=args.overwrite,
        )
        if args.command == "fit-states":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return 0
    else:
        model = _load_state_model(model_path)
        model_sha256 = _sha256(model_path)

    final_rows = transform_batch(
        jobs,
        root=root,
        config=config,
        model=model,
        model_sha256=model_sha256,
        workers=args.workers,
        overwrite=args.overwrite,
        manifest_path=feature_manifest,
    )
    payload = _summary(
        final_rows,
        root=root,
        config=config,
        manifest_path=feature_manifest,
        model_sha256=model_sha256,
    )
    _write_json_atomic(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FeatureBatchError, FeatureExtractionError, OSError, ValueError) as exc:
        print(f"focus-features: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
