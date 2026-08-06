from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from features.batch import (
    _json_hash,
    _read_npz,
    _replace_with_retry,
    _sha256,
    _write_json_atomic,
    _write_npz_atomic,
)
from graphs.transition import TransitionGraph, build_transition_graph
from homology.glmy import PersistentPathResult, persistent_path_homology

TOPOLOGY_MODEL_VERSION = 3
VIEW_FEATURE_COLUMNS = {
    "pitch": ("chroma_relative_path", "chroma_sha256"),
    "rhythm": ("rhythm_relative_path", "rhythm_sha256"),
    "modulation": ("modulation_relative_path", "modulation_sha256"),
    "modulation_tertile": (
        "modulation_tertile_relative_path",
        "modulation_tertile_sha256",
    ),
    "structure": ("structure_relative_path", "structure_sha256"),
}
SEGMENT_COLUMNS = (
    "segment_id",
    "track_id",
    "group",
    "split",
    "scale_seconds",
    "view",
    "feature_relative_path",
    "feature_sha256",
    "graph_relative_path",
    "graph_sha256",
    "persistence_relative_path",
    "persistence_sha256",
    "sensitivity_persistence_relative_path",
    "sensitivity_persistence_sha256",
    "sidecar_relative_path",
    "config_sha256",
    "sequence_length",
    "valid_states",
    "valid_transitions",
    "self_transitions",
    "self_transition_ratio",
    "vertex_count",
    "edge_count",
    "edge_density",
    "reciprocity",
    "transition_entropy",
    "path_entropy",
    "path_entropy_normalized",
    "directed_recurrence",
    "directed_recurrence_unbiased",
    "h0_betti_auc",
    "h1_betti_auc",
    "h0_betti_mean",
    "h1_betti_mean",
    "h0_betti_max",
    "h1_betti_max",
    "h0_interval_count",
    "h1_interval_count",
    "h0_observed_persistence",
    "h1_observed_persistence",
    "h0_censored_count",
    "h1_censored_count",
    "status",
    "processed_at",
    "error",
)
FILTRATION_COLUMNS = (
    "segment_id",
    "track_id",
    "group",
    "split",
    "scale_seconds",
    "view",
    "threshold",
    "vertex_count",
    "edge_count",
    "h0_betti",
    "h0_allowed_path_count",
    "h0_omega_dimension",
    "h0_cycle_dimension",
    "h0_boundary_rank",
    "h1_betti",
    "h1_allowed_path_count",
    "h1_omega_dimension",
    "h1_cycle_dimension",
    "h1_boundary_rank",
)


class TopologyBatchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    views: tuple[str, ...] = ("pitch", "rhythm", "modulation", "structure")
    top_k: int = 6
    include_self_loops: bool = False
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    sensitivity_thresholds: tuple[float, ...] = (
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
    )
    max_reported_dimension: int = 1
    rank_tolerance: float = 1e-9

    def validate(self) -> None:
        if not self.views or set(self.views) - set(VIEW_FEATURE_COLUMNS):
            raise TopologyBatchError(f"unsupported graph views: {self.views}")
        if self.top_k < 1:
            raise TopologyBatchError("graph top_k must be positive")
        if not self.thresholds or any(not 0.0 <= value <= 1.0 for value in self.thresholds):
            raise TopologyBatchError("graph thresholds must be non-empty and in [0, 1]")
        if not self.sensitivity_thresholds or any(
            not 0.0 <= value <= 1.0 for value in self.sensitivity_thresholds
        ):
            raise TopologyBatchError("graph sensitivity_thresholds must be non-empty and in [0, 1]")
        if not set(self.thresholds).issubset(self.sensitivity_thresholds):
            raise TopologyBatchError("sensitivity thresholds must contain all primary thresholds")
        if self.max_reported_dimension != 1:
            raise TopologyBatchError("the production persistence runner currently supports H0/H1")
        if self.rank_tolerance <= 0:
            raise TopologyBatchError("rank_tolerance must be positive")


@dataclass(frozen=True, slots=True)
class TopologyJob:
    segment_id: str
    track_id: str
    group: str
    split: str
    scale_seconds: float
    view: str
    feature_relative_path: str
    feature_sha256: str


def _seconds_token(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else str(seconds).replace(".", "p")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _config_hash(config: TopologyConfig) -> str:
    return _json_hash({"topology_model_version": TOPOLOGY_MODEL_VERSION, **asdict(config)})


def load_topology_config(root: Path) -> TopologyConfig:
    path = root / "configs" / "pipeline.toml"
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    graph = raw.get("graph", {})
    homology = raw.get("homology", {})
    config = TopologyConfig(
        views=tuple(str(value) for value in graph.get("views", TopologyConfig.views)),
        top_k=int(graph.get("top_k", TopologyConfig.top_k)),
        include_self_loops=bool(graph.get("include_self_loops", TopologyConfig.include_self_loops)),
        thresholds=tuple(
            float(value) for value in graph.get("thresholds", TopologyConfig.thresholds)
        ),
        sensitivity_thresholds=tuple(
            float(value)
            for value in graph.get("sensitivity_thresholds", TopologyConfig.sensitivity_thresholds)
        ),
        max_reported_dimension=int(
            homology.get("max_reported_dimension", TopologyConfig.max_reported_dimension)
        ),
        rank_tolerance=float(homology.get("rank_tolerance", TopologyConfig.rank_tolerance)),
    )
    config.validate()
    return config


def _load_feature_jobs(path: Path, config: TopologyConfig) -> list[TopologyJob]:
    if not path.is_file():
        raise TopologyBatchError(f"feature manifest not found: {path}")
    jobs: list[TopologyJob] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "failed":
                continue
            for view in config.views:
                path_column, hash_column = VIEW_FEATURE_COLUMNS[view]
                if not row.get(path_column) or not row.get(hash_column):
                    raise TopologyBatchError(
                        f"feature row {row.get('segment_id', '<unknown>')} lacks {view} state data"
                    )
                jobs.append(
                    TopologyJob(
                        segment_id=row["segment_id"],
                        track_id=row["track_id"],
                        group=row["group"],
                        split=row["split"],
                        scale_seconds=float(row["scale_seconds"]),
                        view=view,
                        feature_relative_path=row[path_column],
                        feature_sha256=row[hash_column],
                    )
                )
    keys = {(job.segment_id, job.view) for job in jobs}
    if len(keys) != len(jobs):
        raise TopologyBatchError("feature manifest contains duplicate segment/view rows")
    return sorted(jobs, key=lambda job: (job.group, job.track_id, job.scale_seconds, job.view))


def _filter_jobs(
    jobs: Sequence[TopologyJob],
    *,
    groups: set[str] | None,
    scales: set[float] | None,
    track_ids: set[str] | None,
) -> list[TopologyJob]:
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
            raise TopologyBatchError(f"unknown or excluded track IDs: {sorted(missing)}")
    return selected


def _output_paths(root: Path, job: TopologyJob) -> tuple[Path, Path, Path, Path]:
    suffix = (
        Path(job.view)
        / f"{_seconds_token(job.scale_seconds)}s"
        / job.group
        / job.split
        / f"{job.segment_id}.npz"
    )
    graph = root / "graphs" / suffix
    persistence = root / "homology" / "persistence" / suffix
    sensitivity_persistence = root / "homology" / "persistence_sensitivity" / suffix
    sidecar = root / "homology" / "descriptors" / suffix.with_suffix(".json")
    return graph, persistence, sensitivity_persistence, sidecar


def _load_state_sequence(path: Path, view: str) -> list[int | None]:
    arrays = _read_npz(path)
    if "states" not in arrays:
        raise TopologyBatchError(f"{path.name} has no fitted state sequence")
    states = np.asarray(arrays["states"])
    if view in {"pitch", "rhythm", "structure", "modulation_tertile"}:
        if states.ndim != 1:
            raise TopologyBatchError(f"{view} states must be one-dimensional")
        return [int(value) if np.isfinite(value) and value >= 0 else None for value in states]
    if states.ndim != 2 or states.shape[1] != 3:
        raise TopologyBatchError("modulation states must have shape (n_windows, 3)")
    output: list[int | None] = []
    for row in states:
        if not np.all(np.isfinite(row)) or np.any(row < 0) or np.any(row > 2):
            output.append(None)
        else:
            output.append(int(row[0]) * 9 + int(row[1]) * 3 + int(row[2]))
    return output


def _graph_metrics(states: Sequence[int | None], graph: TransitionGraph) -> dict[str, Any]:
    valid_states = sum(state is not None for state in states)
    pairs = [
        (source, target)
        for source, target in zip(states, states[1:], strict=False)
        if source is not None and target is not None
    ]
    self_transitions = sum(source == target for source, target in pairs)
    transition_counts = Counter(pairs)
    source_counts = Counter(source for source, _ in pairs)
    transition_total = len(pairs)
    probabilities = np.asarray(list(transition_counts.values()), dtype=float)
    if probabilities.size:
        probabilities /= probabilities.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        if probabilities.size > 1:
            entropy /= math.log(probabilities.size)
    else:
        entropy = 0.0
    if transition_total:
        path_entropy = float(
            -sum(
                (count / transition_total) * math.log(count / source_counts[source])
                for (source, _), count in transition_counts.items()
            )
        )
        directed_recurrence = float(
            sum(count * count for count in transition_counts.values()) / transition_total**2
        )
    else:
        path_entropy = 0.0
        directed_recurrence = 0.0
    observed_state_count = len({state for state in states if state is not None})
    path_entropy_normalized = (
        path_entropy / math.log(observed_state_count) if observed_state_count > 1 else 0.0
    )
    directed_recurrence_unbiased = (
        sum(count * (count - 1) for count in transition_counts.values())
        / (transition_total * (transition_total - 1))
        if transition_total > 1
        else 0.0
    )
    edge_pairs = set(graph.edge_pairs)
    reciprocal = sum((target, source) in edge_pairs for source, target in edge_pairs)
    possible = len(graph.vertices) * max(0, len(graph.vertices) - 1)
    return {
        "sequence_length": len(states),
        "valid_states": valid_states,
        "valid_transitions": len(pairs),
        "self_transitions": self_transitions,
        "self_transition_ratio": self_transitions / len(pairs) if pairs else 0.0,
        "vertex_count": len(graph.vertices),
        "edge_count": len(graph.edges),
        "edge_density": len(graph.edges) / possible if possible else 0.0,
        "reciprocity": reciprocal / len(edge_pairs) if edge_pairs else 0.0,
        "transition_entropy": entropy,
        "path_entropy": path_entropy,
        "path_entropy_normalized": path_entropy_normalized,
        "directed_recurrence": directed_recurrence,
        "directed_recurrence_unbiased": directed_recurrence_unbiased,
    }


def _persistence_arrays(result: PersistentPathResult) -> dict[str, np.ndarray]:
    intervals = result.intervals
    return {
        "thresholds": np.asarray(result.thresholds, dtype=np.float64),
        "h0_betti": np.asarray([row["h0_betti"] for row in result.descriptors], dtype=np.int16),
        "h1_betti": np.asarray([row["h1_betti"] for row in result.descriptors], dtype=np.int16),
        "vertex_count": np.asarray(
            [row["vertex_count"] for row in result.descriptors], dtype=np.int16
        ),
        "edge_count": np.asarray([row["edge_count"] for row in result.descriptors], dtype=np.int16),
        "h0_rank_invariant": result.h0_rank_invariant,
        "h1_rank_invariant": result.h1_rank_invariant,
        "interval_dimension": np.asarray([item.dimension for item in intervals], dtype=np.int8),
        "interval_birth_index": np.asarray(
            [item.birth_index for item in intervals], dtype=np.int16
        ),
        "interval_death_index": np.asarray(
            [-1 if item.death_index is None else item.death_index for item in intervals],
            dtype=np.int16,
        ),
        "interval_birth_threshold": np.asarray(
            [item.birth_threshold for item in intervals], dtype=np.float64
        ),
        "interval_death_threshold": np.asarray(
            [
                np.nan if item.death_threshold is None else item.death_threshold
                for item in intervals
            ],
            dtype=np.float64,
        ),
        "interval_lifetime": np.asarray([item.lifetime for item in intervals], dtype=np.float64),
        "interval_multiplicity": np.asarray(
            [item.multiplicity for item in intervals], dtype=np.int16
        ),
        "interval_censored": np.asarray([item.censored for item in intervals], dtype=bool),
    }


def _topology_metrics(result: PersistentPathResult) -> dict[str, Any]:
    thresholds = np.asarray(result.thresholds, dtype=float)
    output: dict[str, Any] = {}
    for dimension in (0, 1):
        betti = np.asarray([row[f"h{dimension}_betti"] for row in result.descriptors], dtype=float)
        intervals = [item for item in result.intervals if item.dimension == dimension]
        output.update(
            {
                f"h{dimension}_betti_auc": float(np.trapezoid(betti[::-1], thresholds[::-1])),
                f"h{dimension}_betti_mean": float(np.mean(betti)),
                f"h{dimension}_betti_max": int(np.max(betti)),
                f"h{dimension}_interval_count": sum(item.multiplicity for item in intervals),
                f"h{dimension}_observed_persistence": sum(
                    item.lifetime * item.multiplicity for item in intervals
                ),
                f"h{dimension}_censored_count": sum(
                    item.multiplicity for item in intervals if item.censored
                ),
            }
        )
    return output


def _graph_arrays(graph: TransitionGraph) -> dict[str, np.ndarray]:
    return {
        "vertices": np.asarray(graph.vertices, dtype=np.int16),
        "edge_source": np.asarray([edge.source for edge in graph.edges], dtype=np.int16),
        "edge_target": np.asarray([edge.target for edge in graph.edges], dtype=np.int16),
        "edge_weight": np.asarray([edge.weight for edge in graph.edges], dtype=np.float64),
        "edge_count": np.asarray([edge.count for edge in graph.edges], dtype=np.int32),
    }


def _filtration_rows(job: TopologyJob, result: PersistentPathResult) -> list[dict[str, Any]]:
    identity = {
        "segment_id": job.segment_id,
        "track_id": job.track_id,
        "group": job.group,
        "split": job.split,
        "scale_seconds": job.scale_seconds,
        "view": job.view,
    }
    return [{**identity, **row} for row in result.descriptors]


def _verified_existing(
    sidecar_path: Path,
    *,
    root: Path,
    feature_sha256: str,
    config_sha256: str,
) -> dict[str, Any] | None:
    if not sidecar_path.is_file():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if payload.get("feature_sha256") != feature_sha256:
            return None
        if payload.get("config_sha256") != config_sha256:
            return None
        for item in payload["outputs"].values():
            output = root / Path(item["relative_path"])
            if not output.is_file() or _sha256(output) != item["sha256"]:
                return None
        return payload
    except KeyError, OSError, ValueError, json.JSONDecodeError:
        return None


def _row_from_sidecar(payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        **payload["identity"],
        "feature_relative_path": payload["feature_relative_path"],
        "feature_sha256": payload["feature_sha256"],
        "graph_relative_path": payload["outputs"]["graph"]["relative_path"],
        "graph_sha256": payload["outputs"]["graph"]["sha256"],
        "persistence_relative_path": payload["outputs"]["persistence"]["relative_path"],
        "persistence_sha256": payload["outputs"]["persistence"]["sha256"],
        "sensitivity_persistence_relative_path": payload["outputs"]["sensitivity_persistence"][
            "relative_path"
        ],
        "sensitivity_persistence_sha256": payload["outputs"]["sensitivity_persistence"]["sha256"],
        "sidecar_relative_path": payload["sidecar_relative_path"],
        "config_sha256": payload["config_sha256"],
        **payload["graph_metrics"],
        **payload["topology_metrics"],
        "status": status,
        "processed_at": payload["processed_at"],
        "error": "",
        "filtration": payload["filtration"],
        "sensitivity_filtration": payload["sensitivity_filtration"],
    }


def _process_job(
    job: TopologyJob,
    *,
    root: Path,
    config: TopologyConfig,
    config_sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    feature_path = root / Path(job.feature_relative_path)
    if not feature_path.is_file():
        raise TopologyBatchError(f"state archive not found: {feature_path}")
    actual_feature_sha256 = _sha256(feature_path)
    if actual_feature_sha256 != job.feature_sha256:
        raise TopologyBatchError(f"state archive hash mismatch: {job.segment_id}/{job.view}")
    graph_path, persistence_path, sensitivity_persistence_path, sidecar_path = _output_paths(
        root, job
    )
    if not overwrite:
        existing = _verified_existing(
            sidecar_path,
            root=root,
            feature_sha256=actual_feature_sha256,
            config_sha256=config_sha256,
        )
        if existing is not None:
            return _row_from_sidecar(existing, status="verified_existing")

    states = _load_state_sequence(feature_path, job.view)
    graph = build_transition_graph(
        states,
        normalize=True,
        top_k=config.top_k,
        include_self_loops=config.include_self_loops,
    )
    result = persistent_path_homology(
        graph,
        config.thresholds,
        tolerance=config.rank_tolerance,
    )
    sensitivity_result = persistent_path_homology(
        graph,
        config.sensitivity_thresholds,
        tolerance=config.rank_tolerance,
    )
    _write_npz_atomic(graph_path, _graph_arrays(graph))
    _write_npz_atomic(persistence_path, _persistence_arrays(result))
    _write_npz_atomic(sensitivity_persistence_path, _persistence_arrays(sensitivity_result))
    identity = {
        "segment_id": job.segment_id,
        "track_id": job.track_id,
        "group": job.group,
        "split": job.split,
        "scale_seconds": job.scale_seconds,
        "view": job.view,
    }
    sidecar = {
        "schema_version": TOPOLOGY_MODEL_VERSION,
        "identity": identity,
        "feature_relative_path": job.feature_relative_path,
        "feature_sha256": actual_feature_sha256,
        "config_sha256": config_sha256,
        "outputs": {
            "graph": {
                "relative_path": _relative(root, graph_path),
                "sha256": _sha256(graph_path),
            },
            "persistence": {
                "relative_path": _relative(root, persistence_path),
                "sha256": _sha256(persistence_path),
            },
            "sensitivity_persistence": {
                "relative_path": _relative(root, sensitivity_persistence_path),
                "sha256": _sha256(sensitivity_persistence_path),
            },
        },
        "graph_metrics": _graph_metrics(states, graph),
        "topology_metrics": _topology_metrics(result),
        "filtration": _filtration_rows(job, result),
        "sensitivity_filtration": _filtration_rows(job, sensitivity_result),
        "sidecar_relative_path": _relative(root, sidecar_path),
        "processed_at": date.today().isoformat(),
        "error": "",
    }
    _write_json_atomic(sidecar_path, sidecar)
    return _row_from_sidecar(sidecar, status="success")


def _failure_row(job: TopologyJob, exc: Exception, config_sha256: str) -> dict[str, Any]:
    return {
        "segment_id": job.segment_id,
        "track_id": job.track_id,
        "group": job.group,
        "split": job.split,
        "scale_seconds": job.scale_seconds,
        "view": job.view,
        "feature_relative_path": job.feature_relative_path,
        "feature_sha256": job.feature_sha256,
        "config_sha256": config_sha256,
        "status": "failed",
        "processed_at": date.today().isoformat(),
        "error": f"{type(exc).__name__}: {exc}",
        "filtration": [],
        "sensitivity_filtration": [],
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _replace_with_retry(temporary, path)


def run_topology_batch(
    jobs: Sequence[TopologyJob],
    *,
    root: Path,
    config: TopologyConfig,
    workers: int,
    overwrite: bool,
    segment_manifest: Path,
    filtration_manifest: Path,
    sensitivity_filtration_manifest: Path,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise TopologyBatchError("workers must be positive")
    config_sha256 = _config_hash(config)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_job,
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
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                failures = sum(row["status"] == "failed" for row in rows)
                print(
                    f"path homology: {completed}/{len(futures)} complete; failures={failures}",
                    flush=True,
                )
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("group", ""),
            row.get("track_id", ""),
            float(row.get("scale_seconds", 0)),
            row.get("view", ""),
        ),
    )
    _write_csv(segment_manifest, ordered, SEGMENT_COLUMNS)
    filtration = [item for row in ordered for item in row.get("filtration", [])]
    _write_csv(filtration_manifest, filtration, FILTRATION_COLUMNS)
    sensitivity_filtration = [
        item for row in ordered for item in row.get("sensitivity_filtration", [])
    ]
    _write_csv(
        sensitivity_filtration_manifest,
        sensitivity_filtration,
        FILTRATION_COLUMNS,
    )
    return ordered


def _summary(
    rows: Sequence[dict[str, Any]],
    *,
    root: Path,
    config: TopologyConfig,
    segment_manifest: Path,
    filtration_manifest: Path,
    sensitivity_filtration_manifest: Path,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") != "failed"]
    output_paths = [
        root / Path(row[column])
        for row in successful
        for column in (
            "graph_relative_path",
            "persistence_relative_path",
            "sensitivity_persistence_relative_path",
        )
    ]
    output_bytes = sum(path.stat().st_size for path in output_paths if path.is_file())
    return {
        "generated_at": date.today().isoformat(),
        "ok": len(successful) == len(rows),
        "segment_views": len(rows),
        "segments": len({row["segment_id"] for row in rows}),
        "tracks": len({row["track_id"] for row in rows}),
        "status_counts": dict(Counter(row.get("status", "") for row in rows)),
        "view_counts": dict(Counter(row["view"] for row in successful)),
        "group_counts": dict(Counter(row["group"] for row in successful)),
        "split_counts": dict(Counter(row["split"] for row in successful)),
        "scale_counts": dict(
            Counter(f"{_seconds_token(float(row['scale_seconds']))}s" for row in successful)
        ),
        "h1_nonzero_segment_views": sum(float(row["h1_betti_max"]) > 0 for row in successful),
        "output_files": len(output_paths),
        "output_bytes": output_bytes,
        "output_gib": round(output_bytes / (1024**3), 3),
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "segment_manifest": _relative(root, segment_manifest),
        "segment_manifest_sha256": _sha256(segment_manifest),
        "filtration_manifest": _relative(root, filtration_manifest),
        "filtration_manifest_sha256": _sha256(filtration_manifest),
        "sensitivity_filtration_manifest": _relative(root, sensitivity_filtration_manifest),
        "sensitivity_filtration_manifest_sha256": _sha256(sensitivity_filtration_manifest),
    }


def _parse_set(raw: str | None) -> set[str] | None:
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
        raise TopologyBatchError("scales must be comma-separated seconds") from exc
    return values or None


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--groups", default="focus,classical")
    parser.add_argument("--scales", default="180,300")
    parser.add_argument("--track-ids")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--feature-manifest", type=Path, default=Path("metadata/feature_segments.csv")
    )
    parser.add_argument(
        "--topology-manifest", type=Path, default=Path("metadata/topology_segments.csv")
    )
    parser.add_argument(
        "--filtration-manifest", type=Path, default=Path("metadata/topology_filtration.csv")
    )
    parser.add_argument(
        "--sensitivity-filtration-manifest",
        type=Path,
        default=Path("metadata/topology_filtration_sensitivity.csv"),
    )
    parser.add_argument("--summary", type=Path, default=Path("metadata/topology_summary.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-path-analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("model", "statistics", "run"):
        child = subparsers.add_parser(command)
        _add_common_arguments(child)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = load_topology_config(root)
    feature_manifest = _resolve(root, args.feature_manifest)
    topology_manifest = _resolve(root, args.topology_manifest)
    filtration_manifest = _resolve(root, args.filtration_manifest)
    sensitivity_filtration_manifest = _resolve(root, args.sensitivity_filtration_manifest)
    summary_path = _resolve(root, args.summary)
    jobs = _filter_jobs(
        _load_feature_jobs(feature_manifest, config),
        groups=_parse_set(args.groups),
        scales=_parse_scales(args.scales),
        track_ids=_parse_set(args.track_ids),
    )
    if not jobs:
        raise TopologyBatchError("no feature segments matched the requested filters")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "segment_views": len(jobs),
                    "segments": len({job.segment_id for job in jobs}),
                    "tracks": len({job.track_id for job in jobs}),
                    "groups": dict(Counter(job.group for job in jobs)),
                    "splits": dict(Counter(job.split for job in jobs)),
                    "scales": dict(Counter(job.scale_seconds for job in jobs)),
                    "views": dict(Counter(job.view for job in jobs)),
                    "config_sha256": _config_hash(config),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command in {"model", "run"}:
        rows = run_topology_batch(
            jobs,
            root=root,
            config=config,
            workers=args.workers,
            overwrite=args.overwrite,
            segment_manifest=topology_manifest,
            filtration_manifest=filtration_manifest,
            sensitivity_filtration_manifest=sensitivity_filtration_manifest,
        )
        payload = _summary(
            rows,
            root=root,
            config=config,
            segment_manifest=topology_manifest,
            filtration_manifest=filtration_manifest,
            sensitivity_filtration_manifest=sensitivity_filtration_manifest,
        )
        _write_json_atomic(summary_path, payload)
        if not payload["ok"] or args.command == "model":
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload["ok"] else 1

    from .statistics import run_statistics

    statistics = run_statistics(root=root, topology_manifest=topology_manifest)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0 if statistics["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TopologyBatchError, ValueError) as exc:
        print(f"focus-path-analysis: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
