from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.batch import _json_hash, _sha256, _write_json_atomic, _write_npz_atomic
from graphs.transition import build_transition_graph
from homology.glmy import persistent_path_homology
from topology.batch import (
    FILTRATION_COLUMNS,
    SEGMENT_COLUMNS,
    TopologyJob,
    _filtration_rows,
    _graph_arrays,
    _graph_metrics,
    _persistence_arrays,
    _topology_metrics,
    _write_csv,
    load_topology_config,
)
from topology.statistics import _omnibus_and_pairwise

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("classical", "focus")
STATE_LABELS = ("Low", "Medium", "High")
RANDOM_SEED = 20_260_802
CONFIRMATORY_FDR_Q = 0.05
MAX_TRAINING_PER_GROUP = 50_000

MODEL_NPZ = ROOT / "features" / "models" / "modulation_tertile_model.npz"
MODEL_JSON = ROOT / "features" / "models" / "modulation_tertile_model.json"
FEATURE_MANIFEST = ROOT / "metadata" / "modulation_tertile_features.csv"
SEGMENT_MANIFEST = ROOT / "metadata" / "modulation_tertile_topology_segments.csv"
FILTRATION_MANIFEST = ROOT / "metadata" / "modulation_tertile_topology_filtration.csv"
SENSITIVITY_MANIFEST = ROOT / "metadata" / "modulation_tertile_topology_filtration_sensitivity.csv"
TESTS_MANIFEST = ROOT / "metadata" / "modulation_tertile_statistical_tests.csv"
PAIRWISE_MANIFEST = ROOT / "metadata" / "modulation_tertile_pairwise_tests.csv"
SUMMARY_JSON = ROOT / "metadata" / "modulation_tertile_summary.json"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_source_rows() -> list[dict[str, str]]:
    source = ROOT / "metadata" / "feature_segments.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") != "failed"]
    groups = Counter(row["group"] for row in rows)
    scales = Counter(float(row["scale_seconds"]) for row in rows)
    tracks = len({row["track_id"] for row in rows})
    expected_groups = {"classical": 600, "focus": 600}
    expected_scales = {180.0: 600, 300.0: 600}
    if (
        len(rows) != 1_200
        or tracks != 600
        or dict(groups) != expected_groups
        or dict(scales) != expected_scales
    ):
        raise RuntimeError(
            "canonical dataset audit failed: "
            f"rows={len(rows)}, tracks={tracks}, groups={dict(groups)}, scales={dict(scales)}"
        )
    if any(
        not row.get("modulation_relative_path") or not row.get("modulation_sha256") for row in rows
    ):
        raise RuntimeError("one or more source rows lack modulation features")
    return rows


def _intensity(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    key = np.asarray(arrays["key_band_energies"], dtype=np.float64)
    if key.ndim != 2 or key.shape[1] != 3:
        raise RuntimeError(f"expected key_band_energies shape (n, 3), got {key.shape}")
    values = np.sum(key, axis=1)
    valid = np.asarray(arrays["valid"], dtype=bool)
    valid &= np.all(np.isfinite(key), axis=1) & np.isfinite(values) & (values >= 0.0)
    return values, valid


def fit_model(rows: list[dict[str, str]]) -> dict[str, Any]:
    available: dict[str, list[np.ndarray]] = {group: [] for group in GROUPS}
    for row in rows:
        if row["split"] != "discovery" or float(row["scale_seconds"]) != 180.0:
            continue
        arrays = _read_npz(ROOT / row["modulation_relative_path"])
        values, valid = _intensity(arrays)
        available[row["group"]].append(values[valid])
    pooled = {group: np.concatenate(available[group]) for group in GROUPS}
    balanced_count = min(MAX_TRAINING_PER_GROUP, *(values.size for values in pooled.values()))
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = {
        group: pooled[group][rng.choice(pooled[group].size, balanced_count, replace=False)]
        for group in GROUPS
    }
    training = np.concatenate([sampled[group] for group in GROUPS])
    edges = np.quantile(training, [1.0 / 3.0, 2.0 / 3.0]).astype(np.float64)
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f"non-distinct tertile edges: {edges.tolist()}")
    group_quantiles = np.asarray(
        [np.quantile(sampled[group], [0.1, 0.25, 0.5, 0.75, 0.9]) for group in GROUPS],
        dtype=np.float64,
    )
    occupancy = np.bincount(np.digitize(training, edges), minlength=3)
    _write_npz_atomic(
        MODEL_NPZ,
        {
            "tertile_edges": edges,
            "training_counts": np.asarray([balanced_count, balanced_count], dtype=np.int64),
            "training_group_quantiles": group_quantiles,
            "training_state_counts": occupancy.astype(np.int64),
        },
    )
    model_sha256 = _sha256(MODEL_NPZ)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "view": "modulation_tertile",
        "measure": "sum of the three salient-band normalized spectral modulation energy shares",
        "interpretation": (
            "salient-band relative spectral modulation intensity; not absolute modulation power"
        ),
        "source_smp": (
            "mel-band energy envelopes; 4 s window, 2 s hop; normalized 0.5-45 Hz spectrum"
        ),
        "salient_bands_hz": [[8.0, 12.0], [18.0, 20.0], [28.0, 32.0]],
        "fit_scope": "discovery/180s only; equal numbers of valid windows from Classical and Focus",
        "random_seed": RANDOM_SEED,
        "training_groups": list(GROUPS),
        "available_windows": {group: int(pooled[group].size) for group in GROUPS},
        "sampled_windows": {group: int(balanced_count) for group in GROUPS},
        "tertile_edges": edges.tolist(),
        "state_labels": list(STATE_LABELS),
        "state_rule": "Low: x < q1; Medium: q1 <= x < q2; High: x >= q2",
        "training_state_counts": occupancy.astype(int).tolist(),
        "model_relative_path": _relative(MODEL_NPZ),
        "model_sha256": model_sha256,
        "generated_at": date.today().isoformat(),
    }
    _write_json_atomic(MODEL_JSON, payload)
    return payload


def _transform_row(row: dict[str, str], edges: np.ndarray, model_sha256: str) -> dict[str, Any]:
    source = ROOT / row["modulation_relative_path"]
    arrays = _read_npz(source)
    intensity, valid = _intensity(arrays)
    states = np.full(intensity.size, -1, dtype=np.int8)
    states[valid] = np.digitize(intensity[valid], edges).astype(np.int8)
    scale = int(float(row["scale_seconds"]))
    output = (
        ROOT
        / "features"
        / "modulation_tertile"
        / f"{scale}s"
        / row["group"]
        / row["split"]
        / f"{row['segment_id']}.npz"
    )
    _write_npz_atomic(
        output,
        {
            "times": np.asarray(arrays["times"], dtype=np.float32),
            "intensity": intensity.astype(np.float32),
            "states": states,
            "valid": valid,
            "tertile_edges": edges.astype(np.float32),
        },
    )
    valid_states = states[states >= 0]
    counts = np.bincount(valid_states, minlength=3)
    return {
        "segment_id": row["segment_id"],
        "track_id": row["track_id"],
        "group": row["group"],
        "split": row["split"],
        "scale_seconds": float(row["scale_seconds"]),
        "source_modulation_relative_path": row["modulation_relative_path"],
        "source_modulation_sha256": row["modulation_sha256"],
        "modulation_tertile_relative_path": _relative(output),
        "modulation_tertile_sha256": _sha256(output),
        "model_sha256": model_sha256,
        "windows": int(states.size),
        "valid_windows": int(valid.sum()),
        "masked_windows": int((~valid).sum()),
        "low_windows": int(counts[0]),
        "medium_windows": int(counts[1]),
        "high_windows": int(counts[2]),
        "observed_states": int(np.unique(valid_states).size),
        "median_intensity": float(np.median(intensity[valid])) if np.any(valid) else 0.0,
        "status": "success",
        "processed_at": date.today().isoformat(),
        "error": "",
    }


def transform_features(rows: list[dict[str, str]], model: dict[str, Any]) -> list[dict[str, Any]]:
    edges = np.asarray(model["tertile_edges"], dtype=np.float64)
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_transform_row, row, edges, str(model["model_sha256"])): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"modulation_tertile features: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs, key=lambda item: (item["group"], item["track_id"], item["scale_seconds"])
    )
    pd.DataFrame(ordered).to_csv(
        FEATURE_MANIFEST, index=False, encoding="utf-8", lineterminator="\n"
    )
    return ordered


def _topology_paths(row: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    scale = int(float(row["scale_seconds"]))
    suffix = (
        Path("modulation_tertile")
        / f"{scale}s"
        / row["group"]
        / row["split"]
        / f"{row['segment_id']}.npz"
    )
    return (
        ROOT / "graphs" / suffix,
        ROOT / "homology" / "persistence" / suffix,
        ROOT / "homology" / "persistence_sensitivity" / suffix,
        ROOT / "homology" / "descriptors" / suffix.with_suffix(".json"),
    )


def _process_topology_row(row: dict[str, Any], config: Any, config_sha256: str) -> dict[str, Any]:
    arrays = _read_npz(ROOT / row["modulation_tertile_relative_path"])
    raw = np.asarray(arrays["states"], dtype=int)
    states = [int(value) if value >= 0 else None for value in raw]
    graph = build_transition_graph(
        states, normalize=True, top_k=config.top_k, include_self_loops=config.include_self_loops
    )
    primary = persistent_path_homology(graph, config.thresholds, tolerance=config.rank_tolerance)
    sensitivity = persistent_path_homology(
        graph, config.sensitivity_thresholds, tolerance=config.rank_tolerance
    )
    graph_path, primary_path, sensitivity_path, sidecar_path = _topology_paths(row)
    _write_npz_atomic(graph_path, _graph_arrays(graph))
    _write_npz_atomic(primary_path, _persistence_arrays(primary))
    _write_npz_atomic(sensitivity_path, _persistence_arrays(sensitivity))
    identity = {
        "segment_id": row["segment_id"],
        "track_id": row["track_id"],
        "group": row["group"],
        "split": row["split"],
        "scale_seconds": row["scale_seconds"],
        "view": "modulation_tertile",
    }
    job = TopologyJob(
        **identity,
        feature_relative_path=row["modulation_tertile_relative_path"],
        feature_sha256=row["modulation_tertile_sha256"],
    )
    graph_metrics = _graph_metrics(states, graph)
    topology_metrics = _topology_metrics(primary)
    result = {
        **identity,
        "feature_relative_path": row["modulation_tertile_relative_path"],
        "feature_sha256": row["modulation_tertile_sha256"],
        "graph_relative_path": _relative(graph_path),
        "graph_sha256": _sha256(graph_path),
        "persistence_relative_path": _relative(primary_path),
        "persistence_sha256": _sha256(primary_path),
        "sensitivity_persistence_relative_path": _relative(sensitivity_path),
        "sensitivity_persistence_sha256": _sha256(sensitivity_path),
        "sidecar_relative_path": _relative(sidecar_path),
        "config_sha256": config_sha256,
        **graph_metrics,
        **topology_metrics,
        "status": "success",
        "processed_at": date.today().isoformat(),
        "error": "",
        "filtration": _filtration_rows(job, primary),
        "sensitivity_filtration": _filtration_rows(job, sensitivity),
    }
    _write_json_atomic(
        sidecar_path,
        {
            "schema_version": 1,
            "identity": identity,
            "feature_relative_path": result["feature_relative_path"],
            "feature_sha256": result["feature_sha256"],
            "config_sha256": config_sha256,
            "graph_metrics": graph_metrics,
            "topology_metrics": topology_metrics,
            "outputs": {
                "graph": result["graph_relative_path"],
                "persistence": result["persistence_relative_path"],
                "sensitivity_persistence": result["sensitivity_persistence_relative_path"],
            },
            "processed_at": result["processed_at"],
        },
    )
    return result


def run_topology(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    config = load_topology_config(ROOT)
    config_sha256 = _json_hash(
        {
            "view": "modulation_tertile",
            "model_sha256": model["model_sha256"],
            "state_count": 3,
            "invalid_policy": "missing state; adjacent transitions across missing windows excluded",
            "topology": asdict(config),
        }
    )
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_process_topology_row, row, config, config_sha256): row for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(f"modulation_tertile path homology: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs, key=lambda item: (item["group"], item["track_id"], item["scale_seconds"])
    )
    _write_csv(SEGMENT_MANIFEST, ordered, SEGMENT_COLUMNS)
    _write_csv(
        FILTRATION_MANIFEST,
        [item for row in ordered for item in row["filtration"]],
        FILTRATION_COLUMNS,
    )
    _write_csv(
        SENSITIVITY_MANIFEST,
        [item for row in ordered for item in row["sensitivity_filtration"]],
        FILTRATION_COLUMNS,
    )
    return ordered


def _mechanism_example(topology: pd.DataFrame) -> dict[str, Any]:
    subset = topology[
        (topology["split"] == "validation") & np.isclose(topology["scale_seconds"], 180.0)
    ]
    candidates: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        arrays = _read_npz(ROOT / str(row.sensitivity_persistence_relative_path))
        finite = np.flatnonzero(
            (arrays["interval_dimension"].astype(int) == 1)
            & ~arrays["interval_censored"].astype(bool)
        )
        if finite.size:
            best = int(finite[np.argmax(arrays["interval_lifetime"][finite])])
            candidates.append(
                {
                    "segment_id": str(row.segment_id),
                    "track_id": str(row.track_id),
                    "group": str(row.group),
                    "split": "validation",
                    "scale_seconds": 180,
                    "finite_h1_available": True,
                    "finite_h1_intervals": int(finite.size),
                    "birth_threshold": float(arrays["interval_birth_threshold"][best]),
                    "death_threshold": float(arrays["interval_death_threshold"][best]),
                    "lifetime": float(arrays["interval_lifetime"][best]),
                }
            )
    if candidates:
        chosen = sorted(
            candidates,
            key=lambda item: (
                item["group"] != "focus",
                item["finite_h1_intervals"] != 1,
                -item["lifetime"],
                item["segment_id"],
            ),
        )[0]
        chosen["selection_rule"] = (
            "prefer Focus validation/180s with one finite sensitivity H1 interval, "
            "then largest lifetime"
        )
        return chosen
    focus = (
        subset[subset["group"] == "focus"]
        .sort_values(["edge_count", "path_entropy", "segment_id"], ascending=[False, False, True])
        .iloc[0]
    )
    return {
        "segment_id": str(focus.segment_id),
        "track_id": str(focus.track_id),
        "group": "focus",
        "split": "validation",
        "scale_seconds": 180,
        "finite_h1_available": False,
        "finite_h1_intervals": 0,
        "birth_threshold": None,
        "death_threshold": None,
        "lifetime": 0.0,
        "selection_rule": (
            "no finite H1 interval exists; choose Focus validation/180s with most "
            "edges, then path entropy"
        ),
    }


def run_statistics(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    topology = pd.DataFrame(rows).drop(columns=["filtration", "sensitivity_filtration"])
    omnibus, pairwise = _omnibus_and_pairwise(topology)
    omnibus.to_csv(TESTS_MANIFEST, index=False, encoding="utf-8", lineterminator="\n")
    pairwise.to_csv(PAIRWISE_MANIFEST, index=False, encoding="utf-8", lineterminator="\n")
    primary = omnibus[omnibus.analysis_set == "primary_validation_180"]
    sensitivity_tests = omnibus[omnibus.analysis_set == "sensitivity_validation_300"].set_index(
        "metric"
    )
    replicated = 0
    for row in primary[
        primary.p_fdr_bh <= CONFIRMATORY_FDR_Q
    ].itertuples(index=False):
        other = sensitivity_tests.loc[row.metric]
        if float(other.p_fdr_bh) <= CONFIRMATORY_FDR_Q and np.sign(
            row.focus_median - row.classical_median
        ) == np.sign(other.focus_median - other.classical_median):
            replicated += 1
    sensitivity = pd.read_csv(SENSITIVITY_MANIFEST)
    sens_max = sensitivity.groupby(
        ["segment_id", "group", "split", "scale_seconds"], as_index=False
    )["h1_betti"].max()
    validation_sens = sens_max[
        (sens_max.split == "validation") & np.isclose(sens_max.scale_seconds, 180.0)
    ]
    validation = topology[
        (topology.split == "validation") & np.isclose(topology.scale_seconds, 180.0)
    ]
    h1_counts = {
        group: {
            "total": int(len(frame)),
            "primary_nonzero": int((frame.h1_betti_max > 0).sum()),
            "sensitivity_nonzero": int(
                (validation_sens.loc[validation_sens.group == group, "h1_betti"] > 0).sum()
            ),
        }
        for group, frame in validation.groupby("group")
    }
    features = pd.read_csv(FEATURE_MANIFEST)
    occupancy: dict[str, dict[str, float | int]] = {}
    for group, frame in features.groupby("group"):
        counts = frame[["low_windows", "medium_windows", "high_windows"]].sum().to_numpy(int)
        occupancy[str(group)] = {
            "low": int(counts[0]),
            "medium": int(counts[1]),
            "high": int(counts[2]),
            "low_share": float(counts[0] / counts.sum()),
            "medium_share": float(counts[1] / counts.sum()),
            "high_share": float(counts[2] / counts.sum()),
        }
    summary: dict[str, Any] = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "scope": "three-state spectral-modulation Path Homology on Focus/Classical dataset",
        "canonical_focus_source": "Jamendo Open Focus",
        "canonical_groups": list(GROUPS),
        "segment_views": int(len(topology)),
        "tracks": int(topology.track_id.nunique()),
        "status_counts": dict(Counter(topology.status)),
        "state_count": 3,
        "state_labels": list(STATE_LABELS),
        "tertile_edges": model["tertile_edges"],
        "model_sha256": model["model_sha256"],
        "graph_input": "adjacent frozen Low/Medium/High state transitions",
        "ssm_used": False,
        "primary_validation_n": int(len(validation)),
        "primary_tests": int(len(primary)),
        "confirmatory_fdr_q": CONFIRMATORY_FDR_Q,
        "primary_fdr_discoveries": int(
            (primary.p_fdr_bh <= CONFIRMATORY_FDR_Q).sum()
        ),
        "sensitivity_fdr_discoveries": int(
            (sensitivity_tests.p_fdr_bh <= CONFIRMATORY_FDR_Q).sum()
        ),
        "replicated_same_direction": int(replicated),
        "validation_180_h1_counts": h1_counts,
        "all_segment_state_occupancy": occupancy,
        "mechanism_example": _mechanism_example(topology),
        "artifacts": {
            "feature_manifest": _relative(FEATURE_MANIFEST),
            "segment_manifest": _relative(SEGMENT_MANIFEST),
            "filtration_manifest": _relative(FILTRATION_MANIFEST),
            "sensitivity_filtration_manifest": _relative(SENSITIVITY_MANIFEST),
            "statistical_tests": _relative(TESTS_MANIFEST),
            "pairwise_tests": _relative(PAIRWISE_MANIFEST),
        },
    }
    summary["artifact_sha256"] = {
        _relative(path): _sha256(path)
        for path in (
            MODEL_NPZ,
            MODEL_JSON,
            FEATURE_MANIFEST,
            SEGMENT_MANIFEST,
            FILTRATION_MANIFEST,
            SENSITIVITY_MANIFEST,
            TESTS_MANIFEST,
            PAIRWISE_MANIFEST,
        )
    }
    _write_json_atomic(SUMMARY_JSON, summary)
    return summary


def main() -> int:
    rows = _load_source_rows()
    print("modulation_tertile: fitting balanced discovery/180s tertiles", flush=True)
    model = fit_model(rows)
    print("modulation_tertile: transforming all segments", flush=True)
    features = transform_features(rows, model)
    print("modulation_tertile: running persistent path homology", flush=True)
    topology = run_topology(features, model)
    print("modulation_tertile: running two-group statistics", flush=True)
    summary = run_statistics(topology, model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
