# ruff: noqa: E501
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from data.analysis_inputs import audit_analysis_inputs
from features.batch import _json_hash, _sha256, _write_json_atomic, _write_npz_atomic
from features.pitch_v2 import assign_codebook, chroma_to_tonnetz, normalize_chroma
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
RANDOM_SEED = 20_260_716
CONFIRMATORY_FDR_Q = 0.05
V_PITCH = 16
MAX_TRAINING_PER_GROUP = 50_000
SENSITIVITY_K = (8, 12, 16, 24)
GROUPS = ("classical", "focus")
PITCH_NAMES = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")

CODEBOOK_NPZ = ROOT / "features" / "models" / "pitch_v2_codebook.npz"
CODEBOOK_JSON = ROOT / "features" / "models" / "pitch_v2_codebook.json"
DIAGNOSTICS_CSV = ROOT / "metadata" / "pitch_v2_codebook_diagnostics.csv"
FEATURE_MANIFEST = ROOT / "metadata" / "pitch_v2_features.csv"
SEGMENT_MANIFEST = ROOT / "metadata" / "pitch_v2_topology_segments.csv"
FILTRATION_MANIFEST = ROOT / "metadata" / "pitch_v2_topology_filtration.csv"
SENSITIVITY_MANIFEST = ROOT / "metadata" / "pitch_v2_topology_filtration_sensitivity.csv"
TESTS_MANIFEST = ROOT / "metadata" / "pitch_v2_statistical_tests.csv"
PAIRWISE_MANIFEST = ROOT / "metadata" / "pitch_v2_pairwise_tests.csv"
SUMMARY_JSON = ROOT / "metadata" / "pitch_v2_summary.json"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_feature_rows() -> list[dict[str, str]]:
    path = ROOT / "metadata" / "feature_segments.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") != "failed"]
    expected_groups = {group: 600 for group in GROUPS}
    expected_scales = {180.0: 600, 300.0: 600}
    group_counts = Counter(row["group"] for row in rows)
    scale_counts = Counter(float(row["scale_seconds"]) for row in rows)
    track_count = len({row["track_id"] for row in rows})
    if (
        len(rows) != 1_200
        or track_count != 600
        or dict(group_counts) != expected_groups
        or dict(scale_counts) != expected_scales
    ):
        raise RuntimeError(
            "canonical open-dataset audit failed: "
            f"rows={len(rows)}, tracks={track_count}, "
            f"groups={dict(group_counts)}, scales={dict(scale_counts)}"
        )
    return rows


def _training_data(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_group: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        group: [] for group in GROUPS
    }
    for row in rows:
        if row["split"] != "discovery" or float(row["scale_seconds"]) != 180.0:
            continue
        arrays = _read_npz(ROOT / row["chroma_relative_path"])
        chroma = np.asarray(arrays["chroma"], dtype=np.float64)
        valid = np.asarray(arrays["valid"], dtype=bool)
        normalized = normalize_chroma(chroma)
        tonnetz = chroma_to_tonnetz(chroma)
        keep = valid & np.all(np.isfinite(tonnetz), axis=1) & (np.sum(chroma, axis=1) > 1e-8)
        by_group[row["group"]].append((tonnetz[keep], normalized[keep]))

    available = {
        group: (
            np.concatenate([item[0] for item in by_group[group]], axis=0),
            np.concatenate([item[1] for item in by_group[group]], axis=0),
        )
        for group in GROUPS
    }
    balanced_count = min(
        MAX_TRAINING_PER_GROUP,
        *(values[0].shape[0] for values in available.values()),
    )
    rng = np.random.default_rng(RANDOM_SEED)
    tonnetz_samples: list[np.ndarray] = []
    chroma_samples: list[np.ndarray] = []
    group_labels: list[np.ndarray] = []
    for group_index, group in enumerate(GROUPS):
        tonnetz, chroma = available[group]
        selected = rng.choice(tonnetz.shape[0], size=balanced_count, replace=False)
        tonnetz_samples.append(tonnetz[selected])
        chroma_samples.append(chroma[selected])
        group_labels.append(np.full(balanced_count, group_index, dtype=np.int8))
    return (
        np.concatenate(tonnetz_samples, axis=0),
        np.concatenate(chroma_samples, axis=0),
        np.concatenate(group_labels, axis=0),
    )


def _fit_one_k(values: np.ndarray, k: int, seed: int, *, n_init: int) -> MiniBatchKMeans:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4_096,
        n_init=n_init,
        max_iter=300,
        reassignment_ratio=0.01,
    )
    return model.fit(values)


def _codebook_diagnostics(values: np.ndarray) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    selection_size = min(values.shape[0], 60_000)
    selection = rng.choice(values.shape[0], selection_size, replace=False)
    fit_values = values[selection]
    evaluation_size = min(values.shape[0], 3_000)
    evaluation = values[rng.choice(values.shape[0], evaluation_size, replace=False)]
    rows: list[dict[str, float | int]] = []
    for k in SENSITIVITY_K:
        first = _fit_one_k(fit_values, k, RANDOM_SEED + k, n_init=3)
        second = _fit_one_k(fit_values, k, RANDOM_SEED + 1_000 + k, n_init=3)
        labels = first.predict(evaluation)
        counts = np.bincount(first.labels_, minlength=k)
        rows.append(
            {
                "v_pitch": k,
                "silhouette": float(silhouette_score(evaluation, labels)),
                "seed_stability_ari": float(
                    adjusted_rand_score(labels, second.predict(evaluation))
                ),
                "min_cluster_share": float(np.min(counts) / np.sum(counts)),
                "max_cluster_share": float(np.max(counts) / np.sum(counts)),
                "inertia_per_step": float(first.inertia_ / fit_values.shape[0]),
            }
        )
    return rows


def fit_codebook(rows: list[dict[str, str]]) -> dict[str, Any]:
    tonnetz, chroma, groups = _training_data(rows)
    diagnostic_rows = _codebook_diagnostics(tonnetz)
    pd.DataFrame(diagnostic_rows).to_csv(DIAGNOSTICS_CSV, index=False, encoding="utf-8")

    model = _fit_one_k(tonnetz, V_PITCH, RANDOM_SEED, n_init=10)
    labels = model.labels_
    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    prototypes = np.zeros((V_PITCH, 12), dtype=np.float64)
    group_counts = np.zeros((V_PITCH, len(GROUPS)), dtype=np.int64)
    counts = np.bincount(labels, minlength=V_PITCH)
    for state in range(V_PITCH):
        mask = labels == state
        prototypes[state] = np.mean(chroma[mask], axis=0)
        group_counts[state] = np.bincount(groups[mask], minlength=len(GROUPS))

    dominant = np.argmax(prototypes, axis=1)
    second = np.argsort(prototypes, axis=1)[:, -2]
    order = np.asarray(
        sorted(
            range(V_PITCH),
            key=lambda state: (
                int(dominant[state]),
                int(second[state]),
                *np.round(centers[state], 8).tolist(),
            ),
        ),
        dtype=int,
    )
    centers = centers[order]
    prototypes = prototypes[order]
    counts = counts[order]
    group_counts = group_counts[order]
    top_three = np.argsort(prototypes, axis=1)[:, -3:][:, ::-1]
    labels_text = [
        f"S{state:02d} ({'-'.join(PITCH_NAMES[index] for index in top_three[state])})"
        for state in range(V_PITCH)
    ]

    CODEBOOK_NPZ.parent.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(
        CODEBOOK_NPZ,
        {
            "centers": centers,
            "chroma_prototypes": prototypes,
            "training_counts": counts.astype(np.int64),
            "training_group_counts": group_counts.astype(np.int64),
            "top_pitch_classes": top_three.astype(np.int8),
        },
    )
    codebook_sha256 = _sha256(CODEBOOK_NPZ)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "beat-synchronous chroma -> fixed Harte/librosa Tonnetz -> MiniBatchKMeans",
        "fit_scope": "discovery/180s only; group-balanced sampling",
        "v_pitch": V_PITCH,
        "uncertain_policy": "existing pitch-valid mask; invalid beats become missing state -1",
        "random_seed": RANDOM_SEED,
        "training_steps": int(tonnetz.shape[0]),
        "training_groups": list(GROUPS),
        "training_steps_per_group": {
            group: int(np.count_nonzero(groups == index))
            for index, group in enumerate(GROUPS)
        },
        "state_labels": labels_text,
        "diagnostics": diagnostic_rows,
        "codebook_relative_path": _relative(CODEBOOK_NPZ),
        "codebook_sha256": codebook_sha256,
        "generated_at": date.today().isoformat(),
    }
    _write_json_atomic(CODEBOOK_JSON, payload)
    return payload


def _transform_row(row: dict[str, str], centers: np.ndarray, codebook_sha256: str) -> dict[str, Any]:
    source = ROOT / row["chroma_relative_path"]
    arrays = _read_npz(source)
    chroma = np.asarray(arrays["chroma"], dtype=np.float64)
    tonnetz = chroma_to_tonnetz(chroma)
    valid = np.asarray(arrays["valid"], dtype=bool)
    valid &= np.all(np.isfinite(tonnetz), axis=1) & (np.sum(chroma, axis=1) > 1e-8)
    states = assign_codebook(tonnetz, centers, valid=valid)
    distances = np.full(states.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        distances[valid] = np.sqrt(
            np.sum((tonnetz[valid] - centers[states[valid]]) ** 2, axis=1)
        ).astype(np.float32)

    scale = int(float(row["scale_seconds"]))
    output = (
        ROOT
        / "features"
        / "pitch_v2"
        / f"{scale}s"
        / row["group"]
        / row["split"]
        / f"{row['segment_id']}.npz"
    )
    _write_npz_atomic(
        output,
        {
            "times": np.asarray(arrays["times"], dtype=np.float32),
            "tonnetz": tonnetz.astype(np.float32),
            "states": states,
            "valid": valid,
            "codebook_distance": distances,
        },
    )
    return {
        "segment_id": row["segment_id"],
        "track_id": row["track_id"],
        "group": row["group"],
        "split": row["split"],
        "scale_seconds": float(row["scale_seconds"]),
        "source_chroma_relative_path": row["chroma_relative_path"],
        "source_chroma_sha256": row["chroma_sha256"],
        "pitch_v2_relative_path": _relative(output),
        "pitch_v2_sha256": _sha256(output),
        "codebook_sha256": codebook_sha256,
        "steps": int(states.size),
        "valid_steps": int(np.count_nonzero(valid)),
        "masked_steps": int(np.count_nonzero(~valid)),
        "observed_states": int(np.unique(states[states >= 0]).size),
        "median_codebook_distance": float(np.nanmedian(distances)) if np.any(valid) else 0.0,
        "status": "success",
        "processed_at": date.today().isoformat(),
        "error": "",
    }


def transform_features(rows: list[dict[str, str]], codebook: dict[str, Any]) -> list[dict[str, Any]]:
    centers = _read_npz(CODEBOOK_NPZ)["centers"].astype(np.float64)
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_transform_row, row, centers, str(codebook["codebook_sha256"])): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"pitch_v2 features: {completed}/{len(futures)}", flush=True)
    ordered = sorted(outputs, key=lambda item: (item["group"], item["track_id"], item["scale_seconds"]))
    pd.DataFrame(ordered).to_csv(FEATURE_MANIFEST, index=False, encoding="utf-8")
    return ordered


def _topology_paths(feature_row: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    scale = int(float(feature_row["scale_seconds"]))
    suffix = (
        Path("pitch_v2")
        / f"{scale}s"
        / feature_row["group"]
        / feature_row["split"]
        / f"{feature_row['segment_id']}.npz"
    )
    return (
        ROOT / "graphs" / suffix,
        ROOT / "homology" / "persistence" / suffix,
        ROOT / "homology" / "persistence_sensitivity" / suffix,
        ROOT / "homology" / "descriptors" / suffix.with_suffix(".json"),
    )


def _process_topology_row(
    feature_row: dict[str, Any],
    config: Any,
    config_sha256: str,
) -> dict[str, Any]:
    arrays = _read_npz(ROOT / feature_row["pitch_v2_relative_path"])
    raw_states = np.asarray(arrays["states"], dtype=int)
    states = [int(value) if value >= 0 else None for value in raw_states]
    graph = build_transition_graph(
        states,
        normalize=True,
        top_k=config.top_k,
        include_self_loops=config.include_self_loops,
    )
    primary = persistent_path_homology(graph, config.thresholds, tolerance=config.rank_tolerance)
    sensitivity = persistent_path_homology(
        graph,
        config.sensitivity_thresholds,
        tolerance=config.rank_tolerance,
    )
    graph_path, primary_path, sensitivity_path, sidecar_path = _topology_paths(feature_row)
    _write_npz_atomic(graph_path, _graph_arrays(graph))
    _write_npz_atomic(primary_path, _persistence_arrays(primary))
    _write_npz_atomic(sensitivity_path, _persistence_arrays(sensitivity))

    identity = {
        "segment_id": feature_row["segment_id"],
        "track_id": feature_row["track_id"],
        "group": feature_row["group"],
        "split": feature_row["split"],
        "scale_seconds": feature_row["scale_seconds"],
        "view": "pitch_v2",
    }
    job = TopologyJob(
        **{key: identity[key] for key in ("segment_id", "track_id", "group", "split", "scale_seconds", "view")},
        feature_relative_path=feature_row["pitch_v2_relative_path"],
        feature_sha256=feature_row["pitch_v2_sha256"],
    )
    graph_metrics = _graph_metrics(states, graph)
    topology_metrics = _topology_metrics(primary)
    row = {
        **identity,
        "feature_relative_path": feature_row["pitch_v2_relative_path"],
        "feature_sha256": feature_row["pitch_v2_sha256"],
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
            "feature_relative_path": row["feature_relative_path"],
            "feature_sha256": row["feature_sha256"],
            "config_sha256": config_sha256,
            "graph_metrics": graph_metrics,
            "topology_metrics": topology_metrics,
            "outputs": {
                "graph": row["graph_relative_path"],
                "persistence": row["persistence_relative_path"],
                "sensitivity_persistence": row["sensitivity_persistence_relative_path"],
            },
            "processed_at": row["processed_at"],
        },
    )
    return row


def run_topology(feature_rows: list[dict[str, Any]], codebook: dict[str, Any]) -> list[dict[str, Any]]:
    config = load_topology_config(ROOT)
    config_payload = {
        "view": "pitch_v2",
        "codebook_sha256": codebook["codebook_sha256"],
        "v_pitch": V_PITCH,
        "invalid_policy": "missing state; adjacent transitions across missing beats are excluded",
        "topology": asdict(config),
    }
    config_sha256 = _json_hash(config_payload)
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_process_topology_row, row, config, config_sha256): row
            for row in feature_rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(f"pitch_v2 path homology: {completed}/{len(futures)}", flush=True)

    ordered = sorted(outputs, key=lambda item: (item["group"], item["track_id"], item["scale_seconds"]))
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


def select_mechanism_example(topology_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in topology_rows:
        if not (
            row["group"] == "focus"
            and row["split"] == "validation"
            and float(row["scale_seconds"]) == 180.0
        ):
            continue
        arrays = _read_npz(ROOT / row["sensitivity_persistence_relative_path"])
        dimensions = arrays["interval_dimension"].astype(int)
        censored = arrays["interval_censored"].astype(bool)
        finite_h1 = np.flatnonzero((dimensions == 1) & ~censored)
        if finite_h1.size == 0:
            continue
        best = int(finite_h1[np.argmax(arrays["interval_lifetime"][finite_h1])])
        candidates.append(
            {
                "segment_id": row["segment_id"],
                "track_id": row["track_id"],
                "group": "focus",
                "split": "validation",
                "scale_seconds": 180,
                "finite_h1_intervals": int(finite_h1.size),
                "birth_threshold": float(arrays["interval_birth_threshold"][best]),
                "death_threshold": float(arrays["interval_death_threshold"][best]),
                "lifetime": float(arrays["interval_lifetime"][best]),
                "selection_rule": (
                    "prefer exactly one finite sensitivity-filtration H1 interval; then "
                    "largest lifetime; deterministic segment-id tie break"
                ),
            }
        )
    if not candidates:
        raise RuntimeError("no Focus validation/180s segment has a finite pitch_v2 H1 interval")
    return sorted(
        candidates,
        key=lambda item: (
            item["finite_h1_intervals"] != 1,
            -item["lifetime"],
            item["segment_id"],
        ),
    )[0]


def run_statistics(
    topology_rows: list[dict[str, Any]],
    codebook: dict[str, Any],
    example: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    topology = pd.DataFrame(topology_rows).drop(columns=["filtration", "sensitivity_filtration"])
    omnibus, pairwise = _omnibus_and_pairwise(
        topology,
        bootstrap_resamples=3000,
        bootstrap_seed=20260716,
    )
    omnibus.to_csv(TESTS_MANIFEST, index=False, encoding="utf-8")
    pairwise.to_csv(PAIRWISE_MANIFEST, index=False, encoding="utf-8")
    primary = omnibus[omnibus.analysis_set == "primary_validation_180"]
    focus_classical = pairwise[
        (pairwise.analysis_set == "primary_validation_180")
        & (pairwise.group_a == "classical")
        & (pairwise.group_b == "focus")
    ]
    sensitivity_tests = omnibus[omnibus.analysis_set == "sensitivity_validation_300"].set_index(
        "metric"
    )
    replicated = 0
    for row in primary[
        primary.p_fdr_bh <= CONFIRMATORY_FDR_Q
    ].itertuples(index=False):
        other = sensitivity_tests.loc[row.metric]
        primary_delta = float(row.focus_median - row.classical_median)
        sensitivity_delta = float(other.focus_median - other.classical_median)
        if (
            other.p_fdr_bh <= CONFIRMATORY_FDR_Q
            and primary_delta * sensitivity_delta > 0
        ):
            replicated += 1
    sensitivity = pd.read_csv(SENSITIVITY_MANIFEST)
    sensitivity_segments = (
        sensitivity.groupby(["segment_id", "group", "split", "scale_seconds"])["h1_betti"]
        .max()
        .reset_index()
    )
    validation_sensitivity = sensitivity_segments[
        (sensitivity_segments.split == "validation")
        & (sensitivity_segments.scale_seconds == 180.0)
    ]
    primary_rows = topology[
        (topology.split == "validation") & (topology.scale_seconds == 180.0)
    ]
    medians = (
        primary_rows.groupby("group")[[
            "vertex_count",
            "edge_count",
            "self_transition_ratio",
            "path_entropy",
            "directed_recurrence",
            "reciprocity",
            "h0_betti_mean",
            "h1_betti_max",
        ]]
        .median()
        .to_dict(orient="index")
    )
    h1_counts = {
        group: {
            "n": int(len(frame)),
            "primary_nonzero": int(np.count_nonzero(frame.h1_betti_max > 0)),
            "sensitivity_nonzero": int(
                np.count_nonzero(validation_sensitivity.loc[validation_sensitivity.group == group, "h1_betti"] > 0)
            ),
        }
        for group, frame in primary_rows.groupby("group")
    }
    summary = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "scope": "pitch Tonnetz harmonic-codebook path homology on Focus/Classical dataset",
        "canonical_focus_source": "Jamendo Open Focus",
        "canonical_groups": list(GROUPS),
        "input_provenance": input_audit,
        "segment_views": len(topology_rows),
        "tracks": int(topology.track_id.nunique()),
        "status_counts": dict(Counter(topology.status)),
        "v_pitch": V_PITCH,
        "codebook_sha256": codebook["codebook_sha256"],
        "graph_input": "adjacent frozen state transitions",
        "ssm_used": False,
        "primary_validation_n": int(len(primary_rows)),
        "confirmatory_fdr_q": CONFIRMATORY_FDR_Q,
        "primary_fdr_discoveries": int(
            np.count_nonzero(primary.p_fdr_bh <= CONFIRMATORY_FDR_Q)
        ),
        "primary_tests": int(len(primary)),
        "sensitivity_fdr_discoveries": int(
            np.count_nonzero(
                sensitivity_tests.p_fdr_bh <= CONFIRMATORY_FDR_Q
            )
        ),
        "replicated_same_direction": replicated,
        "focus_classical_fdr_discoveries": int(
            np.count_nonzero(focus_classical.p_fdr_bh <= CONFIRMATORY_FDR_Q)
        ),
        "validation_180_group_medians": medians,
        "validation_180_h1_counts": h1_counts,
        "full_primary_h1_nonzero": int(np.count_nonzero(topology.h1_betti_max > 0)),
        "full_sensitivity_h1_nonzero": int(np.count_nonzero(sensitivity_segments.h1_betti > 0)),
        "mechanism_example": example,
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
            CODEBOOK_NPZ,
            CODEBOOK_JSON,
            DIAGNOSTICS_CSV,
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
    input_audit = audit_analysis_inputs(root=ROOT)
    rows = _load_feature_rows()
    print("pitch_v2: fitting discovery-only Tonnetz codebook", flush=True)
    codebook = fit_codebook(rows)
    print("pitch_v2: transforming all feature segments", flush=True)
    feature_rows = transform_features(rows, codebook)
    print("pitch_v2: running persistent path homology", flush=True)
    topology_rows = run_topology(feature_rows, codebook)
    print("pitch_v2: selecting deterministic Focus mechanism example", flush=True)
    example = select_mechanism_example(topology_rows)
    print("pitch_v2: running two-group statistics", flush=True)
    summary = run_statistics(topology_rows, codebook, example, input_audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
