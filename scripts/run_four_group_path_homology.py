from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from features.batch import (
    MANIFEST_COLUMNS,
    FeatureStateModel,
    _load_config,
    _load_jobs,
    _read_npz,
    _sha256,
    _write_json_atomic,
    _write_npz_atomic,
    extract_batch,
    fit_state_model,
)
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
GROUPS = ("classical", "focus", "focus_open", "pop")
RANDOM_SEED = 20_260_716
V_PITCH = 16
MAX_PITCH_TRAINING_PER_GROUP = 50_000
PITCH_SENSITIVITY_K = (8, 12, 16, 24)
PITCH_NAMES = (
    "C",
    "C#/Db",
    "D",
    "D#/Eb",
    "E",
    "F",
    "F#/Gb",
    "G",
    "G#/Ab",
    "A",
    "A#/Bb",
    "B",
)

PREPROCESSED = ROOT / "metadata" / "four_group_preprocessed_segments.csv"
OPEN_RAW_FEATURES = ROOT / "metadata" / "four_group_focus_open_raw_features.csv"
RAW_FEATURES = ROOT / "metadata" / "four_group_raw_feature_segments.csv"
FEATURES = ROOT / "metadata" / "four_group_feature_segments.csv"
PITCH_FEATURES = ROOT / "metadata" / "four_group_pitch_v2_features.csv"
TOPOLOGY = ROOT / "metadata" / "four_group_topology_segments.csv"
FILTRATION = ROOT / "metadata" / "four_group_topology_filtration.csv"
SENSITIVITY = ROOT / "metadata" / "four_group_topology_filtration_sensitivity.csv"
OMNIBUS = ROOT / "metadata" / "four_group_statistical_tests.csv"
PAIRWISE = ROOT / "metadata" / "four_group_pairwise_tests.csv"
SUMMARY = ROOT / "metadata" / "four_group_path_homology_summary.json"
STATE_MODEL = ROOT / "features" / "models" / "four_group_state_model.npz"
STATE_MODEL_JSON = ROOT / "features" / "models" / "four_group_state_model.json"
PITCH_CODEBOOK = ROOT / "features" / "models" / "four_group_pitch_v2_codebook.npz"
PITCH_CODEBOOK_JSON = ROOT / "features" / "models" / "four_group_pitch_v2_codebook.json"
PITCH_DIAGNOSTICS = ROOT / "metadata" / "four_group_pitch_v2_codebook_diagnostics.csv"


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def build_four_group_manifest() -> dict[str, Any]:
    base = pd.read_csv(ROOT / "metadata" / "preprocessed_segments.csv")
    open_focus = pd.read_csv(ROOT / "metadata" / "focus_open_preprocessed_segments.csv")
    if set(base["group"].unique()) != {"classical", "focus", "pop"}:
        raise RuntimeError("base preprocessed manifest has an unexpected group family")
    open_focus = open_focus.copy()
    open_focus["group"] = "focus_open"
    columns = list(dict.fromkeys([*base.columns, *open_focus.columns]))
    combined = pd.concat(
        [base.reindex(columns=columns), open_focus.reindex(columns=columns)],
        ignore_index=True,
    )
    if combined["segment_id"].duplicated().any():
        duplicated = combined.loc[combined["segment_id"].duplicated(), "segment_id"].tolist()
        raise RuntimeError(f"duplicate four-group segment IDs: {duplicated[:5]}")
    if set(combined["group"].unique()) != set(GROUPS):
        raise RuntimeError("four-group preprocessed manifest is incomplete")
    _write_frame(PREPROCESSED, combined)
    counts = (
        combined.groupby(["group", "split", "scale_seconds"]).size().astype(int).to_dict()
    )
    return {
        "segments": int(len(combined)),
        "tracks": int(combined["track_id"].nunique()),
        "counts": {"|".join(map(str, key)): value for key, value in counts.items()},
        "manifest_sha256": _sha256(PREPROCESSED),
    }


def extract_open_focus(*, workers: int) -> list[dict[str, Any]]:
    config = _load_config(ROOT)
    jobs = [job for job in _load_jobs(PREPROCESSED) if job.group == "focus_open"]
    rows = extract_batch(
        jobs,
        root=ROOT,
        config=config,
        workers=workers,
        overwrite=False,
        manifest_path=OPEN_RAW_FEATURES,
    )
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"Focus Open feature extraction failed for {len(failed)} segments")
    return rows


def combine_raw_features(open_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    base = pd.read_csv(ROOT / "metadata" / "feature_segments.csv")
    open_frame = pd.DataFrame(open_rows)
    combined = pd.concat(
        [base.reindex(columns=MANIFEST_COLUMNS), open_frame.reindex(columns=MANIFEST_COLUMNS)],
        ignore_index=True,
    )
    if len(combined) != 2_200 or combined["segment_id"].duplicated().any():
        raise RuntimeError("expected 2,200 unique four-group feature rows")
    if set(combined["group"].unique()) != set(GROUPS):
        raise RuntimeError("raw feature manifest does not contain all four groups")
    _write_frame(RAW_FEATURES, combined)
    return _read_csv(RAW_FEATURES)


def _nearest_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(values * values, axis=1, keepdims=True)
        - 2.0 * values @ centers.T
        + np.sum(centers * centers, axis=1)[None, :]
    )
    return np.argmin(distances, axis=1).astype(np.int16)


def _transform_state_row(row: dict[str, str], model: FeatureStateModel) -> dict[str, Any]:
    rhythm = _read_npz(ROOT / row["rhythm_relative_path"])
    rhythm_values = np.asarray(rhythm["vectors"], dtype=np.float32)
    rhythm_valid = np.asarray(rhythm["valid"], dtype=bool)
    rhythm_filled = np.where(rhythm_valid, rhythm_values, model.rhythm_impute)
    rhythm_scaled = (rhythm_filled - model.rhythm_mean) / model.rhythm_scale
    rhythm["states"] = _nearest_centers(rhythm_scaled, model.rhythm_centers)

    structure = _read_npz(ROOT / row["structure_relative_path"])
    structure_values = np.asarray(structure["block_vectors"], dtype=np.float32)
    structure_valid = np.asarray(structure["valid"], dtype=bool)
    structure_scaled = (structure_values - model.acoustic_mean) / model.acoustic_scale
    structure_reduced = (structure_scaled - model.pca_mean) @ model.pca_components.T
    structure["states"] = _nearest_centers(structure_reduced, model.structure_centers)
    structure["states"][~structure_valid] = -1

    scale = int(float(row["scale_seconds"]))
    suffix = Path(f"{scale}s") / row["group"] / row["split"] / f"{row['segment_id']}.npz"
    rhythm_path = ROOT / "features" / "four_group" / "rhythm" / suffix
    structure_path = ROOT / "features" / "four_group" / "structure" / suffix
    _write_npz_atomic(rhythm_path, rhythm)
    _write_npz_atomic(structure_path, structure)
    output: dict[str, Any] = dict(row)
    output.update(
        {
            "rhythm_relative_path": _relative(rhythm_path),
            "rhythm_sha256": _sha256(rhythm_path),
            "structure_relative_path": _relative(structure_path),
            "structure_sha256": _sha256(structure_path),
            "model_sha256": _sha256(STATE_MODEL),
            "status": "transformed",
            "processed_at": date.today().isoformat(),
            "error": "",
        }
    )
    return output


def fit_and_transform_states(
    raw_rows: list[dict[str, str]], *, workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _load_config(ROOT)
    model, _, model_metadata = fit_state_model(
        raw_rows,
        root=ROOT,
        config=config,
        model_path=STATE_MODEL,
        metadata_path=STATE_MODEL_JSON,
        overwrite=True,
    )
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_transform_state_row, row, model): row for row in raw_rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"four-group states: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs,
        key=lambda row: (row["group"], row["track_id"], float(row["scale_seconds"])),
    )
    _write_frame(FEATURES, pd.DataFrame(ordered).reindex(columns=MANIFEST_COLUMNS))
    return ordered, model_metadata


def _fit_pitch_k(values: np.ndarray, k: int, seed: int, n_init: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4_096,
        n_init=n_init,
        max_iter=300,
        reassignment_ratio=0.01,
    ).fit(values)


def _pitch_training_data(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_group: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {group: [] for group in GROUPS}
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
    balanced = min(
        MAX_PITCH_TRAINING_PER_GROUP,
        *(values[0].shape[0] for values in available.values()),
    )
    rng = np.random.default_rng(RANDOM_SEED)
    tonnetz_parts: list[np.ndarray] = []
    chroma_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for index, group in enumerate(GROUPS):
        tonnetz, chroma = available[group]
        selected = rng.choice(tonnetz.shape[0], size=balanced, replace=False)
        tonnetz_parts.append(tonnetz[selected])
        chroma_parts.append(chroma[selected])
        group_parts.append(np.full(balanced, index, dtype=np.int8))
    return (
        np.concatenate(tonnetz_parts),
        np.concatenate(chroma_parts),
        np.concatenate(group_parts),
    )


def fit_pitch_codebook(rows: list[dict[str, str]]) -> dict[str, Any]:
    tonnetz, chroma, group_labels = _pitch_training_data(rows)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    fit_values = tonnetz[
        rng.choice(tonnetz.shape[0], size=min(60_000, tonnetz.shape[0]), replace=False)
    ]
    evaluation = tonnetz[
        rng.choice(tonnetz.shape[0], size=min(3_000, tonnetz.shape[0]), replace=False)
    ]
    diagnostic_rows: list[dict[str, Any]] = []
    for k in PITCH_SENSITIVITY_K:
        first = _fit_pitch_k(fit_values, k, RANDOM_SEED + k, 3)
        second = _fit_pitch_k(fit_values, k, RANDOM_SEED + 1_000 + k, 3)
        labels = first.predict(evaluation)
        counts = np.bincount(first.labels_, minlength=k)
        diagnostic_rows.append(
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
    _write_frame(PITCH_DIAGNOSTICS, pd.DataFrame(diagnostic_rows))

    model = _fit_pitch_k(tonnetz, V_PITCH, RANDOM_SEED, 10)
    labels = model.labels_
    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    prototypes = np.zeros((V_PITCH, 12), dtype=np.float64)
    group_counts = np.zeros((V_PITCH, len(GROUPS)), dtype=np.int64)
    counts = np.bincount(labels, minlength=V_PITCH)
    for state in range(V_PITCH):
        mask = labels == state
        prototypes[state] = np.mean(chroma[mask], axis=0)
        group_counts[state] = np.bincount(group_labels[mask], minlength=len(GROUPS))
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
        )
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
    _write_npz_atomic(
        PITCH_CODEBOOK,
        {
            "centers": centers,
            "chroma_prototypes": prototypes,
            "training_counts": counts,
            "training_group_counts": group_counts,
            "top_pitch_classes": top_three.astype(np.int8),
        },
    )
    balanced = int(tonnetz.shape[0] // len(GROUPS))
    payload = {
        "schema_version": 2,
        "method": "beat-synchronous chroma -> fixed Tonnetz -> MiniBatchKMeans",
        "fit_scope": "discovery/180s only; strict equal sampling across four groups",
        "groups": list(GROUPS),
        "v_pitch": V_PITCH,
        "random_seed": RANDOM_SEED,
        "training_steps": int(tonnetz.shape[0]),
        "training_steps_per_group": {group: balanced for group in GROUPS},
        "state_labels": labels_text,
        "diagnostics": diagnostic_rows,
        "codebook_relative_path": _relative(PITCH_CODEBOOK),
        "codebook_sha256": _sha256(PITCH_CODEBOOK),
        "generated_at": date.today().isoformat(),
    }
    _write_json_atomic(PITCH_CODEBOOK_JSON, payload)
    return payload


def _transform_pitch_row(
    row: dict[str, str], centers: np.ndarray, codebook_sha256: str
) -> dict[str, Any]:
    arrays = _read_npz(ROOT / row["chroma_relative_path"])
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
        / "four_group"
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


def transform_pitch(
    rows: list[dict[str, str]], codebook: dict[str, Any], *, workers: int
) -> list[dict[str, Any]]:
    centers = _read_npz(PITCH_CODEBOOK)["centers"].astype(np.float64)
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _transform_pitch_row, row, centers, str(codebook["codebook_sha256"])
            ): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"four-group pitch_v2: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs,
        key=lambda row: (row["group"], row["track_id"], float(row["scale_seconds"])),
    )
    _write_frame(PITCH_FEATURES, pd.DataFrame(ordered))
    return ordered


def _topology_paths(row: dict[str, Any], view: str) -> tuple[Path, Path, Path, Path]:
    scale = int(float(row["scale_seconds"]))
    suffix = Path(view) / f"{scale}s" / row["group"] / row["split"] / f"{row['segment_id']}.npz"
    return (
        ROOT / "graphs" / "four_group" / suffix,
        ROOT / "homology" / "four_group" / "persistence" / suffix,
        ROOT / "homology" / "four_group" / "persistence_sensitivity" / suffix,
        ROOT
        / "homology"
        / "four_group"
        / "descriptors"
        / suffix.with_suffix(".json"),
    )


def _process_topology_row(
    row: dict[str, Any], view: str, state_path_column: str, state_hash_column: str
) -> dict[str, Any]:
    config = load_topology_config(ROOT)
    arrays = _read_npz(ROOT / row[state_path_column])
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
        graph, config.sensitivity_thresholds, tolerance=config.rank_tolerance
    )
    graph_path, primary_path, sensitivity_path, sidecar_path = _topology_paths(row, view)
    _write_npz_atomic(graph_path, _graph_arrays(graph))
    _write_npz_atomic(primary_path, _persistence_arrays(primary))
    _write_npz_atomic(sensitivity_path, _persistence_arrays(sensitivity))
    identity = {
        "segment_id": row["segment_id"],
        "track_id": row["track_id"],
        "group": row["group"],
        "split": row["split"],
        "scale_seconds": float(row["scale_seconds"]),
        "view": view,
    }
    job = TopologyJob(
        **identity,
        feature_relative_path=row[state_path_column],
        feature_sha256=row[state_hash_column],
    )
    graph_metrics = _graph_metrics(states, graph)
    topology_metrics = _topology_metrics(primary)
    config_payload = {
        "study": "four_group",
        "view": view,
        "groups": list(GROUPS),
        "topology": asdict(config),
    }
    config_sha256 = __import__("hashlib").sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = {
        **identity,
        "feature_relative_path": row[state_path_column],
        "feature_sha256": row[state_hash_column],
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
            "feature_relative_path": output["feature_relative_path"],
            "feature_sha256": output["feature_sha256"],
            "config_sha256": config_sha256,
            "graph_metrics": graph_metrics,
            "topology_metrics": topology_metrics,
            "outputs": {
                "graph": output["graph_relative_path"],
                "persistence": output["persistence_relative_path"],
                "sensitivity_persistence": output[
                    "sensitivity_persistence_relative_path"
                ],
            },
            "processed_at": output["processed_at"],
        },
    )
    return output


def run_topology(
    feature_rows: list[dict[str, Any]],
    pitch_rows: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    tasks: list[tuple[dict[str, Any], str, str, str]] = []
    for row in feature_rows:
        tasks.append((row, "rhythm", "rhythm_relative_path", "rhythm_sha256"))
        tasks.append((row, "structure", "structure_relative_path", "structure_sha256"))
    for row in pitch_rows:
        tasks.append((row, "pitch_v2", "pitch_v2_relative_path", "pitch_v2_sha256"))
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_topology_row, row, view, path_col, hash_col): view
            for row, view, path_col, hash_col in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"four-group path homology: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs,
        key=lambda row: (
            row["group"],
            row["track_id"],
            float(row["scale_seconds"]),
            row["view"],
        ),
    )
    _write_csv(TOPOLOGY, ordered, SEGMENT_COLUMNS)
    _write_csv(
        FILTRATION,
        [item for row in ordered for item in row["filtration"]],
        FILTRATION_COLUMNS,
    )
    _write_csv(
        SENSITIVITY,
        [item for row in ordered for item in row["sensitivity_filtration"]],
        FILTRATION_COLUMNS,
    )
    return ordered


def run_statistics(topology_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(topology_rows).drop(columns=["filtration", "sensitivity_filtration"])
    omnibus, pairwise = _omnibus_and_pairwise(frame)
    omnibus["fdr_family"] = "analysis_set: 3 views x 20 metrics"
    pairwise["fdr_family"] = "analysis_set: 3 views x 20 metrics x 6 contrasts"
    _write_frame(OMNIBUS, omnibus)
    _write_frame(PAIRWISE, pairwise)
    return omnibus, pairwise


def write_summary(
    manifest_summary: dict[str, Any],
    state_metadata: dict[str, Any],
    codebook: dict[str, Any],
    topology_rows: list[dict[str, Any]],
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> dict[str, Any]:
    topology = pd.DataFrame(topology_rows)
    primary = topology[
        (topology["split"] == "validation") & (topology["scale_seconds"] == 180.0)
    ]
    metrics = [
        "vertex_count",
        "edge_count",
        "self_transition_ratio",
        "path_entropy",
        "directed_recurrence",
        "reciprocity",
        "h0_betti_mean",
        "h1_betti_max",
    ]
    medians: dict[str, Any] = {}
    h1_counts: dict[str, Any] = {}
    for view, view_rows in primary.groupby("view"):
        medians[view] = view_rows.groupby("group")[metrics].median().to_dict(orient="index")
        h1_counts[view] = {
            group: {
                "n": int(len(group_rows)),
                "nonzero": int(np.count_nonzero(group_rows["h1_betti_max"] > 0)),
            }
            for group, group_rows in view_rows.groupby("group")
        }
    primary_tests = omnibus[omnibus["analysis_set"] == "primary_validation_180"]
    primary_pairs = pairwise[pairwise["analysis_set"] == "primary_validation_180"]
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "scope": "four-group structure, pitch_v2, and rhythm Path Homology",
        "groups": list(GROUPS),
        "manifest": manifest_summary,
        "segments": int(topology["segment_id"].nunique()),
        "tracks": int(topology["track_id"].nunique()),
        "segment_views": int(len(topology)),
        "view_counts": topology["view"].value_counts().sort_index().astype(int).to_dict(),
        "status_counts": topology["status"].value_counts().astype(int).to_dict(),
        "primary_validation_n_per_view": int(len(primary) // 3),
        "primary_omnibus_fdr_discoveries": int(np.count_nonzero(primary_tests["p_fdr_bh"] <= 0.10)),
        "primary_pairwise_fdr_discoveries": int(
            np.count_nonzero(primary_pairs["p_fdr_bh"] <= 0.10)
        ),
        "validation_180_group_medians": medians,
        "validation_180_h1_counts": h1_counts,
        "state_model_sha256": _sha256(STATE_MODEL),
        "state_model_training_groups": state_metadata["training_groups"],
        "state_model_sampled_windows": state_metadata["sampled_windows"],
        "pitch_codebook_sha256": codebook["codebook_sha256"],
        "pitch_training_steps_per_group": codebook["training_steps_per_group"],
        "graph_construction": {
            "structure": (
                "SSM-derived macro-boundaries -> frozen section states -> adjacent transitions"
            ),
            "pitch_v2": "adjacent frozen Tonnetz-codebook states; no SSM",
            "rhythm": "adjacent frozen rhythm states; no SSM",
        },
        "inference": {
            "primary": "validation/180s",
            "sensitivity": "validation/300s",
            "exploratory": "discovery/180s",
            "holdout_in_omnibus": False,
            "omnibus_fdr_family": "per analysis set across 3 views x 20 metrics",
            "pairwise_fdr_family": "per analysis set across 3 views x 20 metrics x 6 contrasts",
        },
        "artifacts": {
            "preprocessed_manifest": _relative(PREPROCESSED),
            "raw_feature_manifest": _relative(RAW_FEATURES),
            "feature_manifest": _relative(FEATURES),
            "pitch_feature_manifest": _relative(PITCH_FEATURES),
            "topology_manifest": _relative(TOPOLOGY),
            "filtration_manifest": _relative(FILTRATION),
            "sensitivity_filtration_manifest": _relative(SENSITIVITY),
            "omnibus_tests": _relative(OMNIBUS),
            "pairwise_tests": _relative(PAIRWISE),
        },
    }
    _write_json_atomic(SUMMARY, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("building isolated four-group manifest", flush=True)
    manifest_summary = build_four_group_manifest()
    print("extracting or verifying Focus Open continuous features", flush=True)
    open_rows = extract_open_focus(workers=args.workers)
    raw_rows = combine_raw_features(open_rows)
    print("fitting four-group state model and transforming rhythm/structure", flush=True)
    feature_rows, state_metadata = fit_and_transform_states(raw_rows, workers=args.workers)
    print("fitting four-group Tonnetz pitch_v2 codebook", flush=True)
    codebook = fit_pitch_codebook(raw_rows)
    pitch_rows = transform_pitch(raw_rows, codebook, workers=args.workers)
    print("running persistent Path Homology for three views", flush=True)
    topology_rows = run_topology(feature_rows, pitch_rows, workers=args.workers)
    print("running four-group omnibus and pairwise statistics", flush=True)
    omnibus, pairwise = run_statistics(topology_rows)
    summary = write_summary(
        manifest_summary, state_metadata, codebook, topology_rows, omnibus, pairwise
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
