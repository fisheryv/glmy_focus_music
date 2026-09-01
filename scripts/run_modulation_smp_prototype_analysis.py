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

from data.analysis_inputs import audit_analysis_inputs
from features.batch import _json_hash, _sha256, _write_json_atomic, _write_npz_atomic
from features.modulation_smp import (
    SharedSMPTransform,
    SMPPrototypeCodebook,
    assign_states,
    fit_codebook,
    fit_shared_transform,
)
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
    load_topology_config,
)
from topology.statistics import _omnibus_and_pairwise

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("classical", "focus")
STATE_COUNTS = (8, 10, 12)
PRIMARY_STATE_COUNT = 10
PCA_COMPONENTS = 32
RANDOM_SEED = 20_260_805
FDR_Q = 0.05
WORKERS = 6

SHARED_MODEL_NPZ = ROOT / "features" / "models" / "modulation_smp_shared_transform.npz"
SHARED_MODEL_JSON = ROOT / "features" / "models" / "modulation_smp_shared_transform.json"
FEATURE_MANIFEST = ROOT / "metadata" / "modulation_smp_prototype_features.csv"
SEGMENT_MANIFEST = ROOT / "metadata" / "modulation_smp_prototype_topology_segments.csv"
FILTRATION_MANIFEST = ROOT / "metadata" / "modulation_smp_prototype_topology_filtration.csv"
SENSITIVITY_MANIFEST = (
    ROOT / "metadata" / "modulation_smp_prototype_topology_filtration_sensitivity.csv"
)
TESTS_MANIFEST = ROOT / "metadata" / "modulation_smp_prototype_statistical_tests.csv"
PAIRWISE_MANIFEST = ROOT / "metadata" / "modulation_smp_prototype_pairwise_tests.csv"
SUMMARY_JSON = ROOT / "metadata" / "modulation_smp_prototype_summary.json"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _codebook_npz(state_count: int) -> Path:
    return ROOT / "features" / "models" / f"modulation_smp_proto_k{state_count}.npz"


def _codebook_json(state_count: int) -> Path:
    return ROOT / "features" / "models" / f"modulation_smp_proto_k{state_count}.json"


def _load_source_rows() -> list[dict[str, str]]:
    source = ROOT / "metadata" / "feature_segments.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") != "failed"]
    groups = Counter(row["group"] for row in rows)
    scales = Counter(float(row["scale_seconds"]) for row in rows)
    tracks = len({row["track_id"] for row in rows})
    if (
        len(rows) != 1_200
        or tracks != 600
        or dict(groups) != {"classical": 600, "focus": 600}
        or dict(scales) != {180.0: 600, 300.0: 600}
    ):
        raise RuntimeError(
            "canonical dataset audit failed: "
            f"rows={len(rows)}, tracks={tracks}, groups={dict(groups)}, scales={dict(scales)}"
        )
    if any(not row.get("modulation_relative_path") for row in rows):
        raise RuntimeError("one or more source rows lack modulation features")
    return rows


def _valid_spectrum(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.asarray(arrays["spectrum"], dtype=np.float64)
    valid = np.asarray(arrays["valid"], dtype=bool)
    if spectrum.ndim != 2 or valid.shape != (spectrum.shape[0],):
        raise RuntimeError("invalid SMP spectrum archive")
    valid &= np.all(np.isfinite(spectrum), axis=1) & np.all(spectrum >= 0.0, axis=1)
    valid &= np.sum(spectrum, axis=1) > np.finfo(float).eps
    return spectrum, valid


def _save_shared_model(transform: SharedSMPTransform, training_counts: np.ndarray) -> str:
    _write_npz_atomic(
        SHARED_MODEL_NPZ,
        {
            "frequencies": transform.frequencies,
            "robust_center": transform.robust_center,
            "robust_scale": transform.robust_scale,
            "pca_mean": transform.pca_mean,
            "pca_components": transform.pca_components,
            "explained_variance_ratio": transform.explained_variance_ratio,
            "training_counts": training_counts,
        },
    )
    digest = _sha256(SHARED_MODEL_NPZ)
    _write_json_atomic(
        SHARED_MODEL_JSON,
        {
            "schema_version": 1,
            "view": "modulation_smp_prototype",
            "fit_scope": "balanced Classical/Focus discovery/180s valid SMP windows only",
            "source_smp": "normalized 0.5-45 Hz spectral modulation profile, 4s window/2s hop",
            "transform": "elementwise square root (Hellinger), median/IQR scaling, PCA-32",
            "pca_components": PCA_COMPONENTS,
            "pca_explained_variance": float(np.sum(transform.explained_variance_ratio)),
            "random_seed": RANDOM_SEED,
            "training_groups": list(GROUPS),
            "training_counts": {
                group: int(training_counts[index]) for index, group in enumerate(GROUPS)
            },
            "frequencies_hz": [
                float(transform.frequencies[0]),
                float(transform.frequencies[-1]),
            ],
            "frequency_bins": int(transform.frequencies.size),
            "model_relative_path": _relative(SHARED_MODEL_NPZ),
            "model_sha256": digest,
            "generated_at": date.today().isoformat(),
        },
    )
    return digest


def _save_codebook(codebook: SMPPrototypeCodebook, shared_sha256: str) -> dict[str, Any]:
    npz_path = _codebook_npz(codebook.state_count)
    json_path = _codebook_json(codebook.state_count)
    _write_npz_atomic(
        npz_path,
        {
            "centers": codebook.centers,
            "prototype_spectra": codebook.prototype_spectra,
            "spectral_centroids_hz": codebook.spectral_centroids_hz,
            "training_state_counts": codebook.training_state_counts,
        },
    )
    digest = _sha256(npz_path)
    payload = {
        "schema_version": 1,
        "view": f"modulation_smp_k{codebook.state_count}",
        "state_count": codebook.state_count,
        "role": "primary"
        if codebook.state_count == PRIMARY_STATE_COUNT
        else "representation_sensitivity",
        "fit_scope": "balanced Classical/Focus discovery/180s valid SMP windows only",
        "shared_transform_sha256": shared_sha256,
        "random_seed": RANDOM_SEED + codebook.state_count,
        "state_order": "ascending prototype spectral-modulation centroid",
        "spectral_centroids_hz": codebook.spectral_centroids_hz.tolist(),
        "training_state_counts": codebook.training_state_counts.astype(int).tolist(),
        "model_relative_path": _relative(npz_path),
        "model_sha256": digest,
        "generated_at": date.today().isoformat(),
    }
    _write_json_atomic(json_path, payload)
    return payload


def fit_models(
    rows: list[dict[str, str]],
) -> tuple[SharedSMPTransform, dict[int, SMPPrototypeCodebook], dict[str, Any]]:
    available: dict[str, list[np.ndarray]] = {group: [] for group in GROUPS}
    reference_frequencies: np.ndarray | None = None
    for row in rows:
        if row["split"] != "discovery" or float(row["scale_seconds"]) != 180.0:
            continue
        arrays = _read_npz(ROOT / row["modulation_relative_path"])
        spectrum, valid = _valid_spectrum(arrays)
        frequencies = np.asarray(arrays["frequencies"], dtype=np.float64)
        if reference_frequencies is None:
            reference_frequencies = frequencies
        elif not np.array_equal(frequencies, reference_frequencies):
            raise RuntimeError("SMP frequency grids differ across discovery segments")
        available[row["group"]].append(spectrum[valid])
    if reference_frequencies is None:
        raise RuntimeError("no discovery SMP windows were found")
    pooled = {group: np.concatenate(available[group], axis=0) for group in GROUPS}
    balanced_count = min(values.shape[0] for values in pooled.values())
    rng = np.random.default_rng(RANDOM_SEED)
    sampled = {
        group: values[rng.choice(values.shape[0], balanced_count, replace=False)]
        for group, values in pooled.items()
    }
    training = np.concatenate([sampled[group] for group in GROUPS], axis=0)
    transform, embedded = fit_shared_transform(
        training,
        reference_frequencies,
        n_components=PCA_COMPONENTS,
        random_seed=RANDOM_SEED,
    )
    training_counts = np.asarray([balanced_count, balanced_count], dtype=np.int64)
    shared_sha256 = _save_shared_model(transform, training_counts)
    codebooks: dict[int, SMPPrototypeCodebook] = {}
    codebook_payloads: dict[str, Any] = {}
    for state_count in STATE_COUNTS:
        codebook = fit_codebook(
            embedded,
            training,
            reference_frequencies,
            state_count=state_count,
            random_seed=RANDOM_SEED + state_count,
        )
        codebooks[state_count] = codebook
        codebook_payloads[str(state_count)] = _save_codebook(codebook, shared_sha256)
    model = {
        "shared_model_sha256": shared_sha256,
        "available_windows": {group: int(values.shape[0]) for group, values in pooled.items()},
        "sampled_windows": {group: balanced_count for group in GROUPS},
        "pca_explained_variance": float(np.sum(transform.explained_variance_ratio)),
        "codebooks": codebook_payloads,
    }
    return transform, codebooks, model


def _transform_row(
    row: dict[str, str],
    transform: SharedSMPTransform,
    codebooks: dict[int, SMPPrototypeCodebook],
    model_set_sha256: str,
) -> dict[str, Any]:
    arrays = _read_npz(ROOT / row["modulation_relative_path"])
    spectrum, valid = _valid_spectrum(arrays)
    if not np.array_equal(np.asarray(arrays["frequencies"], dtype=float), transform.frequencies):
        raise RuntimeError(f"SMP frequency grid mismatch: {row['segment_id']}")
    state_arrays = {
        state_count: assign_states(spectrum, valid, transform, codebook)
        for state_count, codebook in codebooks.items()
    }
    scale = int(float(row["scale_seconds"]))
    output = (
        ROOT
        / "features"
        / "modulation_smp_prototype"
        / f"{scale}s"
        / row["group"]
        / row["split"]
        / f"{row['segment_id']}.npz"
    )
    payload = {
        "times": np.asarray(arrays["times"], dtype=np.float32),
        "valid": valid,
        "frequencies": transform.frequencies.astype(np.float32),
    }
    payload.update({f"states_k{k}": states for k, states in state_arrays.items()})
    _write_npz_atomic(output, payload)
    result: dict[str, Any] = {
        "segment_id": row["segment_id"],
        "track_id": row["track_id"],
        "group": row["group"],
        "split": row["split"],
        "scale_seconds": float(row["scale_seconds"]),
        "source_modulation_relative_path": row["modulation_relative_path"],
        "source_modulation_sha256": row["modulation_sha256"],
        "feature_relative_path": _relative(output),
        "feature_sha256": _sha256(output),
        "model_set_sha256": model_set_sha256,
        "windows": int(valid.size),
        "valid_windows": int(valid.sum()),
        "masked_windows": int((~valid).sum()),
        "status": "success",
        "processed_at": date.today().isoformat(),
        "error": "",
    }
    for state_count, states in state_arrays.items():
        usable = states[states >= 0]
        result[f"observed_states_k{state_count}"] = int(np.unique(usable).size)
        result[f"state_coverage_k{state_count}"] = float(np.unique(usable).size / state_count)
        result[f"state_counts_k{state_count}"] = json.dumps(
            np.bincount(usable, minlength=state_count).astype(int).tolist(),
            separators=(",", ":"),
        )
    return result


def transform_features(
    rows: list[dict[str, str]],
    transform: SharedSMPTransform,
    codebooks: dict[int, SMPPrototypeCodebook],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    model_set_sha256 = _json_hash(
        {
            "shared": model["shared_model_sha256"],
            "codebooks": {key: value["model_sha256"] for key, value in model["codebooks"].items()},
        }
    )
    outputs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_transform_row, row, transform, codebooks, model_set_sha256): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"SMP prototype features: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs, key=lambda item: (item["group"], item["track_id"], item["scale_seconds"])
    )
    pd.DataFrame(ordered).to_csv(
        FEATURE_MANIFEST, index=False, encoding="utf-8", lineterminator="\n"
    )
    model["model_set_sha256"] = model_set_sha256
    return ordered


def _topology_paths(row: dict[str, Any], state_count: int) -> tuple[Path, Path, Path, Path]:
    scale = int(float(row["scale_seconds"]))
    suffix = (
        Path("modulation_smp_prototype")
        / f"k{state_count}"
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


def _process_topology_row(
    row: dict[str, Any], state_count: int, config: Any, config_sha256: str
) -> dict[str, Any]:
    arrays = _read_npz(ROOT / row["feature_relative_path"])
    raw = np.asarray(arrays[f"states_k{state_count}"], dtype=int)
    states = [int(value) if value >= 0 else None for value in raw]
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
    graph_path, primary_path, sensitivity_path, sidecar_path = _topology_paths(row, state_count)
    _write_npz_atomic(graph_path, _graph_arrays(graph))
    _write_npz_atomic(primary_path, _persistence_arrays(primary))
    _write_npz_atomic(sensitivity_path, _persistence_arrays(sensitivity))
    identity = {
        "segment_id": row["segment_id"],
        "track_id": row["track_id"],
        "group": row["group"],
        "split": row["split"],
        "scale_seconds": row["scale_seconds"],
        "view": f"modulation_smp_k{state_count}",
    }
    job = TopologyJob(
        **identity,
        feature_relative_path=row["feature_relative_path"],
        feature_sha256=row["feature_sha256"],
    )
    graph_metrics = _graph_metrics(states, graph)
    topology_metrics = _topology_metrics(primary)
    possible_edges = state_count * (state_count - 1)
    diagnostics = {
        "state_count": state_count,
        "state_coverage": graph_metrics["vertex_count"] / state_count,
        "retained_edge_ratio": graph_metrics["edge_count"] / possible_edges,
    }
    result = {
        **identity,
        **diagnostics,
        "feature_relative_path": row["feature_relative_path"],
        "feature_sha256": row["feature_sha256"],
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
            "diagnostics": diagnostics,
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
    hashes = {
        state_count: _json_hash(
            {
                "view": f"modulation_smp_k{state_count}",
                "model_set_sha256": model["model_set_sha256"],
                "codebook_sha256": model["codebooks"][str(state_count)]["model_sha256"],
                "state_count": state_count,
                "invalid_policy": "missing state; no transition across a missing window",
                "topology": asdict(config),
            }
        )
        for state_count in STATE_COUNTS
    }
    outputs: list[dict[str, Any]] = []
    tasks = [(row, state_count) for state_count in STATE_COUNTS for row in rows]
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(
                _process_topology_row,
                row,
                state_count,
                config,
                hashes[state_count],
            ): (row, state_count)
            for row, state_count in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"SMP prototype path homology: {completed}/{len(futures)}", flush=True)
    ordered = sorted(
        outputs,
        key=lambda item: (
            item["state_count"],
            item["group"],
            item["track_id"],
            item["scale_seconds"],
        ),
    )
    clean = [
        {
            key: value
            for key, value in row.items()
            if key not in {"filtration", "sensitivity_filtration"}
        }
        for row in ordered
    ]
    segment_columns = (
        list(SEGMENT_COLUMNS[:6])
        + [
            "state_count",
            "state_coverage",
            "retained_edge_ratio",
        ]
        + list(SEGMENT_COLUMNS[6:])
    )
    pd.DataFrame(clean)[segment_columns].to_csv(
        SEGMENT_MANIFEST, index=False, encoding="utf-8", lineterminator="\n"
    )
    filtration = []
    sensitivity = []
    for row in ordered:
        filtration.extend({**item, "state_count": row["state_count"]} for item in row["filtration"])
        sensitivity.extend(
            {**item, "state_count": row["state_count"]} for item in row["sensitivity_filtration"]
        )
    filtration_columns = (
        list(FILTRATION_COLUMNS[:6]) + ["state_count"] + list(FILTRATION_COLUMNS[6:])
    )
    pd.DataFrame(filtration)[filtration_columns].to_csv(
        FILTRATION_MANIFEST, index=False, encoding="utf-8", lineterminator="\n"
    )
    pd.DataFrame(sensitivity)[filtration_columns].to_csv(
        SENSITIVITY_MANIFEST, index=False, encoding="utf-8", lineterminator="\n"
    )
    return ordered


def _mechanism_example(topology: pd.DataFrame) -> dict[str, Any]:
    subset = topology[
        (topology.state_count == PRIMARY_STATE_COUNT)
        & (topology.split == "validation")
        & np.isclose(topology.scale_seconds, 180.0)
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
            "then longest lifetime"
        )
        return chosen
    chosen = subset.sort_values(
        ["h1_betti_max", "edge_count", "path_entropy", "segment_id"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return {
        "segment_id": str(chosen.segment_id),
        "track_id": str(chosen.track_id),
        "group": str(chosen.group),
        "finite_h1_available": False,
        "finite_h1_intervals": 0,
        "birth_threshold": None,
        "death_threshold": None,
        "lifetime": 0.0,
        "selection_rule": "no finite H1 interval; choose highest H1 max, edges, then path entropy",
    }


def run_statistics(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    topology = pd.DataFrame(rows).drop(columns=["filtration", "sensitivity_filtration"])
    all_omnibus = []
    all_pairwise = []
    for state_count in STATE_COUNTS:
        frame = topology[topology.state_count == state_count]
        omnibus, pairwise = _omnibus_and_pairwise(
            frame,
            bootstrap_resamples=3000,
            bootstrap_seed=20260716,
            bootstrap_views=frozenset({"modulation_smp_k10"}),
        )
        omnibus.insert(5, "state_count", state_count)
        pairwise.insert(5, "state_count", state_count)
        all_omnibus.append(omnibus)
        all_pairwise.append(pairwise)
    omnibus = pd.concat(all_omnibus, ignore_index=True)
    pairwise = pd.concat(all_pairwise, ignore_index=True)
    omnibus.to_csv(TESTS_MANIFEST, index=False, encoding="utf-8", lineterminator="\n")
    pairwise.to_csv(PAIRWISE_MANIFEST, index=False, encoding="utf-8", lineterminator="\n")

    sensitivity_filtration = pd.read_csv(SENSITIVITY_MANIFEST)
    model_summaries: dict[str, Any] = {}
    for state_count in STATE_COUNTS:
        primary = omnibus[
            (omnibus.state_count == state_count)
            & (omnibus.analysis_set == "primary_validation_180")
        ]
        duration = omnibus[
            (omnibus.state_count == state_count)
            & (omnibus.analysis_set == "sensitivity_validation_300")
        ].set_index("metric")
        primary_pair = pairwise[
            (pairwise.state_count == state_count)
            & (pairwise.analysis_set == "primary_validation_180")
        ].set_index("metric")
        duration_pair = pairwise[
            (pairwise.state_count == state_count)
            & (pairwise.analysis_set == "sensitivity_validation_300")
        ].set_index("metric")
        significant = primary[primary.p_fdr_bh <= FDR_Q]
        stable = 0
        for row in significant.itertuples(index=False):
            if float(duration.loc[row.metric, "p_fdr_bh"]) <= FDR_Q and np.sign(
                -primary_pair.loc[row.metric, "rank_biserial_a_minus_b"]
            ) == np.sign(-duration_pair.loc[row.metric, "rank_biserial_a_minus_b"]):
                stable += 1
        validation = topology[
            (topology.state_count == state_count)
            & (topology.split == "validation")
            & np.isclose(topology.scale_seconds, 180.0)
        ]
        sensitivity_max = (
            sensitivity_filtration[
                (sensitivity_filtration.state_count == state_count)
                & (sensitivity_filtration.split == "validation")
                & np.isclose(sensitivity_filtration.scale_seconds, 180.0)
            ]
            .groupby(["segment_id", "group"], as_index=False)["h1_betti"]
            .max()
        )
        h1_counts = {
            group: {
                "total": int(len(frame)),
                "primary_nonzero": int((frame.h1_betti_max > 0).sum()),
                "sensitivity_nonzero": int(
                    (sensitivity_max.loc[sensitivity_max.group == group, "h1_betti"] > 0).sum()
                ),
            }
            for group, frame in validation.groupby("group")
        }
        model_summaries[str(state_count)] = {
            "role": "primary"
            if state_count == PRIMARY_STATE_COUNT
            else "representation_sensitivity",
            "primary_fdr_discoveries": int((primary.p_fdr_bh <= FDR_Q).sum()),
            "duration_fdr_discoveries": int((duration.p_fdr_bh <= FDR_Q).sum()),
            "stable_same_direction_discoveries": int(stable),
            "validation_180_h1_counts": h1_counts,
            "validation_180_diagnostics": {
                "median_observed_states": float(validation.vertex_count.median()),
                "median_state_coverage": float(validation.state_coverage.median()),
                "median_edges": float(validation.edge_count.median()),
                "median_edge_density": float(validation.edge_density.median()),
                "median_retained_edge_ratio": float(validation.retained_edge_ratio.median()),
            },
        }

    summary: dict[str, Any] = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "scope": "shared SMP prototype Path Homology, K=8/10/12",
        "evidence_role": "post-holdout exploratory validation; not a frozen holdout confirmation",
        "canonical_groups": list(GROUPS),
        "input_provenance": input_audit,
        "segment_views": int(len(topology)),
        "source_segments": int(topology.segment_id.nunique()),
        "tracks": int(topology.track_id.nunique()),
        "state_counts": list(STATE_COUNTS),
        "primary_state_count": PRIMARY_STATE_COUNT,
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance": model["pca_explained_variance"],
        "fdr_q": FDR_Q,
        "topology_metrics_per_model": 20,
        "model_set_sha256": model["model_set_sha256"],
        "shared_model_sha256": model["shared_model_sha256"],
        "codebook_sha256": {
            key: value["model_sha256"] for key, value in model["codebooks"].items()
        },
        "available_windows": model["available_windows"],
        "sampled_windows": model["sampled_windows"],
        "models": model_summaries,
        "mechanism_example": _mechanism_example(topology),
        "graph_policy": {
            "top_k": 6,
            "include_self_loops": False,
            "primary_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "sensitivity_thresholds": [
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
            ],
        },
        "artifacts": {
            "feature_manifest": _relative(FEATURE_MANIFEST),
            "segment_manifest": _relative(SEGMENT_MANIFEST),
            "filtration_manifest": _relative(FILTRATION_MANIFEST),
            "sensitivity_filtration_manifest": _relative(SENSITIVITY_MANIFEST),
            "statistical_tests": _relative(TESTS_MANIFEST),
            "pairwise_tests": _relative(PAIRWISE_MANIFEST),
        },
    }
    artifact_paths = [
        SHARED_MODEL_NPZ,
        SHARED_MODEL_JSON,
        *[_codebook_npz(state_count) for state_count in STATE_COUNTS],
        *[_codebook_json(state_count) for state_count in STATE_COUNTS],
        FEATURE_MANIFEST,
        SEGMENT_MANIFEST,
        FILTRATION_MANIFEST,
        SENSITIVITY_MANIFEST,
        TESTS_MANIFEST,
        PAIRWISE_MANIFEST,
    ]
    summary["artifact_sha256"] = {_relative(path): _sha256(path) for path in artifact_paths}
    _write_json_atomic(SUMMARY_JSON, summary)
    return summary


def main() -> int:
    input_audit = audit_analysis_inputs(root=ROOT)
    rows = _load_source_rows()
    print("SMP prototypes: fitting shared balanced discovery/180s transform", flush=True)
    transform, codebooks, model = fit_models(rows)
    print("SMP prototypes: assigning K=8/10/12 states to all segments", flush=True)
    features = transform_features(rows, transform, codebooks, model)
    print("SMP prototypes: running persistent path homology", flush=True)
    topology = run_topology(features, model)
    print("SMP prototypes: running per-K two-group statistics", flush=True)
    summary = run_statistics(topology, model, input_audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
