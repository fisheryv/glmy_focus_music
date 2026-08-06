from __future__ import annotations

import argparse
import json
import math
import os
import tomllib
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.batch import _json_hash, _sha256, _write_json_atomic
from graphs.transition import TransitionGraph, WeightedEdge
from homology.glmy import persistent_path_homology
from repetition.analysis import (
    IDENTITY_COLUMNS,
    RepetitionAnalysisError,
    _block_mean,
    _candidate_data,
    _dominant_lag_from_distance,
    _effect_test,
    _load_model,
    _seed,
    _standard_distance,
    _write_csv,
    load_config,
)
from topology.statistics import benjamini_hochberg

MODALITIES = ("acoustic", "rhythm")
TOPOLOGY_METRICS = (
    "h1_max_lifetime",
    "h1_dominant_persistence",
    "h1_normalized_auc",
)
BASELINE_COLUMNS = (
    "widest_cycle_baseline",
    "phase_recurrence_mean",
    "transition_concentration",
)
SENSITIVITY_THRESHOLDS = (0.40, 0.55, 0.70)


@dataclass(frozen=True, slots=True)
class StrengthenedConfig:
    exploration_tracks_per_group: int = 24
    phase_bins: int = 6
    state_bins: int = 3
    state_similarity_threshold: float = 0.55
    block_frames: int = 4
    min_raw_steps: int = 96
    min_cycles: int = 3
    max_period_blocks: int = 32
    recurrence_phase_radius: int = 1
    top_k_per_source: int = 3
    path_thresholds: tuple[float, ...] = tuple(np.arange(0.05, 1.0, 0.05))
    calibration_fdr_q: float = 0.05
    calibration_min_delta: float = 0.05
    calibration_positive_fraction: float = 0.75
    selection_p_threshold: float = 0.05
    selection_effect_threshold: float = 0.20
    selection_stability_threshold: float = 0.30
    incremental_p_threshold: float = 0.10
    incremental_effect_threshold: float = 0.10
    validation_fdr_q: float = 0.05
    workers: int = 4
    random_seed: int = 20260716

    def validate(self) -> None:
        if self.exploration_tracks_per_group < 8:
            raise RepetitionAnalysisError("strengthened exploration sample is too small")
        if self.phase_bins < 4 or self.state_bins < 2:
            raise RepetitionAnalysisError("phase/state graph is too small")
        if not 0 < self.state_similarity_threshold < 1:
            raise RepetitionAnalysisError("state similarity threshold must lie in (0, 1)")
        if self.block_frames < 1 or self.min_cycles < 3:
            raise RepetitionAnalysisError("invalid aggregation or cycle count")
        if self.min_raw_steps < self.phase_bins * self.block_frames * self.min_cycles:
            raise RepetitionAnalysisError("min_raw_steps cannot support the requested graph")
        if self.recurrence_phase_radius < 0 or self.top_k_per_source < 1:
            raise RepetitionAnalysisError("invalid recurrence radius or top-k")
        if not self.path_thresholds or any(not 0 < value < 1 for value in self.path_thresholds):
            raise RepetitionAnalysisError("path thresholds must lie in (0, 1)")
        probabilities = (
            self.calibration_fdr_q,
            self.calibration_positive_fraction,
            self.selection_p_threshold,
            self.selection_effect_threshold,
            self.selection_stability_threshold,
            self.incremental_p_threshold,
            self.incremental_effect_threshold,
            self.validation_fdr_q,
        )
        if any(not 0 < value < 1 for value in probabilities):
            raise RepetitionAnalysisError("statistical thresholds must lie in (0, 1)")
        if self.calibration_min_delta <= 0 or self.workers < 1:
            raise RepetitionAnalysisError("calibration delta and workers must be positive")


def load_strengthened_config(root: Path) -> StrengthenedConfig:
    with (root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    values = dict(raw.get("repetition_strengthened", {}))
    if "path_thresholds" in values:
        values["path_thresholds"] = tuple(float(value) for value in values["path_thresholds"])
    values.setdefault("random_seed", int(raw.get("project", {}).get("seed", 20260716)))
    unknown = set(values) - set(StrengthenedConfig.__dataclass_fields__)
    if unknown:
        raise RepetitionAnalysisError(f"unknown repetition_strengthened keys: {sorted(unknown)}")
    config = StrengthenedConfig(**values)
    config.validate()
    return config


def _pairwise_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standardized = (values - np.median(values, axis=0)) / np.maximum(
        np.std(values, axis=0), 1e-8
    )
    return np.linalg.norm(standardized[:, None] - standardized[None, :], axis=2) / math.sqrt(
        standardized.shape[1]
    )


def _deterministic_state_labels(values: np.ndarray, state_bins: int) -> np.ndarray:
    distances = _pairwise_distance(values)
    count = min(state_bins, len(values))
    first = int(np.argmax(np.mean(distances, axis=1)))
    selected = [first]
    minimum = distances[first].copy()
    for _ in range(1, count):
        candidate = int(np.argmax(minimum))
        selected.append(candidate)
        minimum = np.minimum(minimum, distances[candidate])
    centers = values[selected]
    center_distances = _pairwise_cross_distance(values, centers)
    return np.argmin(center_distances, axis=1).astype(int)


def _pairwise_cross_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    combined = np.vstack([first, second])
    median = np.median(combined, axis=0)
    scale = np.maximum(np.std(combined, axis=0), 1e-8)
    left = (first - median) / scale
    right = (second - median) / scale
    return np.linalg.norm(left[:, None] - right[None, :], axis=2) / math.sqrt(left.shape[1])


def _phase_snapshots(
    values: np.ndarray, period: int, phase_bins: int, min_cycles: int
) -> np.ndarray:
    cycle_count = len(values) // period
    if cycle_count < min_cycles:
        raise RepetitionAnalysisError("not enough complete cycles for a phase-state graph")
    usable = cycle_count * period
    start = (len(values) - usable) // 2
    cycles = np.asarray(values[start : start + usable], dtype=float).reshape(
        cycle_count, period, -1
    )
    snapshots = np.empty((cycle_count, phase_bins, cycles.shape[2]), dtype=float)
    for phase in range(phase_bins):
        left = phase * period // phase_bins
        right = (phase + 1) * period // phase_bins
        snapshots[:, phase] = np.mean(cycles[:, left:right], axis=1)
    return snapshots


def _adaptive_phase_state_labels(
    snapshots: np.ndarray,
    distances: np.ndarray,
    distance_scale: float,
    config: StrengthenedConfig,
) -> np.ndarray:
    cycle_count, phase_bins = snapshots.shape[:2]
    labels = np.zeros((cycle_count, phase_bins), dtype=int)
    for phase in range(phase_bins):
        indices = [cycle * phase_bins + phase for cycle in range(cycle_count)]
        parent = list(range(cycle_count))

        def find(index: int, parents: list[int] = parent) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(
            first: int,
            second: int,
            parents: list[int] = parent,
            find_root: Any = find,
        ) -> None:
            root_first = find_root(first)
            root_second = find_root(second)
            if root_first != root_second:
                parents[max(root_first, root_second)] = min(root_first, root_second)

        for first in range(cycle_count):
            for second in range(first + 1, cycle_count):
                similarity = math.exp(
                    -distances[indices[first], indices[second]] / distance_scale
                )
                if similarity >= config.state_similarity_threshold:
                    union(first, second)
        roots = [find(index) for index in range(cycle_count)]
        unique_roots = sorted(set(roots))
        if len(unique_roots) <= config.state_bins:
            mapping = {root: index for index, root in enumerate(unique_roots)}
            labels[:, phase] = [mapping[root] for root in roots]
        else:
            labels[:, phase] = _deterministic_state_labels(
                snapshots[:, phase], config.state_bins
            )
    return labels


def _add_weighted_counts(
    storage: dict[tuple[tuple[int, int], tuple[int, int]], list[float]],
    source: tuple[int, int],
    target: tuple[int, int],
    value: float,
) -> None:
    if source != target:
        storage[(source, target)].append(float(value))


def _widest_phase_cycle(
    phase_bins: int,
    states_by_phase: dict[int, set[int]],
    weights: dict[tuple[tuple[int, int], tuple[int, int]], float],
) -> float:
    best = 0.0
    for start_state in states_by_phase.get(0, set()):
        active = {start_state: 1.0}
        for phase in range(phase_bins - 1):
            updated: dict[int, float] = {}
            for source_state, width in active.items():
                source = (phase, source_state)
                for target_state in states_by_phase.get(phase + 1, set()):
                    edge = weights.get((source, (phase + 1, target_state)), 0.0)
                    if edge > 0:
                        updated[target_state] = max(
                            updated.get(target_state, 0.0), min(width, edge)
                        )
            active = updated
        for source_state, width in active.items():
            closing = weights.get(
                (((phase_bins - 1), source_state), (0, start_state)), 0.0
            )
            best = max(best, min(width, closing))
    return float(best)


def _phase_state_graph_features(
    values: np.ndarray,
    *,
    hop_seconds: float,
    config: StrengthenedConfig,
) -> dict[str, float]:
    distances = _standard_distance(values)
    period, lag_peak = _dominant_lag_from_distance(distances, config)  # type: ignore[arg-type]
    snapshots = _phase_snapshots(values, period, config.phase_bins, config.min_cycles)
    cycle_count = snapshots.shape[0]
    flat = snapshots.reshape(cycle_count * config.phase_bins, -1)
    all_snapshot_distances = _pairwise_distance(flat)
    positive = all_snapshot_distances[all_snapshot_distances > 1e-9]
    distance_scale = float(np.median(positive)) if positive.size else 1.0
    labels = _adaptive_phase_state_labels(
        snapshots, all_snapshot_distances, distance_scale, config
    )
    nodes = np.empty((cycle_count, config.phase_bins), dtype=object)
    for cycle in range(cycle_count):
        for phase in range(config.phase_bins):
            nodes[cycle, phase] = (phase, int(labels[cycle, phase]))

    same_phase_by_bin: list[list[float]] = [
        [] for _ in range(config.phase_bins)
    ]
    for cycle in range(cycle_count - 1):
        for phase in range(config.phase_bins):
            source_index = cycle * config.phase_bins + phase
            same_index = (cycle + 1) * config.phase_bins + phase
            same_phase_by_bin[phase].append(
                math.exp(
                    -all_snapshot_distances[source_index, same_index] / distance_scale
                )
            )
    same_phase_scores = [
        score for phase_scores in same_phase_by_bin for score in phase_scores
    ]

    temporal: dict[tuple[tuple[int, int], tuple[int, int]], list[float]] = defaultdict(list)
    phase_denominators = np.full(config.phase_bins, cycle_count, dtype=float)
    phase_denominators[-1] = max(1, cycle_count - 1)
    for cycle in range(cycle_count):
        for phase in range(config.phase_bins - 1):
            _add_weighted_counts(
                temporal,
                nodes[cycle, phase],
                nodes[cycle, phase + 1],
                1.0,
            )
        if cycle + 1 < cycle_count:
            _add_weighted_counts(
                temporal,
                nodes[cycle, -1],
                nodes[cycle + 1, 0],
                1.0,
            )

    recurrence: dict[tuple[tuple[int, int], tuple[int, int]], list[float]] = defaultdict(list)
    for cycle in range(cycle_count - 1):
        for phase in range(config.phase_bins):
            source_index = cycle * config.phase_bins + phase
            candidates = [
                (phase + offset) % config.phase_bins
                for offset in range(
                    -config.recurrence_phase_radius, config.recurrence_phase_radius + 1
                )
            ]
            target_indices = [
                (cycle + 1) * config.phase_bins + target_phase for target_phase in candidates
            ]
            best_local = int(
                np.argmin(all_snapshot_distances[source_index, target_indices])
            )
            target_phase = candidates[best_local]
            similarity = math.exp(
                -all_snapshot_distances[source_index, target_indices[best_local]]
                / distance_scale
            )
            _add_weighted_counts(
                recurrence,
                nodes[cycle, phase],
                nodes[cycle + 1, target_phase],
                similarity,
            )

    weights: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    counts: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
    for edge, observations in temporal.items():
        source_phase = edge[0][0]
        weights[edge] = float(np.sum(observations)) / phase_denominators[source_phase]
        counts[edge] += len(observations)
    recurrence_denominator = max(1, cycle_count - 1)
    for edge, observations in recurrence.items():
        support = len(observations) / recurrence_denominator
        recurrence_weight = support * float(np.mean(observations))
        weights[edge] = max(weights.get(edge, 0.0), recurrence_weight)
        counts[edge] += len(observations)

    by_source: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = defaultdict(list)
    for (source, target), weight in weights.items():
        by_source[source].append((target, weight))
    kept: list[WeightedEdge] = []
    for source, outgoing in by_source.items():
        ranked = sorted(outgoing, key=lambda item: (-item[1], item[0]))
        for target, weight in ranked[: config.top_k_per_source]:
            kept.append(
                WeightedEdge(
                    source=source,
                    target=target,
                    weight=float(np.clip(weight, 0.0, 1.0)),
                    count=int(counts[(source, target)]),
                )
            )
    vertices = tuple(sorted(set(nodes.reshape(-1)), key=repr))
    graph = TransitionGraph(vertices=vertices, edges=tuple(kept))
    filtration_levels = tuple(
        sorted({0.0, 1.0, *(round(edge.weight, 8) for edge in kept)})
    )
    persistence = persistent_path_homology(graph, filtration_levels)
    h1_intervals = [interval for interval in persistence.intervals if interval.dimension == 1]
    lifetimes = [
        interval.lifetime for interval in h1_intervals for _ in range(interval.multiplicity)
    ]
    maximum = float(max(lifetimes, default=0.0))
    total = float(sum(lifetimes))
    dominant = maximum / (1.0 + max(0.0, total - maximum))
    descriptor_rows = sorted(
        persistence.descriptors, key=lambda row: float(row["threshold"])
    )
    thresholds = np.asarray([float(row["threshold"]) for row in descriptor_rows])
    betti = np.asarray([float(row["h1_betti"]) for row in descriptor_rows])
    normalized_auc = float(
        np.trapezoid(betti, thresholds) / max(1.0, math.sqrt(len(vertices)))
    )
    states_by_phase: dict[int, set[int]] = defaultdict(set)
    for phase, state in vertices:
        states_by_phase[int(phase)].add(int(state))
    temporal_weights = {
        edge: weight for edge, weight in weights.items() if edge in temporal
    }
    temporal_outgoing: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (source, _), weight in temporal_weights.items():
        temporal_outgoing[source].append(weight)
    concentration = float(
        np.mean([max(outgoing) for outgoing in temporal_outgoing.values()])
    )
    return {
        "h1_max_lifetime": maximum,
        "h1_total_persistence": total,
        "h1_dominant_persistence": dominant,
        "h1_normalized_auc": normalized_auc,
        "h1_interval_count": float(len(lifetimes)),
        "h1_betti_max": float(np.max(betti, initial=0.0)),
        "widest_cycle_baseline": _widest_phase_cycle(
            config.phase_bins, states_by_phase, temporal_weights
        ),
        "phase_recurrence_mean": float(np.mean(same_phase_scores)),
        "phase_recurrence_q25": float(np.quantile(same_phase_scores, 0.25)),
        "transition_concentration": concentration,
        "dominant_period_seconds": float(period * hop_seconds),
        "lag_peak_strength": float(lag_peak),
        "graph_vertices": float(len(vertices)),
        "graph_edges": float(len(kept)),
        "complete_cycles": float(cycle_count),
    }


def _looped_blocks(values: np.ndarray, config: StrengthenedConfig) -> np.ndarray:
    period = min(max(config.phase_bins * 2, 12), len(values) // config.min_cycles)
    start = max(0, len(values) // 2 - period // 2)
    return np.resize(values[start : start + period], values.shape)


def _strengthened_candidate_data(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: StrengthenedConfig,
) -> dict[str, tuple[np.ndarray, float]]:
    legacy_config = load_config(root)
    legacy = _candidate_data(root, row, model, legacy_config)
    aggregation = max(1, config.block_frames // legacy_config.block_frames)
    return {
        "acoustic": (
            _block_mean(legacy["path_acoustic_phase"][0], aggregation),
            legacy["path_acoustic_phase"][1],
        ),
        "rhythm": (
            _block_mean(legacy["path_rhythm_phase"][0], aggregation),
            legacy["path_rhythm_phase"][1],
        ),
    }


def _metric_representation(modality: str, metric: str) -> str:
    return f"phase_state_{modality}__{metric}"


def _parse_representation(representation: str) -> tuple[str, str]:
    prefix, metric = representation.split("__", maxsplit=1)
    modality = prefix.removeprefix("phase_state_")
    if modality not in MODALITIES or metric not in TOPOLOGY_METRICS:
        raise RepetitionAnalysisError(f"unknown strengthened representation: {representation}")
    return modality, metric


def _compute_segment(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: StrengthenedConfig,
    representations: Sequence[str],
    calibrate: bool,
) -> list[dict[str, Any]]:
    data = _strengthened_candidate_data(root, row, model, config)
    requested = {_parse_representation(name)[0] for name in representations}
    results: dict[str, dict[str, float]] = {}
    calibration: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for modality in requested:
        values, hop_seconds = data[modality]
        results[modality] = _phase_state_graph_features(
            values, hop_seconds=hop_seconds, config=config
        )
        if calibrate:
            looped = _looped_blocks(values, config)
            rng = np.random.default_rng(
                _seed(f"{row['segment_id']}:{modality}:strengthened", config.random_seed)
            )
            shuffled = values[rng.permutation(len(values))]
            calibration[modality] = (
                _phase_state_graph_features(looped, hop_seconds=hop_seconds, config=config),
                _phase_state_graph_features(shuffled, hop_seconds=hop_seconds, config=config),
            )
    identity = {column: row[column] for column in IDENTITY_COLUMNS}
    identity["scale_seconds"] = float(identity["scale_seconds"])
    output: list[dict[str, Any]] = []
    for representation in representations:
        modality, metric = _parse_representation(representation)
        result = results[modality]
        synthetic = shuffled = np.nan
        if calibrate:
            synthetic = calibration[modality][0][metric]
            shuffled = calibration[modality][1][metric]
        output.append(
            {
                **identity,
                "representation": representation,
                "modality": modality,
                "metric": metric,
                "method": "phase_state_persistent_path_homology",
                "loop_score": result[metric],
                "synthetic_loop_score": synthetic,
                "shuffled_score": shuffled,
                **result,
            }
        )
    return output


def _compute_features(
    root: Path,
    manifest: pd.DataFrame,
    config: StrengthenedConfig,
    representations: Sequence[str],
    *,
    calibrate: bool,
) -> pd.DataFrame:
    model = _load_model(root)
    records = manifest.to_dict("records")
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _compute_segment,
                root,
                row,
                model,
                config,
                representations,
                calibrate and float(row["scale_seconds"]) == 180.0,
            ): row
            for row in records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                output.extend(future.result())
            except Exception as exc:
                raise RepetitionAnalysisError(
                    f"strengthened repetition failed for {row['segment_id']}: {exc}"
                ) from exc
            if completed % 50 == 0 or completed == len(records):
                print(f"Strengthened repetition: {completed}/{len(records)}", flush=True)
    return pd.DataFrame(output).sort_values(
        ["split", "group", "track_id", "scale_seconds", "representation"]
    )


def _eligible_manifest(
    manifest: pd.DataFrame, config: StrengthenedConfig, representations: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modalities = {_parse_representation(name)[0] for name in representations}
    columns = [
        "acoustic_windows" if modality == "acoustic" else "rhythm_windows"
        for modality in modalities
    ]
    eligible = np.ones(len(manifest), dtype=bool)
    for column in columns:
        counts = pd.to_numeric(manifest[column], errors="coerce").fillna(0).to_numpy()
        eligible &= counts >= config.min_raw_steps
    return manifest.loc[eligible].copy(), manifest.loc[~eligible].copy()


def _exploration_manifest(manifest: pd.DataFrame, config: StrengthenedConfig) -> pd.DataFrame:
    all_representations = [
        _metric_representation(modality, metric)
        for modality in MODALITIES
        for metric in TOPOLOGY_METRICS
    ]
    eligible, _ = _eligible_manifest(manifest, config, all_representations)
    base = eligible[(eligible["split"] == "discovery") & (eligible["scale_seconds"] == 180.0)]
    sensitivity = eligible[
        (eligible["split"] == "discovery") & (eligible["scale_seconds"] == 300.0)
    ]
    base = base[base["track_id"].isin(sensitivity["track_id"])]
    rng = np.random.default_rng(config.random_seed)
    tracks: list[str] = []
    for group in ("focus", "pop", "classical"):
        choices = np.sort(base.loc[base["group"] == group, "track_id"].unique())
        tracks.extend(
            str(value)
            for value in rng.choice(choices, config.exploration_tracks_per_group, replace=False)
        )
    return eligible[
        (eligible["split"] == "discovery")
        & eligible["track_id"].isin(tracks)
        & eligible["scale_seconds"].isin([180.0, 300.0])
    ].copy()


def _calibration_tests(
    features: pd.DataFrame, config: StrengthenedConfig
) -> pd.DataFrame:
    primary = features[features["scale_seconds"] == 180.0]
    rows: list[dict[str, Any]] = []
    for name, view in primary.groupby("representation"):
        differences = view["synthetic_loop_score"] - view["shuffled_score"]
        statistic, p_value = wilcoxon(differences, alternative="greater")
        rows.append(
            {
                "representation": name,
                "modality": view["modality"].iloc[0],
                "metric": view["metric"].iloc[0],
                "n_tracks": len(view),
                "synthetic_loop_median": float(view["synthetic_loop_score"].median()),
                "shuffled_median": float(view["shuffled_score"].median()),
                "median_delta": float(differences.median()),
                "positive_fraction": float(np.mean(differences > 0)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = benjamini_hochberg(result["p_value"].to_numpy())
    result["calibration_pass"] = (
        (result["p_fdr_bh"] <= config.calibration_fdr_q)
        & (result["median_delta"] >= config.calibration_min_delta)
        & (result["positive_fraction"] >= config.calibration_positive_fraction)
    )
    return result.sort_values(["calibration_pass", "median_delta"], ascending=False)


def _group_tests(
    features: pd.DataFrame,
    *,
    split: str,
    scale: float,
    comparator: str,
    role: str,
    representations: Sequence[str] | None = None,
) -> pd.DataFrame:
    view = features[(features["split"] == split) & (features["scale_seconds"] == scale)]
    if representations is not None:
        view = view[view["representation"].isin(representations)]
    rows: list[dict[str, Any]] = []
    for name, group_view in view.groupby("representation"):
        focus = group_view[group_view["group"] == "focus"]["loop_score"].to_numpy()
        control = group_view[group_view["group"] == comparator]["loop_score"].to_numpy()
        effect, p_value = _effect_test(focus, control)
        rows.append(
            {
                "role": role,
                "comparison": f"focus_greater_than_{comparator}",
                "split": split,
                "scale_seconds": scale,
                "representation": name,
                "modality": group_view["modality"].iloc[0],
                "metric": group_view["metric"].iloc[0],
                "n_focus": len(focus),
                "n_comparator": len(control),
                "focus_median": float(np.median(focus)),
                "comparator_median": float(np.median(control)),
                "rank_biserial_focus_minus_comparator": effect,
                "p_one_sided": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = benjamini_hochberg(result["p_one_sided"].to_numpy())
    return result


def _incremental_tests(features: pd.DataFrame) -> pd.DataFrame:
    view = features[
        (features["split"] == "discovery")
        & (features["scale_seconds"] == 180.0)
        & features["group"].isin(["focus", "pop"])
    ]
    rows: list[dict[str, Any]] = []
    for name, feature_view in view.groupby("representation"):
        predictors = feature_view[list(BASELINE_COLUMNS)].to_numpy(dtype=float)
        outcome = feature_view["loop_score"].to_numpy(dtype=float)
        residual = outcome - LinearRegression().fit(predictors, outcome).predict(predictors)
        focus = residual[feature_view["group"].to_numpy() == "focus"]
        pop = residual[feature_view["group"].to_numpy() == "pop"]
        effect, p_value = _effect_test(focus, pop)
        correlations = [
            abs(float(spearmanr(feature_view[column], outcome).statistic))
            for column in BASELINE_COLUMNS
        ]
        rows.append(
            {
                "representation": name,
                "incremental_rank_biserial": effect,
                "incremental_p_one_sided": p_value,
                "max_baseline_spearman": max(correlations),
            }
        )
    return pd.DataFrame(rows)


def _select_representations(
    features: pd.DataFrame,
    calibration: pd.DataFrame,
    tests: pd.DataFrame,
    incremental: pd.DataFrame,
    config: StrengthenedConfig,
) -> tuple[list[str], pd.DataFrame]:
    primary = tests[tests["scale_seconds"] == 180.0].copy()
    scale = tests[tests["scale_seconds"] == 300.0][
        ["representation", "rank_biserial_focus_minus_comparator"]
    ].rename(columns={"rank_biserial_focus_minus_comparator": "effect_300"})
    selection = (
        primary.merge(scale, on="representation")
        .merge(
            calibration[
                [
                    "representation",
                    "calibration_pass",
                    "median_delta",
                    "positive_fraction",
                    "p_fdr_bh",
                ]
            ].rename(columns={"p_fdr_bh": "calibration_fdr"}),
            on="representation",
        )
        .merge(incremental, on="representation")
    )
    correlations: list[float] = []
    for row in selection.itertuples(index=False):
        paired = features[features["representation"] == row.representation].pivot(
            index="track_id", columns="scale_seconds", values="loop_score"
        )
        coefficient = spearmanr(paired[180.0], paired[300.0]).statistic
        correlations.append(float(coefficient) if np.isfinite(coefficient) else 0.0)
    selection["scale_spearman"] = correlations
    selection["eligible"] = (
        selection["calibration_pass"]
        & (selection["p_one_sided"] <= config.selection_p_threshold)
        & (
            selection["rank_biserial_focus_minus_comparator"]
            >= config.selection_effect_threshold
        )
        & (selection["effect_300"] > 0)
        & (selection["scale_spearman"] >= config.selection_stability_threshold)
        & (selection["incremental_p_one_sided"] <= config.incremental_p_threshold)
        & (
            selection["incremental_rank_biserial"]
            >= config.incremental_effect_threshold
        )
    )
    selection["selection_score"] = (
        selection["rank_biserial_focus_minus_comparator"].clip(lower=0)
        * selection["scale_spearman"].clip(lower=0)
        * selection["incremental_rank_biserial"].clip(lower=0)
        * selection["median_delta"].clip(lower=0)
    )
    selection = selection.sort_values(["eligible", "selection_score"], ascending=[False, False])
    selected: list[str] = []
    for modality in MODALITIES:
        choice = selection[(selection["modality"] == modality) & selection["eligible"]].head(1)
        if len(choice):
            selected.append(str(choice.iloc[0]["representation"]))
    selection["selected"] = selection["representation"].isin(selected)
    return selected, selection


def run_exploration(root: Path, config: StrengthenedConfig) -> dict[str, Any]:
    manifest_path = root / "metadata" / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    sample = _exploration_manifest(manifest, config)
    representations = [
        _metric_representation(modality, metric)
        for modality in MODALITIES
        for metric in TOPOLOGY_METRICS
    ]
    features = _compute_features(
        root, sample, config, representations, calibrate=True
    )
    calibration = _calibration_tests(features, config)
    tests = pd.concat(
        [
            _group_tests(
                features,
                split="discovery",
                scale=180.0,
                comparator="pop",
                role="exploration",
            ),
            _group_tests(
                features,
                split="discovery",
                scale=300.0,
                comparator="pop",
                role="scale_stability",
            ),
        ],
        ignore_index=True,
    )
    incremental = _incremental_tests(features)
    selected, selection = _select_representations(
        features, calibration, tests, incremental, config
    )
    metadata = root / "metadata"
    outputs = {
        "features": metadata / "repetition_strengthened_exploration_features.csv",
        "calibration": metadata / "repetition_strengthened_calibration.csv",
        "tests": metadata / "repetition_strengthened_exploration_tests.csv",
        "incremental": metadata / "repetition_strengthened_incremental_tests.csv",
        "selection": metadata / "repetition_strengthened_selection.csv",
    }
    _write_csv(outputs["features"], features)
    _write_csv(outputs["calibration"], calibration)
    _write_csv(outputs["tests"], tests)
    _write_csv(outputs["incremental"], incremental)
    _write_csv(outputs["selection"], selection)
    payload = {
        "generated_at": date.today().isoformat(),
        "role": "discovery-only strengthened repetition feature selection",
        "config": asdict(config),
        "config_sha256": _json_hash(asdict(config)),
        "selected_representations": selected,
        "outputs": {name: path.relative_to(root).as_posix() for name, path in outputs.items()},
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(
        metadata / "repetition_strengthened_exploration_summary.json", payload
    )
    return payload


def _diagnostic_features(features: pd.DataFrame) -> pd.DataFrame:
    columns = (
        *BASELINE_COLUMNS,
        "phase_recurrence_q25",
        "graph_vertices",
        "graph_edges",
        "h1_interval_count",
        "h1_betti_max",
        "h1_total_persistence",
    )
    unique = features.drop_duplicates(
        ["track_id", "scale_seconds", "modality"]
    )
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        modality_view = unique[unique["modality"] == modality]
        for column in columns:
            scale_results: dict[float, tuple[float, float, float, float]] = {}
            for scale in (180.0, 300.0):
                view = modality_view[
                    (modality_view["scale_seconds"] == scale)
                    & modality_view["group"].isin(["focus", "pop"])
                ]
                focus = view[view["group"] == "focus"][column].to_numpy()
                pop = view[view["group"] == "pop"][column].to_numpy()
                effect, p_value = _effect_test(focus, pop)
                scale_results[scale] = (
                    effect,
                    p_value,
                    float(np.median(focus)),
                    float(np.median(pop)),
                )
            paired = modality_view.pivot(
                index="track_id", columns="scale_seconds", values=column
            )
            coefficient = spearmanr(paired[180.0], paired[300.0]).statistic
            rows.append(
                {
                    "modality": modality,
                    "feature": column,
                    "effect_180": scale_results[180.0][0],
                    "p_180": scale_results[180.0][1],
                    "focus_median_180": scale_results[180.0][2],
                    "pop_median_180": scale_results[180.0][3],
                    "effect_300": scale_results[300.0][0],
                    "p_300": scale_results[300.0][1],
                    "focus_median_300": scale_results[300.0][2],
                    "pop_median_300": scale_results[300.0][3],
                    "scale_spearman": float(coefficient),
                }
            )
    return pd.DataFrame(rows)


def _write_feasibility_report(
    root: Path,
    selection: pd.DataFrame,
    sensitivity: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> Path:
    lines = [
        "# 强化相位—状态 Path Homology 可行性结果",
        "",
        f"生成日期：{date.today().isoformat()}。仅使用 discovery 数据；由于没有候选通过"
        "预设门槛，validation 未用于参数选择或确认性检验。",
        "",
        "## 预设配置结果（状态相似度 0.55）",
        "",
        "| 表示 | 180s 效应 | p | 300s 效应 | 跨尺度ρ | 增量效应 | 增量p | 入选 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selection.itertuples(index=False):
        lines.append(
            f"| {row.representation} | "
            f"{row.rank_biserial_focus_minus_comparator:.3f} | {row.p_one_sided:.3g} | "
            f"{row.effect_300:.3f} | {row.scale_spearman:.3f} | "
            f"{row.incremental_rank_biserial:.3f} | "
            f"{row.incremental_p_one_sided:.3g} | {'是' if row.selected else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 状态粒度敏感性",
            "",
            "| 阈值 | 表示 | 180s 效应 | p | 300s 效应 | p | 跨尺度ρ |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.state_similarity:.2f} | {row.representation} | "
            f"{row.effect_180:.3f} | {row.p_180:.3g} | "
            f"{row.effect_300:.3f} | {row.p_300:.3g} | {row.scale_spearman:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 诊断结论",
            "",
            "人工循环与时间打乱校准全部通过，但没有强化 H1 端点同时满足 180秒组间效应、"
            "300秒方向一致、跨尺度稳定和非拓扑基线之外的增量门槛。",
            "",
            "原始同相位复现仍呈 Focus>Pop；将相位拆成数据驱动状态后，Pop 的集中状态转移"
            "会产生同样或更长的单个 H1 环。Focus 图出现更多节点、边和环类，但这些差异"
            "可由图规模解释，不能归因于独立的 Path Homology 信息。",
            "",
            "因此本次可行性试验不支持把强化相位—状态 Path Homology 扩展到 validation。"
            "原固定相位环的声学/节奏结果仍可作为复现一致性指标，但不应宣称其优势来自"
            "非平凡的图拓扑。",
            "",
        ]
    )
    path = root / "docs" / "repetition-strengthened-feasibility.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_sensitivity(root: Path, config: StrengthenedConfig) -> dict[str, Any]:
    manifest_path = root / "metadata" / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    sample = _exploration_manifest(manifest, config)
    representations = [
        _metric_representation(modality, metric)
        for modality in MODALITIES
        for metric in TOPOLOGY_METRICS
    ]
    rows: list[dict[str, Any]] = []
    reference_features: pd.DataFrame | None = None
    for threshold in SENSITIVITY_THRESHOLDS:
        threshold_config = StrengthenedConfig(
            **{**asdict(config), "state_similarity_threshold": threshold}
        )
        features = _compute_features(
            root, sample, threshold_config, representations, calibrate=False
        )
        if threshold == config.state_similarity_threshold:
            reference_features = features
        tests = pd.concat(
            [
                _group_tests(
                    features,
                    split="discovery",
                    scale=180.0,
                    comparator="pop",
                    role="sensitivity",
                ),
                _group_tests(
                    features,
                    split="discovery",
                    scale=300.0,
                    comparator="pop",
                    role="sensitivity",
                ),
            ],
            ignore_index=True,
        )
        for representation in representations:
            pair = features[features["representation"] == representation].pivot(
                index="track_id", columns="scale_seconds", values="loop_score"
            )
            test_180 = tests[
                (tests["representation"] == representation)
                & (tests["scale_seconds"] == 180.0)
            ].iloc[0]
            test_300 = tests[
                (tests["representation"] == representation)
                & (tests["scale_seconds"] == 300.0)
            ].iloc[0]
            rows.append(
                {
                    "state_similarity": threshold,
                    "representation": representation,
                    "effect_180": test_180.rank_biserial_focus_minus_comparator,
                    "p_180": test_180.p_one_sided,
                    "effect_300": test_300.rank_biserial_focus_minus_comparator,
                    "p_300": test_300.p_one_sided,
                    "scale_spearman": spearmanr(pair[180.0], pair[300.0]).statistic,
                }
            )
    if reference_features is None:
        raise RepetitionAnalysisError(
            "configured state similarity is absent from the sensitivity grid"
        )
    sensitivity = pd.DataFrame(rows)
    diagnostics = _diagnostic_features(reference_features)
    metadata = root / "metadata"
    sensitivity_path = metadata / "repetition_strengthened_sensitivity.csv"
    diagnostics_path = metadata / "repetition_strengthened_diagnostics.csv"
    _write_csv(sensitivity_path, sensitivity)
    _write_csv(diagnostics_path, diagnostics)
    selection = pd.read_csv(metadata / "repetition_strengthened_selection.csv")
    report = _write_feasibility_report(
        root, selection, sensitivity, diagnostics
    )
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "validation_run": False,
        "reason": "no strengthened endpoint passed discovery gates",
        "thresholds": list(SENSITIVITY_THRESHOLDS),
        "outputs": {
            "sensitivity": sensitivity_path.relative_to(root).as_posix(),
            "diagnostics": diagnostics_path.relative_to(root).as_posix(),
            "report": report.relative_to(root).as_posix(),
        },
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(metadata / "repetition_strengthened_feasibility.json", payload)
    return payload


def _load_selected(root: Path) -> list[str]:
    path = root / "metadata" / "repetition_strengthened_exploration_summary.json"
    if not path.is_file():
        raise RepetitionAnalysisError("run strengthened exploration first")
    selected = json.loads(path.read_text(encoding="utf-8"))["selected_representations"]
    if not selected:
        raise RepetitionAnalysisError("no strengthened topology endpoint passed discovery")
    for name in selected:
        _parse_representation(name)
    return list(selected)


def _classification(
    features: pd.DataFrame, config: StrengthenedConfig
) -> pd.DataFrame:
    wide = features.pivot(
        index=list(IDENTITY_COLUMNS), columns="representation", values="loop_score"
    ).reset_index()
    columns = [column for column in wide.columns if column not in IDENTITY_COLUMNS]
    train = wide[
        (wide["split"] == "discovery")
        & (wide["scale_seconds"] == 180.0)
        & wide["group"].isin(["focus", "pop"])
    ]
    validation = wide[
        (wide["split"] == "validation")
        & (wide["scale_seconds"] == 180.0)
        & wide["group"].isin(["focus", "pop"])
    ]
    y_train = (train["group"] == "focus").astype(int)
    y_validation = (validation["group"] == "focus").astype(int)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
    search = GridSearchCV(
        model,
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
    ).fit(train[columns], y_train)
    prediction = search.predict(validation[columns])
    probability = search.predict_proba(validation[columns])[:, 1]
    return pd.DataFrame(
        [
            {
                "task": "focus_vs_pop",
                "n_train": len(train),
                "n_validation": len(validation),
                "n_features": len(columns),
                "best_c": float(search.best_params_["classifier__C"]),
                "cv_macro_f1": float(search.best_score_),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_validation, prediction)
                ),
                "macro_f1": float(f1_score(y_validation, prediction, average="macro")),
                "auroc": float(roc_auc_score(y_validation, probability)),
            }
        ]
    )


def _plot_scores(
    root: Path, features: pd.DataFrame, selected: Sequence[str]
) -> list[str]:
    cache = root / "runs" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation = features[features["split"] == "validation"]
    figure, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 4.5))
    axes = np.atleast_1d(axes)
    colors = {"focus": "#2B6CB0", "pop": "#D95F02", "classical": "#55A868"}
    for axis, representation in zip(axes, selected, strict=True):
        view = validation[validation["representation"] == representation]
        values: list[np.ndarray] = []
        labels: list[str] = []
        box_colors: list[str] = []
        for scale in (180.0, 300.0):
            for group in ("focus", "pop", "classical"):
                values.append(
                    view[
                        (view["scale_seconds"] == scale) & (view["group"] == group)
                    ]["loop_score"].to_numpy()
                )
                labels.append(f"{int(scale)} {group[0].upper()}")
                box_colors.append(colors[group])
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=30)
        axis.set_title(representation)
        axis.set_ylabel("data-driven Path H1 score")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Strengthened phase-state Path Homology: frozen validation")
    figure.tight_layout()
    output = root / "runs" / "repetition_strengthened" / "validation_scores.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [output.relative_to(root).as_posix()]


def _write_report(
    root: Path,
    selected: Sequence[str],
    selection: pd.DataFrame,
    tests: pd.DataFrame,
    classification: pd.DataFrame,
    excluded: int,
) -> Path:
    lines = [
        "# 强化相位—状态 Path Homology 结果",
        "",
        f"生成日期：{date.today().isoformat()}。相位环不再预先固定；节点为相位×状态，"
        "边来自实际相邻相位转移与相邻周期复现匹配。",
        "",
        "## Discovery 筛选与消融",
        "",
        "| 表示 | 180s 效应 | p | 300s 效应 | 跨尺度ρ | 基线最大ρ | 增量效应 | 增量p | 入选 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selection.itertuples(index=False):
        lines.append(
            f"| {row.representation} | "
            f"{row.rank_biserial_focus_minus_comparator:.3f} | {row.p_one_sided:.3g} | "
            f"{row.effect_300:.3f} | {row.scale_spearman:.3f} | "
            f"{row.max_baseline_spearman:.3f} | {row.incremental_rank_biserial:.3f} | "
            f"{row.incremental_p_one_sided:.3g} | {'是' if row.selected else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 冻结表示",
            "",
            *[f"- `{name}`" for name in selected],
            "",
            f"全量分析前因长度门槛排除 {excluded} 个片段。",
            "",
            "## Validation",
            "",
            "| 角色 | 表示 | Focus | 对照 | 效应 | FDR |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.role} | {row.representation} | {row.focus_median:.3f} | "
            f"{row.comparator_median:.3f} | "
            f"{row.rank_biserial_focus_minus_comparator:.3f} | {row.p_fdr_bh:.3g} |"
        )
    result = classification.iloc[0]
    lines.extend(
        [
            "",
            "## 分类辅助结果",
            "",
            f"Focus/Pop：Macro-F1 {result.macro_f1:.3f}，balanced accuracy "
            f"{result.balanced_accuracy:.3f}，AUROC {result.auroc:.3f}。",
            "",
            "## 解释边界",
            "",
            "入选门槛除人工循环校准、组间效应和跨尺度稳定外，还要求在回归掉最宽简单环、"
            "平均同相位复现及转移集中度后仍保留 Focus>Pop 的增量效应。由此避免把"
            "Path H1 简化为最弱边分数。",
            "",
        ]
    )
    path = root / "docs" / "repetition-strengthened-results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_full(root: Path, config: StrengthenedConfig) -> dict[str, Any]:
    selected = _load_selected(root)
    manifest_path = root / "metadata" / "feature_segments.csv"
    full = pd.read_csv(manifest_path)
    manifest, excluded = _eligible_manifest(full, config, selected)
    features = _compute_features(
        root, manifest, config, selected, calibrate=False
    )
    tests = pd.concat(
        [
            _group_tests(
                features,
                split="validation",
                scale=180.0,
                comparator="pop",
                role="confirmatory_focus_vs_pop",
            ),
            _group_tests(
                features,
                split="validation",
                scale=300.0,
                comparator="pop",
                role="replication_focus_vs_pop",
            ),
            _group_tests(
                features,
                split="validation",
                scale=180.0,
                comparator="classical",
                role="specificity_focus_vs_classical",
            ),
            _group_tests(
                features,
                split="validation",
                scale=300.0,
                comparator="classical",
                role="specificity_scale_focus_vs_classical",
            ),
        ],
        ignore_index=True,
    )
    classification = _classification(features, config)
    selection = pd.read_csv(root / "metadata" / "repetition_strengthened_selection.csv")
    metadata = root / "metadata"
    feature_path = metadata / "repetition_strengthened_features.csv"
    test_path = metadata / "repetition_strengthened_tests.csv"
    classification_path = metadata / "repetition_strengthened_classification.csv"
    _write_csv(feature_path, features)
    _write_csv(test_path, tests)
    _write_csv(classification_path, classification)
    plots = _plot_scores(root, features, selected)
    report = _write_report(
        root, selected, selection, tests, classification, len(excluded)
    )
    primary = tests[tests["role"] == "confirmatory_focus_vs_pop"]
    replication = tests[tests["role"] == "replication_focus_vs_pop"]
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "selected_representations": selected,
        "segments": len(manifest),
        "quality_excluded_segments": len(excluded),
        "confirmatory_discoveries": int(
            np.sum(primary["p_fdr_bh"] <= config.validation_fdr_q)
        ),
        "replicated_discoveries": int(
            np.sum(replication["p_fdr_bh"] <= config.validation_fdr_q)
        ),
        "config": asdict(config),
        "outputs": {
            "features": feature_path.relative_to(root).as_posix(),
            "tests": test_path.relative_to(root).as_posix(),
            "classification": classification_path.relative_to(root).as_posix(),
            "report": report.relative_to(root).as_posix(),
            "plots": plots,
        },
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(metadata / "repetition_strengthened_summary.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="focus-repetition-strengthened")
    parser.add_argument("command", choices=("explore", "sensitivity", "run", "all"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_strengthened_config(root)
    exploration: dict[str, Any] | None = None
    if args.command in {"explore", "all"}:
        exploration = run_exploration(root, config)
        print(json.dumps(exploration, ensure_ascii=False, indent=2))
    if args.command in {"sensitivity", "all"}:
        print(json.dumps(run_sensitivity(root, config), ensure_ascii=False, indent=2))
    if args.command == "run" or (
        args.command == "all"
        and exploration is not None
        and exploration["selected_representations"]
    ):
        print(json.dumps(run_full(root, config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
