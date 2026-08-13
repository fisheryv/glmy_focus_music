from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyglmy import vietoris_rips
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features.batch import _json_hash, _read_npz, _sha256, _write_json_atomic
from graphs.transition import TransitionGraph, WeightedEdge
from homology.glmy import persistent_path_homology
from topology.statistics import benjamini_hochberg

SW_REPRESENTATIONS = (
    "sw_acoustic_novelty",
    "sw_loudness",
    "sw_onset",
    "sw_tonal_novelty",
    "sw_modulation",
)
PATH_REPRESENTATIONS = (
    "path_acoustic_phase",
    "path_rhythm_phase",
    "path_chroma_phase",
)
REPRESENTATIONS = (*SW_REPRESENTATIONS, *PATH_REPRESENTATIONS)
IDENTITY_COLUMNS = ("segment_id", "track_id", "group", "split", "scale_seconds")


class RepetitionAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RepetitionConfig:
    exploration_tracks_per_group: int = 24
    phase_bins: int = 6
    block_frames: int = 4
    min_raw_steps: int = 96
    min_sw_points: int = 48
    min_period_seconds: float = 4.0
    max_period_seconds: float = 32.0
    max_period_blocks: int = 32
    min_cycles: int = 3
    max_landmarks: int = 48
    delay_dimension: int = 8
    prominent_lifetime: float = 0.10
    path_thresholds: tuple[float, ...] = tuple(np.arange(0.05, 1.0, 0.05))
    calibration_fdr_q: float = 0.05
    calibration_min_delta: float = 0.05
    calibration_positive_fraction: float = 0.75
    selection_p_threshold: float = 0.05
    selection_effect_threshold: float = 0.30
    selection_stability_threshold: float = 0.30
    max_selected_representations: int = 3
    validation_fdr_q: float = 0.05
    workers: int = 4
    random_seed: int = 20260716

    def validate(self) -> None:
        if self.exploration_tracks_per_group < 8:
            raise RepetitionAnalysisError("exploration sample is too small")
        if self.phase_bins < 3 or self.block_frames < 1 or self.min_cycles < 3:
            raise RepetitionAnalysisError("phase-lifted graph configuration is invalid")
        if self.min_raw_steps < self.phase_bins * self.block_frames * self.min_cycles:
            raise RepetitionAnalysisError("min_raw_steps cannot support the requested cycles")
        if self.max_landmarks < 16 or self.delay_dimension < 3:
            raise RepetitionAnalysisError("sliding-window configuration is too small")
        if self.min_period_seconds >= self.max_period_seconds:
            raise RepetitionAnalysisError("period range is invalid")
        if not self.path_thresholds or any(not 0 < value < 1 for value in self.path_thresholds):
            raise RepetitionAnalysisError("path thresholds must lie in (0, 1)")
        probabilities = (
            self.calibration_fdr_q,
            self.calibration_positive_fraction,
            self.selection_p_threshold,
            self.selection_effect_threshold,
            self.selection_stability_threshold,
            self.validation_fdr_q,
        )
        if any(not 0 < value < 1 for value in probabilities):
            raise RepetitionAnalysisError("statistical thresholds must lie in (0, 1)")
        if self.calibration_min_delta <= 0 or self.workers < 1:
            raise RepetitionAnalysisError("calibration delta and workers must be positive")


def load_config(root: Path) -> RepetitionConfig:
    with (root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    values = dict(raw.get("repetition", {}))
    if "path_thresholds" in values:
        values["path_thresholds"] = tuple(float(value) for value in values["path_thresholds"])
    values.setdefault("random_seed", int(raw.get("project", {}).get("seed", 20260716)))
    unknown = set(values) - set(RepetitionConfig.__dataclass_fields__)
    if unknown:
        raise RepetitionAnalysisError(f"unknown repetition keys: {sorted(unknown)}")
    config = RepetitionConfig(**values)
    config.validate()
    return config


def _seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _smooth(values: np.ndarray, width: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < width:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def _block_mean(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    count = len(values) // size
    if count < 1:
        raise RepetitionAnalysisError("sequence is shorter than one aggregation block")
    return values[: count * size].reshape(count, size, -1).mean(axis=1)


def _standard_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standardized = (values - np.median(values, axis=0)) / np.maximum(np.std(values, axis=0), 1e-8)
    return np.linalg.norm(standardized[:, None] - standardized[None, :], axis=2) / math.sqrt(
        standardized.shape[1]
    )


def transposition_invariant_chroma_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-8)
    similarities = np.stack([values @ np.roll(values, shift, axis=1).T for shift in range(12)])
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.max(similarities, axis=0)))


def _dominant_lag_from_distance(
    distances: np.ndarray, config: RepetitionConfig
) -> tuple[int, float]:
    count = len(distances)
    maximum = min(config.max_period_blocks, count // config.min_cycles)
    lags = np.arange(config.phase_bins, maximum + 1)
    if lags.size == 0:
        raise RepetitionAnalysisError("sequence cannot support a phase cycle")
    scores = np.asarray([np.median(np.diag(distances, k=int(lag))) for lag in lags])
    index = int(np.argmin(scores))
    return int(lags[index]), float(np.median(scores) - scores[index])


def _path_cycle_features(
    values: np.ndarray,
    *,
    hop_seconds: float,
    transposition_invariant: bool,
    config: RepetitionConfig,
) -> dict[str, float]:
    distances = (
        transposition_invariant_chroma_distance(values)
        if transposition_invariant
        else _standard_distance(values)
    )
    upper = distances[np.triu_indices(len(distances), k=3)]
    positive = upper[upper > 1e-9]
    scale = float(np.median(positive)) if positive.size else 1.0
    period, lag_peak = _dominant_lag_from_distance(distances, config)
    recurrence = np.exp(-np.diag(distances, k=period) / max(scale, 1e-8))
    phase = np.arange(len(recurrence)) % period * config.phase_bins // period
    coherence = np.asarray(
        [np.mean(recurrence[phase == index]) for index in range(config.phase_bins)]
    )
    edge_weights = np.minimum(coherence, np.roll(coherence, -1))
    edges = tuple(
        WeightedEdge(
            source=index,
            target=(index + 1) % config.phase_bins,
            weight=float(edge_weights[index]),
            count=int(np.count_nonzero(phase == index)),
        )
        for index in range(config.phase_bins)
    )
    graph = TransitionGraph(vertices=tuple(range(config.phase_bins)), edges=edges)
    persistence = persistent_path_homology(graph, config.path_thresholds)
    ordered = sorted(
        ((float(row["threshold"]), float(row["h1_betti"])) for row in persistence.descriptors),
        key=lambda item: item[0],
    )
    thresholds = np.asarray([item[0] for item in ordered])
    betti = np.asarray([item[1] for item in ordered])
    return {
        "loop_score": float(np.min(edge_weights)),
        "path_h1_cycle_strength": float(np.min(edge_weights)),
        "path_h1_betti_max": float(np.max(betti, initial=0.0)),
        "path_h1_betti_auc": float(np.trapezoid(betti, thresholds)),
        "phase_edge_q25": float(np.quantile(edge_weights, 0.25)),
        "phase_edge_mean": float(np.mean(edge_weights)),
        "dominant_period_seconds": float(period * hop_seconds),
        "lag_peak_strength": lag_peak,
        "h1_max_persistence": 0.0,
        "h1_best_persistence": 0.0,
    }


def _autocorrelation_period(
    values: np.ndarray, hop_seconds: float, config: RepetitionConfig
) -> tuple[int, float]:
    values = np.asarray(values, dtype=float)
    standardized = (values - np.mean(values)) / max(float(np.std(values)), 1e-8)
    minimum = max(8, int(round(config.min_period_seconds / hop_seconds)))
    maximum = min(
        len(values) // config.min_cycles,
        int(round(config.max_period_seconds / hop_seconds)),
    )
    lags = np.arange(minimum, maximum + 1)
    if lags.size == 0:
        raise RepetitionAnalysisError("scalar series cannot support a delay period")
    scores = np.asarray(
        [np.dot(standardized[:-lag], standardized[lag:]) / (len(values) - lag) for lag in lags]
    )
    index = int(np.argmax(scores))
    return int(lags[index]), float(scores[index] - np.median(scores))


def _delay_embedding(values: np.ndarray, period: int, dimension: int) -> np.ndarray:
    delay = max(1, int(round(period / dimension)))
    width = 1 + (dimension - 1) * delay
    standardized = (values - np.mean(values)) / max(float(np.std(values)), 1e-8)
    if len(values) < width + 3:
        raise RepetitionAnalysisError("series is too short for delay embedding")
    return np.column_stack(
        [
            standardized[offset : len(values) - width + 1 + offset]
            for offset in range(0, width, delay)
        ]
    )


def _farthest_landmarks(values: np.ndarray, count: int) -> np.ndarray:
    values = np.unique(np.round(np.asarray(values, dtype=float), 10), axis=0)
    if len(values) <= count:
        return values
    squared = np.sum((values[:, None] - values[None, :]) ** 2, axis=2)
    selected = [int(np.argmax(np.sum((values - values.mean(axis=0)) ** 2, axis=1)))]
    minimum = squared[selected[0]].copy()
    for _ in range(1, count):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, squared[index])
    return values[selected]


def _h1_at_period(values: np.ndarray, period: int, config: RepetitionConfig) -> float:
    cloud = _farthest_landmarks(
        _delay_embedding(values, period, config.delay_dimension), config.max_landmarks
    )
    if len(cloud) < 8:
        return 0.0
    distances = np.linalg.norm(cloud[:, None] - cloud[None, :], axis=2)
    upper = distances[np.triu_indices(len(cloud), k=1)]
    positive = upper[upper > 1e-9]
    if positive.size == 0:
        return 0.0
    distances /= np.median(positive)
    diagram = vietoris_rips(
        distances,
        distance_matrix=True,
        max_dimension=1,
    ).diagram(1)
    lifetimes = diagram[:, 1] - diagram[:, 0] if len(diagram) else np.empty(0)
    return float(np.max(lifetimes, initial=0.0))


def _sliding_window_features(
    values: np.ndarray,
    *,
    hop_seconds: float,
    config: RepetitionConfig,
    forced_period: int | None = None,
) -> dict[str, float]:
    period, lag_peak = (
        _autocorrelation_period(values, hop_seconds, config)
        if forced_period is None
        else (forced_period, 0.0)
    )
    factors = (0.8, 1.0, 1.2) if forced_period is None else (1.0,)
    scores = np.asarray(
        [_h1_at_period(values, max(8, int(round(period * factor))), config) for factor in factors]
    )
    return {
        "loop_score": float(np.median(scores)),
        "h1_max_persistence": float(np.median(scores)),
        "h1_best_persistence": float(np.max(scores, initial=0.0)),
        "dominant_period_seconds": float(period * hop_seconds),
        "lag_peak_strength": lag_peak,
        "path_h1_cycle_strength": 0.0,
        "path_h1_betti_max": 0.0,
        "path_h1_betti_auc": 0.0,
        "phase_edge_q25": 0.0,
        "phase_edge_mean": 0.0,
    }


def _load_model(root: Path) -> dict[str, np.ndarray]:
    return _read_npz(root / "features" / "models" / "state_model.npz")


def _candidate_data(
    root: Path, row: dict[str, Any], model: dict[str, np.ndarray], config: RepetitionConfig
) -> dict[str, tuple[np.ndarray, float, bool]]:
    acoustic = _read_npz(root / Path(str(row["acoustic_relative_path"])))
    rhythm = _read_npz(root / Path(str(row["rhythm_relative_path"])))
    chroma = _read_npz(root / Path(str(row["chroma_relative_path"])))
    modulation = _read_npz(root / Path(str(row["modulation_relative_path"])))

    acoustic_values = (acoustic["vectors"] - model["acoustic_mean"]) / model["acoustic_scale"]
    acoustic_pca = (acoustic_values - model["pca_mean"]) @ model["pca_components"].T
    acoustic_pca = acoustic_pca[:, :8]
    rhythm_values = np.where(rhythm["valid"], rhythm["vectors"], model["rhythm_impute"])
    rhythm_values = (rhythm_values - model["rhythm_mean"]) / model["rhythm_scale"]
    chroma_values = np.asarray(chroma["chroma"], dtype=float)
    chroma_values /= np.maximum(np.linalg.norm(chroma_values, axis=1, keepdims=True), 1e-8)
    tonal_similarity = np.stack(
        [
            np.sum(chroma_values[1:] * np.roll(chroma_values[:-1], shift, axis=1), axis=1)
            for shift in range(12)
        ]
    )
    tonal_novelty = 1.0 - np.max(tonal_similarity, axis=0)
    chroma_hop = float(np.median(np.diff(chroma["times"]))) if len(chroma["times"]) > 1 else 0.5
    return {
        "sw_acoustic_novelty": (
            _smooth(np.linalg.norm(np.diff(acoustic_pca, axis=0), axis=1)),
            0.5,
            False,
        ),
        "sw_loudness": (_smooth(np.mean(acoustic["log_mel"], axis=1)), 0.5, False),
        "sw_onset": (_smooth(rhythm_values[:, 0]), 0.5, False),
        "sw_tonal_novelty": (_smooth(tonal_novelty), chroma_hop, False),
        "sw_modulation": (_smooth(modulation["key_band_energies"][:, 0]), 2.0, False),
        "path_acoustic_phase": (
            _block_mean(acoustic_pca, config.block_frames),
            0.5 * config.block_frames,
            False,
        ),
        "path_rhythm_phase": (
            _block_mean(rhythm_values, config.block_frames),
            0.5 * config.block_frames,
            False,
        ),
        "path_chroma_phase": (
            _block_mean(chroma_values, config.block_frames),
            chroma_hop * config.block_frames,
            True,
        ),
    }


def _looped_scalar(values: np.ndarray, period: int) -> tuple[np.ndarray, int]:
    period = max(24, min(period, min(64, len(values) // 3)))
    best_score = -np.inf
    best = values[:period]
    for start in range(0, len(values) - period + 1, max(1, period // 4)):
        candidate = values[start : start + period]
        variation = float(np.std(candidate))
        closure = abs(float(candidate[0] - candidate[-1])) / max(variation, 1e-8)
        score = variation / (1.0 + closure)
        if score > best_score:
            best_score, best = score, candidate
    return np.resize(best, len(values)), period


def _looped_blocks(values: np.ndarray, config: RepetitionConfig) -> np.ndarray:
    period = min(max(config.phase_bins * 2, 12), len(values) // config.min_cycles)
    start = max(0, len(values) // 2 - period // 2)
    return np.resize(values[start : start + period], values.shape)


def _analyze_candidate(
    name: str,
    values: np.ndarray,
    hop_seconds: float,
    transposition_invariant: bool,
    config: RepetitionConfig,
) -> dict[str, float]:
    if name in PATH_REPRESENTATIONS:
        return _path_cycle_features(
            values,
            hop_seconds=hop_seconds,
            transposition_invariant=transposition_invariant,
            config=config,
        )
    return _sliding_window_features(values, hop_seconds=hop_seconds, config=config)


def _calibrate_candidate(
    name: str,
    values: np.ndarray,
    hop_seconds: float,
    transposition_invariant: bool,
    config: RepetitionConfig,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    shuffled = values[rng.permutation(len(values))]
    if name in PATH_REPRESENTATIONS:
        looped = _looped_blocks(values, config)
        loop_score = _path_cycle_features(
            looped,
            hop_seconds=hop_seconds,
            transposition_invariant=transposition_invariant,
            config=config,
        )["loop_score"]
        shuffle_score = _path_cycle_features(
            shuffled,
            hop_seconds=hop_seconds,
            transposition_invariant=transposition_invariant,
            config=config,
        )["loop_score"]
    else:
        period, _ = _autocorrelation_period(values, hop_seconds, config)
        looped, loop_period = _looped_scalar(values, period)
        loop_score = _sliding_window_features(
            looped,
            hop_seconds=hop_seconds,
            config=config,
            forced_period=loop_period,
        )["loop_score"]
        shuffle_score = _sliding_window_features(shuffled, hop_seconds=hop_seconds, config=config)[
            "loop_score"
        ]
    return float(loop_score), float(shuffle_score)


def _compute_segment(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: RepetitionConfig,
    representations: Sequence[str],
    calibrate: bool,
) -> list[dict[str, Any]]:
    data = _candidate_data(root, row, model, config)
    identity = {column: row[column] for column in IDENTITY_COLUMNS}
    identity["scale_seconds"] = float(identity["scale_seconds"])
    output: list[dict[str, Any]] = []
    for name in representations:
        values, hop_seconds, transposition_invariant = data[name]
        result = _analyze_candidate(name, values, hop_seconds, transposition_invariant, config)
        looped = shuffled = np.nan
        if calibrate:
            looped, shuffled = _calibrate_candidate(
                name,
                values,
                hop_seconds,
                transposition_invariant,
                config,
                _seed(f"{row['segment_id']}:{name}", config.random_seed),
            )
        output.append(
            {
                **identity,
                "representation": name,
                "method": (
                    "phase_lifted_path_homology"
                    if name in PATH_REPRESENTATIONS
                    else "sliding_window_homology"
                ),
                "sequence_points": int(len(values)),
                "synthetic_loop_score": looped,
                "shuffled_score": shuffled,
                **result,
            }
        )
    return output


def _compute_features(
    root: Path,
    manifest: pd.DataFrame,
    config: RepetitionConfig,
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
                    f"repetition analysis failed for {row['segment_id']}: {exc}"
                ) from exc
            if completed % 100 == 0 or completed == len(records):
                print(f"Repetition segments: {completed}/{len(records)}", flush=True)
    return pd.DataFrame(output).sort_values(
        ["split", "group", "track_id", "scale_seconds", "representation"]
    )


def _quality_filter(
    manifest: pd.DataFrame, representations: Sequence[str], config: RepetitionConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds: dict[str, int] = {}
    mapping = {
        "sw_acoustic_novelty": ("acoustic_windows", config.min_sw_points + 1),
        "sw_loudness": ("acoustic_windows", config.min_sw_points),
        "sw_onset": ("rhythm_windows", config.min_sw_points),
        "sw_tonal_novelty": ("pitch_steps", config.min_sw_points + 1),
        "sw_modulation": ("modulation_windows", config.min_sw_points),
        "path_acoustic_phase": ("acoustic_windows", config.min_raw_steps),
        "path_rhythm_phase": ("rhythm_windows", config.min_raw_steps),
        "path_chroma_phase": ("pitch_steps", config.min_raw_steps),
    }
    for name in representations:
        column, threshold = mapping[name]
        thresholds[column] = max(thresholds.get(column, 0), threshold)
    eligible = np.ones(len(manifest), dtype=bool)
    for column, threshold in thresholds.items():
        values = pd.to_numeric(manifest[column], errors="coerce").fillna(0).to_numpy()
        eligible &= values >= threshold
    return manifest.loc[eligible].copy(), manifest.loc[~eligible].copy()


def _exploration_manifest(manifest: pd.DataFrame, config: RepetitionConfig) -> pd.DataFrame:
    eligible, _ = _quality_filter(manifest, REPRESENTATIONS, config)
    base = eligible[(eligible["split"] == "discovery") & (eligible["scale_seconds"] == 180.0)]
    sensitivity = eligible[
        (eligible["split"] == "discovery") & (eligible["scale_seconds"] == 300.0)
    ]
    base = base[base["track_id"].isin(sensitivity["track_id"])]
    rng = np.random.default_rng(config.random_seed)
    tracks: list[str] = []
    for group in ("focus", "pop", "classical"):
        choices = np.sort(base.loc[base["group"] == group, "track_id"].unique())
        if len(choices) < config.exploration_tracks_per_group:
            raise RepetitionAnalysisError(f"not enough eligible {group} tracks")
        tracks.extend(
            str(value)
            for value in rng.choice(choices, config.exploration_tracks_per_group, replace=False)
        )
    return eligible[
        (eligible["split"] == "discovery")
        & (eligible["track_id"].isin(tracks))
        & (eligible["scale_seconds"].isin([180.0, 300.0]))
    ].copy()


def _calibration_tests(features: pd.DataFrame, config: RepetitionConfig) -> pd.DataFrame:
    primary = features[features["scale_seconds"] == 180.0]
    rows: list[dict[str, Any]] = []
    for name, view in primary.groupby("representation"):
        differences = view["synthetic_loop_score"] - view["shuffled_score"]
        statistic, p_value = wilcoxon(differences, alternative="greater")
        rows.append(
            {
                "representation": name,
                "method": view["method"].iloc[0],
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


def _effect_test(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    statistic, p_value = mannwhitneyu(first, second, alternative="greater", method="auto")
    effect = 2.0 * float(statistic) / (len(first) * len(second)) - 1.0
    return effect, float(p_value)


def _group_tests(
    features: pd.DataFrame,
    *,
    split: str,
    scale: float,
    role: str,
    comparator: str,
    representations: Sequence[str] | None = None,
) -> pd.DataFrame:
    subset = features[(features["split"] == split) & (features["scale_seconds"] == scale)]
    if representations is not None:
        subset = subset[subset["representation"].isin(representations)]
    rows: list[dict[str, Any]] = []
    for name, view in subset.groupby("representation"):
        focus = view[view["group"] == "focus"]["loop_score"].to_numpy()
        comparison = view[view["group"] == comparator]["loop_score"].to_numpy()
        effect, p_value = _effect_test(focus, comparison)
        rows.append(
            {
                "role": role,
                "comparison": f"focus_greater_than_{comparator}",
                "split": split,
                "scale_seconds": scale,
                "representation": name,
                "method": view["method"].iloc[0],
                "n_focus": len(focus),
                "n_comparator": len(comparison),
                "focus_median": float(np.median(focus)),
                "comparator_median": float(np.median(comparison)),
                "rank_biserial_focus_minus_comparator": effect,
                "p_one_sided": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = benjamini_hochberg(result["p_one_sided"].to_numpy())
    return result


def _select_representations(
    features: pd.DataFrame,
    calibration: pd.DataFrame,
    discovery_tests: pd.DataFrame,
    config: RepetitionConfig,
) -> tuple[list[str], pd.DataFrame]:
    primary = discovery_tests[discovery_tests["scale_seconds"] == 180.0].copy()
    scale = discovery_tests[discovery_tests["scale_seconds"] == 300.0][
        ["representation", "rank_biserial_focus_minus_comparator"]
    ].rename(columns={"rank_biserial_focus_minus_comparator": "effect_300"})
    selection = primary.merge(scale, on="representation").merge(
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
    correlations: list[float] = []
    for row in selection.itertuples(index=False):
        paired = features[features["representation"] == row.representation].pivot(
            index="track_id", columns="scale_seconds", values="loop_score"
        )
        correlation = spearmanr(paired[180.0], paired[300.0]).statistic
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
    selection["scale_spearman"] = correlations
    selection["eligible"] = (
        selection["calibration_pass"]
        & (selection["p_one_sided"] <= config.selection_p_threshold)
        & (selection["rank_biserial_focus_minus_comparator"] >= config.selection_effect_threshold)
        & (selection["effect_300"] > 0)
        & (selection["scale_spearman"] >= config.selection_stability_threshold)
    )
    selection["selection_score"] = (
        selection["rank_biserial_focus_minus_comparator"].clip(lower=0)
        * selection["scale_spearman"].clip(lower=0)
        * selection["median_delta"].clip(lower=0)
    )
    selection = selection.sort_values(["eligible", "selection_score"], ascending=[False, False])
    selected = selection.loc[selection["eligible"], "representation"].head(
        config.max_selected_representations
    )
    selection["selected"] = selection["representation"].isin(selected)
    return selected.tolist(), selection


def _wide(features: pd.DataFrame) -> pd.DataFrame:
    wide = features.pivot(
        index=list(IDENTITY_COLUMNS), columns="representation", values="loop_score"
    )
    wide.columns = [f"{name}__loop_score" for name in wide.columns]
    return wide.reset_index()


def _classification(features: pd.DataFrame, config: RepetitionConfig) -> pd.DataFrame:
    wide = _wide(features)
    columns = [column for column in wide if "__loop_score" in column]
    rows: list[dict[str, Any]] = []
    for task, groups in (
        ("three_class", ("classical", "focus", "pop")),
        ("focus_vs_pop", ("focus", "pop")),
    ):
        subset = wide[(wide["scale_seconds"] == 180.0) & wide["group"].isin(groups)]
        train = subset[subset["split"] == "discovery"]
        validation = subset[subset["split"] == "validation"]
        encoder = LabelEncoder().fit(train["group"])
        y_train = encoder.transform(train["group"])
        y_validation = encoder.transform(validation["group"])
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
            pipeline,
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="f1_macro",
            cv=folds,
            n_jobs=1,
        ).fit(train[columns], y_train)
        predictions = search.predict(validation[columns])
        probabilities = search.predict_proba(validation[columns])
        auroc = (
            roc_auc_score(y_validation, probabilities[:, 1])
            if len(groups) == 2
            else roc_auc_score(y_validation, probabilities, multi_class="ovr", average="macro")
        )
        rows.append(
            {
                "task": task,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_features": len(columns),
                "best_c": float(search.best_params_["classifier__C"]),
                "cv_macro_f1": float(search.best_score_),
                "balanced_accuracy": float(balanced_accuracy_score(y_validation, predictions)),
                "macro_f1": float(f1_score(y_validation, predictions, average="macro")),
                "macro_auroc": float(auroc),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def run_exploration(root: Path, config: RepetitionConfig) -> dict[str, Any]:
    manifest_path = root / "metadata" / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    sample = _exploration_manifest(manifest, config)
    features = _compute_features(root, sample, config, REPRESENTATIONS, calibrate=True)
    calibration = _calibration_tests(features, config)
    discovery_tests = pd.concat(
        [
            _group_tests(
                features,
                split="discovery",
                scale=180.0,
                role="exploration",
                comparator="pop",
            ),
            _group_tests(
                features,
                split="discovery",
                scale=300.0,
                role="scale_stability",
                comparator="pop",
            ),
        ],
        ignore_index=True,
    )
    selected, selection = _select_representations(features, calibration, discovery_tests, config)
    metadata = root / "metadata"
    outputs = {
        "features": metadata / "repetition_exploration_features.csv",
        "calibration": metadata / "repetition_calibration_tests.csv",
        "tests": metadata / "repetition_exploration_tests.csv",
        "selection": metadata / "repetition_feature_selection.csv",
    }
    _write_csv(outputs["features"], features)
    _write_csv(outputs["calibration"], calibration)
    _write_csv(outputs["tests"], discovery_tests)
    _write_csv(outputs["selection"], selection)
    payload = {
        "generated_at": date.today().isoformat(),
        "role": "discovery-only repetition calibration and feature selection",
        "config": asdict(config),
        "config_sha256": _json_hash(asdict(config)),
        "sample_tracks_per_group": config.exploration_tracks_per_group,
        "candidate_representations": list(REPRESENTATIONS),
        "calibration_passes": calibration.loc[
            calibration["calibration_pass"], "representation"
        ].tolist(),
        "selected_representations": selected,
        "selection_rule": (
            "synthetic-loop calibration passes; discovery Focus>Pop one-sided p threshold "
            "and rank-biserial threshold pass; 300s direction agrees; paired scale "
            "Spearman passes"
        ),
        "outputs": {name: path.relative_to(root).as_posix() for name, path in outputs.items()},
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(metadata / "repetition_exploration_summary.json", payload)
    return payload


def _load_selected(root: Path) -> list[str]:
    path = root / "metadata" / "repetition_exploration_summary.json"
    if not path.is_file():
        raise RepetitionAnalysisError("run repetition exploration first")
    selected = json.loads(path.read_text(encoding="utf-8"))["selected_representations"]
    if not selected:
        raise RepetitionAnalysisError("no repetition representation passed discovery gates")
    if set(selected) - set(REPRESENTATIONS):
        raise RepetitionAnalysisError("selection contains an unknown representation")
    return list(selected)


def _plot_scores(
    root: Path,
    features: pd.DataFrame,
    selected: Sequence[str],
    config: RepetitionConfig,
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
    groups = ("focus", "pop", "classical")
    colors = {"focus": "#2B6CB0", "pop": "#D95F02", "classical": "#55A868"}
    for axis, name in zip(axes, selected, strict=True):
        view = validation[validation["representation"] == name]
        values: list[np.ndarray] = []
        labels: list[str] = []
        box_colors: list[str] = []
        for scale in (180.0, 300.0):
            for group in groups:
                values.append(
                    view[(view["scale_seconds"] == scale) & (view["group"] == group)][
                        "loop_score"
                    ].to_numpy()
                )
                labels.append(f"{int(scale)} {group[0].upper()}")
                box_colors.append(colors[group])
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=30)
        axis.set_title(name)
        axis.set_ylabel("persistent H1 loop score")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Repetition-sensitive homology: frozen validation scores")
    figure.tight_layout()
    output_dir = root / "runs" / "repetition_homology"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_paths = [
        output_dir / "loop_scores_validation.png",
        output_dir / "loop_scores_validation.svg",
    ]
    for path in score_paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    profile, profile_axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 4.5))
    profile_axes = np.atleast_1d(profile_axes)
    thresholds = np.asarray(config.path_thresholds)
    for axis, name in zip(profile_axes, selected, strict=True):
        view = validation[
            (validation["representation"] == name) & (validation["scale_seconds"] == 180.0)
        ]
        for group in groups:
            scores = view[view["group"] == group]["loop_score"].to_numpy()
            survival = np.mean(scores[:, None] >= thresholds[None, :], axis=0)
            axis.plot(
                thresholds,
                survival,
                marker="o",
                markersize=3,
                color=colors[group],
                label=group,
            )
        axis.set_title(name)
        axis.set_xlabel("edge-weight filtration threshold")
        axis.set_ylabel("fraction with Path H1 = 1")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
        axis.legend()
    profile.suptitle("Validation 180s: persistence of the phase-cycle H1")
    profile.tight_layout()
    profile_paths = [
        output_dir / "path_h1_filtration_validation.png",
        output_dir / "path_h1_filtration_validation.svg",
    ]
    for path in profile_paths:
        profile.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(profile)
    return [path.relative_to(root).as_posix() for path in (*score_paths, *profile_paths)]


def _write_report(
    root: Path,
    selected: Sequence[str],
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    tests: pd.DataFrame,
    classification: pd.DataFrame,
    config: RepetitionConfig,
    excluded: int,
    plots: Sequence[str],
) -> Path:
    focus_pop_180 = tests[tests["role"] == "confirmatory_focus_vs_pop"]
    focus_pop_300 = tests[tests["role"] == "replication_focus_vs_pop"]
    focus_classical_180 = tests[tests["role"] == "specificity_focus_vs_classical"]
    comparison = focus_pop_180.merge(focus_pop_300, on="representation", suffixes=("_180", "_300"))
    primary_supported = focus_pop_180.loc[
        focus_pop_180["p_fdr_bh"] <= config.validation_fdr_q, "representation"
    ].tolist()
    supported_text = "、".join(f"`{name}`" for name in primary_supported)
    interpretation = [
        "该结果描述数据集中的可听重复结构，不构成注意力作用的因果证据。",
        f"180秒 Focus/Pop 主确认支持：{supported_text}。",
    ]
    if "path_chroma_phase" not in selected:
        interpretation.append(
            "`path_chroma_phase` 未通过 discovery 冻结门槛，因此没有进入确认性检验。"
        )
    elif "path_chroma_phase" not in primary_supported:
        interpretation.append(
            "`path_chroma_phase` 在180秒 Focus/Pop 主检验中未通过，只能视为辅助信号。"
        )
    lines = [
        "# 循环敏感 Homology / Path Homology 结果",
        "",
        f"生成日期：{date.today().isoformat()}。先用 discovery 小样本执行人工循环/时间打乱校准，"
        "再冻结候选并在 validation 确认。检验方向预先固定为 Focus 的环分数高于对照。",
        "",
        "## 人工循环校准",
        "",
        "| 表示 | 方法 | 人工循环 | 时间打乱 | 中位差 | 正向比例 | FDR | 通过 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in calibration.itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.method} | {row.synthetic_loop_median:.3f} | "
            f"{row.shuffled_median:.3f} | {row.median_delta:.3f} | "
            f"{row.positive_fraction:.2f} | {row.p_fdr_bh:.3g} | "
            f"{'是' if row.calibration_pass else '否'} |"
        )
    lines.extend(
        [
            "",
            "## Discovery 小样本筛选",
            "",
            "| 表示 | 180s 效应 | 单侧 p | 300s 效应 | 跨尺度 Spearman | 入选 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in selection.itertuples(index=False):
        lines.append(
            f"| {row.representation} | "
            f"{row.rank_biserial_focus_minus_comparator:.3f} | "
            f"{row.p_one_sided:.3g} | {row.effect_300:.3f} | "
            f"{row.scale_spearman:.3f} | {'是' if row.selected else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 冻结表示",
            "",
            *[f"- `{name}`" for name in selected],
            "",
            f"全量清单中 {excluded} 个片段未达到时序长度质量门槛，已在统计前排除。",
            "",
            "## Focus vs Pop 确认",
            "",
            "| 表示 | Focus 180s | Pop 180s | 180s 效应 | 180s FDR | 300s 效应 | 300s FDR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.sort_values("p_fdr_bh_180").itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.focus_median_180:.3f} | "
            f"{row.comparator_median_180:.3f} | "
            f"{row.rank_biserial_focus_minus_comparator_180:.3f} | "
            f"{row.p_fdr_bh_180:.3g} | "
            f"{row.rank_biserial_focus_minus_comparator_300:.3f} | "
            f"{row.p_fdr_bh_300:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Classical 特异性复核",
            "",
            "| 表示 | Focus | Classical | 效应 | FDR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in focus_classical_180.sort_values("p_fdr_bh").itertuples(index=False):
        lines.append(
            f"| {row.representation} | {row.focus_median:.3f} | "
            f"{row.comparator_median:.3f} | "
            f"{row.rank_biserial_focus_minus_comparator:.3f} | {row.p_fdr_bh:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 分类辅助结果",
            "",
            "| 任务 | Macro-F1 | Balanced accuracy | AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in classification.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.macro_f1:.3f} | "
            f"{row.balanced_accuracy:.3f} | {row.macro_auroc:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            f"phase-lifted 图使用 {config.phase_bins} 个相位节点形成有向环；边权是相隔一个"
            "主周期的同相位复现一致性。`loop_score` 是该环最弱边的权重，也就是 H1 在"
            "边权过滤中首次完整出现的临界尺度。普通滑动窗口方法只有在延迟嵌入产生"
            "长寿命 H1 时才得到高分。",
            "",
            *interpretation,
            "",
            "## 图形输出",
            "",
            *[f"- `{path}`" for path in plots],
            "",
        ]
    )
    path = root / "docs" / "repetition-homology-results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_full(root: Path, config: RepetitionConfig) -> dict[str, Any]:
    selected = _load_selected(root)
    manifest_path = root / "metadata" / "feature_segments.csv"
    full_manifest = pd.read_csv(manifest_path)
    manifest, excluded = _quality_filter(full_manifest, selected, config)
    features = _compute_features(root, manifest, config, selected, calibrate=False)
    tests = pd.concat(
        [
            _group_tests(
                features,
                split="validation",
                scale=180.0,
                role="confirmatory_focus_vs_pop",
                comparator="pop",
                representations=selected,
            ),
            _group_tests(
                features,
                split="validation",
                scale=300.0,
                role="replication_focus_vs_pop",
                comparator="pop",
                representations=selected,
            ),
            _group_tests(
                features,
                split="validation",
                scale=180.0,
                role="specificity_focus_vs_classical",
                comparator="classical",
                representations=selected,
            ),
            _group_tests(
                features,
                split="validation",
                scale=300.0,
                role="specificity_scale_focus_vs_classical",
                comparator="classical",
                representations=selected,
            ),
        ],
        ignore_index=True,
    )
    classification = _classification(features, config)
    metadata = root / "metadata"
    feature_path = metadata / "repetition_homology_features.csv"
    test_path = metadata / "repetition_homology_tests.csv"
    classification_path = metadata / "repetition_homology_classification.csv"
    _write_csv(feature_path, features)
    _write_csv(test_path, tests)
    _write_csv(classification_path, classification)
    plots = _plot_scores(root, features, selected, config)
    calibration = pd.read_csv(metadata / "repetition_calibration_tests.csv")
    selection = pd.read_csv(metadata / "repetition_feature_selection.csv")
    report_path = _write_report(
        root,
        selected,
        calibration,
        selection,
        tests,
        classification,
        config,
        len(excluded),
        plots,
    )
    primary = tests[
        (tests["role"] == "confirmatory_focus_vs_pop")
        & (tests["p_fdr_bh"] <= config.validation_fdr_q)
    ]
    replication = tests[
        (tests["role"] == "replication_focus_vs_pop")
        & (tests["p_fdr_bh"] <= config.validation_fdr_q)
    ]
    replicated = primary.merge(replication, on="representation")
    replicated = replicated[
        (replicated["rank_biserial_focus_minus_comparator_x"] > 0)
        & (replicated["rank_biserial_focus_minus_comparator_y"] > 0)
    ]
    specificity = tests[
        (tests["role"] == "specificity_focus_vs_classical")
        & (tests["p_fdr_bh"] <= config.validation_fdr_q)
    ]
    outputs = [feature_path, test_path, classification_path, report_path]
    outputs.extend(root / Path(path) for path in plots)
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "selected_representations": selected,
        "segments": int(len(manifest)),
        "quality_excluded_segments": int(len(excluded)),
        "confirmatory_discoveries": int(len(primary)),
        "cross_scale_replicated_discoveries": int(len(replicated)),
        "cross_comparator_discoveries": int(len(primary.merge(specificity, on="representation"))),
        "config": asdict(config),
        "outputs": {
            "features": feature_path.relative_to(root).as_posix(),
            "tests": test_path.relative_to(root).as_posix(),
            "classification": classification_path.relative_to(root).as_posix(),
            "report": report_path.relative_to(root).as_posix(),
            "plots": plots,
        },
        "output_sha256": {path.relative_to(root).as_posix(): _sha256(path) for path in outputs},
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(metadata / "repetition_homology_summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("explore", "run", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = load_config(root)
    if args.command in {"explore", "all"}:
        print(json.dumps(run_exploration(root, config), ensure_ascii=False, indent=2))
    if args.command in {"run", "all"}:
        print(json.dumps(run_full(root, config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
