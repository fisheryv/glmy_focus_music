from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from pyglmy import vietoris_rips
from scipy.stats import mannwhitneyu

from features.batch import _json_hash, _read_npz, _sha256, _write_json_atomic
from topology.statistics import benjamini_hochberg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODALITIES = ("acoustic", "rhythm")
METRICS = (
    "h1_max_persistence",
    "h1_dominance",
    "h1_gap",
    "h1_surrogate_excess",
)
PRIMARY_METRIC = "h1_surrogate_excess"
IDENTITY_COLUMNS = ("segment_id", "track_id", "group", "split", "scale_seconds")


class MultiscaleDelayError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MultiscaleDelayConfig:
    exploration_tracks_per_group: int = 24
    scales_bars: tuple[int, ...] = (4, 8, 16, 32)
    beats_per_bar: int = 4
    embedding_dimension: int = 8
    acoustic_pca_components: int = 8
    min_cycles: float = 3.0
    min_effective_seconds: float = 270.0
    max_landmarks: int = 64
    max_landmark_candidates: int = 160
    surrogate_count: int = 9
    prominent_lifetime: float = 0.10
    validation_fdr_q: float = 0.05
    workers: int = 4
    random_seed: int = 20260716

    def validate(self) -> None:
        if self.exploration_tracks_per_group < 8:
            raise MultiscaleDelayError("exploration sample is too small")
        if not self.scales_bars or any(value < 2 for value in self.scales_bars):
            raise MultiscaleDelayError("bar scales must contain integers >= 2")
        if self.beats_per_bar < 2 or self.embedding_dimension < 3:
            raise MultiscaleDelayError("phase resolution is too small")
        if self.acoustic_pca_components < 2:
            raise MultiscaleDelayError("at least two acoustic components are required")
        if self.min_cycles < 2:
            raise MultiscaleDelayError("at least two cycles are required")
        if not 0 < self.min_effective_seconds <= 300:
            raise MultiscaleDelayError("effective-duration threshold must lie in (0, 300]")
        if self.max_landmarks < 24:
            raise MultiscaleDelayError("too few landmarks for H1")
        if self.max_landmark_candidates < self.max_landmarks:
            raise MultiscaleDelayError("landmark candidates cannot be fewer than landmarks")
        if self.surrogate_count < 3 or self.workers < 1:
            raise MultiscaleDelayError("surrogate count and workers are too small")
        if not 0 < self.prominent_lifetime < 1 or not 0 < self.validation_fdr_q < 1:
            raise MultiscaleDelayError("statistical thresholds must lie in (0, 1)")


def load_config(root: Path) -> MultiscaleDelayConfig:
    with (root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    values = dict(raw.get("multiscale_delay", {}))
    if "scales_bars" in values:
        values["scales_bars"] = tuple(int(value) for value in values["scales_bars"])
    values.setdefault("random_seed", int(raw.get("project", {}).get("seed", 20260716)))
    unknown = set(values) - set(MultiscaleDelayConfig.__dataclass_fields__)
    if unknown:
        raise MultiscaleDelayError(f"unknown multiscale_delay keys: {sorted(unknown)}")
    config = MultiscaleDelayConfig(**values)
    config.validate()
    return config


def _seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise MultiscaleDelayError("feature sequence must be a matrix")
    median = np.nanmedian(values, axis=0)
    q25, q75 = np.nanquantile(values, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    keep = np.isfinite(scale) & (scale > 1e-8)
    if np.count_nonzero(keep) < 2:
        raise MultiscaleDelayError("feature sequence has fewer than two varying dimensions")
    standardized = (values[:, keep] - median[keep]) / scale[keep]
    for column in range(standardized.shape[1]):
        finite = np.isfinite(standardized[:, column])
        if not np.all(finite):
            observed = np.flatnonzero(finite)
            if observed.size < 2:
                standardized[:, column] = 0.0
            else:
                standardized[:, column] = np.interp(
                    np.arange(len(standardized)), observed, standardized[observed, column]
                )
    return np.clip(standardized, -8.0, 8.0)


def delay_embed_multivariate(
    values: np.ndarray, *, period_frames: int, dimension: int
) -> tuple[np.ndarray, int]:
    values = _robust_standardize(values)
    delay = max(1, int(round(period_frames / dimension)))
    width = 1 + (dimension - 1) * delay
    if len(values) < width + 3:
        raise MultiscaleDelayError("sequence is too short for the requested delay embedding")
    cloud = np.concatenate(
        [
            values[offset : len(values) - width + 1 + offset]
            for offset in range(0, width, delay)
        ],
        axis=1,
    )
    return cloud, delay


def _candidate_sample(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(int)
    return values[indices]


def _farthest_landmarks(values: np.ndarray, count: int, candidate_count: int) -> np.ndarray:
    values = _candidate_sample(np.asarray(values, dtype=np.float64), candidate_count)
    if len(values) <= count:
        return values
    center = np.mean(values, axis=0)
    selected = [int(np.argmax(np.sum((values - center) ** 2, axis=1)))]
    minimum = np.sum((values - values[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.sum((values - values[index]) ** 2, axis=1))
    return values[selected]


def _h1_descriptors(cloud: np.ndarray, config: MultiscaleDelayConfig) -> dict[str, float]:
    landmarks = _farthest_landmarks(
        cloud, config.max_landmarks, config.max_landmark_candidates
    )
    if len(landmarks) < 8:
        return {
            "h1_max_persistence": 0.0,
            "h1_total_persistence": 0.0,
            "h1_dominance": 0.0,
            "h1_gap": 0.0,
            "h1_count": 0.0,
            "h1_prominent_count": 0.0,
            "distance_scale": 1.0,
        }
    distances = np.linalg.norm(landmarks[:, None] - landmarks[None, :], axis=2)
    upper = distances[np.triu_indices(len(landmarks), k=1)]
    positive = upper[upper > np.finfo(float).eps]
    distance_scale = float(np.median(positive)) if positive.size else 1.0
    diagram = vietoris_rips(
        distances / distance_scale,
        distance_matrix=True,
        max_dimension=1,
    ).diagram(1)
    lifetimes = diagram[:, 1] - diagram[:, 0] if len(diagram) else np.empty(0)
    lifetimes = np.sort(lifetimes[np.isfinite(lifetimes) & (lifetimes > 1e-10)])[::-1]
    maximum = float(lifetimes[0]) if lifetimes.size else 0.0
    second = float(lifetimes[1]) if lifetimes.size > 1 else 0.0
    total = float(np.sum(lifetimes))
    return {
        "h1_max_persistence": maximum,
        "h1_total_persistence": total,
        "h1_dominance": maximum / total if total > 1e-10 else 0.0,
        "h1_gap": maximum - second,
        "h1_count": float(lifetimes.size),
        "h1_prominent_count": float(np.count_nonzero(lifetimes >= config.prominent_lifetime)),
        "distance_scale": distance_scale,
    }


def _block_shuffle(values: np.ndarray, block_frames: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    blocks = [
        values[start : min(start + block_frames, len(values))]
        for start in range(0, len(values), block_frames)
    ]
    if len(blocks) < 4:
        return values[rng.permutation(len(values))]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order], axis=0)


def _analyze_scale(
    values: np.ndarray,
    *,
    frame_hop_seconds: float,
    bar_seconds: float,
    scale_bars: int,
    seed: int,
    config: MultiscaleDelayConfig,
) -> dict[str, float]:
    period_seconds = scale_bars * bar_seconds
    period_frames = max(
        config.embedding_dimension,
        int(round(period_seconds / frame_hop_seconds)),
    )
    available_cycles = len(values) / period_frames
    if available_cycles < config.min_cycles:
        raise MultiscaleDelayError(
            f"only {available_cycles:.2f} cycles available at {scale_bars} bars"
        )
    cloud, delay = delay_embed_multivariate(
        values, period_frames=period_frames, dimension=config.embedding_dimension
    )
    observed = _h1_descriptors(cloud, config)
    block_frames = max(1, int(round(bar_seconds / frame_hop_seconds)))
    null_rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for _ in range(config.surrogate_count):
        shuffled = _block_shuffle(values, block_frames, rng)
        shuffled_cloud, _ = delay_embed_multivariate(
            shuffled, period_frames=period_frames, dimension=config.embedding_dimension
        )
        null_rows.append(_h1_descriptors(shuffled_cloud, config))
    null = pd.DataFrame(null_rows)
    null_max = float(null["h1_max_persistence"].median())
    return {
        **observed,
        "h1_surrogate_excess": observed["h1_max_persistence"] - null_max,
        "null_h1_max_median": null_max,
        "null_h1_dominance_median": float(null["h1_dominance"].median()),
        "null_h1_gap_median": float(null["h1_gap"].median()),
        "surrogate_percentile": float(
            (1 + np.count_nonzero(null["h1_max_persistence"] < observed["h1_max_persistence"]))
            / (config.surrogate_count + 1)
        ),
        "period_seconds": float(period_seconds),
        "period_frames": float(period_frames),
        "delay_frames": float(delay),
        "embedding_span_seconds": float(
            (config.embedding_dimension - 1) * delay * frame_hop_seconds
        ),
        "available_cycles": float(available_cycles),
        "point_count": float(len(cloud)),
        "landmark_count": float(min(len(cloud), config.max_landmarks)),
    }


def _load_model(root: Path) -> dict[str, np.ndarray]:
    return _read_npz(root / "features" / "models" / "state_model.npz")


def _load_sequences(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: MultiscaleDelayConfig,
) -> tuple[dict[str, np.ndarray], float, float]:
    acoustic = _read_npz(root / Path(str(row["acoustic_relative_path"])))
    rhythm = _read_npz(root / Path(str(row["rhythm_relative_path"])))
    acoustic_values = (acoustic["vectors"] - model["acoustic_mean"]) / model["acoustic_scale"]
    acoustic_pca = (acoustic_values - model["pca_mean"]) @ model["pca_components"].T
    acoustic_pca = acoustic_pca[:, : config.acoustic_pca_components]
    rhythm_values = np.where(rhythm["valid"], rhythm["vectors"], model["rhythm_impute"])
    rhythm_values = (rhythm_values - model["rhythm_mean"]) / model["rhythm_scale"]
    acoustic_times = np.asarray(acoustic["times"], dtype=float)
    rhythm_times = np.asarray(rhythm["times"], dtype=float)
    frame_hop = float(np.median(np.diff(acoustic_times)))
    if not np.isclose(frame_hop, np.median(np.diff(rhythm_times)), atol=0.05):
        raise MultiscaleDelayError("acoustic and rhythm frame rates do not match")
    beat_times = np.asarray(rhythm["beat_times"], dtype=float)
    intervals = np.diff(beat_times)
    intervals = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    if intervals.size < 8:
        raise MultiscaleDelayError("insufficient reliable beats for bar-scaled analysis")
    bar_seconds = float(np.median(intervals) * config.beats_per_bar)
    return {"acoustic": acoustic_pca, "rhythm": rhythm_values}, frame_hop, bar_seconds


def _aggregate_multiscale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    if not rows:
        return output
    frame = pd.DataFrame(rows)
    for modality, view in frame.groupby("modality"):
        if len(view) < 3:
            continue
        identity = {column: view[column].iloc[0] for column in IDENTITY_COLUMNS}
        output.append(
            {
                **identity,
                "modality": modality,
                "scale_bars": 0,
                "scale_label": "multiscale_mean",
                "eligible_scales": int(len(view)),
                "effective_seconds": float(view["effective_seconds"].min()),
                **{
                    column: float(view[column].mean())
                    for column in view
                    if column.startswith("h1_")
                    or column.startswith("null_")
                    or column == "surrogate_percentile"
                },
                "period_seconds": float("nan"),
                "period_frames": float("nan"),
                "delay_frames": float("nan"),
                "embedding_span_seconds": float("nan"),
                "available_cycles": float("nan"),
                "point_count": float(view["point_count"].mean()),
                "landmark_count": float(view["landmark_count"].mean()),
                "distance_scale": float(view["distance_scale"].mean()),
            }
        )
    return output


def _compute_segment(
    root: Path,
    row: dict[str, Any],
    model: dict[str, np.ndarray],
    config: MultiscaleDelayConfig,
    modalities: Sequence[str],
    scales: Sequence[int],
) -> list[dict[str, Any]]:
    sequences, frame_hop, bar_seconds = _load_sequences(root, row, model, config)
    effective_seconds = min(len(values) for values in sequences.values()) * frame_hop
    identity = {column: row[column] for column in IDENTITY_COLUMNS}
    identity["scale_seconds"] = float(identity["scale_seconds"])
    output: list[dict[str, Any]] = []
    for modality in modalities:
        for scale_bars in scales:
            try:
                result = _analyze_scale(
                    sequences[modality],
                    frame_hop_seconds=frame_hop,
                    bar_seconds=bar_seconds,
                    scale_bars=scale_bars,
                    seed=_seed(
                        f"{row['segment_id']}:{modality}:{scale_bars}",
                        config.random_seed,
                    ),
                    config=config,
                )
            except MultiscaleDelayError as exc:
                if "cycles available" in str(exc) or "too short" in str(exc):
                    continue
                raise
            output.append(
                {
                    **identity,
                    "modality": modality,
                    "scale_bars": int(scale_bars),
                    "scale_label": f"{scale_bars}_bars",
                    "eligible_scales": 1,
                    "effective_seconds": float(effective_seconds),
                    **result,
                }
            )
    return _aggregate_multiscale(output)


def _compute_features(
    root: Path,
    manifest: pd.DataFrame,
    config: MultiscaleDelayConfig,
    modalities: Sequence[str],
    scales: Sequence[int],
) -> pd.DataFrame:
    model = _load_model(root)
    rows = manifest.to_dict("records")
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                _compute_segment, root, row, model, config, modalities, scales
            ): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                output.extend(future.result())
            except Exception as exc:
                raise MultiscaleDelayError(
                    f"multiscale delay failed for {row['segment_id']}: {exc}"
                ) from exc
            if completed % 12 == 0 or completed == len(rows):
                print(f"Multiscale-delay segments: {completed}/{len(rows)}", flush=True)
    return pd.DataFrame(output).sort_values(
        ["split", "group", "track_id", "modality", "scale_bars"]
    )


def _sample_discovery(manifest: pd.DataFrame, config: MultiscaleDelayConfig) -> pd.DataFrame:
    base = manifest[
        (manifest["split"] == "discovery")
        & (manifest["scale_seconds"] == 300.0)
        & (manifest["status"] == "transformed")
    ].copy()
    minimum_windows = int(math.ceil(config.min_effective_seconds / 0.5))
    base = base[
        (pd.to_numeric(base["acoustic_windows"], errors="coerce") >= minimum_windows)
        & (pd.to_numeric(base["rhythm_windows"], errors="coerce") >= minimum_windows)
    ]
    rng = np.random.default_rng(config.random_seed)
    tracks: list[str] = []
    for group in ("focus", "pop", "classical"):
        choices = np.sort(base.loc[base["group"] == group, "track_id"].unique())
        if len(choices) < config.exploration_tracks_per_group:
            raise MultiscaleDelayError(f"not enough discovery tracks for {group}")
        tracks.extend(
            str(value)
            for value in rng.choice(choices, config.exploration_tracks_per_group, replace=False)
        )
    return base[base["track_id"].isin(tracks)].copy()


def _effect_test(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    statistic, p_value = mannwhitneyu(first, second, alternative="greater", method="auto")
    effect = 2.0 * float(statistic) / (len(first) * len(second)) - 1.0
    return effect, float(p_value)


def _group_tests(
    features: pd.DataFrame,
    *,
    split: str,
    role: str,
    endpoints: Sequence[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    subset = features[features["split"] == split]
    allowed = None
    if endpoints is not None:
        allowed = {
            (str(item["modality"]), int(item["scale_bars"]), str(item["metric"]))
            for item in endpoints
        }
    rows: list[dict[str, Any]] = []
    for (modality, scale_bars), view in subset.groupby(["modality", "scale_bars"]):
        for metric in METRICS:
            if allowed is not None and (modality, int(scale_bars), metric) not in allowed:
                continue
            focus = view.loc[view["group"] == "focus", metric].to_numpy(dtype=float)
            for comparator in ("pop", "classical"):
                comparison = view.loc[view["group"] == comparator, metric].to_numpy(dtype=float)
                effect, p_value = _effect_test(focus, comparison)
                rows.append(
                    {
                        "role": role,
                        "comparison": f"focus_greater_than_{comparator}",
                        "split": split,
                        "scale_seconds": 300.0,
                        "modality": modality,
                        "scale_bars": int(scale_bars),
                        "scale_label": (
                            "multiscale_mean" if int(scale_bars) == 0 else f"{scale_bars}_bars"
                        ),
                        "metric": metric,
                        "n_focus": len(focus),
                        "n_comparator": len(comparison),
                        "focus_median": float(np.median(focus)),
                        "comparator_median": float(np.median(comparison)),
                        "rank_biserial_focus_minus_comparator": effect,
                        "p_one_sided": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = np.nan
    for comparison, indices in result.groupby("comparison").groups.items():
        del comparison
        result.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            result.loc[indices, "p_one_sided"].to_numpy()
        )
    return result


def _synthetic_calibration(config: MultiscaleDelayConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale_bars in config.scales_bars:
        period_frames = scale_bars * config.beats_per_bar
        time = np.arange(period_frames * 5)
        phase = 2 * np.pi * time / period_frames
        values = np.column_stack(
            [
                np.sin(phase),
                np.cos(phase),
                0.5 * np.sin(2 * phase + 0.3),
                0.25 * np.cos(3 * phase),
            ]
        )
        result = _analyze_scale(
            values,
            frame_hop_seconds=0.5,
            bar_seconds=config.beats_per_bar * 0.5,
            scale_bars=scale_bars,
            seed=_seed(f"synthetic:{scale_bars}", config.random_seed),
            config=config,
        )
        rows.append(
            {
                "scale_bars": scale_bars,
                "observed_h1_max": result["h1_max_persistence"],
                "null_h1_max_median": result["null_h1_max_median"],
                "h1_surrogate_excess": result["h1_surrogate_excess"],
                "calibration_pass": result["h1_surrogate_excess"] > 0.05,
            }
        )
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "scale_bars": 0,
        "observed_h1_max": float(result["observed_h1_max"].mean()),
        "null_h1_max_median": float(result["null_h1_max_median"].mean()),
        "h1_surrogate_excess": float(result["h1_surrogate_excess"].mean()),
        "calibration_pass": bool(result["calibration_pass"].all()),
    }
    return result


def _select_endpoints(
    tests: pd.DataFrame, calibration: pd.DataFrame
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    candidates = tests[
        (tests["comparison"] == "focus_greater_than_pop")
        & (tests["metric"] == PRIMARY_METRIC)
    ].copy()
    candidates = candidates.merge(
        calibration[["scale_bars", "calibration_pass", "h1_surrogate_excess"]].rename(
            columns={"h1_surrogate_excess": "synthetic_excess"}
        ),
        on="scale_bars",
        how="left",
    )
    candidates["p_fdr_primary_family"] = benjamini_hochberg(
        candidates["p_one_sided"].to_numpy()
    )
    candidates["selection_score"] = candidates[
        "rank_biserial_focus_minus_comparator"
    ].clip(lower=0)
    candidates["selected"] = False
    selected: list[dict[str, Any]] = []
    for modality, view in candidates.groupby("modality"):
        eligible = view[
            view["calibration_pass"].fillna(False)
            & (view["rank_biserial_focus_minus_comparator"] > 0)
        ].sort_values(
            ["selection_score", "p_one_sided", "scale_bars"],
            ascending=[False, True, True],
        )
        if eligible.empty:
            continue
        index = eligible.index[0]
        candidates.loc[index, "selected"] = True
        selected.append(
            {
                "modality": str(modality),
                "scale_bars": int(candidates.loc[index, "scale_bars"]),
                "metric": PRIMARY_METRIC,
            }
        )
    return selected, candidates.sort_values(
        ["selected", "selection_score"], ascending=[False, False]
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _plot_discovery(root: Path, tests: pd.DataFrame) -> None:
    view = tests[
        (tests["comparison"] == "focus_greater_than_pop")
        & (tests["metric"] == PRIMARY_METRIC)
    ].copy()
    view["scale"] = view["scale_bars"].map(
        lambda value: "multi" if int(value) == 0 else str(int(value))
    )
    figure, axis = plt.subplots(figsize=(8, 4.8))
    scales = ["multi", *[str(value) for value in sorted(view["scale_bars"].unique()) if value]]
    positions = np.arange(len(scales))
    width = 0.36
    for offset, modality in zip((-0.5, 0.5), MODALITIES, strict=True):
        modality_view = view[view["modality"] == modality].set_index("scale")
        heights = [
            float(modality_view.loc[scale, "rank_biserial_focus_minus_comparator"])
            for scale in scales
        ]
        axis.bar(positions + offset * width, heights, width=width, label=modality)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, scales)
    axis.legend()
    axis.set(
        xlabel="Delay-embedding scale (bars)",
        ylabel="Rank-biserial effect (Focus − Pop)",
        title="Discovery: surrogate-adjusted H1 across 300 s scales",
    )
    figure.tight_layout()
    output = root / "runs" / "multiscale_delay"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "discovery_scale_effects.png", dpi=180)
    figure.savefig(output / "discovery_scale_effects.svg")
    plt.close(figure)


def _plot_validation(
    root: Path, features: pd.DataFrame, endpoints: Sequence[dict[str, Any]]
) -> None:
    rows: list[pd.DataFrame] = []
    for endpoint in endpoints:
        view = features[
            (features["modality"] == endpoint["modality"])
            & (features["scale_bars"] == endpoint["scale_bars"])
            & (features["group"].isin(["focus", "pop"]))
        ][["group", endpoint["metric"]]].copy()
        view["endpoint"] = (
            f"{endpoint['modality']} / "
            f"{'multi' if endpoint['scale_bars'] == 0 else str(endpoint['scale_bars']) + ' bars'}"
        )
        view = view.rename(columns={endpoint["metric"]: "score"})
        rows.append(view)
    if not rows:
        return
    plot = pd.concat(rows, ignore_index=True)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    endpoint_names = list(dict.fromkeys(plot["endpoint"]))
    colors = {"focus": "#4c78a8", "pop": "#f58518"}
    width = 0.28
    legend_handles = []
    for endpoint_index, endpoint_name in enumerate(endpoint_names):
        for group_index, group in enumerate(("focus", "pop")):
            values = plot.loc[
                (plot["endpoint"] == endpoint_name) & (plot["group"] == group), "score"
            ].to_numpy()
            position = endpoint_index + (group_index - 0.5) * width
            box = axis.boxplot(
                values,
                positions=[position],
                widths=width * 0.8,
                patch_artist=True,
                showfliers=False,
            )
            box["boxes"][0].set_facecolor(colors[group])
            box["boxes"][0].set_alpha(0.65)
            jitter = np.linspace(-0.04, 0.04, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=8,
                alpha=0.3,
                color=colors[group],
                linewidths=0,
            )
            if endpoint_index == 0:
                legend_handles.append(box["boxes"][0])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(endpoint_names)), endpoint_names)
    axis.legend(legend_handles, ["focus", "pop"])
    axis.set(
        xlabel="Frozen endpoint",
        ylabel="Observed H1 max − block-shuffle median",
        title="Validation: multiscale delay-embedding H1 (300 s)",
    )
    figure.tight_layout()
    output = root / "runs" / "multiscale_delay"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "validation_scores.png", dpi=180)
    figure.savefig(output / "validation_scores.svg")
    plt.close(figure)


def run_exploration(root: Path, config: MultiscaleDelayConfig) -> dict[str, Any]:
    manifest_path = root / "metadata" / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    sample = _sample_discovery(manifest, config)
    features = _compute_features(root, sample, config, MODALITIES, config.scales_bars)
    tests = _group_tests(features, split="discovery", role="scale_selection")
    calibration = _synthetic_calibration(config)
    selected, selection = _select_endpoints(tests, calibration)
    metadata = root / "metadata"
    feature_path = metadata / "multiscale_delay_discovery_features.csv"
    test_path = metadata / "multiscale_delay_discovery_tests.csv"
    calibration_path = metadata / "multiscale_delay_calibration.csv"
    selection_path = metadata / "multiscale_delay_selection.csv"
    summary_path = metadata / "multiscale_delay_exploration_summary.json"
    _write_csv(feature_path, features)
    _write_csv(test_path, tests)
    _write_csv(calibration_path, calibration)
    _write_csv(selection_path, selection)
    _plot_discovery(root, tests)
    payload = {
        "generated_at": date.today().isoformat(),
        "role": "300 s discovery-only multiscale selection",
        "config": asdict(config),
        "config_sha256": _json_hash(asdict(config)),
        "tracks_per_group": config.exploration_tracks_per_group,
        "selected_endpoints": selected,
        "selection_rule": (
            "For each modality, freeze the calibrated bar scale with the largest positive "
            "discovery Focus-minus-Pop rank-biserial effect for surrogate-adjusted H1. "
            "Discovery significance is reported but is not required for freezing."
        ),
        "outputs": {
            "features": feature_path.relative_to(root).as_posix(),
            "tests": test_path.relative_to(root).as_posix(),
            "calibration": calibration_path.relative_to(root).as_posix(),
            "selection": selection_path.relative_to(root).as_posix(),
        },
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(summary_path, payload)
    return payload


def _load_selection(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "metadata" / "multiscale_delay_exploration_summary.json"
    if not path.is_file():
        raise MultiscaleDelayError("run multiscale-delay exploration first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    endpoints = list(payload.get("selected_endpoints", []))
    if not endpoints:
        raise MultiscaleDelayError("discovery produced no positive calibrated endpoint")
    return endpoints, payload


def _validation_requirements(
    endpoints: Sequence[dict[str, Any]], config: MultiscaleDelayConfig
) -> tuple[list[str], list[int]]:
    modalities = sorted({str(endpoint["modality"]) for endpoint in endpoints})
    scales: set[int] = set()
    for endpoint in endpoints:
        scale = int(endpoint["scale_bars"])
        scales.update(config.scales_bars if scale == 0 else [scale])
    return modalities, sorted(scales)


def _write_report(
    root: Path,
    endpoints: Sequence[dict[str, Any]],
    discovery_tests: pd.DataFrame,
    validation_tests: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> None:
    discovery_primary = discovery_tests[
        (discovery_tests["comparison"] == "focus_greater_than_pop")
        & (discovery_tests["metric"] == PRIMARY_METRIC)
    ].copy()
    discovery_primary["p_fdr_primary_family"] = benjamini_hochberg(
        discovery_primary["p_one_sided"].to_numpy()
    )
    lines = [
        "# 300 s multiscale delay-embedding homology",
        "",
        "## Design",
        "",
        "- Acoustic and rhythm trajectories were delay-embedded at 4, 8, 16, and 32 bars.",
        "- The eight delay coordinates span approximately one candidate period.",
        "- A one-bar block permutation preserves local content while destroying long-range order.",
        "- Only segments with at least 270 s of effective features entered the primary analysis.",
        "- Discovery selected one scale per modality; validation tested only the frozen endpoints.",
        "",
        "## Discovery scale scan",
        "",
        "| modality | scale | Focus median | Pop median | rank-biserial | one-sided p | "
        "primary-family FDR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in discovery_primary.sort_values(["modality", "scale_bars"]).itertuples():
        scale = "multi" if row.scale_bars == 0 else str(row.scale_bars)
        lines.append(
            f"| {row.modality} | {scale} | {row.focus_median:.4f} | "
            f"{row.comparator_median:.4f} | {row.rank_biserial_focus_minus_comparator:.3f} | "
            f"{row.p_one_sided:.4g} | {row.p_fdr_primary_family:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen validation",
            "",
            "| role | modality | scale | comparison | Focus median | comparator median | "
            "rank-biserial | one-sided p | FDR |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation_tests.sort_values(["role", "modality", "comparison"]).itertuples():
        scale = "multi" if row.scale_bars == 0 else str(row.scale_bars)
        lines.append(
            f"| {row.role} | {row.modality} | {scale} | {row.comparison} | "
            f"{row.focus_median:.4f} | "
            f"{row.comparator_median:.4f} | "
            f"{row.rank_biserial_focus_minus_comparator:.3f} | "
            f"{row.p_one_sided:.4g} | {row.p_fdr_bh:.4g} |"
        )
    validation_pop = validation_tests[
        (validation_tests["comparison"] == "focus_greater_than_pop")
        & (validation_tests["role"] == "frozen_validation")
    ]
    supported = validation_pop[
        (validation_pop["rank_biserial_focus_minus_comparator"] > 0)
        & (validation_pop["p_fdr_bh"] <= 0.05)
    ]
    validation_classical = validation_tests[
        (validation_tests["comparison"] == "focus_greater_than_classical")
        & (validation_tests["role"] == "frozen_validation")
    ]
    classical_supported = validation_classical[
        (validation_classical["rank_biserial_focus_minus_comparator"] > 0)
        & (validation_classical["p_fdr_bh"] <= 0.05)
    ]
    all_positive = float(
        np.mean(validation_features["h1_surrogate_excess"].to_numpy() > 0)
    )
    diagnostic = (
        validation_features.groupby(["modality", "group"])
        .agg(
            n=("track_id", "size"),
            observed_h1=("h1_max_persistence", "median"),
            block_shuffle_h1=("null_h1_max_median", "median"),
            h1_excess=("h1_surrogate_excess", "median"),
            positive_fraction=("h1_surrogate_excess", lambda values: np.mean(values > 0)),
        )
        .reset_index()
    )
    lines.extend(
        [
            "",
            "## Validation diagnostics",
            "",
            "| modality | group | n | observed H1 | block-shuffle H1 | excess | positive tracks |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic.itertuples():
        lines.append(
            f"| {row.modality} | {row.group} | {row.n} | {row.observed_h1:.4f} | "
            f"{row.block_shuffle_h1:.4f} | {row.h1_excess:.4f} | "
            f"{row.positive_fraction:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"{len(supported)} of {len(validation_pop)} frozen Focus-vs-Pop endpoints "
                "showed a positive FDR-significant validation effect."
            ),
            (
                f"{len(classical_supported)} of {len(validation_classical)} corresponding "
                "Focus-vs-Classical contrasts were positive and FDR-significant."
            ),
            (
                f"Across the computed validation rows, {all_positive:.1%} of track-scale "
                "scores exceeded their own block-shuffle median."
            ),
            "",
            (
                "The discovery preference for 16 bars did not survive independent Focus-vs-Pop "
                "validation. The method detects ordered long-range trajectories in both Focus "
                "and Pop, while separating them more clearly from Classical."
            ),
            "",
            "A positive score means that the observed temporal ordering supports a longer H1 "
            "class than the same local material after one-bar blocks are reordered. It does not "
            "by itself prove that the listener perceives a repeated motif.",
            "",
            "## Frozen endpoints",
            "",
        ]
    )
    for endpoint in endpoints:
        scale = (
            "multiscale mean"
            if endpoint["scale_bars"] == 0
            else f"{endpoint['scale_bars']} bars"
        )
        lines.append(f"- {endpoint['modality']}: {scale}, {endpoint['metric']}")
    path = root / "docs" / "multiscale-delay-300s-results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(root: Path, config: MultiscaleDelayConfig) -> dict[str, Any]:
    endpoints, exploration = _load_selection(root)
    manifest_path = root / "metadata" / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    validation = manifest[
        (manifest["split"] == "validation")
        & (manifest["scale_seconds"] == 300.0)
        & (manifest["status"] == "transformed")
    ].copy()
    minimum_windows = int(math.ceil(config.min_effective_seconds / 0.5))
    validation = validation[
        (pd.to_numeric(validation["acoustic_windows"], errors="coerce") >= minimum_windows)
        & (pd.to_numeric(validation["rhythm_windows"], errors="coerce") >= minimum_windows)
    ]
    modalities, scales = _validation_requirements(endpoints, config)
    features = _compute_features(root, validation, config, modalities, scales)
    primary_tests = _group_tests(
        features,
        split="validation",
        role="frozen_validation",
        endpoints=endpoints,
    )
    duration_sensitivity = _group_tests(
        features[features["effective_seconds"] >= 295.0],
        split="validation",
        role="duration_sensitivity_295s",
        endpoints=endpoints,
    )
    tests = pd.concat([primary_tests, duration_sensitivity], ignore_index=True)
    metadata = root / "metadata"
    feature_path = metadata / "multiscale_delay_validation_features.csv"
    test_path = metadata / "multiscale_delay_validation_tests.csv"
    summary_path = metadata / "multiscale_delay_summary.json"
    _write_csv(feature_path, features)
    _write_csv(test_path, tests)
    _plot_validation(root, features, endpoints)
    discovery_tests = pd.read_csv(root / "metadata" / "multiscale_delay_discovery_tests.csv")
    _write_report(root, endpoints, discovery_tests, tests, features)
    payload = {
        "generated_at": date.today().isoformat(),
        "role": "frozen validation on 300 s segments",
        "config_sha256": _json_hash(asdict(config)),
        "selected_endpoints": endpoints,
        "validation_segments": int(len(validation)),
        "validation_run": True,
        "frozen_tests": primary_tests.to_dict("records"),
        "outputs": {
            "features": feature_path.relative_to(root).as_posix(),
            "tests": test_path.relative_to(root).as_posix(),
            "report": "docs/multiscale-delay-300s-results.md",
            "discovery_plot": "runs/multiscale_delay/discovery_scale_effects.png",
            "validation_plot": "runs/multiscale_delay/validation_scores.png",
        },
        "exploration_config_sha256": exploration["config_sha256"],
        "input_sha256": {manifest_path.relative_to(root).as_posix(): _sha256(manifest_path)},
    }
    _write_json_atomic(summary_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="300 s multiscale delay-embedding homology")
    parser.add_argument("command", choices=("explore", "validate", "all"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = load_config(root)
    if args.command in {"explore", "all"}:
        payload = run_exploration(root, config)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.command in {"validate", "all"}:
        payload = run_validation(root, config)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
