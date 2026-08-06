from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from features.batch import (
    _json_hash,
    _read_npz,
    _replace_with_retry,
    _sha256,
    _write_json_atomic,
)

IDENTITY_COLUMNS = (
    "segment_id",
    "track_id",
    "group",
    "split",
    "scale_seconds",
    "view",
)
COLORS = {"classical": "#4472C4", "focus": "#ED7D31", "pop": "#70AD47"}


class HypothesisAnalysisError(RuntimeError):
    pass


def _write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    _replace_with_retry(temporary, path)


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _curve_key(row: Any) -> tuple[str, str, str, str, float, str]:
    return (
        str(row.segment_id),
        str(row.track_id),
        str(row.group),
        str(row.split),
        float(row.scale_seconds),
        str(row.view),
    )


def load_filtration_curves(
    path: Path,
) -> tuple[np.ndarray, dict[tuple[str, str, str, str, float, str], np.ndarray]]:
    frame = pd.read_csv(path)
    required = set(IDENTITY_COLUMNS) | {"threshold", "h1_betti"}
    missing = required - set(frame.columns)
    if missing:
        raise HypothesisAnalysisError(f"{path.name} lacks columns: {sorted(missing)}")
    frame["scale_seconds"] = frame["scale_seconds"].astype(float)
    frame["threshold"] = frame["threshold"].astype(float)
    frame["h1_betti"] = frame["h1_betti"].astype(float)
    thresholds = np.asarray(sorted(frame["threshold"].unique(), reverse=True), dtype=float)
    curves: dict[tuple[str, str, str, str, float, str], np.ndarray] = {}
    for identity, rows in frame.groupby(list(IDENTITY_COLUMNS), sort=False):
        ordered = rows.set_index("threshold")["h1_betti"].reindex(thresholds)
        if ordered.isna().any() or len(rows) != len(thresholds):
            raise HypothesisAnalysisError(f"incomplete filtration for {identity}")
        key = (
            str(identity[0]),
            str(identity[1]),
            str(identity[2]),
            str(identity[3]),
            float(identity[4]),
            str(identity[5]),
        )
        curves[key] = ordered.to_numpy(float)
    return thresholds, curves


def normalized_profile_dispersion(
    curve: np.ndarray, center: np.ndarray, thresholds: np.ndarray
) -> float:
    curve_values = np.asarray(curve, dtype=float)
    center_values = np.asarray(center, dtype=float)
    threshold_values = np.asarray(thresholds, dtype=float)
    if curve_values.shape != center_values.shape or curve_values.shape != threshold_values.shape:
        raise ValueError("curve, center, and thresholds must have matching shapes")
    span = float(np.max(threshold_values) - np.min(threshold_values))
    if span <= 0:
        return 0.0
    return float(
        np.trapezoid(
            np.abs(curve_values - center_values)[::-1],
            threshold_values[::-1],
        )
        / span
    )


def _profile_auc(curve: np.ndarray, thresholds: np.ndarray) -> float:
    return float(np.trapezoid(curve[::-1], thresholds[::-1]))


def _h1_interval_summary(arrays: dict[str, np.ndarray]) -> tuple[int, float]:
    dimensions = arrays["interval_dimension"]
    selected = dimensions == 1
    multiplicity = arrays["interval_multiplicity"][selected].astype(int)
    lifetime = arrays["interval_lifetime"][selected].astype(float)
    return int(np.sum(multiplicity)), float(np.sum(lifetime * multiplicity))


def _interval_rows(
    identity: dict[str, Any], variant: str, arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, dimension in enumerate(arrays["interval_dimension"]):
        if int(dimension) != 1:
            continue
        death = float(arrays["interval_death_threshold"][index])
        rows.append(
            {
                **identity,
                "filtration_variant": variant,
                "dimension": 1,
                "birth_index": int(arrays["interval_birth_index"][index]),
                "death_index": int(arrays["interval_death_index"][index]),
                "birth_threshold": float(arrays["interval_birth_threshold"][index]),
                "death_threshold": "" if np.isnan(death) else death,
                "lifetime": float(arrays["interval_lifetime"][index]),
                "multiplicity": int(arrays["interval_multiplicity"][index]),
                "censored": bool(arrays["interval_censored"][index]),
            }
        )
    return rows


def build_hypothesis_metrics(
    root: Path,
    topology: pd.DataFrame,
    filtrations: dict[
        str,
        tuple[np.ndarray, dict[tuple[str, str, str, str, float, str], np.ndarray]],
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers: dict[tuple[str, str, str, float], np.ndarray] = {}
    for variant, (_, curves) in filtrations.items():
        grouped: dict[tuple[str, str, float], list[np.ndarray]] = {}
        for key, curve in curves.items():
            _, _, group, split, scale, view = key
            if split == "discovery":
                grouped.setdefault((group, view, scale), []).append(curve)
        for (group, view, scale), values in grouped.items():
            centers[(variant, group, view, scale)] = np.median(np.stack(values), axis=0)

    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for row in topology.itertuples(index=False):
        key = _curve_key(row)
        identity = {column: getattr(row, column) for column in IDENTITY_COLUMNS}
        output: dict[str, Any] = {
            **identity,
            "path_entropy": float(row.path_entropy),
            "path_entropy_normalized": float(row.path_entropy_normalized),
            "directed_recurrence": float(row.directed_recurrence),
            "directed_recurrence_unbiased": float(row.directed_recurrence_unbiased),
        }
        paths = {
            "primary": root / Path(row.persistence_relative_path),
            "expanded_sensitivity": root / Path(row.sensitivity_persistence_relative_path),
        }
        for variant, path in paths.items():
            thresholds, curves = filtrations[variant]
            if key not in curves:
                raise HypothesisAnalysisError(f"filtration curve missing for {key}")
            curve = curves[key]
            center = centers.get((variant, str(row.group), str(row.view), float(row.scale_seconds)))
            arrays = _read_npz(path)
            interval_count, total_persistence = _h1_interval_summary(arrays)
            prefix = "primary" if variant == "primary" else "sensitivity"
            output.update(
                {
                    f"{prefix}_h1_profile_dispersion": (
                        normalized_profile_dispersion(curve, center, thresholds)
                        if center is not None
                        else np.nan
                    ),
                    f"{prefix}_h1_betti_auc": _profile_auc(curve, thresholds),
                    f"{prefix}_h1_zero": bool(np.max(curve) == 0),
                    f"{prefix}_h1_total_persistence": total_persistence,
                    f"{prefix}_h1_interval_count": interval_count,
                }
            )
            interval_rows.extend(_interval_rows(identity, variant, arrays))
        metric_rows.append(output)
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows)


def _bootstrap_median_difference(
    focus: np.ndarray,
    comparator: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=float)
    for index in range(resamples):
        left = rng.choice(focus, size=focus.size, replace=True)
        right = rng.choice(comparator, size=comparator.size, replace=True)
        differences[index] = np.median(left) - np.median(right)
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def _test_endpoint(
    subset: pd.DataFrame,
    *,
    metric: str,
    endpoint: str,
    comparator: str,
    alternative: str,
    filtration_variant: str,
    role: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    focus = subset.loc[subset["group"] == "focus", metric].to_numpy(float)
    other = subset.loc[subset["group"] == comparator, metric].to_numpy(float)
    if focus.size == 0 or other.size == 0:
        raise HypothesisAnalysisError(f"missing focus/{comparator} values for {endpoint}")
    directional = mannwhitneyu(focus, other, alternative=alternative)
    two_sided = mannwhitneyu(focus, other, alternative="two-sided")
    median_difference = float(np.median(focus) - np.median(other))
    ci_low, ci_high = _bootstrap_median_difference(
        focus, other, resamples=resamples, seed=seed
    )
    predicted = median_difference < 0 if alternative == "less" else median_difference > 0
    observed_direction = (
        "lower" if median_difference < 0 else "higher" if median_difference > 0 else "equal"
    )
    return {
        "analysis_role": role,
        "split": str(subset["split"].iloc[0]),
        "scale_seconds": float(subset["scale_seconds"].iloc[0]),
        "view": str(subset["view"].iloc[0]),
        "contrast": f"focus_vs_{comparator}",
        "endpoint": endpoint,
        "metric_column": metric,
        "filtration_variant": filtration_variant,
        "expected_direction": alternative,
        "observed_direction": observed_direction,
        "focus_n": int(focus.size),
        "comparator_n": int(other.size),
        "focus_median": float(np.median(focus)),
        "comparator_median": float(np.median(other)),
        "median_difference_focus_minus_comparator": median_difference,
        "median_difference_ci_low": ci_low,
        "median_difference_ci_high": ci_high,
        "mann_whitney_u": float(directional.statistic),
        "p_one_sided": float(directional.pvalue),
        "p_two_sided": float(two_sided.pvalue),
        "rank_biserial_focus_minus_comparator": float(
            2 * directional.statistic / (focus.size * other.size) - 1
        ),
        "predicted_direction_observed": bool(predicted),
    }


def run_hypothesis_tests(
    metrics: pd.DataFrame,
    *,
    fdr_q: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    specs = (
        ("path_entropy", "path_entropy", "sequence", "less", True),
        (
            "path_entropy_normalized",
            "path_entropy_normalized",
            "sequence_normalized",
            "less",
            False,
        ),
        (
            "directed_recurrence",
            "directed_recurrence",
            "sequence",
            "greater",
            True,
        ),
        (
            "directed_recurrence_unbiased",
            "directed_recurrence_unbiased",
            "sequence_unbiased",
            "greater",
            False,
        ),
        (
            "primary_h1_profile_dispersion",
            "beta1_profile_dispersion",
            "primary",
            "less",
            True,
        ),
        (
            "sensitivity_h1_profile_dispersion",
            "beta1_profile_dispersion",
            "expanded_sensitivity",
            "less",
            False,
        ),
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for scale in (180.0, 300.0):
        for view in ("modulation", "pitch", "rhythm"):
            subset = metrics[
                (metrics["split"] == "validation")
                & (metrics["scale_seconds"] == scale)
                & (metrics["view"] == view)
            ]
            for comparator in ("pop", "classical"):
                for metric, endpoint, variant, alternative, confirmatory_default in specs:
                    is_core = (
                        scale == 180.0
                        and view == "modulation"
                        and comparator == "pop"
                        and confirmatory_default
                    )
                    rows.append(
                        _test_endpoint(
                            subset,
                            metric=metric,
                            endpoint=endpoint,
                            comparator=comparator,
                            alternative=alternative,
                            filtration_variant=variant,
                            role="confirmatory_core" if is_core else "robustness_or_sensitivity",
                            resamples=bootstrap_resamples,
                            seed=seed + offset,
                        )
                    )
                    offset += 1
    tests = pd.DataFrame(rows)
    tests["p_fdr_bh"] = np.nan
    core = tests["analysis_role"] == "confirmatory_core"
    tests.loc[core, "p_fdr_bh"] = _benjamini_hochberg(
        tests.loc[core, "p_one_sided"].to_numpy(float)
    )
    tests.loc[~core, "p_fdr_bh"] = _benjamini_hochberg(
        tests.loc[~core, "p_one_sided"].to_numpy(float)
    )
    tests["verdict"] = np.where(
        tests["predicted_direction_observed"] & (tests["p_fdr_bh"] <= fdr_q),
        "supported",
        "not_supported",
    )
    supported = int(np.count_nonzero(tests.loc[core, "verdict"] == "supported"))
    composite = (
        "supported"
        if supported == 3
        else "partially_supported" if supported else "not_supported"
    )
    return tests, composite


def select_group_medoids(
    metrics: pd.DataFrame,
    sensitivity_curves: dict[tuple[str, str, str, str, float, str], np.ndarray],
) -> list[dict[str, Any]]:
    selected = metrics[
        (metrics["split"] == "validation")
        & (metrics["scale_seconds"] == 180.0)
        & (metrics["view"] == "modulation")
    ].copy()
    curve_matrix = np.stack(
        [
            sensitivity_curves[
                (
                    str(row.segment_id),
                    str(row.track_id),
                    str(row.group),
                    str(row.split),
                    float(row.scale_seconds),
                    str(row.view),
                )
            ]
            for row in selected.itertuples(index=False)
        ]
    )
    matrix = np.column_stack(
        [
            selected["path_entropy"].to_numpy(float),
            selected["directed_recurrence"].to_numpy(float),
            curve_matrix,
        ]
    )
    scale = np.std(matrix, axis=0, ddof=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    standardized = (matrix - np.mean(matrix, axis=0)) / scale
    representatives: list[dict[str, Any]] = []
    for group in ("focus", "pop", "classical"):
        positions = np.flatnonzero(selected["group"].to_numpy() == group)
        values = standardized[positions]
        distances = np.sqrt(np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2))
        totals = np.sum(distances, axis=1)
        candidates = positions[np.isclose(totals, np.min(totals), rtol=1e-12, atol=1e-12)]
        labels = selected.iloc[candidates]["segment_id"].astype(str).to_numpy()
        chosen = int(candidates[np.argmin(labels)])
        row = selected.iloc[chosen]
        representatives.append(
            {
                "group": group,
                "segment_id": str(row["segment_id"]),
                "track_id": str(row["track_id"]),
                "split": "validation",
                "scale_seconds": 180,
                "view": "modulation",
                "criterion": (
                    "minimum total standardized Euclidean distance within group over "
                    "path entropy, directed recurrence, and expanded beta1 curve"
                ),
            }
        )
    return representatives


def _save_figure(figure: Any, output_directory: Path, stem: str) -> list[str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, options in (
        ("png", {"dpi": 300}),
        ("svg", {"metadata": {"Date": None, "Creator": "focus-music-glmy"}}),
    ):
        path = output_directory / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path.as_posix())
    return paths


def _bootstrap_curve_interval(
    matrix: np.ndarray, *, resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples, matrix.shape[1]), dtype=float)
    for index in range(resamples):
        positions = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        samples[index] = np.mean(matrix[positions], axis=0)
    return (
        np.mean(matrix, axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
    )


def _all_interval_records(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, dimension in enumerate(arrays["interval_dimension"]):
        death = float(arrays["interval_death_threshold"][index])
        records.append(
            {
                "dimension": int(dimension),
                "birth_threshold": float(arrays["interval_birth_threshold"][index]),
                "death_threshold": None if np.isnan(death) else death,
                "lifetime": float(arrays["interval_lifetime"][index]),
                "multiplicity": int(arrays["interval_multiplicity"][index]),
                "censored": bool(arrays["interval_censored"][index]),
            }
        )
    return records


def persistence_coordinates(
    birth_threshold: float,
    death_threshold: float | None,
    *,
    terminal_threshold: float,
) -> tuple[float, float]:
    birth = 1.0 - float(birth_threshold)
    death = 1.0 - (
        float(terminal_threshold) if death_threshold is None else float(death_threshold)
    )
    if death + np.finfo(float).eps < birth:
        raise ValueError("persistence death must not precede birth after filtration transform")
    return birth, death


def make_hypothesis_plots(
    root: Path,
    topology: pd.DataFrame,
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    sensitivity_thresholds: np.ndarray,
    sensitivity_curves: dict[tuple[str, str, str, str, float, str], np.ndarray],
    representatives: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise HypothesisAnalysisError("matplotlib is required for paper figures") from exc

    matplotlib.rcParams["svg.hashsalt"] = f"focus-music-glmy-{seed}"

    output_directory = root / "runs" / "topology_statistics"
    generated: list[str] = []
    primary = metrics[
        (metrics["split"] == "validation")
        & (metrics["scale_seconds"] == 180.0)
        & (metrics["view"] == "modulation")
    ]
    core_tests = tests[tests["analysis_role"] == "confirmatory_core"].set_index("endpoint")
    groups = ("focus", "pop", "classical")
    rng = np.random.default_rng(seed)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for axis, metric, label in (
        (axes[0], "path_entropy", "Path entropy (nats)"),
        (axes[1], "directed_recurrence", "Directed recurrence"),
    ):
        values = [
            primary.loc[primary["group"] == group, metric].to_numpy(float)
            for group in groups
        ]
        violin = axis.violinplot(values, positions=np.arange(3), showextrema=False)
        for body, group in zip(violin["bodies"], groups, strict=True):
            body.set_facecolor(COLORS[group])
            body.set_alpha(0.25)
        box = axis.boxplot(
            values,
            positions=np.arange(3),
            widths=0.22,
            patch_artist=True,
            showfliers=False,
        )
        for patch, group in zip(box["boxes"], groups, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        for position, (group, group_values) in enumerate(zip(groups, values, strict=True)):
            jitter = rng.uniform(-0.09, 0.09, size=group_values.size)
            axis.scatter(
                position + jitter,
                group_values,
                s=12,
                alpha=0.46,
                color=COLORS[group],
                edgecolors="none",
            )
        endpoint = "path_entropy" if metric == "path_entropy" else "directed_recurrence"
        row = core_tests.loc[endpoint]
        readable_verdict = str(row["verdict"]).replace("_", " ").capitalize()
        axis.set_title(
            f"{label}\nFocus vs Pop: q={row['p_fdr_bh']:.3g}, {readable_verdict}"
        )
        axis.set_xticks(np.arange(3), ["Focus", "Pop", "Classical"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Core directed-transition hypotheses (validation, 180 s, modulation)")
    generated.extend(
        _save_figure(figure, output_directory, "h2_path_entropy_recurrence_validation_180")
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    x = sensitivity_thresholds[::-1]
    for index, group in enumerate(groups):
        group_keys = [
            key
            for key in sensitivity_curves
            if key[2] == group
            and key[3] == "validation"
            and key[4] == 180.0
            and key[5] == "modulation"
        ]
        matrix = np.stack([sensitivity_curves[key] for key in group_keys])
        mean, low, high = _bootstrap_curve_interval(
            matrix, resamples=bootstrap_resamples, seed=seed + 100 + index
        )
        axis.plot(x, mean[::-1], color=COLORS[group], label=group.capitalize(), linewidth=2)
        axis.fill_between(x, low[::-1], high[::-1], color=COLORS[group], alpha=0.18)
    axis.set(
        title="Expanded-filtration directed H1 profiles (exploratory)",
        xlabel="Transition-probability threshold",
        ylabel=r"Mean $\beta_1^{path}$ (95% bootstrap CI)",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    generated.extend(_save_figure(figure, output_directory, "h2_beta1_profiles_validation_180"))
    plt.close(figure)

    representative_records: dict[str, list[dict[str, Any]]] = {}
    for representative in representatives:
        row = topology[
            (topology["segment_id"] == representative["segment_id"])
            & (topology["view"] == "modulation")
        ].iloc[0]
        arrays = _read_npz(root / Path(row["sensitivity_persistence_relative_path"]))
        representative_records[representative["group"]] = _all_interval_records(arrays)

    figure, axes = plt.subplots(2, 3, figsize=(12.3, 6.4), sharex=True, constrained_layout=True)
    for column, representative in enumerate(representatives):
        group = representative["group"]
        records = representative_records[group]
        for dimension in (0, 1):
            axis = axes[dimension, column]
            intervals = sorted(
                [record for record in records if record["dimension"] == dimension],
                key=lambda record: (-record["lifetime"], record["birth_threshold"]),
            )
            y = 0
            for record in intervals:
                for _ in range(record["multiplicity"]):
                    birth, death = persistence_coordinates(
                        record["birth_threshold"],
                        record["death_threshold"],
                        terminal_threshold=float(np.min(sensitivity_thresholds)),
                    )
                    axis.hlines(y, birth, death, color=COLORS[group], linewidth=1.5)
                    if record["censored"]:
                        axis.plot(death, y, marker=">", color=COLORS[group], markersize=4)
                    y += 1
            if not intervals:
                axis.text(
                    0.5,
                    0.5,
                    "No intervals",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_yticks([])
            axis.set_xlim(0, 1)
            axis.grid(axis="x", alpha=0.16)
            if column == 0:
                axis.set_ylabel(f"H{dimension} barcode")
            if dimension == 0:
                axis.set_title(f"{group.capitalize()} medoid\n{representative['segment_id']}")
            else:
                axis.set_xlabel(r"Filtration time $t=1-\tau$")
    figure.suptitle("Representative persistent path barcodes (expanded filtration)")
    generated.extend(
        _save_figure(figure, output_directory, "h2_barcode_modulation_validation_180")
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, representative in zip(axes, representatives, strict=True):
        group = representative["group"]
        axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        for dimension, marker in ((0, "o"), (1, "^")):
            for record in representative_records[group]:
                if record["dimension"] != dimension:
                    continue
                birth, death = persistence_coordinates(
                    record["birth_threshold"],
                    record["death_threshold"],
                    terminal_threshold=float(np.min(sensitivity_thresholds)),
                )
                axis.scatter(
                    birth,
                    death,
                    s=26 + 6 * (record["multiplicity"] - 1),
                    marker=marker,
                    facecolors="none" if record["censored"] else COLORS[group],
                    edgecolors=COLORS[group],
                    alpha=0.8,
                    label=(
                        f"H{dimension}"
                        if not any(
                            collection.get_label() == f"H{dimension}"
                            for collection in axis.collections
                        )
                        else None
                    ),
                )
        axis.set(xlim=(0, 1), ylim=(0, 1), title=group.capitalize())
        axis.set_xlabel(r"Birth $t=1-\tau$")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel(r"Death $t=1-\tau$")
    figure.suptitle("Representative persistent path diagrams; open markers are right-censored")
    generated.extend(
        _save_figure(figure, output_directory, "h2_persistence_diagram_modulation_validation_180")
    )
    plt.close(figure)
    return [str(Path(path).relative_to(root).as_posix()) for path in generated]


def run_hypothesis_analysis(
    *,
    root: Path,
    topology: pd.DataFrame,
    fdr_q: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata = root / "metadata"
    filtration_paths = {
        "primary": metadata / "topology_filtration.csv",
        "expanded_sensitivity": metadata / "topology_filtration_sensitivity.csv",
    }
    for path in filtration_paths.values():
        if not path.is_file():
            raise HypothesisAnalysisError(f"filtration manifest not found: {path}")
    filtrations = {
        variant: load_filtration_curves(path)
        for variant, path in filtration_paths.items()
    }
    metrics, intervals = build_hypothesis_metrics(root, topology, filtrations)
    tests, composite = run_hypothesis_tests(
        metrics,
        fdr_q=fdr_q,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    sensitivity_thresholds, sensitivity_curves = filtrations["expanded_sensitivity"]
    representatives = select_group_medoids(metrics, sensitivity_curves)
    plots = make_hypothesis_plots(
        root,
        topology,
        metrics,
        tests,
        sensitivity_thresholds,
        sensitivity_curves,
        representatives,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )

    metrics_path = metadata / "topology_hypothesis_metrics.csv"
    tests_path = metadata / "topology_hypothesis_tests.csv"
    intervals_path = metadata / "topology_hypothesis_h1_intervals.csv"
    _write_frame_atomic(metrics_path, metrics)
    _write_frame_atomic(tests_path, tests)
    _write_frame_atomic(intervals_path, intervals)
    core = tests[tests["analysis_role"] == "confirmatory_core"]
    analysis_config = {
        "fdr_q": fdr_q,
        "bootstrap_resamples": bootstrap_resamples,
        "random_seed": seed,
        "primary_thresholds": filtrations["primary"][0].tolist(),
        "sensitivity_thresholds": filtrations["expanded_sensitivity"][0].tolist(),
    }
    topology_manifest = metadata / "topology_segments.csv"
    output_paths = [
        metrics_path,
        tests_path,
        intervals_path,
        *(root / Path(path) for path in plots),
    ]
    summary_path = metadata / "topology_hypothesis_summary.json"
    payload: dict[str, Any] = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "core_definition": {
            "split": "validation",
            "scale_seconds": 180,
            "view": "modulation",
            "contrast": "focus_vs_pop",
            "fdr_q": fdr_q,
            "family_size": 3,
        },
        "core_verdict": composite,
        "core_supported_endpoints": int(np.count_nonzero(core["verdict"] == "supported")),
        "analysis_config": analysis_config,
        "analysis_config_sha256": _json_hash(analysis_config),
        "primary_h1_nonzero_segment_views": int(
            np.count_nonzero(~metrics["primary_h1_zero"])
        ),
        "sensitivity_h1_nonzero_segment_views": int(
            np.count_nonzero(~metrics["sensitivity_h1_zero"])
        ),
        "core_tests": [
            {
                "endpoint": str(row.endpoint),
                "focus_median": float(row.focus_median),
                "pop_median": float(row.comparator_median),
                "rank_biserial": float(row.rank_biserial_focus_minus_comparator),
                "p_one_sided": float(row.p_one_sided),
                "p_fdr_bh": float(row.p_fdr_bh),
                "verdict": str(row.verdict),
            }
            for row in core.itertuples(index=False)
        ],
        "holdout_policy": (
            "excluded from group tests because the holdout contains Focus tracks only"
        ),
        "representatives": representatives,
        "inputs": {
            topology_manifest.relative_to(root).as_posix(): _sha256(topology_manifest),
            **{
                path.relative_to(root).as_posix(): _sha256(path)
                for path in filtration_paths.values()
            },
        },
        "outputs": {
            "metrics": metrics_path.relative_to(root).as_posix(),
            "tests": tests_path.relative_to(root).as_posix(),
            "h1_intervals": intervals_path.relative_to(root).as_posix(),
            "plots": plots,
        },
        "output_sha256": {
            path.relative_to(root).as_posix(): _sha256(path) for path in output_paths
        },
    }
    _write_json_atomic(summary_path, payload)
    return payload, tests
