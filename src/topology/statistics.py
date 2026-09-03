from __future__ import annotations

import csv
import hashlib
import os
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.batch import (
    _json_hash,
    _read_npz,
    _replace_with_retry,
    _sha256,
    _write_json_atomic,
)
from topology.metrics import TOPOLOGY_METRICS

HYPOTHESIS_AUXILIARY_METRICS = (
    "path_entropy_normalized",
    "directed_recurrence_unbiased",
)
IDENTITY_COLUMNS = ("segment_id", "track_id", "group", "split", "scale_seconds")
ANALYSIS_SETS = (
    ("primary_validation_180", "validation", 180.0, "confirmatory"),
    ("sensitivity_validation_300", "validation", 300.0, "sensitivity"),
    ("exploratory_discovery_180", "discovery", 180.0, "exploratory"),
)
LOCAL_EFFECT_BOOTSTRAP_RESAMPLES = 3000
LOCAL_EFFECT_BOOTSTRAP_SEED = 20260716
LOCAL_EFFECT_BOOTSTRAP_ANALYSIS_SETS = frozenset(
    {"primary_validation_180", "sensitivity_validation_300"}
)


class StatisticalAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StatisticalConfig:
    primary_alpha: float = 0.05
    fdr_q: float = 0.10
    primary_distance: str = "mahalanobis"
    primary_split: str = "validation"
    primary_scale_seconds: float = 180.0
    sensitivity_scale_seconds: float = 300.0
    permutations: int = 999
    classification_folds: int = 5
    bootstrap_resamples: int = 1000
    random_seed: int = 20260716

    def validate(self) -> None:
        if not 0 < self.primary_alpha < 1 or not 0 < self.fdr_q < 1:
            raise StatisticalAnalysisError("alpha and FDR q must be in (0, 1)")
        if self.primary_distance != "mahalanobis":
            raise StatisticalAnalysisError(
                "only the preregistered Mahalanobis distance is supported"
            )
        if self.permutations < 99:
            raise StatisticalAnalysisError("at least 99 permutations are required")
        if self.classification_folds < 2 or self.bootstrap_resamples < 100:
            raise StatisticalAnalysisError("classification folds or bootstrap count is too small")


def _load_config(root: Path) -> StatisticalConfig:
    with (root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle).get("evaluation", {})
    known = set(StatisticalConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise StatisticalAnalysisError(f"unknown evaluation configuration keys: {sorted(unknown)}")
    config = StatisticalConfig(**raw)
    config.validate()
    return config


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("p-values must be a finite one-dimensional sequence")
    count = values.size
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    _replace_with_retry(temporary, path)


def _load_topology_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise StatisticalAnalysisError(f"topology manifest not found: {path}")
    frame = pd.read_csv(path)
    required = (
        set(IDENTITY_COLUMNS)
        | {"view", "status"}
        | set(TOPOLOGY_METRICS)
        | set(HYPOTHESIS_AUXILIARY_METRICS)
        | {
            "persistence_relative_path",
            "sensitivity_persistence_relative_path",
        }
    )
    missing = required - set(frame.columns)
    if missing:
        raise StatisticalAnalysisError(f"topology manifest lacks columns: {sorted(missing)}")
    failed = frame[frame["status"] == "failed"]
    if not failed.empty:
        raise StatisticalAnalysisError(f"topology manifest contains {len(failed)} failed rows")
    frame["scale_seconds"] = frame["scale_seconds"].astype(float)
    for column in (*TOPOLOGY_METRICS, *HYPOTHESIS_AUXILIARY_METRICS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric_columns = [*TOPOLOGY_METRICS, *HYPOTHESIS_AUXILIARY_METRICS]
    if frame[numeric_columns].isna().any().any():
        raise StatisticalAnalysisError("topology metrics contain non-numeric or missing values")
    return frame


def _wide_topology(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index=list(IDENTITY_COLUMNS), columns="view", values=list(TOPOLOGY_METRICS))
    wide.columns = [f"{view}__{metric}" for metric, view in wide.columns]
    wide = wide.reset_index()
    if wide[list(wide.columns[len(IDENTITY_COLUMNS) :])].isna().any().any():
        raise StatisticalAnalysisError("one or more segments lack a configured graph view")
    return wide


def _group_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def _bootstrap_rank_biserial_interval(
    first: np.ndarray,
    second: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the rank-biserial effect first minus second."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or right.size == 0:
        raise ValueError("bootstrap samples must be non-empty one-dimensional arrays")
    if resamples < 100:
        raise ValueError("rank-biserial bootstrap requires at least 100 resamples")

    rng = np.random.default_rng(seed)
    left_counts = rng.multinomial(
        left.size,
        np.full(left.size, 1.0 / left.size),
        size=resamples,
    ).astype(float)
    right_counts = rng.multinomial(
        right.size,
        np.full(right.size, 1.0 / right.size),
        size=resamples,
    ).astype(float)
    pair_sign = np.sign(left[:, None] - right[None, :])
    effects = np.sum((left_counts @ pair_sign) * right_counts, axis=1) / (
        left.size * right.size
    )
    lower, upper = np.quantile(effects, [0.025, 0.975])
    return float(lower), float(upper)


def _local_effect_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _omnibus_and_pairwise(
    frame: pd.DataFrame,
    *,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = LOCAL_EFFECT_BOOTSTRAP_SEED,
    bootstrap_analysis_sets: frozenset[str] = LOCAL_EFFECT_BOOTSTRAP_ANALYSIS_SETS,
    bootstrap_views: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bootstrap_resamples and bootstrap_resamples < 100:
        raise ValueError("rank-biserial bootstrap requires at least 100 resamples")
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    expected_groups = sorted(frame["group"].dropna().unique())
    if len(expected_groups) < 2:
        raise StatisticalAnalysisError("topology statistics require at least two groups")
    for analysis_set, split, scale, role in ANALYSIS_SETS:
        subset = frame[(frame["split"] == split) & (frame["scale_seconds"] == scale)]
        groups = sorted(subset["group"].unique())
        if groups != expected_groups:
            missing = sorted(set(expected_groups) - set(groups))
            extra = sorted(set(groups) - set(expected_groups))
            raise StatisticalAnalysisError(
                f"{analysis_set} does not contain the full group family; "
                f"missing={missing}, extra={extra}"
            )
        for view in sorted(subset["view"].unique()):
            view_rows = subset[subset["view"] == view]
            for metric in TOPOLOGY_METRICS:
                samples = {
                    group: view_rows.loc[view_rows["group"] == group, metric].to_numpy(float)
                    for group in groups
                }
                if all(np.ptp(values) == 0 for values in samples.values()):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = kruskal(*samples.values())
                total = sum(values.size for values in samples.values())
                effect = max(0.0, (float(statistic) - len(groups) + 1) / (total - len(groups)))
                row: dict[str, Any] = {
                    "analysis_set": analysis_set,
                    "role": role,
                    "split": split,
                    "scale_seconds": scale,
                    "view": view,
                    "metric": metric,
                    "test": "Kruskal-Wallis",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "epsilon_squared": effect,
                    "n_total": total,
                }
                for group, values in samples.items():
                    summary = _group_summary(values)
                    row[f"n_{group}"] = values.size
                    for name, value in summary.items():
                        row[f"{group}_{name}"] = value
                omnibus_rows.append(row)

                for index, group_a in enumerate(groups):
                    for group_b in groups[index + 1 :]:
                        values_a, values_b = samples[group_a], samples[group_b]
                        if np.ptp(np.concatenate([values_a, values_b])) == 0:
                            u_statistic, pair_p = values_a.size * values_b.size / 2, 1.0
                        else:
                            u_statistic, pair_p = mannwhitneyu(
                                values_a, values_b, alternative="two-sided", method="asymptotic"
                            )
                        rank_biserial = 2 * float(u_statistic) / (
                            values_a.size * values_b.size
                        ) - 1
                        run_bootstrap = (
                            bootstrap_resamples > 0
                            and analysis_set in bootstrap_analysis_sets
                            and (bootstrap_views is None or view in bootstrap_views)
                        )
                        if run_bootstrap:
                            row_seed = _local_effect_seed(
                                bootstrap_seed,
                                analysis_set,
                                view,
                                metric,
                                group_a,
                                group_b,
                            )
                            ci_low, ci_high = _bootstrap_rank_biserial_interval(
                                values_a,
                                values_b,
                                resamples=bootstrap_resamples,
                                seed=row_seed,
                            )
                        else:
                            row_seed = np.nan
                            ci_low, ci_high = np.nan, np.nan
                        pairwise_rows.append(
                            {
                                "analysis_set": analysis_set,
                                "role": role,
                                "split": split,
                                "scale_seconds": scale,
                                "view": view,
                                "metric": metric,
                                "group_a": group_a,
                                "group_b": group_b,
                                "test": "Mann-Whitney U",
                                "statistic": float(u_statistic),
                                "p_value": float(pair_p),
                                "rank_biserial_a_minus_b": rank_biserial,
                                "rank_biserial_ci95_low": ci_low,
                                "rank_biserial_ci95_high": ci_high,
                                "bootstrap_resamples": (
                                    bootstrap_resamples if run_bootstrap else 0
                                ),
                                "bootstrap_seed": row_seed,
                                "n_a": values_a.size,
                                "n_b": values_b.size,
                            }
                        )

    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    for _, indices in omnibus.groupby("analysis_set").groups.items():
        omnibus.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            omnibus.loc[indices, "p_value"].to_numpy(float)
        )
    # Pairwise adjustment is a separate, explicitly labeled family per analysis set.
    for _, indices in pairwise.groupby("analysis_set").groups.items():
        pairwise.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            pairwise.loc[indices, "p_value"].to_numpy(float)
        )
    return omnibus, pairwise


def _pseudo_f_statistic(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    grand = np.mean(matrix, axis=0)
    between = 0.0
    within = 0.0
    for label in unique:
        group = matrix[labels == label]
        centroid = np.mean(group, axis=0)
        offset = centroid - grand
        between += group.shape[0] * float(offset @ offset)
        centered = group - centroid
        within += float(np.sum(centered**2))
    degrees_between = len(unique) - 1
    degrees_within = matrix.shape[0] - len(unique)
    if within <= np.finfo(float).eps:
        return 0.0 if between <= np.finfo(float).eps else float("inf")
    return (between / degrees_between) / (within / degrees_within)


def permanova_mahalanobis(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    permutations: int,
    seed: int,
    reference_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """One-factor permutation MANOVA under a frozen Mahalanobis metric.

    A discovery reference may be supplied when testing validation data. This
    prevents the confirmation labels and covariance from determining the
    distance metric and avoids unstable self-whitening in small samples.
    """

    values = np.asarray(matrix, dtype=float)
    label_array = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] != label_array.size:
        raise ValueError("matrix and labels have incompatible shapes")
    reference = values if reference_matrix is None else np.asarray(reference_matrix, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != values.shape[1]:
        raise ValueError("reference matrix has incompatible dimensions")
    variances = np.nanvar(reference, axis=0)
    keep = np.isfinite(variances) & (variances > np.finfo(float).eps)
    values = values[:, keep]
    reference = reference[:, keep]
    if values.shape[1] == 0:
        return {"pseudo_f": 0.0, "p_value": 1.0, "effective_dimensions": 0}
    imputer = SimpleImputer(strategy="median").fit(reference)
    reference = imputer.transform(reference)
    imputed = imputer.transform(values)
    covariance = np.cov(reference, rowvar=False)
    covariance = np.atleast_2d(covariance)
    inverse = np.linalg.pinv(covariance, rcond=1e-10)
    eigenvalues, eigenvectors = np.linalg.eigh(inverse)
    positive = eigenvalues > np.finfo(float).eps
    whitening = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    transformed = imputed @ whitening
    observed = _pseudo_f_statistic(transformed, label_array)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(label_array)
        statistic = _pseudo_f_statistic(transformed, permuted)
        exceedances += statistic >= observed
    return {
        "pseudo_f": observed,
        "p_value": (exceedances + 1) / (permutations + 1),
        "effective_dimensions": int(np.linalg.matrix_rank(covariance)),
    }


def _run_permanova(wide: pd.DataFrame, config: StatisticalConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [column for column in wide.columns if "__" in column]
    for offset, (analysis_set, split, scale, role) in enumerate(ANALYSIS_SETS):
        subset = wide[(wide["split"] == split) & (wide["scale_seconds"] == scale)]
        reference = wide[
            (wide["split"] == "discovery") & (wide["scale_seconds"] == scale)
        ]
        result = permanova_mahalanobis(
            subset[metric_columns].to_numpy(float),
            subset["group"].astype(str).to_numpy(),
            permutations=config.permutations,
            seed=config.random_seed + offset,
            reference_matrix=reference[metric_columns].to_numpy(float),
        )
        rows.append(
            {
                "analysis_set": analysis_set,
                "role": role,
                "split": split,
                "scale_seconds": scale,
                "distance": config.primary_distance,
                "permutations": config.permutations,
                "n_tracks": len(subset),
                "n_features": len(metric_columns),
                **result,
            }
        )
    return pd.DataFrame(rows)


def _safe_aggregate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=float)[np.asarray(valid, dtype=bool)]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.shape[0] == 0:
        return np.full(values.shape[1] * 2, np.nan, dtype=float)
    return np.concatenate([np.mean(selected, axis=0), np.std(selected, axis=0)])


def _load_continuous_features(
    root: Path,
    *,
    scale: float,
    segment_ids: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    manifest = root / "metadata" / "feature_segments.csv"
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["segment_id"] not in segment_ids or float(row["scale_seconds"]) != scale:
                continue
            acoustic = _read_npz(root / Path(row["acoustic_relative_path"]))
            modulation = _read_npz(root / Path(row["modulation_relative_path"]))
            acoustic_vector = _safe_aggregate(acoustic["vectors"], acoustic["valid"])
            modulation_values = np.concatenate(
                [modulation["band_energies"], modulation["key_band_energies"]], axis=1
            )
            modulation_vector = _safe_aggregate(modulation_values, modulation["valid"])
            output[row["segment_id"]] = (acoustic_vector, modulation_vector)
    missing = segment_ids - set(output)
    if missing:
        raise StatisticalAnalysisError(f"continuous features missing for {len(missing)} segments")
    return output


def _bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    strata = {label: np.flatnonzero(y_true == label) for label in classes}
    balanced: list[float] = []
    macro_f1: list[float] = []
    for _ in range(resamples):
        indices = np.concatenate(
            [rng.choice(group, size=group.size, replace=True) for group in strata.values()]
        )
        balanced.append(balanced_accuracy_score(y_true[indices], y_pred[indices]))
        macro_f1.append(f1_score(y_true[indices], y_pred[indices], average="macro"))
    return {
        "balanced_accuracy_ci_low": float(np.quantile(balanced, 0.025)),
        "balanced_accuracy_ci_high": float(np.quantile(balanced, 0.975)),
        "macro_f1_ci_low": float(np.quantile(macro_f1, 0.025)),
        "macro_f1_ci_high": float(np.quantile(macro_f1, 0.975)),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    indicators = np.column_stack([(y_true == label).astype(int) for label in classes])
    auroc = np.mean(
        [
            roc_auc_score(indicators[:, index], probabilities[:, index])
            for index in range(len(classes))
        ]
    )
    auprc = np.mean(
        [
            average_precision_score(indicators[:, index], probabilities[:, index])
            for index in range(len(classes))
        ]
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_auroc_ovr": float(auroc),
        "macro_auprc": float(auprc),
    }


def _run_classification(
    root: Path,
    wide: pd.DataFrame,
    config: StatisticalConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    topology_columns = [column for column in wide.columns if "__" in column]
    scales = (config.primary_scale_seconds, config.sensitivity_scale_seconds)
    for scale_index, scale in enumerate(scales):
        subset = wide[
            (wide["scale_seconds"] == scale) & wide["split"].isin(["discovery", "validation"])
        ].copy()
        segment_ids = set(subset["segment_id"].astype(str))
        continuous = _load_continuous_features(root, scale=scale, segment_ids=segment_ids)
        acoustic = np.stack([continuous[value][0] for value in subset["segment_id"]])
        modulation = np.stack([continuous[value][1] for value in subset["segment_id"]])
        topology = subset[topology_columns].to_numpy(float)
        feature_sets = {
            "acoustic": acoustic,
            "modulation": modulation,
            "topology": topology,
            "acoustic_modulation": np.concatenate([acoustic, modulation], axis=1),
            "all": np.concatenate([acoustic, modulation, topology], axis=1),
        }
        train = (subset["split"] == "discovery").to_numpy()
        validation = (subset["split"] == "validation").to_numpy()
        y = subset["group"].astype(str).to_numpy()
        folds = min(config.classification_folds, int(pd.Series(y[train]).value_counts().min()))
        cross_validation = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=config.random_seed
        )
        for feature_index, (feature_set, matrix) in enumerate(feature_sets.items()):
            pipeline = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("variance", VarianceThreshold()),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=3000,
                            solver="lbfgs",
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            )
            search = GridSearchCV(
                pipeline,
                {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
                scoring="f1_macro",
                cv=cross_validation,
                n_jobs=1,
                refit=True,
            )
            search.fit(matrix[train], y[train])
            predicted = search.predict(matrix[validation])
            probabilities = search.predict_proba(matrix[validation])
            classes = search.best_estimator_.named_steps["classifier"].classes_
            metrics = _classification_metrics(y[validation], predicted, probabilities, classes)
            intervals = _bootstrap_metric_intervals(
                y[validation],
                predicted,
                resamples=config.bootstrap_resamples,
                seed=config.random_seed + scale_index * 100 + feature_index,
            )
            matrix_confusion = confusion_matrix(y[validation], predicted, labels=classes)
            key = f"{_scale_token(scale)}_{feature_set}"
            confusion_payload[key] = {
                "scale_seconds": scale,
                "feature_set": feature_set,
                "labels": classes.tolist(),
                "matrix": matrix_confusion.tolist(),
            }
            variance = search.best_estimator_.named_steps["variance"]
            rows.append(
                {
                    "analysis_set": (
                        "primary_validation_180"
                        if scale == config.primary_scale_seconds
                        else "sensitivity_validation_300"
                    ),
                    "scale_seconds": scale,
                    "train_split": "discovery",
                    "test_split": "validation",
                    "feature_set": feature_set,
                    "classifier": "L2 multinomial logistic regression",
                    "selection_metric": "macro_f1",
                    "cv_folds": folds,
                    "best_c": search.best_params_["classifier__C"],
                    "cv_macro_f1": search.best_score_,
                    "input_dimensions": matrix.shape[1],
                    "retained_dimensions": int(np.count_nonzero(variance.get_support())),
                    "n_train": int(np.count_nonzero(train)),
                    "n_validation": int(np.count_nonzero(validation)),
                    **metrics,
                    **intervals,
                }
            )
    return pd.DataFrame(rows), confusion_payload


def _scale_token(scale: float) -> str:
    return f"{int(scale)}s" if float(scale).is_integer() else f"{scale}s"


def _svg_text(
    x: float,
    y: float,
    value: str,
    size: int = 13,
    anchor: str = "start",
) -> str:
    escaped = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#222">{escaped}</text>'
    )


def _write_svg(path: Path, *, width: int, height: int, elements: Sequence[str]) -> None:
    payload = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *elements,
        "</svg>",
    ]
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text("\n".join(payload) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, path)


def _linear_map(values: np.ndarray, low: float, high: float) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if maximum <= minimum:
        return np.full(values.shape, (low + high) / 2)
    return low + (values - minimum) * (high - low) / (maximum - minimum)


def _make_svg_plots(
    root: Path,
    wide: pd.DataFrame,
    classification: pd.DataFrame,
    filtration_path: Path,
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Dependency-free vector plots for minimal analysis environments."""

    output_directory = root / "runs" / "topology_statistics"
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    errors: list[str] = []
    colors = {"classical": "#4472C4", "focus": "#ED7D31", "pop": "#70AD47"}
    groups = sorted(str(value) for value in wide["group"].unique())
    try:
        subset = wide[(wide["split"] == "validation") & (wide["scale_seconds"] == 180.0)]
        discovery = wide[
            (wide["split"] == "discovery") & (wide["scale_seconds"] == 180.0)
        ]
        metric_columns = [column for column in wide.columns if "__" in column]
        scaler = StandardScaler().fit(discovery[metric_columns])
        reducer = PCA(n_components=2, random_state=seed).fit(
            scaler.transform(discovery[metric_columns])
        )
        coordinates = reducer.transform(scaler.transform(subset[metric_columns]))
        x = _linear_map(coordinates[:, 0], 65, 705)
        y = _linear_map(coordinates[:, 1], 465, 55)
        elements = [
            _svg_text(385, 28, "Validation 180 s topology space", size=18, anchor="middle"),
            '<line x1="55" y1="475" x2="720" y2="475" stroke="#555"/>',
            '<line x1="55" y1="475" x2="55" y2="45" stroke="#555"/>',
            _svg_text(385, 510, "PC1 (discovery-fitted)", anchor="middle"),
            _svg_text(18, 250, "PC2", anchor="middle"),
        ]
        for x_value, y_value, group in zip(x, y, subset["group"], strict=True):
            elements.append(
                f'<circle cx="{x_value:.2f}" cy="{y_value:.2f}" r="3.6" '
                f'fill="{colors[group]}" fill-opacity="0.72"/>'
            )
        for index, group in enumerate(groups):
            y_legend = 65 + index * 22
            elements.append(
                f'<circle cx="620" cy="{y_legend}" r="5" fill="{colors[group]}"/>'
            )
            elements.append(_svg_text(632, y_legend + 4, group))
        path = output_directory / "topology_pca_validation_180.svg"
        _write_svg(path, width=760, height=530, elements=elements)
        generated.append(path.relative_to(root).as_posix())
    except Exception as exc:
        errors.append(f"SVG PCA plot failed: {type(exc).__name__}: {exc}")

    try:
        filtration = pd.read_csv(filtration_path)
        subset = filtration[
            (filtration["split"] == "validation") & (filtration["scale_seconds"] == 180.0)
        ]
        views = sorted(subset["view"].unique())
        width, height = 1080, 390
        elements = [
            _svg_text(width / 2, 27, "Validation 180 s directed H1 filtration", 18, "middle")
        ]
        for panel, view in enumerate(views):
            left, right = 45 + panel * 350, 330 + panel * 350
            top, bottom = 58, 320
            selected = subset[subset["view"] == view]
            aggregates = {
                group: (
                    selected[selected["group"] == group]
                    .groupby("threshold")["h1_betti"]
                    .mean()
                    .sort_index()
                )
                for group in sorted(selected["group"].unique())
            }
            maximum = max(1.0, max(float(values.max()) for values in aggregates.values()))
            elements.extend(
                [
                    f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
                    'stroke="#555"/>',
                    f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" '
                    'stroke="#555"/>',
                    _svg_text((left + right) / 2, 50, view, 15, "middle"),
                    _svg_text(left, bottom + 20, "0.50", 11, "middle"),
                    _svg_text(right, bottom + 20, "0.95", 11, "middle"),
                    _svg_text(left - 8, bottom + 4, "0", 11, "end"),
                    _svg_text(left - 8, top + 4, f"{maximum:.2f}", 11, "end"),
                ]
            )
            for group, aggregate in aggregates.items():
                x = left + (aggregate.index.to_numpy(float) - 0.5) / 0.45 * (right - left)
                y = bottom - aggregate.to_numpy(float) / maximum * (bottom - top)
                points = " ".join(
                    f"{x_value:.2f},{y_value:.2f}"
                    for x_value, y_value in zip(x, y, strict=True)
                )
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{colors[group]}" '
                    'stroke-width="2.2"/>'
                )
        for index, group in enumerate(groups):
            x_legend = 420 + index * 110
            elements.append(
                f'<line x1="{x_legend}" y1="365" x2="{x_legend + 22}" y2="365" '
                f'stroke="{colors[group]}" stroke-width="3"/>'
            )
            elements.append(_svg_text(x_legend + 28, 369, group, 12))
        path = output_directory / "h1_filtration_validation_180.svg"
        _write_svg(path, width=width, height=height, elements=elements)
        generated.append(path.relative_to(root).as_posix())
    except Exception as exc:
        errors.append(f"SVG filtration plot failed: {type(exc).__name__}: {exc}")

    try:
        subset = classification[classification["scale_seconds"] == 180.0].sort_values(
            "macro_f1"
        )
        width, height = 760, 390
        elements = [
            _svg_text(width / 2, 28, "Validation 180 s classification", 18, "middle"),
            '<line x1="190" y1="335" x2="720" y2="335" stroke="#555"/>',
        ]
        for index, (_, row) in enumerate(subset.iterrows()):
            y = 65 + index * 54
            x_end = 190 + float(row["macro_f1"]) * 530
            ci_low = 190 + float(row["macro_f1_ci_low"]) * 530
            ci_high = 190 + float(row["macro_f1_ci_high"]) * 530
            elements.extend(
                [
                    _svg_text(178, y + 16, str(row["feature_set"]), 12, "end"),
                    f'<rect x="190" y="{y}" width="{x_end - 190:.2f}" height="25" '
                    'fill="#5B9BD5"/>',
                    f'<line x1="{ci_low:.2f}" y1="{y + 12.5}" x2="{ci_high:.2f}" '
                    f'y2="{y + 12.5}" stroke="#222" stroke-width="1.5"/>',
                    f'<line x1="{ci_low:.2f}" y1="{y + 7}" x2="{ci_low:.2f}" '
                    f'y2="{y + 18}" stroke="#222"/>',
                    f'<line x1="{ci_high:.2f}" y1="{y + 7}" x2="{ci_high:.2f}" '
                    f'y2="{y + 18}" stroke="#222"/>',
                    _svg_text(x_end + 7, y + 17, f"{row['macro_f1']:.3f}", 12),
                ]
            )
        for tick in np.linspace(0, 1, 6):
            x_tick = 190 + tick * 530
            elements.append(
                f'<line x1="{x_tick:.1f}" y1="335" x2="{x_tick:.1f}" y2="341" '
                'stroke="#555"/>'
            )
            elements.append(_svg_text(x_tick, 357, f"{tick:.1f}", 11, "middle"))
        elements.append(_svg_text(455, 380, "Macro-F1 (95% bootstrap CI)", 13, "middle"))
        path = output_directory / "classification_macro_f1_validation_180.svg"
        _write_svg(path, width=width, height=height, elements=elements)
        generated.append(path.relative_to(root).as_posix())
    except Exception as exc:
        errors.append(f"SVG classification plot failed: {type(exc).__name__}: {exc}")
    return generated, errors


def _make_plots(
    root: Path,
    topology: pd.DataFrame,
    wide: pd.DataFrame,
    classification: pd.DataFrame,
    filtration_path: Path,
    *,
    seed: int,
) -> tuple[list[str], list[str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return _make_svg_plots(
            root,
            wide,
            classification,
            filtration_path,
            seed=seed,
        )

    output_directory = root / "runs" / "topology_statistics"
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    errors: list[str] = []
    colors = {"classical": "#4472C4", "focus": "#ED7D31", "pop": "#70AD47"}
    try:
        subset = wide[(wide["split"] == "validation") & (wide["scale_seconds"] == 180.0)]
        discovery = wide[
            (wide["split"] == "discovery") & (wide["scale_seconds"] == 180.0)
        ]
        metric_columns = [column for column in wide.columns if "__" in column]
        scaler = StandardScaler().fit(discovery[metric_columns])
        discovery_matrix = scaler.transform(discovery[metric_columns])
        reducer = PCA(n_components=2, random_state=seed).fit(discovery_matrix)
        coordinates = reducer.transform(scaler.transform(subset[metric_columns]))
        figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
        for group in sorted(subset["group"].unique()):
            mask = subset["group"].to_numpy() == group
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=24,
                alpha=0.72,
                label=group,
                color=colors[group],
            )
        axis.set(title="Validation 180 s topology space", xlabel="PC1", ylabel="PC2")
        axis.legend(frameon=False)
        path = output_directory / "topology_pca_validation_180.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        generated.append(path.relative_to(root).as_posix())
    except Exception as exc:  # plots are secondary to numerical outputs
        errors.append(f"PCA plot failed: {type(exc).__name__}: {exc}")

    try:
        filtration = pd.read_csv(filtration_path)
        subset = filtration[
            (filtration["split"] == "validation") & (filtration["scale_seconds"] == 180.0)
        ]
        views = sorted(subset["view"].unique())
        figure, axes = plt.subplots(1, len(views), figsize=(13.5, 4.2), sharey=False)
        for axis, view in zip(np.atleast_1d(axes), views, strict=True):
            selected = subset[subset["view"] == view]
            for group in sorted(selected["group"].unique()):
                aggregate = (
                    selected[selected["group"] == group]
                    .groupby("threshold")["h1_betti"]
                    .agg(["mean", "sem"])
                    .sort_index()
                )
                axis.plot(aggregate.index, aggregate["mean"], label=group, color=colors[group])
                axis.fill_between(
                    aggregate.index,
                    aggregate["mean"] - aggregate["sem"].fillna(0),
                    aggregate["mean"] + aggregate["sem"].fillna(0),
                    alpha=0.18,
                    color=colors[group],
                )
            axis.set(title=view, xlabel="transition threshold", ylabel="mean H1 Betti")
        axes[0].legend(frameon=False)
        figure.suptitle("Validation 180 s directed H1 filtration")
        figure.tight_layout()
        path = output_directory / "h1_filtration_validation_180.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        generated.append(path.relative_to(root).as_posix())
    except Exception as exc:
        errors.append(f"filtration plot failed: {type(exc).__name__}: {exc}")

    try:
        subset = classification[classification["scale_seconds"] == 180.0].sort_values("macro_f1")
        figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        positions = np.arange(len(subset))
        lower = subset["macro_f1"] - subset["macro_f1_ci_low"]
        upper = subset["macro_f1_ci_high"] - subset["macro_f1"]
        axis.barh(positions, subset["macro_f1"], color="#5B9BD5", xerr=[lower, upper])
        axis.set_yticks(positions, subset["feature_set"])
        axis.set(xlabel="validation macro-F1 (95% stratified bootstrap CI)", xlim=(0, 1))
        path = output_directory / "classification_macro_f1_validation_180.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        generated.append(path.relative_to(root).as_posix())
    except Exception as exc:
        errors.append(f"classification plot failed: {type(exc).__name__}: {exc}")
    return generated, errors


def _format_p(value: float) -> str:
    return f"{value:.3g}" if value >= 0.001 else f"{value:.2e}"


def _write_report(
    path: Path,
    *,
    topology: pd.DataFrame,
    omnibus: pd.DataFrame,
    permanova: pd.DataFrame,
    classification: pd.DataFrame,
    hypothesis_tests: pd.DataFrame,
    hypothesis_summary: dict[str, Any],
    config: StatisticalConfig,
    plots: Sequence[str],
) -> None:
    primary_tests = omnibus[omnibus["analysis_set"] == "primary_validation_180"].sort_values(
        ["p_fdr_bh", "epsilon_squared"], ascending=[True, False]
    )
    significant = primary_tests[primary_tests["p_fdr_bh"] <= config.fdr_q]
    h1_tests = primary_tests[primary_tests["metric"].str.startswith("h1_")]
    h1_significant = h1_tests[h1_tests["p_fdr_bh"] <= config.fdr_q]
    primary_permanova = permanova.loc[
        permanova["analysis_set"] == "primary_validation_180"
    ].iloc[0]
    sensitivity_permanova = permanova.loc[
        permanova["analysis_set"] == "sensitivity_validation_300"
    ].iloc[0]
    primary_classification = classification[
        classification["analysis_set"] == "primary_validation_180"
    ].sort_values("macro_f1", ascending=False)
    view_names = sorted(str(value) for value in topology["view"].unique())
    view_text = "、".join(view_names)
    groups = sorted(str(value) for value in topology["group"].unique())
    group_labels = {"classical": "Classical", "focus": "Focus", "pop": "Pop"}
    group_header = " | ".join(f"{group_labels.get(group, group)} 中位数" for group in groups)
    group_separator = "|".join("---:" for _ in groups)
    replication_text = ""
    if groups == ["classical", "focus"]:
        sensitivity_tests = omnibus[
            omnibus["analysis_set"] == "sensitivity_validation_300"
        ].set_index(["view", "metric"])
        replicated = 0
        for _, row in significant.iterrows():
            sensitivity = sensitivity_tests.loc[(row["view"], row["metric"])]
            primary_delta = float(row["focus_median"] - row["classical_median"])
            sensitivity_delta = float(
                sensitivity["focus_median"] - sensitivity["classical_median"]
            )
            if sensitivity["p_fdr_bh"] <= config.fdr_q and primary_delta * sensitivity_delta > 0:
                replicated += 1
        replication_text = (
            f"其中 {replicated}/{len(significant)} 个在 validation / 300 秒仍通过 FDR 且方向一致。"
        )
    lines = [
        "# Path Homology 拓扑建模与统计分析结果",
        "",
        f"生成日期：{date.today().isoformat()}。主验证集为 validation / 180 秒；"
        "discovery 仅用于探索和分类器拟合，300 秒为敏感性分析，holdout 未参与本轮统计。",
        "",
        "## 数据与方法",
        "",
        f"共建模 {topology['segment_id'].nunique():,} 个片段、{len(topology):,} 个片段-视图。"
        f"对 {view_text} 共 {len(view_names)} 类冻结状态序列构建 top-k 出向概率有向图；"
        "在 0.50–0.95 的固定阈值上计算实系数 GLMY H0/H1，并以链空间包含映射计算"
        "有限过滤的持久秩不变量和条形码。",
        "",
        f"单变量检验采用 Kruskal–Wallis，分析集内 Benjamini–Hochberg FDR q={config.fdr_q:.2f}；"
        f"多变量检验采用 Mahalanobis PERMANOVA（{config.permutations} 次置换）。"
        "分类基线仅在 discovery 拟合与选参，在 validation 报告。",
        "",
        "## 主结果（validation / 180 秒）",
        "",
        f"多变量 PERMANOVA：pseudo-F={primary_permanova['pseudo_f']:.3f}，"
        f"p={_format_p(float(primary_permanova['p_value']))}，n={int(primary_permanova['n_tracks'])}。",
        "",
        f"300 秒敏感性分析的 PERMANOVA 为 pseudo-F={sensitivity_permanova['pseudo_f']:.3f}，"
        f"p={_format_p(float(sensitivity_permanova['p_value']))}，"
        f"n={int(sensitivity_permanova['n_tracks'])}。",
        "",
        f"{len(primary_tests)} 个预设视图-指标检验中，FDR q≤{config.fdr_q:.2f} 的结果有 "
        f"{len(significant)} 个。"
        f"{replication_text}效应最大的显著结果如下：",
        "",
        f"| 视图 | 指标 | {group_header} | ε² | FDR p |",
        f"|---|---|{group_separator}|---:|---:|",
    ]
    for _, row in significant.sort_values("epsilon_squared", ascending=False).head(12).iterrows():
        group_values = " | ".join(f"{row[f'{group}_median']:.3f}" for group in groups)
        lines.append(
            f"| {row['view']} | {row['metric']} | {group_values} | "
            f"{row['epsilon_squared']:.3f} | {_format_p(row['p_fdr_bh'])} |"
        )
    if significant.empty:
        empty_groups = " | ".join("—" for _ in groups)
        lines.append(
            f"| — | 未发现通过预设 FDR 阈值的单变量结果 | {empty_groups} | — | — |"
        )
    lines.extend(
        [
            "",
            "### H1 专项结果",
            "",
            f"{len(h1_tests)} 个 H1 视图-指标检验中有 {len(h1_significant)} 个通过 FDR。"
            "本轮差异主要由 H0 连通结构和图转移描述子驱动；在当前 ≥0.50 的稀疏过滤下，"
            "多数曲目的 H1 为 0，因此不能将多变量显著性解释为有向一维洞的组间差异。",
        ]
    )
    if hypothesis_summary.get("applicable", True):
        core_hypothesis = hypothesis_tests[
            hypothesis_tests["analysis_role"] == "confirmatory_core"
        ]
        verdict_labels = {
            "supported": "支持",
            "partially_supported": "部分支持",
            "not_supported": "不支持",
        }
        lines.extend(
            [
                "",
                "## H2 核心有向拓扑假设",
                "",
                "主检验固定为 validation / 180 秒 / modulation / Focus vs Pop。",
                "",
                "| 子假设 | Focus 中位数 | Pop 中位数 | Rank-biserial | 单侧 p | FDR q | 判定 |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        endpoint_labels = {
            "path_entropy": "Path entropy 更低",
            "directed_recurrence": "Directed recurrence 更高",
            "beta1_profile_dispersion": "β1 分布更集中",
        }
        for _, row in core_hypothesis.iterrows():
            lines.append(
                f"| {endpoint_labels[row['endpoint']]} | {row['focus_median']:.4f} | "
                f"{row['comparator_median']:.4f} | "
                f"{row['rank_biserial_focus_minus_comparator']:.3f} | "
                f"{_format_p(float(row['p_one_sided']))} | "
                f"{_format_p(float(row['p_fdr_bh']))} | "
                f"{verdict_labels[row['verdict']]} |"
            )
        composite = str(hypothesis_summary["core_verdict"])
        lines.extend(
            [
                "",
                f"三个子假设的合取判定为：**{verdict_labels[composite]}**。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 原 H2 专项假设的适用性",
                "",
                "原 H2 专项检验预先固定为 Focus vs Pop。由于新的规范数据集已移除 Pop，"
                "本轮不执行该专项检验，也不将比较组事后改为 Classical；其状态记为"
                "**不适用（comparator absent）**。通用 H0/H1 两组统计仍按冻结设置执行。",
            ]
        )
    lines.extend(
        [
            "",
            "## 分类基线",
            "",
            "| 特征 | Macro-F1 | 95% CI | 平衡准确率 | Macro-AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in primary_classification.iterrows():
        lines.append(
            f"| {row['feature_set']} | {row['macro_f1']:.3f} | "
            f"[{row['macro_f1_ci_low']:.3f}, {row['macro_f1_ci_high']:.3f}] | "
            f"{row['balanced_accuracy']:.3f} | {row['macro_auroc_ovr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这些结果描述 Focus 与 Classical 两组音频状态转移拓扑的分布差异，"
            "不构成注意力提升或因果效果证据。"
            "300 秒与扩展过滤结果用于敏感性分析。holdout 仅含 Focus 曲目、没有对照组，"
            "因此不用于本轮组间假设检验。",
        ]
    )
    if plots:
        lines.extend(["", "## 图形输出", ""])
        lines.extend(f"- `{item}`" for item in plots)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, path)


def run_statistics(*, root: Path, topology_manifest: Path) -> dict[str, Any]:
    matplotlib_cache = root / "runs" / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    config = _load_config(root)
    topology = _load_topology_frame(topology_manifest)
    wide = _wide_topology(topology)
    holdout_rows = int(np.count_nonzero(topology["split"] == "holdout"))
    omnibus, pairwise = _omnibus_and_pairwise(topology)
    permanova = _run_permanova(wide, config)
    classification, confusion_payload = _run_classification(root, wide, config)

    metadata = root / "metadata"
    omnibus_path = metadata / "topology_statistical_tests.csv"
    pairwise_path = metadata / "topology_pairwise_tests.csv"
    permanova_path = metadata / "topology_permanova.csv"
    classification_path = metadata / "classification_results.csv"
    confusion_path = metadata / "classification_confusion_matrices.json"
    filtration_path = metadata / "topology_filtration.csv"
    _write_frame_atomic(omnibus_path, omnibus)
    _write_frame_atomic(pairwise_path, pairwise)
    _write_frame_atomic(permanova_path, permanova)
    _write_frame_atomic(classification_path, classification)
    _write_json_atomic(confusion_path, confusion_payload)

    plots, plot_errors = _make_plots(
        root,
        topology,
        wide,
        classification,
        filtration_path,
        seed=config.random_seed,
    )
    groups = set(str(value) for value in topology["group"].unique())
    if {"focus", "pop"}.issubset(groups):
        from topology.hypothesis import run_hypothesis_analysis

        hypothesis_summary, hypothesis_tests = run_hypothesis_analysis(
            root=root,
            topology=topology,
            fdr_q=config.fdr_q,
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.random_seed,
        )
        plots.extend(hypothesis_summary["outputs"]["plots"])
    else:
        hypothesis_summary = {
            "generated_at": date.today().isoformat(),
            "applicable": False,
            "reason": "the preregistered H2 comparator Pop is absent from the canonical dataset",
            "core_verdict": "not_applicable_comparator_absent",
            "core_supported_endpoints": 0,
            "outputs": {"plots": []},
        }
        hypothesis_tests = pd.DataFrame()
        _write_json_atomic(metadata / "topology_hypothesis_summary.json", hypothesis_summary)
    report_path = root / "docs" / "topology-analysis-results.md"
    _write_report(
        report_path,
        topology=topology,
        omnibus=omnibus,
        permanova=permanova,
        classification=classification,
        hypothesis_tests=hypothesis_tests,
        hypothesis_summary=hypothesis_summary,
        config=config,
        plots=plots,
    )
    primary = omnibus[omnibus["analysis_set"] == "primary_validation_180"]
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "config": asdict(config),
        "config_sha256": _json_hash(asdict(config)),
        "topology_manifest": topology_manifest.relative_to(root).as_posix(),
        "topology_manifest_sha256": _sha256(topology_manifest),
        "feature_manifest_sha256": _sha256(root / "metadata" / "feature_segments.csv"),
        "analysis_track_counts": {
            name: int(
                wide[(wide["split"] == split) & (wide["scale_seconds"] == scale)].shape[0]
            )
            for name, split, scale, _ in ANALYSIS_SETS
        },
        "holdout_segment_views_available": holdout_rows,
        "holdout_segment_views_analyzed": 0,
        "primary_omnibus_tests": int(len(primary)),
        "primary_fdr_discoveries": int(np.count_nonzero(primary["p_fdr_bh"] <= config.fdr_q)),
        "h2_core_verdict": hypothesis_summary["core_verdict"],
        "h2_core_supported_endpoints": hypothesis_summary["core_supported_endpoints"],
        "h2_core_applicable": bool(hypothesis_summary.get("applicable", True)),
        "outputs": {
            "omnibus": omnibus_path.relative_to(root).as_posix(),
            "pairwise": pairwise_path.relative_to(root).as_posix(),
            "permanova": permanova_path.relative_to(root).as_posix(),
            "classification": classification_path.relative_to(root).as_posix(),
            "confusion_matrices": confusion_path.relative_to(root).as_posix(),
            "hypothesis_summary": "metadata/topology_hypothesis_summary.json",
            "report": report_path.relative_to(root).as_posix(),
            "plots": plots,
        },
        "plot_errors": plot_errors,
    }
    artifact_paths = [
        omnibus_path,
        pairwise_path,
        permanova_path,
        classification_path,
        confusion_path,
        root / "metadata" / "topology_hypothesis_summary.json",
        report_path,
        *(root / Path(item) for item in plots),
    ]
    payload["output_sha256"] = {
        path.relative_to(root).as_posix(): _sha256(path) for path in artifact_paths
    }
    summary_path = metadata / "topology_statistics_summary.json"
    _write_json_atomic(summary_path, payload)
    return payload
