from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from topology.multiview_fusion import (
    DiscoveryMahalanobisBlock,
    equal_block_fusion,
    paired_incremental_permutation,
    permutation_pseudo_f,
    stratified_bootstrap_differences,
)
from topology.statistics import TOPOLOGY_METRICS, _pseudo_f_statistic, benjamini_hochberg

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
OUTPUT = ROOT / "runs" / "multiscale_hierarchical_fusion"
FIGURES = OUTPUT / "figures"

VIEW_FILES = {
    "pitch": METADATA / "pitch_v2_topology_segments.csv",
    "rhythm": METADATA / "rhythm_topology_segments.csv",
    "modulation": METADATA / "modulation_tertile_topology_segments.csv",
    "structure": METADATA / "structure_topology_segments.csv",
}
PHASE_FILE = METADATA / "phase_lifted_path_homology_features.csv"
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
LOCAL_VIEWS = ("pitch", "rhythm", "modulation")
PRIMARY_PHASE = ("path_acoustic_phase", "path_chroma_phase")
ALL_PHASE = (*PRIMARY_PHASE, "path_rhythm_phase")
SCALES = (180.0, 300.0)
PERMUTATIONS = 999
BOOTSTRAPS = 1000
SEED = 20260716

PERMANOVA_PATH = METADATA / "multiscale_hierarchical_fusion_permanova.csv"
INCREMENTAL_PATH = METADATA / "multiscale_hierarchical_fusion_incremental.csv"
CLASSIFICATION_PATH = METADATA / "multiscale_hierarchical_fusion_classification.csv"
CLASSIFICATION_DELTA_PATH = (
    METADATA / "multiscale_hierarchical_fusion_classification_deltas.csv"
)
CORRELATION_PATH = METADATA / "multiscale_hierarchical_fusion_correlations.csv"
RESIDUAL_PATH = METADATA / "multiscale_hierarchical_fusion_residuals.csv"
PHASE_SENSITIVITY_PATH = (
    METADATA / "multiscale_hierarchical_fusion_phase_sensitivity.csv"
)
HOLDOUT_PATH = METADATA / "multiscale_hierarchical_fusion_holdout_descriptive.csv"
SUMMARY_PATH = METADATA / "multiscale_hierarchical_fusion_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    canonical_index: pd.MultiIndex | None = None
    for view, path in VIEW_FILES.items():
        frame = pd.read_csv(path)
        required = set(IDENTITY) | set(TOPOLOGY_METRICS) | {"status"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{view} is missing columns: {sorted(missing)}")
        if frame.duplicated(IDENTITY).any():
            raise RuntimeError(f"{view} contains duplicate identity rows")
        if (frame["status"] == "failed").any():
            raise RuntimeError(f"{view} contains failed rows")
        indexed = frame.set_index(IDENTITY).sort_index()
        if canonical_index is None:
            canonical_index = indexed.index
        elif not indexed.index.equals(canonical_index):
            raise RuntimeError(f"{view} identities do not align")
        numeric = indexed.loc[:, TOPOLOGY_METRICS].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise RuntimeError(f"{view} contains missing topology metrics")
        frames[view] = indexed
    assert canonical_index is not None

    phase = pd.read_csv(PHASE_FILE)
    required_phase = set(IDENTITY) | {"representation", "loop_score"}
    if required_phase - set(phase.columns):
        raise RuntimeError("phase file is missing required columns")
    if phase.duplicated([*IDENTITY, "representation"]).any():
        raise RuntimeError("phase file contains duplicate identity/representation rows")
    pivot = phase.pivot(index=IDENTITY, columns="representation", values="loop_score")
    pivot = pivot.reindex(canonical_index)
    if pivot.loc[:, list(ALL_PHASE)].isna().sum().max() > 2:
        raise RuntimeError("unexpected phase missingness")
    identity = canonical_index.to_frame(index=False)
    return identity, frames, pivot


def _mask(identity: pd.DataFrame, split: str, scale: float) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _fit_all_blocks(
    identity: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    phase: pd.DataFrame,
    scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    discovery = _mask(identity, "discovery", scale)
    blocks: dict[str, np.ndarray] = {}
    ranks: dict[str, int] = {}
    missing: dict[str, int] = {}
    for view, frame in frames.items():
        raw = frame.loc[:, TOPOLOGY_METRICS].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        blocks[view] = transformer.transform(raw)
        ranks[view] = transformer.effective_rank
        missing[view] = 0
    for representation in ALL_PHASE:
        raw = phase.loc[:, [representation]].to_numpy(float)
        missing[representation] = int(np.isnan(raw[discovery]).sum())
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        blocks[representation] = transformer.transform(raw)
        ranks[representation] = transformer.effective_rank
    return blocks, ranks, missing


def _feature_sets(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    local = equal_block_fusion([blocks[name] for name in LOCAL_VIEWS])
    phase_primary = equal_block_fusion([blocks[name] for name in PRIMARY_PHASE])
    phase_all = equal_block_fusion([blocks[name] for name in ALL_PHASE])
    return {
        "L": local,
        "P": phase_primary,
        "LP": equal_block_fusion([local, phase_primary]),
        "S": blocks["structure"],
        "LPS": equal_block_fusion([local, phase_primary, blocks["structure"]]),
        "P_all3": phase_all,
        "LP_all3": equal_block_fusion([local, phase_all]),
        "LPS_all3": equal_block_fusion([local, phase_all, blocks["structure"]]),
    }


def _macro_auroc(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    values = [
        roc_auc_score((labels == label).astype(int), probabilities[:, index])
        for index, label in enumerate(classes)
    ]
    return float(np.mean(values))


def _macro_auprc(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    values = [
        average_precision_score((labels == label).astype(int), probabilities[:, index])
        for index, label in enumerate(classes)
    ]
    return float(np.mean(values))


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "macro_auroc_ovr": _macro_auroc(labels, probabilities, classes),
        "macro_auprc": _macro_auprc(labels, probabilities, classes),
    }


def _fit_classifier(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
) -> tuple[GridSearchCV, np.ndarray, np.ndarray, np.ndarray]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=SEED,
        ),
        {"C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    search.fit(train, train_labels)
    predictions = search.predict(validation)
    probabilities = search.predict_proba(validation)
    return search, predictions, probabilities, search.best_estimator_.classes_


def _metric_functions(classes: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": lambda y, p, q: float(balanced_accuracy_score(y, p)),
        "macro_f1": lambda y, p, q: float(f1_score(y, p, average="macro")),
        "macro_auroc_ovr": lambda y, p, q: _macro_auroc(y, q, classes),
        "macro_auprc": lambda y, p, q: _macro_auprc(y, q, classes),
    }


def _bootstrap_single(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> dict[str, float]:
    strata = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    rng = np.random.default_rng(seed)
    names = ("balanced_accuracy", "macro_f1", "macro_auroc_ovr", "macro_auprc")
    samples = {name: np.empty(BOOTSTRAPS, dtype=float) for name in names}
    for index in range(BOOTSTRAPS):
        selected = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in strata.values()]
        )
        scores = _classification_metrics(
            labels[selected], predictions[selected], probabilities[selected], classes
        )
        for name, value in scores.items():
            samples[name][index] = value
    output: dict[str, float] = {}
    for name, values in samples.items():
        output[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        output[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return output


def _centroid_distance(matrix: np.ndarray, labels: np.ndarray) -> float:
    groups = np.unique(labels)
    if groups.size != 2:
        raise RuntimeError("centroid distance requires two groups")
    return float(
        np.linalg.norm(matrix[labels == groups[0]].mean(0) - matrix[labels == groups[1]].mean(0))
    )


def _ellipse(ax: Any, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * 2.4477 * np.sqrt(np.maximum(values, 0.0))
    center = np.mean(points, axis=0)
    ax.add_patch(
        Ellipse(
            center,
            width,
            height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.6,
        )
    )


def _save_figure(figure: Any, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _plot_permanova(frame: pd.DataFrame) -> None:
    order = ["L", "P", "LP", "S", "LPS"]
    labels = ["Local L", "Phase P", "L + P", "Structure S", "L + P + S"]
    figure, ax = plt.subplots(figsize=(9.0, 4.9), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    for offset, scale, color in (
        (-width / 2, 180.0, "#4472C4"),
        (width / 2, 300.0, "#70AD47"),
    ):
        subset = frame[frame["scale_seconds"] == scale].set_index("feature_set")
        values = np.array([subset.loc[name, "pseudo_f"] for name in order])
        bars = ax.bar(x + offset, values, width, color=color, label=f"{int(scale)} s")
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.12,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation permutation pseudo-F")
    ax.set_title("Hierarchical multiscale group separation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(figure, "hierarchical_permanova")


def _plot_incremental(frame: pd.DataFrame) -> None:
    labels = {"LP_minus_L": "Add phase: LP - L", "LPS_minus_LP": "Add structure: LPS - LP"}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharex=True, constrained_layout=True)
    colors = {180.0: "#4472C4", 300.0: "#70AD47"}
    for ax, comparison in zip(axes, labels, strict=True):
        subset = frame[frame["comparison"] == comparison].sort_values("scale_seconds")
        for y, row in enumerate(subset.itertuples(index=False)):
            ax.plot(
                [row.null_ci_low, row.null_ci_high],
                [y, y],
                color=colors[row.scale_seconds],
                linewidth=5,
                alpha=0.35,
            )
            ax.scatter(
                row.delta_pseudo_f,
                y,
                color=colors[row.scale_seconds],
                marker="D",
                s=55,
                zorder=3,
            )
            ax.text(
                row.delta_pseudo_f,
                y + 0.18,
                f"Δ={row.delta_pseudo_f:.2f}, p={row.p_value_one_sided:.3f}",
                ha="center",
                fontsize=8,
            )
        ax.axvline(0.0, color="#666666", linewidth=0.9, linestyle="--")
        ax.set_yticks([0, 1], ["180 s", "300 s"])
        ax.set_xlabel("Observed Δpseudo-F; thick line = null 95% interval")
        ax.set_title(labels[comparison])
        ax.grid(axis="x", alpha=0.2)
    _save_figure(figure, "hierarchical_incremental_tests")


def _plot_classification(frame: pd.DataFrame) -> None:
    order = ["L", "P", "LP", "S", "LPS"]
    labels = ["L", "P", "L+P", "S", "L+P+S"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, constrained_layout=True)
    for ax, scale in zip(axes, SCALES, strict=True):
        subset = frame[frame["scale_seconds"] == scale].set_index("feature_set")
        y = np.arange(len(order))
        balanced = np.array([subset.loc[name, "balanced_accuracy"] for name in order])
        auroc = np.array([subset.loc[name, "macro_auroc_ovr"] for name in order])
        ax.scatter(balanced, y - 0.10, marker="o", color="#4472C4", label="Balanced accuracy")
        ax.scatter(auroc, y + 0.10, marker="s", color="#D95F02", label="Macro AUROC")
        for index, value in enumerate(balanced):
            ax.text(value + 0.008, index - 0.10, f"{value:.3f}", va="center", fontsize=8)
        for index, value in enumerate(auroc):
            ax.text(value + 0.008, index + 0.10, f"{value:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xlim(0.48, 1.04)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Validation score")
        ax.set_title(f"{int(scale)} s")
        ax.grid(axis="x", alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
    )
    figure.suptitle("Discovery-trained classification by fusion stage")
    _save_figure(figure, "hierarchical_classification")


def _plot_correlations(frame: pd.DataFrame) -> None:
    blocks = ["L", "P", "S"]
    figure, axes = plt.subplots(1, 2, figsize=(8.7, 3.9), constrained_layout=True)
    image = None
    for ax, scale in zip(axes, SCALES, strict=True):
        subset = frame[frame["scale_seconds"] == scale]
        matrix = np.eye(3)
        for row in subset.itertuples(index=False):
            left = blocks.index(row.block_a)
            right = blocks.index(row.block_b)
            matrix[left, right] = matrix[right, left] = row.spearman_rho
        image = ax.imshow(matrix, vmin=-0.1, vmax=1.0, cmap="Blues")
        ax.set_xticks(range(3), blocks)
        ax.set_yticks(range(3), blocks)
        ax.set_title(f"{int(scale)} s")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label="Spearman distance correlation")
    figure.suptitle("Agreement among local, phase, and structure spaces")
    _save_figure(figure, "hierarchical_distance_correlations")


def _plot_pca(payload: dict[str, Any]) -> None:
    colors = {"classical": "#4472C4", "focus": "#D95F02"}
    labels = {"classical": "Classical", "focus": "Open Focus"}
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for ax, name in zip(axes, ("L", "LP", "LPS"), strict=True):
        points = payload[name]["validation"]
        groups = payload[name]["groups"]
        variance = payload[name]["variance"]
        for group in ("classical", "focus"):
            subset = points[groups == group]
            ax.scatter(subset[:, 0], subset[:, 1], s=22, alpha=0.55, color=colors[group])
            _ellipse(ax, subset, colors[group])
            center = subset.mean(0)
            ax.scatter(center[0], center[1], marker="X", s=75, color=colors[group])
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.axvline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.set_title(name.replace("LP", "L + P").replace("S", " + S"))
        ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[group], label=label)
        for group, label in labels.items()
    ]
    figure.legend(handles=handles, frameon=False, loc="center right")
    figure.suptitle("Validation 180 s in discovery-fitted fusion spaces")
    _save_figure(figure, "hierarchical_pca_validation_180")


def _plot_residuals(frame: pd.DataFrame) -> None:
    order = ["P_given_L", "S_given_LP"]
    labels = ["Phase residual P | L", "Structure residual S | L+P"]
    figure, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    x = np.arange(2)
    width = 0.36
    for offset, scale, color in (
        (-width / 2, 180.0, "#4472C4"),
        (width / 2, 300.0, "#70AD47"),
    ):
        subset = frame[frame["scale_seconds"] == scale].set_index("conditional_test")
        values = np.array([subset.loc[name, "pseudo_f"] for name in order])
        bars = ax.bar(x + offset, values, width, color=color, label=f"{int(scale)} s")
        for bar, name, value in zip(bars, order, values, strict=True):
            p_value = subset.loc[name, "p_value"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.05,
                f"F={value:.2f}\np={p_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Residual-space pseudo-F")
    ax.set_title("Conditional group separation after discovery-fitted regression")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(figure, "hierarchical_conditional_residuals")


def main() -> int:
    identity, frames, phase = _load_inputs()
    permanova_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    classification_delta_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    phase_sensitivity_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    ranks_by_scale: dict[str, Any] = {}
    missing_by_scale: dict[str, Any] = {}
    pca_payload: dict[str, Any] = {}

    for scale_index, scale in enumerate(SCALES):
        blocks, ranks, missing = _fit_all_blocks(identity, frames, phase, scale)
        sets = _feature_sets(blocks)
        masks = {
            split: _mask(identity, split, scale)
            for split in ("discovery", "validation", "holdout")
        }
        labels = {
            split: identity.loc[mask, "group"].astype(str).to_numpy()
            for split, mask in masks.items()
        }
        ranks_by_scale[str(int(scale))] = ranks
        missing_by_scale[str(int(scale))] = missing

        primary_names = ("L", "P", "LP", "S", "LPS")
        for feature_index, name in enumerate(primary_names):
            matrix = sets[name][masks["validation"]]
            test = permutation_pseudo_f(
                matrix,
                labels["validation"],
                permutations=PERMUTATIONS,
                seed=SEED + scale_index * 100 + feature_index,
            )
            permanova_rows.append(
                {
                    "analysis_set": "exploratory_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "input_dimensions": matrix.shape[1],
                    "n_validation": matrix.shape[0],
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        comparisons = (
            ("LP_minus_L", "LP", "L"),
            ("LPS_minus_LP", "LPS", "LP"),
        )
        for comparison_index, (name, candidate, baseline) in enumerate(comparisons):
            test = paired_incremental_permutation(
                sets[candidate][masks["validation"]],
                sets[baseline][masks["validation"]],
                labels["validation"],
                permutations=PERMUTATIONS,
                seed=SEED + 200 + scale_index * 10 + comparison_index,
            )
            incremental_rows.append(
                {
                    "analysis_set": "exploratory_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "comparison": name,
                    "candidate": candidate,
                    "baseline": baseline,
                    "alternative": "candidate pseudo-F > baseline pseudo-F",
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        holdout_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for feature_index, name in enumerate(primary_names):
            search, predicted, probability, classes = _fit_classifier(
                sets[name][masks["discovery"]],
                labels["discovery"],
                sets[name][masks["validation"]],
            )
            scores = _classification_metrics(
                labels["validation"], predicted, probability, classes
            )
            intervals = _bootstrap_single(
                labels["validation"],
                predicted,
                probability,
                classes,
                SEED + 400 + scale_index * 100 + feature_index,
            )
            classification_rows.append(
                {
                    "analysis_set": "exploratory_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "best_c": float(search.best_params_["C"]),
                    "discovery_cv_macro_f1": float(search.best_score_),
                    "n_train": int(masks["discovery"].sum()),
                    "n_validation": int(masks["validation"].sum()),
                    **scores,
                    **intervals,
                }
            )
            predictions[name] = (predicted, probability, classes)
            hold_predicted = search.predict(sets[name][masks["holdout"]])
            hold_probability = search.predict_proba(sets[name][masks["holdout"]])
            holdout_predictions[name] = (hold_predicted, hold_probability, classes)

        for comparison_index, (name, candidate, baseline) in enumerate(comparisons):
            cand_pred, cand_prob, classes = predictions[candidate]
            base_pred, base_prob, base_classes = predictions[baseline]
            if not np.array_equal(classes, base_classes):
                raise RuntimeError("classifier class orders do not match")
            candidate_scores = _classification_metrics(
                labels["validation"], cand_pred, cand_prob, classes
            )
            baseline_scores = _classification_metrics(
                labels["validation"], base_pred, base_prob, classes
            )
            intervals = stratified_bootstrap_differences(
                labels["validation"],
                cand_pred,
                cand_prob,
                base_pred,
                base_prob,
                metric_functions=_metric_functions(classes),
                resamples=BOOTSTRAPS,
                seed=SEED + 600 + scale_index * 10 + comparison_index,
            )
            for metric, interval in intervals.items():
                classification_delta_rows.append(
                    {
                        "analysis_set": "exploratory_validation_180"
                        if scale == 180.0
                        else "sensitivity_validation_300",
                        "scale_seconds": scale,
                        "comparison": name,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "candidate_score": candidate_scores[metric],
                        "baseline_score": baseline_scores[metric],
                        "delta": candidate_scores[metric] - baseline_scores[metric],
                        "bootstrap_resamples": BOOTSTRAPS,
                        **interval,
                    }
                )

        validation_distances = {
            name: pdist(sets[name][masks["validation"]], metric="euclidean")
            for name in ("L", "P", "S")
        }
        for left_index, left in enumerate(("L", "P", "S")):
            for right in ("L", "P", "S")[left_index + 1 :]:
                result = spearmanr(validation_distances[left], validation_distances[right])
                correlation_rows.append(
                    {
                        "analysis_set": "exploratory_validation_180"
                        if scale == 180.0
                        else "sensitivity_validation_300",
                        "scale_seconds": scale,
                        "block_a": left,
                        "block_b": right,
                        "spearman_rho": float(result.statistic),
                        "p_value_descriptive": float(result.pvalue),
                        "n_pairwise_distances": validation_distances[left].size,
                    }
                )

        conditional_specs = (
            ("P_given_L", "L", "P"),
            ("S_given_LP", "LP", "S"),
        )
        for residual_index, (name, predictor, outcome) in enumerate(conditional_specs):
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 17), cv=5).fit(
                sets[predictor][masks["discovery"]],
                sets[outcome][masks["discovery"]],
            )
            residual = sets[outcome][masks["validation"]] - ridge.predict(
                sets[predictor][masks["validation"]]
            )
            test = permutation_pseudo_f(
                residual,
                labels["validation"],
                permutations=PERMUTATIONS,
                seed=SEED + 800 + scale_index * 10 + residual_index,
            )
            residual_rows.append(
                {
                    "analysis_set": "exploratory_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "conditional_test": name,
                    "predictor": predictor,
                    "outcome": outcome,
                    "ridge_alpha": float(ridge.alpha_),
                    "discovery_r2": float(
                        ridge.score(
                            sets[predictor][masks["discovery"]],
                            sets[outcome][masks["discovery"]],
                        )
                    ),
                    "validation_r2": float(
                        ridge.score(
                            sets[predictor][masks["validation"]],
                            sets[outcome][masks["validation"]],
                        )
                    ),
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        sensitivity_specs = (
            ("P", "P_all3"),
            ("LP", "LP_all3"),
            ("LPS", "LPS_all3"),
        )
        for pair_index, (primary, all_three) in enumerate(sensitivity_specs):
            phase_variants = (
                ("primary_acoustic_chroma", primary),
                ("all_three", all_three),
            )
            for phase_role, feature_set in phase_variants:
                matrix = sets[feature_set][masks["validation"]]
                test = permutation_pseudo_f(
                    matrix,
                    labels["validation"],
                    permutations=PERMUTATIONS,
                    seed=(
                        SEED
                        + 1000
                        + scale_index * 20
                        + pair_index * 2
                        + (phase_role == "all_three")
                    ),
                )
                phase_sensitivity_rows.append(
                    {
                        "analysis_set": "exploratory_phase_definition_sensitivity",
                        "scale_seconds": scale,
                        "fusion_stage": primary,
                        "phase_definition": phase_role,
                        "feature_set": feature_set,
                        "permutations": PERMUTATIONS,
                        **test,
                    }
                )

        for name in primary_names:
            matrix = sets[name][masks["holdout"]]
            predicted, probability, classes = holdout_predictions[name]
            scores = _classification_metrics(
                labels["holdout"], predicted, probability, classes
            )
            holdout_rows.append(
                {
                    "analysis_set": "opened_holdout_descriptive_only",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "n_holdout": matrix.shape[0],
                    "pseudo_f_descriptive": float(
                        _pseudo_f_statistic(matrix, labels["holdout"])
                    ),
                    "centroid_distance_descriptive": _centroid_distance(
                        matrix, labels["holdout"]
                    ),
                    **scores,
                }
            )

        if scale == 180.0:
            for name in ("L", "LP", "LPS"):
                pca = PCA(n_components=2, random_state=SEED).fit(
                    sets[name][masks["discovery"]]
                )
                pca_payload[name] = {
                    "validation": pca.transform(sets[name][masks["validation"]]),
                    "groups": labels["validation"],
                    "variance": pca.explained_variance_ratio_,
                }

    permanova = pd.DataFrame(permanova_rows)
    for _, indices in permanova.groupby("scale_seconds").groups.items():
        permanova.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            permanova.loc[indices, "p_value"].to_numpy(float)
        )
    incremental = pd.DataFrame(incremental_rows)
    for _, indices in incremental.groupby("scale_seconds").groups.items():
        incremental.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            incremental.loc[indices, "p_value_one_sided"].to_numpy(float)
        )
    classification = pd.DataFrame(classification_rows)
    classification_deltas = pd.DataFrame(classification_delta_rows)
    correlations = pd.DataFrame(correlation_rows)
    residuals = pd.DataFrame(residual_rows)
    for _, indices in residuals.groupby("scale_seconds").groups.items():
        residuals.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            residuals.loc[indices, "p_value"].to_numpy(float)
        )
    phase_sensitivity = pd.DataFrame(phase_sensitivity_rows)
    holdout = pd.DataFrame(holdout_rows)

    outputs = {
        PERMANOVA_PATH: permanova,
        INCREMENTAL_PATH: incremental,
        CLASSIFICATION_PATH: classification,
        CLASSIFICATION_DELTA_PATH: classification_deltas,
        CORRELATION_PATH: correlations,
        RESIDUAL_PATH: residuals,
        PHASE_SENSITIVITY_PATH: phase_sensitivity,
        HOLDOUT_PATH: holdout,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    _plot_permanova(permanova)
    _plot_incremental(incremental)
    _plot_classification(classification)
    _plot_correlations(correlations)
    _plot_pca(pca_payload)
    _plot_residuals(residuals)

    primary = permanova[permanova["scale_seconds"] == 180.0].set_index("feature_set")
    increments = incremental[incremental["scale_seconds"] == 180.0].set_index("comparison")
    primary_classification = classification[
        classification["scale_seconds"] == 180.0
    ].set_index("feature_set")
    primary_residuals = residuals[residuals["scale_seconds"] == 180.0].set_index(
        "conditional_test"
    )
    artifacts = [*outputs, *sorted(FIGURES.glob("*"))]
    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "exploratory L -> L+P -> L+P+S hierarchical multiscale fusion",
        "evidence_role": {
            "validation_180": "primary exploratory analysis",
            "validation_300": "same-track duration sensitivity, not independent replication",
            "holdout": "already opened; descriptive only, no p-values or tuning",
        },
        "design": {
            "L": "equal-block fusion of pitch, rhythm, modulation",
            "P_primary": "equal-block fusion of acoustic and chroma phase loop_score",
            "P_sensitivity": "equal-block fusion of acoustic, chroma, rhythm phase loop_score",
            "LP_weights": {"L": 0.5, "P": 0.5},
            "LPS_weights": {"L": 1 / 3, "P": 1 / 3, "S": 1 / 3},
            "block_transform": "discovery-fitted rank-normalized Mahalanobis coordinates",
            "permutations": PERMUTATIONS,
            "bootstrap_resamples": BOOTSTRAPS,
            "seed": SEED,
        },
        "sample_counts_per_scale": {
            split: int(_mask(identity, split, 180.0).sum())
            for split in ("discovery", "validation", "holdout")
        },
        "effective_ranks": ranks_by_scale,
        "discovery_phase_missing_imputed": missing_by_scale,
        "primary_180": {
            "permanova": {
                name: primary.loc[name, ["pseudo_f", "p_value"]].to_dict()
                for name in ("L", "P", "LP", "S", "LPS")
            },
            "increments": {
                name: increments.loc[
                    name, ["delta_pseudo_f", "p_value_one_sided", "p_fdr_bh"]
                ].to_dict()
                for name in ("LP_minus_L", "LPS_minus_LP")
            },
            "classification": {
                name: primary_classification.loc[
                    name, ["balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
                ].to_dict()
                for name in ("L", "P", "LP", "S", "LPS")
            },
            "conditional_residuals": {
                name: primary_residuals.loc[
                    name, ["pseudo_f", "p_value", "p_fdr_bh", "validation_r2"]
                ].to_dict()
                for name in ("P_given_L", "S_given_LP")
            },
        },
        "decision": {
            "phase_positive_increment_over_L": bool(
                increments.loc["LP_minus_L", "delta_pseudo_f"] > 0
                and increments.loc["LP_minus_L", "p_fdr_bh"] <= 0.05
            ),
            "structure_positive_increment_over_LP": bool(
                increments.loc["LPS_minus_LP", "delta_pseudo_f"] > 0
                and increments.loc["LPS_minus_LP", "p_fdr_bh"] <= 0.05
            ),
            "results_are_confirmatory": False,
            "holdout_used_for_model_selection": False,
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (*VIEW_FILES.values(), PHASE_FILE)
        },
        "artifacts": [path.relative_to(ROOT).as_posix() for path in artifacts],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
