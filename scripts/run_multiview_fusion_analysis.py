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
    hierarchical_fusion,
    paired_incremental_permutation,
    permutation_pseudo_f,
    stratified_bootstrap_differences,
)
from topology.statistics import TOPOLOGY_METRICS, benjamini_hochberg

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
OUTPUT = ROOT / "runs" / "multiview_fusion"
FIGURES = OUTPUT / "figures"

VIEW_FILES = {
    "pitch": METADATA / "pitch_v2_topology_segments.csv",
    "rhythm": METADATA / "rhythm_topology_segments.csv",
    "modulation": METADATA / "modulation_tertile_topology_segments.csv",
    "structure": METADATA / "structure_topology_segments.csv",
}
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
LOCAL_VIEWS = ("pitch", "rhythm", "modulation")
SCALES = (180.0, 300.0)
PERMUTATIONS = 999
BOOTSTRAPS = 1000
SEED = 20260716

PERMANOVA_PATH = METADATA / "multiview_fusion_permanova.csv"
INCREMENTAL_PATH = METADATA / "multiview_fusion_incremental.csv"
CLASSIFICATION_PATH = METADATA / "multiview_fusion_classification.csv"
CLASSIFICATION_DELTA_PATH = METADATA / "multiview_fusion_classification_deltas.csv"
CORRELATION_PATH = METADATA / "multiview_fusion_distance_correlations.csv"
WEIGHT_PATH = METADATA / "multiview_fusion_structure_weight_sensitivity.csv"
RESIDUAL_PATH = METADATA / "multiview_fusion_structure_residual.csv"
SCORES_PATH = METADATA / "multiview_fusion_validation_scores.csv"
SUMMARY_PATH = METADATA / "multiview_fusion_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_aligned() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
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
            raise RuntimeError(f"{view} sample identities do not align")
        numeric = indexed.loc[:, TOPOLOGY_METRICS].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise RuntimeError(f"{view} contains missing topology metrics")
        frames[view] = indexed
    assert canonical_index is not None
    identity = canonical_index.to_frame(index=False)
    return identity, frames


def _analysis_mask(identity: pd.DataFrame, split: str, scale: float) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _fit_blocks(
    identity: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    discovery = _analysis_mask(identity, "discovery", scale)
    validation = _analysis_mask(identity, "validation", scale)
    blocks: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for view, frame in frames.items():
        raw = frame.loc[:, TOPOLOGY_METRICS].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        blocks[view] = (
            transformer.transform(raw[discovery]),
            transformer.transform(raw[validation]),
            transformer.effective_rank,
        )
    return discovery, validation, blocks


def _feature_sets(
    blocks: dict[str, tuple[np.ndarray, np.ndarray, int]],
    position: int,
) -> dict[str, np.ndarray]:
    values = {view: block[position] for view, block in blocks.items()}
    local = equal_block_fusion([values[view] for view in LOCAL_VIEWS])
    return {
        "pitch": values["pitch"],
        "rhythm": values["rhythm"],
        "modulation": values["modulation"],
        "structure": values["structure"],
        "pitch_rhythm": equal_block_fusion([values["pitch"], values["rhythm"]]),
        "pitch_modulation": equal_block_fusion([values["pitch"], values["modulation"]]),
        "rhythm_modulation": equal_block_fusion([values["rhythm"], values["modulation"]]),
        "local": local,
        "hierarchical": hierarchical_fusion(local, values["structure"], structure_weight=0.5),
        "all_equal_views": equal_block_fusion(
            [values[view] for view in (*LOCAL_VIEWS, "structure")]
        ),
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


def _bootstrap_single(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    strata = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    rng = np.random.default_rng(seed)
    samples = {
        name: np.empty(BOOTSTRAPS, dtype=float)
        for name in ("balanced_accuracy", "macro_f1", "macro_auroc_ovr", "macro_auprc")
    }
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


def _fit_classifier(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
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
    classes = search.best_estimator_.classes_
    return search, predictions, probabilities, classes


def _metric_functions(classes: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": lambda y, prediction, probability: float(
            balanced_accuracy_score(y, prediction)
        ),
        "macro_f1": lambda y, prediction, probability: float(
            f1_score(y, prediction, average="macro")
        ),
        "macro_auroc_ovr": lambda y, prediction, probability: _macro_auroc(y, probability, classes),
        "macro_auprc": lambda y, prediction, probability: _macro_auprc(y, probability, classes),
    }


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
            linewidth=1.8,
        )
    )


def _save_figure(figure: Any, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _plot_pca(pca_payload: dict[str, Any]) -> None:
    colors = {"classical": "#4472C4", "focus": "#D95F02"}
    labels = {"classical": "Classical", "focus": "Open Focus"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, feature_set in zip(axes, ("local", "hierarchical"), strict=True):
        payload = pca_payload[feature_set]
        coordinates = payload["validation"]
        groups = payload["groups"]
        for group in ("classical", "focus"):
            points = coordinates[groups == group]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=24,
                alpha=0.55,
                color=colors[group],
                label=labels[group],
                edgecolors="none",
            )
            _ellipse(ax, points, colors[group])
            center = np.mean(points, axis=0)
            ax.scatter(
                center[0], center[1], s=85, marker="X", color=colors[group], edgecolor="white"
            )
        variance = payload["variance"]
        title = (
            "Local: pitch + rhythm + modulation"
            if feature_set == "local"
            else "Hierarchical: local + structure"
        )
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.axvline(0.0, color="#BBBBBB", linewidth=0.7)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("Validation 180 s: discovery-fitted multiview spaces", fontsize=13)
    _save_figure(figure, "multiview_pca_validation_180")


def _plot_permanova(permanova: pd.DataFrame) -> None:
    order = ["pitch", "rhythm", "modulation", "structure", "local", "hierarchical"]
    labels = ["Pitch", "Rhythm", "Modulation", "Structure", "Local fusion", "Local + structure"]
    figure, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    y = np.arange(len(order))
    width = 0.36
    for offset, scale, color in ((-width / 2, 180.0, "#4472C4"), (width / 2, 300.0, "#70AD47")):
        subset = permanova[permanova["scale_seconds"] == scale].set_index("feature_set")
        values = np.array([subset.loc[name, "pseudo_f"] for name in order], dtype=float)
        bars = ax.barh(
            y + offset, values, height=width, label=f"{int(scale)} s", color=color, alpha=0.88
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 0.10,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=8,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation pseudo-F")
    ax.set_title("Group separation by view and fusion stage")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    _save_figure(figure, "multiview_permanova_ablation")


def _plot_classification(classification: pd.DataFrame) -> None:
    order = ["pitch", "rhythm", "modulation", "structure", "local", "hierarchical"]
    labels = ["Pitch", "Rhythm", "Modulation", "Structure", "Local fusion", "Local + structure"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True, constrained_layout=True)
    for ax, scale in zip(axes, SCALES, strict=True):
        subset = classification[classification["scale_seconds"] == scale].set_index("feature_set")
        y = np.arange(len(order))
        balanced = np.array([subset.loc[name, "balanced_accuracy"] for name in order])
        auroc = np.array([subset.loc[name, "macro_auroc_ovr"] for name in order])
        ax.scatter(balanced, y - 0.11, marker="o", color="#4472C4", label="Balanced accuracy")
        ax.scatter(auroc, y + 0.11, marker="s", color="#D95F02", label="Macro AUROC")
        for index, value in enumerate(balanced):
            ax.text(value + 0.008, index - 0.11, f"{value:.3f}", va="center", fontsize=8)
        for index, value in enumerate(auroc):
            ax.text(value + 0.008, index + 0.11, f"{value:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color="#888888", linewidth=0.8, linestyle="--")
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
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    figure.suptitle("Discovery-trained logistic regression", fontsize=13)
    _save_figure(figure, "multiview_classification_ablation")


def _plot_correlations(correlations: pd.DataFrame) -> None:
    views = ["pitch", "rhythm", "modulation", "structure"]
    labels = ["Pitch", "Rhythm", "Modulation", "Structure"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), constrained_layout=True)
    image = None
    for ax, scale in zip(axes, SCALES, strict=True):
        subset = correlations[correlations["scale_seconds"] == scale]
        matrix = np.eye(len(views))
        for row in subset.itertuples(index=False):
            left = views.index(row.view_a)
            right = views.index(row.view_b)
            matrix[left, right] = matrix[right, left] = row.spearman_rho
        image = ax.imshow(matrix, vmin=-0.1, vmax=1.0, cmap="Blues")
        ax.set_xticks(range(len(views)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(views)), labels)
        ax.set_title(f"{int(scale)} s")
        for i in range(len(views)):
            for j in range(len(views)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label="Spearman correlation of pairwise distances")
    figure.suptitle("Validation distance-matrix agreement", fontsize=13)
    _save_figure(figure, "multiview_distance_correlations")


def _plot_weight_sensitivity(weight_frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = {180.0: "#4472C4", 300.0: "#70AD47"}
    for scale in SCALES:
        subset = weight_frame[weight_frame["scale_seconds"] == scale].sort_values(
            "structure_weight"
        )
        label = f"{int(scale)} s"
        axes[0].plot(
            subset["structure_weight"],
            subset["pseudo_f"],
            marker="o",
            color=colors[scale],
            label=label,
        )
        axes[1].plot(
            subset["structure_weight"],
            subset["balanced_accuracy"],
            marker="o",
            color=colors[scale],
            label=label,
        )
    axes[0].set_ylabel("Permutation pseudo-F")
    axes[1].set_ylabel("Balanced accuracy")
    for ax in axes:
        ax.set_xlabel("Structure block weight")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    figure.suptitle("Frozen structure-weight sensitivity (not model selection)", fontsize=13)
    _save_figure(figure, "multiview_structure_weight_sensitivity")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity, frames = _load_aligned()
    permanova_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    classification_delta_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    pca_payload: dict[str, Any] = {}

    comparison_pairs = (
        ("local_vs_pitch", "local", "pitch", "local fusion vs single view"),
        ("local_vs_rhythm", "local", "rhythm", "local fusion vs single view"),
        ("local_vs_modulation", "local", "modulation", "local fusion vs single view"),
        ("add_pitch", "local", "rhythm_modulation", "leave-one-view-out"),
        ("add_rhythm", "local", "pitch_modulation", "leave-one-view-out"),
        ("add_modulation", "local", "pitch_rhythm", "leave-one-view-out"),
        ("add_structure", "hierarchical", "local", "hierarchical increment"),
    )

    for scale_index, scale in enumerate(SCALES):
        discovery, validation, blocks = _fit_blocks(identity, frames, scale)
        train_sets = _feature_sets(blocks, 0)
        validation_sets = _feature_sets(blocks, 1)
        train_labels = identity.loc[discovery, "group"].astype(str).to_numpy()
        validation_labels = identity.loc[validation, "group"].astype(str).to_numpy()
        validation_identity = identity.loc[validation].reset_index(drop=True)

        for feature_index, (name, matrix) in enumerate(validation_sets.items()):
            test = permutation_pseudo_f(
                matrix,
                validation_labels,
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
                    "distance": "discovery-fitted rank-normalized block Mahalanobis",
                    "permutations": PERMUTATIONS,
                    "n_validation": int(validation.sum()),
                    "input_dimensions": int(matrix.shape[1]),
                    **test,
                }
            )

        for comparison_index, (name, candidate, baseline, family) in enumerate(comparison_pairs):
            test = paired_incremental_permutation(
                validation_sets[candidate],
                validation_sets[baseline],
                validation_labels,
                permutations=PERMUTATIONS,
                seed=SEED + 300 + scale_index * 100 + comparison_index,
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
                    "family": family,
                    "alternative": "candidate pseudo-F > baseline pseudo-F",
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        prediction_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for feature_index, name in enumerate(train_sets):
            search, predictions, probabilities, classes = _fit_classifier(
                train_sets[name],
                train_labels,
                validation_sets[name],
                seed=SEED + 500 + scale_index * 100 + feature_index,
            )
            scores = _classification_metrics(validation_labels, predictions, probabilities, classes)
            intervals = _bootstrap_single(
                validation_labels,
                predictions,
                probabilities,
                classes,
                seed=SEED + 700 + scale_index * 100 + feature_index,
            )
            classification_rows.append(
                {
                    "analysis_set": "exploratory_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "classifier": "L2 logistic regression on frozen block coordinates",
                    "selection_metric": "discovery CV macro-F1",
                    "best_c": float(search.best_params_["C"]),
                    "cv_macro_f1": float(search.best_score_),
                    "n_train": int(discovery.sum()),
                    "n_validation": int(validation.sum()),
                    **scores,
                    **intervals,
                }
            )
            prediction_payload[name] = (predictions, probabilities, classes)
            for row_index, row in validation_identity.iterrows():
                score_rows.append(
                    {
                        "segment_id": row["segment_id"],
                        "track_id": row["track_id"],
                        "group": row["group"],
                        "split": row["split"],
                        "scale_seconds": scale,
                        "feature_set": name,
                        "predicted_group": predictions[row_index],
                        "probability_focus": float(
                            probabilities[row_index, list(classes).index("focus")]
                        ),
                    }
                )

        for comparison_index, (name, candidate, baseline, family) in enumerate(comparison_pairs):
            candidate_prediction, candidate_probability, classes = prediction_payload[candidate]
            baseline_prediction, baseline_probability, baseline_classes = prediction_payload[
                baseline
            ]
            if not np.array_equal(classes, baseline_classes):
                raise RuntimeError("classifier class orders do not match")
            candidate_scores = _classification_metrics(
                validation_labels, candidate_prediction, candidate_probability, classes
            )
            baseline_scores = _classification_metrics(
                validation_labels, baseline_prediction, baseline_probability, classes
            )
            intervals = stratified_bootstrap_differences(
                validation_labels,
                candidate_prediction,
                candidate_probability,
                baseline_prediction,
                baseline_probability,
                metric_functions=_metric_functions(classes),
                resamples=BOOTSTRAPS,
                seed=SEED + 900 + scale_index * 100 + comparison_index,
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
                        "family": family,
                        "metric": metric,
                        "candidate_score": candidate_scores[metric],
                        "baseline_score": baseline_scores[metric],
                        "delta": candidate_scores[metric] - baseline_scores[metric],
                        "bootstrap_resamples": BOOTSTRAPS,
                        **interval,
                    }
                )

        view_distances = {
            view: pdist(blocks[view][1], metric="euclidean") for view in (*LOCAL_VIEWS, "structure")
        }
        views = list(view_distances)
        for left_index, left in enumerate(views):
            for right in views[left_index + 1 :]:
                result = spearmanr(view_distances[left], view_distances[right])
                correlation_rows.append(
                    {
                        "analysis_set": "exploratory_validation_180"
                        if scale == 180.0
                        else "sensitivity_validation_300",
                        "scale_seconds": scale,
                        "view_a": left,
                        "view_b": right,
                        "spearman_rho": float(result.statistic),
                        "p_value": float(result.pvalue),
                        "n_pairwise_distances": int(view_distances[left].size),
                    }
                )

        local_train = train_sets["local"]
        local_validation = validation_sets["local"]
        structure_train = blocks["structure"][0]
        structure_validation = blocks["structure"][1]
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 17), cv=5).fit(local_train, structure_train)
        residual = structure_validation - ridge.predict(local_validation)
        residual_test = permutation_pseudo_f(
            residual,
            validation_labels,
            permutations=PERMUTATIONS,
            seed=SEED + 1100 + scale_index,
        )
        residual_rows.append(
            {
                "analysis_set": "exploratory_validation_180"
                if scale == 180.0
                else "sensitivity_validation_300",
                "scale_seconds": scale,
                "method": "discovery-fitted ridge residual structure conditional on local block",
                "ridge_alpha": float(ridge.alpha_),
                "discovery_r2": float(ridge.score(local_train, structure_train)),
                "validation_r2": float(ridge.score(local_validation, structure_validation)),
                "permutations": PERMUTATIONS,
                **residual_test,
            }
        )

        for weight_index, structure_weight in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            train = hierarchical_fusion(
                local_train, structure_train, structure_weight=structure_weight
            )
            validation_matrix = hierarchical_fusion(
                local_validation,
                structure_validation,
                structure_weight=structure_weight,
            )
            test = permutation_pseudo_f(
                validation_matrix,
                validation_labels,
                permutations=PERMUTATIONS,
                seed=SEED + 1200 + scale_index * 10 + weight_index,
            )
            search, predictions, probabilities, classes = _fit_classifier(
                train,
                train_labels,
                validation_matrix,
                seed=SEED + 1300 + scale_index * 10 + weight_index,
            )
            scores = _classification_metrics(validation_labels, predictions, probabilities, classes)
            weight_rows.append(
                {
                    "analysis_set": "exploratory_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "structure_weight": structure_weight,
                    "role": "frozen_primary" if structure_weight == 0.5 else "weight_sensitivity",
                    "pseudo_f": test["pseudo_f"],
                    "p_value": test["p_value"],
                    "best_c": float(search.best_params_["C"]),
                    **scores,
                }
            )

        if scale == 180.0:
            for name in ("local", "hierarchical"):
                pca = PCA(n_components=2, random_state=SEED).fit(train_sets[name])
                pca_payload[name] = {
                    "validation": pca.transform(validation_sets[name]),
                    "groups": validation_labels,
                    "variance": pca.explained_variance_ratio_,
                }

    incremental = pd.DataFrame(incremental_rows)
    for _, indices in incremental.groupby("scale_seconds").groups.items():
        incremental.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            incremental.loc[indices, "p_value_one_sided"].to_numpy(float)
        )

    permanova = pd.DataFrame(permanova_rows)
    classification = pd.DataFrame(classification_rows)
    classification_deltas = pd.DataFrame(classification_delta_rows)
    correlations = pd.DataFrame(correlation_rows)
    weights = pd.DataFrame(weight_rows)
    residuals = pd.DataFrame(residual_rows)
    validation_scores = pd.DataFrame(score_rows)

    permanova.to_csv(PERMANOVA_PATH, index=False, encoding="utf-8", lineterminator="\n")
    incremental.to_csv(INCREMENTAL_PATH, index=False, encoding="utf-8", lineterminator="\n")
    classification.to_csv(CLASSIFICATION_PATH, index=False, encoding="utf-8", lineterminator="\n")
    classification_deltas.to_csv(
        CLASSIFICATION_DELTA_PATH, index=False, encoding="utf-8", lineterminator="\n"
    )
    correlations.to_csv(CORRELATION_PATH, index=False, encoding="utf-8", lineterminator="\n")
    weights.to_csv(WEIGHT_PATH, index=False, encoding="utf-8", lineterminator="\n")
    residuals.to_csv(RESIDUAL_PATH, index=False, encoding="utf-8", lineterminator="\n")
    validation_scores.to_csv(SCORES_PATH, index=False, encoding="utf-8", lineterminator="\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    _plot_pca(pca_payload)
    _plot_permanova(permanova)
    _plot_classification(classification)
    _plot_correlations(correlations)
    _plot_weight_sensitivity(weights)

    primary_permanova = permanova[permanova["scale_seconds"] == 180.0].set_index("feature_set")
    sensitivity_permanova = permanova[permanova["scale_seconds"] == 300.0].set_index("feature_set")
    primary_classification = classification[classification["scale_seconds"] == 180.0].set_index(
        "feature_set"
    )
    sensitivity_classification = classification[classification["scale_seconds"] == 300.0].set_index(
        "feature_set"
    )
    structure_increment = incremental[
        (incremental["scale_seconds"] == 180.0) & (incremental["comparison"] == "add_structure")
    ].iloc[0]
    structure_increment_sensitivity = incremental[
        (incremental["scale_seconds"] == 300.0) & (incremental["comparison"] == "add_structure")
    ].iloc[0]
    local_increment = incremental[
        (incremental["scale_seconds"] == 180.0)
        & (incremental["comparison"] == "local_vs_pitch")
    ].iloc[0]
    residual_primary = residuals[residuals["scale_seconds"] == 180.0].iloc[0]
    residual_sensitivity = residuals[residuals["scale_seconds"] == 300.0].iloc[0]

    artifacts = [
        PERMANOVA_PATH,
        INCREMENTAL_PATH,
        CLASSIFICATION_PATH,
        CLASSIFICATION_DELTA_PATH,
        CORRELATION_PATH,
        WEIGHT_PATH,
        RESIDUAL_PATH,
        SCORES_PATH,
        *sorted(FIGURES.glob("*")),
    ]
    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "validation-stage two-stage multiview Path Homology integration",
        "evidence_role": {
            "validation_180": (
                "exploratory integration because fusion was specified after viewing "
                "single-view validation results"
            ),
            "validation_300": "duration sensitivity",
            "holdout": (
                "withheld from all summaries and tests until the hashed analysis gate "
                "is written"
            ),
        },
        "design": {
            "local_views": list(LOCAL_VIEWS),
            "macro_view": "structure",
            "metrics_per_view": len(TOPOLOGY_METRICS),
            "block_transform": "discovery-fitted rank-normalized Mahalanobis",
            "local_weights": {view: 1 / 3 for view in LOCAL_VIEWS},
            "hierarchical_weights": {"local": 0.5, "structure": 0.5},
            "permutations": PERMUTATIONS,
            "bootstrap_resamples": BOOTSTRAPS,
            "seed": SEED,
        },
        "sample_counts": {
            "total_rows_per_view": int(len(identity)),
            "unique_tracks": int(identity["track_id"].nunique()),
            "discovery_per_scale": int(_analysis_mask(identity, "discovery", 180.0).sum()),
            "validation_per_scale": int(_analysis_mask(identity, "validation", 180.0).sum()),
            "withheld_holdout_per_scale": int(_analysis_mask(identity, "holdout", 180.0).sum()),
        },
        "primary_180": {
            "local_permanova": primary_permanova.loc["local", ["pseudo_f", "p_value"]].to_dict(),
            "pitch_permanova": primary_permanova.loc["pitch", ["pseudo_f", "p_value"]].to_dict(),
            "hierarchical_permanova": primary_permanova.loc[
                "hierarchical", ["pseudo_f", "p_value"]
            ].to_dict(),
            "local_classification": primary_classification.loc[
                "local", ["balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
            ].to_dict(),
            "pitch_classification": primary_classification.loc[
                "pitch", ["balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
            ].to_dict(),
            "hierarchical_classification": primary_classification.loc[
                "hierarchical", ["balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
            ].to_dict(),
            "structure_increment": structure_increment[
                ["delta_pseudo_f", "p_value_one_sided", "p_fdr_bh"]
            ].to_dict(),
            "conditional_structure_residual": residual_primary[
                ["pseudo_f", "p_value", "validation_r2"]
            ].to_dict(),
        },
        "sensitivity_300": {
            "local_permanova": sensitivity_permanova.loc[
                "local", ["pseudo_f", "p_value"]
            ].to_dict(),
            "hierarchical_permanova": sensitivity_permanova.loc[
                "hierarchical", ["pseudo_f", "p_value"]
            ].to_dict(),
            "local_classification": sensitivity_classification.loc[
                "local", ["balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
            ].to_dict(),
            "hierarchical_classification": sensitivity_classification.loc[
                "hierarchical", ["balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
            ].to_dict(),
            "conditional_structure_residual": residual_sensitivity[
                ["pseudo_f", "p_value", "validation_r2"]
            ].to_dict(),
        },
        "decision": {
            "local_fusion_supported_as_group_separating_representation": bool(
                primary_permanova.loc["local", "p_value"] <= 0.05
            ),
            "local_fusion_has_validation_increment_over_pitch": bool(
                local_increment["delta_pseudo_f"] > 0
                and local_increment["p_fdr_bh"] <= 0.05
            ),
            "structure_has_validation_increment_over_local": bool(
                structure_increment["delta_pseudo_f"] > 0
                and structure_increment["p_fdr_bh"] <= 0.05
            ),
            "structure_increment_direction_stable_300": bool(
                np.sign(structure_increment["delta_pseudo_f"])
                == np.sign(structure_increment_sensitivity["delta_pseudo_f"])
            ),
            "frozen_holdout_primary_feature_set": "local",
            "frozen_holdout_secondary_feature_set": "hierarchical",
            "frozen_local_weights": {view: 1 / 3 for view in LOCAL_VIEWS},
            "frozen_hierarchical_weights": {"local": 0.5, "structure": 0.5},
            "weights_changed_after_validation": False,
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in VIEW_FILES.values()
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
