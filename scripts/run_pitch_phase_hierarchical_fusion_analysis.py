from __future__ import annotations

# ruff: noqa: I001

import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from topology.multiview_fusion import (  # noqa: E402
    DiscoveryMahalanobisBlock,
    equal_block_fusion,
    paired_incremental_permutation,
    permutation_pseudo_f,
    stratified_bootstrap_differences,
)
from topology.statistics import (  # noqa: E402
    TOPOLOGY_METRICS,
    _pseudo_f_statistic,
    benjamini_hochberg,
)


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
OUTPUT = ROOT / "runs" / "pitch_phase_hierarchical_fusion"
FIGURES = OUTPUT / "figures"

PITCH_FILE = METADATA / "pitch_v2_topology_segments.csv"
PHASE_FILE = METADATA / "phase_lifted_path_homology_features.csv"
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
PHASE_VIEWS = ("path_acoustic_phase", "path_chroma_phase")
SCALES = (180.0, 300.0)
PERMUTATIONS = 999
BOOTSTRAPS = 1000
SEED = 20260716
FDR_Q = 0.05

FEATURE_ORDER = ("Pitch", "Phase", "PitchPhase")
DISPLAY = {
    "Pitch": "Pitch",
    "Phase": "Acoustic + Chroma phase",
    "PitchPhase": "Pitch + Phase",
}
COMPARISONS = (
    ("PitchPhase_minus_Pitch", "PitchPhase", "Pitch"),
    ("PitchPhase_minus_Phase", "PitchPhase", "Phase"),
)
CONDITIONAL_SPECS = (
    ("Phase_given_Pitch", "Pitch", "Phase"),
    ("Pitch_given_Phase", "Phase", "Pitch"),
)

PERMANOVA_PATH = METADATA / "pitch_phase_hierarchical_permanova.csv"
INCREMENTAL_PATH = METADATA / "pitch_phase_hierarchical_incremental.csv"
RESIDUAL_PATH = METADATA / "pitch_phase_hierarchical_residuals.csv"
CLASSIFICATION_PATH = METADATA / "pitch_phase_hierarchical_classification.csv"
CLASSIFICATION_DELTA_PATH = METADATA / "pitch_phase_hierarchical_classification_deltas.csv"
CORRELATION_PATH = METADATA / "pitch_phase_hierarchical_correlations.csv"
HOLDOUT_PATH = METADATA / "pitch_phase_hierarchical_holdout_descriptive.csv"
SUMMARY_PATH = METADATA / "pitch_phase_hierarchical_summary.json"

PRIMARY_COLOR = "#28536B"
SENSITIVITY_COLOR = "#D97706"
FOCUS_COLOR = "#E07A5F"
CLASSICAL_COLOR = "#4472C4"
NEUTRAL_COLOR = "#59636E"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask(identity: pd.DataFrame, split: str, scale: float) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pitch = pd.read_csv(PITCH_FILE)
    required_pitch = set(IDENTITY) | set(TOPOLOGY_METRICS) | {"status"}
    missing_pitch = required_pitch - set(pitch.columns)
    if missing_pitch:
        raise RuntimeError(f"pitch file is missing columns: {sorted(missing_pitch)}")
    if pitch.duplicated(IDENTITY).any():
        raise RuntimeError("pitch file contains duplicate identity rows")
    if (pitch["status"] == "failed").any():
        raise RuntimeError("pitch file contains failed rows")
    pitch_indexed = pitch.set_index(IDENTITY).sort_index()
    pitch_numeric = pitch_indexed.loc[:, TOPOLOGY_METRICS].apply(
        pd.to_numeric, errors="coerce"
    )
    if pitch_numeric.isna().any().any():
        raise RuntimeError("pitch file contains missing topology metrics")

    phase = pd.read_csv(PHASE_FILE)
    required_phase = set(IDENTITY) | {"representation", "loop_score"}
    missing_phase = required_phase - set(phase.columns)
    if missing_phase:
        raise RuntimeError(f"phase file is missing columns: {sorted(missing_phase)}")
    if phase.duplicated([*IDENTITY, "representation"]).any():
        raise RuntimeError("phase file contains duplicate identity/representation rows")
    phase_pivot = phase.pivot(
        index=IDENTITY, columns="representation", values="loop_score"
    ).reindex(pitch_indexed.index)
    if phase_pivot.loc[:, list(PHASE_VIEWS)].isna().sum().max() > 2:
        raise RuntimeError("unexpected phase missingness")

    identity = pitch_indexed.index.to_frame(index=False)
    return identity, pitch_numeric, phase_pivot


def _fit_sets(
    identity: pd.DataFrame,
    pitch: pd.DataFrame,
    phase: pd.DataFrame,
    scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    discovery = _mask(identity, "discovery", scale)
    raw_pitch = pitch.to_numpy(float)
    pitch_transform = DiscoveryMahalanobisBlock().fit(raw_pitch[discovery])
    pitch_block = pitch_transform.transform(raw_pitch)

    phase_blocks = []
    ranks: dict[str, int] = {"Pitch": int(pitch_transform.effective_rank)}
    missing: dict[str, int] = {"Pitch": 0}
    for representation in PHASE_VIEWS:
        raw = phase.loc[:, [representation]].to_numpy(float)
        missing[representation] = int(np.isnan(raw[discovery]).sum())
        transform = DiscoveryMahalanobisBlock().fit(raw[discovery])
        phase_blocks.append(transform.transform(raw))
        ranks[representation] = int(transform.effective_rank)

    phase_block = equal_block_fusion(phase_blocks)
    return (
        {
            "Pitch": pitch_block,
            "Phase": phase_block,
            "PitchPhase": equal_block_fusion([pitch_block, phase_block]),
        },
        ranks,
        missing,
    )


def _macro_auroc(
    labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    values = [
        roc_auc_score((labels == label).astype(int), probabilities[:, index])
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
        "auroc": _macro_auroc(labels, probabilities, classes),
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


def _bootstrap_single(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> dict[str, float]:
    strata = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    rng = np.random.default_rng(seed)
    samples = {
        name: np.empty(BOOTSTRAPS, dtype=float)
        for name in ("balanced_accuracy", "macro_f1", "auroc")
    }
    for index in range(BOOTSTRAPS):
        selected = np.concatenate(
            [
                rng.choice(indices, size=indices.size, replace=True)
                for indices in strata.values()
            ]
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


def _metric_functions(classes: np.ndarray) -> dict[str, Callable[..., float]]:
    return {
        "balanced_accuracy": lambda y, p, q: float(
            balanced_accuracy_score(y, p)
        ),
        "macro_f1": lambda y, p, q: float(f1_score(y, p, average="macro")),
        "auroc": lambda y, p, q: _macro_auroc(y, q, classes),
    }


def _centroid_distance(matrix: np.ndarray, labels: np.ndarray) -> float:
    groups = np.unique(labels)
    if groups.size != 2:
        raise RuntimeError("centroid distance requires exactly two groups")
    return float(
        np.linalg.norm(
            matrix[labels == groups[0]].mean(axis=0)
            - matrix[labels == groups[1]].mean(axis=0)
        )
    )


def _save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    figure.savefig(
        FIGURES / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(figure)


def _plot_permanova(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
    x = np.arange(len(FEATURE_ORDER))
    width = 0.36
    for offset, scale, color in (
        (-width / 2, 180.0, PRIMARY_COLOR),
        (width / 2, 300.0, SENSITIVITY_COLOR),
    ):
        rows = frame[frame["scale_seconds"] == scale].set_index("feature_set").loc[
            list(FEATURE_ORDER)
        ]
        values = rows["pseudo_f"].to_numpy(float)
        bars = axis.bar(
            x + offset,
            values,
            width,
            color=color,
            alpha=0.88,
            label=f"{int(scale)} s",
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.18,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x, [DISPLAY[name] for name in FEATURE_ORDER], rotation=15, ha="right")
    axis.set_ylabel("PERMANOVA pseudo-F")
    axis.set_title("Pitch and phase Path Homology fusion and ablation")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, "pitch_phase_permanova_ablation")


def _plot_incremental(frame: pd.DataFrame) -> None:
    order = [name for name, *_ in COMPARISONS]
    labels = ["Add phase to Pitch", "Add Pitch to phase"]
    figure, axis = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    for offset, scale, color in (
        (-width / 2, 180.0, PRIMARY_COLOR),
        (width / 2, 300.0, SENSITIVITY_COLOR),
    ):
        rows = frame[frame["scale_seconds"] == scale].set_index("comparison").loc[order]
        values = rows["delta_pseudo_f"].to_numpy(float)
        passed = (values > 0) & (rows["p_fdr_bh"].to_numpy(float) <= FDR_Q)
        bars = axis.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor=color,
            alpha=0.9,
            label=f"{int(scale)} s",
        )
        for bar, value, q_value, is_passed in zip(
            bars, values, rows["p_fdr_bh"], passed, strict=True
        ):
            if not is_passed:
                bar.set_alpha(0.28)
            vertical = 0.12 if value >= 0 else -0.12
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + vertical,
                f"q={float(q_value):.3g}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    axis.axhline(0, color=NEUTRAL_COLOR, lw=1)
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"Paired increment $\Delta$ pseudo-F")
    axis.set_title("Does each scale add group-separation geometry?")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(
        handles=[
            Patch(facecolor=PRIMARY_COLOR, label="180 s"),
            Patch(facecolor=SENSITIVITY_COLOR, label="300 s"),
            Patch(
                facecolor="#7B8794",
                alpha=0.28,
                label=r"faded: not positive at BH $q\leq0.05$",
            ),
        ],
        frameon=False,
    )
    _save_figure(figure, "pitch_phase_incremental_tests")


def _plot_residuals(frame: pd.DataFrame) -> None:
    order = [name for name, *_ in CONDITIONAL_SPECS]
    labels = ["Phase | Pitch", "Pitch | Phase"]
    figure, axis = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    for offset, scale, color in (
        (-width / 2, 180.0, PRIMARY_COLOR),
        (width / 2, 300.0, SENSITIVITY_COLOR),
    ):
        rows = frame[frame["scale_seconds"] == scale].set_index("conditional_test").loc[
            order
        ]
        values = rows["pseudo_f"].to_numpy(float)
        bars = axis.bar(
            x + offset,
            values,
            width,
            color=color,
            alpha=0.88,
            label=f"{int(scale)} s",
        )
        for bar, q_value in zip(bars, rows["p_fdr_bh"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"q={float(q_value):.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Residual PERMANOVA pseudo-F")
    axis.set_title("Conditional non-redundancy of Pitch and phase")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, "pitch_phase_conditional_residuals")


def _plot_classification(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 5.0), sharey=True, constrained_layout=True
    )
    x = np.arange(len(FEATURE_ORDER))
    width = 0.36
    for axis, metric, title in zip(
        axes,
        ("balanced_accuracy", "auroc"),
        ("Balanced accuracy", "AUROC"),
        strict=True,
    ):
        for offset, scale, color in (
            (-width / 2, 180.0, PRIMARY_COLOR),
            (width / 2, 300.0, SENSITIVITY_COLOR),
        ):
            rows = frame[frame["scale_seconds"] == scale].set_index("feature_set").loc[
                list(FEATURE_ORDER)
            ]
            values = rows[metric].to_numpy(float)
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=color,
                alpha=0.88,
                label=f"{int(scale)} s",
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.axhline(0.5, color="#7B8794", lw=1, ls="--")
        axis.set_xticks(
            x,
            [DISPLAY[name] for name in FEATURE_ORDER],
            rotation=18,
            ha="right",
        )
        axis.set_title(title)
        axis.set_ylim(0.45, 1.04)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Validation score")
    axes[0].legend(frameon=False)
    figure.suptitle("Auxiliary discovery-trained classification ablation", fontsize=13)
    _save_figure(figure, "pitch_phase_classification_ablation")


def _ellipse(axis: plt.Axes, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * 1.96 * np.sqrt(np.maximum(values, 0))
    axis.add_patch(
        Ellipse(
            np.mean(points, axis=0),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            lw=1.5,
        )
    )


def _plot_pca(
    payload: dict[str, tuple[np.ndarray, np.ndarray]], labels: np.ndarray
) -> dict[str, dict[str, float]]:
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 5.2))
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.21, top=0.87, wspace=0.28)
    colors = {"classical": CLASSICAL_COLOR, "focus": FOCUS_COLOR}
    display = {"classical": "Classical", "focus": "Open Focus"}
    panels = (
        ("Pitch", "L"),
        ("Phase", "P"),
        ("PitchPhase", "L + P"),
    )
    summary: dict[str, dict[str, float]] = {}
    for axis, (feature_set, title) in zip(axes, panels, strict=True):
        discovery, validation = payload[feature_set]
        pca = PCA(n_components=2, random_state=SEED).fit(discovery)
        coordinates = pca.transform(validation)
        for group in ("classical", "focus"):
            points = coordinates[labels == group]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=28,
                alpha=0.72,
                color=colors[group],
                label=display[group],
            )
            _ellipse(axis, points, colors[group])
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axis.set_title(title)
        axis.grid(alpha=0.18)
        summary[title] = {
            "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "pc2_variance_ratio": float(pca.explained_variance_ratio_[1]),
        }
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle(
        "Validation/180 s discovery-fitted PCA projections", fontsize=13, y=0.98
    )
    _save_figure(figure, "pitch_phase_pca_validation_180")
    return summary


def _plot_holdout(frame: pd.DataFrame) -> None:
    rows = frame[frame["scale_seconds"] == 180.0].set_index("feature_set").loc[
        list(FEATURE_ORDER)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    x = np.arange(len(FEATURE_ORDER))
    values = rows["pseudo_f_descriptive"].to_numpy(float)
    bars = axes[0].bar(x, values, color=PRIMARY_COLOR, alpha=0.88, width=0.58)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )
    axes[0].set_xticks(
        x, [DISPLAY[name] for name in FEATURE_ORDER], rotation=18, ha="right"
    )
    axes[0].set_ylabel("Descriptive pseudo-F")
    axes[0].set_title("Opened holdout: distance geometry")
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.36
    for offset, metric, color, label in (
        (-width / 2, "balanced_accuracy", PRIMARY_COLOR, "Balanced accuracy"),
        (width / 2, "auroc", SENSITIVITY_COLOR, "AUROC"),
    ):
        scores = rows[metric].to_numpy(float)
        bars = axes[1].bar(
            x + offset,
            scores,
            width,
            color=color,
            alpha=0.88,
            label=label,
        )
        for bar, value in zip(bars, scores, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
            )
    axes[1].axhline(0.5, color="#7B8794", lw=1, ls="--")
    axes[1].set_xticks(
        x, [DISPLAY[name] for name in FEATURE_ORDER], rotation=18, ha="right"
    )
    axes[1].set_ylim(0.45, 1.04)
    axes[1].set_ylabel("Descriptive score")
    axes[1].set_title("Opened holdout: classification")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle("Opened holdout / 180 s (descriptive only; no new inference)")
    _save_figure(figure, "pitch_phase_holdout_descriptive")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    identity, pitch, phase = _load_inputs()
    labels = identity["group"].astype(str).to_numpy()
    permanova_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    classification_delta_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    ranks_by_scale: dict[str, Any] = {}
    missing_by_scale: dict[str, Any] = {}
    pca_payload: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    for scale_index, scale in enumerate(SCALES):
        sets, ranks, missing = _fit_sets(identity, pitch, phase, scale)
        ranks_by_scale[str(int(scale))] = ranks
        missing_by_scale[str(int(scale))] = missing
        discovery = _mask(identity, "discovery", scale)
        validation = _mask(identity, "validation", scale)
        holdout = _mask(identity, "holdout", scale)

        for feature_index, name in enumerate(FEATURE_ORDER):
            matrix = sets[name][validation]
            test = permutation_pseudo_f(
                matrix,
                labels[validation],
                permutations=PERMUTATIONS,
                seed=SEED + scale_index * 100 + feature_index,
            )
            permanova_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "same_track_sensitivity_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "input_dimensions": matrix.shape[1],
                    "n_validation": matrix.shape[0],
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        for comparison_index, (name, candidate, baseline) in enumerate(COMPARISONS):
            test = paired_incremental_permutation(
                sets[candidate][validation],
                sets[baseline][validation],
                labels[validation],
                permutations=PERMUTATIONS,
                seed=SEED + 200 + scale_index * 10 + comparison_index,
            )
            incremental_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "same_track_sensitivity_300",
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
        for feature_index, name in enumerate(FEATURE_ORDER):
            search, predicted, probability, classes = _fit_classifier(
                sets[name][discovery], labels[discovery], sets[name][validation]
            )
            scores = _classification_metrics(
                labels[validation], predicted, probability, classes
            )
            intervals = _bootstrap_single(
                labels[validation],
                predicted,
                probability,
                classes,
                SEED + 400 + scale_index * 100 + feature_index,
            )
            classification_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "same_track_sensitivity_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "best_c": float(search.best_params_["C"]),
                    "discovery_cv_macro_f1": float(search.best_score_),
                    "n_train": int(discovery.sum()),
                    "n_validation": int(validation.sum()),
                    **scores,
                    **intervals,
                }
            )
            predictions[name] = (predicted, probability, classes)
            holdout_predictions[name] = (
                search.predict(sets[name][holdout]),
                search.predict_proba(sets[name][holdout]),
                classes,
            )

        for comparison_index, (name, candidate, baseline) in enumerate(COMPARISONS):
            cand_pred, cand_prob, classes = predictions[candidate]
            base_pred, base_prob, base_classes = predictions[baseline]
            if not np.array_equal(classes, base_classes):
                raise RuntimeError("classifier class order mismatch")
            candidate_scores = _classification_metrics(
                labels[validation], cand_pred, cand_prob, classes
            )
            baseline_scores = _classification_metrics(
                labels[validation], base_pred, base_prob, classes
            )
            intervals = stratified_bootstrap_differences(
                labels[validation],
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
                        "analysis_set": "primary_validation_180"
                        if scale == 180.0
                        else "same_track_sensitivity_300",
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

        distances = {
            name: pdist(sets[name][validation], metric="euclidean")
            for name in ("Pitch", "Phase")
        }
        correlation = spearmanr(distances["Pitch"], distances["Phase"])
        correlation_rows.append(
            {
                "analysis_set": "primary_validation_180"
                if scale == 180.0
                else "same_track_sensitivity_300",
                "scale_seconds": scale,
                "block_a": "Pitch",
                "block_b": "Phase",
                "spearman_rho": float(correlation.statistic),
                "p_value_descriptive": float(correlation.pvalue),
                "n_pairwise_distances": distances["Pitch"].size,
            }
        )

        for residual_index, (name, predictor, outcome) in enumerate(CONDITIONAL_SPECS):
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 17), cv=5).fit(
                sets[predictor][discovery], sets[outcome][discovery]
            )
            residual = sets[outcome][validation] - ridge.predict(
                sets[predictor][validation]
            )
            test = permutation_pseudo_f(
                residual,
                labels[validation],
                permutations=PERMUTATIONS,
                seed=SEED + 800 + scale_index * 10 + residual_index,
            )
            residual_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "same_track_sensitivity_300",
                    "scale_seconds": scale,
                    "conditional_test": name,
                    "predictor": predictor,
                    "outcome": outcome,
                    "ridge_alpha": float(ridge.alpha_),
                    "discovery_r2": float(
                        ridge.score(sets[predictor][discovery], sets[outcome][discovery])
                    ),
                    "validation_r2": float(
                        ridge.score(sets[predictor][validation], sets[outcome][validation])
                    ),
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        for name in FEATURE_ORDER:
            matrix = sets[name][holdout]
            predicted, probability, classes = holdout_predictions[name]
            scores = _classification_metrics(labels[holdout], predicted, probability, classes)
            holdout_rows.append(
                {
                    "analysis_set": "opened_holdout_descriptive_only",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "n_holdout": matrix.shape[0],
                    "pseudo_f_descriptive": float(
                        _pseudo_f_statistic(matrix, labels[holdout])
                    ),
                    "centroid_distance_descriptive": _centroid_distance(
                        matrix, labels[holdout]
                    ),
                    **scores,
                }
            )

        if scale == 180.0:
            pca_payload = {
                name: (sets[name][discovery], sets[name][validation])
                for name in FEATURE_ORDER
            }

    permanova = pd.DataFrame(permanova_rows)
    for indices in permanova.groupby("scale_seconds").groups.values():
        permanova.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            permanova.loc[indices, "p_value"].to_numpy(float)
        )
    incremental = pd.DataFrame(incremental_rows)
    for indices in incremental.groupby("scale_seconds").groups.values():
        incremental.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            incremental.loc[indices, "p_value_one_sided"].to_numpy(float)
        )
    residuals = pd.DataFrame(residual_rows)
    for indices in residuals.groupby("scale_seconds").groups.values():
        residuals.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            residuals.loc[indices, "p_value"].to_numpy(float)
        )
    classification = pd.DataFrame(classification_rows)
    classification_deltas = pd.DataFrame(classification_delta_rows)
    correlations = pd.DataFrame(correlation_rows)
    holdout_frame = pd.DataFrame(holdout_rows)

    outputs = {
        PERMANOVA_PATH: permanova,
        INCREMENTAL_PATH: incremental,
        RESIDUAL_PATH: residuals,
        CLASSIFICATION_PATH: classification,
        CLASSIFICATION_DELTA_PATH: classification_deltas,
        CORRELATION_PATH: correlations,
        HOLDOUT_PATH: holdout_frame,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    _plot_permanova(permanova)
    _plot_incremental(incremental)
    _plot_residuals(residuals)
    _plot_classification(classification)
    _plot_holdout(holdout_frame)
    if pca_payload is None:
        raise RuntimeError("missing validation/180 PCA payload")
    pca_summary = _plot_pca(pca_payload, labels[_mask(identity, "validation", 180.0)])

    primary_permanova = permanova[permanova["scale_seconds"] == 180.0].set_index(
        "feature_set"
    )
    primary_incremental = incremental[
        incremental["scale_seconds"] == 180.0
    ].set_index("comparison")
    primary_residuals = residuals[residuals["scale_seconds"] == 180.0].set_index(
        "conditional_test"
    )
    primary_classification = classification[
        classification["scale_seconds"] == 180.0
    ].set_index("feature_set")
    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "Pitch local block -> Acoustic/Chroma phase block hierarchical fusion",
        "evidence_role": {
            "validation_180": "primary analysis after user-requested refreeze",
            "validation_300": "same-track duration sensitivity, not independent replication",
            "holdout": "already opened; descriptive only, no p-values or tuning",
        },
        "design": {
            "local_block": "Pitch Path Homology only",
            "phase_block": "equal-block Acoustic phase and Chroma phase loop_score",
            "fusion_weights": {"Pitch": 0.5, "Phase": 0.5},
            "block_transform": (
                "scale-specific discovery-fitted rank-normalized "
                "Mahalanobis coordinates"
            ),
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
                name: primary_permanova.loc[
                    name, ["pseudo_f", "p_value", "p_fdr_bh"]
                ].to_dict()
                for name in FEATURE_ORDER
            },
            "increments": {
                name: primary_incremental.loc[
                    name, ["delta_pseudo_f", "p_value_one_sided", "p_fdr_bh"]
                ].to_dict()
                for name, *_ in COMPARISONS
            },
            "conditional_residuals": {
                name: primary_residuals.loc[
                    name, ["pseudo_f", "p_value", "p_fdr_bh", "validation_r2"]
                ].to_dict()
                for name, *_ in CONDITIONAL_SPECS
            },
            "classification": {
                name: primary_classification.loc[
                    name, ["balanced_accuracy", "macro_f1", "auroc"]
                ].to_dict()
                for name in FEATURE_ORDER
            },
            "pca": pca_summary,
        },
        "input_sha256": {
            str(PITCH_FILE.relative_to(ROOT)): _sha256(PITCH_FILE),
            str(PHASE_FILE.relative_to(ROOT)): _sha256(PHASE_FILE),
        },
        "artifacts": [
            str(path.relative_to(ROOT))
            for path in [*outputs, *sorted(FIGURES.glob("*"))]
        ],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
