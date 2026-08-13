# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "runs" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from topology.multiview_fusion import (
    DiscoveryMahalanobisBlock,
    equal_block_fusion,
    paired_incremental_permutation,
    permutation_pseudo_f,
    stratified_bootstrap_differences,
)
from topology.statistics import TOPOLOGY_METRICS, benjamini_hochberg

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
OUTPUT = ROOT / "runs" / "local_smp_k10_fusion"
FIGURES = OUTPUT / "figures"
REPORT = ROOT / "docs" / "path-homology-local-smp-k10-fusion-analysis.md"

VIEW_FILES = {
    "pitch": METADATA / "pitch_v2_topology_segments.csv",
    "rhythm": METADATA / "rhythm_topology_segments.csv",
    "modulation_smp_k10": METADATA / "modulation_smp_prototype_topology_segments.csv",
}
VIEWS = tuple(VIEW_FILES)
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
SCALES = (180.0, 300.0)
PERMUTATIONS = 999
BOOTSTRAPS = 1000
SEED = 20_260_716
FDR_Q = 0.05

FEATURE_ORDER = (
    "pitch",
    "rhythm",
    "modulation_smp_k10",
    "pitch_rhythm",
    "pitch_modulation",
    "rhythm_modulation",
    "local_all",
)
DISPLAY = {
    "pitch": "Pitch",
    "rhythm": "Rhythm",
    "modulation_smp_k10": "Modulation SMP K=10",
    "pitch_rhythm": "Pitch + Rhythm",
    "pitch_modulation": "Pitch + Modulation",
    "rhythm_modulation": "Rhythm + Modulation",
    "local_all": "Pitch + Rhythm + Modulation",
}
COMPARISONS = (
    ("full_vs_pitch", "local_all", "pitch", "full versus single view"),
    ("full_vs_rhythm", "local_all", "rhythm", "full versus single view"),
    (
        "full_vs_modulation",
        "local_all",
        "modulation_smp_k10",
        "full versus single view",
    ),
    ("add_pitch", "local_all", "rhythm_modulation", "leave-one-view-out"),
    ("add_rhythm", "local_all", "pitch_modulation", "leave-one-view-out"),
    ("add_modulation", "local_all", "pitch_rhythm", "leave-one-view-out"),
)

PERMANOVA_PATH = METADATA / "local_smp_k10_fusion_permanova.csv"
INCREMENTAL_PATH = METADATA / "local_smp_k10_fusion_incremental.csv"
RESIDUAL_PATH = METADATA / "local_smp_k10_fusion_conditional_residual.csv"
CORRELATION_PATH = METADATA / "local_smp_k10_fusion_distance_correlations.csv"
CLASSIFICATION_PATH = METADATA / "local_smp_k10_fusion_classification.csv"
CLASSIFICATION_DELTA_PATH = METADATA / "local_smp_k10_fusion_classification_deltas.csv"
SCORES_PATH = METADATA / "local_smp_k10_fusion_validation_scores.csv"
SUMMARY_PATH = METADATA / "local_smp_k10_fusion_summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_aligned() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    canonical_index: pd.MultiIndex | None = None
    for view, path in VIEW_FILES.items():
        frame = pd.read_csv(path)
        if view == "modulation_smp_k10":
            frame = frame[frame["state_count"].astype(int) == 10].copy()
        required = set(IDENTITY) | set(TOPOLOGY_METRICS) | {"status"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{view} is missing columns: {sorted(missing)}")
        if frame.duplicated(IDENTITY).any():
            raise RuntimeError(f"{view} contains duplicate identity rows")
        if (frame["status"].astype(str) == "failed").any():
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
    return canonical_index.to_frame(index=False), frames


def _mask(identity: pd.DataFrame, split: str, scale: float) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _fit_blocks(
    identity: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, dict[float, np.ndarray], dict[str, dict[str, Any]]]:
    discovery = _mask(identity, "discovery", 180.0)
    validation_masks = {scale: _mask(identity, "validation", scale) for scale in SCALES}
    blocks: dict[str, dict[str, Any]] = {}
    for view, frame in frames.items():
        raw = frame.loc[:, TOPOLOGY_METRICS].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(raw[discovery])
        blocks[view] = {
            "discovery_180": transformer.transform(raw[discovery]),
            "validation": {
                scale: transformer.transform(raw[validation_masks[scale]])
                for scale in SCALES
            },
            "effective_rank": transformer.effective_rank,
        }
    return discovery, validation_masks, blocks


def _feature_sets(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pitch = values["pitch"]
    rhythm = values["rhythm"]
    modulation = values["modulation_smp_k10"]
    return {
        "pitch": pitch,
        "rhythm": rhythm,
        "modulation_smp_k10": modulation,
        "pitch_rhythm": equal_block_fusion([pitch, rhythm]),
        "pitch_modulation": equal_block_fusion([pitch, modulation]),
        "rhythm_modulation": equal_block_fusion([rhythm, modulation]),
        "local_all": equal_block_fusion([pitch, rhythm, modulation]),
    }


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    focus_index = list(classes).index("focus")
    focus_labels = (labels == "focus").astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "auroc": float(roc_auc_score(focus_labels, probabilities[:, focus_index])),
    }


def _metric_functions(classes: np.ndarray) -> dict[str, Callable[..., float]]:
    focus_index = list(classes).index("focus")
    return {
        "balanced_accuracy": lambda y, pred, prob: float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_f1": lambda y, pred, prob: float(f1_score(y, pred, average="macro")),
        "auroc": lambda y, pred, prob: float(
            roc_auc_score((y == "focus").astype(int), prob[:, focus_index])
        ),
    }


def _fit_classifier(
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
) -> tuple[GridSearchCV, np.ndarray, np.ndarray, np.ndarray]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="f1_macro",
        cv=folds,
        refit=True,
    ).fit(train, train_labels)
    predictions = search.predict(validation)
    probabilities = search.predict_proba(validation)
    return search, predictions, probabilities, search.classes_


def _bootstrap_single(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    functions = _metric_functions(classes)
    strata = {group: np.flatnonzero(labels == group) for group in np.unique(labels)}
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(BOOTSTRAPS, dtype=float) for name in functions}
    for index in range(BOOTSTRAPS):
        selected = np.concatenate(
            [rng.choice(rows, size=rows.size, replace=True) for rows in strata.values()]
        )
        for name, function in functions.items():
            samples[name][index] = function(
                labels[selected], predictions[selected], probabilities[selected]
            )
    output: dict[str, float] = {}
    for name, values in samples.items():
        output[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        output[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return output


def _save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_permanova(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 6.2), constrained_layout=True)
    x = np.arange(len(FEATURE_ORDER))
    width = 0.36
    for offset, scale, color in ((-width / 2, 180.0, "#28536B"), (width / 2, 300.0, "#D97706")):
        rows = frame[frame["scale_seconds"] == scale].set_index("feature_set").loc[
            list(FEATURE_ORDER)
        ]
        bars = axis.bar(
            x + offset,
            rows["pseudo_f"].to_numpy(float),
            width,
            color=color,
            alpha=0.88,
            label=f"{int(scale)} s",
        )
        for bar, value in zip(bars, rows["pseudo_f"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.12,
                f"{float(value):.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x, [DISPLAY[name] for name in FEATURE_ORDER], rotation=25, ha="right")
    axis.set_ylabel("PERMANOVA pseudo-F")
    axis.set_title("Local Path Homology fusion and ablation")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, "local_smp_k10_permanova_ablation")


def _plot_incremental(frame: pd.DataFrame) -> None:
    order = [name for name, *_ in COMPARISONS]
    labels = [
        "Full − Pitch",
        "Full − Rhythm",
        "Full − Modulation",
        "Add Pitch",
        "Add Rhythm",
        "Add Modulation",
    ]
    figure, axis = plt.subplots(figsize=(11.0, 6.1), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    for offset, scale, color in ((-width / 2, 180.0, "#28536B"), (width / 2, 300.0, "#D97706")):
        rows = frame[frame["scale_seconds"] == scale].set_index("comparison").loc[order]
        values = rows["delta_pseudo_f"].to_numpy(float)
        passed = (values > 0) & (rows["p_fdr_bh"].to_numpy(float) <= FDR_Q)
        bars = axis.bar(
            x + offset,
            values,
            width,
            color=color,
            alpha=0.9,
            edgecolor=color,
            label=f"{int(scale)} s",
        )
        for bar, value, q_value, is_passed in zip(
            bars, values, rows["p_fdr_bh"], passed, strict=True
        ):
            if not is_passed:
                bar.set_alpha(0.28)
            vertical = 0.14 if value >= 0 else -0.14
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + vertical,
                f"q={float(q_value):.3g}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.5,
            )
    axis.axhline(0, color="#59636E", lw=1)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel(r"Paired increment $\Delta$ pseudo-F")
    axis.set_title("Does each view add group-separation geometry?")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(
        handles=[
            Patch(facecolor="#28536B", label="180 s"),
            Patch(facecolor="#D97706", label="300 s"),
            Patch(
                facecolor="#7B8794",
                alpha=0.28,
                label=r"faded: not positive at BH $q\leq0.05$",
            ),
        ],
        frameon=False,
    )
    _save_figure(figure, "local_smp_k10_incremental_tests")


def _plot_classification(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True, constrained_layout=True)
    x = np.arange(len(FEATURE_ORDER))
    width = 0.36
    for axis, metric, title in zip(
        axes,
        ("balanced_accuracy", "auroc"),
        ("Balanced accuracy", "AUROC"),
        strict=True,
    ):
        for offset, scale, color in ((-width / 2, 180.0, "#28536B"), (width / 2, 300.0, "#D97706")):
            rows = frame[frame["scale_seconds"] == scale].set_index("feature_set").loc[
                list(FEATURE_ORDER)
            ]
            axis.bar(
                x + offset,
                rows[metric].to_numpy(float),
                width,
                color=color,
                alpha=0.88,
                label=f"{int(scale)} s",
            )
        axis.axhline(0.5, color="#7B8794", lw=1, ls="--")
        axis.set_xticks(x, [DISPLAY[name] for name in FEATURE_ORDER], rotation=28, ha="right")
        axis.set_title(title)
        axis.set_ylim(0.45, 1.02)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Validation score")
    axes[0].legend(frameon=False)
    figure.suptitle("Auxiliary discovery-trained classification ablation", fontsize=13)
    _save_figure(figure, "local_smp_k10_classification_ablation")


def _plot_correlations(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    labels = ["Pitch", "Rhythm", "Modulation"]
    for axis, scale in zip(axes, SCALES, strict=True):
        matrix = np.eye(3)
        rows = frame[frame["scale_seconds"] == scale]
        for row in rows.itertuples(index=False):
            left = VIEWS.index(row.view_a)
            right = VIEWS.index(row.view_b)
            matrix[left, right] = matrix[right, left] = float(row.spearman_rho)
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(3), labels, rotation=25, ha="right")
        axis.set_yticks(range(3), labels)
        axis.set_title(f"Validation {int(scale)} s")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, fraction=0.035, pad=0.03, label="Spearman rho of pairwise distances")
    figure.suptitle("Complementarity of view-specific distance geometries", fontsize=13)
    _save_figure(figure, "local_smp_k10_distance_correlations")


def _plot_residuals(frame: pd.DataFrame) -> None:
    labels = ["Pitch | Rhythm+Mod", "Rhythm | Pitch+Mod", "Mod | Pitch+Rhythm"]
    order = list(VIEWS)
    figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    x = np.arange(3)
    width = 0.36
    for offset, scale, color in ((-width / 2, 180.0, "#28536B"), (width / 2, 300.0, "#D97706")):
        rows = frame[frame["scale_seconds"] == scale].set_index("target_view").loc[order]
        bars = axis.bar(
            x + offset,
            rows["pseudo_f"].to_numpy(float),
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
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Residual PERMANOVA pseudo-F")
    axis.set_title("Conditional non-redundancy of each local view")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, "local_smp_k10_conditional_residuals")


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


def _plot_pca(train: np.ndarray, validation: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=SEED).fit(train)
    coordinates = pca.transform(validation)
    figure, axis = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
    colors = {"classical": "#4472C4", "focus": "#E07A5F"}
    for group in ("classical", "focus"):
        points = coordinates[labels == group]
        axis.scatter(points[:, 0], points[:, 1], s=34, alpha=0.72, color=colors[group], label=group.title())
        _ellipse(axis, points, colors[group])
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axis.set_title("Validation/180 s fused local topology")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    _save_figure(figure, "local_smp_k10_pca_validation_180")


def _format(value: float) -> str:
    return f"{float(value):.3g}"


def _table_permanova(frame: pd.DataFrame) -> str:
    rows = []
    for name in FEATURE_ORDER:
        primary = frame[(frame["scale_seconds"] == 180.0) & (frame["feature_set"] == name)].iloc[0]
        sensitivity = frame[(frame["scale_seconds"] == 300.0) & (frame["feature_set"] == name)].iloc[0]
        rows.append(
            f"| {DISPLAY[name]} | {int(primary.input_dimensions)} | {_format(primary.pseudo_f)} | "
            f"{_format(primary.p_value)} | {_format(primary.p_fdr_bh)} | "
            f"{_format(sensitivity.pseudo_f)} | {_format(sensitivity.p_fdr_bh)} |"
        )
    return "\n".join(rows)


def _table_incremental(frame: pd.DataFrame) -> str:
    labels = {
        "full_vs_pitch": "Full − Pitch",
        "full_vs_rhythm": "Full − Rhythm",
        "full_vs_modulation": "Full − Modulation",
        "add_pitch": "Add Pitch to Rhythm+Modulation",
        "add_rhythm": "Add Rhythm to Pitch+Modulation",
        "add_modulation": "Add Modulation to Pitch+Rhythm",
    }
    rows = []
    for name, *_ in COMPARISONS:
        primary = frame[(frame["scale_seconds"] == 180.0) & (frame["comparison"] == name)].iloc[0]
        sensitivity = frame[(frame["scale_seconds"] == 300.0) & (frame["comparison"] == name)].iloc[0]
        rows.append(
            f"| {labels[name]} | {_format(primary.delta_pseudo_f)} | {_format(primary.p_fdr_bh)} | "
            f"{_format(sensitivity.delta_pseudo_f)} | {_format(sensitivity.p_fdr_bh)} |"
        )
    return "\n".join(rows)


def _table_classification(frame: pd.DataFrame) -> str:
    rows = []
    for name in FEATURE_ORDER:
        primary = frame[(frame["scale_seconds"] == 180.0) & (frame["feature_set"] == name)].iloc[0]
        sensitivity = frame[(frame["scale_seconds"] == 300.0) & (frame["feature_set"] == name)].iloc[0]
        rows.append(
            f"| {DISPLAY[name]} | {_format(primary.balanced_accuracy)} | {_format(primary.macro_f1)} | "
            f"{_format(primary.auroc)} | {_format(sensitivity.balanced_accuracy)} | "
            f"{_format(sensitivity.auroc)} |"
        )
    return "\n".join(rows)


def _table_residuals(frame: pd.DataFrame) -> str:
    rows = []
    for view in VIEWS:
        primary = frame[
            (frame["scale_seconds"] == 180.0) & (frame["target_view"] == view)
        ].iloc[0]
        sensitivity = frame[
            (frame["scale_seconds"] == 300.0) & (frame["target_view"] == view)
        ].iloc[0]
        rows.append(
            f"| {DISPLAY[view]} | {_format(primary.pseudo_f)} | "
            f"{_format(primary.p_fdr_bh)} | {_format(primary.validation_r2)} | "
            f"{_format(sensitivity.pseudo_f)} | {_format(sensitivity.p_fdr_bh)} |"
        )
    return "\n".join(rows)


def _write_report(
    permanova: pd.DataFrame,
    incremental: pd.DataFrame,
    residuals: pd.DataFrame,
    correlations: pd.DataFrame,
    classification: pd.DataFrame,
    classification_deltas: pd.DataFrame,
    blocks: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
) -> None:
    primary = permanova[permanova["scale_seconds"] == 180.0].set_index("feature_set")
    sensitivity = permanova[permanova["scale_seconds"] == 300.0].set_index("feature_set")
    full_primary = primary.loc["local_all"]
    full_sensitivity = sensitivity.loc["local_all"]
    best_single_name = max(VIEWS, key=lambda name: float(primary.loc[name, "pseudo_f"]))
    best_single = primary.loc[best_single_name]
    add_primary = incremental[
        (incremental["scale_seconds"] == 180.0)
        & incremental["comparison"].isin(["add_pitch", "add_rhythm", "add_modulation"])
    ]
    supported_additions = add_primary[
        (add_primary["delta_pseudo_f"] > 0) & (add_primary["p_fdr_bh"] <= FDR_Q)
    ]["comparison"].tolist()
    residual_primary = residuals[residuals["scale_seconds"] == 180.0]
    residual_supported = residual_primary[
        residual_primary["p_fdr_bh"] <= FDR_Q
    ]["target_view"].tolist()
    full_classifier = classification[
        (classification["scale_seconds"] == 180.0)
        & (classification["feature_set"] == "local_all")
    ].iloc[0]
    best_single_classifier = classification[
        (classification["scale_seconds"] == 180.0)
        & (classification["feature_set"] == best_single_name)
    ].iloc[0]
    full_pitch_balanced_delta = classification_deltas[
        (classification_deltas["scale_seconds"] == 180.0)
        & (classification_deltas["comparison"] == "full_vs_pitch")
        & (classification_deltas["metric"] == "balanced_accuracy")
    ].iloc[0]
    add_modulation_balanced_delta = classification_deltas[
        (classification_deltas["scale_seconds"] == 180.0)
        & (classification_deltas["comparison"] == "add_modulation")
        & (classification_deltas["metric"] == "balanced_accuracy")
    ].iloc[0]
    supported_text = (
        "、".join(name.replace("add_", "") for name in supported_additions)
        if supported_additions
        else "无"
    )
    residual_text = (
        "、".join(DISPLAY[name] for name in residual_supported)
        if residual_supported
        else "无"
    )
    rank_text = "、".join(
        f"{DISPLAY[view]} {int(blocks[view]['effective_rank'])}"
        for view in VIEWS
    )
    report = rf"""# 音高—节奏—SMP 调制局部 Path Homology 融合与消融分析

生成日期：{date.today().isoformat()}。本研究将 [音高视角](path-homology-pitch-v2-analysis.md)、[节奏视角](path-homology-rhythm-analysis.md) 与更新后的 [SMP 调制 K=10 视角](path-homology-modulation-smp-k10-analysis.md) 视为三个短时间尺度状态转移块，进行等权融合、留一视角消融和条件非冗余检验。所有归一化、协方差估计和分类器选择只使用 discovery/180 s；主分析为 validation/180 s，validation/300 s 仅作同曲目时长敏感性。由于 SMP K=10 方法是在旧 holdout 打开后提出，本次整合属于**探索性验证**，不检验或重新解释旧 holdout。多重检验统一要求 BH-FDR $q\le0.05$。

## 1. 结论摘要

- 三视角等权融合在 validation/180 s 上形成显著组间距离几何：pseudo-$F={_format(full_primary.pseudo_f)}$，置换 $p={_format(full_primary.p_value)}$，BH $q={_format(full_primary.p_fdr_bh)}$；300 s 为 pseudo-$F={_format(full_sensitivity.pseudo_f)}$、$q={_format(full_sensitivity.p_fdr_bh)}$。
- 180 s 单视角中 pseudo-$F$ 最大的是 {DISPLAY[best_single_name]}（{_format(best_single.pseudo_f)}）。完整融合相对它的差值为 {_format(float(full_primary.pseudo_f)-float(best_single.pseudo_f))}；因此“融合空间可分”不能自动写成“融合优于最佳单视角”。
- 留一视角增量中，通过 $q\le0.05$ 且 $\Delta$ pseudo-$F>0$ 的加入项为：{supported_text}。条件残差仍可分的视角为：{residual_text}。前者检验几何增量，后者检验新块是否含有无法由另外两块预测的组别信息。
- 辅助分类中，完整融合的 balanced accuracy={_format(full_classifier.balanced_accuracy)}、Macro-F1={_format(full_classifier.macro_f1)}、AUROC={_format(full_classifier.auroc)}；{DISPLAY[best_single_name]} 分别为 {_format(best_single_classifier.balanced_accuracy)}、{_format(best_single_classifier.macro_f1)}、{_format(best_single_classifier.auroc)}。分类结果不替代主要的距离置换检验。
- 完整融合相对 Pitch 的 balanced accuracy 增量为 {_format(full_pitch_balanced_delta.delta)}，分层 bootstrap 95% CI [{_format(full_pitch_balanced_delta.ci_low)}, {_format(full_pitch_balanced_delta.ci_high)}]。因此当前数据呈现“分类改善、pseudo-$F$ 不改善”的指标分歧，必须并列报告，而不能只挑有利指标。
- 本研究支持的是 Focus 与 Classical 在短时状态转移组织上的观察性差异；不支持注意力改善、功能疗效、生成质量或因果机制。三个视角的普通图与 $H_0$ 描述子贡献较多，单视角报告均不支持稳定、普遍的 $H_1$ 差异。

## 2. 为什么先融合三个短时视角

三个视角都把局部时间窗口或节拍映射为离散状态，并由相邻状态构造有向转移图：音高使用按拍 Tonnetz 原型，节奏使用 1 s/0.5 s 八维节奏块，SMP 调制使用 4 s/2 s 调制谱形与共享 PCA-32、固定 $K=10$ 原型。它们共享“局部状态—相邻转移—有向过滤”的数学接口，但观察的物理内容不同，因此适合在进入相位或宏观结构层之前先构造局部块 $L$。

```mermaid
flowchart LR
    P["Pitch：按拍 Tonnetz，K=16"] --> WP["Discovery Mahalanobis block"]
    R["Rhythm：1 s / 0.5 s，K=10"] --> WR["Discovery Mahalanobis block"]
    M["SMP modulation：4 s / 2 s，PCA-32，K=10"] --> WM["Discovery Mahalanobis block"]
    WP --> L["Equal-block local fusion L"]
    WR --> L
    WM --> L
    L --> A["PERMANOVA + leave-one-view-out + conditional residual"]
```

## 3. 单视角 Path Homology 接口

对状态序列 $s_t$，相邻非自转移计数和条件边权为

$$
C_{{ij}}=\left|\{{t:s_t=i,s_{{t+1}}=j,i\ne j}}\right|,
\qquad
w_{{ij}}=\frac{{C_{{ij}}}}{{\sum_{{k\ne i}}C_{{ik}}}}.
$$

每个源节点最多保留 top-6 非自环边。主过滤为

$$
G_\tau=(V,\{{(i,j):w_{{ij}}\ge\tau}}),
\qquad
\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}.
$$

允许有向 $p$-路径张成 $A_p$，$\Omega_p=A_p\cap\partial^{{-1}}A_{{p-1}}$，路径同调为

$$
H_p^{{\mathrm{{path}}}}(G)=
\frac{{\ker(\partial_p|_{{\Omega_p}})}}{{\operatorname{{im}}(\partial_{{p+1}}|_{{\Omega_{{p+1}}}})}}.
$$

每个视角最终使用同一组 20 个预设图、$H_0$ 与 $H_1$ 描述子。SMP K=10 的单视角稳定发现为状态数、边数更高而边密度、互惠性更低；音高与节奏视角的稳定结果详见各自报告。

## 4. 块归一化与等权融合

直接拼接会让有效维数较高或协方差尺度较大的视角主导距离。对视角 $v$ 的 discovery/180 s 描述子矩阵，去除常量列并估计均值 $\mu_v$ 与协方差 $\Sigma_v$。设有效秩为 $r_v$，其伪逆特征分解诱导白化坐标

$$
z_v(x)=\frac{{(x-\mu_v)W_v}}{{\sqrt{{r_v}}}},
\qquad
W_vW_v^\mathsf T=\Sigma_v^+.
$$

本轮有效秩为：{rank_text}。除以 $\sqrt{{r_v}}$ 后，每个块的期望平方距离处于相近尺度。三视角等权融合定义为

$$
z_L(x)=\frac1{{\sqrt3}}\left[z_P(x)\;\Vert\;z_R(x)\;\Vert\;z_M(x)\right].
$$

两视角消融使用相同规则，例如 $z_{{PR}}=2^{{-1/2}}[z_P\Vert z_R]$。没有根据 validation 结果调节 $1/3$ 权重。

## 5. 统计检验

### 5.1 整体分离

对欧氏距离 $d_{{ij}}$，PERMANOVA pseudo-$F$ 写为

$$
SS_T=\frac1N\sum_{{i<j}}d_{{ij}}^2,
\qquad
SS_W=\sum_g\frac1{{n_g}}\sum_{{i<j\in g}}d_{{ij}}^2,
$$

$$
F^*=\frac{{(SS_T-SS_W)/(G-1)}}{{SS_W/(N-G)}}.
$$

每个表示使用 999 次标签置换；七个表示按时长分别作 BH 校正。

### 5.2 配对增量与消融

在同一次标签排列下比较候选与基线：

$$
\Delta F=F^*_{{\mathrm{{candidate}}}}-F^*_{{\mathrm{{baseline}}}}.
$$

主要消融为 $L-(R+M)$、$L-(P+M)$、$L-(P+R)$，分别检验加入音高、节奏和 SMP 调制是否带来正增量。另将完整融合与每个单视角比较。六个增量检验按时长分别作 BH 校正，要求 $\Delta F>0$ 且 $q\le0.05$。

### 5.3 条件残差

仅有正增量仍不能说明信息不可预测。对目标视角 $v$，在 discovery/180 s 上用另外两个块拟合多输出岭回归 $\widehat f_v$，在 validation 中计算

$$
R_v=Z_v-\widehat f_v(Z_{{-v}}),
$$

再对 $R_v$ 做 PERMANOVA。三个残差检验按时长作 BH 校正。这是“条件非冗余”诊断，不是因果分解。

## 6. 结果

### 6.1 整体融合与组合消融

| 表示 | 维数 | 180 s pseudo-F | 180 s p | 180 s FDR | 300 s pseudo-F | 300 s FDR |
|---|---:|---:|---:|---:|---:|---:|
{_table_permanova(permanova)}

![融合与消融](../runs/local_smp_k10_fusion/figures/local_smp_k10_permanova_ablation.png)

[SVG](../runs/local_smp_k10_fusion/figures/local_smp_k10_permanova_ablation.svg)

### 6.2 增量检验

| 比较 | 180 s $\Delta F$ | 180 s FDR | 300 s $\Delta F$ | 300 s FDR |
|---|---:|---:|---:|---:|
{_table_incremental(incremental)}

![增量检验](../runs/local_smp_k10_fusion/figures/local_smp_k10_incremental_tests.png)

[SVG](../runs/local_smp_k10_fusion/figures/local_smp_k10_incremental_tests.svg)

### 6.3 条件非冗余与视角互补性

| 目标视角（条件于其余两视角） | 180 s residual pseudo-F | 180 s FDR | validation $R^2$ | 300 s residual pseudo-F | 300 s FDR |
|---|---:|---:|---:|---:|---:|
{_table_residuals(residuals)}

![条件残差](../runs/local_smp_k10_fusion/figures/local_smp_k10_conditional_residuals.png)

[SVG](../runs/local_smp_k10_fusion/figures/local_smp_k10_conditional_residuals.svg)

![距离相关](../runs/local_smp_k10_fusion/figures/local_smp_k10_distance_correlations.png)

[SVG](../runs/local_smp_k10_fusion/figures/local_smp_k10_distance_correlations.svg)

距离相关只描述三个视角对样本两两关系的相似程度；低相关支持互补可能性，但不能单独证明加入后提高组间分离。

### 6.4 二维投影

![融合 PCA](../runs/local_smp_k10_fusion/figures/local_smp_k10_pca_validation_180.png)

[SVG](../runs/local_smp_k10_fusion/figures/local_smp_k10_pca_validation_180.svg)

PCA 仅用于显示 discovery 拟合坐标下的 validation 投影，不参与置换检验。二维重叠不能否定高维距离差异，二维分离也不能替代 PERMANOVA。

### 6.5 辅助分类

| 表示 | 180 s balanced accuracy | 180 s Macro-F1 | 180 s AUROC | 300 s balanced accuracy | 300 s AUROC |
|---|---:|---:|---:|---:|---:|
{_table_classification(classification)}

![分类消融](../runs/local_smp_k10_fusion/figures/local_smp_k10_classification_ablation.png)

[SVG](../runs/local_smp_k10_fusion/figures/local_smp_k10_classification_ablation.svg)

分类器为 discovery/180 s 内部五折选择 $C$ 的 L2 logistic regression；validation 只报告，不参与调参。分类差值的分层 bootstrap 结果保存在数值产物中。分类性能衡量逐曲判别，pseudo-$F$ 衡量组间/组内距离比，两者不要求同方向变化。

值得单独披露的是：在 Pitch+Rhythm 上加入 SMP 调制后，balanced accuracy 增加 {_format(add_modulation_balanced_delta.delta)}，95% CI [{_format(add_modulation_balanced_delta.ci_low)}, {_format(add_modulation_balanced_delta.ci_high)}]；但对应的距离增量 $\Delta$ pseudo-$F$ 为 {_format(incremental[(incremental['scale_seconds'] == 180.0) & (incremental['comparison'] == 'add_modulation')].iloc[0].delta_pseudo_f)}、FDR={_format(incremental[(incremental['scale_seconds'] == 180.0) & (incremental['comparison'] == 'add_modulation')].iloc[0].p_fdr_bh)}。这说明 SMP 调制含有条件信息，却在当前等权距离中稀释组间/组内距离比；不能据此事后降低其权重。

## 7. 证据边界与最终判断

- **探索性主分析：** validation/180 s 的三视角融合、七表示 PERMANOVA、六个增量和三个条件残差 family，统一 $q\le0.05$。
- **敏感性：** validation/300 s 使用同一个 discovery/180 s 变换和分类器，只检验时长稳健性，不称为独立复制。
- **未使用：** 当前融合没有检验 holdout。SMP K=10 在旧 holdout 打开后提出，因此旧 holdout 不能转化为其确认性证据。
- **融合判定：** 只有完整块显著不足以证明融合有益；必须同时查看相对单视角、留一视角增量与条件残差。分类只作辅助。
- **拓扑解释：** 结果描述离散状态覆盖、转移集中度、连通过程与少量低发生率环；不把 $H_1$ 零膨胀改写为“音乐缺少循环”。
- **因果边界：** 数据集比较不能推出专注效果、治疗作用、认知机制或 ACE-Step 生成改善。

## 8. 复现与审计

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
.\.venv\Scripts\python.exe scripts\run_local_smp_k10_fusion_analysis.py
```

输入 SHA-256：

{chr(10).join(f'- `{path}`: `{digest}`' for path, digest in input_hashes.items())}

主要数值产物：`metadata/local_smp_k10_fusion_permanova.csv`、`metadata/local_smp_k10_fusion_incremental.csv`、`metadata/local_smp_k10_fusion_conditional_residual.csv`、`metadata/local_smp_k10_fusion_distance_correlations.csv`、`metadata/local_smp_k10_fusion_classification.csv`、`metadata/local_smp_k10_fusion_classification_deltas.csv`、`metadata/local_smp_k10_fusion_validation_scores.csv` 与 `metadata/local_smp_k10_fusion_summary.json`。
"""
    REPORT.write_text(report, encoding="utf-8")


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

    identity, frames = _load_aligned()
    discovery, validation_masks, blocks = _fit_blocks(identity, frames)
    train_values = {view: blocks[view]["discovery_180"] for view in VIEWS}
    train_sets = _feature_sets(train_values)
    train_labels = identity.loc[discovery, "group"].astype(str).to_numpy()

    permanova_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    pca_payload: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for scale_index, scale in enumerate(SCALES):
        validation = validation_masks[scale]
        validation_values = {
            view: blocks[view]["validation"][scale] for view in VIEWS
        }
        validation_sets = _feature_sets(validation_values)
        labels = identity.loc[validation, "group"].astype(str).to_numpy()
        validation_identity = identity.loc[validation].reset_index(drop=True)

        for feature_index, name in enumerate(FEATURE_ORDER):
            matrix = validation_sets[name]
            test = permutation_pseudo_f(
                matrix,
                labels,
                permutations=PERMUTATIONS,
                seed=SEED + scale_index * 100 + feature_index,
            )
            permanova_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "input_dimensions": int(matrix.shape[1]),
                    "n_validation": int(validation.sum()),
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        for comparison_index, (name, candidate, baseline, family) in enumerate(COMPARISONS):
            test = paired_incremental_permutation(
                validation_sets[candidate],
                validation_sets[baseline],
                labels,
                permutations=PERMUTATIONS,
                seed=SEED + 200 + scale_index * 100 + comparison_index,
            )
            incremental_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "comparison": name,
                    "candidate": candidate,
                    "baseline": baseline,
                    "family": family,
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        for target_index, target in enumerate(VIEWS):
            predictors = [view for view in VIEWS if view != target]
            train_predictors = equal_block_fusion([train_values[view] for view in predictors])
            validation_predictors = equal_block_fusion(
                [validation_values[view] for view in predictors]
            )
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 17), cv=5).fit(
                train_predictors, train_values[target]
            )
            residual = validation_values[target] - ridge.predict(validation_predictors)
            test = permutation_pseudo_f(
                residual,
                labels,
                permutations=PERMUTATIONS,
                seed=SEED + 400 + scale_index * 100 + target_index,
            )
            residual_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "target_view": target,
                    "conditioned_on": "+".join(predictors),
                    "ridge_alpha": float(ridge.alpha_),
                    "discovery_r2": float(ridge.score(train_predictors, train_values[target])),
                    "validation_r2": float(
                        ridge.score(validation_predictors, validation_values[target])
                    ),
                    "permutations": PERMUTATIONS,
                    **test,
                }
            )

        distances = {
            view: pdist(validation_values[view], metric="euclidean") for view in VIEWS
        }
        for left_index, left in enumerate(VIEWS):
            for right in VIEWS[left_index + 1 :]:
                result = spearmanr(distances[left], distances[right])
                correlation_rows.append(
                    {
                        "analysis_set": "primary_validation_180"
                        if scale == 180.0
                        else "sensitivity_validation_300",
                        "scale_seconds": scale,
                        "view_a": left,
                        "view_b": right,
                        "spearman_rho": float(result.statistic),
                        "p_value": float(result.pvalue),
                        "n_pairwise_distances": int(distances[left].size),
                    }
                )

        predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for feature_index, name in enumerate(FEATURE_ORDER):
            search, predicted, probabilities, classes = _fit_classifier(
                train_sets[name],
                train_labels,
                validation_sets[name],
                seed=SEED + 600 + feature_index,
            )
            metrics = _classification_metrics(labels, predicted, probabilities, classes)
            intervals = _bootstrap_single(
                labels,
                predicted,
                probabilities,
                classes,
                seed=SEED + 800 + scale_index * 100 + feature_index,
            )
            classification_rows.append(
                {
                    "analysis_set": "primary_validation_180"
                    if scale == 180.0
                    else "sensitivity_validation_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "classifier": "L2 logistic regression",
                    "selection": "discovery/180s five-fold CV macro-F1",
                    "best_c": float(search.best_params_["C"]),
                    "cv_macro_f1": float(search.best_score_),
                    "n_train": int(discovery.sum()),
                    "n_validation": int(validation.sum()),
                    **metrics,
                    **intervals,
                }
            )
            predictions[name] = (predicted, probabilities, classes)
            for row_index, row in validation_identity.iterrows():
                score_rows.append(
                    {
                        "segment_id": row["segment_id"],
                        "track_id": row["track_id"],
                        "group": row["group"],
                        "split": row["split"],
                        "scale_seconds": scale,
                        "feature_set": name,
                        "predicted_group": predicted[row_index],
                        "probability_focus": float(
                            probabilities[row_index, list(classes).index("focus")]
                        ),
                    }
                )

        for comparison_index, (name, candidate, baseline, family) in enumerate(COMPARISONS):
            candidate_prediction, candidate_probability, classes = predictions[candidate]
            baseline_prediction, baseline_probability, baseline_classes = predictions[baseline]
            if not np.array_equal(classes, baseline_classes):
                raise RuntimeError("classifier class orders do not match")
            candidate_scores = _classification_metrics(
                labels, candidate_prediction, candidate_probability, classes
            )
            baseline_scores = _classification_metrics(
                labels, baseline_prediction, baseline_probability, classes
            )
            intervals = stratified_bootstrap_differences(
                labels,
                candidate_prediction,
                candidate_probability,
                baseline_prediction,
                baseline_probability,
                metric_functions=_metric_functions(classes),
                resamples=BOOTSTRAPS,
                seed=SEED + 1000 + scale_index * 100 + comparison_index,
            )
            for metric, interval in intervals.items():
                delta_rows.append(
                    {
                        "analysis_set": "primary_validation_180"
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

        if scale == 180.0:
            pca_payload = (train_sets["local_all"], validation_sets["local_all"], labels)

    permanova = pd.DataFrame(permanova_rows)
    incremental = pd.DataFrame(incremental_rows)
    residuals = pd.DataFrame(residual_rows)
    correlations = pd.DataFrame(correlation_rows)
    classification = pd.DataFrame(classification_rows)
    classification_deltas = pd.DataFrame(delta_rows)
    scores = pd.DataFrame(score_rows)

    for frame, p_column in (
        (permanova, "p_value"),
        (incremental, "p_value_one_sided"),
        (residuals, "p_value"),
    ):
        frame["p_fdr_bh"] = np.nan
        for _, indices in frame.groupby("scale_seconds").groups.items():
            frame.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
                frame.loc[indices, p_column].to_numpy(float)
            )

    for frame, path in (
        (permanova, PERMANOVA_PATH),
        (incremental, INCREMENTAL_PATH),
        (residuals, RESIDUAL_PATH),
        (correlations, CORRELATION_PATH),
        (classification, CLASSIFICATION_PATH),
        (classification_deltas, CLASSIFICATION_DELTA_PATH),
        (scores, SCORES_PATH),
    ):
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    assert pca_payload is not None
    _plot_permanova(permanova)
    _plot_incremental(incremental)
    _plot_classification(classification)
    _plot_correlations(correlations)
    _plot_residuals(residuals)
    _plot_pca(*pca_payload)

    input_hashes = {
        str(path.relative_to(ROOT).as_posix()): _sha256(path)
        for path in VIEW_FILES.values()
    }
    output_paths = (
        PERMANOVA_PATH,
        INCREMENTAL_PATH,
        RESIDUAL_PATH,
        CORRELATION_PATH,
        CLASSIFICATION_PATH,
        CLASSIFICATION_DELTA_PATH,
        SCORES_PATH,
        *sorted(FIGURES.glob("*")),
    )
    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "exploratory local fusion of pitch, rhythm, and modulation SMP K=10 Path Homology",
        "evidence_role": {
            "validation_180": "exploratory primary integration",
            "validation_300": "same-track duration sensitivity",
            "holdout": "not tested because modulation SMP K=10 postdates the old holdout opening",
        },
        "design": {
            "fit_split": "discovery/180s only",
            "views": list(VIEWS),
            "metrics_per_view": len(TOPOLOGY_METRICS),
            "fusion": "rank-normalized Mahalanobis blocks with equal expected squared-distance weight",
            "permutations": PERMUTATIONS,
            "bootstraps": BOOTSTRAPS,
            "fdr_q": FDR_Q,
            "seed": SEED,
        },
        "effective_rank": {
            view: int(blocks[view]["effective_rank"]) for view in VIEWS
        },
        "input_sha256": input_hashes,
        "output_sha256": {
            str(path.relative_to(ROOT).as_posix()): _sha256(path) for path in output_paths
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_report(
        permanova,
        incremental,
        residuals,
        correlations,
        classification,
        classification_deltas,
        blocks,
        input_hashes,
    )
    print(REPORT)
    print(SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
