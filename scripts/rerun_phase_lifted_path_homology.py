# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.analysis_inputs import audit_analysis_inputs
from features.batch import _sha256, _write_json_atomic
from repetition.analysis import (
    PATH_REPRESENTATIONS,
    _calibration_tests,
    _candidate_data,
    _compute_features,
    _dominant_lag_from_distance,
    _load_model,
    _quality_filter,
    _standard_distance,
    load_config,
    transposition_invariant_chroma_distance,
)
from topology.statistics import benjamini_hochberg

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "phase_lifted_path_homology_20260802"
FIGURE_DIR = RUN_DIR / "figures"
METADATA = ROOT / "metadata"
FEATURE_PATH = METADATA / "phase_lifted_path_homology_features.csv"
TEST_PATH = METADATA / "phase_lifted_path_homology_tests.csv"
CALIBRATION_PATH = METADATA / "phase_lifted_path_homology_calibration.csv"
STABILITY_PATH = METADATA / "phase_lifted_path_homology_scale_stability.csv"
CLASSIFICATION_PATH = METADATA / "phase_lifted_path_homology_classification.csv"
REPRESENTATIVE_PATH = METADATA / "phase_lifted_path_homology_representative_edges.csv"
EXCLUSION_PATH = METADATA / "phase_lifted_path_homology_exclusions.csv"
SUMMARY_PATH = METADATA / "phase_lifted_path_homology_summary.json"
REPORT_PATH = ROOT / "docs" / "phase-lifted-path-homology-analysis.md"

LABELS = {
    "path_acoustic_phase": "Acoustic phase",
    "path_rhythm_phase": "Rhythm phase",
    "path_chroma_phase": "Chroma phase",
}
GROUP_LABELS = {"focus": "Open Focus", "classical": "Classical"}
COLORS = {"focus": "#2B6CB0", "classical": "#D95F02"}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _balanced_calibration_manifest(
    manifest: pd.DataFrame, tracks_per_group: int, seed: int
) -> pd.DataFrame:
    base = manifest[(manifest["split"] == "discovery") & (manifest["scale_seconds"] == 180.0)]
    sensitivity = manifest[
        (manifest["split"] == "discovery") & (manifest["scale_seconds"] == 300.0)
    ]
    base = base[base["track_id"].isin(sensitivity["track_id"])]
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for group in ("focus", "classical"):
        candidates = np.sort(base.loc[base["group"] == group, "track_id"].unique())
        if len(candidates) < tracks_per_group:
            raise RuntimeError(f"not enough eligible {group} tracks for calibration")
        chosen.extend(
            str(value) for value in rng.choice(candidates, tracks_per_group, replace=False)
        )
    return manifest[
        (manifest["split"] == "discovery")
        & manifest["track_id"].isin(chosen)
        & manifest["scale_seconds"].isin([180.0, 300.0])
    ].copy()


def _rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    statistic = mannwhitneyu(first, second, alternative="two-sided", method="auto").statistic
    return 2.0 * float(statistic) / (len(first) * len(second)) - 1.0


def _bootstrap_effect_interval(
    first: np.ndarray, second: np.ndarray, *, seed: int, repetitions: int = 3000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    effects = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        left = rng.choice(first, len(first), replace=True)
        right = rng.choice(second, len(second), replace=True)
        effects[index] = np.mean(left[:, None] > right[None, :]) - np.mean(
            left[:, None] < right[None, :]
        )
    lower, upper = np.quantile(effects, [0.025, 0.975])
    return float(lower), float(upper)


def _comparison_tests(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roles = (
        ("discovery_descriptive", "discovery", 180.0),
        ("primary_validation", "validation", 180.0),
        ("duration_sensitivity", "validation", 300.0),
    )
    for role, split, scale in roles:
        subset = features[(features["split"] == split) & (features["scale_seconds"] == scale)]
        for representation in PATH_REPRESENTATIONS:
            view = subset[subset["representation"] == representation]
            focus = view.loc[view["group"] == "focus", "loop_score"].to_numpy(float)
            classical = view.loc[view["group"] == "classical", "loop_score"].to_numpy(float)
            two_sided = mannwhitneyu(focus, classical, alternative="two-sided", method="auto")
            greater = mannwhitneyu(focus, classical, alternative="greater", method="auto")
            effect = 2.0 * float(two_sided.statistic) / (len(focus) * len(classical)) - 1.0
            digest = hashlib.sha256(f"{seed}:{role}:{representation}".encode()).digest()
            ci_low, ci_high = _bootstrap_effect_interval(
                focus,
                classical,
                seed=int.from_bytes(digest[:8], "little"),
            )
            rows.append(
                {
                    "role": role,
                    "split": split,
                    "scale_seconds": scale,
                    "representation": representation,
                    "method": "phase_lifted_path_homology",
                    "n_focus": len(focus),
                    "n_classical": len(classical),
                    "focus_median": float(np.median(focus)),
                    "classical_median": float(np.median(classical)),
                    "rank_biserial_focus_minus_classical": effect,
                    "effect_ci95_low": ci_low,
                    "effect_ci95_high": ci_high,
                    "p_two_sided": float(two_sided.pvalue),
                    "p_focus_greater": float(greater.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    result["p_two_sided_fdr_bh"] = np.nan
    result["p_focus_greater_fdr_bh"] = np.nan
    for role, indices in result.groupby("role").groups.items():
        del role
        result.loc[indices, "p_two_sided_fdr_bh"] = benjamini_hochberg(
            result.loc[indices, "p_two_sided"].to_numpy(float)
        )
        result.loc[indices, "p_focus_greater_fdr_bh"] = benjamini_hochberg(
            result.loc[indices, "p_focus_greater"].to_numpy(float)
        )
    return result.sort_values(["role", "p_two_sided_fdr_bh", "representation"])


def _scale_stability(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("discovery", "validation", "holdout"):
        for representation in PATH_REPRESENTATIONS:
            subset = features[
                (features["split"] == split) & (features["representation"] == representation)
            ]
            for group in ("focus", "classical"):
                paired = subset[subset["group"] == group].pivot(
                    index="track_id", columns="scale_seconds", values="loop_score"
                )
                paired = paired.dropna(subset=[180.0, 300.0])
                correlation = spearmanr(paired[180.0], paired[300.0])
                differences = paired[300.0] - paired[180.0]
                if np.allclose(differences, 0.0):
                    signed_p = 1.0
                else:
                    signed_p = float(wilcoxon(differences, alternative="two-sided").pvalue)
                rows.append(
                    {
                        "split": split,
                        "group": group,
                        "representation": representation,
                        "n_tracks": len(paired),
                        "spearman_rho_180_vs_300": float(correlation.statistic),
                        "spearman_p": float(correlation.pvalue),
                        "median_300_minus_180": float(np.median(differences)),
                        "wilcoxon_p_two_sided": signed_p,
                    }
                )
    return pd.DataFrame(rows)


def _classification(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    wide = features.pivot(
        index=["segment_id", "track_id", "group", "split", "scale_seconds"],
        columns="representation",
        values="loop_score",
    ).reset_index()
    columns = list(PATH_REPRESENTATIONS)
    subset = wide[wide["scale_seconds"] == 180.0]
    train = subset[subset["split"] == "discovery"]
    validation = subset[subset["split"] == "validation"]
    y_train = (train["group"] == "focus").astype(int).to_numpy()
    y_validation = (validation["group"] == "focus").astype(int).to_numpy()
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline,
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
    ).fit(train[columns], y_train)
    predictions = search.predict(validation[columns])
    probabilities = search.predict_proba(validation[columns])[:, 1]
    return pd.DataFrame(
        [
            {
                "task": "open_focus_vs_classical",
                "role": "auxiliary_not_primary",
                "scale_seconds": 180.0,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_features": len(columns),
                "features": ";".join(columns),
                "best_c": float(search.best_params_["classifier__C"]),
                "cv_macro_f1": float(search.best_score_),
                "balanced_accuracy": float(balanced_accuracy_score(y_validation, predictions)),
                "macro_f1": float(f1_score(y_validation, predictions, average="macro")),
                "auroc": float(roc_auc_score(y_validation, probabilities)),
            }
        ]
    )


def _phase_edge_details(
    values: np.ndarray,
    hop_seconds: float,
    transposition_invariant: bool,
    phase_bins: int,
    config,
) -> tuple[int, np.ndarray, np.ndarray]:
    distances = (
        transposition_invariant_chroma_distance(values)
        if transposition_invariant
        else _standard_distance(values)
    )
    upper = distances[np.triu_indices(len(distances), k=3)]
    positive = upper[upper > 1e-9]
    scale = float(np.median(positive)) if positive.size else 1.0
    period, _ = _dominant_lag_from_distance(distances, config)
    recurrence = np.exp(-np.diag(distances, k=period) / max(scale, 1e-8))
    phase = np.arange(len(recurrence)) % period * phase_bins // period
    coherence = np.asarray([np.mean(recurrence[phase == index]) for index in range(phase_bins)])
    edge_weights = np.minimum(coherence, np.roll(coherence, -1))
    return period, coherence, edge_weights


def _representative_edges(manifest: pd.DataFrame, features: pd.DataFrame, config) -> pd.DataFrame:
    model = _load_model(ROOT)
    rows: list[dict[str, object]] = []
    target_representation = "path_rhythm_phase"
    view = features[
        (features["split"] == "validation")
        & (features["scale_seconds"] == 180.0)
        & (features["representation"] == target_representation)
    ]
    for group in ("focus", "classical"):
        group_view = view[view["group"] == group].copy()
        median = float(group_view["loop_score"].median())
        chosen = group_view.iloc[
            int(np.argmin(np.abs(group_view["loop_score"].to_numpy(float) - median)))
        ]
        manifest_row = manifest[manifest["segment_id"] == chosen["segment_id"]].iloc[0].to_dict()
        values, hop_seconds, transposition_invariant = _candidate_data(
            ROOT, manifest_row, model, config
        )[target_representation]
        period, coherence, weights = _phase_edge_details(
            values,
            hop_seconds,
            transposition_invariant,
            config.phase_bins,
            config,
        )
        for source, weight in enumerate(weights):
            rows.append(
                {
                    "group": group,
                    "representation": target_representation,
                    "segment_id": chosen["segment_id"],
                    "track_id": chosen["track_id"],
                    "source_phase": source,
                    "target_phase": (source + 1) % config.phase_bins,
                    "source_coherence": float(coherence[source]),
                    "target_coherence": float(coherence[(source + 1) % config.phase_bins]),
                    "edge_weight": float(weight),
                    "dominant_period_blocks": period,
                    "dominant_period_seconds": float(period * hop_seconds),
                    "loop_score": float(np.min(weights)),
                }
            )
    return pd.DataFrame(rows)


def _configure_matplotlib() -> None:
    cache = RUN_DIR / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "phase-lifted-path-homology-20260802"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


def _plot_distributions(features: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    validation = features[features["split"] == "validation"]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for axis, representation in zip(axes, PATH_REPRESENTATIONS, strict=True):
        view = validation[validation["representation"] == representation]
        values: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        for scale in (180.0, 300.0):
            for group in ("focus", "classical"):
                values.append(
                    view[(view["scale_seconds"] == scale) & (view["group"] == group)][
                        "loop_score"
                    ].to_numpy(float)
                )
                labels.append(f"{int(scale)}s\n{GROUP_LABELS[group]}")
                colors.append(COLORS[group])
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.65)
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
        axis.set_title(LABELS[representation])
        axis.grid(axis="y", alpha=0.22)
        axis.set_xlabel("Duration and group")
    axes[0].set_ylabel("Phase-cycle loop score")
    figure.suptitle("Validation distributions under the frozen phase-lifted construction")
    figure.tight_layout()
    paths = [
        FIGURE_DIR / "validation_loop_score_distributions.png",
        FIGURE_DIR / "validation_loop_score_distributions.svg",
    ]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return paths


def _plot_filtration(features: pd.DataFrame, thresholds: np.ndarray) -> list[Path]:
    import matplotlib.pyplot as plt

    subset = features[(features["split"] == "validation") & (features["scale_seconds"] == 180.0)]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    for axis, representation in zip(axes, PATH_REPRESENTATIONS, strict=True):
        view = subset[subset["representation"] == representation]
        for group in ("focus", "classical"):
            scores = view.loc[view["group"] == group, "loop_score"].to_numpy(float)
            survival = np.mean(scores[:, None] >= thresholds[None, :], axis=0)
            axis.plot(
                thresholds,
                survival,
                marker="o",
                markersize=3.2,
                linewidth=1.8,
                color=COLORS[group],
                label=GROUP_LABELS[group],
            )
        axis.set_title(LABELS[representation])
        axis.set_xlabel("Superlevel threshold τ")
        axis.grid(alpha=0.22)
        axis.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Fraction of segments with β₁(τ) = 1")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle("Validation 180s: persistence of the directed phase cycle")
    figure.tight_layout()
    paths = [
        FIGURE_DIR / "validation_h1_filtration.png",
        FIGURE_DIR / "validation_h1_filtration.svg",
    ]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return paths


def _plot_effects(tests: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    view = tests[tests["role"].isin(["primary_validation", "duration_sensitivity"])].copy()
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    y_base = np.arange(len(PATH_REPRESENTATIONS), dtype=float)
    offsets = {180.0: -0.14, 300.0: 0.14}
    markers = {180.0: "o", 300.0: "s"}
    for scale in (180.0, 300.0):
        part = (
            view[view["scale_seconds"] == scale]
            .set_index("representation")
            .loc[list(PATH_REPRESENTATIONS)]
        )
        effects = part["rank_biserial_focus_minus_classical"].to_numpy(float)
        low = part["effect_ci95_low"].to_numpy(float)
        high = part["effect_ci95_high"].to_numpy(float)
        axis.errorbar(
            effects,
            y_base + offsets[scale],
            xerr=np.vstack([effects - low, high - effects]),
            fmt=markers[scale],
            capsize=3,
            color="#4C4C4C" if scale == 180.0 else "#8C8C8C",
            label=f"{int(scale)}s",
        )
    axis.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axis.set_yticks(y_base, [LABELS[name] for name in PATH_REPRESENTATIONS])
    axis.set_xlabel("Rank-biserial effect (Open Focus − Classical), bootstrap 95% CI")
    axis.set_xlim(-1.02, 1.02)
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False)
    axis.set_title("Validation group effects and duration sensitivity")
    figure.tight_layout()
    paths = [FIGURE_DIR / "validation_effect_sizes.png", FIGURE_DIR / "validation_effect_sizes.svg"]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return paths


def _plot_calibration(calibration: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    ordered = calibration.set_index("representation").loc[list(PATH_REPRESENTATIONS)]
    x = np.arange(len(ordered))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.bar(
        x - width / 2,
        ordered["synthetic_loop_median"],
        width,
        label="Synthetic loop",
        color="#2B6CB0",
        alpha=0.75,
    )
    axis.bar(
        x + width / 2,
        ordered["shuffled_median"],
        width,
        label="Time shuffled",
        color="#D95F02",
        alpha=0.75,
    )
    axis.set_xticks(x, [LABELS[name] for name in ordered.index], rotation=15, ha="right")
    axis.set_ylabel("Median loop score")
    axis.set_title("Discovery-only construct calibration (24 tracks per group)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    paths = [FIGURE_DIR / "construct_calibration.png", FIGURE_DIR / "construct_calibration.svg"]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return paths


def _plot_representative_cycles(edges: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 6, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    for axis, group in zip(axes, ("focus", "classical"), strict=True):
        view = edges[edges["group"] == group].sort_values("source_phase")
        weights = view["edge_weight"].to_numpy(float)
        for source, weight in enumerate(weights):
            target = (source + 1) % 6
            arrow = FancyArrowPatch(
                positions[source],
                positions[target],
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0 + 5.0 * weight,
                color=COLORS[group],
                alpha=0.78,
                shrinkA=15,
                shrinkB=15,
                connectionstyle="arc3,rad=0.06",
            )
            axis.add_patch(arrow)
            midpoint = (positions[source] + positions[target]) / 2
            axis.text(
                midpoint[0] * 1.08,
                midpoint[1] * 1.08,
                f"{weight:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=420,
            color=COLORS[group],
            alpha=0.22,
            edgecolors=COLORS[group],
            linewidths=1.5,
        )
        for index, (x, y) in enumerate(positions):
            axis.text(x, y, str(index), ha="center", va="center", fontweight="bold")
        first = view.iloc[0]
        axis.set_title(
            f"{GROUP_LABELS[group]} representative\n"
            f"loop={first.loop_score:.3f}, period={first.dominant_period_seconds:.1f}s"
        )
        axis.set_xlim(-1.4, 1.4)
        axis.set_ylim(-1.35, 1.35)
        axis.set_aspect("equal")
        axis.axis("off")
    figure.suptitle("Median-like validation 180s rhythm phase cycles")
    figure.tight_layout()
    paths = [
        FIGURE_DIR / "representative_rhythm_phase_cycles.png",
        FIGURE_DIR / "representative_rhythm_phase_cycles.svg",
    ]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return paths


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _markdown_table(
    frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 3
) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(columns))) + "|",
    ]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        for key, _ in columns:
            value = getattr(row, key)
            if isinstance(value, (float, np.floating)):
                values.append(_fmt(float(value), digits))
            elif isinstance(value, (bool, np.bool_)):
                values.append("是" if value else "否")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    *,
    manifest: pd.DataFrame,
    excluded: pd.DataFrame,
    features: pd.DataFrame,
    calibration: pd.DataFrame,
    tests: pd.DataFrame,
    stability: pd.DataFrame,
    classification: pd.DataFrame,
    representatives: pd.DataFrame,
    config,
    figure_paths: list[Path],
) -> None:
    del manifest, representatives
    primary = tests[tests["role"] == "primary_validation"].copy()
    sensitivity = tests[tests["role"] == "duration_sensitivity"].copy()
    comparison = primary.merge(
        sensitivity[
            [
                "representation",
                "rank_biserial_focus_minus_classical",
                "p_two_sided_fdr_bh",
            ]
        ],
        on="representation",
        suffixes=("_180", "_300"),
    )
    comparison["representation"] = comparison["representation"].map(LABELS)
    calibration_table = calibration.copy()
    calibration_table["representation"] = calibration_table["representation"].map(LABELS)
    validation_stability = stability[stability["split"] == "validation"].copy()
    validation_stability["representation"] = validation_stability["representation"].map(LABELS)
    validation_stability["group"] = validation_stability["group"].map(GROUP_LABELS)
    holdout = (
        features[(features["split"] == "holdout") & (features["scale_seconds"] == 180.0)]
        .groupby(["representation", "group"])["loop_score"]
        .agg(["count", "median", "mean"])
        .reset_index()
    )
    holdout["representation"] = holdout["representation"].map(LABELS)
    holdout["group"] = holdout["group"].map(GROUP_LABELS)
    significant = primary[primary["p_two_sided_fdr_bh"] <= 0.05]
    significant_names = [LABELS[name] for name in significant["representation"]]
    significance_text = "、".join(significant_names) if significant_names else "无"
    calibration_passes = [
        LABELS[name] for name in calibration.loc[calibration["calibration_pass"], "representation"]
    ]
    figure_links = {
        path.stem: f"../{path.relative_to(ROOT).as_posix()}"
        for path in figure_paths
        if path.suffix == ".png"
    }
    state_model = json.loads(
        (ROOT / "features" / "models" / "state_model.json").read_text(encoding="utf-8")
    )
    lines = [
        "# 相位提升路径同调：Open Focus 与 Classical 的重新分析",
        "",
        f"生成日期：{date.today().isoformat()}。本报告对应 2026-08-02 两组数据迁移后的独立重跑。",
        "",
        "## 摘要",
        "",
        f"本次在当前 600 首曲目的 Open Focus/Classical 数据上重新执行相位提升路径同调。方法参数沿用原冻结配置，不根据本次结果调节：6 个相位节点、4 帧块聚合、至少 96 个原始时间步、至少 3 个周期、候选周期不超过 32 个块、超水平阈值 0.05–0.95。三个预定义视角 Acoustic、Rhythm、Chroma 全部纳入，未再次依据组间 p 值筛选。共分析 {features['segment_id'].nunique():,} 个合格片段、{len(features):,} 个片段-视角；质量门槛排除 {len(excluded):,} 个片段。",
        "",
        f"在 discovery-only 构造校准中，通过人工循环优于时间打乱门槛的视角为：{'、'.join(calibration_passes) if calibration_passes else '无'}。validation/180s 的三视角双侧 Mann–Whitney 检验在三视角检验家族内进行 BH-FDR 后，q≤0.05 的视角为：{significance_text}。这是一项迁移后的重新分析，不是对原 Focus>Pop 假设的确认性复制；Classical 比较对象发生改变，而且当前 validation 已在其他分析中被查看，因此结果应解释为当前数据上的观察性组间差异。",
        "",
        "## 1. 方法思想",
        "",
        "普通状态转移图回答“哪些声学状态彼此转换”；相位提升则回答“一个候选重复周期内部，各相位是否按稳定顺序闭合”。它先从声学轨迹估计主导重复周期，再把周期位置压缩到 6 个相位节点。若所有相邻相位都能跨周期稳定复现，就形成有向环，其一维路径同调非零。最弱的一条相位边决定该环能够承受多高的边权过滤阈值。",
        "",
        "```mermaid",
        "flowchart LR",
        '    A["声学时序 x_t"] --> B["4 帧块聚合"]',
        '    B --> C["距离矩阵 D_ij"]',
        '    C --> D["估计主导周期 P*"]',
        '    D --> E["映射为 6 个相位"]',
        '    E --> F["相位一致性 c_k"]',
        '    F --> G["有向环边权 w_k"]',
        '    G --> H["超水平过滤 G_tau"]',
        '    H --> I["Path H1 与 loop score"]',
        "```",
        "",
        "## 2. 数学原理与公式",
        "",
        "### 2.1 三种输入表示",
        "",
        "- Acoustic：对 discovery 拟合的声学标准化与 PCA 表示取前 8 维，再按 4 帧求均值；块间隔为 2 秒。",
        "- Rhythm：对节奏向量按 discovery 模型插补、标准化，再按 4 帧求均值；块间隔为 2 秒。",
        "- Chroma：逐帧 L2 归一化后按 4 帧求均值；距离对 12 种循环移调取最优匹配。",
        "",
        "对 Acoustic 与 Rhythm，令 z_i 为块级向量，距离为",
        "",
        r"$$D_{ij}=\frac{\lVert z_i-z_j\rVert_2}{\sqrt d},\qquad z_{ij}=\frac{x_{ij}-\operatorname{median}_i x_{ij}}{\max(\operatorname{sd}_i x_{ij},10^{-8})}.$$",
        "",
        r"对 Chroma，先单位化 $u_i=x_i/\lVert x_i\rVert_2$，再定义移调不变距离",
        "",
        r"$$D_{ij}=\sqrt{\max\left(0,2-2\max_{s\in\{0,\ldots,11\}}u_i^\top R_su_j\right)}.$$",
        "",
        "其中 $R_s$ 表示循环移动 $s$ 个半音。",
        "",
        "### 2.2 主导周期与相位提升",
        "",
        r"在候选集合 $\mathcal P=\{K,\ldots,\min(P_{\max},\lfloor N/C\rfloor)\}$ 中选择",
        "",
        r"$$P^*=\arg\min_{P\in\mathcal P}\operatorname{median}_i D_{i,i+P}.$$",
        "",
        rf"其中相位数 $K={config.phase_bins}$、最小周期数 $C={config.min_cycles}$、$P_{{\max}}={config.max_period_blocks}$。以非邻近距离的正值中位数 $s$ 为尺度，跨周期复现强度为",
        "",
        r"$$r_i=\exp\left(-D_{i,i+P^*}/s\right).$$",
        "",
        "将周期位置映射到离散相位",
        "",
        r"$$q_i=\left\lfloor\frac{(i\bmod P^*)K}{P^*}\right\rfloor,\qquad c_k=\operatorname{mean}\{r_i:q_i=k\}.$$",
        "",
        r"并构造有向边 $k\to(k+1)\bmod K$，边权为",
        "",
        r"$$w_k=\min(c_k,c_{k+1}).$$",
        "",
        "取相邻相位一致性的较小值，是为了让每条边同时受其起点和终点的稳定性约束。",
        "",
        "### 2.3 GLMY 路径同调",
        "",
        r"在有向图中，允许的 $p$-路径是顶点序列 $e_{i_0\ldots i_p}$，相邻顶点之间均存在有向边。边界算子为",
        "",
        r"$$\partial e_{i_0\ldots i_p}=\sum_{q=0}^{p}(-1)^q e_{i_0\ldots\widehat{i_q}\ldots i_p}.$$",
        "",
        "并取保持允许性的链空间",
        "",
        r"$$\Omega_p=\{v\in A_p:\partial v\in A_{p-1}\},\qquad H_p=\ker(\partial_p|_{\Omega_p})/\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}}).$$",
        "",
        "Betti 数为",
        "",
        r"$$\beta_p=\dim\ker(\partial_p|_{\Omega_p})-\operatorname{rank}(\partial_{p+1}|_{\Omega_{p+1}}).$$",
        "",
        "对边权采用超水平过滤",
        "",
        r"$$G_\tau=(V,\{e:w(e)\ge\tau\}),\qquad \tau\in\{0.05,0.10,\ldots,0.95\}.$$",
        "",
        r"因为本方法构造的是单一 6 节点有向环，所以当且仅当所有 6 条边均保留时 $\beta_1=1$。因此连续临界值",
        "",
        r"$$\lambda=\min_k w_k$$",
        "",
        r"就是 `loop_score`；在离散阈值上，$\beta_1(\tau)=\mathbf 1[\tau\le\lambda]$。该指标衡量最弱相位连接，而不是一般有向图中所有可能 H1 类的总复杂度。",
        "",
        "## 3. 数据与分析协议",
        "",
        "- 当前规范数据：Open Focus 300 首、Classical 300 首；每首均有 180s 与 300s 片段。",
        "- 对称切分：每组 discovery 195、validation 60、holdout 45。",
        "- 质量排除：`classical_musicnet_2305` 的 180s/300s chroma 时间步均为 94，低于冻结门槛 96；排除发生在 discovery，validation 仍为完整的 60+60。逐行记录见 `metadata/phase_lifted_path_homology_exclusions.csv`。",
        "- 状态模型只来自 discovery/180s；模型 SHA-256：`"
        + str(state_model["model_sha256"])
        + "`。",
        "- 构造校准：从 discovery 中按固定随机种子每组抽取 24 首，比较人工循环与时间打乱。校准不用于从三视角中删选当前组间结果。",
        "- 主要比较：validation/180s；双侧 Mann–Whitney U，三视角 BH-FDR。原先的 Focus>Pop 单侧方向仅作为历史背景，不转移为新的确认性主检验。",
        "- 时长敏感性：同一 validation 曲目的 300s 结果，以及 180/300s Spearman 相关。",
        "- holdout：仅报告描述统计，不进行新的显著性开启或调参。",
        "- 分类：三项 loop score 的逻辑回归，discovery 训练、validation 测试；仅作辅助预测检查。",
        "",
        "## 4. 构造校准",
        "",
        *_markdown_table(
            calibration_table,
            [
                ("representation", "视角"),
                ("n_tracks", "n"),
                ("synthetic_loop_median", "人工循环中位数"),
                ("shuffled_median", "打乱中位数"),
                ("median_delta", "中位差"),
                ("positive_fraction", "正差比例"),
                ("p_fdr_bh", "FDR q"),
                ("calibration_pass", "通过"),
            ],
        ),
        "",
        f"![Construct calibration]({figure_links['construct_calibration']})",
        "",
        "人工循环由同一个块序列精确平铺，因此三个视角的合成分数都达到理论上限 1.0；这是一项管线/构造校准，不是额外的经验发现。该校准只说明指标能否对预设的循环化操作作出响应，并不能证明真实音乐中的高分必然对应感知到的重复或专注效果。",
        "",
        "## 5. validation 组间结果",
        "",
        *_markdown_table(
            comparison,
            [
                ("representation", "视角"),
                ("n_focus", "Focus n"),
                ("n_classical", "Classical n"),
                ("focus_median", "Focus 180s"),
                ("classical_median", "Classical 180s"),
                ("rank_biserial_focus_minus_classical_180", "180s 效应"),
                ("effect_ci95_low", "95% CI 下限"),
                ("effect_ci95_high", "95% CI 上限"),
                ("p_two_sided_fdr_bh_180", "180s FDR"),
                ("rank_biserial_focus_minus_classical_300", "300s 效应"),
                ("p_two_sided_fdr_bh_300", "300s FDR"),
            ],
        ),
        "",
        f"![Validation distributions]({figure_links['validation_loop_score_distributions']})",
        "",
        f"![Validation effects]({figure_links['validation_effect_sizes']})",
        "",
        "效应量为 rank-biserial(Open Focus − Classical)：正值表示 Open Focus 的 loop score 倾向更高，负值表示 Classical 更高。置信区间由固定随机种子的 3,000 次分组内 bootstrap 得到。",
        "",
        "## 6. H1 过滤曲线",
        "",
        f"![Path H1 filtration]({figure_links['validation_h1_filtration']})",
        "",
        "纵轴是在给定阈值仍保有完整 6 相位有向环的片段比例。曲线右移意味着更多片段的最弱边仍较强；它是 loop score 生存函数的路径同调解释，而不是另一个独立统计终点。",
        "",
        "## 7. 代表性相位环",
        "",
        f"![Representative phase cycles]({figure_links['representative_rhythm_phase_cycles']})",
        "",
        "图中选取 validation/180s 各组 rhythm loop score 最接近该组中位数的曲目。节点是 6 个相位，箭头数字是边权；最小边权即 loop score。线宽用于帮助观察边权差异，代表图不参与显著性检验。逐边数值见 `metadata/phase_lifted_path_homology_representative_edges.csv`。",
        "",
        "## 8. 时长稳定性",
        "",
        *_markdown_table(
            validation_stability,
            [
                ("representation", "视角"),
                ("group", "组别"),
                ("n_tracks", "n"),
                ("spearman_rho_180_vs_300", "Spearman ρ"),
                ("median_300_minus_180", "300−180 中位差"),
                ("wilcoxon_p_two_sided", "配对 p"),
            ],
        ),
        "",
        "300s 不是独立样本，而是同一曲目的时长敏感性视图；因此不能把 180s 与 300s 的同向显著误写为独立复制。",
        "",
        "## 9. holdout 描述统计",
        "",
        *_markdown_table(
            holdout,
            [
                ("representation", "视角"),
                ("group", "组别"),
                ("count", "n"),
                ("median", "中位数"),
                ("mean", "均值"),
            ],
        ),
        "",
        "此处没有对 holdout 计算或报告新的 p 值。当前 Classical holdout 不含钢琴独奏，并且其中部分曲目在旧切分中曾属于 discovery，不能称为 pristine 外部确认集。",
        "",
        "## 10. 辅助分类",
        "",
        *_markdown_table(
            classification,
            [
                ("n_train", "训练 n"),
                ("n_validation", "验证 n"),
                ("cv_macro_f1", "CV Macro-F1"),
                ("balanced_accuracy", "Balanced accuracy"),
                ("macro_f1", "Validation Macro-F1"),
                ("auroc", "AUROC"),
            ],
        ),
        "",
        "分类结果只回答三个 loop score 是否具有联合判别信息，不等价于拓扑机制、感知效果或因果效应。",
        "",
        "## 11. 结论与证据边界",
        "",
        "### 可以支持",
        "",
        "- 相位提升构造可被当前特征层稳定执行，并能以一个明确临界值连接相位一致性与 Path H1 过滤。",
        "- discovery-only 人工循环/打乱校准可用于判断指标是否响应顺序化循环结构；具体通过情况见校准表。",
        "- validation 中观察到的组间差异及其 300s 时长敏感性，可以描述为 Open Focus 与 Classical 在候选周期相位闭合强度上的差异。",
        "",
        "### 不能支持",
        "",
        "- 不能把本次 Classical 比较解释为原 Focus>Pop 假设的确认性复制。",
        "- 不能由 6 节点人工相位环推出音乐本身具有一般意义上的复杂 H1 拓扑；这里的 H1 由预定义相位闭环构造诱导。",
        "- 不能推出专注力改善、临床效果、生成质量或任何因果机制。",
        "- 不能把 300s 结果当作独立数据集复制，也不能把 holdout 描述视为 pristine 外部验证。",
        "",
        "### 方法局限",
        "",
        "- 主导周期由全段距离对角线的中位数最小化得到，可能把缓慢结构重复与局部节拍重复混合。",
        "- 相位节点数固定为 6，压缩了周期内部的细粒度变化；本次不做事后 K 值优化。",
        "- `loop_score` 是最弱边统计量，对单个薄弱相位敏感；它与过滤曲线不是独立证据。",
        "- Acoustic/Rhythm 使用固定 4 帧块，而 Chroma 的实际秒级步长取决于其时间戳；跨视角数值不应当直接作绝对大小比较。",
        "- Classical 的风格与乐器构成可能成为组间差异来源，结果不能自动推广到所有非专注音乐。",
        "",
        "## 12. 可复现产物",
        "",
        "- `scripts/rerun_phase_lifted_path_homology.py`",
        "- `metadata/phase_lifted_path_homology_features.csv`",
        "- `metadata/phase_lifted_path_homology_tests.csv`",
        "- `metadata/phase_lifted_path_homology_calibration.csv`",
        "- `metadata/phase_lifted_path_homology_scale_stability.csv`",
        "- `metadata/phase_lifted_path_homology_classification.csv`",
        "- `metadata/phase_lifted_path_homology_representative_edges.csv`",
        "- `metadata/phase_lifted_path_homology_exclusions.csv`",
        "- `metadata/phase_lifted_path_homology_summary.json`",
        "- `runs/phase_lifted_path_homology_20260802/figures/`（PNG 与 SVG）",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    input_audit = audit_analysis_inputs(root=ROOT)
    config = load_config(ROOT)
    manifest_path = METADATA / "feature_segments.csv"
    manifest = pd.read_csv(manifest_path)
    groups = set(manifest["group"].astype(str).unique())
    if groups != {"focus", "classical"}:
        raise RuntimeError(f"expected current two-group manifest, found {sorted(groups)}")
    eligible, excluded = _quality_filter(manifest, PATH_REPRESENTATIONS, config)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    features = _compute_features(
        ROOT,
        eligible,
        config,
        PATH_REPRESENTATIONS,
        calibrate=False,
    )
    calibration_manifest = _balanced_calibration_manifest(
        eligible, config.exploration_tracks_per_group, config.random_seed
    )
    calibration_features = _compute_features(
        ROOT,
        calibration_manifest,
        config,
        PATH_REPRESENTATIONS,
        calibrate=True,
    )
    calibration = _calibration_tests(calibration_features, config)
    tests = _comparison_tests(features, config.random_seed)
    stability = _scale_stability(features)
    classification = _classification(features, config.random_seed)
    representative_edges = _representative_edges(eligible, features, config)

    _write_csv(FEATURE_PATH, features)
    _write_csv(TEST_PATH, tests)
    _write_csv(CALIBRATION_PATH, calibration)
    _write_csv(STABILITY_PATH, stability)
    _write_csv(CLASSIFICATION_PATH, classification)
    _write_csv(REPRESENTATIVE_PATH, representative_edges)
    _write_csv(EXCLUSION_PATH, excluded)

    _configure_matplotlib()
    figure_paths: list[Path] = []
    figure_paths.extend(_plot_distributions(features))
    figure_paths.extend(_plot_filtration(features, np.asarray(config.path_thresholds)))
    figure_paths.extend(_plot_effects(tests))
    figure_paths.extend(_plot_calibration(calibration))
    figure_paths.extend(_plot_representative_cycles(representative_edges))
    _write_report(
        manifest=eligible,
        excluded=excluded,
        features=features,
        calibration=calibration,
        tests=tests,
        stability=stability,
        classification=classification,
        representatives=representative_edges,
        config=config,
        figure_paths=figure_paths,
    )

    artifacts = [
        FEATURE_PATH,
        TEST_PATH,
        CALIBRATION_PATH,
        STABILITY_PATH,
        CLASSIFICATION_PATH,
        REPRESENTATIVE_PATH,
        EXCLUSION_PATH,
        REPORT_PATH,
        *figure_paths,
    ]
    primary = tests[tests["role"] == "primary_validation"]
    sensitivity = tests[tests["role"] == "duration_sensitivity"]
    payload = {
        "generated_at": date.today().isoformat(),
        "scope": "phase-lifted Path Homology rerun on current Open Focus/Classical data",
        "evidence_status": "post-migration observational reanalysis; not a replication of the former Focus-vs-Pop hypothesis",
        "ok": True,
        "input_provenance": input_audit,
        "representations": list(PATH_REPRESENTATIONS),
        "segments": int(features["segment_id"].nunique()),
        "segment_views": int(len(features)),
        "tracks": int(features["track_id"].nunique()),
        "quality_excluded_segments": int(len(excluded)),
        "group_counts_segments": {
            str(key): int(value)
            for key, value in eligible.groupby("group").size().to_dict().items()
        },
        "split_counts_segments": {
            str(key): int(value)
            for key, value in eligible.groupby("split").size().to_dict().items()
        },
        "calibration_passes": calibration.loc[
            calibration["calibration_pass"], "representation"
        ].tolist(),
        "primary_validation_fdr_discoveries": primary.loc[
            primary["p_two_sided_fdr_bh"] <= config.validation_fdr_q,
            "representation",
        ].tolist(),
        "duration_sensitivity_fdr_discoveries": sensitivity.loc[
            sensitivity["p_two_sided_fdr_bh"] <= config.validation_fdr_q,
            "representation",
        ].tolist(),
        "config": asdict(config),
        "input_sha256": {
            manifest_path.relative_to(ROOT).as_posix(): _sha256(manifest_path),
            "features/models/state_model.npz": _sha256(
                ROOT / "features" / "models" / "state_model.npz"
            ),
        },
        "outputs": [path.relative_to(ROOT).as_posix() for path in artifacts],
        "output_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in artifacts},
    }
    _write_json_atomic(SUMMARY_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
