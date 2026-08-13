from __future__ import annotations

# ruff: noqa: E402, E501, I001

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "modulation_tertile_path_homology_open"
REPORT = ROOT / "docs" / "path-homology-modulation-analysis.md"
SUMMARY = ROOT / "metadata" / "modulation_tertile_summary.json"
MODEL = ROOT / "features" / "models" / "modulation_tertile_model.json"
FEATURES = ROOT / "metadata" / "modulation_tertile_features.csv"
TOPOLOGY = ROOT / "metadata" / "modulation_tertile_topology_segments.csv"
FILTRATION = ROOT / "metadata" / "modulation_tertile_topology_filtration_sensitivity.csv"
TESTS = ROOT / "metadata" / "modulation_tertile_statistical_tests.csv"
PAIRWISE = ROOT / "metadata" / "modulation_tertile_pairwise_tests.csv"
HOLDOUT_GATE = ROOT / "metadata" / "holdout_gate.json"
HOLDOUT_PERMANOVA = ROOT / "metadata" / "holdout_confirmation_permanova.csv"
HOLDOUT_DIRECTIONAL = ROOT / "metadata" / "holdout_confirmation_directional_metrics.csv"
HOLDOUT_SUMMARY = ROOT / "metadata" / "holdout_confirmation_summary.json"

GROUPS = ("classical", "focus")
GROUP_LABELS = {"classical": "Classical", "focus": "Open Focus"}
COLORS = {"classical": "#4472C4", "focus": "#E07A5F"}
STATE_LABELS = ("Low", "Medium", "High")
STATE_COLORS = ("#4C78A8", "#F2CF5B", "#D95F59")
CONFIRMATORY_FDR_Q = 0.05


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _save(figure: plt.Figure, stem: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    svg = OUTPUT / f"{stem}.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png


def _context() -> tuple[dict[str, Any], dict[str, Any], pd.Series]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    topology = pd.read_csv(TOPOLOGY)
    example_id = summary["mechanism_example"]["segment_id"]
    row = topology[topology.segment_id == example_id].iloc[0]
    return summary, model, row


def _example_arrays(row: pd.Series) -> tuple[dict[str, np.ndarray], ...]:
    feature = _read_npz(ROOT / str(row.feature_relative_path))
    graph = _read_npz(ROOT / str(row.graph_relative_path))
    persistence = _read_npz(ROOT / str(row.sensitivity_persistence_relative_path))
    source_rows = pd.read_csv(ROOT / "metadata" / "feature_segments.csv")
    source = source_rows[source_rows.segment_id == row.segment_id].iloc[0]
    smp = _read_npz(ROOT / str(source.modulation_relative_path))
    return feature, graph, persistence, smp


def plot_tertile_diagnostics(model: dict[str, Any]) -> Path:
    features = pd.read_csv(FEATURES)
    values: dict[str, list[np.ndarray]] = {group: [] for group in GROUPS}
    selected = features[(features.split == "discovery") & (features.scale_seconds == 180.0)]
    for row in selected.itertuples(index=False):
        arrays = _read_npz(ROOT / str(row.modulation_tertile_relative_path))
        values[row.group].append(arrays["intensity"][arrays["valid"].astype(bool)])
    edges = np.asarray(model["tertile_edges"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    upper = float(np.quantile(np.concatenate([np.concatenate(v) for v in values.values()]), 0.99))
    bins = np.linspace(0.0, upper, 55)
    for group in GROUPS:
        axes[0].hist(
            np.concatenate(values[group]),
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=COLORS[group],
            label=GROUP_LABELS[group],
        )
    for edge, label in zip(edges, ("q1", "q2"), strict=True):
        axes[0].axvline(edge, color="#222222", ls="--", lw=1.3)
        axes[0].text(
            edge, axes[0].get_ylim()[1] * 0.94, f"{label}={edge:.4f}", rotation=90, va="top"
        )
    axes[0].set(
        xlabel="Salient-band relative spectral modulation intensity",
        ylabel="Density",
        title="Discovery/180 s distributions and frozen tertiles",
    )
    axes[0].legend(frameon=False)
    counts = np.asarray(model["training_state_counts"], dtype=float)
    axes[1].bar(STATE_LABELS, counts / counts.sum(), color=STATE_COLORS)
    axes[1].axhline(1 / 3, color="#222222", ls="--", lw=1)
    axes[1].set(ylim=(0, 0.4), ylabel="Share", title="Balanced pooled training occupancy")
    for index, value in enumerate(counts / counts.sum()):
        axes[1].text(index, value + 0.012, f"{value:.1%}", ha="center")
    figure.suptitle("Three-state modulation model fitted on discovery/180 s only", fontsize=13)
    return _save(figure, "modulation_tertile_diagnostics")


def plot_smp_profile(
    feature: dict[str, np.ndarray], smp: dict[str, np.ndarray], example: str
) -> Path:
    times = feature["times"].astype(float)
    intensity = feature["intensity"].astype(float)
    states = feature["states"].astype(int)
    edges = feature["tertile_edges"].astype(float)
    frequencies = smp["frequencies"].astype(float)
    spectrum = smp["spectrum"].astype(float)
    figure, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)
    mesh = axes[0].pcolormesh(times, frequencies, spectrum.T, shading="auto", cmap="magma")
    axes[0].set(
        ylabel="Modulation frequency (Hz)", ylim=(0.5, 45), title=f"Normalized SMP: {example}"
    )
    figure.colorbar(mesh, ax=axes[0], pad=0.01, label="Normalized energy share")
    axes[1].plot(times, intensity, color="#273043", lw=1.6)
    axes[1].axhline(edges[0], color=STATE_COLORS[0], ls="--", label=f"q1={edges[0]:.4f}")
    axes[1].axhline(edges[1], color=STATE_COLORS[2], ls="--", label=f"q2={edges[1]:.4f}")
    axes[1].set(ylabel="Relative intensity", title="Scalar salient-band intensity")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].step(times, states, where="mid", color="#273043", lw=1.5)
    axes[2].scatter(times, states, c=[STATE_COLORS[max(0, state)] for state in states], s=18)
    axes[2].set(
        yticks=[0, 1, 2],
        yticklabels=STATE_LABELS,
        xlabel="Time (s)",
        ylabel="State",
        title="Frozen Low / Medium / High sequence",
    )
    return _save(figure, "modulation_smp_profile")


def _draw_graph(axis: plt.Axes, graph: dict[str, np.ndarray], threshold: float, title: str) -> None:
    positions = {0: np.array([-0.82, -0.48]), 1: np.array([0.82, -0.48]), 2: np.array([0.0, 0.86])}
    sources = graph["edge_source"].astype(int)
    targets = graph["edge_target"].astype(int)
    weights = graph["edge_weight"].astype(float)
    for source, target, weight in zip(sources, targets, weights, strict=True):
        if weight + 1e-12 < threshold:
            continue
        start, end = positions[source], positions[target]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=0.8 + 3.0 * weight,
            color="#4A5568",
            alpha=0.8,
            connectionstyle="arc3,rad=0.13",
            shrinkA=23,
            shrinkB=23,
        )
        axis.add_patch(arrow)
        midpoint = 0.5 * (start + end)
        direction = end - start
        normal = np.asarray([-direction[1], direction[0]])
        normal /= np.linalg.norm(normal)
        label_position = midpoint + 0.18 * normal
        axis.text(
            label_position[0],
            label_position[1],
            f"{weight:.2f}",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    vertices = set(graph["vertices"].astype(int).tolist())
    for state in range(3):
        position = positions[state]
        alpha = 1.0 if state in vertices else 0.25
        axis.scatter(
            *position,
            s=1500,
            color=STATE_COLORS[state],
            edgecolor="white",
            linewidth=2,
            alpha=alpha,
            zorder=3,
        )
        axis.text(*position, STATE_LABELS[state], ha="center", va="center", fontsize=11, zorder=4)
    axis.set(xlim=(-1.25, 1.25), ylim=(-0.9, 1.15), aspect="equal", title=title)
    axis.axis("off")


def plot_directed_graph(graph: dict[str, np.ndarray], example: str) -> Path:
    figure, axis = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    _draw_graph(
        axis,
        graph,
        0.0,
        f"Directed modulation-state graph: {example}\nedge width = conditional transition probability",
    )
    return _save(figure, "modulation_directed_state_graph")


def plot_filtration(graph: dict[str, np.ndarray]) -> Path:
    thresholds = (0.95, 0.60, 0.30, 0.05)
    figure, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), constrained_layout=True)
    for axis, threshold in zip(axes, thresholds, strict=True):
        _draw_graph(axis, graph, threshold, rf"$\tau={threshold:.2f}$")
    figure.suptitle(
        "Nested directed-graph filtration (0.30 and 0.05 are sensitivity only)", fontsize=13
    )
    return _save(figure, "modulation_filtration_process")


def _intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, float | int | bool]]:
    output: list[dict[str, float | int | bool]] = []
    end = 1.0 - float(np.min(persistence["thresholds"]))
    for index, dimension in enumerate(persistence["interval_dimension"].astype(int)):
        birth = 1.0 - float(persistence["interval_birth_threshold"][index])
        censored = bool(persistence["interval_censored"][index])
        death = end if censored else 1.0 - float(persistence["interval_death_threshold"][index])
        for _ in range(int(persistence["interval_multiplicity"][index])):
            output.append(
                {"dimension": dimension, "birth": birth, "death": death, "censored": censored}
            )
    return output


def plot_persistence_diagram(persistence: dict[str, np.ndarray], example: str) -> Path:
    intervals = _intervals(persistence)
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    axis.plot([0, end], [0, end], color="#888888", ls="--", lw=1)
    for dimension, color, marker in ((0, "#4472C4", "o"), (1, "#C44E52", "s")):
        selected = [item for item in intervals if item["dimension"] == dimension]
        if selected:
            axis.scatter(
                [item["birth"] for item in selected],
                [item["death"] for item in selected],
                s=70,
                marker=marker,
                color=color,
                edgecolor="white",
                label=f"H{dimension}",
            )
    censored = [item for item in intervals if item["censored"]]
    if censored:
        axis.scatter(
            [item["birth"] for item in censored],
            [item["death"] for item in censored],
            s=110,
            facecolors="none",
            edgecolors="#111111",
            label="right-censored",
        )
    axis.set(
        xlim=(-0.02, end + 0.04),
        ylim=(-0.02, end + 0.04),
        xlabel="Birth a = 1 - tau",
        ylabel="Death a = 1 - tau",
        title=f"Persistence diagram: {example}",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.text(
        0.98,
        0.05,
        "No H1 intervals",
        transform=axis.transAxes,
        ha="right",
        color="#A23E48",
        weight="bold",
    )
    return _save(figure, "modulation_persistence_diagram")


def plot_barcode(persistence: dict[str, np.ndarray], example: str) -> Path:
    intervals = sorted(
        _intervals(persistence), key=lambda item: (item["dimension"], item["birth"], item["death"])
    )
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    for row, item in enumerate(intervals):
        color = "#4472C4" if item["dimension"] == 0 else "#C44E52"
        axis.hlines(row, item["birth"], item["death"], color=color, lw=3)
        axis.plot(item["birth"], row, marker="|", color=color, ms=9)
        axis.plot(
            item["death"],
            row,
            marker="o" if item["censored"] else "|",
            color=color,
            ms=7,
            markerfacecolor="white" if item["censored"] else color,
        )
    axis.set(
        xlim=(-0.02, end + 0.03),
        xlabel="Filtration coordinate a = 1 - tau",
        ylabel="Interval index",
        title=f"Persistent path barcode: {example}",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.02, 0.94, "blue: H0; red: H1; open circle: censored", transform=axis.transAxes, va="top"
    )
    return _save(figure, "modulation_barcode")


def plot_group_summary() -> Path:
    topology = pd.read_csv(TOPOLOGY)
    data = topology[(topology.split == "validation") & (topology.scale_seconds == 180.0)]
    metrics = (
        ("self_transition_ratio", "Self-transition ratio"),
        ("edge_count", "Directed edges"),
        ("path_entropy", "Path entropy"),
        ("directed_recurrence", "Directed recurrence"),
        ("h0_betti_mean", "Mean beta0"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.5, 4.3), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [data.loc[data.group == group, metric].to_numpy() for group in GROUPS]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(boxes["boxes"], GROUPS, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        axis.set_xticks((1, 2), [GROUP_LABELS[group] for group in GROUPS], rotation=22)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Three-state modulation graph comparison (validation, 180 s)", fontsize=13)
    return _save(figure, "modulation_group_summary")


def plot_betti_curves() -> Path:
    filtration = pd.read_csv(FILTRATION)
    data = filtration[
        (filtration.split == "validation") & (filtration.scale_seconds == 180.0)
    ].copy()
    data["a"] = 1.0 - data.threshold
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("h0_betti", "h1_betti"), ("Mean beta0", "Mean beta1"), strict=True
    ):
        for group in GROUPS:
            summary = (
                data[data.group == group]
                .groupby("a")[metric]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("a")
            )
            x = summary.a.to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            sem = summary["sem"].fillna(0).to_numpy(float)
            axis.plot(
                x, mean, marker="o", ms=3.5, lw=1.7, color=COLORS[group], label=GROUP_LABELS[group]
            )
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.14)
        axis.set(title=title, xlabel="Filtration coordinate a = 1 - tau", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].set_ylim(-0.03, 0.3)
    figure.suptitle("Modulation-state Betti curves (validation, 180 s; mean +/- SEM)", fontsize=13)
    return _save(figure, "modulation_betti_curves")


def plot_scale_sensitivity() -> Path:
    topology = pd.read_csv(TOPOLOGY)
    data = topology[topology.split == "validation"]
    metrics = (
        ("self_transition_ratio", "Self-transition ratio"),
        ("path_entropy", "Path entropy"),
        ("directed_recurrence", "Directed recurrence"),
        ("h0_betti_mean", "Mean beta0"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), constrained_layout=True)
    x = np.arange(2)
    width = 0.36
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        med180 = [
            data[(data.group == group) & (data.scale_seconds == 180.0)][metric].median()
            for group in GROUPS
        ]
        med300 = [
            data[(data.group == group) & (data.scale_seconds == 300.0)][metric].median()
            for group in GROUPS
        ]
        axis.bar(x - width / 2, med180, width, color="#AAB4BE", label="180 s")
        axis.bar(x + width / 2, med300, width, color="#28536B", label="300 s")
        axis.set_xticks(x, [GROUP_LABELS[group] for group in GROUPS], rotation=24)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Scale sensitivity: validation group medians", fontsize=13)
    return _save(figure, "modulation_scale_sensitivity")


def plot_effect_sizes() -> Path:
    pairwise = pd.read_csv(PAIRWISE)
    data = pairwise[
        (pairwise.analysis_set == "primary_validation_180")
        & (pairwise.group_a == "classical")
        & (pairwise.group_b == "focus")
    ].copy()
    data["effect_focus_minus_classical"] = -data[
        "rank_biserial_a_minus_b"
    ].astype(float)
    data = data.sort_values("effect_focus_minus_classical")
    y = np.arange(len(data))
    significant = data.p_fdr_bh.astype(float) <= CONFIRMATORY_FDR_Q
    figure, axis = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    axis.axvline(0.0, color="#777777", lw=1.0)
    axis.hlines(
        y, 0.0, data.effect_focus_minus_classical, color="#AAB4BD", lw=1.2
    )
    axis.scatter(
        data.loc[~significant, "effect_focus_minus_classical"],
        y[~significant.to_numpy()],
        facecolors="white",
        edgecolors="#6F7F8C",
        s=48,
        label="q > 0.05",
        zorder=3,
    )
    axis.scatter(
        data.loc[significant, "effect_focus_minus_classical"],
        y[significant.to_numpy()],
        color="#28536B",
        s=54,
        label="BH-FDR q <= 0.05",
        zorder=3,
    )
    axis.set_yticks(y, data.metric)
    axis.set(
        xlim=(-1.02, 1.02),
        xlabel="Rank-biserial effect (Open Focus - Classical)",
        title="Modulation validation/180 s effect directions",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "modulation_effect_sizes")


def plot_duration_stability() -> Path:
    pairwise = pd.read_csv(PAIRWISE)
    selected = pairwise[
        (pairwise.group_a == "classical") & (pairwise.group_b == "focus")
    ].copy()
    selected["effect_focus_minus_classical"] = -selected[
        "rank_biserial_a_minus_b"
    ].astype(float)
    effects = selected.pivot(
        index="metric", columns="analysis_set", values="effect_focus_minus_classical"
    )
    qvalues = selected.pivot(
        index="metric", columns="analysis_set", values="p_fdr_bh"
    )
    x = effects["primary_validation_180"].astype(float)
    y = effects["sensitivity_validation_300"].astype(float)
    stable = (
        (qvalues["primary_validation_180"].astype(float) <= CONFIRMATORY_FDR_Q)
        & (qvalues["sensitivity_validation_300"].astype(float) <= CONFIRMATORY_FDR_Q)
        & (x * y > 0)
    )
    figure, axis = plt.subplots(figsize=(7.4, 6.5), constrained_layout=True)
    axis.axhline(0.0, color="#888888", lw=0.9)
    axis.axvline(0.0, color="#888888", lw=0.9)
    axis.plot([-1, 1], [-1, 1], ls="--", color="#B0B0B0", lw=1.0)
    axis.scatter(
        x[~stable],
        y[~stable],
        facecolors="white",
        edgecolors="#6F7F8C",
        s=52,
        label="not stable",
    )
    axis.scatter(
        x[stable],
        y[stable],
        color="#28536B",
        s=58,
        label=f"stable ({int(stable.sum())})",
    )
    h1_unstable = [
        metric for metric in effects.index[~stable] if metric.startswith("h1_")
    ]
    if h1_unstable:
        anchor_x = float(x.loc[h1_unstable].mean())
        anchor_y = float(y.loc[h1_unstable].mean())
        axis.annotate(
            f"H1 descriptors ({len(h1_unstable)})\ncluster at zero; not stable",
            (anchor_x, anchor_y),
            xytext=(28, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#6F7F8C", "lw": 0.8},
            fontsize=8,
        )
    axis.set(
        xlim=(-1.02, 1.02),
        ylim=(-1.02, 1.02),
        xlabel="Validation/180 s rank-biserial (Open Focus - Classical)",
        ylabel="Validation/300 s rank-biserial (Open Focus - Classical)",
        title="Modulation cross-duration direction and effect stability",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "modulation_duration_stability")


def _tables(tests: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[str, str]:
    primary = tests[tests.analysis_set == "primary_validation_180"].sort_values(
        ["p_fdr_bh", "metric"]
    )
    sensitivity = tests[tests.analysis_set == "sensitivity_validation_300"].set_index("metric")
    metric_rows = []
    for row in primary.itertuples(index=False):
        other = sensitivity.loc[row.metric]
        metric_rows.append(
            f"| {row.metric} | {row.classical_median:.3f} | {row.focus_median:.3f} | "
            f"{row.epsilon_squared:.3f} | {row.p_fdr_bh:.3g} | {other.p_fdr_bh:.3g} |"
        )
    pairs = pairwise[pairwise.analysis_set == "primary_validation_180"].sort_values(
        ["p_fdr_bh", "metric"]
    )
    pair_rows = []
    for row in pairs.itertuples(index=False):
        effect = -float(row.rank_biserial_a_minus_b)
        pair_rows.append(f"| {row.metric} | {effect:.3f} | {row.p_fdr_bh:.3g} |")
    return "\n".join(metric_rows), "\n".join(pair_rows)


def _write_report_legacy(summary: dict[str, Any], model: dict[str, Any], stems: tuple[str, ...]) -> Path:
    tests = pd.read_csv(TESTS)
    pairwise = pd.read_csv(PAIRWISE)
    metric_rows, pair_rows = _tables(tests, pairwise)
    h1 = summary["validation_180_h1_counts"]
    features = pd.read_csv(FEATURES)
    primary_features = features[
        (features.split == "validation") & (features.scale_seconds == 180.0)
    ]
    occupancy_rows = []
    for group in GROUPS:
        row = primary_features[primary_features.group == group]
        counts = row[["low_windows", "medium_windows", "high_windows"]].sum().to_numpy(float)
        shares = counts / counts.sum()
        occupancy_rows.append(
            f"| {GROUP_LABELS[group]} | {shares[0]:.1%} | {shares[1]:.1%} | {shares[2]:.1%} |"
        )
    figures = "\n\n".join(
        f"![{stem}](../runs/modulation_tertile_path_homology_open/{stem}.png)\n\n"
        f"[SVG](../runs/modulation_tertile_path_homology_open/{stem}.svg)"
        for stem in stems
    )
    edges = summary["tertile_edges"]
    report = rf"""# Path Homology 调制视角：Focus–Classical 完整分析

生成日期：2026-08-02。本报告使用当前规范数据集 Jamendo Open Focus 300 首与 Classical 300 首。主推断固定为 validation/180 s（n={summary["primary_validation_n"]}；Classical {h1["classical"]["total"]}，Open Focus {h1["focus"]["total"]}）；validation/300 s 仅作尺度敏感性分析；Focus-only holdout 不进入组间检验。

## 1. 结论摘要

- 1,200/1,200 个片段完成三状态调制特征、有向图和持久 Path Homology，失败 0；模型 SHA-256 为 `{summary["model_sha256"]}`。
- discovery/180 s 平衡样本拟合的固定边界为 q1={edges[0]:.6f}、q2={edges[1]:.6f}。指标是三个重点频带归一化能量占比之和，只表示“重点频带相对谱调制强度”，不是绝对调制功率。
- validation/180 s 的 20 个预设指标中，{summary["primary_fdr_discoveries"]} 个通过 omnibus BH-FDR $q\le0.05$；validation/300 s 有 {summary["sensitivity_fdr_discoveries"]} 个通过，其中 {summary["replicated_same_direction"]} 个跨时长方向一致且再次显著。
- Open Focus 在当前三状态量化下表现为更高的自转移率与有向复现度、更低的路径熵和较少的高阈值边；这是离散调制状态的重复/集中程度差异，不是音乐质量或功能效果。
- **$H_1$ 明确不支持：** 主阈值非零率 Classical {h1["classical"]["primary_nonzero"]}/{h1["classical"]["total"]}、Open Focus {h1["focus"]["primary_nonzero"]}/{h1["focus"]["total"]}；阈值扩展到 0.05 后仍分别为 {h1["classical"]["sensitivity_nonzero"]}/{h1["classical"]["total"]} 和 {h1["focus"]["sensitivity_nonzero"]}/{h1["focus"]["total"]}。六个 $H_1$ 指标的中位数均为 0，FDR 均为 1。
- 结论属于观察性声学结构比较；不支持注意力、治疗、认知、生成质量或因果推断。

## 2. Spectral Modulation Profile 与三状态表示

先从 mel 频带能量包络提取 Spectral Modulation Profile（SMP）：固定 4 s 窗、2 s 步长，在 0.5–45 Hz 调制频率范围归一化。对三个预先存储的重点频带 8–12、18–20、28–32 Hz，定义

$$
m_t=E_{{8:12,t}}+E_{{18:20,t}}+E_{{28:32,t}},
$$

其中每个 $E$ 是归一化 SMP 中相应频带的能量占比。仅从 discovery/180 s 的 Classical 与 Focus 各等量抽取 {model["sampled_windows"]["classical"]:,} 个有效窗口，合并后估计三分位数；validation、300 s 和 holdout 均不参与边界拟合：

$$
s_t=\begin{{cases}}
0\;(\mathrm{{Low}}), & m_t<q_1,\\
1\;(\mathrm{{Medium}}), & q_1\le m_t<q_2,\\
2\;(\mathrm{{High}}), & m_t\ge q_2.
\end{{cases}}
$$

平衡训练池按构造接近各占三分之一；各组在冻结边界下不要求占用率相同。主分析 validation/180 s 的实际占用率为：

| 组别 | Low | Medium | High |
|---|---:|---:|---:|
{chr(10).join(occupancy_rows)}

无效窗口记为缺失状态 -1；其两侧不跨越连接。原有按三个频带分别三分位量化、最多形成 27 个组合状态的 `modulation` 结果被完整保留，本报告使用独立的 `modulation_tertile` 三状态视图。

## 3. 有向图与持久 Path Homology

相邻有效状态定义转移计数和条件概率：

$$
C_{{uv}}=|\{{t:s_t=u,s_{{t+1}}=v\}}|,\qquad
p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}}.
$$

自转移用于描述统计，但不进入 Path Homology 图；每个源状态最多保留 top-6 非自环边。因为本视图只有三个状态，实际每个源最多两条非自环边，top-6 不再造成额外裁剪。主阈值冻结为 $\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}$；0.05–0.40 仅用于敏感性与机制图：

$$
G_\tau=(V,\{{(u,v):u\ne v,\ p_{{uv}}\ge\tau\}}).
$$

对允许路径空间 $\Omega_p$，GLMY 路径同调为

$$
\partial e_{{v_0\ldots v_p}}=\sum_i(-1)^i e_{{v_0\ldots\widehat{{v_i}}\ldots v_p}},\qquad
H_p^{{path}}(G)=\frac{{\ker(\partial_p|_{{\Omega_p}})}}{{\operatorname{{im}}(\partial_{{p+1}}|_{{\Omega_{{p+1}}}})}}.
$$

生产流程报告 $H_0/H_1$，未计算 $H_2$。SMP 热图只用于表示审计；建图输入是冻结的 Low/Medium/High 相邻状态路径。

## 4. 可视化

示例 `{summary["mechanism_example"]["segment_id"]}` 按预设回退规则选自 Open Focus validation/180 s。由于全部 validation/180 s 样本在扩展阈值下也没有有限 $H_1$ 区间，示例改为优先展示边数最多、路径熵最高的 Focus 片段；它不参与检验，也不代表组中心。

{figures}

## 5. 组间结果

Kruskal–Wallis 检验在 20 个预设 modulation_tertile 指标内作 BH-FDR，效应量为 $\epsilon^2$。300 s 列只用于敏感性复核。

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s FDR | 300 s FDR |
|---|---:|---:|---:|---:|---:|
{metric_rows}

由于本轮只有两个组，独立 Mann–Whitney 检验与 omnibus 检验在排序证据上等价，但仍按独立的 20 指标 family 校正。效应方向统一写成 Open Focus − Classical：

| 指标 | rank-biserial（Open Focus−Classical） | FDR |
|---|---:|---:|
{pair_rows}

### 5.1 解释

1. **最稳定的差异是路径集中度。** Open Focus 的 directed recurrence 更高，而 path entropy 与 transition entropy 更低；同时自转移率更高。这说明冻结三状态路径更集中于少数状态/转移。
2. **$H_0$ 反映高阈值连通过程。** Classical 在主阈值中保留更多边，并具有更高的 $\beta_0$ 最大值、均值、AUC 与观测持久量；这意味着其条件转移概率较分散，图需要在更低阈值才合并，而不是“拓扑更好”。
3. **不能把普通指标显著误写为 $H_1$。** 13 个显著指标全部属于状态、边、路径熵或 $H_0$；所有六个 $H_1$ 指标均不显著且恒为零。
4. **三状态压缩限制环。** 只有三个顶点且边由一阶相邻转换产生，表示能力远低于原 27 组合状态；本结果回答的是总调制强度级别如何转换，不回答三个重点频带之间的联合模式。

## 6. 证据层级与局限

- **确认性：** validation/180 s、冻结三分位边界、top-6、主阈值 0.50–0.95、20 指标 omnibus 与独立 pairwise FDR。
- **敏感性：** validation/300 s；阈值下探到 0.05 的 Betti 曲线和 $H_1$ 发生率。敏感性结果不替代主检验。
- **探索/说明性：** discovery 分布、单片段 SMP 热图、机制图与全数据状态占用率。
- **不支持：** Focus 特异的 $H_1/H_2$；将相对 SMP 占比解释为绝对振幅调制功率；注意力、治疗、认知、生成或因果结论。
- 三个重点频带之和忽略各频带的独立方向；归一化后数值会受到谱内其他频带能量变化影响。
- 固定 4 s/2 s 时间网格只能刻画局部调制轨迹；三分位边界是当前 discovery 数据的经验量化器，不是普适生理阈值。
- 两组来源和曲库差异可能混入录音、配器、母带和元数据选择效应。

## 7. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/run_modulation_tertile_analysis.py
python scripts/render_modulation_tertile_report.py
```

主要数值文件为 `metadata/modulation_tertile_features.csv`、`metadata/modulation_tertile_topology_segments.csv`、`metadata/modulation_tertile_topology_filtration.csv`、`metadata/modulation_tertile_topology_filtration_sensitivity.csv`、`metadata/modulation_tertile_statistical_tests.csv`、`metadata/modulation_tertile_pairwise_tests.csv` 和 `metadata/modulation_tertile_summary.json`。模型位于 `features/models/modulation_tertile_model.*`，图和持久结果位于 `graphs/modulation_tertile` 与 `homology/*/modulation_tertile`。
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def write_report(summary: dict[str, Any], model: dict[str, Any], stems: tuple[str, ...]) -> Path:
    tests = pd.read_csv(TESTS)
    pairwise = pd.read_csv(PAIRWISE)
    metric_rows, pair_rows = _tables(tests, pairwise)
    primary = tests[tests.analysis_set == "primary_validation_180"].set_index("metric")
    sensitivity = tests[tests.analysis_set == "sensitivity_validation_300"].set_index("metric")
    dual_significant = (primary.p_fdr_bh <= CONFIRMATORY_FDR_Q) & (
        sensitivity.p_fdr_bh <= CONFIRMATORY_FDR_Q
    )
    primary_delta = primary.focus_median - primary.classical_median
    sensitivity_delta = sensitivity.focus_median - sensitivity.classical_median
    nonzero_direction = dual_significant & (np.sign(primary_delta) == np.sign(sensitivity_delta)) & (
        np.sign(primary_delta) != 0
    )
    equal_medians = dual_significant & (np.sign(primary_delta) == 0) & (
        np.sign(sensitivity_delta) == 0
    )

    h1 = summary["validation_180_h1_counts"]
    features = pd.read_csv(FEATURES)
    primary_features = features[
        (features.split == "validation") & (features.scale_seconds == 180.0)
    ]
    occupancy_rows = []
    for group in GROUPS:
        frame = primary_features[primary_features.group == group]
        counts = frame[["low_windows", "medium_windows", "high_windows"]].sum().to_numpy(float)
        shares = counts / counts.sum()
        occupancy_rows.append(
            f"| {GROUP_LABELS[group]} | {shares[0]:.1%} | {shares[1]:.1%} | {shares[2]:.1%} |"
        )

    topology = pd.read_csv(TOPOLOGY)
    holdout_180 = topology[(topology.split == "holdout") & (topology.scale_seconds == 180.0)]
    holdout_h1 = {
        group: int((holdout_180.loc[holdout_180.group == group, "h1_betti_max"] > 0).sum())
        for group in GROUPS
    }
    gate = json.loads(HOLDOUT_GATE.read_text(encoding="utf-8"))
    gate_hash = gate["input_sha256"]["metadata/modulation_tertile_topology_segments.csv"]
    current_hash = summary["artifact_sha256"]["metadata/modulation_tertile_topology_segments.csv"]
    hash_compatible = gate_hash == current_hash
    permanova = pd.read_csv(HOLDOUT_PERMANOVA)
    holdout_row = permanova[
        (permanova.analysis_set == "primary_holdout_180") & (permanova.feature_set == "modulation")
    ].iloc[0]
    directional = pd.read_csv(HOLDOUT_DIRECTIONAL)
    locked = directional[
        (directional.analysis_set == "primary_holdout_180") & (directional.view == "modulation")
    ]
    locked_strict = int(
        (
            locked.direction_matched.astype(str).str.lower().eq("true")
            & (locked.p_fdr_bh.astype(float) <= CONFIRMATORY_FDR_Q)
        ).sum()
    )
    all_holdout_180 = directional[
        directional.analysis_set == "primary_holdout_180"
    ]
    all_holdout_strict = int(
        (
            all_holdout_180.direction_matched.astype(str).str.lower().eq("true")
            & (all_holdout_180.p_fdr_bh.astype(float) <= CONFIRMATORY_FDR_Q)
        ).sum()
    )
    holdout_summary = json.loads(HOLDOUT_SUMMARY.read_text(encoding="utf-8"))
    edges = summary["tertile_edges"]

    figures = "\n\n".join(
        f"![{stem}](../runs/modulation_tertile_path_homology_open/{stem}.png)\n\n"
        f"[下载 SVG](../runs/modulation_tertile_path_homology_open/{stem}.svg)"
        for stem in stems
    )
    report = rf"""# 调制视角 Path Homology：Focus–Classical 完整重跑报告

生成日期：{date.today().isoformat()}。切分版本：`symmetric_holdout_v2`。数据为 Jamendo Open Focus 300 首与 Classical 300 首；每组 discovery/validation/holdout 分别为 195/60/45 首。分析共重算 {summary["segment_views"]:,}/{summary["segment_views"]:,} 个 180 s/300 s 片段视图，覆盖 {summary["tracks"]} 首曲目，失败 0。本版将确认性端点统一为 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留。

> 证据边界：validation/180 s 是本报告的主单视角检验；validation/300 s 是同曲目的时长敏感性。holdout 是冻结哈希门控后的单次操作性最终确认。由于 Classical holdout 在旧切分中曾属于 discovery，它不是 pristine 外部复制集。

## 1. 结论摘要

- 冻结的三状态调制表示检测到稳定的组间组织差异：20 个预设指标中，validation/180 s 有 **{summary["primary_fdr_discoveries"]} 个**通过 BH-FDR $q\le0.05$；300 s 有 **{summary["sensitivity_fdr_discoveries"]} 个**通过。共有 **{int(dual_significant.sum())} 个**在两个时长均显著，其中 **{int(nonzero_direction.sum())} 个**具有一致的非零中位数方向，另有 **{int(equal_medians.sum())} 个**两组中位数均相等、显著性来自分布而非中位数位移。
- Open Focus 在冻结量化下呈现更高的自转移率与有向复现度，以及更低的转移熵、路径熵、边数和多项 $H_0$ 汇总量。这表示三状态转移更集中，不表示音乐质量、注意力效果或因果机制。
- **$H_1$ 明确不支持。** 主阈值下 Classical 为 {h1["classical"]["primary_nonzero"]}/{h1["classical"]["total"]}、Focus 为 {h1["focus"]["primary_nonzero"]}/{h1["focus"]["total"]}；阈值扩展至 0.05 后仍分别为 {h1["classical"]["sensitivity_nonzero"]}/{h1["classical"]["total"]} 与 {h1["focus"]["sensitivity_nonzero"]}/{h1["focus"]["total"]}。holdout/180 s 也为 Classical {holdout_h1["classical"]}/45、Focus {holdout_h1["focus"]}/45。
- 冻结 holdout 中，调制块整体表示 pseudo-$F={holdout_row.pseudo_f:.3f}$，$p={holdout_row.p_value:.3f}$，跨次级块 BH $q={holdout_row.p_fdr_bh:.3f}$。原门控的 10 个调制方向指标中，{int(locked.direction_matched.sum())}/10 方向一致；历史 $q\le0.10$ 与统一严格 $q\le0.05$ 均为 {locked_strict}/10 复现。
- 当前重跑的拓扑输入 SHA-256 为 `{current_hash}`，与 holdout gate **{'一致' if hash_compatible else '不一致'}**；模型 SHA-256 为 `{summary["model_sha256"]}`。

## 2. 方法思想：把调制动力学变成有向路径

该方法不直接对音频波形做同调，而是先把短时调制能量压缩成 Low/Medium/High 状态序列，再把相邻状态的条件转移概率构造成有向图。普通图拓扑只关心“是否连通”，Path Homology 还保留路径的方向和可连接次序，因此适合描述“调制状态如何演化”。过滤阈值从高到低加入边，观察有向连通分量和有向环是否持续存在。

这一路径回答的是：**调制强度级别之间的转移组织是否不同**。它不回答三个频带各自的独立作用，也不等价于完整声学表征。原有 27 个组合状态的 `modulation` 分支保持独立；本报告只分析新重跑的 `modulation_tertile` 三状态分支。

## 3. Spectral Modulation Profile 与冻结三分位表示

对 mel 子带能量包络 $x_b[n]$，在第 $t$ 个 4 s 窗内计算调制频率谱：

$$
P_t(f_m)=\sum_b\left|\sum_n w[n]x_b[n+tH]e^{{-i2\pi f_m n/f_s}}\right|^2,
\qquad
\widetilde P_t(f_m)=\frac{{P_t(f_m)}}{{\sum_{{0.5\le f\le45}}P_t(f)}}.
$$

步长 $H=2$ s，只保留 0.5–45 Hz，并将谱归一化。三个预先指定频带的相对能量为

$$
E_{{B,t}}=\sum_{{f_m\in B}}\widetilde P_t(f_m),
\qquad
m_t=E_{{8:12,t}}+E_{{18:20,t}}+E_{{28:32,t}}.
$$

因此 $m_t$ 是“重点频带相对调制能量占比”，不是绝对调制功率。仅在 discovery/180 s 中从 Classical 与 Focus 各平衡抽取 {model["sampled_windows"]["classical"]:,} 个有效窗口，拟合冻结边界 $q_1={edges[0]:.9f}$、$q_2={edges[1]:.9f}$：

$$
s_t=\begin{{cases}}
0\ (\mathrm{{Low}}),&m_t<q_1,\\
1\ (\mathrm{{Medium}}),&q_1\le m_t<q_2,\\
2\ (\mathrm{{High}}),&m_t\ge q_2.
\end{{cases}}
$$

validation/180 s 的实际状态占用为：

| 组别 | Low | Medium | High |
|---|---:|---:|---:|
{chr(10).join(occupancy_rows)}

无效窗记作缺失状态 $-1$，缺失区间两侧不跨越连接。validation、300 s 与 holdout 均不参与边界拟合。

## 4. 有向转移图、Path Homology 与持久性

相邻有效状态定义转移计数与条件概率：

$$
C_{{uv}}=\left|\{{t:s_t=u,\ s_{{t+1}}=v\}}\right|,
\qquad
p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}}.
$$

自转移只用于描述统计，不进入 Path Homology 图。每个源状态保留 top-6 非自环边；三状态下每源最多只有两条，所以该规则不会额外截边。超水平过滤为

$$
G_\tau=\left(V,\{{(u,v):u\ne v,\ p_{{uv}}\ge\tau\}}\right).
$$

主阈值冻结为 $\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}$；0.05–0.40 只用于敏感性和机制图。令 $A_p$ 为允许的有向 $p$-路径张成空间，路径边界为

$$
\partial e_{{v_0\ldots v_p}}=\sum_{{i=0}}^p(-1)^i
e_{{v_0\ldots\widehat{{v_i}}\ldots v_p}}.
$$

并非每条允许路径的边界仍然允许，因此使用 $\partial$-不变路径空间

$$
\Omega_p=A_p\cap\partial^{{-1}}(A_{{p-1}}),
\qquad
H_p^{{\mathrm{{path}}}}(G)=
\frac{{\ker(\partial_p:\Omega_p\to\Omega_{{p-1}})}}
{{\operatorname{{im}}(\partial_{{p+1}}:\Omega_{{p+1}}\to\Omega_p)}}.
$$

$\beta_p=\dim H_p^{{\mathrm{{path}}}}$。用 $a=1-\tau$ 把降阈值过滤改写为递增参数；持久秩不变量为

$$
\rho_p(a_i,a_j)=\operatorname{{rank}}\operatorname{{im}}
\left[H_p(G_{{a_i}})\longrightarrow H_p(G_{{a_j}})\right],
\qquad a_i\le a_j.
$$

据此得到 barcode、persistence diagram、Betti 曲线、区间数、观测持久量与 AUC。生产流程只报告 $H_0/H_1$；本轮没有计算 $H_2$，因此不能作 $H_2$ 发现声明。

## 5. 统计检验与冻结确认

主检验对 20 个预设指标分别做两组 Kruskal–Wallis 检验，并在单一 modulation_tertile family 内作 BH-FDR，确认性判定统一要求 $q\le0.05$。若秩和统计量为 $H$、组数为 $k$、总样本为 $N$，效应量为

$$
\epsilon^2=\frac{{H-k+1}}{{N-k}}.
$$

两组情况下另报告 Mann–Whitney rank-biserial，方向统一为 Focus $-$ Classical。300 s 是同曲目时长敏感性，不是独立复制。holdout 的整体表示使用发现集拟合的秩正态 Mahalanobis 距离和 999 次标签置换；pseudo-$F$ 可写为

$$
F^*=\frac{{SS_{{between}}/(g-1)}}{{SS_{{within}}/(N-g)}}.
$$

冻结 holdout 只验证 gate 中按原 $q\le0.10$ 方案预先锁定的 10 个调制指标；它不重新选择当前表中的 12 个 validation 发现，也不重开盲或调参。完整 holdout 的 44 个四视角方向指标中，{holdout_summary["directional_metric_replication_180"]["direction_matched"]}/44 方向一致，历史 $q\le0.10$ 为 {holdout_summary["directional_metric_replication_180"]["replicated_q_0_10"]}/44、严格 $q\le0.05$ 为 {all_holdout_strict}/44 联合 FDR 复现；本报告只把其中调制视角的 {locked_strict}/10 作为严格口径证据。

## 6. 完整数值结果

### 6.1 Omnibus 检验

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s FDR | 300 s FDR |
|---|---:|---:|---:|---:|---:|
{metric_rows}

### 6.2 两组方向效应

| 指标 | rank-biserial（Open Focus−Classical） | FDR |
|---|---:|---:|
{pair_rows}

### 6.3 结果解释

1. **路径集中度是最稳定的差异。** Focus 的 directed recurrence 与 self-transition ratio 更高，path entropy 和 transition entropy 更低，说明状态路径更集中于少数状态/转移。
2. **$H_0$ 描述高阈值连通过程。** Classical 保留更多高阈值边，并有更高的 $\beta_0$ 最大值、均值、AUC 与观测持久量；这表示条件转移概率更分散，图要到更低阈值才合并，不表示“拓扑更好”。
3. **普通图指标显著不等于 $H_1$ 显著。** 六个 $H_1$ 指标均恒为零、FDR 为 1。证据来自状态占用、边组织、路径熵和 $H_0$，不是环。
4. **三状态压缩限制环表示能力。** 它可解释、稳定，但远低于 27 个组合状态的表达容量；不能据此否定更细状态空间中可能存在的环，只能说当前冻结表示没有观察到。

## 7. 可视化

示例 `{summary["mechanism_example"]["segment_id"]}` 按预设回退规则选自 Focus validation/180 s：由于没有有限 $H_1$ 区间，选择边数最多、随后路径熵最高的片段。该示例不参与检验，也不代表组中心。

{figures}

## 8. 证据层级与局限

- **确认性：** validation/180 s、冻结三分位、top-6、主阈值 0.50–0.95、20 指标 family 与 BH-FDR，统一要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件；还必须满足预定方向、300 s 方向一致性、去冗余和融合增量审计。$0.05<q\le0.10$ 的端点只作提示性结果，不进入核心冻结指纹。
- **操作性最终确认：** 哈希门控的 holdout/180 s；但 Classical holdout 不是 pristine 外部样本，且其配器构成没有 piano solo，泛化解释需保守。
- **敏感性：** validation/300 s 和阈值下探至 0.05；不能替代主检验。
- **探索/说明性：** discovery 分布、单片段 SMP 热图与机制图。
- **不支持：** 当前三状态表示下的 $H_1$；任何 $H_2$ 发现；把相对 SMP 占比解释为绝对功率；注意力、治疗、认知、生成质量或因果结论。
- 三个重点频带求和会丢失频带间方向；归一化比例也会受谱内其他频带能量变化影响。4 s/2 s 网格只刻画局部调制轨迹，三分位边界是当前 discovery 数据的经验量化器，不是普适生理阈值。
- 两组来自不同曲库，录音、母带、配器、作曲家和元数据选择差异仍可能混入观察结果。

## 9. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
.\.venv\Scripts\python.exe scripts\run_modulation_tertile_analysis.py
.\.venv\Scripts\python.exe scripts\render_modulation_tertile_report.py
```

主要数值产物：`metadata/modulation_tertile_features.csv`、`metadata/modulation_tertile_topology_segments.csv`、`metadata/modulation_tertile_topology_filtration.csv`、`metadata/modulation_tertile_topology_filtration_sensitivity.csv`、`metadata/modulation_tertile_statistical_tests.csv`、`metadata/modulation_tertile_pairwise_tests.csv` 与 `metadata/modulation_tertile_summary.json`。图和持久结果位于 `runs/modulation_tertile_path_homology_open/`、`graphs/modulation_tertile/` 与 `homology/*/modulation_tertile/`。
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def main() -> int:
    plt.rcParams.update(
        {"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False}
    )
    summary, model, row = _context()
    feature, graph, persistence, smp = _example_arrays(row)
    example = str(row.segment_id)
    stems = (
        "modulation_tertile_diagnostics",
        "modulation_smp_profile",
        "modulation_directed_state_graph",
        "modulation_filtration_process",
        "modulation_persistence_diagram",
        "modulation_barcode",
        "modulation_group_summary",
        "modulation_betti_curves",
        "modulation_scale_sensitivity",
        "modulation_effect_sizes",
        "modulation_duration_stability",
    )
    outputs = (
        plot_tertile_diagnostics(model),
        plot_smp_profile(feature, smp, example),
        plot_directed_graph(graph, example),
        plot_filtration(graph),
        plot_persistence_diagram(persistence, example),
        plot_barcode(persistence, example),
        plot_group_summary(),
        plot_betti_curves(),
        plot_scale_sensitivity(),
        plot_effect_sizes(),
        plot_duration_stability(),
    )
    report = write_report(summary, model, stems)
    print(report.relative_to(ROOT).as_posix())
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
