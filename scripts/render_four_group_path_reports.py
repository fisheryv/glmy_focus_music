from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "four_group_path_homology"
GROUPS = ("classical", "focus", "focus_open", "pop")
GROUP_LABELS = {
    "classical": "Classical",
    "focus": "Focus",
    "focus_open": "Focus Open",
    "pop": "Pop",
}
COLORS = {
    "classical": "#4472C4",
    "focus": "#ED7D31",
    "focus_open": "#8E6BBE",
    "pop": "#70AD47",
}
VIEW_LABELS = {
    "structure": "结构",
    "pitch_v2": "音高（Tonnetz pitch_v2）",
    "rhythm": "节奏",
}
PITCH_LABELS = (
    "C",
    "C#/Db",
    "D",
    "D#/Eb",
    "E",
    "F",
    "F#/Gb",
    "G",
    "G#/Ab",
    "A",
    "A#/Bb",
    "B",
)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _save(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    svg = OUTPUT / f"{stem}.svg"
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, svg


def _example(topology: pd.DataFrame, view: str) -> pd.Series:
    candidates = topology[
        (topology["view"] == view)
        & (topology["group"] == "focus_open")
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"no Focus Open validation example for {view}")
    candidates = candidates.sort_values(
        ["h1_observed_persistence", "h1_betti_max", "edge_count", "segment_id"],
        ascending=[False, False, False, True],
    )
    return candidates.iloc[0]


def _feature_row(features: pd.DataFrame, segment_id: str) -> pd.Series:
    selected = features[features["segment_id"] == segment_id]
    if len(selected) != 1:
        raise RuntimeError(f"feature row lookup failed for {segment_id}")
    return selected.iloc[0]


def _draw_graph(
    axis: plt.Axes,
    graph: dict[str, np.ndarray],
    *,
    threshold: float,
    title: str,
    edge_labels: bool = False,
) -> None:
    vertices = graph["vertices"].astype(int)
    ordered = sorted(vertices)
    if not ordered:
        axis.set_title(title)
        axis.axis("off")
        return
    positions = {
        state: np.asarray(
            [
                np.cos(np.pi / 2 - 2 * np.pi * index / len(ordered)),
                np.sin(np.pi / 2 - 2 * np.pi * index / len(ordered)),
            ]
        )
        for index, state in enumerate(ordered)
    }
    for source, target, weight in zip(
        graph["edge_source"].astype(int),
        graph["edge_target"].astype(int),
        graph["edge_weight"].astype(float),
        strict=True,
    ):
        if weight < threshold or source not in positions or target not in positions:
            continue
        start, end = positions[source], positions[target]
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8 + 5 * weight,
            linewidth=0.7 + 2.4 * weight,
            color="#46647A",
            alpha=0.35 + 0.6 * weight,
            shrinkA=15,
            shrinkB=15,
            connectionstyle="arc3,rad=0.09",
        )
        axis.add_patch(patch)
        if edge_labels:
            midpoint = (start + end) / 2
            axis.text(
                midpoint[0],
                midpoint[1],
                f"{weight:.2f}",
                fontsize=6.5,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
            )
    for state in ordered:
        position = positions[state]
        axis.scatter(
            position[0],
            position[1],
            s=420,
            color=plt.get_cmap("tab20")(state % 20),
            edgecolor="#263B4A",
            zorder=5,
        )
        axis.text(
            position[0], position[1], str(state), ha="center", va="center", fontsize=8, zorder=6
        )
    axis.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal", title=title)
    axis.axis("off")


def plot_structure_input(example: pd.Series, features: pd.DataFrame) -> None:
    row = _feature_row(features, str(example["segment_id"]))
    arrays = _read_npz(ROOT / row["structure_relative_path"])
    ssm = arrays["self_similarity"].astype(float)
    novelty = arrays["novelty"].astype(float)
    boundary_times = arrays["boundary_times"].astype(float)
    times = arrays["times"].astype(float)
    states = arrays["states"].astype(int)
    duration = float(example["scale_seconds"])
    figure = plt.figure(figsize=(9.5, 8.6), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(4.5, 1.2, 1.0))
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(
        ssm,
        origin="lower",
        extent=(0, duration, 0, duration),
        aspect="equal",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    for boundary in boundary_times:
        axis.axvline(boundary, color="white", lw=0.65, alpha=0.75)
        axis.axhline(boundary, color="white", lw=0.65, alpha=0.75)
    axis.set(
        title=f"Structure SSM and boundaries: {example['segment_id']}",
        xlabel="Time (s)",
        ylabel="Time (s)",
    )
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label="Cosine similarity")
    novelty_axis = figure.add_subplot(grid[1])
    novelty_times = np.linspace(0, duration, novelty.size)
    novelty_axis.plot(novelty_times, novelty, color="#28536B", lw=1.1)
    for boundary in boundary_times:
        novelty_axis.axvline(boundary, color="#C44E52", lw=0.8, alpha=0.8)
    novelty_axis.set(title="Checkerboard novelty and selected boundaries", ylabel="Novelty")
    state_axis = figure.add_subplot(grid[2])
    state_axis.step(times, states, where="post", color="#28536B", lw=1.4)
    for boundary in boundary_times:
        state_axis.axvline(boundary, color="#C44E52", lw=0.7, alpha=0.65)
    state_axis.set(
        xlim=(0, duration), xlabel="Time (s)", ylabel="State", title="Macro-section state path"
    )
    _save(figure, "structure_ssm_boundaries")


def plot_pitch_input(example: pd.Series, pitch_features: pd.DataFrame) -> None:
    row = _feature_row(pitch_features, str(example["segment_id"]))
    source = _read_npz(ROOT / row["source_chroma_relative_path"])
    feature = _read_npz(ROOT / row["pitch_v2_relative_path"])
    times = feature["times"].astype(float)
    valid = feature["valid"].astype(bool)
    states = feature["states"].astype(float)
    states[~valid] = np.nan
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 6.8), constrained_layout=True)
    chroma = axes[0].imshow(
        source["chroma"].astype(float).T,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], -0.5, 11.5),
        cmap="magma",
    )
    axes[0].set(
        yticks=range(12), yticklabels=PITCH_LABELS, title=f"Beat chroma: {example['segment_id']}"
    )
    figure.colorbar(chroma, ax=axes[0], fraction=0.02, pad=0.01, label="Chroma")
    for index, label in enumerate(("5x", "5y", "m3x", "m3y", "M3x", "M3y")):
        axes[1].plot(times, feature["tonnetz"][:, index] + index * 1.15, lw=0.7, label=label)
    axes[1].set(
        yticks=np.arange(6) * 1.15,
        yticklabels=("5x", "5y", "m3x", "m3y", "M3x", "M3y"),
        title="Tonnetz coordinates",
    )
    axes[2].step(times, states, where="mid", color="#28536B", lw=1.2)
    axes[2].set(
        xlabel="Time (s)",
        ylabel="State",
        title="Frozen harmonic-state path (direct transition input; no SSM)",
    )
    _save(figure, "pitch_v2_state_sequence")


def plot_rhythm_input(example: pd.Series, features: pd.DataFrame) -> None:
    row = _feature_row(features, str(example["segment_id"]))
    arrays = _read_npz(ROOT / row["rhythm_relative_path"])
    times = arrays["times"].astype(float)
    values = arrays["vectors"].astype(float)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = np.clip((values - center) / scale, -4, 4)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), constrained_layout=True)
    heatmap = axes[0].imshow(
        standardized.T,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], -0.5, values.shape[1] - 0.5),
        cmap="coolwarm",
        vmin=-4,
        vmax=4,
    )
    axes[0].set(title=f"Rhythm descriptors: {example['segment_id']}", ylabel="Descriptor index")
    figure.colorbar(heatmap, ax=axes[0], fraction=0.02, pad=0.01, label="Robust z-score")
    axes[1].step(times, arrays["states"].astype(int), where="mid", color="#28536B", lw=1.2)
    axes[1].set(
        xlabel="Time (s)",
        ylabel="State",
        title="Frozen rhythm-state path (direct transition input; no SSM)",
    )
    _save(figure, "rhythm_state_sequence")


def _expanded_intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    end = 1.0 - float(np.min(persistence["thresholds"]))
    intervals: list[dict[str, Any]] = []
    for index, dimension in enumerate(persistence["interval_dimension"].astype(int)):
        censored = bool(persistence["interval_censored"][index])
        birth = 1.0 - float(persistence["interval_birth_threshold"][index])
        death = end if censored else 1.0 - float(persistence["interval_death_threshold"][index])
        for _ in range(int(persistence["interval_multiplicity"][index])):
            intervals.append(
                {"dimension": dimension, "birth": birth, "death": death, "censored": censored}
            )
    return intervals


def plot_topology_example(example: pd.Series, view: str) -> None:
    graph = _read_npz(ROOT / example["graph_relative_path"])
    persistence = _read_npz(ROOT / example["sensitivity_persistence_relative_path"])
    figure, axis = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    _draw_graph(
        axis,
        graph,
        threshold=0.0,
        title=f"{VIEW_LABELS[view]} directed state graph\n{example['segment_id']}",
        edge_labels=True,
    )
    _save(figure, f"{view}_directed_state_graph")

    thresholds = persistence["thresholds"].astype(float)
    targets = (0.95, 0.60, 0.30, 0.05)
    figure, axes = plt.subplots(1, 4, figsize=(14.0, 3.7), constrained_layout=True)
    for axis, target in zip(axes, targets, strict=True):
        index = int(np.argmin(np.abs(thresholds - target)))
        threshold = float(thresholds[index])
        title = (
            f"tau={threshold:.2f}\n"
            f"edges={int(persistence['edge_count'][index])}, "
            f"beta0={int(persistence['h0_betti'][index])}, "
            f"beta1={int(persistence['h1_betti'][index])}"
        )
        _draw_graph(axis, graph, threshold=threshold, title=title)
    figure.suptitle(f"{VIEW_LABELS[view]} persistent Path Homology filtration", fontsize=13)
    _save(figure, f"{view}_filtration_process")

    intervals = _expanded_intervals(persistence)
    end = 1.0 - float(np.min(thresholds))
    figure, axis = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    axis.plot([0, end], [0, end], ls="--", color="#777777", lw=1)
    for dimension, marker, color in ((0, "o", "#4472C4"), (1, "^", "#C44E52")):
        selected = [item for item in intervals if item["dimension"] == dimension]
        axis.scatter(
            [item["birth"] for item in selected],
            [item["death"] for item in selected],
            marker=marker,
            s=60,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=f"H{dimension}",
        )
    censored = [item for item in intervals if item["censored"]]
    axis.scatter(
        [item["birth"] for item in censored],
        [item["death"] for item in censored],
        s=95,
        facecolors="none",
        edgecolors="#111111",
        label="right-censored",
    )
    axis.set(
        xlim=(-0.02, end + 0.04),
        ylim=(-0.02, end + 0.04),
        xlabel="Birth a = 1 - tau",
        ylabel="Death a = 1 - tau",
        title=f"{VIEW_LABELS[view]} persistence diagram",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save(figure, f"{view}_persistence_diagram")

    intervals.sort(key=lambda item: (item["dimension"], item["birth"], item["death"]))
    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for row_index, item in enumerate(intervals):
        color = "#4472C4" if item["dimension"] == 0 else "#C44E52"
        axis.hlines(row_index, item["birth"], item["death"], color=color, lw=3)
        axis.plot(item["birth"], row_index, marker="|", color=color, ms=8)
        axis.plot(
            item["death"],
            row_index,
            marker="o" if item["censored"] else "|",
            markerfacecolor="white" if item["censored"] else color,
            color=color,
            ms=6 if item["censored"] else 8,
        )
    axis.set(
        xlim=(-0.02, end + 0.03),
        xlabel="Filtration coordinate a = 1 - tau",
        ylabel="Interval index",
        title=f"{VIEW_LABELS[view]} barcode",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(0.02, 0.96, "blue: H0   red: H1   open: censored", transform=axis.transAxes, va="top")
    _save(figure, f"{view}_barcode")


def plot_group_summary(topology: pd.DataFrame, filtration: pd.DataFrame, view: str) -> None:
    data = topology[
        (topology["view"] == view)
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ]
    metrics = (
        ("vertex_count", "Observed states"),
        ("edge_count", "Directed edges"),
        ("path_entropy", "Path entropy"),
        ("directed_recurrence", "Directed recurrence"),
        ("h0_betti_mean", "Mean beta0"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.5, 4.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [data.loc[data["group"] == group, metric].to_numpy() for group in GROUPS]
        plot = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(plot["boxes"], GROUPS, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        axis.set_xticks(range(1, 5), [GROUP_LABELS[group] for group in GROUPS], rotation=25)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"{VIEW_LABELS[view]} four-group comparison (validation, 180 s)", fontsize=13)
    _save(figure, f"{view}_group_summary")

    data_f = filtration[
        (filtration["view"] == view)
        & (filtration["split"] == "validation")
        & (filtration["scale_seconds"] == 180.0)
    ].copy()
    data_f["a"] = 1.0 - data_f["threshold"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("h0_betti", "h1_betti"), ("Mean beta0", "Mean beta1"), strict=True
    ):
        for group in GROUPS:
            selected = data_f[data_f["group"] == group]
            summary = selected.groupby("a")[metric].agg(["mean", "sem"]).reset_index()
            summary = summary.sort_values("a")
            x = summary["a"].to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            sem = summary["sem"].fillna(0).to_numpy(float)
            axis.plot(
                x, mean, marker="o", ms=3.2, lw=1.6, color=COLORS[group], label=GROUP_LABELS[group]
            )
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.12)
        axis.set(title=title, xlabel="a = 1 - tau", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"{VIEW_LABELS[view]} Betti curves (validation, 180 s; mean +/- SEM)", fontsize=13
    )
    _save(figure, f"{view}_betti_curves")


def plot_pitch_codebook() -> None:
    codebook = _read_npz(ROOT / "features" / "models" / "four_group_pitch_v2_codebook.npz")
    metadata = json.loads(
        (ROOT / "features" / "models" / "four_group_pitch_v2_codebook.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = pd.read_csv(ROOT / "metadata" / "four_group_pitch_v2_codebook_diagnostics.csv")
    figure = plt.figure(figsize=(11.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 2.0))
    axis = figure.add_subplot(grid[0])
    prototypes = codebook["chroma_prototypes"].astype(float)
    image = axis.imshow(prototypes, aspect="auto", cmap="magma")
    axis.set_xticks(range(12), PITCH_LABELS, rotation=30, ha="right")
    axis.set_yticks(range(16), metadata["state_labels"])
    axis.set(
        xlabel="Pitch class", ylabel="Frozen state", title="Four-group Discovery Tonnetz codebook"
    )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Mean normalized chroma")
    subgrid = grid[1].subgridspec(1, 3)
    for subaxis, metric, title in zip(
        (figure.add_subplot(subgrid[0, index]) for index in range(3)),
        ("silhouette", "seed_stability_ari", "inertia_per_step"),
        ("Silhouette", "Seed stability ARI", "Inertia per step"),
        strict=True,
    ):
        subaxis.plot(diagnostics["v_pitch"], diagnostics[metric], marker="o", color="#28536B")
        chosen = diagnostics[diagnostics["v_pitch"] == 16].iloc[0]
        subaxis.scatter([16], [chosen[metric]], s=80, color="#C44E52", zorder=3)
        subaxis.set(xticks=diagnostics["v_pitch"], xlabel="V_pitch", title=title)
        subaxis.grid(alpha=0.2)
    _save(figure, "pitch_v2_codebook")


def _fmt(value: float) -> str:
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _result_tables(
    topology: pd.DataFrame, omnibus: pd.DataFrame, pairwise: pd.DataFrame, view: str
) -> tuple[str, str, str, int, int, str]:
    primary = topology[
        (topology["view"] == view)
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ]
    metrics = (
        "vertex_count",
        "edge_count",
        "path_entropy",
        "directed_recurrence",
        "h0_betti_mean",
        "h1_betti_max",
    )
    medians = primary.groupby("group")[list(metrics)].median()
    lines = ["| 指标 | Classical | Focus | Focus Open | Pop |", "|---|---:|---:|---:|---:|"]
    for metric in metrics:
        lines.append(
            f"| `{metric}` | "
            + " | ".join(_fmt(float(medians.loc[group, metric])) for group in GROUPS)
            + " |"
        )
    tests = omnibus[
        (omnibus["analysis_set"] == "primary_validation_180") & (omnibus["view"] == view)
    ].sort_values(["p_fdr_bh", "epsilon_squared"])
    discoveries = tests[tests["p_fdr_bh"] <= 0.10]
    sensitivity = omnibus[
        (omnibus["analysis_set"] == "sensitivity_validation_300")
        & (omnibus["view"] == view)
        & (omnibus["p_fdr_bh"] <= 0.10)
    ]
    replicated = sorted(set(discoveries["metric"]) & set(sensitivity["metric"]))
    sensitivity_note = (
        f"Validation/300s 敏感性中 {len(sensitivity)}/20 个指标通过独立 FDR；"
        "与主分析共同通过的指标为 "
        f"{', '.join(f'`{metric}`' for metric in replicated) if replicated else '无'}。"
    )
    shown = tests.head(12)
    test_lines = ["| 指标 | $\\epsilon^2$ | FDR $q$ |", "|---|---:|---:|"]
    for _, row in shown.iterrows():
        metric = row["metric"]
        effect = _fmt(float(row["epsilon_squared"]))
        adjusted = _fmt(float(row["p_fdr_bh"]))
        test_lines.append(f"| `{metric}` | {effect} | {adjusted} |")
    key_pairs = pairwise[
        (pairwise["analysis_set"] == "primary_validation_180")
        & (pairwise["view"] == view)
        & (
            ((pairwise["group_a"] == "focus") & (pairwise["group_b"] == "focus_open"))
            | ((pairwise["group_a"] == "focus_open") & (pairwise["group_b"] == "pop"))
        )
    ].sort_values(["p_fdr_bh", "metric"])
    pair_lines = ["| 对比 | 指标 | rank-biserial | FDR $q$ |", "|---|---|---:|---:|"]
    for _, row in key_pairs.head(16).iterrows():
        pair_lines.append(
            f"| {GROUP_LABELS[row['group_a']]} vs {GROUP_LABELS[row['group_b']]} | "
            f"`{row['metric']}` | {_fmt(float(row['rank_biserial_a_minus_b']))} | "
            f"{_fmt(float(row['p_fdr_bh']))} |"
        )
    h1_nonzero = int(np.count_nonzero(primary["h1_betti_max"] > 0))
    return (
        "\n".join(lines),
        "\n".join(test_lines),
        "\n".join(pair_lines),
        len(discoveries),
        h1_nonzero,
        sensitivity_note,
    )


def _method_text(view: str) -> str:
    common = r"""
给定状态序列 $z_1,\ldots,z_T$，相邻有效状态产生计数

$$
C_{ij}=\sum_{t=1}^{T-1}\mathbf 1[z_t=i,z_{t+1}=j],\qquad
P_{ij}=\frac{C_{ij}}{\sum_k C_{ik}}.
$$

每个源状态保留至多 `top_k=6` 条边，并按下降阈值构造

$$
G_\tau=(V,\{(i,j):P_{ij}\ge\tau\}),\qquad
\tau\in\{0.95,0.90,\ldots,0.05\}.
$$

允许 $p$-路径空间记为 $\Omega_p(G_\tau)$，边界算子为

$$
\partial(v_0\cdots v_p)=\sum_{q=0}^{p}(-1)^q
v_0\cdots\widehat{v_q}\cdots v_p,
$$

Path Homology 与 Betti 数为

$$
H_p^{\mathrm{path}}(G_\tau)=
\frac{\ker(\partial_p|_{\Omega_p})}
{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})},
\qquad \beta_p(\tau)=\dim H_p^{\mathrm{path}}(G_\tau).
$$

持久图使用递增坐标 $a=1-\tau$。主分析仅使用
$\tau\in\{0.50,0.60,0.70,0.80,0.90,0.95\}$；扩展阈值只用于敏感性和示例。
"""
    if view == "structure":
        specific = r"""
结构视角先对宏观声学块 $x_t$ 构造余弦自相似矩阵

$$
S_{ij}=\frac{\langle x_i,x_j\rangle}
{\|x_i\|_2\|x_j\|_2},
$$

再用沿对角线移动的棋盘核 $K_L$ 计算新颖度

$$
\nu(t)=\sum_{i=-L}^{L}\sum_{j=-L}^{L}K_L(i,j)S_{t+i,t+j}.
$$

新颖度峰值经最短/最长段约束形成边界 $b_0<\cdots<b_M$；段向量为

$$
s_m=\operatorname{pool}\{x_t:b_m\le t<b_{m+1}\}.
$$

Discovery/180s 四组等量抽样后，对标准化、PCA 投影的 $s_m$ 拟合 K-means，得到
冻结结构原型；Validation 和 300s 只做映射，不参与拟合。SSM 在此视角中是边界构造的一部分。
"""
    elif view == "pitch_v2":
        specific = r"""
音高视角从 12 维 Chroma 向量 $c_t$ 开始，先作能量归一化

$$
\widetilde c_t(k)=\frac{c_t(k)}{\sum_{r=0}^{11}c_t(r)+\varepsilon}.
$$

Harte Tonnetz 将每个音级嵌入五度、小三度与大三度三个圆的正余弦坐标：

$$
T(c)=\sum_{k=0}^{11}\widetilde c(k)
\begin{bmatrix}
\cos(7\pi k/6)\\ \sin(7\pi k/6)\\
\cos(3\pi k/2)\\ \sin(3\pi k/2)\\
\cos(2\pi k/3)\\ \sin(2\pi k/3)
\end{bmatrix}.
$$

四组 Discovery/180s 严格等量抽样，在六维 Tonnetz 上拟合

$$
\min_{\mu_1,\ldots,\mu_{16}}\sum_t\min_v\|T(c_t)-\mu_v\|_2^2,
$$

得到固定 $V_{pitch}=16$ 的谐波骨架码本。音高图直接由相邻冻结状态建立；SSM 不参与主图构造。
"""
    else:
        specific = r"""
节奏视角把每个分析窗表示为多维节奏描述向量 $r_t$（局部 onset 强度、节拍/速度、
IOI 与 tempogram 形态等）。缺失维使用 Discovery 中位数 $m_d$ 填补，再标准化：

$$
\widetilde r_{td}=\frac{r_{td}-\mu_d}{\sigma_d},\qquad
r_{td}^{\mathrm{fill}}=\begin{cases}r_{td},&\text{valid},\\m_d,&\text{missing}.\end{cases}
$$

四组 Discovery/180s 严格等量抽样后拟合冻结 K-means 节奏码本：

$$
z_t=\arg\min_v\|\widetilde r_t-\mu_v^{(r)}\|_2^2.
$$

节奏图直接由相邻冻结状态建立；SSM 不参与主图构造。
"""
    return specific + common


def write_report(
    topology: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    summary: dict[str, Any],
    examples: dict[str, pd.Series],
    view: str,
) -> Path:
    medians, tests, pairs, discoveries, h1_nonzero, sensitivity_note = _result_tables(
        topology, omnibus, pairwise, view
    )
    example = examples[view]
    n_primary = int(summary["primary_validation_n_per_view"])
    figure_stems = {
        "structure": "structure_ssm_boundaries",
        "pitch_v2": "pitch_v2_state_sequence",
        "rhythm": "rhythm_state_sequence",
    }
    extra_pitch = (
        "![pitch_v2_codebook](../runs/four_group_path_homology/pitch_v2_codebook.png)\n\n"
        if view == "pitch_v2"
        else ""
    )
    content = f"""# 四组音乐 {VIEW_LABELS[view]} Path Homology 完整分析报告

## 1. 研究范围与证据边界

本报告把新建的 `focus_open_music` 作为独立的 **Focus Open** 组，与原 Focus、Pop、Classical
共同分析，不把它当作原 Focus 的替代品。四组共 {summary["segments"]:,} 个 180/300 秒片段、
{summary["tracks"]:,} 首曲目；本视角共 {summary["view_counts"][view]:,} 个片段结果，零失败。

- 码本/状态模型拟合：仅 Discovery/180s，四组严格等量抽样；
- 主推断：Validation/180s，共 {n_primary} 个片段；
- 敏感性：Validation/300s；Discovery/180s 仅探索；
- Holdout 未进入四组 omnibus，因为原三组并不都具有同构 holdout；
- FDR 家族：每个分析集统一校正 3 视角 × 20 指标；六个两两对比另成一族；
- 这是观察性声学结构比较，不能推出注意力、认知收益或因果效应。

## 2. 视角思想、原理与公式

{_method_text(view)}

## 3. 可视化的持续同调过程（示例）

示例自动选自 Focus Open validation/180s：`{example["segment_id"]}`。选择规则优先最大
$H_1$ 观测持久性，其次为 $H_1$ 峰值、边数和稳定 ID；因此它用于解释机制，不作为“典型曲目”证据。

![input](../runs/four_group_path_homology/{figure_stems[view]}.png)

{extra_pitch}![directed_graph](../runs/four_group_path_homology/{view}_directed_state_graph.png)

![filtration](../runs/four_group_path_homology/{view}_filtration_process.png)

![persistence](../runs/four_group_path_homology/{view}_persistence_diagram.png)

![barcode](../runs/four_group_path_homology/{view}_barcode.png)

## 4. 四组结果

Validation/180s 中位数：

{medians}

主 omnibus 中，本视角有 **{discoveries}/20** 个指标通过统一 FDR $q\\le0.10$；
$H_1$ 非零片段为 **{h1_nonzero}/{n_primary}**。中位数为零时，即使秩检验显著，也应解释为
零膨胀发生率或尾部分布差异，而不是“普遍存在环”。

{sensitivity_note}

按统一 FDR 排序的前 12 个 omnibus 指标：

{tests}

Focus–Focus Open 与 Focus Open–Pop 的关键两两比较（前 16 项；正 rank-biserial 表示前者更高）：

{pairs}

![group_summary](../runs/four_group_path_homology/{view}_group_summary.png)

![betti_curves](../runs/four_group_path_homology/{view}_betti_curves.png)

## 5. 解读

1. **Focus Open 必须保持独立。** 它的来源、授权条件、筛选机制和原 Focus 不同；拓扑相似只能支持
   “在本表示下接近”，不能证明数据分布等价。
2. **优先看效应量和方向。** Kruskal–Wallis 的 $\\epsilon^2$ 回答四组总体可分程度；
   rank-biserial 回答具体两组方向。统一 FDR 后未通过的差异一律标为不支持。
3. **$H_0$ 与图描述量通常比 $H_1$ 更稳定。** 状态数、边数、路径熵、复现率和连通分支变化反映
   状态组织方式；$H_1$ 只在少量曲目出现时应作为稀有结构现象报告。
4. **阈值敏感性不是主结论。** 扩展过滤用于展示类的出生/死亡；主统计仍冻结在 0.50–0.95。
5. **跨视角结论需查阅三份报告。** 结构描述宏观段落，pitch_v2 描述谐波骨架，节奏描述局部
   时间组织；任何单一视角都不能代表完整音乐结构。

## 6. 复现与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/run_four_group_path_homology.py --workers 6
python scripts/render_four_group_path_reports.py
```

数值产物：`metadata/four_group_topology_segments.csv`、
`metadata/four_group_topology_filtration.csv`、
`metadata/four_group_topology_filtration_sensitivity.csv`、
`metadata/four_group_statistical_tests.csv`、`metadata/four_group_pairwise_tests.csv`。
模型、特征、图和同调结果均位于带 `four_group` 的隔离命名空间；PNG 与 SVG 同步输出到
`runs/four_group_path_homology/`，未覆盖原三组验证产物。
"""
    path = ROOT / "docs" / f"path-homology-{view.replace('_v2', '-v2')}-four-group-analysis.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_combined_report(
    topology: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    primary_tests = omnibus[omnibus["analysis_set"] == "primary_validation_180"]
    primary_pairs = pairwise[pairwise["analysis_set"] == "primary_validation_180"]
    view_rows = [
        "| 视角 | 主 Omnibus | 300s Omnibus | 主/300s 交集 | Focus vs Focus Open | "
        "Focus Open vs Pop FDR 通过 | H1 非零片段 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    details: list[str] = []
    for view in VIEW_LABELS:
        view_tests = primary_tests[primary_tests["view"] == view]
        sensitivity_tests = omnibus[
            (omnibus["analysis_set"] == "sensitivity_validation_300") & (omnibus["view"] == view)
        ]
        primary_metrics = set(view_tests.loc[view_tests["p_fdr_bh"] <= 0.10, "metric"])
        sensitivity_metrics = set(
            sensitivity_tests.loc[sensitivity_tests["p_fdr_bh"] <= 0.10, "metric"]
        )
        focus_open = primary_pairs[
            (primary_pairs["view"] == view)
            & (primary_pairs["group_a"] == "focus")
            & (primary_pairs["group_b"] == "focus_open")
        ]
        open_pop = primary_pairs[
            (primary_pairs["view"] == view)
            & (primary_pairs["group_a"] == "focus_open")
            & (primary_pairs["group_b"] == "pop")
        ]
        primary = topology[
            (topology["view"] == view)
            & (topology["split"] == "validation")
            & (topology["scale_seconds"] == 180.0)
        ]
        h1 = int(np.count_nonzero(primary["h1_betti_max"] > 0))
        view_rows.append(
            f"| {VIEW_LABELS[view]} | {len(primary_metrics)}/20 | "
            f"{len(sensitivity_metrics)}/20 | {len(primary_metrics & sensitivity_metrics)}/20 | "
            f"{int((focus_open['p_fdr_bh'] <= 0.10).sum())}/20 | "
            f"{int((open_pop['p_fdr_bh'] <= 0.10).sum())}/20 | {h1}/{len(primary)} |"
        )
        focus_metrics = focus_open.loc[focus_open["p_fdr_bh"] <= 0.10, "metric"].tolist()
        pop_metrics = open_pop.loc[open_pop["p_fdr_bh"] <= 0.10, "metric"].tolist()
        details.append(
            f"- **{VIEW_LABELS[view]}**：Focus vs Focus Open 支持差异的指标为 "
            f"{', '.join(f'`{metric}`' for metric in focus_metrics) if focus_metrics else '无'}；"
            f"Focus Open vs Pop 为 "
            f"{', '.join(f'`{metric}`' for metric in pop_metrics) if pop_metrics else '无'}。"
        )
    content = f"""# 四组音乐三视角 Path Homology 综合报告

## 1. 分析设计

本轮将 Focus Open 作为独立第四组，完成结构、Tonnetz pitch_v2 与节奏三个视角的全量重跑。
共有 {summary["segments"]:,} 个片段、{summary["tracks"]:,} 首曲目和
{summary["segment_views"]:,} 个片段-视角结果。原三组验证产物保持不变。

主证据层为 Validation/180s；Validation/300s 是敏感性，Discovery/180s 是探索性。
状态模型只在四组 Discovery/180s 严格等量拟合。统一 omnibus FDR 家族覆盖三个视角的 60 个指标；
两两比较覆盖 60 指标 × 6 对比。Holdout 不进入四组检验。

## 2. 主结果概览

{chr(10).join(view_rows)}

关键两两对比：

{chr(10).join(details)}

## 3. 跨视角解读

1. **不能把 Focus Open 与 Focus 合并。** 两者即使在部分指标上没有检出差异，也只能说明在当前
   样本量、表示和冻结检验下“未拒绝相同”，不等价于证明同分布。
2. **三视角回答不同问题。** 结构视角的 SSM 与边界描述宏观段落组织；pitch_v2 描述五度/三度
   谐波骨架的有向移动；节奏视角描述局部时间组织。跨视角同时出现且方向一致的差异更值得关注，
   但仍是观察性证据。
3. **优先解释图组织与 H0。** 状态数、边数、路径熵、复现率、互惠性与 H0 通常比稀疏 H1 稳定。
   当各组 H1 中位数均为零时，显著秩检验只表明零膨胀率或尾部不同。
4. **Focus Open 的位置不是单一“更像谁”。** 应分别在结构、音高和节奏报告中查看效应方向；
   不用一个未经预注册的综合距离把多维差异压缩成单一相似度排名。
5. **没有认知或因果结论。** 本分析不能证明某类音乐提升专注，也不能把拓扑差异解释为机制因果。

## 4. 报告与产物

- [结构四组报告](path-homology-structure-four-group-analysis.md)
- [音高四组报告](path-homology-pitch-v2-four-group-analysis.md)
- [节奏四组报告](path-homology-rhythm-four-group-analysis.md)

数值清单位于 `metadata/four_group_*`，图与同调结果分别位于带 `four_group` 的隔离目录；
每幅论文图同时提供 PNG 和 SVG。
"""
    path = ROOT / "docs" / "path-homology-four-group-analysis.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    topology = pd.read_csv(ROOT / "metadata" / "four_group_topology_segments.csv")
    filtration = pd.read_csv(ROOT / "metadata" / "four_group_topology_filtration_sensitivity.csv")
    omnibus = pd.read_csv(ROOT / "metadata" / "four_group_statistical_tests.csv")
    pairwise = pd.read_csv(ROOT / "metadata" / "four_group_pairwise_tests.csv")
    features = pd.read_csv(ROOT / "metadata" / "four_group_feature_segments.csv")
    pitch_features = pd.read_csv(ROOT / "metadata" / "four_group_pitch_v2_features.csv")
    summary = json.loads(
        (ROOT / "metadata" / "four_group_path_homology_summary.json").read_text(encoding="utf-8")
    )
    examples = {view: _example(topology, view) for view in VIEW_LABELS}
    plot_structure_input(examples["structure"], features)
    plot_pitch_input(examples["pitch_v2"], pitch_features)
    plot_rhythm_input(examples["rhythm"], features)
    plot_pitch_codebook()
    for view in VIEW_LABELS:
        plot_topology_example(examples[view], view)
        plot_group_summary(topology, filtration, view)
        path = write_report(topology, omnibus, pairwise, summary, examples, view)
        print(path.relative_to(ROOT).as_posix())
    combined = write_combined_report(topology, omnibus, pairwise, summary)
    print(combined.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
