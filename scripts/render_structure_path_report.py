# ruff: noqa: E501
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "structure_path_homology"
EXAMPLE_ID = "pop_jamendo_1045184__180s"
EXAMPLE_GROUP = "pop"
EXAMPLE_SPLIT = "discovery"
COLORS = {"classical": "#4472C4", "focus": "#ED7D31", "pop": "#70AD47"}


def _latexify_report(text: str) -> str:
    """Convert the generated report's mathematical notation to Markdown LaTeX."""

    replacements = (
        (
            "对第 i 帧声学向量 x_i∈R^d，先以训练片段内部的中位数与 MAD 做稳健标准化：",
            r"对第 $i$ 帧声学向量 $\mathbf{x}_i\in\mathbb{R}^d$，先以训练片段内部的中位数与 MAD 做稳健标准化：",
        ),
        (
            "z_i=(x_i-med(x))/(1.4826·MAD(x)+epsilon)。",
            r"""$$
\mathbf{z}_i=
\frac{\mathbf{x}_i-\operatorname{med}(\mathbf{x})}
{1.4826\,\operatorname{MAD}(\mathbf{x})+\varepsilon}.
$$""",
        ),
        (
            "u_i=[z_i,1]/||[z_i,1]||_2。",
            r"""$$
\mathbf{u}_i=
\frac{[\mathbf{z}_i,1]}
{\left\lVert[\mathbf{z}_i,1]\right\rVert_2}.
$$""",
        ),
        (
            "S_ij=(1+u_i^T u_j)/2，S_ij∈[0,1]。",
            r"""$$
S_{ij}=\frac{1+\mathbf{u}_i^{\mathsf T}\mathbf{u}_j}{2},
\qquad S_{ij}\in[0,1].
$$""",
        ),
        (
            "令 L_t=[t-h,t)，R_t=[t,t+h)。Foote 棋盘核 novelty 为：",
            r"令 $L_t=[t-h,t)$、$R_t=[t,t+h)$。Foote 棋盘核 novelty 为：",
        ),
        (
            "nu(t)={sum_(i,j∈L_t) S_ij + sum_(i,j∈R_t) S_ij - sum_(i∈L_t,j∈R_t) S_ij - sum_(i∈R_t,j∈L_t) S_ij}/(2h^2)。",
            r"""$$
\nu(t)=\frac{1}{2h^2}
\left(
\sum_{i,j\in L_t}S_{ij}
+\sum_{i,j\in R_t}S_{ij}
-\sum_{\substack{i\in L_t\\j\in R_t}}S_{ij}
-\sum_{\substack{i\in R_t\\j\in L_t}}S_{ij}
\right).
$$""",
        ),
        (
            "同侧相似而跨侧不相似时，nu(t) 形成峰值。实现中使用 8 s 半窗、[0.25,0.5,0.25] 平滑、median+1.5·MAD 峰值阈值，并约束段长为 8–45 s；过长的均质区间在局部最大 novelty 处分割。",
            r"""同侧相似而跨侧不相似时，$\nu(t)$ 形成峰值。实现中使用 $8\,\mathrm{s}$ 半窗、$[0.25,0.5,0.25]$ 平滑，以及

$$
\operatorname{median}(\nu)+1.5\,\operatorname{MAD}(\nu)
$$

作为峰值阈值，并约束段长为 $8$–$45\,\mathrm{s}$；过长的均质区间在局部最大 novelty 处分割。""",
        ),
        (
            "边界 0=b_0<...<b_K=T 确定 K 个块。第 k 个块向量为有效帧均值：",
            r"边界 $0=b_0<b_1<\cdots<b_K=T$ 确定 $K$ 个块。第 $k$ 个块向量为有效帧均值：",
        ),
        (
            "q_k=(1/|I_k|) sum_(i∈I_k) x_i。",
            r"""$$
\mathbf{q}_k=
\frac{1}{|I_k|}\sum_{i\in I_k}\mathbf{x}_i.
$$""",
        ),
        (
            "使用 discovery/180s 三组平衡抽样拟合声学标准化与 32 维 PCA，再以 16 个 MiniBatch K-means 中心 c_m 定义共享结构状态：",
            r"使用 discovery/180s 三组平衡抽样拟合声学标准化与 32 维 PCA，再以 16 个 MiniBatch K-means 中心 $\mathbf{c}_m$ 定义共享结构状态：",
        ),
        (
            "s_k=argmin_m ||P D^(-1)(q_k-mu)-c_m||_2^2。",
            r"""$$
s_k=\underset{m}{\arg\min}\;
\left\lVert
\mathbf{P}\mathbf{D}^{-1}(\mathbf{q}_k-\boldsymbol{\mu})-\mathbf{c}_m
\right\rVert_2^2.
$$""",
        ),
        (
            "对状态路径 (s_0,...,s_K)，转移计数 C_uv=#{k:s_k=u,s_(k+1)=v}，出向概率\n\np_uv=C_uv/sum_w C_uw。",
            r"""对状态路径 $(s_0,\ldots,s_K)$，转移计数与出向概率分别定义为

$$
C_{uv}=\left|\left\{k:s_k=u,\ s_{k+1}=v\right\}\right|,
\qquad
p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.
$$""",
        ),
        (
            "过滤图 G_tau 保留 p_uv≥tau 的非自环边，并限制每个源状态最多 top-6 条出边。阈值从 0.95 下降到 0.05 时只增边，因此形成 G_0.95⊆...⊆G_0.05。",
            r"""过滤图 $G_\tau$ 保留满足 $p_{uv}\geq\tau$ 的非自环边，并限制每个源状态最多 top-6 条出边。阈值从 $0.95$ 下降到 $0.05$ 时只增边，因此形成

$$
G_{0.95}\subseteq G_{0.90}\subseteq\cdots\subseteq G_{0.05}.
$$""",
        ),
        (
            "对允许 p-路径 e_(v0...vp)，GLMY 边界算子为：\n\npartial e_(v0...vp)=sum_(i=0)^p (-1)^i e_(v0...vhat_i...vp)。",
            r"""对允许的 $p$-路径 $e_{v_0\ldots v_p}$，GLMY 边界算子为：

$$
\partial e_{v_0\ldots v_p}
=\sum_{i=0}^{p}(-1)^i
e_{v_0\ldots\widehat{v_i}\ldots v_p}.
$$""",
        ),
        (
            "令 A_p 为允许路径空间，Omega_p={a∈A_p:partial a∈A_(p-1)}，则\n\nH_p^path(G)=ker(partial_p|Omega_p)/im(partial_(p+1)|Omega_(p+1))，beta_p=dim H_p^path(G)。",
            r"""令 $A_p$ 为允许路径空间，并定义

$$
\Omega_p=
\left\{a\in A_p:\partial a\in A_{p-1}\right\}.
$$

则 Path Homology 群及其 Betti 数为

$$
H_p^{\mathrm{path}}(G)=
\frac{\ker\!\left(\partial_p\rvert_{\Omega_p}\right)}
{\operatorname{im}\!\left(\partial_{p+1}\rvert_{\Omega_{p+1}}\right)},
\qquad
\beta_p=\dim H_p^{\mathrm{path}}(G).
$$""",
        ),
        (
            "过滤图之间的包含映射诱导持久模。本文报告 H0/H1 秩不变量、持久区间、barcode 和持久图。为符合常规横轴递增表示，绘图使用 a=1-tau。",
            r"""过滤图之间的包含映射诱导持久模。本文报告 $H_0/H_1$ 秩不变量、持久区间、barcode 和持久图。为符合常规横轴递增表示，绘图使用

$$
a=1-\tau.
$$""",
        ),
        (
            "- tau=0.95：仅 3→10、10→6，beta0=1、beta1=0；\n- tau=0.60：加入 6→3，形成 6→3→10→6 的有向一维类，beta1 从 0 变为 1；\n- tau=0.30：加入 6→10，新增的允许 2-路径边界使该 H1 类死亡，beta1 回到 0。",
            r"""- $\tau=0.95$：仅 $3\to10$、$10\to6$，$\beta_0=1$、$\beta_1=0$；
- $\tau=0.60$：加入 $6\to3$，形成 $6\to3\to10\to6$ 的有向一维类，$\beta_1$ 从 0 变为 1；
- $\tau=0.30$：加入 $6\to10$，新增的允许 2-路径边界使该 $H_1$ 类死亡，$\beta_1$ 回到 0。""",
        ),
        (
            "因此 H1 在阈值坐标中出生于 tau=0.60、死亡于 tau=0.30、寿命为 0.30；在递增坐标 a=1-tau 中是区间 [0.40,0.70)。H0 从 tau=0.95 起一直存活到观察终点，属于右删失区间。",
            r"因此 $H_1$ 在阈值坐标中出生于 $\tau=0.60$、死亡于 $\tau=0.30$、寿命为 $0.30$；在递增坐标 $a=1-\tau$ 中是区间 $[0.40,0.70)$。$H_0$ 从 $\tau=0.95$ 起一直存活到观察终点，属于右删失区间。",
        ),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(
        r"pseudo-F=([0-9.]+)，p=([0-9.]+)，n=([0-9]+)，有效维数=([0-9]+)",
        lambda match: (
            r"$\mathrm{pseudo}\text{-}F="
            + match.group(1)
            + r"$，$p="
            + match.group(2)
            + r"$，$n="
            + match.group(3)
            + r"$，有效维数为 $"
            + match.group(4)
            + "$"
        ),
        text,
    )
    return (
        text.replace("FDR q≤0.10", r"FDR $q\leq0.10$")
        .replace("| epsilon² | FDR q |", r"| $\epsilon^2$ | FDR $q$ |")
    )


def _save(figure: plt.Figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _example_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature_path = (
        ROOT
        / "features"
        / "structure"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz"
    )
    graph_path = (
        ROOT
        / "graphs"
        / "structure"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz"
    )
    persistence_path = (
        ROOT
        / "homology"
        / "persistence_sensitivity"
        / "structure"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz"
    )
    return tuple(
        {name: np.asarray(archive[name]) for name in archive.files}
        for archive in (np.load(feature_path), np.load(graph_path), np.load(persistence_path))
    )  # type: ignore[return-value]


def plot_ssm(feature: dict[str, np.ndarray]) -> Path:
    boundaries = feature["boundary_times"].astype(float)
    states = feature["states"].astype(int)
    similarity = feature["self_similarity"].astype(float)
    novelty = feature["novelty"].astype(float)
    duration = float(boundaries[-1])
    times = np.linspace(0.0, duration, novelty.size, endpoint=False)

    figure = plt.figure(figsize=(9.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(6.0, 1.5, 0.9))
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(
        similarity,
        origin="lower",
        extent=(0, duration, 0, duration),
        cmap="magma",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    for boundary in boundaries[1:-1]:
        axis.axvline(boundary, color="white", lw=0.8, alpha=0.85)
        axis.axhline(boundary, color="white", lw=0.8, alpha=0.85)
    axis.set(
        title=f"Self-Similarity Matrix and detected macro boundaries: {EXAMPLE_ID}",
        xlabel="Time (s)",
        ylabel="Time (s)",
    )
    figure.colorbar(image, ax=axis, label="cosine similarity mapped to [0, 1]", shrink=0.82)

    novelty_axis = figure.add_subplot(grid[1], sharex=axis)
    novelty_axis.plot(times, novelty, color="#4C78A8", lw=1.5)
    novelty_axis.fill_between(times, novelty, color="#4C78A8", alpha=0.22)
    for boundary in boundaries[1:-1]:
        novelty_axis.axvline(boundary, color="#D62728", lw=1.0, alpha=0.8)
    novelty_axis.set(ylabel="novelty", xlim=(0, duration), ylim=(0, 1.05))

    state_axis = figure.add_subplot(grid[2], sharex=axis)
    palette = plt.get_cmap("tab20")
    for index, state in enumerate(states):
        left, right = boundaries[index : index + 2]
        state_axis.broken_barh(
            [(left, right - left)],
            (0, 1),
            facecolors=palette(int(state) % 20),
            edgecolors="white",
        )
        state_axis.text((left + right) / 2, 0.5, str(state), ha="center", va="center", fontsize=9)
    state_axis.set(xlabel="Time (s)", ylabel="state", yticks=[], ylim=(0, 1), xlim=(0, duration))
    return _save(figure, "structure_ssm_boundaries.png")


def _draw_graph(
    axis: plt.Axes,
    vertices: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float,
    title: str,
    beta: tuple[int, int] | None = None,
) -> None:
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(vertices), endpoint=False)
    positions = {int(vertex): np.array([np.cos(angle), np.sin(angle)]) for vertex, angle in zip(vertices, angles, strict=True)}
    active = weights >= threshold - 1e-12
    for source, target, weight in zip(sources[active], targets[active], weights[active], strict=True):
        start, end = positions[int(source)], positions[int(target)]
        direction = end - start
        norm = float(np.linalg.norm(direction))
        unit = direction / norm
        left = start + unit * 0.18
        right = end - unit * 0.18
        bend = 0.14 if np.any((sources == target) & (targets == source)) else 0.0
        patch = FancyArrowPatch(
            left,
            right,
            arrowstyle="-|>",
            mutation_scale=14,
            connectionstyle=f"arc3,rad={bend}",
            lw=1.2 + 2.8 * float(weight),
            color="#4C78A8",
            alpha=0.9,
        )
        axis.add_patch(patch)
        midpoint = (start + end) / 2
        axis.text(midpoint[0], midpoint[1], f"{weight:.2f}", fontsize=8, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    for vertex in vertices:
        point = positions[int(vertex)]
        axis.scatter(point[0], point[1], s=720, color="#F2C14E", edgecolor="#333333", zorder=3)
        axis.text(point[0], point[1], str(int(vertex)), ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    subtitle = f"tau >= {threshold:g}"
    if beta is not None:
        subtitle += f" | beta0={beta[0]}, beta1={beta[1]}"
    axis.set_title(f"{title}\n{subtitle}", fontsize=11)
    axis.set_aspect("equal")
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.35, 1.35)
    axis.axis("off")


def plot_state_graph(graph: dict[str, np.ndarray]) -> Path:
    figure, axis = plt.subplots(figsize=(6.4, 5.5), constrained_layout=True)
    _draw_graph(
        axis,
        graph["vertices"],
        graph["edge_source"],
        graph["edge_target"],
        graph["edge_weight"],
        threshold=0.0,
        title=f"Directed macro-state transition graph: {EXAMPLE_ID}",
    )
    return _save(figure, "structure_directed_state_graph.png")


def plot_filtration(graph: dict[str, np.ndarray], persistence: dict[str, np.ndarray]) -> Path:
    thresholds = persistence["thresholds"].astype(float)
    beta0 = persistence["h0_betti"].astype(int)
    beta1 = persistence["h1_betti"].astype(int)
    selected = (0.95, 0.6, 0.3)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    titles = ("Before the cycle", "H1 is born", "H1 is killed")
    for axis, threshold, title in zip(axes, selected, titles, strict=True):
        index = int(np.argmin(np.abs(thresholds - threshold)))
        _draw_graph(
            axis,
            graph["vertices"],
            graph["edge_source"],
            graph["edge_target"],
            graph["edge_weight"],
            threshold=threshold,
            title=title,
            beta=(int(beta0[index]), int(beta1[index])),
        )
    figure.suptitle("Persistent path homology filtration example", fontsize=14)
    return _save(figure, "structure_filtration_process.png")


def _interval_rows(persistence: dict[str, np.ndarray]) -> list[dict[str, float | int | bool]]:
    terminal = float(np.min(persistence["thresholds"]))
    rows: list[dict[str, float | int | bool]] = []
    for dimension, birth, death, censored, multiplicity in zip(
        persistence["interval_dimension"],
        persistence["interval_birth_threshold"],
        persistence["interval_death_threshold"],
        persistence["interval_censored"],
        persistence["interval_multiplicity"],
        strict=True,
    ):
        birth_x = 1.0 - float(birth)
        death_x = 1.0 - (terminal if bool(censored) else float(death))
        rows.append(
            {
                "dimension": int(dimension),
                "birth": birth_x,
                "death": death_x,
                "censored": bool(censored),
                "multiplicity": int(multiplicity),
            }
        )
    return rows


def plot_persistence(persistence: dict[str, np.ndarray]) -> tuple[Path, Path]:
    rows = _interval_rows(persistence)
    colors = {0: "#4C78A8", 1: "#E45756"}
    markers = {0: "o", 1: "^"}

    figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="#777777", lw=1, ls="--")
    for row in rows:
        dimension = int(row["dimension"])
        axis.scatter(
            float(row["birth"]),
            float(row["death"]),
            s=85,
            marker=markers[dimension],
            color=colors[dimension],
            label=f"H{dimension}",
        )
        if bool(row["censored"]):
            axis.annotate("censored", (float(row["birth"]), float(row["death"])), xytext=(6, -12), textcoords="offset points", fontsize=8)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axis.legend(unique.values(), unique.keys(), frameon=False)
    axis.set(
        title="Persistent path diagram",
        xlabel="birth filtration value a = 1 - tau",
        ylabel="death filtration value a = 1 - tau",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    diagram = _save(figure, "structure_persistence_diagram.png")

    expanded: list[dict[str, float | int | bool]] = []
    for row in rows:
        expanded.extend([row] * int(row["multiplicity"]))
    figure, axis = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    for index, row in enumerate(expanded):
        dimension = int(row["dimension"])
        axis.hlines(index, float(row["birth"]), float(row["death"]), color=colors[dimension], lw=5)
        axis.scatter(float(row["birth"]), index, color=colors[dimension], s=28, zorder=3)
        if bool(row["censored"]):
            axis.scatter(float(row["death"]), index, facecolor="white", edgecolor=colors[dimension], s=36, zorder=3)
        else:
            axis.scatter(float(row["death"]), index, color=colors[dimension], marker="x", s=45, zorder=3)
    axis.set_yticks(range(len(expanded)), [f"H{int(row['dimension'])}" for row in expanded])
    axis.set(
        title="Persistent path barcode",
        xlabel="filtration value a = 1 - tau",
        xlim=(0, 1),
        ylim=(-0.7, len(expanded) - 0.3),
    )
    barcode = _save(figure, "structure_barcode.png")
    return diagram, barcode


def plot_group_summary(topology: pd.DataFrame) -> Path:
    subset = topology[
        (topology["view"] == "structure")
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ].copy()
    groups = ["classical", "focus", "pop"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    for group in groups:
        values = subset.loc[subset["group"] == group, "self_transition_ratio"].to_numpy(float)
        axes[0].boxplot(
            values,
            positions=[groups.index(group)],
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": COLORS[group], "alpha": 0.55},
            medianprops={"color": "black"},
        )
    axes[0].set_xticks(range(3), groups, rotation=20)
    axes[0].set(title="Self-transition ratio", ylabel="ratio")

    medians = subset.groupby("group")["vertex_count"].median().reindex(groups)
    axes[1].bar(groups, medians, color=[COLORS[group] for group in groups], alpha=0.8)
    axes[1].set(title="Median macro-state count", ylabel="vertices")
    axes[1].tick_params(axis="x", rotation=20)

    prevalence = (
        subset.assign(nonzero=subset["h1_betti_max"] > 0)
        .groupby("group")["nonzero"]
        .mean()
        .reindex(groups)
    )
    axes[2].bar(groups, prevalence, color=[COLORS[group] for group in groups], alpha=0.8)
    axes[2].set(title="Nonzero H1 prevalence", ylabel="fraction", ylim=(0, max(0.2, float(prevalence.max()) * 1.25)))
    axes[2].tick_params(axis="x", rotation=20)
    figure.suptitle("Structure-view validation summary (180 s)", fontsize=14)
    return _save(figure, "structure_group_summary.png")


def write_report(
    feature: dict[str, np.ndarray],
    graph: dict[str, np.ndarray],
    persistence: dict[str, np.ndarray],
    figures: list[Path],
) -> Path:
    topology = pd.read_csv(ROOT / "metadata" / "topology_segments.csv")
    tests = pd.read_csv(ROOT / "metadata" / "topology_statistical_tests.csv")
    permanova = pd.read_csv(ROOT / "metadata" / "topology_permanova.csv")
    classification = pd.read_csv(ROOT / "metadata" / "classification_results.csv")
    summary = json.loads((ROOT / "metadata" / "topology_summary.json").read_text(encoding="utf-8"))
    feature_summary = json.loads((ROOT / "metadata" / "feature_summary.json").read_text(encoding="utf-8"))
    structure_tests = tests[
        (tests["analysis_set"] == "primary_validation_180") & (tests["view"] == "structure")
    ].sort_values(["p_fdr_bh", "epsilon_squared"])
    significant = structure_tests[structure_tests["p_fdr_bh"] <= 0.10]
    primary_permanova = permanova[permanova["analysis_set"] == "primary_validation_180"].iloc[0]
    primary_classification = classification[
        classification["analysis_set"] == "primary_validation_180"
    ].set_index("feature_set")
    structure_rows = topology[
        (topology["view"] == "structure")
        & (topology["split"] == "validation")
        & (topology["scale_seconds"] == 180.0)
    ]
    nonzero = structure_rows.groupby("group")["h1_betti_max"].apply(lambda values: int((values > 0).sum()))
    counts = structure_rows.groupby("group").size()
    boundaries = ", ".join(f"{value:.2f}" for value in feature["boundary_times"])
    states = " → ".join(str(int(value)) for value in feature["states"])
    edges = ", ".join(
        f"{int(source)}→{int(target)} ({float(weight):.3f})"
        for source, target, weight in zip(
            graph["edge_source"], graph["edge_target"], graph["edge_weight"], strict=True
        )
    )
    figure_links = "\n".join(
        f"![{path.stem}](../runs/structure_path_homology/{path.name})" for path in figures
    )
    table_rows = "\n".join(
        f"| {row.metric} | {row.classical_median:.3f} | {row.focus_median:.3f} | "
        f"{row.pop_median:.3f} | {row.epsilon_squared:.3f} | {row.p_fdr_bh:.3g} |"
        for row in significant.itertuples(index=False)
    )
    report = f"""# Path Homology 结构视角：方法、原理与完整重分析报告

生成日期：2026-08-01。结构视角已加入批量 Path Homology 管线，并在 1,600 个片段、800 首曲目上完成四视角重分析。全部结果是音乐结构描述，不构成注意力提升、治疗或因果效果证据。

## 1. 核心思想

原有 pitch、rhythm、modulation 视角刻画局部音高、节奏和调制状态；结构视角把时间分辨率提升到“段落”。它先用自相似矩阵（SSM）寻找声学纹理发生改变的位置，再把每个边界区间汇聚成宏观声学块，映射到由 discovery/180s 数据拟合并冻结的共享原型。这样得到的状态路径不再表示逐帧音色，而是表示 A/B/C 等高阶段落形态及其有向转换。

流程为：短时声学向量 → SSM → novelty → 段落边界 → 块向量 → 共享结构状态 → 有向转移图 → 持久 Path Homology。

## 2. 原理与公式

### 2.1 自相似矩阵

对第 i 帧声学向量 x_i∈R^d，先以训练片段内部的中位数与 MAD 做稳健标准化：

z_i=(x_i-med(x))/(1.4826·MAD(x)+epsilon)。

加入常数偏置坐标并单位化，避免位于稳健中心的帧退化为零向量：

u_i=[z_i,1]/||[z_i,1]||_2。

余弦自相似矩阵定义为：

S_ij=(1+u_i^T u_j)/2，S_ij∈[0,1]。

矩阵对角附近的高值表示局部连续性，远离对角线的重复亮块表示非相邻段落复现。

### 2.2 棋盘核 novelty 与边界

令 L_t=[t-h,t)，R_t=[t,t+h)。Foote 棋盘核 novelty 为：

nu(t)={{sum_(i,j∈L_t) S_ij + sum_(i,j∈R_t) S_ij - sum_(i∈L_t,j∈R_t) S_ij - sum_(i∈R_t,j∈L_t) S_ij}}/(2h^2)。

同侧相似而跨侧不相似时，nu(t) 形成峰值。实现中使用 8 s 半窗、[0.25,0.5,0.25] 平滑、median+1.5·MAD 峰值阈值，并约束段长为 8–45 s；过长的均质区间在局部最大 novelty 处分割。

### 2.3 宏观块与高阶状态

边界 0=b_0<...<b_K=T 确定 K 个块。第 k 个块向量为有效帧均值：

q_k=(1/|I_k|) sum_(i∈I_k) x_i。

使用 discovery/180s 三组平衡抽样拟合声学标准化与 32 维 PCA，再以 16 个 MiniBatch K-means 中心 c_m 定义共享结构状态：

s_k=argmin_m ||P D^(-1)(q_k-mu)-c_m||_2^2。

本轮用 {feature_summary['quality']['structure_blocks']:,} 个宏观块，得到 16 个实际被使用的结构原型；validation 与 holdout 不参与原型拟合。

### 2.4 有向图与 Path Homology

对状态路径 (s_0,...,s_K)，转移计数 C_uv=#{{k:s_k=u,s_(k+1)=v}}，出向概率

p_uv=C_uv/sum_w C_uw。

过滤图 G_tau 保留 p_uv≥tau 的非自环边，并限制每个源状态最多 top-6 条出边。阈值从 0.95 下降到 0.05 时只增边，因此形成 G_0.95⊆...⊆G_0.05。

对允许 p-路径 e_(v0...vp)，GLMY 边界算子为：

partial e_(v0...vp)=sum_(i=0)^p (-1)^i e_(v0...vhat_i...vp)。

令 A_p 为允许路径空间，Omega_p={{a∈A_p:partial a∈A_(p-1)}}，则

H_p^path(G)=ker(partial_p|Omega_p)/im(partial_(p+1)|Omega_(p+1))，beta_p=dim H_p^path(G)。

过滤图之间的包含映射诱导持久模。本文报告 H0/H1 秩不变量、持久区间、barcode 和持久图。为符合常规横轴递增表示，绘图使用 a=1-tau。

## 3. 持久过程示例

示例 `{EXAMPLE_ID}` 的结构状态路径为：{states}。

边界时间（s）：{boundaries}。

完整有向边及出向概率：{edges}。

- tau=0.95：仅 3→10、10→6，beta0=1、beta1=0；
- tau=0.60：加入 6→3，形成 6→3→10→6 的有向一维类，beta1 从 0 变为 1；
- tau=0.30：加入 6→10，新增的允许 2-路径边界使该 H1 类死亡，beta1 回到 0。

因此 H1 在阈值坐标中出生于 tau=0.60、死亡于 tau=0.30、寿命为 0.30；在递增坐标 a=1-tau 中是区间 [0.40,0.70)。H0 从 tau=0.95 起一直存活到观察终点，属于右删失区间。

## 4. 全量重分析结果

- 片段-视图数：{summary['segment_views']:,}（4 视角 × 1,600 片段），零失败；
- 结构视角：1,600 个片段，宏观块 {feature_summary['quality']['structure_blocks']:,} 个；
- 所有视图非零 H1 片段-视图：{summary['h1_nonzero_segment_views']:,}；
- validation/180s 的结构 H1 非零数：Classical {nonzero['classical']}/{counts['classical']}，Focus {nonzero['focus']}/{counts['focus']}，Pop {nonzero['pop']}/{counts['pop']}；
- 四视角 Mahalanobis PERMANOVA：pseudo-F={primary_permanova['pseudo_f']:.3f}，p={primary_permanova['p_value']:.3g}，n={int(primary_permanova['n_tracks'])}，有效维数={int(primary_permanova['effective_dimensions'])}；
- 80 个预设视图-指标检验中 57 个通过 FDR q≤0.10；
- 仅拓扑分类基线 validation Macro-F1={primary_classification.loc['topology','macro_f1']:.3f}（95% CI {primary_classification.loc['topology','macro_f1_ci_low']:.3f}–{primary_classification.loc['topology','macro_f1_ci_high']:.3f}），Macro-AUROC={primary_classification.loc['topology','macro_auroc_ovr']:.3f}。

结构视角通过 FDR q≤0.10 的指标：

| 指标 | Classical 中位数 | Focus 中位数 | Pop 中位数 | epsilon² | FDR q |
|---|---:|---:|---:|---:|---:|
{table_rows}

H1 指标虽有若干秩检验通过 FDR，但三组中位数均为 0，属于明显零膨胀；不能把它表述为普遍存在的有向环差异。更稳妥的结论是：结构视角主要新增了宏观自转移、状态数、边密度与边方向互惠性的组间信息；个别曲目出现可解释的 H1 生命周期。

加入结构视角后，四视角 PERMANOVA 仍显著，但 pseudo-F 由旧三视角的 3.143 变为 2.365。维数增加会改变协方差白化和伪 F 的尺度，因此不应直接把数值下降解释为“模型变差”；应同时参考结构视角的单变量效应和验证集分类结果。仅拓扑 Macro-F1 从 0.776 上升到 0.801，说明结构信息提供了小幅增量判别力。

## 5. 可视化

{figure_links}

## 6. 复现与产物

1. `python -m features.batch backfill-structure --root . --workers 6`
2. `python -m features.batch fit-states --root . --overwrite`
3. `python -m features.batch transform-states --root . --workers 6 --overwrite`
4. `python -m topology.batch model --root . --workers 6`
5. `python -m topology.batch statistics --root .`
6. `python scripts/render_structure_path_report.py`

数值结果位于 `metadata/topology_segments.csv`、`metadata/topology_filtration.csv`、`metadata/topology_filtration_sensitivity.csv` 和 `metadata/topology_statistical_tests.csv`。结构 SSM/边界/状态位于 `features/structure/`，有向图位于 `graphs/structure/`，持久结果位于 `homology/persistence/structure/` 与 `homology/persistence_sensitivity/structure/`。

## 7. 局限性

- SSM 基于声学向量相似性，边界是算法性分段，不等同于人工标注的曲式边界；
- 16 个结构原型用于跨曲目可比性，但状态编号本身没有固定音乐语义；
- 段落数较少导致 H1 零膨胀，单曲 barcode 比组中位数更适合解释结构环；
- 当前最高报告维度为 H1；更高阶路径同调需要独立的计算量与稳定性评估；
- 统计比较是观察性描述，holdout 仅含 Focus 曲目，未用于三组检验。

## 参考文献

1. Foote, J. (2000). Automatic Audio Segmentation Using a Measure of Audio Novelty. ICME.
2. Müller, M. (2015). Fundamentals of Music Processing. Springer.
3. Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S.-T. (2012). Homologies of path complexes and digraphs.
4. Chowdhury, S., & Mémoli, F. (2018). Persistent path homology of directed networks. SODA.
"""
    report = _latexify_report(report)
    path = ROOT / "docs" / "path-homology-structure-analysis.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        }
    )
    feature, graph, persistence = _example_arrays()
    topology = pd.read_csv(ROOT / "metadata" / "topology_segments.csv")
    figures = [
        plot_ssm(feature),
        plot_state_graph(graph),
        plot_filtration(graph, persistence),
        *plot_persistence(persistence),
        plot_group_summary(topology),
    ]
    report = write_report(feature, graph, persistence, figures)
    print(report)
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
