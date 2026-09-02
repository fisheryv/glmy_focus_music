# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import render_modulation_smp_prototype_report as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "modulation_smp_k10_path_homology_open"
REPORT = ROOT / "docs" / "path-homology-modulation-smp-k10-analysis.md"
SUMMARY_PATH = ROOT / "metadata" / "modulation_smp_prototype_summary.json"
FEATURES_PATH = ROOT / "metadata" / "modulation_smp_prototype_features.csv"
TOPOLOGY_PATH = ROOT / "metadata" / "modulation_smp_prototype_topology_segments.csv"
TESTS_PATH = ROOT / "metadata" / "modulation_smp_prototype_statistical_tests.csv"
PAIRWISE_PATH = ROOT / "metadata" / "modulation_smp_prototype_pairwise_tests.csv"
K = 10
FDR_Q = 0.05
COLORS = {"classical": "#4472C4", "focus": "#ED7D31"}
GROUP_LABELS = {"classical": "Classical", "focus": "Open Focus"}


def save(figure: plt.Figure, stem: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_codebook(context: dict[str, Any]) -> Path:
    model = context["models"][K]
    frequencies = context["states"]["frequencies"].astype(float)
    profiles = model["prototype_spectra"].astype(float)
    centroids = model["spectral_centroids_hz"].astype(float)
    counts = model["training_state_counts"].astype(int)
    shares = counts / counts.sum()

    figure = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.2, 1.8))
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(
        profiles,
        origin="lower",
        aspect="auto",
        extent=(frequencies[0], frequencies[-1], -0.5, K - 0.5),
        cmap="magma",
        interpolation="nearest",
    )
    axis.scatter(
        centroids,
        np.arange(K),
        color="cyan",
        marker="|",
        s=32,
        label="prototype spectral centroid",
    )
    axis.set(
        yticks=range(K),
        yticklabels=[f"P{i:02d}" for i in range(K)],
        xlabel="Modulation frequency (Hz)",
        ylabel="Frozen prototype state",
        title="Discovery-only K=10 SMP prototype codebook",
    )
    axis.legend(frameon=False, loc="upper right")
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01, label="Mean normalized SMP energy")

    axis = figure.add_subplot(grid[1])
    bars = axis.bar(np.arange(K), shares, color=plt.get_cmap("tab10")(np.arange(K)))
    axis.set(
        xticks=range(K),
        xticklabels=[f"P{i:02d}" for i in range(K)],
        xlabel="Frozen prototype state",
        ylabel="Balanced discovery share",
        title="Training-state occupancy",
    )
    axis.grid(axis="y", alpha=0.2)
    for bar, share in zip(bars, shares, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{share:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.suptitle("Fixed K=10 shared SMP representation", fontsize=14)
    return save(figure, "modulation_smp_k10_codebook")


def plot_smp_ssm(context: dict[str, Any]) -> Path:
    smp = context["smp"]
    spectrum = smp["spectrum"].astype(float)
    valid = smp["valid"].astype(bool)
    times = smp["times"].astype(float)
    frequencies = smp["frequencies"].astype(float)
    states = context["states"]["states_k10"].astype(int)
    hellinger = np.sqrt(np.maximum(spectrum, 0.0))
    norms = np.linalg.norm(hellinger, axis=1)
    similarity = hellinger @ hellinger.T
    similarity /= np.maximum(norms[:, None] * norms[None, :], np.finfo(float).eps)
    similarity[~valid, :] = np.nan
    similarity[:, ~valid] = np.nan

    figure = plt.figure(figsize=(10.5, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(3.2, 0.9, 5.0))
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(
        np.log10(spectrum.T + 1e-7),
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], frequencies[0], frequencies[-1]),
        cmap="viridis",
    )
    axis.set(
        title=f"Normalized SMP: {context['summary']['mechanism_example']['segment_id']}",
        ylabel="Modulation frequency (Hz)",
    )
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01, label="log10 normalized energy")

    state_axis = figure.add_subplot(grid[1], sharex=axis)
    state_axis.step(times, states, where="mid", color="#28536B", lw=1.1)
    state_axis.set(
        yticks=range(K),
        yticklabels=[f"P{i:02d}" for i in range(K)],
        xlabel="Time (s)",
        ylabel="State",
    )
    state_axis.grid(alpha=0.2)

    ssm_axis = figure.add_subplot(grid[2])
    ssm = ssm_axis.imshow(
        similarity,
        origin="lower",
        aspect="equal",
        extent=(times[0], times[-1], times[0], times[-1]),
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ssm_axis.set(
        title="Hellinger SMP self-similarity matrix",
        xlabel="Time (s)",
        ylabel="Time (s)",
    )
    figure.colorbar(ssm, ax=ssm_axis, fraction=0.03, pad=0.02, label="Cosine similarity")
    return save(figure, "modulation_smp_k10_ssm")


def plot_persistence_diagram(context: dict[str, Any]) -> Path:
    persistence = context["persistence"]
    intervals = base.expanded_intervals(persistence)
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    axis.plot([0, end], [0, end], ls="--", color="#777777", lw=1)
    for dimension, marker, color in ((0, "o", "#4472C4"), (1, "^", "#C44E52")):
        selected = [item for item in intervals if item["dimension"] == dimension]
        axis.scatter(
            [item["birth"] for item in selected],
            [item["death"] for item in selected],
            marker=marker,
            s=65,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=f"H{dimension}",
        )
    censored = [item for item in intervals if item["censored"]]
    axis.scatter(
        [item["birth"] for item in censored],
        [item["death"] for item in censored],
        s=100,
        facecolors="none",
        edgecolors="#111111",
        linewidth=1.2,
        label="right-censored",
    )
    axis.set(
        xlim=(-0.02, end + 0.04),
        ylim=(-0.02, end + 0.04),
        xlabel="Birth a = 1 - tau",
        ylabel="Death a = 1 - tau",
        title=f"K=10 persistence diagram: {context['summary']['mechanism_example']['segment_id']}",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return save(figure, "modulation_smp_k10_persistence_diagram")


def plot_barcode(context: dict[str, Any]) -> Path:
    persistence = context["persistence"]
    intervals = base.expanded_intervals(persistence)
    intervals.sort(key=lambda item: (item["dimension"], item["birth"], item["death"]))
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for row, item in enumerate(intervals):
        color = "#4472C4" if item["dimension"] == 0 else "#C44E52"
        start, stop = float(item["birth"]), float(item["death"])
        axis.hlines(row, start, stop, color=color, lw=3)
        axis.plot(start, row, marker="|", color=color, ms=8)
        axis.plot(
            stop,
            row,
            marker="o" if item["censored"] else "|",
            color=color,
            ms=6 if item["censored"] else 8,
            markerfacecolor="white" if item["censored"] else color,
        )
    axis.set(
        xlim=(-0.02, end + 0.03),
        xlabel="Filtration coordinate a = 1 - tau",
        ylabel="Interval index",
        title=f"K=10 persistent path barcode: {context['summary']['mechanism_example']['segment_id']}",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.02,
        0.96,
        "blue: H0    red: H1    open circle: censored",
        transform=axis.transAxes,
        va="top",
    )
    return save(figure, "modulation_smp_k10_barcode")


def plot_group_summary(context: dict[str, Any]) -> Path:
    topology = context["topology"]
    data = topology[
        (topology.state_count == K)
        & (topology.split == "validation")
        & np.isclose(topology.scale_seconds, 180.0)
    ]
    metrics = (
        ("vertex_count", "Observed states"),
        ("edge_count", "Directed edges"),
        ("edge_density", "Edge density"),
        ("reciprocity", "Reciprocity"),
        ("h0_betti_auc", "H0 Betti AUC"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [
            data.loc[data.group == group, metric].to_numpy(float)
            for group in ("classical", "focus")
        ]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(boxes["boxes"], ("classical", "focus"), strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        axis.set_xticks(
            (1, 2),
            (GROUP_LABELS["classical"], GROUP_LABELS["focus"]),
            rotation=25,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Fixed K=10 group comparison (validation, 180 s)", fontsize=13)
    return save(figure, "modulation_smp_k10_group_summary")


def plot_betti_curves(context: dict[str, Any]) -> Path:
    filtration = context["filtration"]
    data = filtration[
        (filtration.state_count == K)
        & (filtration.split == "validation")
        & np.isclose(filtration.scale_seconds, 180.0)
    ].copy()
    data["a"] = 1.0 - data.threshold
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("h0_betti", "h1_betti"),
        ("Mean beta0", "Mean beta1"),
        strict=True,
    ):
        for group in ("classical", "focus"):
            stats = (
                data[data.group == group]
                .groupby("a")[metric]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("a")
            )
            x = stats.a.to_numpy(float)
            mean = stats["mean"].to_numpy(float)
            sem = stats["sem"].fillna(0.0).to_numpy(float)
            axis.plot(
                x,
                mean,
                marker="o",
                ms=3.5,
                lw=1.7,
                color=COLORS[group],
                label=GROUP_LABELS[group],
            )
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.14)
        axis.set(title=title, xlabel="Filtration coordinate a = 1 - tau", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Fixed K=10 Betti curves (validation, 180 s; mean +/- SEM)", fontsize=13)
    return save(figure, "modulation_smp_betti_curves")


def figure_manifest(stems: tuple[str, ...]) -> Path:
    payload = {
        "generated_at": date.today().isoformat(),
        "fixed_state_count": K,
        "figures": {
            stem: {
                "png_sha256": sha256(OUTPUT / f"{stem}.png"),
                "svg_sha256": sha256(OUTPUT / f"{stem}.svg"),
            }
            for stem in stems
        },
    }
    path = OUTPUT / "figure_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_report(context: dict[str, Any], stems: tuple[str, ...]) -> Path:
    summary = context["summary"]
    tests = context["tests"]
    pairwise = context["pairwise"]
    features = context["features"]
    primary = tests[
        (tests.state_count == K) & (tests.analysis_set == "primary_validation_180")
    ].sort_values(["p_fdr_bh", "metric"])
    duration = tests[
        (tests.state_count == K) & (tests.analysis_set == "sensitivity_validation_300")
    ].set_index("metric")
    pair = pairwise[
        (pairwise.state_count == K)
        & (pairwise.analysis_set == "primary_validation_180")
        & (pairwise.group_a == "classical")
        & (pairwise.group_b == "focus")
    ].sort_values(["p_fdr_bh", "metric"])
    metric_rows = "\n".join(
        "| {metric} | {classical:.3f} | {focus:.3f} | {effect:.3f} | {q180:.3g} | {q300:.3g} |".format(
            metric=row.metric,
            classical=float(row.classical_median),
            focus=float(row.focus_median),
            effect=float(row.epsilon_squared),
            q180=float(row.p_fdr_bh),
            q300=float(duration.loc[row.metric, "p_fdr_bh"]),
        )
        for row in primary.itertuples(index=False)
    )
    pair_rows = "\n".join(
        f"| {row.metric} | {-float(row.rank_biserial_a_minus_b):.3f} | "
        f"[{-float(row.rank_biserial_ci95_high):.3f}, "
        f"{-float(row.rank_biserial_ci95_low):.3f}] | {float(row.p_fdr_bh):.3g} |"
        for row in pair[pair.p_fdr_bh <= FDR_Q].itertuples(index=False)
    )
    model = context["models"][K]
    counts = model["training_state_counts"].astype(int)
    centroids = model["spectral_centroids_hz"].astype(float)
    shares = counts / counts.sum()
    state_rows = "\n".join(
        f"| P{state:02d} | {centroids[state]:.3f} | {counts[state]:,} | {shares[state]:.3%} |"
        for state in range(K)
    )
    figures = "\n\n".join(
        f"![{stem}](../runs/modulation_smp_k10_path_homology_open/{stem}.png)\n\n"
        f"[SVG](../runs/modulation_smp_k10_path_homology_open/{stem}.svg)"
        for stem in stems
    )
    validation_features = features[
        (features.split == "validation") & np.isclose(features.scale_seconds, 180.0)
    ].copy()
    validation_features["valid_share"] = (
        validation_features.valid_windows / validation_features.windows
    )
    valid_medians = validation_features.groupby("group").valid_share.median().to_dict()
    main = summary["models"]["10"]
    h1 = main["validation_180_h1_counts"]
    primary_h1 = primary[primary.metric.str.startswith("h1_")]
    duration_h1 = duration.loc[[metric for metric in duration.index if metric.startswith("h1_")]]
    h1_primary_discoveries = int((primary_h1.p_fdr_bh <= FDR_Q).sum())
    h1_duration_discoveries = int((duration_h1.p_fdr_bh <= FDR_Q).sum())
    example = summary["mechanism_example"]
    codebook_sha = summary["codebook_sha256"]["10"]
    topology_sha = summary["artifact_sha256"][
        "metadata/modulation_smp_prototype_topology_segments.csv"
    ]
    report = rf"""# Path Homology modulation_smp_k10：Focus–Classical 调制视角完整分析

生成日期：{date.today().isoformat()}。本文使用当前规范数据集Jamendo Open Focus 300首与Classical 300首。两组均分为discovery 195、validation 60、holdout 45；每首有180 s与300 s两个片段，共1,200个片段。状态数固定为$K=10$。主推断为validation/180 s（n=120，每组60），validation/300 s仅作时长敏感性。该模型在既有holdout打开后提出，因此本报告是探索性验证，不把旧holdout倒写为当前模型的确认结果。统计阈值统一为BH-FDR $q\le0.05$。

## 1. 结论摘要

- 仅用discovery/180 s的Classical与Focus各14,715个有效SMP窗口拟合共享变换与固定$K=10$码本；完成1,200/1,200个片段的状态转换及1,200个$K=10$有向图与持续Path Homology，失败0。码本SHA-256为{codebook_sha}。
- 20个预设指标中，validation/180 s有{main["primary_fdr_discoveries"]}个通过BH-FDR，validation/300 s有{main["duration_fdr_discoveries"]}个通过；其中{main["stable_same_direction_discoveries"]}个在两种时长均显著且方向一致。
- 稳定差异为：Open Focus观察状态更多、边数更多，但在已观察节点上的边密度更低、互惠性更低。这描述调制谱形路径覆盖与连接组织，不表示音乐质量高低。
- $H_1$主阈值非零率为Classical {h1["classical"]["primary_nonzero"]}/60、Open Focus {h1["focus"]["primary_nonzero"]}/60；预设$H_1$指标在180 s有{h1_primary_discoveries}个、300 s有{h1_duration_discoveries}个通过FDR。因此当前不支持稳定的组间$H_1$差异。
- 结论属于观察性声学结构比较；不支持疗效、认知提升、生成质量或因果结论。

## 2. 表示与冻结设计

对mel子带能量包络$x_b[n]$，在4 s窗、2 s步长上计算归一化调制频谱：

$$
P_t(f_m)=\sum_b\left|\sum_n w[n]x_b[n+tH]e^{{-i2\pi f_mn/f_s}}\right|^2,\qquad
\widetilde P_t(f_m)=\frac{{P_t(f_m)}}{{\sum_{{0.5\le f\le45}}P_t(f)}}.
$$

保留0.5–45 Hz的178维相对SMP。先作Hellinger映射与discovery拟合的中位数/IQR标准化：

$$
h_{{tj}}=\sqrt{{\widetilde P_t(f_j)}},\qquad
z_{{tj}}=
\frac{{h_{{tj}}-\operatorname{{median}}_D(h_{{\cdot j}})}}
{{Q_{{0.75,D}}(h_{{\cdot j}})-Q_{{0.25,D}}(h_{{\cdot j}})}}.
$$

共享PCA-32为$y_t=W_{{32}}(z_t-\mu_D)$，累计解释方差为{summary["pca_explained_variance"]:.3f}。固定MiniBatch K-means状态数$K=10$：

$$
s_t=\arg\min_{{v\in\{{0,\ldots,9\}}}}\|y_t-\boldsymbol\mu_v\|_2^2.
$$

原型按原始SMP频谱质心从低到高编号。validation/180 s有效窗口比例中位数为Classical {valid_medians["classical"]:.1%}、Open Focus {valid_medians["focus"]:.1%}。无效窗口记为缺失，不建立额外状态，也不跨缺失位置连接转移。

状态数在本报告前固定为10；下表仅描述discovery平衡训练样本，不用于事后增减$K$：

| 状态 | SMP频谱质心（Hz） | 训练窗口 | 训练占比 |
|---|---:|---:|---:|
{state_rows}

训练占比范围为{shares.min():.3%}–{shares.max():.3%}，说明码本存在明显占用不均衡；这属于表示局限，不能通过删除低占用状态来重新优化当前结果。

## 3. 有向图与持续 Path Homology

相邻有效状态定义转移计数与条件概率：

$$
C_{{uv}}=|\{{t:s_t=u,s_{{t+1}}=v\}}|,\qquad
p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}}.
$$

每个源状态最多保留top-6非自环边。主阈值固定为$\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}$；扩展至0.05的网格只用于敏感性和机制图。过滤图为

$$
G_\tau=(V,\{{(u,v):u\ne v,\ p_{{uv}}\ge\tau\}}).
$$

对允许路径空间使用GLMY边界与路径同调：

$$
\partial e_{{v_0\ldots v_p}}=\sum_i(-1)^i e_{{v_0\ldots\widehat{{v_i}}\ldots v_p}},\qquad
\Omega_p=A_p\cap\partial^{{-1}}(A_{{p-1}}),
$$

$$
H_p^{{path}}(G)=
\frac{{\ker(\partial_p|_{{\Omega_p}})}}
{{\operatorname{{im}}(\partial_{{p+1}}|_{{\Omega_{{p+1}}}})}}.
$$

其中$A_p$由图中允许的有向$p$-路径张成，$\beta_p=\dim H_p^{{path}}$。阈值下降时边逐步加入，得到秩不变量

$$
\rho_p(\tau_i,\tau_j)=
\operatorname{{rank}}\operatorname{{im}}
\left[H_p(G_{{\tau_i}})\to H_p(G_{{\tau_j}})\right],
\qquad \tau_i\ge\tau_j.
$$

条形码和持续图使用$a=1-\tau$；生产流程只报告$H_0/H_1$，不作$H_2$声明。

## 4. 可视化

示例{example["segment_id"]}按固定说明性规则选出：优先选择Open Focus validation/180 s中仅含一个有限$H_1$区间的片段，再按lifetime与segment ID确定性排序。该区间在$\tau={float(example["birth_threshold"]):.2f}$出生、$\tau={float(example["death_threshold"]):.2f}$死亡；它不代表组中心，也不参与假设检验。SSM仅作SMP谱形诊断，主图直接由相邻状态转移构造。

{figures}

## 5. 组间结果

| 指标 | Classical中位数 | Open Focus中位数 | $\epsilon^2$ | 180 s BH-FDR $q$ | 300 s BH-FDR $q$ |
|---|---:|---:|---:|---:|---:|
{metric_rows}

Open Focus与Classical在主尺度通过独立两两FDR的指标：

| 指标 | rank-biserial（Open Focus−Classical） | bootstrap 95% CI | BH-FDR $q$ |
|---|---:|---:|---:|
{pair_rows}

### 5.1 解读

1. Open Focus在同一10状态码本中访问更多状态并形成更多绝对边，但边密度和互惠性更低；这意味着覆盖更广而连接更选择性，不等于“拓扑更好”。
2. 只有同时通过validation/180 s FDR，并在300 s同方向且再次显著的指标，才视为跨时长稳定差异；本轮共{main["stable_same_direction_discoveries"]}项。
3. h0_censored_count在180 s显著但300 s不显著，且两组180 s中位数同为1；它是分布形状差异，不应写成中位数位移。
4. 主尺度$H_1$发生率低且六个$H_1$指标均未通过FDR，不能改写为“普遍存在稳定调制环”。

### 5.2 统计原理

每个指标在validation/180 s做两组Kruskal–Wallis omnibus检验，并以$\epsilon^2=(H-k+1)/(N-k)$报告秩效应量；独立两两表使用Mann–Whitney $U$与rank-biserial。20个指标在预先定义的单一$K=10$ family内分别做Benjamini–Hochberg校正，判定要求$q\le0.05$。300 s重复同一套检验，但只解释为同曲目的时长敏感性。

## 6. holdout兼容性与不可确认边界

- 当前$K=10$共享SMP模型是在旧holdout打开后提出；旧holdout gate锁定的是modulation_tertile三状态分支，不包含modulation_smp_k10模型、码本哈希或指标方向。
- 虽然全部holdout片段已按同一码本转换以保证产物完整，本报告不对其做组间检验，也不引用旧modulation holdout数值作为$K=10$确认。
- 当前拓扑总表SHA-256为{topology_sha}。它证明本次结果可审计，不证明与旧门控兼容。
- 若要获得确认性证据，应冻结当前共享变换、$K=10$码本、top-6、阈值、20指标family与哈希，并在未参与本次设计的新数据上验证。

## 7. 证据层级与局限

- **探索性主分析：** validation/180 s、固定$K=10$、top-6、主阈值0.50–0.95、20指标omnibus及pairwise FDR，统一要求$q\le0.05$。
- **时长敏感性：** validation/300 s；只报告跨时长显著性和方向，不称为独立复制。
- **说明性：** discovery码本占用、扩展至0.05的过滤、SMP SSM与单曲birth/death图。
- **不具备：** 当前模型的冻结holdout确认或外部独立复制。
- **不支持：** 稳定组间$H_1$差异、$H_2$发现、注意力/治疗/认知/生成质量或因果结论。
- **表示局限：** PCA-32只保留约82.1%的稳健标准化方差，码本占用不均衡；每片段约88或148个SMP窗口，低占用原型及$H_1$均可能零膨胀。
- Path Homology只使用状态ID、边方向和转移概率；原型频谱质心用于解释，不直接进入链复形。

## 8. 复现入口与产物

PowerShell：

    $env:PYTHONPATH = "packages/pyglmy/src;src"
    .\.venv\Scripts\python.exe scripts\run_modulation_analysis.py
    .\.venv\Scripts\python.exe scripts\render_modulation_smp_k10_report.py

主要数值文件为metadata/modulation_smp_prototype_features.csv、metadata/modulation_smp_prototype_topology_segments.csv、metadata/modulation_smp_prototype_topology_filtration.csv、metadata/modulation_smp_prototype_topology_filtration_sensitivity.csv、metadata/modulation_smp_prototype_statistical_tests.csv、metadata/modulation_smp_prototype_pairwise_tests.csv与metadata/modulation_smp_prototype_summary.json。$K=10$模型为features/models/modulation_smp_proto_k10.npz/json；图和哈希清单位于runs/modulation_smp_k10_path_homology_open。
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def main() -> int:
    plt.rcParams.update(
        {"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False}
    )
    base.OUTPUT = OUTPUT
    context = base.load_context()
    stems = (
        "modulation_smp_k10_codebook",
        "modulation_smp_k10_ssm",
        "modulation_smp_directed_graph",
        "modulation_smp_filtration",
        "modulation_smp_k10_persistence_diagram",
        "modulation_smp_k10_barcode",
        "modulation_smp_k10_group_summary",
        "modulation_smp_betti_curves",
        "modulation_smp_effect_sizes",
        "modulation_smp_duration_stability",
    )
    outputs = [
        plot_codebook(context),
        plot_smp_ssm(context),
        *base.plot_graph_and_filtration(context),
        plot_persistence_diagram(context),
        plot_barcode(context),
        plot_group_summary(context),
        plot_betti_curves(context),
        *base.plot_effects_and_duration(context),
    ]
    report = write_report(context, stems)
    manifest = figure_manifest(stems)
    print(report.relative_to(ROOT).as_posix())
    print(manifest.relative_to(ROOT).as_posix())
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
