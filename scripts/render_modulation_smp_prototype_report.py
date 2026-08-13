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
import pandas as pd
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "modulation_smp_prototype_path_homology"
REPORT = ROOT / "docs" / "path-homology-modulation-smp-prototype-analysis.md"
SUMMARY = ROOT / "metadata" / "modulation_smp_prototype_summary.json"
FEATURES = ROOT / "metadata" / "modulation_smp_prototype_features.csv"
TOPOLOGY = ROOT / "metadata" / "modulation_smp_prototype_topology_segments.csv"
FILTRATION = ROOT / "metadata" / "modulation_smp_prototype_topology_filtration_sensitivity.csv"
TESTS = ROOT / "metadata" / "modulation_smp_prototype_statistical_tests.csv"
PAIRWISE = ROOT / "metadata" / "modulation_smp_prototype_pairwise_tests.csv"
STATE_COUNTS = (8, 10, 12)
COLORS = {"classical": "#4472C4", "focus": "#ED7D31"}
LABELS = {"classical": "Classical", "focus": "Open Focus"}
METRICS = (
    "vertex_count",
    "edge_count",
    "edge_density",
    "reciprocity",
    "self_transition_ratio",
    "transition_entropy",
    "path_entropy",
    "directed_recurrence",
    "h0_betti_auc",
    "h0_betti_mean",
    "h0_betti_max",
    "h0_interval_count",
    "h0_observed_persistence",
    "h0_censored_count",
    "h1_betti_auc",
    "h1_betti_mean",
    "h1_betti_max",
    "h1_interval_count",
    "h1_observed_persistence",
    "h1_censored_count",
)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(figure: plt.Figure, stem: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png


def load_context() -> dict[str, Any]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    features = pd.read_csv(FEATURES)
    topology = pd.read_csv(TOPOLOGY)
    example_id = summary["mechanism_example"]["segment_id"]
    feature_row = features[features.segment_id == example_id].iloc[0]
    topology_row = topology[
        (topology.state_count == 10) & (topology.segment_id == example_id)
    ].iloc[0]
    return {
        "summary": summary,
        "features": features,
        "topology": topology,
        "filtration": pd.read_csv(FILTRATION),
        "tests": pd.read_csv(TESTS),
        "pairwise": pd.read_csv(PAIRWISE),
        "feature_row": feature_row,
        "topology_row": topology_row,
        "states": read_npz(ROOT / str(feature_row.feature_relative_path)),
        "smp": read_npz(ROOT / str(feature_row.source_modulation_relative_path)),
        "graph": read_npz(ROOT / str(topology_row.graph_relative_path)),
        "persistence": read_npz(ROOT / str(topology_row.sensitivity_persistence_relative_path)),
        "models": {
            k: read_npz(ROOT / "features" / "models" / f"modulation_smp_proto_k{k}.npz")
            for k in STATE_COUNTS
        },
    }


def effect_table(context: dict[str, Any], state_count: int, analysis_set: str) -> pd.DataFrame:
    pair = context["pairwise"]
    tests = context["tests"]
    pair = pair[(pair.state_count == state_count) & (pair.analysis_set == analysis_set)].copy()
    pair["r"] = -pair["rank_biserial_a_minus_b"].astype(float)
    pair["ci95_low"] = -pair["rank_biserial_ci95_high"].astype(float)
    pair["ci95_high"] = -pair["rank_biserial_ci95_low"].astype(float)
    omnibus = tests[(tests.state_count == state_count) & (tests.analysis_set == analysis_set)][
        ["metric", "p_fdr_bh", "classical_median", "focus_median"]
    ]
    return pair[["metric", "r", "ci95_low", "ci95_high"]].merge(
        omnibus,
        on="metric",
    )


def plot_prototypes(context: dict[str, Any]) -> Path:
    frequencies = context["states"]["frequencies"].astype(float)
    figure, axes = plt.subplots(3, 1, figsize=(12.0, 10.5), constrained_layout=True)
    for axis, state_count in zip(axes, STATE_COUNTS, strict=True):
        model = context["models"][state_count]
        profiles = model["prototype_spectra"].astype(float)
        image = axis.imshow(
            profiles,
            origin="lower",
            aspect="auto",
            extent=(frequencies[0], frequencies[-1], -0.5, state_count - 0.5),
            cmap="magma",
            interpolation="nearest",
        )
        centroids = model["spectral_centroids_hz"].astype(float)
        axis.scatter(centroids, np.arange(state_count), marker="|", s=25, color="cyan")
        axis.set(
            yticks=range(state_count),
            yticklabels=[f"P{i:02d}" for i in range(state_count)],
            ylabel=f"K={state_count}",
        )
        figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Modulation frequency (Hz)")
    figure.suptitle("Discovery-fitted shared SMP prototype codebooks", fontsize=14)
    return save(figure, "modulation_smp_prototypes")


def plot_example(context: dict[str, Any]) -> Path:
    smp = context["smp"]
    spectrum = smp["spectrum"].astype(float)
    times = smp["times"].astype(float)
    frequencies = smp["frequencies"].astype(float)
    states = context["states"]["states_k10"].astype(int)
    centroids = context["models"][10]["spectral_centroids_hz"].astype(float)
    path = np.where(states >= 0, centroids[np.maximum(states, 0)], np.nan)
    figure = plt.figure(figsize=(12.0, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 1.2))
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(
        np.log10(spectrum.T + 1e-7),
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], frequencies[0], frequencies[-1]),
        cmap="viridis",
    )
    axis.plot(times, path, color="white", lw=1.2, label="prototype spectral centroid")
    axis.set(ylabel="Modulation frequency (Hz)", title="Mechanism-example SMP")
    axis.legend(frameon=False)
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01, label="log10 normalized energy")
    state_axis = figure.add_subplot(grid[1], sharex=axis)
    state_axis.step(times, states, where="mid", color="#28536B")
    state_axis.set(
        yticks=range(10),
        yticklabels=[f"P{i:02d}" for i in range(10)],
        xlabel="Time (s)",
        ylabel="State",
    )
    state_axis.grid(alpha=0.2)
    return save(figure, "modulation_smp_example_trajectory")


def positions(vertices: np.ndarray) -> dict[int, np.ndarray]:
    ordered = sorted(int(value) for value in vertices)
    return {
        state: np.array(
            [
                np.cos(np.pi / 2 - 2 * np.pi * index / len(ordered)),
                np.sin(np.pi / 2 - 2 * np.pi * index / len(ordered)),
            ]
        )
        for index, state in enumerate(ordered)
    }


def draw_graph(
    axis: plt.Axes, graph: dict[str, np.ndarray], threshold: float, label_edges: bool
) -> None:
    vertices = graph["vertices"].astype(int)
    node_positions = positions(vertices)
    for source, target, weight in zip(
        graph["edge_source"].astype(int),
        graph["edge_target"].astype(int),
        graph["edge_weight"].astype(float),
        strict=True,
    ):
        if weight < threshold:
            continue
        start, stop = node_positions[source], node_positions[target]
        axis.add_patch(
            FancyArrowPatch(
                start,
                stop,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.7 + 2.5 * weight,
                color="#46647A",
                alpha=0.35 + 0.6 * weight,
                shrinkA=20,
                shrinkB=20,
                connectionstyle="arc3,rad=0.10",
            )
        )
        if label_edges:
            midpoint = (start + stop) / 2
            axis.text(
                midpoint[0],
                midpoint[1],
                f"{weight:.2f}",
                fontsize=7,
                ha="center",
                va="center",
                bbox={"fc": "white", "ec": "none", "alpha": 0.75},
            )
    cmap = plt.get_cmap("tab10")
    for state in vertices:
        point = node_positions[int(state)]
        axis.scatter(*point, s=700, color=cmap(int(state) % 10), edgecolor="#263B4A", zorder=5)
        axis.text(*point, f"P{int(state):02d}", ha="center", va="center", fontsize=8, zorder=6)
    axis.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal")
    axis.axis("off")


def plot_graph_and_filtration(context: dict[str, Any]) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    draw_graph(axis, context["graph"], 0.0, True)
    axis.set_title("K=10 directed SMP prototype graph")
    graph_path = save(figure, "modulation_smp_directed_graph")

    persistence = context["persistence"]
    finite = np.flatnonzero(
        (persistence["interval_dimension"].astype(int) == 1)
        & ~persistence["interval_censored"].astype(bool)
    )
    best = int(finite[np.argmax(persistence["interval_lifetime"][finite])])
    birth = float(persistence["interval_birth_threshold"][best])
    death = float(persistence["interval_death_threshold"][best])
    thresholds = persistence["thresholds"].astype(float)
    previous = thresholds[thresholds > birth]
    selected = (float(np.min(previous)), birth, death)
    labels = ("before H1 birth", "H1 birth", "H1 death")
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    for axis, threshold, label in zip(axes, selected, labels, strict=True):
        index = int(np.argmin(np.abs(thresholds - threshold)))
        draw_graph(axis, context["graph"], threshold, False)
        axis.set_title(
            f"tau={threshold:.2f}: {label}\n"
            f"edges={int(persistence['edge_count'][index])}, "
            f"beta0={int(persistence['h0_betti'][index])}, "
            f"beta1={int(persistence['h1_betti'][index])}",
            fontsize=10,
        )
    figure.suptitle("Descending-threshold path-homology filtration", fontsize=13)
    return graph_path, save(figure, "modulation_smp_filtration")


def expanded_intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    terminal = 1.0 - float(np.min(persistence["thresholds"]))
    rows = []
    for index, dimension in enumerate(persistence["interval_dimension"].astype(int)):
        censored = bool(persistence["interval_censored"][index])
        birth = 1.0 - float(persistence["interval_birth_threshold"][index])
        death = (
            terminal if censored else 1.0 - float(persistence["interval_death_threshold"][index])
        )
        for _ in range(int(persistence["interval_multiplicity"][index])):
            rows.append(
                {
                    "dimension": int(dimension),
                    "birth": birth,
                    "death": death,
                    "censored": censored,
                }
            )
    return rows


def plot_persistence(context: dict[str, Any]) -> Path:
    intervals = expanded_intervals(context["persistence"])
    terminal = 1.0 - float(np.min(context["persistence"]["thresholds"]))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    axes[0].plot([0, terminal], [0, terminal], ls="--", color="#777777")
    for dimension, marker, color in ((0, "o", "#4472C4"), (1, "^", "#C44E52")):
        selected = [row for row in intervals if row["dimension"] == dimension]
        axes[0].scatter(
            [row["birth"] for row in selected],
            [row["death"] for row in selected],
            marker=marker,
            color=color,
            label=f"H{dimension}",
        )
    axes[0].set(
        xlabel="Birth a = 1 - tau",
        ylabel="Death a = 1 - tau",
        title="Persistence diagram",
    )
    axes[0].legend(frameon=False)
    intervals.sort(key=lambda row: (row["dimension"], row["birth"], row["death"]))
    for index, row in enumerate(intervals):
        color = "#4472C4" if row["dimension"] == 0 else "#C44E52"
        axes[1].hlines(index, row["birth"], row["death"], color=color, lw=3)
        axes[1].plot(row["death"], index, marker="o" if row["censored"] else "|", color=color)
    axes[1].set(
        xlabel="Filtration coordinate a = 1 - tau",
        ylabel="Interval index",
        title="Barcode",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("K=10 persistent path homology of the mechanism example", fontsize=13)
    return save(figure, "modulation_smp_persistence")


def plot_group_and_betti(context: dict[str, Any]) -> tuple[Path, Path]:
    data = context["topology"]
    data = data[
        (data.state_count == 10)
        & (data.split == "validation")
        & np.isclose(data.scale_seconds, 180.0)
    ]
    selected_metrics = (
        "vertex_count",
        "edge_count",
        "edge_density",
        "reciprocity",
        "h0_betti_auc",
        "h1_betti_max",
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)
    for axis, metric in zip(axes.flat, selected_metrics, strict=True):
        values = [
            data.loc[data.group == group, metric].to_numpy(float)
            for group in ("classical", "focus")
        ]
        boxes = axis.boxplot(
            values,
            tick_labels=[LABELS["classical"], LABELS["focus"]],
            patch_artist=True,
            showfliers=False,
        )
        for patch, group in zip(boxes["boxes"], ("classical", "focus"), strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.55)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("K=10 validation/180s group distributions", fontsize=13)
    group_path = save(figure, "modulation_smp_group_distributions")

    filtration = context["filtration"]
    filtration = filtration[
        (filtration.state_count == 10)
        & (filtration.split == "validation")
        & np.isclose(filtration.scale_seconds, 180.0)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for axis, dimension in zip(axes, (0, 1), strict=True):
        metric = f"h{dimension}_betti"
        for group in ("classical", "focus"):
            stats = (
                filtration[filtration.group == group]
                .groupby("threshold")[metric]
                .agg(["mean", "sem"])
                .sort_index(ascending=False)
            )
            x = stats.index.to_numpy(float)
            mean = stats["mean"].to_numpy(float)
            sem = stats["sem"].fillna(0.0).to_numpy(float)
            axis.plot(x, mean, marker="o", color=COLORS[group], label=LABELS[group])
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.16)
        axis.invert_xaxis()
        axis.set(
            xlabel="Threshold tau",
            ylabel=f"Mean beta{dimension}",
            title=f"H{dimension} Betti curve",
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("K=10 validation/180s sensitivity filtration, mean ± SE", fontsize=13)
    return group_path, save(figure, "modulation_smp_betti_curves")


def plot_effects_and_duration(context: dict[str, Any]) -> tuple[Path, Path]:
    data = effect_table(context, 10, "primary_validation_180")
    data = data.sort_values("r")
    y = np.arange(len(data))
    significant = data.p_fdr_bh.astype(float) <= 0.05
    figure, axis = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    axis.axvline(0.0, color="#777777", lw=1.0)
    axis.hlines(y, 0.0, data.r, color="#AAB4BD", lw=1.2)
    axis.errorbar(
        data.r,
        y,
        xerr=np.vstack([data.r - data.ci95_low, data.ci95_high - data.r]),
        fmt="none",
        ecolor="#526773",
        elinewidth=1.1,
        capsize=2.5,
        zorder=2,
    )
    axis.scatter(
        data.loc[~significant, "r"],
        y[~significant.to_numpy()],
        facecolors="white",
        edgecolors="#6F7F8C",
        s=48,
        label="q > 0.05",
        zorder=3,
    )
    axis.scatter(
        data.loc[significant, "r"],
        y[significant.to_numpy()],
        color="#28536B",
        s=54,
        label="BH-FDR q <= 0.05",
        zorder=3,
    )
    axis.set_yticks(y, data.metric)
    axis.set(
        xlim=(-1.02, 1.02),
        xlabel="Rank-biserial effect (Open Focus - Classical), bootstrap 95% CI",
        title="Modulation SMP K=10 validation/180 s effect directions",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    effect_path = save(figure, "modulation_smp_effect_sizes")

    first = effect_table(context, 10, "primary_validation_180").set_index("metric")
    second = effect_table(context, 10, "sensitivity_validation_300").set_index("metric")
    effects = pd.DataFrame(
        {
            "primary_validation_180": first["r"],
            "sensitivity_validation_300": second["r"],
        }
    )
    qvalues = pd.DataFrame(
        {
            "primary_validation_180": first["p_fdr_bh"],
            "sensitivity_validation_300": second["p_fdr_bh"],
        }
    )
    x = effects["primary_validation_180"].astype(float)
    y = effects["sensitivity_validation_300"].astype(float)
    stable = (
        (qvalues["primary_validation_180"].astype(float) <= 0.05)
        & (qvalues["sensitivity_validation_300"].astype(float) <= 0.05)
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
    if "edge_density" in effects.index:
        axis.annotate(
            "edge_density",
            (x.loc["edge_density"], y.loc["edge_density"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    h1_unstable = [metric for metric in effects.index[~stable] if metric.startswith("h1_")]
    if h1_unstable:
        anchor_x = float(x.loc[h1_unstable].mean())
        anchor_y = float(y.loc[h1_unstable].mean())
        axis.annotate(
            f"H1 descriptors ({len(h1_unstable)})\ncluster near zero; not stable",
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
        title="Cross-duration direction and effect stability",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    return effect_path, save(figure, "modulation_smp_duration_stability")


def plot_k_sensitivity(context: dict[str, Any]) -> Path:
    topology = context["topology"]
    summary = context["summary"]
    validation = topology[
        (topology.split == "validation") & np.isclose(topology.scale_seconds, 180.0)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    for group, marker in (("classical", "o"), ("focus", "s")):
        coverage = [
            validation[
                (validation.state_count == k) & (validation.group == group)
            ].state_coverage.median()
            for k in STATE_COUNTS
        ]
        density = [
            validation[
                (validation.state_count == k) & (validation.group == group)
            ].edge_density.median()
            for k in STATE_COUNTS
        ]
        axes[0].plot(
            STATE_COUNTS, coverage, marker=marker, color=COLORS[group], label=LABELS[group]
        )
        axes[1].plot(STATE_COUNTS, density, marker=marker, color=COLORS[group])
    axes[0].set(title="State coverage", ylabel="Median observed / K")
    axes[0].legend(frameon=False)
    axes[1].set(title="Observed-vertex density", ylabel="Median directed edge density")
    primary = [summary["models"][str(k)]["primary_fdr_discoveries"] for k in STATE_COUNTS]
    stable = [summary["models"][str(k)]["stable_same_direction_discoveries"] for k in STATE_COUNTS]
    x = np.arange(3)
    axes[2].bar(x - 0.18, primary, 0.36, color="#4472C4", label="180s")
    axes[2].bar(x + 0.18, stable, 0.36, color="#70AD47", label="stable at 300s")
    axes[2].set(
        xticks=x,
        xticklabels=[f"K={k}" for k in STATE_COUNTS],
        title="FDR findings",
        ylabel="Metrics out of 20",
    )
    axes[2].legend(frameon=False)
    for axis in axes[:2]:
        axis.set(xticks=STATE_COUNTS, xlabel="Prototype states K")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Representation sensitivity without significance-driven K selection", fontsize=13
    )
    return save(figure, "modulation_smp_k_sensitivity")


def fmt(value: float) -> str:
    return f"{value:.2e}" if 0 < abs(value) < 0.0005 else f"{value:.3f}"


def write_report(context: dict[str, Any], stems: tuple[str, ...]) -> Path:
    summary = context["summary"]
    first = effect_table(context, 10, "primary_validation_180")
    second = effect_table(context, 10, "sensitivity_validation_300")[
        ["metric", "r", "p_fdr_bh"]
    ].rename(columns={"r": "r300", "p_fdr_bh": "q300"})
    results = first.merge(second, on="metric").set_index("metric").loc[list(METRICS)].reset_index()
    result_rows = "\n".join(
        f"| {row.metric} | {fmt(row.classical_median)} | {fmt(row.focus_median)} | "
        f"{fmt(row.r)} | {fmt(row.p_fdr_bh)} | {fmt(row.r300)} | {fmt(row.q300)} |"
        for row in results.itertuples(index=False)
    )
    k_rows = "\n".join(
        f"| {k} | {summary['models'][str(k)]['role']} | "
        f"{summary['models'][str(k)]['primary_fdr_discoveries']} | "
        f"{summary['models'][str(k)]['duration_fdr_discoveries']} | "
        f"{summary['models'][str(k)]['stable_same_direction_discoveries']} | "
        f"{summary['models'][str(k)]['validation_180_diagnostics']['median_observed_states']:.1f}/{k} | "
        f"{summary['models'][str(k)]['validation_180_diagnostics']['median_retained_edge_ratio']:.3f} |"
        for k in STATE_COUNTS
    )
    h1_rows = "\n".join(
        f"| {k} | "
        f"{summary['models'][str(k)]['validation_180_h1_counts']['classical']['primary_nonzero']}/60 | "
        f"{summary['models'][str(k)]['validation_180_h1_counts']['focus']['primary_nonzero']}/60 | "
        f"{summary['models'][str(k)]['validation_180_h1_counts']['classical']['sensitivity_nonzero']}/60 | "
        f"{summary['models'][str(k)]['validation_180_h1_counts']['focus']['sensitivity_nonzero']}/60 |"
        for k in STATE_COUNTS
    )
    figures = "\n\n".join(
        f"![{stem}](../runs/modulation_smp_prototype_path_homology/{stem}.png)\n\n"
        f"[SVG](../runs/modulation_smp_prototype_path_homology/{stem}.svg)"
        for stem in stems
    )
    main = summary["models"]["10"]
    example = summary["mechanism_example"]
    report = rf"""# 调制视角共享SMP原型 Path Homology：完整重分析报告

生成日期：{date.today().isoformat()}。Jamendo Open Focus 300首与Classical 300首，每首含180 s与300 s视图；处理1,200个源片段、3,600个片段乘状态规模图，失败0。主模型固定为 $K=10$，$K=8,12$只用于表示敏感性，不按显著性选择状态数。

> 证据边界：模型只在平衡的discovery/180 s窗口上拟合；validation/180 s为主分析，validation/300 s是同曲目时长敏感性。方案在既有holdout打开后提出，因此属于探索性验证，不能更新旧holdout gate，也不能称为冻结外部确认。

## 1. 结论摘要

- $K=10$ 的20项预设指标中，180 s有 **{main["primary_fdr_discoveries"]}项**通过BH-FDR $q\le0.05$；其中 **{main["stable_same_direction_discoveries"]}项**在300 s同方向且仍通过FDR：状态数、边数、边密度与互惠性。
- Open Focus观察到更多原型状态和更多有向边，但相对图更稀疏、互惠边比例更低。这说明SMP谱形转移覆盖更广、连接更选择性，不是音乐质量、专注效果或因果证据。
- **$H_1$组间差异不受支持。** $K=10$主阈值下非零$H_1$为Classical 2/60、Focus 3/60；阈值扩展到0.05后为4/60与7/60。六项$H_1$指标均未通过180 s FDR。
- $K=8,10,12$的180 s发现数为3、5、6，跨时长稳定数为3、4、4。中位状态覆盖从$4/8$、$5/10$降至$5/12$，不能因发现数更多就偏好$K=12$。
- 相比旧三状态模型，本模型保留完整SMP谱形并出现少量有限$H_1$区间，但证据仍主要来自普通有向图组织与$H_0$。

## 2. SMP共享原型模型

对mel子带能量包络 $x_b[n]$，在4 s窗、2 s步长上计算

$$
P_t(f_m)=\sum_b\left|\sum_n w[n]x_b[n+tH]e^{{-i2\pi f_mn/f_s}}\right|^2,\qquad
\widetilde P_t(f_m)=\frac{{P_t(f_m)}}{{\sum_{{0.5\le f\le45}}P_t(f)}}.
$$

保留0.5–45 Hz的178维相对SMP。先作Hellinger平方根映射，再作discovery拟合的稳健标准化：

$$
h_{{tj}}=\sqrt{{\widetilde P_t(f_j)}},\qquad
z_{{tj}}=
\frac{{h_{{tj}}-\operatorname{{median}}_D(h_{{\cdot j}})}}{{Q_{{0.75,D}}(h_{{\cdot j}})-Q_{{0.25,D}}(h_{{\cdot j}})}}.
$$

共享PCA-32为 $y_t=W_{{32}}(z_t-\mu_D)$，累计解释方差为{summary["pca_explained_variance"]:.3f}。Classical有14,715个、Focus有16,822个可用discovery窗口；各平衡抽14,715个。对$K\in\{{8,10,12\}}$分别求

$$
\min_{{c_1,\ldots,c_K}}\sum_t\min_k\|y_t-c_k\|_2^2,\qquad
s_t=\arg\min_k\|y_t-c_k\|_2^2.
$$

三个码本共享同一预处理；原型按原始SMP频谱质心由低到高排序。

## 3. 有向图与Path Homology

$$
C_{{uv}}=|\{{t:s_t=u,\ s_{{t+1}}=v\}}|,\qquad
p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}},\qquad
G_\tau=(V,\{{(u,v):u\ne v,\ p_{{uv}}\ge\tau\}}).
$$

无效窗口两侧不跨越连边；自转移不进入图。每个源状态保留至多6条非自环边。主阈值为$\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}$，0.05–0.40只作敏感性。

对允许的正则有向路径空间$A_p$，

$$
\partial e_{{v_0\ldots v_p}}=\sum_{{i=0}}^p(-1)^ie_{{v_0\ldots\widehat{{v_i}}\ldots v_p}},
$$

$$
\Omega_p=A_p\cap\partial^{{-1}}(A_{{p-1}}),\qquad
H_p^{{\mathrm{{path}}}}(G)=
\frac{{\ker(\partial_p:\Omega_p\to\Omega_{{p-1}})}}
{{\operatorname{{im}}(\partial_{{p+1}}:\Omega_{{p+1}}\to\Omega_p)}}.
$$

令$a=1-\tau$，持久秩不变量为

$$
\rho_p(a_i,a_j)=\operatorname{{rank}}\operatorname{{im}}
[H_p(G_{{a_i}})\to H_p(G_{{a_j}})],\qquad a_i\le a_j.
$$

实现只计算$H_0/H_1$，不作$H_2$声明。

## 4. 统计设计

每个$K$独立形成20指标family，做Kruskal–Wallis检验及BH-FDR，阈值$q\le0.05$。方向效应为

$$
r_{{F-C}}=\frac{{2U_F}}{{n_Fn_C}}-1.
$$

300 s统一绘制$r_{{180}}$对$r_{{300}}$，它不是独立复制。

## 5. 数值结果

### 5.1 K敏感性

| K | 角色 | 180 s FDR发现 | 300 s FDR发现 | 跨时长稳定 | 中位观察状态 | 中位保留边比例 |
|---:|---|---:|---:|---:|---:|---:|
{k_rows}

保留边比例为$|E|/[K(K-1)]$。它随$K$下降，反映单曲欠覆盖及冻结top-6/高阈值共同作用。

### 5.2 K=10完整20指标

| 指标 | Classical中位数 | Focus中位数 | $r_{{180}}$ | $q_{{180}}$ | $r_{{300}}$ | $q_{{300}}$ |
|---|---:|---:|---:|---:|---:|---:|
{result_rows}

### 5.3 H1稀疏性

| K | Classical主阈值非零 | Focus主阈值非零 | Classical扩展阈值非零 | Focus扩展阈值非零 |
|---:|---:|---:|---:|---:|
{h1_rows}

机制示例为 {example["segment_id"]}。有限$H_1$区间出生于$\tau={example["birth_threshold"]:.2f}$、死亡于$\tau={example["death_threshold"]:.2f}$，寿命{example["lifetime"]:.2f}。它只解释计算机制，不是组间证据。

## 6. 可视化

{figures}

## 7. 解释与局限

1. 共享SMP原型保留谱峰位置、宽度和多峰形状，比三状态总强度更有表达力。
2. Focus覆盖更多SMP原型，但图相对更稀疏、互惠性更低；该模式在$K=8,10,12$及180/300 s间大体一致。
3. 共享原型提高了环的可表达性，但六项$H_1$组间检验不显著，不能宣称稳定$H_1$差异。
4. top-6和0.50–0.95来自旧冻结图族，未针对10状态图调参；这避免结果驱动调参，但可能压低分散转移概率。
5. K=12的中位覆盖仅5/12，不建议继续无约束增加$K$。
6. 本方案在旧holdout打开后提出，不能并入旧确认性fingerprint；升级前必须冻结当前哈希并用新数据验证。
7. 新分支未覆盖modulation_tertile。旧模型解释总调制强度级别转移；本模型解释完整SMP谱形原型转移。

## 8. 复现与审计

PowerShell命令：

    $env:PYTHONPATH = "packages/pyglmy/src;src"
    .\.venv\Scripts\python.exe scripts\run_modulation_smp_prototype_analysis.py
    .\.venv\Scripts\python.exe scripts\render_modulation_smp_prototype_report.py

模型集合SHA-256：{summary["model_set_sha256"]}。共享变换SHA-256：{summary["shared_model_sha256"]}。数值表位于metadata/modulation_smp_prototype系列文件，模型位于features/models/modulation_smp系列文件，PNG/SVG及哈希清单位于runs/modulation_smp_prototype_path_homology。
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def write_manifest(stems: tuple[str, ...]) -> Path:
    payload = {
        "generated_at": date.today().isoformat(),
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


def main() -> int:
    plt.rcParams.update(
        {"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False}
    )
    context = load_context()
    stems = (
        "modulation_smp_prototypes",
        "modulation_smp_example_trajectory",
        "modulation_smp_directed_graph",
        "modulation_smp_filtration",
        "modulation_smp_persistence",
        "modulation_smp_group_distributions",
        "modulation_smp_betti_curves",
        "modulation_smp_effect_sizes",
        "modulation_smp_duration_stability",
        "modulation_smp_k_sensitivity",
    )
    outputs = [
        plot_prototypes(context),
        plot_example(context),
        *plot_graph_and_filtration(context),
        plot_persistence(context),
        *plot_group_and_betti(context),
        *plot_effects_and_duration(context),
        plot_k_sensitivity(context),
    ]
    report = write_report(context, stems)
    manifest = write_manifest(stems)
    print(report.relative_to(ROOT).as_posix())
    print(manifest.relative_to(ROOT).as_posix())
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
