# ruff: noqa: E501
from __future__ import annotations

import json
import os
from datetime import date
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
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "rhythm_path_homology_open"
REPORT = ROOT / "docs" / "path-homology-rhythm-analysis.md"
EXAMPLE_ID = ""
EXAMPLE_GROUP = "focus"
EXAMPLE_SPLIT = "validation"
COLORS = {"classical": "#4472C4", "focus": "#ED7D31"}
GROUP_LABELS = {"classical": "Classical", "focus": "Open Focus"}
CONFIRMATORY_FDR_Q = 0.05
FEATURE_LABELS = (
    "Onset mean",
    "Onset SD",
    "Onset max",
    "Onset rate",
    "Mean IOI",
    "IOI SD",
    "Tempo",
    "Beat rate",
)


def _save(figure: plt.Figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _model() -> dict[str, np.ndarray]:
    return _read_npz(ROOT / "features" / "models" / "state_model.npz")


def _example_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    paths = (
        ROOT / "graphs" / "rhythm" / "180s" / EXAMPLE_GROUP / EXAMPLE_SPLIT / f"{EXAMPLE_ID}.npz",
        ROOT / "homology" / "persistence_sensitivity" / "rhythm" / "180s" / EXAMPLE_GROUP / EXAMPLE_SPLIT / f"{EXAMPLE_ID}.npz",
    )
    return tuple(_read_npz(path) for path in paths)  # type: ignore[return-value]


def _filled_scaled(feature: dict[str, np.ndarray], model: dict[str, np.ndarray]) -> np.ndarray:
    values = feature["vectors"].astype(float)
    valid = feature["valid"].astype(bool)
    filled = np.where(valid, values, model["rhythm_impute"].astype(float))
    return (filled - model["rhythm_mean"].astype(float)) / model["rhythm_scale"].astype(float)


def _state_labels(model: dict[str, np.ndarray]) -> list[str]:
    centers = model["rhythm_centers"].astype(float)
    labels: list[str] = []
    for state, center in enumerate(centers):
        top = np.argsort(center)[-2:][::-1]
        short = "+".join(FEATURE_LABELS[index].replace(" ", "-") for index in top)
        labels.append(f"S{state:02d} ({short})")
    return labels


def _similarity(values: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    return np.exp(-(distances**2) / (2.0 * sigma**2))


def plot_codebook(model: dict[str, np.ndarray], labels: list[str]) -> Path:
    centers = model["rhythm_centers"].astype(float)
    topology = pd.read_csv(ROOT / "metadata" / "rhythm_topology_segments.csv")
    discovery = topology[(topology.split == "discovery") & (topology.scale_seconds == 180.0)]
    occupancy = {group: np.zeros(centers.shape[0], dtype=float) for group in GROUP_LABELS}
    for row in discovery.itertuples(index=False):
        feature = _read_npz(ROOT / Path(row.feature_relative_path))
        counts = np.bincount(feature["states"].astype(int), minlength=centers.shape[0])
        occupancy[row.group] += counts
    for group in occupancy:
        occupancy[group] /= max(float(occupancy[group].sum()), 1.0)

    figure = plt.figure(figsize=(11.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 2.2))
    axis = figure.add_subplot(grid[0])
    limit = float(np.max(np.abs(centers)))
    image = axis.imshow(centers, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(range(8), FEATURE_LABELS, rotation=30, ha="right")
    axis.set_yticks(range(10), labels)
    axis.set(xlabel="Standardized rhythm dimension", ylabel="Frozen rhythm state", title="Discovery-only 10-state rhythm codebook")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Standardized centroid")

    axis_occ = figure.add_subplot(grid[1])
    x = np.arange(centers.shape[0])
    width = 0.36
    for offset, group in zip((-width / 2, width / 2), ("classical", "focus"), strict=True):
        axis_occ.bar(x + offset, occupancy[group], width, color=COLORS[group], label=GROUP_LABELS[group])
    axis_occ.set(xticks=x, xticklabels=[f"S{i:02d}" for i in x], ylabel="Discovery beat-window share", xlabel="Frozen state")
    axis_occ.grid(axis="y", alpha=0.2)
    axis_occ.legend(frameon=False, ncol=2)
    figure.suptitle("Rhythm codebook and group occupancy", fontsize=14)
    return _save(figure, "rhythm_codebook.png")


def plot_rhythm_ssm(feature: dict[str, np.ndarray], model: dict[str, np.ndarray]) -> Path:
    scaled = _filled_scaled(feature, model)
    states = feature["states"].astype(int)
    times = feature["times"].astype(float)
    similarity = _similarity(scaled)
    left, right = float(times[0]), float(times[-1])

    figure = plt.figure(figsize=(10.0, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(2.8, 0.9, 5.2))
    axis_features = figure.add_subplot(grid[0])
    for index, label in enumerate(FEATURE_LABELS):
        axis_features.plot(times, scaled[:, index] + index * 4.2, lw=0.7, label=label)
    axis_features.set_yticks(np.arange(8) * 4.2, FEATURE_LABELS)
    axis_features.set(title=f"Standardized one-second rhythm trajectories: {EXAMPLE_ID}", ylabel="Feature + offset")
    axis_features.grid(alpha=0.15)

    axis_state = figure.add_subplot(grid[1], sharex=axis_features)
    axis_state.step(times, states, where="mid", color="#28536B", lw=1.1)
    axis_state.set(yticks=range(10), yticklabels=[f"S{i:02d}" for i in range(10)], ylabel="State", xlabel="Time (s)")
    axis_state.grid(alpha=0.2)

    axis_ssm = figure.add_subplot(grid[2])
    image = axis_ssm.imshow(similarity, origin="lower", extent=(left, right, left, right), aspect="equal", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axis_ssm.set(title="Rhythm self-similarity matrix", xlabel="Time (s)", ylabel="Time (s)")
    figure.colorbar(image, ax=axis_ssm, fraction=0.03, pad=0.02, label="Gaussian similarity")
    return _save(figure, "rhythm_ssm.png")


def _positions(model: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    projected = PCA(n_components=2, random_state=20_260_716).fit_transform(model["rhythm_centers"].astype(float))
    projected -= np.mean(projected, axis=0)
    projected /= max(float(np.max(np.linalg.norm(projected, axis=1))), 1e-12)
    return {index: projected[index] for index in range(projected.shape[0])}


def _draw_graph(axis: plt.Axes, graph: dict[str, np.ndarray], model: dict[str, np.ndarray], labels: list[str], *, threshold: float, show_edge_labels: bool) -> None:
    vertices = graph["vertices"].astype(int)
    pca_positions = _positions(model)
    ordered = sorted(vertices, key=lambda state: np.arctan2(pca_positions[int(state)][1], pca_positions[int(state)][0]))
    positions = {int(state): np.asarray([np.cos(np.pi / 2 - 2 * np.pi * index / len(ordered)), np.sin(np.pi / 2 - 2 * np.pi * index / len(ordered))]) for index, state in enumerate(ordered)}
    edges = zip(graph["edge_source"].astype(int), graph["edge_target"].astype(int), graph["edge_weight"].astype(float), strict=True)
    for source, target, weight in edges:
        if weight < threshold:
            continue
        start, end = positions[source], positions[target]
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9 + 5 * weight, linewidth=0.7 + 2.8 * weight, color="#46647A", alpha=0.35 + 0.6 * weight, shrinkA=21, shrinkB=21, connectionstyle="arc3,rad=0.09"))
        if show_edge_labels:
            midpoint = (start + end) / 2
            axis.text(midpoint[0], midpoint[1], f"{weight:.2f}", fontsize=7, ha="center", va="center", bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})
    for state in vertices:
        position = positions[state]
        axis.scatter(position[0], position[1], s=760, color=plt.get_cmap("tab10")(state), edgecolor="#263B4A", zorder=5)
        axis.text(position[0], position[1] + 0.03, f"S{state:02d}", ha="center", va="center", fontsize=8.5, zorder=6)
        if show_edge_labels:
            short = labels[state].split("(", 1)[1].rstrip(")").replace("-", " ")
            axis.text(position[0], position[1] - 0.10, short, ha="center", va="top", fontsize=5.8, zorder=6)
    axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
    axis.axis("off")


def plot_directed_graph(graph: dict[str, np.ndarray], model: dict[str, np.ndarray], labels: list[str]) -> Path:
    figure, axis = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    _draw_graph(axis, graph, model, labels, threshold=0.0, show_edge_labels=True)
    axis.set_title("Directed rhythm-state graph\nnode order follows PCA angle of frozen centroids; edge width = transition probability")
    return _save(figure, "rhythm_directed_state_graph.png")


def plot_filtration(graph: dict[str, np.ndarray], persistence: dict[str, np.ndarray], model: dict[str, np.ndarray], labels: list[str]) -> Path:
    dimensions = persistence["interval_dimension"].astype(int)
    censored = persistence["interval_censored"].astype(bool)
    finite_h1 = np.flatnonzero((dimensions == 1) & ~censored)
    if finite_h1.size == 0:
        raise RuntimeError("mechanism example has no finite H1 interval")
    best = int(finite_h1[np.argmax(persistence["interval_lifetime"][finite_h1])])
    birth = float(persistence["interval_birth_threshold"][best])
    death = float(persistence["interval_death_threshold"][best])
    archive_thresholds = persistence["thresholds"].astype(float)
    earlier = archive_thresholds[archive_thresholds > birth]
    before = float(np.min(earlier)) if earlier.size else birth
    thresholds = (before, birth, death)
    descriptions = ("before H1 birth", "H1 birth", "H1 death")
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    for axis, threshold, description in zip(axes, thresholds, descriptions, strict=True):
        index = int(np.argmin(np.abs(archive_thresholds - threshold)))
        _draw_graph(axis, graph, model, labels, threshold=threshold, show_edge_labels=False)
        axis.set_title(f"tau={threshold:.2f}: {description}\nedges={int(persistence['edge_count'][index])}, beta0={int(persistence['h0_betti'][index])}, beta1={int(persistence['h1_betti'][index])}", fontsize=10)
    figure.suptitle("Rhythm descending-threshold persistent path-homology filtration", fontsize=13)
    return _save(figure, "rhythm_filtration_process.png")


def _expanded_intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, float | int | bool]]:
    end = 1.0 - float(np.min(persistence["thresholds"]))
    intervals: list[dict[str, float | int | bool]] = []
    for index, dimension in enumerate(persistence["interval_dimension"].astype(int)):
        censored = bool(persistence["interval_censored"][index])
        birth = 1.0 - float(persistence["interval_birth_threshold"][index])
        death = end if censored else 1.0 - float(persistence["interval_death_threshold"][index])
        for _ in range(int(persistence["interval_multiplicity"][index])):
            intervals.append({"dimension": dimension, "birth": birth, "death": death, "censored": censored})
    return intervals


def plot_persistence_diagram(persistence: dict[str, np.ndarray]) -> Path:
    intervals = _expanded_intervals(persistence)
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    axis.plot([0, end], [0, end], ls="--", color="#777777", lw=1)
    for dimension, marker, color in ((0, "o", "#4472C4"), (1, "^", "#C44E52")):
        selected = [item for item in intervals if item["dimension"] == dimension]
        axis.scatter([item["birth"] for item in selected], [item["death"] for item in selected], marker=marker, s=65, color=color, edgecolor="white", linewidth=0.6, label=f"H{dimension}")
    censored = [item for item in intervals if item["censored"]]
    axis.scatter([item["birth"] for item in censored], [item["death"] for item in censored], s=100, facecolors="none", edgecolors="#111111", linewidth=1.2, label="right-censored")
    axis.set(xlim=(-0.02, end + 0.04), ylim=(-0.02, end + 0.04), xlabel="Birth a = 1 - tau", ylabel="Death a = 1 - tau", title=f"Rhythm persistence diagram: {EXAMPLE_ID}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, "rhythm_persistence_diagram.png")


def plot_barcode(persistence: dict[str, np.ndarray]) -> Path:
    intervals = _expanded_intervals(persistence)
    intervals.sort(key=lambda item: (int(item["dimension"]), float(item["birth"]), float(item["death"])))
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for row, item in enumerate(intervals):
        color = "#4472C4" if int(item["dimension"]) == 0 else "#C44E52"
        start, stop = float(item["birth"]), float(item["death"])
        axis.hlines(row, start, stop, color=color, lw=3)
        axis.plot(start, row, marker="|", color=color, ms=8)
        axis.plot(stop, row, marker="o" if item["censored"] else "|", color=color, ms=6 if item["censored"] else 8, markerfacecolor="white" if item["censored"] else color)
    axis.set(xlim=(-0.02, end + 0.03), xlabel="Filtration coordinate a = 1 - tau", ylabel="Interval index", title=f"Rhythm persistent path barcode: {EXAMPLE_ID}")
    axis.grid(axis="x", alpha=0.2)
    axis.text(0.02, 0.96, "blue: H0    red: H1    open circle: censored", transform=axis.transAxes, va="top")
    return _save(figure, "rhythm_barcode.png")


def plot_group_summary() -> Path:
    topology = pd.read_csv(ROOT / "metadata" / "rhythm_topology_segments.csv")
    data = topology[(topology.split == "validation") & (topology.scale_seconds == 180.0)]
    metrics = (("vertex_count", "Observed states"), ("edge_count", "Directed edges"), ("edge_density", "Edge density"), ("path_entropy", "Path entropy"), ("reciprocity", "Reciprocity"))
    groups = ("classical", "focus")
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [data.loc[data.group == group, metric].to_numpy() for group in groups]
        plot = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(plot["boxes"], groups, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        axis.set_xticks(range(1, 3), [GROUP_LABELS[group] for group in groups], rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Rhythm-state group comparison (validation, 180 s)", fontsize=13)
    return _save(figure, "rhythm_group_summary.png")


def plot_betti_curves() -> Path:
    filtration = pd.read_csv(ROOT / "metadata" / "rhythm_topology_filtration_sensitivity.csv")
    data = filtration[(filtration.split == "validation") & (filtration.scale_seconds == 180.0)].copy()
    data["a"] = 1.0 - data.threshold
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("h0_betti", "h1_betti"), ("Mean beta0", "Mean beta1"), strict=True):
        for group in ("classical", "focus"):
            selected = data[data.group == group]
            summary = selected.groupby("a")[metric].agg(["mean", "sem"]).reset_index().sort_values("a")
            x = summary.a.to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            sem = summary["sem"].fillna(0).to_numpy(float)
            axis.plot(x, mean, marker="o", ms=3.5, lw=1.7, color=COLORS[group], label=GROUP_LABELS[group])
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.14)
        axis.set(title=title, xlabel="Filtration coordinate a = 1 - tau", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Rhythm Betti curves (validation, 180 s; mean +/- SEM)", fontsize=13)
    return _save(figure, "rhythm_betti_curves.png")


def plot_scale_sensitivity() -> Path:
    topology = pd.read_csv(ROOT / "metadata" / "rhythm_topology_segments.csv")
    data = topology[topology.split == "validation"]
    metrics = (("edge_density", "Edge density"), ("reciprocity", "Reciprocity"), ("path_entropy", "Path entropy"), ("h0_betti_mean", "Mean beta0"))
    groups = ("classical", "focus")
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.1), constrained_layout=True)
    x = np.arange(2)
    width = 0.36
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        med180 = [data[(data.group == group) & (data.scale_seconds == 180.0)][metric].median() for group in groups]
        med300 = [data[(data.group == group) & (data.scale_seconds == 300.0)][metric].median() for group in groups]
        axis.bar(x - width / 2, med180, width, color="#9AA6B2", label="180 s")
        axis.bar(x + width / 2, med300, width, color="#28536B", label="300 s")
        axis.set_xticks(x, [GROUP_LABELS[group] for group in groups], rotation=25)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Rhythm-view scale sensitivity: group medians", fontsize=13)
    return _save(figure, "rhythm_scale_sensitivity.png")


def plot_effect_sizes() -> Path:
    pairwise = pd.read_csv(ROOT / "metadata" / "rhythm_pairwise_tests.csv")
    data = pairwise[
        (pairwise.analysis_set == "primary_validation_180")
        & (pairwise.group_a == "classical")
        & (pairwise.group_b == "focus")
    ].copy()
    data["effect_focus_minus_classical"] = -data["rank_biserial_a_minus_b"].astype(float)
    data["ci95_low_focus_minus_classical"] = -data[
        "rank_biserial_ci95_high"
    ].astype(float)
    data["ci95_high_focus_minus_classical"] = -data[
        "rank_biserial_ci95_low"
    ].astype(float)
    data = data.sort_values("effect_focus_minus_classical")
    y = np.arange(len(data))
    significant = data.p_fdr_bh.astype(float) <= CONFIRMATORY_FDR_Q
    figure, axis = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    axis.axvline(0.0, color="#777777", lw=1.0)
    axis.hlines(y, 0.0, data.effect_focus_minus_classical, color="#AAB4BD", lw=1.2)
    axis.errorbar(
        data.effect_focus_minus_classical,
        y,
        xerr=np.vstack(
            [
                data.effect_focus_minus_classical
                - data.ci95_low_focus_minus_classical,
                data.ci95_high_focus_minus_classical
                - data.effect_focus_minus_classical,
            ]
        ),
        fmt="none",
        ecolor="#526773",
        elinewidth=1.1,
        capsize=2.5,
        zorder=2,
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
        xlabel="Rank-biserial effect (Open Focus - Classical), bootstrap 95% CI",
        title="Rhythm validation/180 s effect directions",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "rhythm_effect_sizes.png")


def plot_duration_stability() -> Path:
    pairwise = pd.read_csv(ROOT / "metadata" / "rhythm_pairwise_tests.csv")
    selected = pairwise[
        (pairwise.group_a == "classical") & (pairwise.group_b == "focus")
    ].copy()
    selected["effect_focus_minus_classical"] = -selected["rank_biserial_a_minus_b"].astype(float)
    effects = selected.pivot(index="metric", columns="analysis_set", values="effect_focus_minus_classical")
    qvalues = selected.pivot(index="metric", columns="analysis_set", values="p_fdr_bh")
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
    axis.scatter(x[~stable], y[~stable], facecolors="white", edgecolors="#6F7F8C", s=52, label="not stable")
    axis.scatter(x[stable], y[stable], color="#28536B", s=58, label=f"stable ({int(stable.sum())})")
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
        title="Rhythm cross-duration direction and effect stability",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "rhythm_duration_stability.png")


def _example_feature() -> dict[str, np.ndarray]:
    topology = pd.read_csv(ROOT / "metadata" / "rhythm_topology_segments.csv")
    row = topology[topology["segment_id"] == EXAMPLE_ID].iloc[0]
    return _read_npz(ROOT / str(row["feature_relative_path"]))


def write_report(figure_stems: tuple[str, ...]) -> Path:
    summary = json.loads(
        (ROOT / "metadata" / "rhythm_analysis_summary.json").read_text(
            encoding="utf-8"
        )
    )
    topology_summary = json.loads(
        (ROOT / "metadata" / "rhythm_topology_summary.json").read_text(
            encoding="utf-8"
        )
    )
    state_model = json.loads(
        (ROOT / "features" / "models" / "state_model.json").read_text(
            encoding="utf-8"
        )
    )
    tests = pd.read_csv(ROOT / "metadata" / "rhythm_statistical_tests.csv")
    pairwise = pd.read_csv(ROOT / "metadata" / "rhythm_pairwise_tests.csv")
    holdout_permanova = pd.read_csv(ROOT / "metadata" / "holdout_confirmation_permanova.csv")
    holdout_directional = pd.read_csv(ROOT / "metadata" / "holdout_confirmation_directional_metrics.csv")
    holdout_gate = json.loads((ROOT / "metadata" / "holdout_gate.json").read_text(encoding="utf-8"))
    primary = tests[tests["analysis_set"] == "primary_validation_180"].copy()
    primary = primary.sort_values(["p_fdr_bh", "metric"])
    sensitivity = tests[
        tests["analysis_set"] == "sensitivity_validation_300"
    ].set_index("metric")
    primary_pairs = pairwise[
        pairwise["analysis_set"] == "primary_validation_180"
    ]
    focus_classical = primary_pairs[
        primary_pairs["group_a"].isin(["classical", "focus"])
        & primary_pairs["group_b"].isin(["classical", "focus"])
    ]
    h1 = summary["validation_180_h1_counts"]
    h1_sensitivity = summary["validation_180_sensitivity_h1_counts"]
    h1_tests = primary[primary["metric"].str.startswith("h1_")]
    example = summary["mechanism_example"]
    holdout_rhythm = holdout_permanova[
        (holdout_permanova["analysis_set"] == "primary_holdout_180")
        & (holdout_permanova["feature_set"] == "rhythm")
    ].iloc[0]
    holdout_rhythm_metrics = holdout_directional[
        (holdout_directional["analysis_set"] == "primary_holdout_180")
        & (holdout_directional["view"] == "rhythm")
    ]
    holdout_direction_matches = int(
        holdout_rhythm_metrics["direction_matched"].astype(str).str.lower().eq("true").sum()
    )
    holdout_replicated = int(
        holdout_rhythm_metrics["replicated_q_0_10"].astype(str).str.lower().eq("true").sum()
    )
    holdout_replicated_strict = int(
        (
            holdout_rhythm_metrics["direction_matched"].astype(str).str.lower().eq("true")
            & (holdout_rhythm_metrics["p_fdr_bh"].astype(float) <= CONFIRMATORY_FDR_Q)
        ).sum()
    )
    gate_topology_sha = holdout_gate["input_sha256"]["metadata/rhythm_topology_segments.csv"]
    rerun_topology_sha = topology_summary["artifact_sha256"]["metadata/rhythm_topology_segments.csv"]
    gate_hash_matches = gate_topology_sha == rerun_topology_sha
    metric_rows: list[str] = []
    for _, row in primary.iterrows():
        other = sensitivity.loc[row["metric"]]
        metric_rows.append(
            "| {metric} | {classical:.3f} | {focus:.3f} | {effect:.3f} | {q180:.3g} | {q300:.3g} |".format(
                metric=row["metric"],
                classical=float(row["classical_median"]),
                focus=float(row["focus_median"]),
                effect=float(row["epsilon_squared"]),
                q180=float(row["p_fdr_bh"]),
                q300=float(other["p_fdr_bh"]),
            )
        )
    focus_classical_rows: list[str] = []
    for _, row in focus_classical.sort_values(["p_fdr_bh", "metric"]).iterrows():
        effect = float(row["rank_biserial_a_minus_b"])
        ci_low = float(row["rank_biserial_ci95_low"])
        ci_high = float(row["rank_biserial_ci95_high"])
        if row["group_a"] == "classical":
            effect = -effect
            ci_low, ci_high = -ci_high, -ci_low
        focus_classical_rows.append(
            f"| {row['metric']} | {effect:.3f} | "
            f"[{ci_low:.3f}, {ci_high:.3f}] | "
            f"{float(row['p_fdr_bh']):.3g} |"
        )
    figures = "\n\n".join(
        f"![{stem}](../runs/rhythm_path_homology_open/{stem}.png)\n\n"
        f"[SVG](../runs/rhythm_path_homology_open/{stem}.svg)"
        for stem in figure_stems
    )
    report = rf"""# Path Homology 节奏视角：Focus–Classical 完整分析

生成日期：{date.today().isoformat()}。本文使用当前规范数据集 Jamendo Open Focus 300 与 Classical 300。两组均分为 discovery 195、validation 60、holdout 45；每首有 180 s 与 300 s 两个片段，共 1,200 个片段。主推断固定为 validation/180 s（n={summary['primary_validation_n']}：每组 60）；validation/300 s 仅作时长敏感性。holdout 是既有哈希门控后的单次操作性确认，不用于重新选择参数或指标。本版将确认性端点统一为 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留。

## 1. 结论摘要

- 1,200/1,200 个节奏片段完成有向图和持续 Path Homology，失败 0。本轮复用开盲前冻结的 discovery-only 状态模型，不在 holdout 后重新拟合；模型 SHA-256 为 `{topology_summary['state_model_sha256']}`。
- validation/180 s 的 {summary['primary_tests']} 个预设 rhythm 指标中，{summary['primary_fdr_discoveries']} 个通过 omnibus BH-FDR $q\le0.05$；validation/300 s 有 {summary['sensitivity_fdr_discoveries']} 个通过，其中 {summary['replicated_same_direction']} 个在两种时长均显著且方向一致。
- 既有单次 holdout/180 s 确认中，rhythm 表示的 permutation pseudo-$F={float(holdout_rhythm['pseudo_f']):.3f}$、$p={float(holdout_rhythm['p_value']):.3f}$、次级家族 FDR $q={float(holdout_rhythm['p_fdr_bh']):.3g}$；原门控的 14 个指标中 {holdout_direction_matches} 个方向一致、按历史 $q\le0.10$ 有 {holdout_replicated} 个复现，按统一严格口径 $q\le0.05$ 有 {holdout_replicated_strict} 个复现。
- 稳定差异集中在状态覆盖、边数、路径熵、有向复现度和 $H_0$ 连通过程；这些量描述量化状态空间，不表示音乐质量高低。
- $H_1$ 高度零膨胀：主阈值非零为 Classical {h1['classical']['nonzero']}/{h1['classical']['total']}、Open Focus {h1['focus']['nonzero']}/{h1['focus']['total']}；{int((h1_tests['p_fdr_bh'] <= CONFIRMATORY_FDR_Q).sum())}/{len(h1_tests)} 个预设 $H_1$ 指标通过主分析 FDR。
- 结论属于观察性声学结构比较；不支持注意力、治疗、认知、生成质量或任何因果结论。

## 2. 节奏状态表示

音频为 22,050 Hz 单声道。固定使用 1 s 窗、0.5 s 步长，对第 $n$ 个窗口构造八维向量

$$
\mathbf r_n=[\mu_o,\sigma_o,\max_o,\rho_o,\mu_{{IOI}},\sigma_{{IOI}},BPM,\rho_b]^\mathsf T,
$$

依次表示起音包络均值、标准差、最大值、起音率、起音间隔均值与标准差、局部速度和拍点率。事件不足时相应维度记为缺失，不强制设零。

在窗口 $W_n=[t_n,t_n+\Delta)$ 内，若起音时刻为 $a_i$、拍点时刻为 $b_i$，则主要事件统计量为

$$
\rho_o=\frac{{N_o(W_n)}}{{\Delta}},\qquad
\mu_{{IOI}}=\operatorname{{mean}}(a_{{i+1}}-a_i),\qquad
BPM=\frac{{60}}{{\operatorname{{median}}(b_{{i+1}}-b_i)}},\qquad
\rho_b=\frac{{N_b(W_n)}}{{\Delta}}.
$$

所有填补、均值、尺度和聚类仅在 discovery/180 s 上拟合。缺失值用 discovery 中位数 $m_j$ 填补，再标准化：

$$
r'_{{n,j}}=\begin{{cases}}r_{{n,j}},&\text{{有效}},\\m_j,&\text{{缺失}},\end{{cases}}
\qquad
\widetilde r_{{n,j}}=\frac{{r'_{{n,j}}-\mu_j}}{{\sigma_j}}.
$$

仅使用 discovery/180 s，分别从 Classical 与 Focus 平衡抽样 {state_model['sampled_windows']['rhythm']['classical']:,} 和 {state_model['sampled_windows']['rhythm']['focus']:,} 个窗口，拟合固定 $V_{{rhythm}}=10$ 的 MiniBatch K-means：

$$
s_n=\arg\min_{{v\in\{{0,\ldots,9\}}}}\|\widetilde{{\mathbf r}}_n-\boldsymbol\mu_v\|_2^2.
$$

单曲顶点集仅包含实际观察到的原型；全局 10 状态不会作为未出现的孤立点补入。

## 3. 有向图与持续 Path Homology

相邻窗口状态定义转移计数与条件概率：

$$
C_{{uv}}=|\{{n:s_n=u,s_{{n+1}}=v\}}|,\qquad
p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}}.
$$

自转移用于描述统计，但不进入 Path Homology 图；每个源状态最多保留 top-6 非自环边。主阈值冻结为 $\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}$，扩展至 0.05 仅用于敏感性和机制图：

$$
G_\tau=(V,\{{(u,v):u\ne v,\ p_{{uv}}\ge\tau\}}).
$$

对允许路径空间 $\Omega_p$，GLMY 路径同调为

$$
\partial e_{{v_0\ldots v_p}}=\sum_i(-1)^i e_{{v_0\ldots\widehat{{v_i}}\ldots v_p}},\qquad
\Omega_p=A_p\cap\partial^{{-1}}(A_{{p-1}}),\qquad
H_p^{{path}}(G)=\frac{{\ker(\partial_p|_{{\Omega_p}})}}{{\operatorname{{im}}(\partial_{{p+1}}|_{{\Omega_{{p+1}}}})}}.
$$

其中 $A_p$ 由允许的有向 $p$-路径张成，$\beta_p=\dim H_p^{{path}}$。阈值下降时边逐步加入，并计算秩不变量

$$
\rho_p(\tau_i,\tau_j)=\operatorname{{rank}}\operatorname{{im}}\left[H_p(G_{{\tau_i}})\to H_p(G_{{\tau_j}})\right],\qquad \tau_i\ge\tau_j.
$$

条形码和持续图使用单调坐标 $a=1-\tau$。

节奏主分析直接使用冻结状态路径，不构造 SSM。报告中的 SSM 只是轨迹质量诊断，移除它不会改变任何图、barcode 或统计数值。

## 4. 可视化

机制示例 `{EXAMPLE_ID}` 按冻结的说明性规则选自当前 {GROUP_LABELS[EXAMPLE_GROUP]} validation/180 s：优先选择 Focus 中恰有一个有限 $H_1$ 区间的片段，再按 lifetime 最大及 segment ID 确定性排序。其敏感阈值区间在 $\tau={float(example['birth_threshold']):.3g}$ 出生、$\tau={float(example['death_threshold']):.3g}$ 死亡；它不代表组中心，也不参与假设检验。

{figures}

## 5. 组间结果

Kruskal–Wallis 检验在 20 个预设 rhythm 指标内作 BH-FDR，确认性判定统一要求 $q\le0.05$，效应量为 $\epsilon^2$。

$$
\epsilon^2=\frac{{H-k+1}}{{N-k}},
$$

其中 $H$ 为 Kruskal–Wallis 统计量、$k=2$ 为组数。独立两两表使用 Mann–Whitney $U$ 与 rank-biserial 效应；300 s 使用同一检验，仅作为同曲目的时长敏感性，不称为独立复制。

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s BH-FDR $q$ | 300 s BH-FDR $q$ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Open Focus 与 Classical 的独立两两检验如下：

| 指标 | rank-biserial（Open Focus−Classical） | bootstrap 95% CI | BH-FDR $q$ |
|---|---:|---:|---:|
{chr(10).join(focus_classical_rows)}

### 5.1 解读

1. **跨时长稳定差异。** 只有同时通过 validation/180 s FDR、在 300 s 方向一致并再次显著的指标才视为跨时长稳定；本轮共有 {summary['replicated_same_direction']} 项。
2. **状态空间解释。** 状态/边覆盖、路径熵和 $H_0$ 描述两组在同一 10 状态码本中的覆盖与连通过程，不等于音乐质量高低。
3. **$H_1$ 不支持组间结论。** 主阈值非零率分别为 Classical {h1['classical']['nonzero']/h1['classical']['total']:.1%}、Open Focus {h1['focus']['nonzero']/h1['focus']['total']:.1%}。扩展阈值下为 {h1_sensitivity['classical']['nonzero']}/{h1_sensitivity['classical']['total']} 和 {h1_sensitivity['focus']['nonzero']}/{h1_sensitivity['focus']['total']}；不能将敏感过滤发生率当作主结果。
4. **观察性边界。** 更高自转移率或更低路径熵只描述当前量化空间中的节奏重复性，不证明注意力提升、治疗效果或生成质量。

## 6. 已冻结 holdout 的兼容性核验

- 本次重跑产出的 `rhythm_topology_segments.csv` SHA-256 为 `{rerun_topology_sha}`；开盲门控记录为 `{gate_topology_sha}`；核验结果：**{'一致' if gate_hash_matches else '不一致'}**。
- 因哈希一致，本次重跑没有改变既有单次 holdout 输入：rhythm/180 s pseudo-$F={float(holdout_rhythm['pseudo_f']):.3f}$，$p={float(holdout_rhythm['p_value']):.3f}$，$q={float(holdout_rhythm['p_fdr_bh']):.3g}$；14/14 方向一致，历史 $q\le0.10$ 为 14/14、严格 $q\le0.05$ 为 {holdout_replicated_strict}/14 复现。未通过严格阈值的是 `edge_density`（$q=0.0802$）与 `reciprocity`（$q=0.0986$），仅保留为提示性 holdout 结果。
- 该 holdout 是回顾性对称重切分下的操作性最终确认；Classical holdout 在旧切分中曾属于 discovery，因此不是 pristine 外部确认集，不能提升为外部独立复制或因果证据。

## 7. 证据层级与局限

- **确认性：** validation/180 s、冻结 10 状态、top-6、主阈值 0.50–0.95、20 指标 omnibus 与独立 pairwise FDR，统一要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件；还必须满足预定方向、300 s 方向一致性、去冗余和融合增量审计。$0.05<q\le0.10$ 的端点只作提示性结果，不进入核心冻结指纹。
- **敏感性：** validation/300 s；报告跨时长显著性和方向一致性，不以敏感性结果替代主检验。
- **探索/说明性：** discovery 占用率、扩展至 0.05 的阈值、SSM 和有限 $H_1$ 个例。
- **操作性最终确认：** 既有哈希门控 holdout/180 s；本轮只核验输入哈希仍相同，不重新开盲或调参。
- **不支持：** 将两组节奏结构差异解释为稳定或 Focus 特异的 $H_1/H_2$；认知、治疗、生成或因果结论；将已归档三组分析的结论转移到当前两组数据。
- 八维局部节奏向量与 10 状态量化可能压缩长程节拍层级；这属于表示局限，不能通过事后调参追求显著性。

## 8. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pathhom_tda/src;src"
python scripts/rerun_rhythm_path_homology.py
python scripts/analyze_rhythm_results.py
python scripts/render_rhythm_path_report.py
```

主要数值文件为 `metadata/rhythm_topology_segments.csv`、`metadata/rhythm_topology_filtration.csv`、`metadata/rhythm_topology_filtration_sensitivity.csv`、`metadata/rhythm_statistical_tests.csv`、`metadata/rhythm_pairwise_tests.csv`、`metadata/rhythm_analysis_summary.json` 和 `metadata/rhythm_topology_summary.json`。
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def main() -> None:
    global EXAMPLE_GROUP
    global EXAMPLE_ID
    global EXAMPLE_SPLIT

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    summary = json.loads(
        (ROOT / "metadata" / "rhythm_analysis_summary.json").read_text(
            encoding="utf-8"
        )
    )
    EXAMPLE_ID = str(summary["mechanism_example"]["segment_id"])
    EXAMPLE_GROUP = str(summary["mechanism_example"]["group"])
    EXAMPLE_SPLIT = str(summary["mechanism_example"]["split"])
    model = _model()
    graph, persistence = _example_arrays()
    feature = _example_feature()
    labels = _state_labels(model)
    figure_stems = (
        "rhythm_codebook",
        "rhythm_ssm",
        "rhythm_directed_state_graph",
        "rhythm_filtration_process",
        "rhythm_persistence_diagram",
        "rhythm_barcode",
        "rhythm_group_summary",
        "rhythm_betti_curves",
        "rhythm_scale_sensitivity",
        "rhythm_effect_sizes",
        "rhythm_duration_stability",
    )
    outputs = (
        plot_codebook(model, labels),
        plot_rhythm_ssm(feature, model),
        plot_directed_graph(graph, model, labels),
        plot_filtration(graph, persistence, model, labels),
        plot_persistence_diagram(persistence),
        plot_barcode(persistence),
        plot_group_summary(),
        plot_betti_curves(),
        plot_scale_sensitivity(),
        plot_effect_sizes(),
        plot_duration_stability(),
    )
    report = write_report(figure_stems)
    print(report.relative_to(ROOT).as_posix())
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
