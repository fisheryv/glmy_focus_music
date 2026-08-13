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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "structure_path_homology_open"
REPORT = ROOT / "docs" / "path-homology-structure-analysis.md"
SELECTION = ROOT / "metadata" / "structure_representative_selection.csv"
TESTS = ROOT / "metadata" / "structure_statistical_tests.csv"
PAIRWISE = ROOT / "metadata" / "structure_pairwise_tests.csv"
HOLDOUT_GATE = ROOT / "metadata" / "holdout_gate.json"
HOLDOUT_PERMANOVA = ROOT / "metadata" / "holdout_confirmation_permanova.csv"
HOLDOUT_DIRECTIONAL = ROOT / "metadata" / "holdout_confirmation_directional_metrics.csv"
GROUPS = ("classical", "focus")
GROUP_LABELS = {"classical": "Classical", "focus": "Open Focus"}
COLORS = {"classical": "#4472C4", "focus": "#ED7D31"}
CONFIRMATORY_FDR_Q = 0.05
PEDAGOGICAL_EXAMPLE = ""
SELECTION_METRICS = (
    "vertex_count",
    "edge_count",
    "edge_density",
    "self_transition_ratio",
    "path_entropy",
    "directed_recurrence",
    "reciprocity",
    "h0_betti_mean",
    "h1_betti_max",
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _model() -> dict[str, np.ndarray]:
    return _load_npz(ROOT / "features" / "models" / "state_model.npz")


def _example_data(
    topology: pd.DataFrame,
) -> tuple[pd.Series, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    row = _example_row(topology)
    feature = _load_npz(ROOT / str(row["feature_relative_path"]))
    graph = _load_npz(ROOT / str(row["graph_relative_path"]))
    persistence = _load_npz(ROOT / str(row["sensitivity_persistence_relative_path"]))
    return row, feature, graph, persistence


def _save(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    svg = OUTPUT / f"{stem}.svg"
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, svg


def _validation(topology: pd.DataFrame) -> pd.DataFrame:
    return topology[
        (topology["split"] == "validation")
        & np.isclose(topology["scale_seconds"].astype(float), 180.0)
    ].copy()


def select_representatives(topology: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for group in GROUPS:
        frame = _validation(topology)
        frame = frame[frame["group"] == group].copy()
        values = frame.loc[:, SELECTION_METRICS].to_numpy(float)
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median), axis=0)
        scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
        distance = np.sum(((values - median) / scale) ** 2, axis=1)
        frame["robust_medoid_distance"] = distance
        rows.append(frame.sort_values(["robust_medoid_distance", "segment_id"]).iloc[0])
    selected = pd.DataFrame(rows)
    columns = [
        "group",
        "segment_id",
        "track_id",
        "split",
        "scale_seconds",
        "robust_medoid_distance",
        *SELECTION_METRICS,
        "feature_relative_path",
        "graph_relative_path",
        "sensitivity_persistence_relative_path",
    ]
    selected.loc[:, columns].to_csv(SELECTION, index=False, encoding="utf-8")
    return selected


def _draw_graph(
    axis: plt.Axes,
    graph: dict[str, np.ndarray],
    states: np.ndarray,
    *,
    threshold: float,
    title: str,
    beta: tuple[int, int] | None = None,
) -> None:
    vertices = graph["vertices"].astype(int)
    sources = graph["edge_source"].astype(int)
    targets = graph["edge_target"].astype(int)
    weights = graph["edge_weight"].astype(float)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(vertices), endpoint=False)
    positions = {
        int(vertex): np.array([np.cos(angle), np.sin(angle)])
        for vertex, angle in zip(vertices, angles, strict=True)
    }
    active = weights >= threshold - 1e-12
    for source, target, weight in zip(
        sources[active], targets[active], weights[active], strict=True
    ):
        start, end = positions[int(source)], positions[int(target)]
        direction = end - start
        unit = direction / float(np.linalg.norm(direction))
        left, right = start + unit * 0.20, end - unit * 0.20
        reciprocal = np.any((sources == target) & (targets == source) & active)
        bend = 0.16 if reciprocal and source < target else (-0.16 if reciprocal else 0.0)
        axis.add_patch(
            FancyArrowPatch(
                left,
                right,
                arrowstyle="-|>",
                mutation_scale=13,
                connectionstyle=f"arc3,rad={bend}",
                linewidth=1.0 + 3.0 * float(weight),
                color="#4C78A8",
                alpha=0.88,
            )
        )
        midpoint = (start + end) / 2
        axis.text(
            midpoint[0],
            midpoint[1],
            f"{weight:.2f}",
            fontsize=7.5,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
        )
    counts = {int(vertex): int(np.count_nonzero(states == vertex)) for vertex in vertices}
    maximum = max(counts.values(), default=1)
    for vertex in vertices:
        point = positions[int(vertex)]
        size = 430 + 530 * counts[int(vertex)] / maximum
        axis.scatter(
            point[0],
            point[1],
            s=size,
            color="#F2C14E",
            edgecolor="#333333",
            linewidth=1.0,
            zorder=3,
        )
        axis.text(
            point[0],
            point[1],
            f"S{int(vertex)}\n(n={counts[int(vertex)]})",
            ha="center",
            va="center",
            fontsize=8.5,
            zorder=4,
        )
    subtitle = f"τ ≥ {threshold:g}"
    if beta is not None:
        subtitle += f" · β₀={beta[0]}, β₁={beta[1]}"
    axis.set_title(f"{title}\n{subtitle}", fontsize=10.5)
    axis.set_aspect("equal")
    axis.set_xlim(-1.42, 1.42)
    axis.set_ylim(-1.42, 1.42)
    axis.axis("off")


def plot_representative_graphs(selected: pd.DataFrame) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), constrained_layout=True)
    for axis, group in zip(axes, GROUPS, strict=True):
        row = selected[selected["group"] == group].iloc[0]
        graph = _load_npz(ROOT / str(row["graph_relative_path"]))
        feature = _load_npz(ROOT / str(row["feature_relative_path"]))
        _draw_graph(
            axis,
            graph,
            feature["states"].astype(int),
            threshold=0.0,
            title=f"{GROUP_LABELS[group]}\n{row['segment_id']}",
        )
    figure.suptitle("Representative directed macro-state graphs (validation, 180 s)", fontsize=14)
    return _save(figure, "structure_representative_state_graphs")


def plot_representative_ssm(selected: pd.DataFrame) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    for axis, group in zip(axes, GROUPS, strict=True):
        row = selected[selected["group"] == group].iloc[0]
        feature = _load_npz(ROOT / str(row["feature_relative_path"]))
        similarity = feature["self_similarity"].astype(float)
        boundaries = feature["boundary_times"].astype(float)
        duration = float(boundaries[-1])
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
            axis.axvline(boundary, color="white", linewidth=0.65, alpha=0.8)
            axis.axhline(boundary, color="white", linewidth=0.65, alpha=0.8)
        axis.set(
            title=f"{GROUP_LABELS[group]} · {len(boundaries) - 1} blocks",
            xlabel="time (s)",
            ylabel="time (s)",
        )
    figure.colorbar(image, ax=axes, label="similarity [0, 1]", shrink=0.78)
    figure.suptitle("Representative acoustic self-similarity matrices and boundaries", fontsize=14)
    return _save(figure, "structure_representative_ssm")


def plot_codebook(topology: pd.DataFrame) -> tuple[Path, Path]:
    centers = _model()["structure_centers"].astype(float)
    discovery = topology[
        (topology["split"] == "discovery")
        & np.isclose(topology["scale_seconds"].astype(float), 180.0)
    ]
    occupancy = {group: np.zeros(centers.shape[0], dtype=float) for group in GROUPS}
    for row in discovery.itertuples(index=False):
        feature = _load_npz(ROOT / str(row.feature_relative_path))
        occupancy[str(row.group)] += np.bincount(
            feature["states"].astype(int), minlength=centers.shape[0]
        )
    for group in GROUPS:
        occupancy[group] /= max(float(occupancy[group].sum()), 1.0)

    figure = plt.figure(figsize=(13.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 2.0))
    axis = figure.add_subplot(grid[0])
    limit = max(float(np.max(np.abs(centers))), 1e-12)
    image = axis.imshow(
        centers, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit
    )
    axis.set_xticks(range(0, centers.shape[1], 2), [f"PC{i + 1}" for i in range(0, centers.shape[1], 2)], rotation=45, ha="right")
    axis.set_yticks(range(centers.shape[0]), [f"S{i:02d}" for i in range(centers.shape[0])])
    axis.set(
        xlabel="Shared acoustic PCA coordinate",
        ylabel="Frozen structure state",
        title="Discovery-only 16-state structure codebook",
    )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="PCA centroid value")

    axis_occ = figure.add_subplot(grid[1])
    x = np.arange(centers.shape[0])
    width = 0.36
    for offset, group in zip((-width / 2, width / 2), GROUPS, strict=True):
        axis_occ.bar(
            x + offset,
            occupancy[group],
            width,
            color=COLORS[group],
            label=GROUP_LABELS[group],
        )
    axis_occ.set(
        xticks=x,
        xticklabels=[f"S{i:02d}" for i in x],
        xlabel="Frozen structure state",
        ylabel="Discovery block share",
    )
    axis_occ.grid(axis="y", alpha=0.2)
    axis_occ.legend(frameon=False, ncol=2)
    figure.suptitle("Structure codebook and group occupancy", fontsize=14)
    return _save(figure, "structure_codebook")


def plot_example_ssm(topology: pd.DataFrame) -> tuple[Path, Path]:
    _, feature, _, _ = _example_data(topology)
    similarity = feature["self_similarity"].astype(float)
    novelty = feature["novelty"].astype(float)
    boundaries = feature["boundary_times"].astype(float)
    times = feature["times"].astype(float)
    states = feature["states"].astype(int)
    duration = float(boundaries[-1])
    novelty_times = np.linspace(0.0, duration, novelty.size, endpoint=False)

    figure = plt.figure(figsize=(10.2, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(1.8, 1.0, 5.0))
    axis_novelty = figure.add_subplot(grid[0])
    axis_novelty.plot(novelty_times, novelty, color="#7A5195", linewidth=1.1)
    for boundary in boundaries[1:-1]:
        axis_novelty.axvline(boundary, color="#E45756", linewidth=0.8, alpha=0.75)
    axis_novelty.set(
        title=f"Foote novelty and selected boundaries: {PEDAGOGICAL_EXAMPLE}",
        ylabel="novelty",
        xlim=(0, duration),
    )
    axis_novelty.grid(alpha=0.2)

    axis_states = figure.add_subplot(grid[1], sharex=axis_novelty)
    axis_states.step(times, states, where="mid", color="#28536B", linewidth=1.4)
    axis_states.set(ylabel="state", xlabel="time (s)", xlim=(0, duration))
    axis_states.grid(alpha=0.2)

    axis_ssm = figure.add_subplot(grid[2])
    image = axis_ssm.imshow(
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
        axis_ssm.axvline(boundary, color="white", linewidth=0.65, alpha=0.8)
        axis_ssm.axhline(boundary, color="white", linewidth=0.65, alpha=0.8)
    axis_ssm.set(title="Acoustic self-similarity matrix", xlabel="time (s)", ylabel="time (s)")
    figure.colorbar(image, ax=axis_ssm, fraction=0.03, pad=0.02, label="similarity [0, 1]")
    return _save(figure, "structure_ssm")


def plot_example_graph(topology: pd.DataFrame) -> tuple[Path, Path]:
    _, feature, graph, _ = _example_data(topology)
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    _draw_graph(
        axis,
        graph,
        feature["states"].astype(int),
        threshold=0.0,
        title=f"Directed macro-state graph: {PEDAGOGICAL_EXAMPLE}",
    )
    return _save(figure, "structure_directed_state_graph")


def _example_row(topology: pd.DataFrame) -> pd.Series:
    return topology[topology["segment_id"] == PEDAGOGICAL_EXAMPLE].iloc[0]


def plot_filtration(topology: pd.DataFrame) -> tuple[Path, Path]:
    row = _example_row(topology)
    graph = _load_npz(ROOT / str(row["graph_relative_path"]))
    feature = _load_npz(ROOT / str(row["feature_relative_path"]))
    persistence = _load_npz(ROOT / str(row["sensitivity_persistence_relative_path"]))
    thresholds = persistence["thresholds"].astype(float)
    beta0 = persistence["h0_betti"].astype(int)
    beta1 = persistence["h1_betti"].astype(int)
    dimensions = persistence["interval_dimension"].astype(int)
    censored = persistence["interval_censored"].astype(bool)
    finite_h1 = np.flatnonzero((dimensions == 1) & ~censored)
    if finite_h1.size:
        best = int(finite_h1[np.argmax(persistence["interval_lifetime"][finite_h1])])
        birth = float(persistence["interval_birth_threshold"][best])
        death = float(persistence["interval_death_threshold"][best])
        earlier = thresholds[thresholds > birth]
        before = float(np.min(earlier)) if earlier.size else birth
        selected_thresholds = (before, birth, death)
        titles = ("Before H₁ birth", "H₁ born", "H₁ killed")
        suptitle = f"Persistent Path Homology example · {PEDAGOGICAL_EXAMPLE}"
    else:
        selected_thresholds = (0.95, 0.50, 0.05)
        titles = ("High threshold", "Primary lower bound", "Sensitivity lower bound")
        suptitle = f"Filtration without a finite H₁ interval · {PEDAGOGICAL_EXAMPLE}"
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), constrained_layout=True)
    for axis, threshold, title in zip(
        axes, selected_thresholds, titles, strict=True
    ):
        index = int(np.argmin(np.abs(thresholds - threshold)))
        _draw_graph(
            axis,
            graph,
            feature["states"].astype(int),
            threshold=threshold,
            title=title,
            beta=(int(beta0[index]), int(beta1[index])),
        )
    figure.suptitle(suptitle, fontsize=14)
    return _save(figure, "structure_filtration_process")


def _intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, float | int | bool]]:
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
        rows.append(
            {
                "dimension": int(dimension),
                "birth": 1.0 - float(birth),
                "death": 1.0 - (terminal if bool(censored) else float(death)),
                "censored": bool(censored),
                "multiplicity": int(multiplicity),
            }
        )
    return rows


def plot_persistence(topology: pd.DataFrame) -> tuple[Path, Path]:
    row = _example_row(topology)
    persistence = _load_npz(ROOT / str(row["sensitivity_persistence_relative_path"]))
    rows = _intervals(persistence)
    colors = {0: "#4C78A8", 1: "#E45756"}
    markers = {0: "o", 1: "^"}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    diagram, barcode = axes
    diagram.plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
    for dimension in (0, 1):
        subset = [item for item in rows if item["dimension"] == dimension]
        diagram.scatter(
            [float(item["birth"]) for item in subset],
            [float(item["death"]) for item in subset],
            s=80,
            marker=markers[dimension],
            color=colors[dimension],
            label=f"H{dimension}",
        )
    diagram.set(
        title="Persistence diagram",
        xlabel="birth a = 1 − τ",
        ylabel="death a = 1 − τ",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    diagram.legend(frameon=False)
    expanded: list[dict[str, float | int | bool]] = []
    for item in rows:
        expanded.extend([item] * int(item["multiplicity"]))
    for index, item in enumerate(expanded):
        dimension = int(item["dimension"])
        barcode.hlines(
            index,
            float(item["birth"]),
            float(item["death"]),
            color=colors[dimension],
            linewidth=5,
        )
        barcode.scatter(float(item["birth"]), index, color=colors[dimension], s=28)
        if bool(item["censored"]):
            barcode.scatter(
                float(item["death"]),
                index,
                facecolor="white",
                edgecolor=colors[dimension],
                s=38,
            )
        else:
            barcode.scatter(
                float(item["death"]), index, color=colors[dimension], marker="x", s=45
            )
    barcode.set_yticks(
        range(len(expanded)), [f"H{int(item['dimension'])}" for item in expanded]
    )
    barcode.set(
        title="Barcode (○ = right-censored)",
        xlabel="filtration a = 1 − τ",
        xlim=(0, 1),
        ylim=(-0.7, len(expanded) - 0.3),
    )
    figure.suptitle(f"Persistent intervals · {PEDAGOGICAL_EXAMPLE}", fontsize=14)
    return _save(figure, "structure_persistence_diagram_barcode")


def _expanded_intervals(
    persistence: dict[str, np.ndarray],
) -> list[dict[str, float | int | bool]]:
    expanded: list[dict[str, float | int | bool]] = []
    for item in _intervals(persistence):
        expanded.extend([item.copy() for _ in range(int(item["multiplicity"]))])
    return expanded


def plot_persistence_diagram(topology: pd.DataFrame) -> tuple[Path, Path]:
    _, _, _, persistence = _example_data(topology)
    rows = _expanded_intervals(persistence)
    colors = {0: "#4C78A8", 1: "#E45756"}
    markers = {0: "o", 1: "^"}
    terminal = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    axis.plot([0, terminal], [0, terminal], color="#777777", linewidth=1, linestyle="--")
    for dimension in (0, 1):
        subset = [item for item in rows if item["dimension"] == dimension]
        axis.scatter(
            [float(item["birth"]) for item in subset],
            [float(item["death"]) for item in subset],
            s=80,
            marker=markers[dimension],
            color=colors[dimension],
            edgecolor="white",
            linewidth=0.7,
            label=f"H{dimension}",
        )
    censored = [item for item in rows if bool(item["censored"])]
    axis.scatter(
        [float(item["birth"]) for item in censored],
        [float(item["death"]) for item in censored],
        s=105,
        facecolors="none",
        edgecolors="#111111",
        linewidth=1.2,
        label="right-censored",
    )
    axis.set(
        title=f"Structure persistence diagram: {PEDAGOGICAL_EXAMPLE}",
        xlabel="birth a = 1 − τ",
        ylabel="death a = 1 − τ",
        xlim=(-0.02, terminal + 0.04),
        ylim=(-0.02, terminal + 0.04),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, "structure_persistence_diagram")


def plot_barcode(topology: pd.DataFrame) -> tuple[Path, Path]:
    _, _, _, persistence = _example_data(topology)
    rows = _expanded_intervals(persistence)
    rows.sort(key=lambda item: (int(item["dimension"]), float(item["birth"]), float(item["death"])))
    colors = {0: "#4C78A8", 1: "#E45756"}
    terminal = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for index, item in enumerate(rows):
        dimension = int(item["dimension"])
        start, stop = float(item["birth"]), float(item["death"])
        axis.hlines(index, start, stop, color=colors[dimension], linewidth=4)
        axis.plot(start, index, marker="|", color=colors[dimension], markersize=8)
        axis.plot(
            stop,
            index,
            marker="o" if bool(item["censored"]) else "|",
            color=colors[dimension],
            markerfacecolor="white" if bool(item["censored"]) else colors[dimension],
            markersize=6 if bool(item["censored"]) else 8,
        )
    axis.set_yticks(range(len(rows)), [f"H{int(item['dimension'])}" for item in rows])
    axis.set(
        title=f"Structure persistent path barcode: {PEDAGOGICAL_EXAMPLE}",
        xlabel="filtration coordinate a = 1 − τ",
        ylabel="interval",
        xlim=(-0.02, terminal + 0.03),
        ylim=(-0.7, len(rows) - 0.3),
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(0.02, 0.96, "blue: H0    red: H1    open circle: censored", transform=axis.transAxes, va="top")
    return _save(figure, "structure_barcode")


def plot_group_summary(topology: pd.DataFrame) -> tuple[Path, Path]:
    frame = _validation(topology)
    metrics = (
        ("vertex_count", "Observed states"),
        ("edge_count", "Directed edges"),
        ("edge_density", "Edge density"),
        ("reciprocity", "Reciprocity"),
        ("self_transition_ratio", "Self-transition ratio"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.2, 4.2), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values = [frame.loc[frame["group"] == group, metric].to_numpy(float) for group in GROUPS]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(boxes["boxes"], GROUPS, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.65)
        axis.set_xticks(range(1, len(GROUPS) + 1), [GROUP_LABELS[group] for group in GROUPS], rotation=22)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Structure-state group comparison (validation, 180 s)", fontsize=14)
    return _save(figure, "structure_group_summary")


def plot_betti_curves() -> tuple[Path, Path]:
    filtration = pd.read_csv(
        ROOT / "metadata" / "structure_topology_filtration_sensitivity.csv"
    )
    frame = filtration[
        (filtration["split"] == "validation")
        & np.isclose(filtration["scale_seconds"].astype(float), 180.0)
    ].copy()
    frame["a"] = 1.0 - frame["threshold"].astype(float)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, label in zip(
        axes,
        ("h0_betti", "h1_betti"),
        ("Mean β₀", "Mean β₁"),
        strict=True,
    ):
        for group in GROUPS:
            selected = frame[frame["group"] == group]
            summary = selected.groupby("a")[metric].agg(["mean", "sem"]).reset_index().sort_values("a")
            x = summary["a"].to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            sem = summary["sem"].fillna(0).to_numpy(float)
            axis.plot(x, mean, marker="o", markersize=3.5, linewidth=1.7, color=COLORS[group], label=GROUP_LABELS[group])
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.14)
        axis.set(title=label, xlabel="filtration coordinate a = 1 − τ", ylabel=label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Structure Betti curves (validation, 180 s; mean ± SEM)", fontsize=14)
    return _save(figure, "structure_betti_curves")


def plot_scale_sensitivity(topology: pd.DataFrame) -> tuple[Path, Path]:
    frame = topology[topology["split"] == "validation"]
    metrics = (
        ("edge_density", "Edge density"),
        ("reciprocity", "Reciprocity"),
        ("self_transition_ratio", "Self-transition ratio"),
        ("h0_betti_mean", "Mean β₀"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(13.2, 4.1), constrained_layout=True)
    x = np.arange(len(GROUPS))
    width = 0.36
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        med180 = [frame[(frame["group"] == group) & np.isclose(frame["scale_seconds"], 180.0)][metric].median() for group in GROUPS]
        med300 = [frame[(frame["group"] == group) & np.isclose(frame["scale_seconds"], 300.0)][metric].median() for group in GROUPS]
        axis.bar(x - width / 2, med180, width, color="#9AA6B2", label="180 s")
        axis.bar(x + width / 2, med300, width, color="#28536B", label="300 s")
        axis.set_xticks(x, [GROUP_LABELS[group] for group in GROUPS], rotation=22)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Structure-view scale sensitivity: group medians", fontsize=14)
    return _save(figure, "structure_scale_sensitivity")


def plot_group_results(topology: pd.DataFrame) -> tuple[Path, Path]:
    frame = _validation(topology)
    metrics = (
        ("reciprocity", "Reciprocity"),
        ("edge_density", "Edge density"),
        ("self_transition_ratio", "Self-transition ratio"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values = [frame.loc[frame["group"] == group, metric].to_numpy(float) for group in GROUPS]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.58, showfliers=False)
        for patch, group in zip(boxes["boxes"], GROUPS, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.62)
        axis.set_xticks(range(1, len(GROUPS) + 1), [GROUP_LABELS[group] for group in GROUPS], rotation=18)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Structure-view validation distributions (180 s)", fontsize=14)
    return _save(figure, "structure_group_results")


def plot_effect_sizes() -> tuple[Path, Path]:
    pairwise = pd.read_csv(PAIRWISE)
    data = pairwise[
        (pairwise["analysis_set"] == "primary_validation_180")
        & (pairwise["group_a"] == "classical")
        & (pairwise["group_b"] == "focus")
    ].copy()
    data["effect_focus_minus_classical"] = -data[
        "rank_biserial_a_minus_b"
    ].astype(float)
    data = data.sort_values("effect_focus_minus_classical")
    y = np.arange(len(data))
    significant = data["p_fdr_bh"].astype(float) <= CONFIRMATORY_FDR_Q
    figure, axis = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    axis.axvline(0.0, color="#777777", lw=1.0)
    axis.hlines(
        y, 0.0, data["effect_focus_minus_classical"], color="#AAB4BD", lw=1.2
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
    axis.set_yticks(y, data["metric"])
    axis.set(
        xlim=(-1.02, 1.02),
        xlabel="Rank-biserial effect (Open Focus - Classical)",
        title="Structure validation/180 s effect directions",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "structure_effect_sizes")


def plot_duration_stability() -> tuple[Path, Path]:
    pairwise = pd.read_csv(PAIRWISE)
    selected = pairwise[
        (pairwise["group_a"] == "classical")
        & (pairwise["group_b"] == "focus")
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
            f"H1 descriptors ({len(h1_unstable)})\nnear zero; not stable",
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
        title="Structure cross-duration direction and effect stability",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "structure_duration_stability")


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def write_report(
    topology: pd.DataFrame,
    selected: pd.DataFrame,
    figure_stems: list[str],
) -> Path:
    tests = pd.read_csv(TESTS)
    pairwise = pd.read_csv(PAIRWISE)
    topology_summary = json.loads(
        (ROOT / "metadata" / "structure_topology_summary.json").read_text(encoding="utf-8")
    )
    analysis_summary = json.loads(
        (ROOT / "metadata" / "structure_analysis_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads(
        (ROOT / "metadata" / "feature_summary.json").read_text(encoding="utf-8")
    )
    gate = json.loads(HOLDOUT_GATE.read_text(encoding="utf-8"))
    holdout_permanova = pd.read_csv(HOLDOUT_PERMANOVA)
    holdout_directional = pd.read_csv(HOLDOUT_DIRECTIONAL)
    holdout_row = holdout_permanova[
        (holdout_permanova["analysis_set"] == "primary_holdout_180")
        & (holdout_permanova["feature_set"] == "structure")
    ].iloc[0]
    holdout_locked = holdout_directional[
        (holdout_directional["analysis_set"] == "primary_holdout_180")
        & (holdout_directional["view"] == "structure")
    ]
    holdout_strict = int(
        (
            holdout_locked["direction_matched"].astype(str).str.lower().eq("true")
            & (holdout_locked["p_fdr_bh"].astype(float) <= CONFIRMATORY_FDR_Q)
        ).sum()
    )
    gate_hash = gate["input_sha256"]["metadata/structure_topology_segments.csv"]
    current_hash = topology_summary["artifact_sha256"][
        "metadata/structure_topology_segments.csv"
    ]
    holdout_primary = topology[
        (topology["split"] == "holdout")
        & np.isclose(topology["scale_seconds"].astype(float), 180.0)
    ]
    holdout_h1 = {
        group: int(
            (
                holdout_primary.loc[holdout_primary["group"] == group, "h1_betti_max"]
                .astype(float)
                > 0
            ).sum()
        )
        for group in GROUPS
    }
    holdout_sensitivity = pd.read_csv(
        ROOT / "metadata" / "structure_topology_filtration_sensitivity.csv"
    )
    holdout_sensitivity = holdout_sensitivity[
        (holdout_sensitivity["split"] == "holdout")
        & np.isclose(holdout_sensitivity["scale_seconds"].astype(float), 180.0)
    ]
    holdout_sensitivity_max = holdout_sensitivity.groupby(
        ["segment_id", "group"], as_index=False
    )["h1_betti"].max()
    holdout_sensitivity_h1 = {
        group: int(
            (
                holdout_sensitivity_max.loc[
                    holdout_sensitivity_max["group"] == group, "h1_betti"
                ].astype(float)
                > 0
            ).sum()
        )
        for group in GROUPS
    }
    primary_frame = tests[tests["analysis_set"] == "primary_validation_180"].copy()
    primary_frame = primary_frame.sort_values(["p_fdr_bh", "metric"])
    primary = primary_frame.set_index("metric")
    primary_pairwise = pairwise[pairwise["analysis_set"] == "primary_validation_180"]
    sensitivity = tests[tests["analysis_set"] == "sensitivity_validation_300"].set_index("metric")
    significant_metrics = primary_frame.loc[
        primary_frame["p_fdr_bh"] <= CONFIRMATORY_FDR_Q, "metric"
    ].tolist()
    metric_rows = "\n".join(
        f"| {metric} | {_fmt(primary.loc[metric, 'classical_median'])} | {_fmt(primary.loc[metric, 'focus_median'])} | {_fmt(primary.loc[metric, 'epsilon_squared'])} | {_fmt(primary.loc[metric, 'p_fdr_bh'])} | {_fmt(sensitivity.loc[metric, 'p_fdr_bh'])} |"
        for metric in primary_frame["metric"]
    )
    pair_rows = primary_pairwise.sort_values(["p_fdr_bh", "metric"])
    pair_table = "\n".join(
        f"| {row.metric} | {GROUP_LABELS[row.group_a]} − {GROUP_LABELS[row.group_b]} | {_fmt(row.rank_biserial_a_minus_b)} | {_fmt(row.p_fdr_bh)} |"
        for row in pair_rows.itertuples()
    )
    representative_rows = "\n".join(
        f"| {GROUP_LABELS[row.group]} | `{row.segment_id}` | {_fmt(row.robust_medoid_distance)} | {int(row.vertex_count)} | {int(row.edge_count)} | {_fmt(row.self_transition_ratio)} | {_fmt(row.path_entropy)} |"
        for row in selected.itertuples()
    )
    h1 = analysis_summary["validation_180_h1_counts"]
    h1_sensitivity = analysis_summary["validation_180_sensitivity_h1_counts"]
    example = analysis_summary["mechanism_example"]
    if example.get("available", False):
        mechanism_text = (
            f"示例 `{example['segment_id']}` 按冻结规则选自 "
            f"{GROUP_LABELS[str(example['group'])]} validation/180 s。有限 H1 区间在 "
            f"$\\tau={float(example['birth_threshold']):.3g}$ 出生、"
            f"$\\tau={float(example['death_threshold']):.3g}$ 死亡，寿命 "
            f"{float(example['lifetime']):.3g}。它只用于解释过滤过程，不参与检验。"
        )
    else:
        mechanism_text = (
            f"validation/180 s 的扩展阈值中没有有限 H1 区间，因此不挑选循环机制个例。"
            f"图中使用冻结的 Focus 稳健代表 `{PEDAGOGICAL_EXAMPLE}` 展示 0.95、0.50、0.05 "
            f"三个阈值和其 persistence/barcode；这是一张负结果说明图，不参与检验。"
        )
    figure_markup = {
        stem: (
            f"![{stem}](../runs/structure_path_homology_open/{stem}.png)\n\n"
            f"[SVG](../runs/structure_path_homology_open/{stem}.svg)"
        )
        for stem in figure_stems
    }
    significant_names = "、".join(significant_metrics)
    report = rf"""# Path Homology 结构视角：Focus–Classical 完整分析

生成日期：{date.today().isoformat()}。切分版本：`symmetric_holdout_v2`。本文使用当前规范数据集 Open Focus 300 与 Classical 300；每组 discovery/validation/holdout 分别为 195/60/45 首。结构状态模型仅在两组 discovery/180 s 上拟合；本专项的主检验固定为 validation/180 s（n={analysis_summary['primary_validation_n']}：Classical {h1['classical']['total']}、Open Focus {h1['focus']['total']}），validation/300 s 仅作同曲目时长敏感性。holdout 是哈希门控后的单次操作性最终确认，但 Classical holdout 在旧切分中曾属于 discovery，并非 pristine 外部复制集。结构视角是原三视角确认性家族之外的扩展，因此其证据层级保持为探索性验证。本版在该层级内同样采用统一的 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留。

## 1. 结论摘要

- 1,200/1,200 个结构片段成功完成有向图与持续 Path Homology，失败 0；状态模型 SHA-256 为 `{topology_summary['state_model_sha256']}`。
- validation/180 s 的 {analysis_summary['primary_tests']} 个预设结构指标中，{analysis_summary['primary_fdr_discoveries_q_0_05']} 个通过 BH-FDR $q\le0.05$；validation/300 s 有 {analysis_summary['sensitivity_fdr_discoveries_q_0_05']} 个通过，其中 {analysis_summary['replicated_same_direction']} 个在两种时长均显著且方向一致。
- 通过主分析 FDR 的指标为 {significant_names}；它们描述两组在共享宏观状态码本中的覆盖、转移方向和连通过程，不表示曲式质量或复杂度高低。
- $H_1$ 高度零膨胀：validation/180 s 非零曲目为 Classical {h1['classical']['nonzero']}/{h1['classical']['total']}、Open Focus {h1['focus']['nonzero']}/{h1['focus']['total']}；不得把扩展阈值个例改写成主分析中的普遍结构环。
- holdout/180 s 的结构块整体表示 pseudo-$F={float(holdout_row['pseudo_f']):.3f}$、$p={float(holdout_row['p_value']):.3f}$、跨次级视角 BH $q={float(holdout_row['p_fdr_bh']):.3f}$。原门控的 {len(holdout_locked)} 个方向指标中，{int(holdout_locked['direction_matched'].sum())}/{len(holdout_locked)} 方向一致；历史 $q\le0.10$ 有 {int(holdout_locked['replicated_q_0_10'].sum())}/{len(holdout_locked)}、严格 $q\le0.05$ 有 {holdout_strict}/{len(holdout_locked)} 复现，属于部分而非完整复制。
- 当前拓扑输入 SHA-256 为 `{current_hash}`，与 holdout gate **{'一致' if current_hash == gate_hash else '不一致'}**；没有在重跑后改阈值、重选指标或调参。
- 结论属于观察性声学结构比较，不支持注意力、治疗、认知、生成质量或因果结论。

## 2. 结构视角的构建思想

音高、节奏和调制视角描述局部状态，而结构视角把音乐表示为“宏观声学段落的有向演化”。它先比较不同时刻的声学纹理，利用自相似矩阵定位变化边界，再把每个段落映射到共享结构原型。这样得到的路径不是逐帧音色序列，而是 A→B→A、A→B→C 等高阶段落组织。

### 2.1 自相似矩阵

对第 $i$ 个短时声学向量 $\mathbf{{x}}_i\in\mathbb{{R}}^d$ 作稳健标准化：

$$
\mathbf{{z}}_i=\frac{{\mathbf{{x}}_i-\operatorname{{med}}(\mathbf{{x}})}}{{1.4826\,\operatorname{{MAD}}(\mathbf{{x}})+\varepsilon}},
\qquad
\mathbf{{u}}_i=\frac{{[\mathbf{{z}}_i,1]}}{{\lVert[\mathbf{{z}}_i,1]\rVert_2}}.
$$

声学自相似矩阵为

$$
S_{{ij}}=\frac{{1+\mathbf{{u}}_i^\mathsf{{T}}\mathbf{{u}}_j}}{{2}},\qquad S_{{ij}}\in[0,1].
$$

块状对角结构表示相似段落，块之间的突变提示结构边界。

### 2.2 Foote 棋盘 novelty 与边界

令 $L_t=[t-h,t)$、$R_t=[t,t+h)$，棋盘 novelty 为

$$
\nu(t)=\frac{{1}}{{2h^2}}\left(
\sum_{{i,j\in L_t}}S_{{ij}}+\sum_{{i,j\in R_t}}S_{{ij}}
-\sum_{{i\in L_t,j\in R_t}}S_{{ij}}-\sum_{{i\in R_t,j\in L_t}}S_{{ij}}
\right).
$$

实现固定使用 8 s 核、$\operatorname{{median}}(\nu)+1.5\operatorname{{MAD}}(\nu)$ 峰值阈值，并把段长约束在 8–45 s。边界 $0=b_0<\cdots<b_K=T$ 把音频划分为 $K$ 个宏观块。

### 2.3 高阶状态码本

第 $k$ 块的声学向量为

$$
\mathbf{{q}}_k=\frac{{1}}{{|I_k|}}\sum_{{i\in I_k}}\mathbf{{x}}_i.
$$

仅用 discovery/180 s 的 Focus/Classical 平衡数据拟合稳健标准化、32 维 PCA 和 16 个 MiniBatch K-means 原型 $\mathbf{{c}}_m$。状态分配为

$$
s_k=\arg\min_m\left\lVert\mathbf{{P}}\mathbf{{D}}^{{-1}}(\mathbf{{q}}_k-\boldsymbol{{\mu}})-\mathbf{{c}}_m\right\rVert_2^2.
$$

本轮共得到 {feature_summary['quality']['structure_blocks']:,} 个宏观块和 {feature_summary['quality']['structure_boundaries']:,} 个边界；16 个原型均被使用。

## 3. 有向图与 Path Homology

对状态路径 $(s_0,\ldots,s_K)$，定义相邻转移计数和条件概率：

$$
C_{{uv}}=|\{{k:s_k=u,\ s_{{k+1}}=v\}}|,
\qquad p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}}.
$$

自环用于 `self_transition_ratio`，但不进入 Path Homology 图。每个源状态最多保留 top-6 非自环边。过滤图 $G_\tau$ 保留 $p_{{uv}}\ge\tau$ 的边；阈值从 0.95 降至 0.05 时只增加边：

$$
G_{{0.95}}\subseteq G_{{0.90}}\subseteq\cdots\subseteq G_{{0.05}}.
$$

对允许的 $p$-路径 $e_{{v_0\ldots v_p}}$，GLMY 边界算子为

$$
\partial e_{{v_0\ldots v_p}}=\sum_{{i=0}}^p(-1)^i e_{{v_0\ldots\widehat{{v_i}}\ldots v_p}}.
$$

令 $\Omega_p=\{{a\in A_p:\partial a\in A_{{p-1}}\}}$，则

$$
H_p^{{\mathrm{{path}}}}(G)=
\frac{{\ker(\partial_p|_{{\Omega_p}})}}{{\operatorname{{im}}(\partial_{{p+1}}|_{{\Omega_{{p+1}}}})}},
\qquad \beta_p=\dim H_p^{{\mathrm{{path}}}}(G).
$$

持久图和 barcode 使用递增坐标 $a=1-\tau$。对 $a_i\le a_j$，持久秩不变量为

$$
\rho_p(a_i,a_j)=\operatorname{{rank}}\operatorname{{im}}
\left[H_p(G_{{a_i}})\longrightarrow H_p(G_{{a_j}})\right].
$$

$H_0$ 表示有向可达结构的连通变化，$H_1$ 表示不能由允许 2-路径边界填充的有向一维类。生产流程只报告 $H_0/H_1$，没有计算 $H_2$。

## 4. 可视化与代表样本

### 4.1 稳健代表样本选择

代表曲目只用于可视化，不参与假设检验。对 validation/180 s 中每一组，在 9 个预设描述子上计算稳健组中心。若 $\mathbf{{m}}$ 为组中位数、$r_j=1.4826\operatorname{{MAD}}_j$，并令 $\widetilde r_j=r_j$（当 $r_j>10^{{-9}}$）否则 $\widetilde r_j=1$，选择

$$
i^*=\arg\min_i\sum_j\left(\frac{{x_{{ij}}-m_j}}{{\widetilde r_j}}\right)^2.
$$

这避免人工挑选“最像预期”的图。选择记录保存在 `metadata/structure_representative_selection.csv`。

| 组别 | 代表片段 | 稳健距离 | 顶点 | 边 | 自转移率 | 路径熵 |
|---|---|---:|---:|---:|---:|---:|
{representative_rows}

### 4.2 码本、单曲结构轨迹与有向图

{figure_markup['structure_codebook']}

{figure_markup['structure_ssm']}

{figure_markup['structure_directed_state_graph']}

### 4.3 两组稳健代表

{figure_markup['structure_representative_state_graphs']}

{figure_markup['structure_representative_ssm']}

### 4.4 持久 Path Homology 过程

{mechanism_text}

{figure_markup['structure_filtration_process']}

{figure_markup['structure_persistence_diagram']}

{figure_markup['structure_barcode']}

### 4.5 组间分布、Betti 曲线与尺度敏感性

{figure_markup['structure_group_summary']}

{figure_markup['structure_betti_curves']}

{figure_markup['structure_scale_sensitivity']}

{figure_markup['structure_effect_sizes']}

{figure_markup['structure_duration_stability']}

## 5. 组间结果

Kruskal–Wallis 检验在 20 个预设结构指标内作 BH-FDR，判定统一要求 $q\le0.05$。若秩和统计量为 $H$、组数为 $k$、总样本为 $N$，效应量为

$$
\epsilon^2=\frac{{H-k+1}}{{N-k}}.
$$

两组情况下另报告 Mann–Whitney rank-biserial。holdout 的整体结构块使用 discovery 拟合的秩正态 Mahalanobis 距离与 999 次标签置换，其 pseudo-$F$ 为

$$
F^*=\frac{{SS_{{between}}/(g-1)}}{{SS_{{within}}/(N-g)}}.
$$

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s FDR | 300 s FDR |
|---|---:|---:|---:|---:|---:|
{metric_rows}

Open Focus 与 Classical 的完整独立两两检验：

| 指标 | 比较 | rank-biserial（前者−后者） | FDR |
|---|---|---:|---:|
{pair_table}

### 5.1 解读

1. **跨时长稳定性。** 只有同时通过 validation/180 s FDR、在 300 s 方向一致并再次显著的指标才视为跨时长稳定；本轮共有 {analysis_summary['replicated_same_direction']} 项。
2. **状态空间解释。** 边密度、互惠性与自转移率只描述两组在同一 16 原型空间中的组织，不等于曲式标签、复杂度或质量。
3. **$H_1$ 不支持稳定组间差异。** 主尺度非零比例为 Classical {h1['classical']['nonzero']/h1['classical']['total']:.1%}、Open Focus {h1['focus']['nonzero']/h1['focus']['total']:.1%}；敏感阈值下为 {h1_sensitivity['classical']['nonzero']}/{h1_sensitivity['classical']['total']} 和 {h1_sensitivity['focus']['nonzero']}/{h1_sensitivity['focus']['total']}。holdout/180 s 主阈值为 Classical {holdout_h1['classical']}/45、Open Focus {holdout_h1['focus']}/45，扩展阈值为 {holdout_sensitivity_h1['classical']}/45 与 {holdout_sensitivity_h1['focus']}/45。主分析六个 $H_1$ 汇总量均未通过 FDR；300 s 中若干接近阈值的结果没有主尺度支持，而且所有 validation/180 s $H_1$ 区间均为右删失或不存在，没有有限 birth–death 区间。
4. **观察性边界。** 任何显著声学结构差异都不能推出注意力、治疗、生成质量或因果效应。

## 6. 证据层级与局限

- **探索性验证：** 结构视角并非原三视角确认性家族；本轮在该扩展内部固定 validation/180 s、表示、阈值和 20 指标 FDR，并按统一标准要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件，但结构还必须对 $L+P$ 提供预定的正融合增量；当前未满足该增量条件，因此结构端点不进入主要冻结指纹，只保留为宏观辅助描述。
- **操作性最终确认：** holdout/180 s 的整体结构块显著，但 6 个原锁定方向指标仅 5 个同方向；历史 $q\le0.10$ 为 4/6、严格 $q\le0.05$ 为 {holdout_strict}/6 联合 FDR 复现。Classical holdout 也不是 pristine 外部样本。
- **敏感性：** validation/300 s；4 个主尺度差异再次显著且方向一致。`directed_recurrence` 的 300 s $q=0.0946$ 现降为提示性结果，300 s 新出现的 $H_1$ 边缘结果不能替代主分析。
- **说明性：** 两个稳健组中心代表图、SSM，以及无有限 $H_1$ 区间时的过滤负结果示例。
- **不支持：** 任何稳定或 Focus 特异的 $H_1$；本轮未计算 $H_2$，因此不作 $H_2$ 发现声明；注意力提升、ADHD 疗效、生成效果或因果机制；将已归档三组结论转移到当前两组数据。
- 结构码本描述的是声学段落原型，不等同于音乐学上的主歌、副歌或奏鸣曲式标签。
- 边界检测和 16 状态量化仍可能压缩弱渐变结构；未来应做边界扰动、状态数和 top-k 稳定性分析，但不得据此调参追求显著性。

## 7. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/rerun_structure_path_homology.py
python scripts/analyze_structure_results.py
python scripts/render_structure_open_report.py
```

主要数值文件：

- `metadata/structure_topology_segments.csv`
- `metadata/structure_topology_filtration.csv`
- `metadata/structure_topology_filtration_sensitivity.csv`
- `metadata/structure_statistical_tests.csv`
- `metadata/structure_pairwise_tests.csv`
- `metadata/structure_analysis_summary.json`
- `metadata/structure_representative_selection.csv`

## 8. 参考文献

1. Foote, J. (2000). Automatic Audio Segmentation Using a Measure of Audio Novelty. ICME.
2. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.
3. Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S.-T. (2012). Homologies of path complexes and digraphs.
4. Chowdhury, S., & Mémoli, F. (2018). Persistent path homology of directed networks. SODA.
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def main() -> int:
    global PEDAGOGICAL_EXAMPLE

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        }
    )
    analysis_summary = json.loads(
        (ROOT / "metadata" / "structure_analysis_summary.json").read_text(
            encoding="utf-8"
        )
    )
    topology = pd.read_csv(ROOT / "metadata" / "structure_topology_segments.csv")
    selected = select_representatives(topology)
    mechanism = analysis_summary["mechanism_example"]
    if mechanism.get("available", True) and mechanism.get("segment_id"):
        PEDAGOGICAL_EXAMPLE = str(mechanism["segment_id"])
    else:
        PEDAGOGICAL_EXAMPLE = str(
            selected[selected["group"] == "focus"].iloc[0]["segment_id"]
        )
    figure_stems = [
        "structure_codebook",
        "structure_ssm",
        "structure_directed_state_graph",
        "structure_representative_state_graphs",
        "structure_representative_ssm",
        "structure_filtration_process",
        "structure_persistence_diagram",
        "structure_barcode",
        "structure_group_summary",
        "structure_betti_curves",
        "structure_scale_sensitivity",
        "structure_effect_sizes",
        "structure_duration_stability",
    ]
    plot_codebook(topology)
    plot_example_ssm(topology)
    plot_example_graph(topology)
    plot_representative_graphs(selected)
    plot_representative_ssm(selected)
    plot_filtration(topology)
    plot_persistence_diagram(topology)
    plot_barcode(topology)
    plot_group_summary(topology)
    plot_betti_curves()
    plot_scale_sensitivity(topology)
    plot_effect_sizes()
    plot_duration_stability()
    report = write_report(topology, selected, figure_stems)
    print(report)
    for stem in figure_stems:
        print(OUTPUT / f"{stem}.png")
        print(OUTPUT / f"{stem}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
