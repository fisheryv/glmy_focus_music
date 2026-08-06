# ruff: noqa: E501
from __future__ import annotations

import os
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
OUTPUT = ROOT / "runs" / "pitch_path_homology"
EXAMPLE_ID = "focus_brainfm_80d50779ccb9__180s"
EXAMPLE_GROUP = "focus"
EXAMPLE_SPLIT = "discovery"
COLORS = {"classical": "#4472C4", "focus": "#ED7D31", "pop": "#70AD47"}
GROUP_LABELS = {"classical": "Classical", "focus": "Focus", "pop": "Pop"}
PITCH_LABELS = {
    0: "C",
    1: "C#/Db",
    2: "D",
    3: "D#/Eb",
    4: "E",
    5: "F",
    6: "F#/Gb",
    7: "G",
    8: "G#/Ab",
    9: "A",
    10: "A#/Bb",
    11: "B",
    12: "U",
}


def _save(figure: plt.Figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _example_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    paths = (
        ROOT
        / "features"
        / "chroma"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz",
        ROOT
        / "graphs"
        / "pitch"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz",
        ROOT
        / "homology"
        / "persistence_sensitivity"
        / "pitch"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz",
    )
    output: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path) as archive:
            output.append({name: np.asarray(archive[name]) for name in archive.files})
    return output[0], output[1], output[2]


def _pitch_ssm(chroma: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    normalized = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 1e-12)
    return np.clip(normalized @ normalized.T, 0.0, 1.0)


def plot_chromagram_ssm(feature: dict[str, np.ndarray]) -> Path:
    chroma = feature["chroma"].astype(float)
    states = feature["states"].astype(int)
    times = feature["times"].astype(float)
    similarity = _pitch_ssm(chroma)
    left = max(0.0, float(times[0]))
    right = float(times[-1])
    extent = (left, right, left, right)

    figure = plt.figure(figsize=(10.0, 9.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(2.6, 1.0, 5.5))
    axis_chroma = figure.add_subplot(grid[0])
    image_chroma = axis_chroma.imshow(
        chroma.T,
        origin="lower",
        aspect="auto",
        extent=(left, right, -0.5, 11.5),
        cmap="magma",
        interpolation="nearest",
    )
    axis_chroma.set_yticks(range(12), [PITCH_LABELS[index] for index in range(12)])
    axis_chroma.set_ylabel("Pitch class")
    axis_chroma.set_title(f"Beat-synchronous chromagram: {EXAMPLE_ID}")
    figure.colorbar(image_chroma, ax=axis_chroma, fraction=0.025, pad=0.02, label="Chroma energy")

    axis_state = figure.add_subplot(grid[1], sharex=axis_chroma)
    axis_state.step(times, states, where="mid", color="#28536B", lw=1.1)
    uncertain = states == 12
    axis_state.scatter(times[uncertain], states[uncertain], s=8, color="#C44E52", label="U")
    axis_state.set_yticks([0, 4, 7, 11, 12], ["C", "E", "G", "B", "U"])
    axis_state.set_ylabel("State")
    axis_state.set_xlabel("Time (s)")
    axis_state.grid(alpha=0.2)
    axis_state.legend(loc="upper right", fontsize=8, frameon=False)

    axis_ssm = figure.add_subplot(grid[2])
    image_ssm = axis_ssm.imshow(
        similarity,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis_ssm.set(
        title="Pitch self-similarity matrix (cosine similarity of beat chroma)",
        xlabel="Time (s)",
        ylabel="Time (s)",
    )
    figure.colorbar(image_ssm, ax=axis_ssm, fraction=0.035, pad=0.02, label="Similarity")
    return _save(figure, "pitch_chromagram_ssm.png")


def _layout(vertices: np.ndarray) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    present = {int(value) for value in vertices}
    for state in range(12):
        if state in present:
            angle = np.pi / 2.0 - state * 2.0 * np.pi / 12.0
            positions[state] = np.array([np.cos(angle), np.sin(angle)])
    if 12 in present:
        positions[12] = np.array([0.0, 0.0])
    return positions


def _draw_graph(
    axis: plt.Axes,
    graph: dict[str, np.ndarray],
    *,
    threshold: float,
    label_edges: bool,
) -> None:
    vertices = graph["vertices"].astype(int)
    positions = _layout(vertices)
    edges = list(
        zip(
            graph["edge_source"].astype(int),
            graph["edge_target"].astype(int),
            graph["edge_weight"].astype(float),
            strict=True,
        )
    )
    selected = [(source, target, weight) for source, target, weight in edges if weight >= threshold]
    for source, target, weight in selected:
        start = positions[source]
        end = positions[target]
        delta = end - start
        if np.linalg.norm(delta) < 1e-10:
            continue
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8 + 5 * weight,
            linewidth=0.4 + 2.6 * weight,
            color="#46647A",
            alpha=0.25 + 0.65 * weight,
            shrinkA=13,
            shrinkB=13,
            connectionstyle="arc3,rad=0.07",
        )
        axis.add_patch(patch)
        if label_edges and weight >= 0.15:
            midpoint = (start + end) / 2.0
            axis.text(
                midpoint[0],
                midpoint[1],
                f"{weight:.2f}",
                fontsize=6.5,
                color="#263B4A",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.72},
            )
    for state, position in positions.items():
        face = "#C44E52" if state == 12 else "#F6C85F"
        axis.scatter(position[0], position[1], s=540, color=face, edgecolor="#263B4A", zorder=5)
        axis.text(position[0], position[1], PITCH_LABELS[state], ha="center", va="center", fontsize=8, zorder=6)
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_directed_graph(graph: dict[str, np.ndarray]) -> Path:
    figure, axis = plt.subplots(figsize=(8.3, 8.0), constrained_layout=True)
    _draw_graph(axis, graph, threshold=0.0, label_edges=True)
    axis.set_title(
        "Full directed pitch-state graph\nedge width = outgoing transition probability; labels shown for p >= 0.15",
        fontsize=12,
    )
    return _save(figure, "pitch_directed_state_graph.png")


def plot_filtration(graph: dict[str, np.ndarray], persistence: dict[str, np.ndarray]) -> Path:
    thresholds = [0.6, 0.5, 0.15]
    archive_thresholds = persistence["thresholds"].astype(float)
    h0 = persistence["h0_betti"].astype(int)
    h1 = persistence["h1_betti"].astype(int)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    descriptions = ("before H1 birth", "H1 birth", "H1 death")
    for axis, threshold, description in zip(axes, thresholds, descriptions, strict=True):
        index = int(np.argmin(np.abs(archive_thresholds - threshold)))
        _draw_graph(axis, graph, threshold=threshold, label_edges=False)
        axis.set_title(
            f"tau = {threshold:.2f}: {description}\n"
            f"edges={int(persistence['edge_count'][index])}, beta0={h0[index]}, beta1={h1[index]}",
            fontsize=10,
        )
    figure.suptitle("Descending-threshold persistent path-homology filtration", fontsize=13)
    return _save(figure, "pitch_filtration_process.png")


def _expanded_intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, float | int | bool]]:
    intervals: list[dict[str, float | int | bool]] = []
    end = 1.0 - float(np.min(persistence["thresholds"]))
    for index, dimension in enumerate(persistence["interval_dimension"].astype(int)):
        birth_threshold = float(persistence["interval_birth_threshold"][index])
        death_threshold = float(persistence["interval_death_threshold"][index])
        censored = bool(persistence["interval_censored"][index])
        birth = 1.0 - birth_threshold
        death = end if censored else 1.0 - death_threshold
        multiplicity = int(persistence["interval_multiplicity"][index])
        for _ in range(multiplicity):
            intervals.append(
                {"dimension": dimension, "birth": birth, "death": death, "censored": censored}
            )
    return intervals


def plot_persistence_diagram(persistence: dict[str, np.ndarray]) -> Path:
    intervals = _expanded_intervals(persistence)
    figure, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    end = 1.0 - float(np.min(persistence["thresholds"]))
    axis.plot([0, end], [0, end], ls="--", color="#7A7A7A", lw=1)
    for dimension, marker, color in ((0, "o", "#4472C4"), (1, "^", "#C44E52")):
        selected = [item for item in intervals if item["dimension"] == dimension]
        axis.scatter(
            [float(item["birth"]) for item in selected],
            [float(item["death"]) for item in selected],
            marker=marker,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.86,
            label=f"H{dimension}",
        )
    censored = [item for item in intervals if bool(item["censored"])]
    axis.scatter(
        [float(item["birth"]) for item in censored],
        [float(item["death"]) for item in censored],
        marker="o",
        s=90,
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
        title=f"Persistent path-homology diagram: {EXAMPLE_ID}",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, "pitch_persistence_diagram.png")


def plot_barcode(persistence: dict[str, np.ndarray]) -> Path:
    intervals = _expanded_intervals(persistence)
    intervals.sort(key=lambda item: (int(item["dimension"]), float(item["birth"]), float(item["death"])))
    figure, axis = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
    end = 1.0 - float(np.min(persistence["thresholds"]))
    for row, item in enumerate(intervals):
        dimension = int(item["dimension"])
        color = "#4472C4" if dimension == 0 else "#C44E52"
        start = float(item["birth"])
        stop = float(item["death"])
        axis.hlines(row, start, stop, color=color, lw=3)
        axis.plot(start, row, marker="|", color=color, ms=8)
        axis.plot(
            stop,
            row,
            marker="o" if bool(item["censored"]) else "|",
            color=color,
            ms=6 if bool(item["censored"]) else 8,
            markerfacecolor="white" if bool(item["censored"]) else color,
        )
    axis.set(
        xlim=(-0.02, end + 0.03),
        xlabel="Filtration coordinate a = 1 - tau",
        ylabel="Interval index",
        title=f"Persistent path barcode: {EXAMPLE_ID}",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(0.02, 0.96, "blue: H0    red: H1    open circle: censored", transform=axis.transAxes, va="top")
    return _save(figure, "pitch_barcode.png")


def plot_group_summary() -> Path:
    topology = pd.read_csv(ROOT / "metadata" / "pitch_topology_segments.csv")
    data = topology[
        (topology["split"] == "validation") & (topology["scale_seconds"] == 180.0)
    ]
    metrics = (
        ("vertex_count", "Observed states"),
        ("edge_count", "Directed edges"),
        ("path_entropy", "Path entropy"),
        ("h0_betti_mean", "Mean beta0"),
        ("directed_recurrence", "Directed recurrence"),
    )
    groups = ("classical", "focus", "pop")
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.3), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [data.loc[data.group == group, metric].dropna().to_numpy() for group in groups]
        plot = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(plot["boxes"], groups, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        axis.set_xticks(range(1, 4), [GROUP_LABELS[group] for group in groups], rotation=25)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Pitch-view group comparison (validation, 180 s)", fontsize=13)
    return _save(figure, "pitch_group_summary.png")


def plot_betti_curves() -> Path:
    filtration = pd.read_csv(ROOT / "metadata" / "pitch_topology_filtration_sensitivity.csv")
    data = filtration[
        (filtration["split"] == "validation") & (filtration["scale_seconds"] == 180.0)
    ].copy()
    data["a"] = 1.0 - data["threshold"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("h0_betti", "h1_betti"), ("Mean beta0", "Mean beta1"), strict=True):
        for group in ("classical", "focus", "pop"):
            selected = data[data.group == group]
            summary = selected.groupby("a")[metric].agg(["mean", "sem"]).reset_index().sort_values("a")
            x = summary["a"].to_numpy(dtype=float)
            mean = summary["mean"].to_numpy(dtype=float)
            sem = summary["sem"].fillna(0).to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", ms=3.5, lw=1.7, color=COLORS[group], label=GROUP_LABELS[group])
            axis.fill_between(x, mean - sem, mean + sem, color=COLORS[group], alpha=0.14)
        axis.set(title=title, xlabel="Filtration coordinate a = 1 - tau", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Pitch-view Betti curves (validation, 180 s; mean +/- SEM)", fontsize=13)
    return _save(figure, "pitch_betti_curves_by_group.png")


def main() -> None:
    feature, graph, persistence = _example_arrays()
    outputs = (
        plot_chromagram_ssm(feature),
        plot_directed_graph(graph),
        plot_filtration(graph, persistence),
        plot_persistence_diagram(persistence),
        plot_barcode(persistence),
        plot_group_summary(),
        plot_betti_curves(),
    )
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
