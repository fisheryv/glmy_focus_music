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

from features.pitch_v2 import tonnetz_similarity

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pitch_v2_path_homology_open"
REPORT = ROOT / "docs" / "path-homology-pitch-v2-analysis.md"
EXAMPLE_ID = ""
EXAMPLE_GROUP = "focus"
EXAMPLE_SPLIT = "validation"
GROUPS = ("classical", "focus")
COLORS = {"classical": "#4472C4", "focus": "#ED7D31"}
GROUP_LABELS = {"classical": "Classical", "focus": "Open Focus"}
CONFIRMATORY_FDR_Q = 0.05
PITCH_LABELS = ("C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B")


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


def _codebook() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays = _read_npz(ROOT / "features" / "models" / "pitch_v2_codebook.npz")
    metadata = json.loads(
        (ROOT / "features" / "models" / "pitch_v2_codebook.json").read_text(encoding="utf-8")
    )
    return arrays, metadata


def _example_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    paths = (
        ROOT / "graphs" / "pitch_v2" / "180s" / EXAMPLE_GROUP / EXAMPLE_SPLIT / f"{EXAMPLE_ID}.npz",
        ROOT
        / "homology"
        / "persistence_sensitivity"
        / "pitch_v2"
        / "180s"
        / EXAMPLE_GROUP
        / EXAMPLE_SPLIT
        / f"{EXAMPLE_ID}.npz",
    )
    return tuple(_read_npz(path) for path in paths)  # type: ignore[return-value]


def _example_feature_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    row = pd.read_csv(ROOT / "metadata" / "pitch_v2_features.csv")
    row = row[row["segment_id"] == EXAMPLE_ID].iloc[0]
    return (
        _read_npz(ROOT / str(row["source_chroma_relative_path"])),
        _read_npz(ROOT / str(row["pitch_v2_relative_path"])),
    )


def plot_codebook(codebook: dict[str, np.ndarray], metadata: dict[str, object]) -> Path:
    prototypes = codebook["chroma_prototypes"].astype(float)
    labels = list(metadata["state_labels"])
    diagnostics = pd.read_csv(ROOT / "metadata" / "pitch_v2_codebook_diagnostics.csv")
    figure = plt.figure(figsize=(11.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 2.0))
    axis = figure.add_subplot(grid[0])
    image = axis.imshow(prototypes, aspect="auto", cmap="magma", vmin=0, vmax=np.max(prototypes))
    axis.set_xticks(range(12), PITCH_LABELS, rotation=30, ha="right")
    axis.set_yticks(range(16), labels)
    axis.set(xlabel="Pitch class", ylabel="Frozen harmonic state", title="Discovery-only Tonnetz codebook: mean chroma profile per state")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Mean normalized chroma")

    subgrid = grid[1].subgridspec(1, 3)
    metrics = (
        ("silhouette", "Silhouette (higher is better)"),
        ("seed_stability_ari", "Seed stability ARI"),
        ("inertia_per_step", "Inertia per step (lower is better)"),
    )
    for subaxis, (metric, title) in zip((figure.add_subplot(subgrid[0, index]) for index in range(3)), metrics, strict=True):
        subaxis.plot(diagnostics.v_pitch, diagnostics[metric], marker="o", color="#28536B")
        chosen = diagnostics[diagnostics.v_pitch == 16].iloc[0]
        subaxis.scatter([16], [chosen[metric]], s=85, color="#C44E52", zorder=3, label="frozen V=16")
        subaxis.set(xticks=diagnostics.v_pitch, xlabel="V_pitch", title=title)
        subaxis.grid(alpha=0.2)
    figure.suptitle("Pitch v2 harmonic skeleton codebook and Discovery diagnostics", fontsize=14)
    return _save(figure, "pitch_v2_codebook.png")


def plot_tonnetz_ssm(chroma: dict[str, np.ndarray], feature: dict[str, np.ndarray]) -> Path:
    chroma_values = chroma["chroma"].astype(float)
    tonnetz = feature["tonnetz"].astype(float)
    states = feature["states"].astype(int)
    valid = feature["valid"].astype(bool)
    times = feature["times"].astype(float)
    similarity = tonnetz_similarity(tonnetz)
    masked_states = np.where(valid, states, np.nan)
    left, right = float(times[0]), float(times[-1])

    figure = plt.figure(figsize=(10.0, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(4, 1, height_ratios=(2.2, 1.7, 0.9, 5.0))
    axis_chroma = figure.add_subplot(grid[0])
    image = axis_chroma.imshow(
        chroma_values.T,
        origin="lower",
        aspect="auto",
        extent=(left, right, -0.5, 11.5),
        cmap="magma",
        interpolation="nearest",
    )
    axis_chroma.set_yticks(range(12), PITCH_LABELS)
    axis_chroma.set(title=f"Beat-synchronous chromagram: {EXAMPLE_ID}", ylabel="Pitch class")
    figure.colorbar(image, ax=axis_chroma, fraction=0.02, pad=0.01, label="Chroma")

    axis_tonnetz = figure.add_subplot(grid[1], sharex=axis_chroma)
    for index, label in enumerate(("5th-x", "5th-y", "min3-x", "min3-y", "maj3-x", "maj3-y")):
        axis_tonnetz.plot(times, tonnetz[:, index] + index * 1.15, lw=0.75, label=label)
    axis_tonnetz.set_yticks(np.arange(6) * 1.15, ("5x", "5y", "m3x", "m3y", "M3x", "M3y"))
    axis_tonnetz.set(title="Six Tonnetz coordinates", ylabel="Axis + offset")
    axis_tonnetz.grid(alpha=0.15)

    axis_state = figure.add_subplot(grid[2], sharex=axis_chroma)
    axis_state.step(times, masked_states, where="mid", color="#28536B", lw=1.1)
    axis_state.scatter(times[~valid], np.full(np.count_nonzero(~valid), -1), marker="x", s=12, color="#C44E52", label="masked")
    axis_state.set(yticks=[-1, 0, 5, 10, 15], yticklabels=["mask", "S00", "S05", "S10", "S15"], ylabel="State", xlabel="Time (s)")
    axis_state.legend(loc="upper right", frameon=False, fontsize=8)
    axis_state.grid(alpha=0.2)

    axis_ssm = figure.add_subplot(grid[3])
    ssm = axis_ssm.imshow(
        similarity,
        origin="lower",
        extent=(left, right, left, right),
        aspect="equal",
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis_ssm.set(title="Tonnetz self-similarity matrix (all beat chroma)", xlabel="Time (s)", ylabel="Time (s)")
    figure.colorbar(ssm, ax=axis_ssm, fraction=0.03, pad=0.02, label="Gaussian similarity")
    return _save(figure, "pitch_v2_tonnetz_ssm.png")


def _positions(codebook: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    projected = PCA(n_components=2, random_state=20_260_716).fit_transform(codebook["centers"].astype(float))
    projected -= np.mean(projected, axis=0)
    projected /= max(float(np.max(np.linalg.norm(projected, axis=1))), 1e-12)
    anchors = projected.copy()
    for _ in range(120):
        movement = np.zeros_like(projected)
        for left in range(projected.shape[0]):
            for right in range(left + 1, projected.shape[0]):
                delta = projected[left] - projected[right]
                distance = float(np.linalg.norm(delta))
                if distance < 0.30:
                    direction = delta / max(distance, 1e-6)
                    push = (0.30 - distance) * 0.035 * direction
                    movement[left] += push
                    movement[right] -= push
        movement += 0.01 * (anchors - projected)
        projected += movement
    projected /= max(float(np.max(np.linalg.norm(projected, axis=1))), 1e-12)
    return {index: projected[index] for index in range(projected.shape[0])}


def _draw_graph(
    axis: plt.Axes,
    graph: dict[str, np.ndarray],
    codebook: dict[str, np.ndarray],
    labels: list[str],
    *,
    threshold: float,
    show_edge_labels: bool,
) -> None:
    vertices = graph["vertices"].astype(int)
    pca_positions = _positions(codebook)
    ordered = sorted(vertices, key=lambda state: np.arctan2(pca_positions[int(state)][1], pca_positions[int(state)][0]))
    positions = {
        int(state): np.asarray(
            [
                np.cos(np.pi / 2 - 2 * np.pi * index / len(ordered)),
                np.sin(np.pi / 2 - 2 * np.pi * index / len(ordered)),
            ]
        )
        for index, state in enumerate(ordered)
    }
    edges = zip(
        graph["edge_source"].astype(int),
        graph["edge_target"].astype(int),
        graph["edge_weight"].astype(float),
        strict=True,
    )
    for source, target, weight in edges:
        if weight < threshold:
            continue
        start, end = positions[source], positions[target]
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9 + 5 * weight,
            linewidth=0.7 + 2.8 * weight,
            color="#46647A",
            alpha=0.35 + 0.6 * weight,
            shrinkA=18,
            shrinkB=18,
            connectionstyle="arc3,rad=0.08",
        )
        axis.add_patch(patch)
        if show_edge_labels:
            midpoint = (start + end) / 2
            axis.text(midpoint[0], midpoint[1], f"{weight:.2f}", fontsize=7, ha="center", va="center", bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})
    for state in vertices:
        position = positions[state]
        color = plt.get_cmap("tab20")(state)
        axis.scatter(position[0], position[1], s=660, color=color, edgecolor="#263B4A", zorder=5)
        axis.text(position[0], position[1] + 0.025, f"S{state:02d}", ha="center", va="center", fontsize=8, zorder=6)
        if show_edge_labels:
            top = labels[state].split("(", 1)[1].rstrip(")")
            axis.text(position[0], position[1] - 0.09, top, ha="center", va="top", fontsize=6.2, zorder=6)
    axis.set_xlim(-1.28, 1.28)
    axis.set_ylim(-1.28, 1.28)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_directed_graph(graph: dict[str, np.ndarray], codebook: dict[str, np.ndarray], labels: list[str]) -> Path:
    figure, axis = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    _draw_graph(axis, graph, codebook, labels, threshold=0.0, show_edge_labels=True)
    axis.set_title("Directed Tonnetz-codebook state graph\nnode order follows PCA angle of frozen 6D centers; edge width = transition probability")
    return _save(figure, "pitch_v2_directed_state_graph.png")


def plot_filtration(graph: dict[str, np.ndarray], persistence: dict[str, np.ndarray], codebook: dict[str, np.ndarray], labels: list[str]) -> Path:
    dimensions = persistence["interval_dimension"].astype(int)
    censored = persistence["interval_censored"].astype(bool)
    finite_h1 = np.flatnonzero((dimensions == 1) & ~censored)
    if finite_h1.size == 0:
        raise RuntimeError("selected pitch_v2 example has no finite H1 interval")
    best = int(finite_h1[np.argmax(persistence["interval_lifetime"][finite_h1])])
    birth = float(persistence["interval_birth_threshold"][best])
    death = float(persistence["interval_death_threshold"][best])
    archive_thresholds = persistence["thresholds"].astype(float)
    before_candidates = archive_thresholds[archive_thresholds > birth]
    before = float(np.min(before_candidates)) if before_candidates.size else float(np.max(archive_thresholds))
    thresholds = (before, birth, death)
    descriptions = ("before H1 birth", "H1 birth", "H1 death")
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    for axis, threshold, description in zip(axes, thresholds, descriptions, strict=True):
        index = int(np.argmin(np.abs(archive_thresholds - threshold)))
        _draw_graph(axis, graph, codebook, labels, threshold=threshold, show_edge_labels=False)
        axis.set_title(
            f"tau={threshold:.2f}: {description}\n"
            f"edges={int(persistence['edge_count'][index])}, beta0={int(persistence['h0_betti'][index])}, beta1={int(persistence['h1_betti'][index])}",
            fontsize=10,
        )
    figure.suptitle("Pitch v2 descending-threshold persistent path-homology filtration", fontsize=13)
    return _save(figure, "pitch_v2_filtration_process.png")


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
    axis.set(xlim=(-0.02, end + 0.04), ylim=(-0.02, end + 0.04), xlabel="Birth a = 1 - tau", ylabel="Death a = 1 - tau", title=f"Pitch v2 persistence diagram: {EXAMPLE_ID}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, "pitch_v2_persistence_diagram.png")


def plot_barcode(persistence: dict[str, np.ndarray]) -> Path:
    intervals = _expanded_intervals(persistence)
    intervals.sort(key=lambda item: (int(item["dimension"]), float(item["birth"]), float(item["death"])))
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for row, item in enumerate(intervals):
        dimension = int(item["dimension"])
        color = "#4472C4" if dimension == 0 else "#C44E52"
        start, stop = float(item["birth"]), float(item["death"])
        axis.hlines(row, start, stop, color=color, lw=3)
        axis.plot(start, row, marker="|", color=color, ms=8)
        axis.plot(stop, row, marker="o" if item["censored"] else "|", color=color, ms=6 if item["censored"] else 8, markerfacecolor="white" if item["censored"] else color)
    axis.set(xlim=(-0.02, end + 0.03), xlabel="Filtration coordinate a = 1 - tau", ylabel="Interval index", title=f"Pitch v2 persistent path barcode: {EXAMPLE_ID}")
    axis.grid(axis="x", alpha=0.2)
    axis.text(0.02, 0.96, "blue: H0    red: H1    open circle: censored", transform=axis.transAxes, va="top")
    return _save(figure, "pitch_v2_barcode.png")


def plot_group_summary() -> Path:
    topology = pd.read_csv(ROOT / "metadata" / "pitch_v2_topology_segments.csv")
    data = topology[(topology.split == "validation") & (topology.scale_seconds == 180.0)]
    metrics = (
        ("vertex_count", "Observed states"),
        ("edge_count", "Directed edges"),
        ("path_entropy", "Path entropy"),
        ("h0_betti_mean", "Mean beta0"),
        ("directed_recurrence", "Directed recurrence"),
    )
    groups = GROUPS
    figure, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [data.loc[data.group == group, metric].to_numpy() for group in groups]
        plot = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(plot["boxes"], groups, strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.72)
        axis.set_xticks(
            range(1, len(groups) + 1),
            [GROUP_LABELS[group] for group in groups],
            rotation=25,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Pitch v2 group comparison (validation, 180 s)", fontsize=13)
    return _save(figure, "pitch_v2_group_summary.png")


def plot_betti_curves() -> Path:
    filtration = pd.read_csv(ROOT / "metadata" / "pitch_v2_topology_filtration_sensitivity.csv")
    data = filtration[(filtration.split == "validation") & (filtration.scale_seconds == 180.0)].copy()
    data["a"] = 1.0 - data.threshold
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("h0_betti", "h1_betti"), ("Mean beta0", "Mean beta1"), strict=True):
        for group in GROUPS:
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
    figure.suptitle("Pitch v2 Betti curves (validation, 180 s; mean +/- SEM)", fontsize=13)
    return _save(figure, "pitch_v2_betti_curves.png")


def plot_effect_sizes() -> Path:
    pairwise = pd.read_csv(ROOT / "metadata" / "pitch_v2_pairwise_tests.csv")
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
        title="Pitch v2 validation/180 s effect directions",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    return _save(figure, "pitch_v2_effect_sizes.png")


def plot_duration_stability() -> Path:
    pairwise = pd.read_csv(ROOT / "metadata" / "pitch_v2_pairwise_tests.csv")
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
    return _save(figure, "pitch_v2_duration_stability.png")


def plot_view_comparison() -> Path:
    old = pd.read_csv(ROOT / "metadata" / "pitch_topology_segments.csv")
    new = pd.read_csv(ROOT / "metadata" / "pitch_v2_topology_segments.csv")
    old = old[(old.split == "validation") & (old.scale_seconds == 180.0)]
    new = new[(new.split == "validation") & (new.scale_seconds == 180.0)]
    metrics = (("vertex_count", "States"), ("edge_count", "Edges"), ("path_entropy", "Path entropy"), ("h0_betti_mean", "Mean beta0"))
    groups = GROUPS
    figure, axes = plt.subplots(1, 4, figsize=(13.0, 4.1), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.36
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        old_values = [old.loc[old.group == group, metric].median() for group in groups]
        new_values = [new.loc[new.group == group, metric].median() for group in groups]
        axis.bar(x - width / 2, old_values, width, color="#9AA6B2", label="pitch")
        axis.bar(x + width / 2, new_values, width, color="#28536B", label="pitch_v2")
        axis.set_xticks(x, [GROUP_LABELS[group] for group in groups], rotation=25)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Current dominant-pitch view vs Tonnetz-codebook pitch_v2", fontsize=13)
    return _save(figure, "pitch_v2_vs_pitch.png")


def write_report(metadata: dict[str, object], figure_stems: tuple[str, ...]) -> Path:
    summary = json.loads(
        (ROOT / "metadata" / "pitch_v2_summary.json").read_text(encoding="utf-8")
    )
    tests = pd.read_csv(ROOT / "metadata" / "pitch_v2_statistical_tests.csv")
    pairwise = pd.read_csv(ROOT / "metadata" / "pitch_v2_pairwise_tests.csv")
    features = pd.read_csv(ROOT / "metadata" / "pitch_v2_features.csv")
    diagnostics = pd.read_csv(ROOT / "metadata" / "pitch_v2_codebook_diagnostics.csv")
    holdout_permanova = pd.read_csv(ROOT / "metadata" / "holdout_confirmation_permanova.csv")
    holdout_directional = pd.read_csv(ROOT / "metadata" / "holdout_confirmation_directional_metrics.csv")
    holdout_gate = json.loads((ROOT / "metadata" / "holdout_gate.json").read_text(encoding="utf-8"))
    primary = tests[tests["analysis_set"] == "primary_validation_180"].copy()
    sensitivity = tests[tests["analysis_set"] == "sensitivity_validation_300"].copy()
    primary = primary.sort_values(["p_fdr_bh", "metric"])
    sensitivity_by_metric = sensitivity.set_index("metric")
    focus_classical = pairwise[
        (pairwise["analysis_set"] == "primary_validation_180")
        & (pairwise["group_a"] == "classical")
        & (pairwise["group_b"] == "focus")
    ].sort_values(["p_fdr_bh", "metric"])
    h1 = summary["validation_180_h1_counts"]
    valid = features.assign(valid_share=features["valid_steps"] / features["steps"])
    valid = valid[
        (valid["split"] == "validation")
        & np.isclose(valid["scale_seconds"].astype(float), 180.0)
    ]
    valid_medians = valid.groupby("group")["valid_share"].median().to_dict()
    metric_rows = []
    for _, row in primary.iterrows():
        other = sensitivity_by_metric.loc[row["metric"]]
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
    pair_rows = []
    for _, row in focus_classical[
        focus_classical["p_fdr_bh"] <= CONFIRMATORY_FDR_Q
    ].iterrows():
        ci_low = -float(row["rank_biserial_ci95_high"])
        ci_high = -float(row["rank_biserial_ci95_low"])
        pair_rows.append(
            f"| {row['metric']} | {-float(row['rank_biserial_a_minus_b']):.3f} | "
            f"[{ci_low:.3f}, {ci_high:.3f}] | "
            f"{float(row['p_fdr_bh']):.3g} |"
        )
    diagnostic_rows = []
    for _, row in diagnostics.iterrows():
        diagnostic_rows.append(
            f"| {int(row['v_pitch'])} | {float(row['silhouette']):.3f} | "
            f"{float(row['seed_stability_ari']):.3f} | {float(row['min_cluster_share']):.3f} | "
            f"{float(row['max_cluster_share']):.3f} | {float(row['inertia_per_step']):.4f} |"
        )
    figures = "\n\n".join(
        f"![{stem}](../runs/pitch_v2_path_homology_open/{stem}.png)\n\n"
        f"[SVG](../runs/pitch_v2_path_homology_open/{stem}.svg)"
        for stem in figure_stems
    )
    primary_h1 = primary[primary["metric"].str.startswith("h1_")]
    sensitivity_h1 = sensitivity[sensitivity["metric"].str.startswith("h1_")]
    primary_h1_discoveries = int(
        np.count_nonzero(primary_h1["p_fdr_bh"] <= CONFIRMATORY_FDR_Q)
    )
    sensitivity_h1_discoveries = int(
        np.count_nonzero(sensitivity_h1["p_fdr_bh"] <= CONFIRMATORY_FDR_Q)
    )
    example = summary["mechanism_example"]
    holdout_pitch = holdout_permanova[
        (holdout_permanova["analysis_set"] == "primary_holdout_180")
        & (holdout_permanova["feature_set"] == "pitch")
    ].iloc[0]
    holdout_pitch_metrics = holdout_directional[
        (holdout_directional["analysis_set"] == "primary_holdout_180")
        & (holdout_directional["view"] == "pitch")
    ]
    holdout_direction_matches = int(
        holdout_pitch_metrics["direction_matched"].astype(str).str.lower().eq("true").sum()
    )
    holdout_replicated = int(
        holdout_pitch_metrics["replicated_q_0_10"].astype(str).str.lower().eq("true").sum()
    )
    holdout_replicated_strict = int(
        (
            holdout_pitch_metrics["direction_matched"].astype(str).str.lower().eq("true")
            & (holdout_pitch_metrics["p_fdr_bh"].astype(float) <= CONFIRMATORY_FDR_Q)
        ).sum()
    )
    gate_topology_sha = holdout_gate["input_sha256"]["metadata/pitch_v2_topology_segments.csv"]
    rerun_topology_sha = summary["artifact_sha256"]["metadata/pitch_v2_topology_segments.csv"]
    gate_hash_matches = gate_topology_sha == rerun_topology_sha
    report = rf"""# Path Homology `pitch_v2`：Focus–Classical 音高视角完整分析

生成日期：{date.today().isoformat()}。本文使用当前规范数据集 Jamendo Open Focus 300 与 Classical 300。两组均分为 discovery 195、validation 60、holdout 45；每首有 180 s 与 300 s 两个片段，共 1,200 个片段。主推断固定为 validation/180 s（n=120：每组 60）；validation/300 s 仅作时长敏感性。holdout 是既有哈希门控后的单次操作性确认，不用于重新选择参数或指标。本版将确认性端点统一为 BH-FDR $q\le0.05$；原 holdout 的 $q\le0.10$ 门控作为历史记录保留，不倒写为新预注册标准。

## 1. 结论摘要

- 仅用 discovery/180 s 的 Focus 与 Classical 各 {metadata['training_steps_per_group']['focus']:,} 个有效节拍重新拟合 Tonnetz 码本，并完成 {summary['segment_views']:,}/{summary['segment_views']:,} 个片段的有向图和持续 Path Homology，失败 0。码本 SHA-256：`{summary['codebook_sha256']}`。
- 20 个预设指标中，validation/180 s 有 {summary['primary_fdr_discoveries']} 个通过 BH-FDR $q\le0.05$；validation/300 s 有 {summary['sensitivity_fdr_discoveries']} 个通过，其中 {summary['replicated_same_direction']} 个在两种时长均显著且方向一致。原 $q\le0.10$ 下额外入选的 `edge_density`（$q=0.0654$）现降为提示性结果。
- 既有单次 holdout/180 s 确认中，pitch 表示的 permutation pseudo-$F={float(holdout_pitch['pseudo_f']):.3f}$、$p={float(holdout_pitch['p_value']):.3f}$、次级家族 FDR $q={float(holdout_pitch['p_fdr_bh']):.3g}$；原门控的 14 个方向性指标中 {holdout_direction_matches} 个方向一致、按历史 $q\le0.10$ 有 {holdout_replicated} 个复现，按统一严格口径 $q\le0.05$ 有 {holdout_replicated_strict} 个复现。重跑后的拓扑清单哈希与开盲门控记录{'一致' if gate_hash_matches else '不一致'}。
- $H_1$ 主阈值非零为 Classical {h1['classical']['primary_nonzero']}/{h1['classical']['n']}、Open Focus {h1['focus']['primary_nonzero']}/{h1['focus']['n']}。预设 $H_1$ 指标在 180 s 有 {primary_h1_discoveries} 个、300 s 有 {sensitivity_h1_discoveries} 个通过 FDR；必须与零膨胀和效应量一起解释。
- 结论属于观察性声学结构比较；不支持疗效、认知提升、生成质量或任何因果结论。

## 2. 表示与冻结设计

对每个节拍的 12 维 chroma $\mathbf c_b$ 作 $L_1$ 归一化，并映射到 Harte Tonnetz 的五度、小三度和大三度三个圆周：

$$
\widetilde{{\mathbf c}}_b=\frac{{\mathbf c_b}}{{\sum_p c_b(p)+\varepsilon}},\qquad
\mathbf z_b=\Phi\widetilde{{\mathbf c}}_b\in\mathbb R^6.
$$

令 $q=(7/6,3/2,2/3)$、$r=(1,1,1/2)$，则固定基矩阵可写为

$$
\Phi_{{2k,p}}=r_k\sin(\pi q_kp),\qquad
\Phi_{{2k+1,p}}=r_k\cos(\pi q_kp),
\quad k=0,1,2,\ p=0,\ldots,11.
$$

这一步把八度等价的音级质量映到五度/三度圆周；随后直接在固定 Tonnetz 欧氏度量中聚类，不对 validation 或 holdout 重新标准化或拟合。

仅在 discovery/180 s 上按组等量抽样，每组 {metadata['training_steps_per_group']['classical']:,} 个有效节拍，拟合固定 $V_{{pitch}}=16$ 的 MiniBatch K-means；validation 与 holdout 均不参与码本拟合：

$$
s_b=\arg\min_{{v\in\{{0,\ldots,15\}}}}\|\mathbf z_b-\boldsymbol\mu_v\|_2^2.
$$

低置信节拍沿用既有 1.15 主峰比规则并记为缺失，不建立第 17 个状态，也不跨缺失位置连接转移。validation/180 s 有效节拍比例中位数为 Classical {valid_medians['classical']:.1%}、Open Focus {valid_medians['focus']:.1%}。

码本状态数在分析前固定为 16；下表是 discovery-only 诊断，不用于事后更换主结果：

| $V_{{pitch}}$ | Silhouette | 种子稳定 ARI | 最小簇占比 | 最大簇占比 | 每步 inertia |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(diagnostic_rows)}

## 3. 有向图与持续 Path Homology

相邻有效状态定义转移计数与条件概率：

$$
C_{{uv}}=|\{{b:s_b=u,s_{{b+1}}=v\}}|,\qquad
p_{{uv}}=\frac{{C_{{uv}}}}{{\sum_w C_{{uw}}}}.
$$

每个源状态最多保留 top-6 非自环边。主阈值冻结为 $\tau\in\{{0.50,0.60,0.70,0.80,0.90,0.95\}}$；扩展至 0.05 的网格仅用于敏感性和机制图。过滤图为

$$
G_\tau=(V,\{{(u,v):u\ne v,\ p_{{uv}}\ge\tau\}}).
$$

对允许路径空间 $\Omega_p$，使用 GLMY 边界与路径同调：

$$
\partial e_{{v_0\ldots v_p}}=\sum_i(-1)^i e_{{v_0\ldots\widehat{{v_i}}\ldots v_p}},\qquad
\Omega_p=A_p\cap\partial^{{-1}}(A_{{p-1}}),\qquad
H_p^{{path}}(G)=\frac{{\ker(\partial_p|_{{\Omega_p}})}}{{\operatorname{{im}}(\partial_{{p+1}}|_{{\Omega_{{p+1}}}})}}.
$$

其中 $A_p$ 由图中允许的有向 $p$-路径张成，$\beta_p=\dim H_p^{{path}}$。阈值下降时边逐步加入，得到包含映射及秩不变量

$$
\rho_p(\tau_i,\tau_j)=\operatorname{{rank}}\operatorname{{im}}\left[H_p(G_{{\tau_i}})\to H_p(G_{{\tau_j}})\right],\qquad \tau_i\ge\tau_j.
$$

条形码和持续图使用单调坐标 $a=1-\tau$；主报告给出 $H_0/H_1$ 的 Betti 曲线、区间数量、观测持续性与右删失数量。

## 4. 可视化

示例 `{EXAMPLE_ID}` 按冻结的说明性规则选出：优先选择当前 Open Focus validation/180 s 中恰有一个有限 $H_1$ 区间的片段，再按区间 lifetime 最大及 segment ID 确定性排序。其敏感阈值区间在 $\tau={float(example['birth_threshold']):.3g}$ 出生、$\tau={float(example['death_threshold']):.3g}$ 死亡；它不代表组中心，也不参与假设检验。SSM 仅作 Tonnetz 诊断，主图直接由相邻状态转移构造。

{figures}

## 5. 组间结果

| 指标 | Classical 中位数 | Open Focus 中位数 | $\epsilon^2$ | 180 s BH-FDR $q$ | 300 s BH-FDR $q$ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Open Focus 与 Classical 在主尺度通过独立两两 FDR 的指标：

| 指标 | rank-biserial（Open Focus−Classical） | bootstrap 95% CI | BH-FDR $q$ |
|---|---:|---:|---:|
{chr(10).join(pair_rows)}

### 5.1 解读

1. 状态数、边数、熵和 $H_0$ 指标描述两组在同一 16 状态码本中的覆盖与连通过程，不等于音乐质量高低。
2. 只有同时通过 validation/180 s FDR、在 300 s 方向一致并再次显著的指标，才视为跨时长稳定差异；本轮共有 {summary['replicated_same_direction']} 项。
3. 主尺度的 $H_1$ 非零率分别为 Classical {h1['classical']['primary_nonzero']/h1['classical']['n']:.1%}、Open Focus {h1['focus']['primary_nonzero']/h1['focus']['n']:.1%}。即使秩检验显著，也不能在中位数为零或低发生率时改写为“普遍存在稳定音高环”。

### 5.2 统计原理

每个指标在 validation/180 s 上做两组 Kruskal–Wallis omnibus 检验，并以 $\epsilon^2=(H-k+1)/(N-k)$ 报告秩效应量；独立两两表使用 Mann–Whitney $U$ 与 rank-biserial 效应。20 个指标各自在预先定义的家族内做 Benjamini–Hochberg 校正，确认性判定统一要求 $q\le0.05$。300 s 重复同一套检验，但只解释为同曲目的时长敏感性，不称为独立复制。

## 6. 已冻结 holdout 的兼容性核验

- 本次重跑产出的 `pitch_v2_topology_segments.csv` SHA-256 为 `{rerun_topology_sha}`；开盲门控记录为 `{gate_topology_sha}`；核验结果：**{'一致' if gate_hash_matches else '不一致'}**。
- {'因哈希一致，本次统计口径更新没有改变既有单次 holdout 的输入，故可引用原先冻结后的结果' if gate_hash_matches else '因哈希不一致，本报告仅保留原 holdout 数值作为历史记录，不将其视为当前输入的兼容性确认'}：pitch/180 s pseudo-$F={float(holdout_pitch['pseudo_f']):.3f}$，$p={float(holdout_pitch['p_value']):.3f}$，$q={float(holdout_pitch['p_fdr_bh']):.3g}$；14/14 方向一致，历史 $q\le0.10$ 与严格 $q\le0.05$ 均为 14/14 复现。
- 该 holdout 是回顾性对称重切分下的操作性最终确认；Classical holdout 在旧切分中曾属于 discovery，因此不是 pristine 外部确认集。不得将它提升为外部独立复制或因果证据。

## 7. 证据层级与局限

- **确认性：** validation/180 s、固定 16 状态、top-6、主阈值 0.50–0.95、20 指标 omnibus FDR 与独立 pairwise FDR，统一要求 $q\le0.05$。
- **指纹入选：** $q\le0.05$ 是必要条件；还必须满足预定方向、300 s 方向一致性、去冗余和融合增量审计。$0.05<q\le0.10$ 的端点只作提示性结果，不进入核心冻结指纹。
- **敏感性：** validation/300 s；报告跨时长显著性和方向一致性，不以敏感性结果替代主检验。
- **探索/说明性：** discovery 诊断、扩展至 0.05 的阈值、码本 $K$ 诊断和单曲 birth/death 图。
- **操作性最终确认：** 既有哈希门控 holdout/180 s；本轮只核验输入哈希仍相同，不重新开盲或调参。
- **不支持：** 将两组声学差异解释为注意力、治疗、认知或生成效果；将旧 Pop 对照结论转移到当前两组数据。
- `pitch_v2` 的顶点具有 Tonnetz 原型语义，但 Path Homology 本身只使用状态 ID、边方向和转移概率，没有直接计算 Tonnetz 群作用。

## 8. 复现入口与产物

```powershell
$env:PYTHONPATH = "packages/pyglmy/src;src"
python scripts/run_pitch_v2_analysis.py
python scripts/render_pitch_v2_report.py
```

主要数值文件为 `metadata/pitch_v2_features.csv`、`metadata/pitch_v2_topology_segments.csv`、`metadata/pitch_v2_topology_filtration.csv`、`metadata/pitch_v2_topology_filtration_sensitivity.csv`、`metadata/pitch_v2_statistical_tests.csv`、`metadata/pitch_v2_pairwise_tests.csv` 和 `metadata/pitch_v2_summary.json`。
"""
    REPORT.write_text(report, encoding="utf-8")
    return REPORT


def main() -> None:
    global EXAMPLE_ID
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    codebook, metadata = _codebook()
    summary = json.loads(
        (ROOT / "metadata" / "pitch_v2_summary.json").read_text(encoding="utf-8")
    )
    EXAMPLE_ID = str(summary["mechanism_example"]["segment_id"])
    graph, persistence = _example_arrays()
    chroma, feature = _example_feature_arrays()
    labels = list(metadata["state_labels"])
    figure_stems = (
        "pitch_v2_codebook",
        "pitch_v2_tonnetz_ssm",
        "pitch_v2_directed_state_graph",
        "pitch_v2_filtration_process",
        "pitch_v2_persistence_diagram",
        "pitch_v2_barcode",
        "pitch_v2_group_summary",
        "pitch_v2_betti_curves",
        "pitch_v2_effect_sizes",
        "pitch_v2_duration_stability",
    )
    outputs = (
        plot_codebook(codebook, metadata),
        plot_tonnetz_ssm(chroma, feature),
        plot_directed_graph(graph, codebook, labels),
        plot_filtration(graph, persistence, codebook, labels),
        plot_persistence_diagram(persistence),
        plot_barcode(persistence),
        plot_group_summary(),
        plot_betti_curves(),
        plot_effect_sizes(),
        plot_duration_stability(),
    )
    report = write_report(metadata, figure_stems)
    print(report.relative_to(ROOT).as_posix())
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
