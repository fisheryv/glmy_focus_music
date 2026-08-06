# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features.batch import _sha256, _write_json_atomic
from topology.statistics import (
    TOPOLOGY_METRICS,
    _load_topology_frame,
    _omnibus_and_pairwise,
    permanova_mahalanobis,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "pitch_path_homology_open"
METADATA = ROOT / "metadata"
SEGMENTS = METADATA / "pitch_topology_segments.csv"
FILTRATION = METADATA / "pitch_topology_filtration.csv"
SENSITIVITY = METADATA / "pitch_topology_filtration_sensitivity.csv"
TESTS = METADATA / "pitch_statistical_tests.csv"
PAIRWISE = METADATA / "pitch_pairwise_tests.csv"
PERMANOVA = METADATA / "pitch_permanova.csv"
STABILITY = METADATA / "pitch_scale_stability.csv"
SUMMARY = METADATA / "pitch_path_homology_analysis_summary.json"
REPORT = ROOT / "docs" / "path-homology-pitch-analysis.md"

COLORS = {"classical": "#D95F02", "focus": "#2B6CB0"}
GROUP_LABELS = {"classical": "Classical", "focus": "Open Focus"}
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
METRIC_LABELS = {
    "self_transition_ratio": "Self-transition ratio",
    "vertex_count": "Observed states",
    "edge_count": "Directed edges",
    "edge_density": "Edge density",
    "reciprocity": "Reciprocity",
    "transition_entropy": "Transition entropy",
    "path_entropy": "Path entropy",
    "directed_recurrence": "Directed recurrence",
    "h0_betti_auc": "H0 Betti AUC",
    "h1_betti_auc": "H1 Betti AUC",
    "h0_betti_mean": "Mean beta0",
    "h1_betti_mean": "Mean beta1",
    "h0_betti_max": "Maximum beta0",
    "h1_betti_max": "Maximum beta1",
    "h0_interval_count": "H0 interval count",
    "h1_interval_count": "H1 interval count",
    "h0_observed_persistence": "H0 observed persistence",
    "h1_observed_persistence": "H1 observed persistence",
    "h0_censored_count": "H0 censored count",
    "h1_censored_count": "H1 censored count",
}
METRIC_LABELS_ZH = {
    "self_transition_ratio": "自转移比",
    "vertex_count": "观察状态数",
    "edge_count": "有向边数",
    "edge_density": "边密度",
    "reciprocity": "互惠性",
    "transition_entropy": "转移熵",
    "path_entropy": "路径熵",
    "directed_recurrence": "有向复现度",
    "h0_betti_auc": "H0 Betti AUC",
    "h1_betti_auc": "H1 Betti AUC",
    "h0_betti_mean": "平均 beta0",
    "h1_betti_mean": "平均 beta1",
    "h0_betti_max": "最大 beta0",
    "h1_betti_max": "最大 beta1",
    "h0_interval_count": "H0 区间数",
    "h1_interval_count": "H1 区间数",
    "h0_observed_persistence": "H0 观察持久量",
    "h1_observed_persistence": "H1 观察持久量",
    "h0_censored_count": "H0 右删失数",
    "h1_censored_count": "H1 右删失数",
}


def _configure_matplotlib() -> None:
    cache = OUTPUT / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "pitch-path-homology-open-20260802"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _save(figure, stem: str) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / f"{stem}.png", OUTPUT / f"{stem}.svg"]
    for path in paths:
        figure.savefig(path, dpi=210, bbox_inches="tight", metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(figure)
    return paths


def _select_representative(topology: pd.DataFrame) -> pd.Series:
    validation = topology[
        (topology["split"] == "validation") & (topology["scale_seconds"] == 180.0)
    ].copy()
    nonzero = validation[validation["h1_betti_max"] > 0]
    if not nonzero.empty:
        return nonzero.sort_values(
            ["h1_observed_persistence", "path_entropy"], ascending=False
        ).iloc[0]
    sensitivity = pd.read_csv(SENSITIVITY)
    h1_max = (
        sensitivity.groupby("segment_id", as_index=False)["h1_betti"]
        .max()
        .rename(columns={"h1_betti": "sensitivity_h1_max"})
    )
    candidates = validation.merge(h1_max, on="segment_id")
    return candidates.sort_values(["sensitivity_h1_max", "path_entropy"], ascending=False).iloc[0]


def _pitch_ssm(chroma: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    normalized = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 1e-12)
    return np.clip(normalized @ normalized.T, 0.0, 1.0)


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


def _draw_graph(axis, graph: dict[str, np.ndarray], *, threshold: float, labels: bool) -> None:
    from matplotlib.patches import FancyArrowPatch

    vertices = graph["vertices"].astype(int)
    positions = _layout(vertices)
    edges = zip(
        graph["edge_source"].astype(int),
        graph["edge_target"].astype(int),
        graph["edge_weight"].astype(float),
        strict=True,
    )
    for source, target, weight in edges:
        if weight < threshold or source == target:
            continue
        start, end = positions[source], positions[target]
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8 + 5 * weight,
            linewidth=0.45 + 2.8 * weight,
            color="#46647A",
            alpha=0.24 + 0.68 * weight,
            shrinkA=13,
            shrinkB=13,
            connectionstyle="arc3,rad=0.07",
        )
        axis.add_patch(patch)
        if labels and weight >= 0.15:
            midpoint = (start + end) / 2.0
            axis.text(
                midpoint[0],
                midpoint[1],
                f"{weight:.2f}",
                fontsize=6.3,
                color="#263B4A",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.72,
                },
            )
    for state, position in positions.items():
        face = "#C44E52" if state == 12 else "#F6C85F"
        axis.scatter(position[0], position[1], s=520, color=face, edgecolor="#263B4A", zorder=5)
        axis.text(
            position[0],
            position[1],
            PITCH_LABELS[state],
            ha="center",
            va="center",
            fontsize=8,
            zorder=6,
        )
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_chromagram_ssm(feature: dict[str, np.ndarray], segment_id: str) -> list[Path]:
    import matplotlib.pyplot as plt

    chroma = feature["chroma"].astype(float)
    states = feature["states"].astype(int)
    times = feature["times"].astype(float)
    similarity = _pitch_ssm(chroma)
    left, right = max(0.0, float(times[0])), float(times[-1])
    figure = plt.figure(figsize=(10.0, 9.1), constrained_layout=True)
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
    axis_chroma.set_title(f"Beat-synchronous chromagram: {segment_id}")
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
        extent=(left, right, left, right),
        aspect="auto",
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
    return _save(figure, "pitch_chromagram_ssm")


def plot_directed_graph(graph: dict[str, np.ndarray], segment_id: str) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.3, 8.0), constrained_layout=True)
    _draw_graph(axis, graph, threshold=0.0, labels=True)
    axis.set_title(
        f"Full directed pitch-state graph: {segment_id}\n"
        "edge width = outgoing probability; labels shown for p ≥ 0.15",
        fontsize=12,
    )
    return _save(figure, "pitch_directed_state_graph")


def _filtration_snapshots(persistence: dict[str, np.ndarray]) -> list[tuple[int, str]]:
    thresholds = persistence["thresholds"].astype(float)
    h1 = persistence["h1_betti"].astype(int)
    nonzero = np.flatnonzero(h1 > 0)
    if nonzero.size == 0:
        candidates = [0, len(thresholds) // 2, len(thresholds) - 1]
        return [(index, "sampled level") for index in sorted(set(candidates))]
    birth = int(nonzero[0])
    before = max(0, birth - 1)
    after_candidates = np.flatnonzero((np.arange(len(h1)) > birth) & (h1 == 0))
    after = int(after_candidates[0]) if after_candidates.size else len(h1) - 1
    return [(before, "before H1"), (birth, "H1 present"), (after, "after H1")]


def plot_filtration(graph: dict[str, np.ndarray], persistence: dict[str, np.ndarray]) -> list[Path]:
    import matplotlib.pyplot as plt

    snapshots = _filtration_snapshots(persistence)
    thresholds = persistence["thresholds"].astype(float)
    h0 = persistence["h0_betti"].astype(int)
    h1 = persistence["h1_betti"].astype(int)
    figure, axes = plt.subplots(1, len(snapshots), figsize=(14.0, 4.7), constrained_layout=True)
    for axis, (index, label) in zip(np.atleast_1d(axes), snapshots, strict=True):
        threshold = float(thresholds[index])
        _draw_graph(axis, graph, threshold=threshold, labels=False)
        axis.set_title(
            f"τ={threshold:.3f}: {label}\n"
            f"edges={int(persistence['edge_count'][index])}, β0={h0[index]}, β1={h1[index]}",
            fontsize=10,
        )
    figure.suptitle("Descending-threshold persistent path-homology filtration", fontsize=13)
    return _save(figure, "pitch_filtration_process")


def _expanded_intervals(persistence: dict[str, np.ndarray]) -> list[dict[str, object]]:
    intervals: list[dict[str, object]] = []
    end = 1.0 - float(np.min(persistence["thresholds"]))
    for index, dimension in enumerate(persistence["interval_dimension"].astype(int)):
        birth_threshold = float(persistence["interval_birth_threshold"][index])
        death_threshold = float(persistence["interval_death_threshold"][index])
        censored = bool(persistence["interval_censored"][index])
        birth = 1.0 - birth_threshold
        death = end if censored else 1.0 - death_threshold
        for _ in range(int(persistence["interval_multiplicity"][index])):
            intervals.append(
                {"dimension": dimension, "birth": birth, "death": death, "censored": censored}
            )
    return intervals


def plot_persistence_diagram(persistence: dict[str, np.ndarray], segment_id: str) -> list[Path]:
    import matplotlib.pyplot as plt

    intervals = _expanded_intervals(persistence)
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
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
        xlabel="Birth a = 1 − τ",
        ylabel="Death a = 1 − τ",
        title=f"Persistent path-homology diagram: {segment_id}",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, "pitch_persistence_diagram")


def plot_barcode(persistence: dict[str, np.ndarray], segment_id: str) -> list[Path]:
    import matplotlib.pyplot as plt

    intervals = _expanded_intervals(persistence)
    intervals.sort(
        key=lambda item: (int(item["dimension"]), float(item["birth"]), float(item["death"]))
    )
    end = 1.0 - float(np.min(persistence["thresholds"]))
    figure, axis = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
    for row, item in enumerate(intervals):
        dimension = int(item["dimension"])
        color = "#4472C4" if dimension == 0 else "#C44E52"
        start, stop = float(item["birth"]), float(item["death"])
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
        xlabel="Filtration coordinate a = 1 − τ",
        ylabel="Interval index",
        title=f"Persistent path barcode: {segment_id}",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.02,
        0.96,
        "blue: H0    red: H1    open circle: censored",
        transform=axis.transAxes,
        va="top",
    )
    return _save(figure, "pitch_barcode")


def plot_group_summary(topology: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    data = topology[(topology["split"] == "validation") & (topology["scale_seconds"] == 180.0)]
    metrics = (
        "vertex_count",
        "edge_count",
        "self_transition_ratio",
        "path_entropy",
        "directed_recurrence",
        "h0_observed_persistence",
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        values = [
            data.loc[data.group == group, metric].dropna().to_numpy()
            for group in ("focus", "classical")
        ]
        boxes = axis.boxplot(values, patch_artist=True, widths=0.62, showfliers=False)
        for patch, group in zip(boxes["boxes"], ("focus", "classical"), strict=True):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.68)
        axis.set_xticks([1, 2], [GROUP_LABELS["focus"], GROUP_LABELS["classical"]], rotation=15)
        axis.set_title(METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Pitch-view group comparison (validation, 180s)", fontsize=13)
    return _save(figure, "pitch_group_summary")


def plot_betti_curves(filtration: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    data = filtration[
        (filtration["split"] == "validation") & (filtration["scale_seconds"] == 180.0)
    ].copy()
    data["a"] = 1.0 - data["threshold"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("h0_betti", "h1_betti"), ("Mean beta0", "Mean beta1"), strict=True
    ):
        for group in ("focus", "classical"):
            selected = data[data.group == group]
            summary = (
                selected.groupby("a")[metric].agg(["mean", "sem"]).reset_index().sort_values("a")
            )
            x = summary["a"].to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            sem = summary["sem"].fillna(0).to_numpy(float)
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
        axis.set(title=title, xlabel="Filtration coordinate a = 1 − τ", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Pitch-view Betti curves (validation, 180s; mean ± SEM)", fontsize=13)
    return _save(figure, "pitch_betti_curves")


def _bootstrap_effects(
    topology: pd.DataFrame, metrics: list[str], *, repetitions: int = 2000
) -> pd.DataFrame:
    subset = topology[(topology["split"] == "validation") & (topology["scale_seconds"] == 180.0)]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        focus = subset.loc[subset.group == "focus", metric].to_numpy(float)
        classical = subset.loc[subset.group == "classical", metric].to_numpy(float)
        effect = np.mean(focus[:, None] > classical[None, :]) - np.mean(
            focus[:, None] < classical[None, :]
        )
        digest = hashlib.sha256(f"20260716:pitch:{metric}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        sampled = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            left = rng.choice(focus, len(focus), replace=True)
            right = rng.choice(classical, len(classical), replace=True)
            sampled[index] = np.mean(left[:, None] > right[None, :]) - np.mean(
                left[:, None] < right[None, :]
            )
        low, high = np.quantile(sampled, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "effect_focus_minus_classical": effect,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_effect_sizes(topology: pd.DataFrame, tests: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    primary = tests[tests.analysis_set == "primary_validation_180"].copy()
    metrics = primary.sort_values("epsilon_squared", ascending=False).metric.head(14).tolist()
    effects = _bootstrap_effects(topology, metrics).merge(
        primary[["metric", "p_fdr_bh"]], on="metric"
    )
    effects = effects.sort_values("effect_focus_minus_classical")
    y = np.arange(len(effects))
    values = effects.effect_focus_minus_classical.to_numpy(float)
    low = effects.ci_low.to_numpy(float)
    high = effects.ci_high.to_numpy(float)
    figure, axis = plt.subplots(figsize=(9.8, 7.2), constrained_layout=True)
    axis.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="o",
        color="#4C4C4C",
        capsize=3,
    )
    significant = effects.p_fdr_bh.to_numpy(float) <= 0.10
    axis.scatter(
        values[significant], y[significant], s=62, color="#2B6CB0", zorder=4, label="BH q ≤ 0.10"
    )
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
    axis.set_yticks(y, [METRIC_LABELS[name] for name in effects.metric])
    axis.set_xlabel("Rank-biserial effect (Open Focus − Classical), bootstrap 95% CI")
    axis.set_title("Largest pitch-view effects (validation, 180s)")
    axis.set_xlim(-1.05, 1.05)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, "pitch_effect_sizes")


def _permanova(topology: pd.DataFrame) -> pd.DataFrame:
    metrics = list(TOPOLOGY_METRICS)
    reference = topology[(topology["split"] == "discovery") & (topology["scale_seconds"] == 180.0)][
        metrics
    ].to_numpy(float)
    rows: list[dict[str, object]] = []
    for role, scale in (("primary_validation_180", 180.0), ("duration_sensitivity_300", 300.0)):
        subset = topology[
            (topology["split"] == "validation") & (topology["scale_seconds"] == scale)
        ]
        result = permanova_mahalanobis(
            subset[metrics].to_numpy(float),
            subset["group"].to_numpy(),
            permutations=999,
            seed=20260716,
            reference_matrix=reference,
        )
        rows.append(
            {
                "role": role,
                "split": "validation",
                "scale_seconds": scale,
                "n": len(subset),
                "metrics": len(metrics),
                "pseudo_f": result["pseudo_f"],
                "p_value": result["p_value"],
                "permutations": 999,
                "effective_dimensions": result["effective_dimensions"],
                "covariance_reference": "discovery/180s",
            }
        )
    return pd.DataFrame(rows)


def _scale_stability(topology: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in ("focus", "classical"):
        for metric in TOPOLOGY_METRICS:
            subset = topology[(topology["split"] == "validation") & (topology["group"] == group)]
            paired = subset.pivot(index="track_id", columns="scale_seconds", values=metric).dropna()
            if (
                np.ptp(paired[180.0].to_numpy(float)) == 0
                or np.ptp(paired[300.0].to_numpy(float)) == 0
            ):
                statistic, p_value = np.nan, np.nan
            else:
                result = spearmanr(paired[180.0], paired[300.0])
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_tracks": len(paired),
                    "spearman_rho_180_vs_300": statistic,
                    "p_value": p_value,
                    "median_300_minus_180": float(np.median(paired[300.0] - paired[180.0])),
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(columns))) + "|",
    ]
    for row in frame.itertuples(index=False):
        rendered: list[str] = []
        for key, _ in columns:
            value = getattr(row, key)
            if isinstance(value, (float, np.floating)):
                rendered.append(_fmt(float(value)))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _write_report(
    topology: pd.DataFrame,
    tests: pd.DataFrame,
    pairwise: pd.DataFrame,
    permanova: pd.DataFrame,
    stability: pd.DataFrame,
    representative: pd.Series,
    feature: dict[str, np.ndarray],
    sensitivity_persistence: dict[str, np.ndarray],
    figures: list[Path],
) -> None:
    primary = tests[tests.analysis_set == "primary_validation_180"].copy()
    duration = tests[tests.analysis_set == "sensitivity_validation_300"].copy()
    merged = primary.merge(
        duration[["metric", "focus_median", "classical_median", "p_fdr_bh"]],
        on="metric",
        suffixes=("_180", "_300"),
    )
    primary_pair = pairwise[pairwise.analysis_set == "primary_validation_180"][
        ["metric", "rank_biserial_a_minus_b"]
    ].copy()
    primary_pair["effect_focus_minus_classical"] = -primary_pair["rank_biserial_a_minus_b"]
    merged = merged.merge(primary_pair[["metric", "effect_focus_minus_classical"]], on="metric")
    duration_pair = pairwise[pairwise.analysis_set == "sensitivity_validation_300"][
        ["metric", "rank_biserial_a_minus_b"]
    ].copy()
    duration_pair["effect_300"] = -duration_pair["rank_biserial_a_minus_b"]
    merged = merged.merge(duration_pair[["metric", "effect_300"]], on="metric")
    merged["metric_label"] = merged.metric.map(METRIC_LABELS_ZH)
    merged = merged.sort_values(["p_fdr_bh_180", "metric"])
    significant = primary[primary.p_fdr_bh <= 0.10]
    significant_metrics = significant.metric.tolist()
    duration_lookup = duration.set_index("metric")
    replicated = [
        metric
        for metric in significant_metrics
        if duration_lookup.loc[metric, "p_fdr_bh"] <= 0.10
        and np.sign(merged.loc[merged.metric == metric, "effect_focus_minus_classical"].iloc[0])
        == np.sign(merged.loc[merged.metric == metric, "effect_300"].iloc[0])
    ]
    validation = topology[(topology.split == "validation") & (topology.scale_seconds == 180.0)]
    sensitivity = pd.read_csv(SENSITIVITY)
    sensitivity_validation = sensitivity[
        (sensitivity.split == "validation") & (sensitivity.scale_seconds == 180.0)
    ]
    h1_primary = (
        validation.groupby("group")
        .h1_betti_max.apply(lambda values: int(np.count_nonzero(values > 0)))
        .to_dict()
    )
    h1_sensitivity = (
        sensitivity_validation.groupby(["group", "segment_id"])
        .h1_betti.max()
        .gt(0)
        .groupby("group")
        .sum()
        .astype(int)
        .to_dict()
    )
    uncertain = pd.read_csv(METADATA / "feature_segments.csv")
    uncertain = uncertain[
        (uncertain.split == "validation") & (uncertain.scale_seconds == 180.0)
    ].copy()
    uncertain["uncertain_ratio"] = uncertain.uncertain_pitch_steps / uncertain.pitch_steps
    uncertainty_summary = (
        uncertain.groupby("group")["uncertain_ratio"].agg(["count", "median", "mean"]).reset_index()
    )
    uncertainty_summary["group"] = uncertainty_summary.group.map(GROUP_LABELS)
    holdout = topology[(topology.split == "holdout") & (topology.scale_seconds == 180.0)]
    holdout_metrics = [
        "vertex_count",
        "edge_count",
        "self_transition_ratio",
        "path_entropy",
        "directed_recurrence",
        "h0_observed_persistence",
        "h1_betti_max",
    ]
    holdout_summary = (
        holdout.groupby("group")[holdout_metrics]
        .median()
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    holdout_summary["metric_label"] = holdout_summary.metric.map(METRIC_LABELS_ZH)
    stable_primary = stability[stability.metric.isin(significant_metrics)]
    stability_summary = (
        stable_primary.groupby("group")
        .spearman_rho_180_vs_300.agg(["median", "min", "max"])
        .reset_index()
    )
    stability_summary["group"] = stability_summary.group.map(GROUP_LABELS)
    intervals = _expanded_intervals(sensitivity_persistence)
    h1_intervals = [item for item in intervals if item["dimension"] == 1]
    longest_h1 = max(
        (float(item["death"]) - float(item["birth"]) for item in h1_intervals),
        default=0.0,
    )
    links = {
        path.stem: f"../{path.relative_to(ROOT).as_posix()}"
        for path in figures
        if path.suffix == ".png"
    }
    report_table = merged[
        [
            "metric_label",
            "classical_median_180",
            "focus_median_180",
            "effect_focus_minus_classical",
            "p_fdr_bh_180",
            "effect_300",
            "p_fdr_bh_300",
        ]
    ].copy()
    report_table.columns = [
        "metric_label",
        "classical_180",
        "focus_180",
        "effect_180",
        "q_180",
        "effect_300",
        "q_300",
    ]
    representative_uncertain = float(np.mean(feature["states"].astype(int) == 12))
    lines = [
        "# 音高视角 Path Homology：Open Focus 与 Classical 重新分析",
        "",
        "生成日期：2026-08-02。分析对象为当前 Open Focus 300 首与 Classical 300 首的两组规范数据。",
        "",
        "## 摘要",
        "",
        f"本次按冻结配置重新计算全部 1,200 个 pitch 片段视图，覆盖 600 首曲目的 180s/300s 版本，成功 1,200、失败 0。validation/180s 的 20 个预设指标中有 {len(significant_metrics)} 个通过 BH-FDR q≤0.10；其中 {len(replicated)} 个在同曲目的 validation/300s 中方向一致并再次通过 FDR。20 指标联合 Mahalanobis PERMANOVA 得到 pseudo-F={permanova.iloc[0].pseudo_f:.3f}、p={permanova.iloc[0].p_value:.3f}。主要差异来自状态字母表、边数、H0 连通过程、路径熵和转移集中度，而不是稳定 H1。",
        "",
        f"主阈值 0.50–0.95 下，validation/180s 非零 H1 仅为 Open Focus {h1_primary.get('focus', 0)}/60、Classical {h1_primary.get('classical', 0)}/60；扩展阈值至 0.05 后变为 Open Focus {h1_sensitivity.get('focus', 0)}/60、Classical {h1_sensitivity.get('classical', 0)}/60。故不能声称存在稳健或 Focus 特异的音高 H1。当前 validation 已在其他项目分析中被查看，本报告是冻结方法在迁移后数据上的观察性重新分析，而非新的 pristine 确认性实验。",
        "",
        "## 1. 方法思想",
        "",
        "音高视角不直接比较曲调名称或调性标签，而是把每个节拍区间编码成主导音级状态，再研究状态之间的有向转移。强边反映某个音级状态后经常出现的下一状态；随着转移概率阈值降低，弱边逐步进入图，连通分量合并，并可能形成或填充有向一维路径同调类。",
        "",
        "```mermaid",
        "flowchart LR",
        '    A["音频"] --> B["谐波 chroma"]',
        '    B --> C["节拍同步池化"]',
        '    C --> D["12 音级 + 不确定态 U"]',
        '    D --> E["相邻状态转移计数"]',
        '    E --> F["按源状态归一化 + top-6"]',
        '    F --> G["超水平过滤 G_tau"]',
        '    G --> H["GLMY H0/H1 与持久区间"]',
        "```",
        "",
        "## 2. 音高状态表示",
        "",
        "### 2.1 节拍同步 chroma",
        "",
        "把 STFT 谐波功率谱按八度折叠到 12 个音级。对第 b 个相邻节拍区间的帧集合 I_b，池化表示为",
        "",
        r"$$\bar{\mathbf c}_b=\frac{1}{|I_b|}\sum_{t\in I_b}\mathbf c_t,\qquad \bar{\mathbf c}_b\in\mathbb R_{\ge0}^{12}.$$",
        "",
        "节拍同步减少逐帧颤音、起音偏移和局部时间伸缩造成的状态抖动，但其质量仍依赖节拍估计。",
        "",
        "### 2.2 主导音级与不确定态",
        "",
        "令 c_b^(1)、c_b^(2) 为最大和次大 chroma 分量，冻结的不确定比为 1.15：",
        "",
        r"$$s_b=\begin{cases}\arg\max_{p\in\{0,\ldots,11\}}\bar c_b(p),&c_b^{(1)}/\max(c_b^{(2)},10^{-8})\ge1.15\ \text{且}\ c_b^{(1)}>10^{-8},\\U,&\text{其他情况}.\end{cases}$$",
        "",
        "U 编码为 12。虽然特征文件还保存 `valid = states != 12`，当前研究批处理将所有非负整数状态都作为图顶点，因此 U 是第 13 个可观测状态，而不是缺失值。",
        "",
        *_markdown_table(
            uncertainty_summary,
            [("group", "组别"), ("count", "n"), ("median", "U 比例中位数"), ("mean", "U 比例均值")],
        ),
        "",
        "### 2.3 音高自相似矩阵",
        "",
        r"对解释性图形，先令 $\widehat{\mathbf c}_i=\bar{\mathbf c}_i/(\|\bar{\mathbf c}_i\|_2+\varepsilon)$，再计算 $S_{ij}^{\mathrm{pitch}}=\widehat{\mathbf c}_i^{\mathsf T}\widehat{\mathbf c}_j$。远离对角线的亮块表示不同时段具有相近的音级能量配置；SSM 不参与状态图边权或组间检验。",
        "",
        f"![Pitch chromagram and SSM]({links['pitch_chromagram_ssm']})",
        "",
        f"代表片段为 `{representative.segment_id}`（{GROUP_LABELS[str(representative.group)]}，validation/180s）；U 比例为 {representative_uncertain:.3f}。",
        "",
        "## 3. 有向状态图",
        "",
        "相邻状态转移计数与按源状态归一化概率为",
        "",
        r"$$C_{uv}=|\{t:s_t=u,\ s_{t+1}=v\}|,\qquad p_{uv}=\frac{C_{uv}}{\sum_w C_{uw}}.$$",
        "",
        "每个源状态最多保留概率最大的 6 条非自环边。自转移不进入 GLMY 正则路径图，但仍以描述量保留：",
        "",
        r"$$r_{\mathrm{self}}=\frac{\sum_u C_{uu}}{\sum_{u,v}C_{uv}}.$$",
        "",
        "路径熵和有向复现度分别为",
        "",
        r"$$H_{\mathrm{path}}=-\sum_{u,v}\frac{C_{uv}}{N}\log\frac{C_{uv}}{\sum_w C_{uw}},\qquad R_{\mathrm{dir}}=\sum_{u,v}\left(\frac{C_{uv}}N\right)^2.$$",
        "",
        "前者衡量条件转移的多样性，后者衡量概率质量是否集中在少数转移。",
        "",
        f"![Directed pitch graph]({links['pitch_directed_state_graph']})",
        "",
        "## 4. GLMY 路径同调与持续过滤",
        "",
        "对阈值 τ 定义超水平过滤",
        "",
        r"$$G_\tau=(V,E_\tau),\qquad E_\tau=\{(u,v):u\ne v,\ p_{uv}\ge\tau\}.$$",
        "",
        "主分析阈值冻结为 {0.50,0.60,0.70,0.80,0.90,0.95}；敏感性阈值扩展到 {0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95}。实现按超水平方向从高阈值向低阈值加入边。",
        "",
        "允许的正则 p-路径 e_(v0...vp) 要求相邻顶点均由有向边连接。边界算子为",
        "",
        r"$$\partial e_{v_0\ldots v_p}=\sum_{i=0}^{p}(-1)^i e_{v_0\ldots\widehat{v_i}\ldots v_p}.$$",
        "",
        "删除中间顶点后所得路径可能不再允许，因此链空间限制为",
        "",
        r"$$\Omega_p=\{a\in A_p:\partial a\in A_{p-1}\}.$$",
        "",
        "路径同调群和 Betti 数为",
        "",
        r"$$H_p^{\mathrm{path}}(G)=\frac{\ker(\partial_p|_{\Omega_p})}{\operatorname{im}(\partial_{p+1}|_{\Omega_{p+1}})},\qquad \beta_p=\dim H_p^{\mathrm{path}}(G).$$",
        "",
        "β0 描述弱连通分量；β1 描述未被允许 2-路径边界填充的独立有向一维类，它不等于简单环计数。过滤包含映射诱导同调映射，其秩不变量为",
        "",
        r"$$\rho_p(i,j)=\operatorname{rank}\bigl(H_p(G_i)\to H_p(G_j)\bigr),\qquad i\le j.$$",
        "",
        "持久区间由完整秩不变量分解；在观测末端仍存活的类标为右删失。绘图采用递增坐标 a=1−τ。",
        "",
        f"![Pitch filtration]({links['pitch_filtration_process']})",
        "",
        f"![Pitch persistence diagram]({links['pitch_persistence_diagram']})",
        "",
        f"![Pitch barcode]({links['pitch_barcode']})",
        "",
        f"代表片段的敏感性过滤含 {len(h1_intervals)} 个 H1 区间；最长观测跨度为 {longest_h1:.3f}。该样本用于展示边的加入如何形成或填充路径类，不代表总体典型性。",
        "",
        "## 5. 数据与统计协议",
        "",
        "- 数据：Open Focus 300、Classical 300；每首各有 180s 与 300s。",
        "- 对称切分：每组 discovery 195、validation 60、holdout 45。",
        "- pitch 状态无需拟合码本；1.15 不确定比、top-6、非自环和过滤阈值均来自冻结配置。",
        "- 主要分析：validation/180s；20 指标 Kruskal–Wallis（两组时与双侧 Mann–Whitney 秩检验等价），在 20 指标家族内 BH-FDR q≤0.10。",
        "- 多变量检查：20 指标 Mahalanobis PERMANOVA，协方差只由 discovery/180s 冻结参考估计，999 次置换。",
        "- 时长敏感性：validation/300s；同曲目而非独立复制。",
        "- holdout：只报告描述统计，不根据其结果调参或再次开启显著性家族。",
        "",
        "## 6. validation/180s 主要结果",
        "",
        f"20 个预设指标中 {len(significant_metrics)} 个通过 q≤0.10。正效应表示 Open Focus 倾向更高，负效应表示 Classical 倾向更高。",
        "",
        *_markdown_table(
            report_table,
            [
                ("metric_label", "指标"),
                ("classical_180", "Classical 180s"),
                ("focus_180", "Focus 180s"),
                ("effect_180", "180s 效应"),
                ("q_180", "180s FDR"),
                ("effect_300", "300s 效应"),
                ("q_300", "300s FDR"),
            ],
        ),
        "",
        f"![Pitch group summary]({links['pitch_group_summary']})",
        "",
        f"![Pitch effect sizes]({links['pitch_effect_sizes']})",
        "",
        "方向最稳定的模式是：Classical 具有更多状态、更多边、更高路径熵和更大的 H0 连通持久过程；Open Focus 具有更高自转移、互惠性、边密度和有向复现度。这是音高状态组织差异，不是价值排序。H0 较高也不能单独解释为“更复杂”，因为它强烈受状态字母表大小影响。",
        "",
        "## 7. 多变量结果",
        "",
        *_markdown_table(
            permanova,
            [
                ("role", "角色"),
                ("scale_seconds", "时长"),
                ("n", "n"),
                ("pseudo_f", "pseudo-F"),
                ("p_value", "置换 p"),
                ("effective_dimensions", "有效维数"),
            ],
        ),
        "",
        "PERMANOVA 支持两组在联合 pitch 拓扑描述空间中可分，但不能判断哪一单项指标是原因，也不能排除配器、古典子类型、制作方式和状态数等混杂。",
        "",
        "## 8. H0/H1 过滤行为",
        "",
        f"![Pitch Betti curves]({links['pitch_betti_curves']})",
        "",
        f"主阈值下 H1 几乎完全零膨胀（Focus {h1_primary.get('focus', 0)}/60；Classical {h1_primary.get('classical', 0)}/60）。扩展到低概率边后，H1 出现率反而是 Classical {h1_sensitivity.get('classical', 0)}/60、Focus {h1_sensitivity.get('focus', 0)}/60。这个方向变化和阈值依赖说明：H1 适合做单曲解释与敏感性诊断，不适合作为稳健组间核心结论。",
        "",
        "## 9. 180s/300s 稳定性",
        "",
        *_markdown_table(
            stability_summary,
            [
                ("group", "组别"),
                ("median", "显著指标 ρ 中位数"),
                ("min", "最小 ρ"),
                ("max", "最大 ρ"),
            ],
        ),
        "",
        f"主要 14 个指标中，{len(replicated)} 个在 300s 中同方向且再次通过 FDR。由于 300s 包含同曲目的前 180s，它只是时长敏感性，不是独立样本复制。逐指标相关见 `metadata/pitch_scale_stability.csv`。",
        "",
        "## 10. holdout 描述",
        "",
        *_markdown_table(
            holdout_summary,
            [
                ("metric_label", "指标"),
                ("classical", "Classical 中位数"),
                ("focus", "Focus 中位数"),
            ],
        ),
        "",
        "holdout 未用于本报告的新显著性检验。Classical holdout 不含钢琴独奏，且其中曲目在旧切分中曾属于 discovery，不能称为 pristine 外部确认集。",
        "",
        "## 11. 结论与证据边界",
        "",
        "### 支持",
        "",
        "- 当前 Open Focus 与 Classical 在节拍级主导音高状态的有向组织上存在强而稳定的观察性差异。",
        "- 差异主要由状态覆盖、转移网络规模、路径熵、转移集中度以及 H0 连通过程贡献。",
        "- validation/300s 保持绝大多数主要指标的方向和 FDR 结果。",
        "",
        "### 不支持",
        "",
        "- 不支持稳健、普遍或 Focus 特异的 pitch H1；主阈值 H1 极稀疏，低阈值结果明显依赖弱边。",
        "- 不支持由 H0 较低或转移更集中直接推出‘更有序’、‘更优’或‘更适合专注’。",
        "- 不支持注意力提升、治疗作用、生成质量或其他因果结论。",
        "",
        "### 局限",
        "",
        "- Chroma 折叠八度并混合旋律、和声、伴奏与泛音，不能替代音符级转录。",
        "- 节拍跟踪误差会改变池化边界与状态序列长度。",
        "- U 同时反映复音、低能量和主峰不明确，不是和弦或休止标签；将 U 改作缺失值会构成另一套分析。",
        "- H0 与可观察状态数高度耦合，必须与状态数、边数和归一化指标联合解释。",
        "- 两组在来源、配器和体裁上不同，组间差异不等于 Focus 功能机制。",
        "",
        "## 12. 可复现产物",
        "",
        "- `scripts/rerun_pitch_path_homology.py`",
        "- `scripts/render_pitch_path_report_current.py`",
        "- `metadata/pitch_topology_segments.csv`",
        "- `metadata/pitch_topology_filtration.csv`",
        "- `metadata/pitch_topology_filtration_sensitivity.csv`",
        "- `metadata/pitch_statistical_tests.csv`",
        "- `metadata/pitch_pairwise_tests.csv`",
        "- `metadata/pitch_permanova.csv`",
        "- `metadata/pitch_scale_stability.csv`",
        "- `metadata/pitch_path_homology_analysis_summary.json`",
        "- `runs/pitch_path_homology_open/`（全部图均有 PNG 与 SVG）",
        "",
        "复现命令：",
        "",
        "```powershell",
        "$env:PYTHONPATH='packages/pathhom_tda/src;src'",
        ".\\.venv\\Scripts\\python.exe scripts\\rerun_pitch_path_homology.py",
        ".\\.venv\\Scripts\\python.exe scripts\\render_pitch_path_report_current.py",
        "```",
        "",
        "## 参考文献",
        "",
        "1. Müller, M. (2015). *Fundamentals of Music Processing*. Springer.",
        "2. Ellis, D. P. W., & Poliner, G. E. (2007). Identifying cover songs with chroma features and dynamic programming beat tracking. *ICASSP*.",
        "3. Grigor'yan, A., Lin, Y., Muranov, Y., & Yau, S.-T. (2012). Homologies of path complexes and digraphs. arXiv:1207.2834.",
        "4. Chowdhury, S., & Mémoli, F. (2018). Persistent path homology of directed networks. *SODA*.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    _configure_matplotlib()
    topology = _load_topology_frame(SEGMENTS)
    expected = topology.groupby(["split", "group", "scale_seconds"]).size()
    if len(topology) != 1200 or set(topology.group) != {"focus", "classical"}:
        raise RuntimeError("pitch topology manifest is not the current 1,200-row two-group run")
    if not (
        expected.loc[("validation", "focus", 180.0)] == 60
        and expected.loc[("validation", "classical", 180.0)] == 60
    ):
        raise RuntimeError("validation/180s is not balanced 60+60")
    tests, pairwise = _omnibus_and_pairwise(topology)
    tests["role"] = tests.role.replace({"confirmatory": "primary_reanalysis"})
    pairwise["role"] = pairwise.role.replace({"confirmatory": "primary_reanalysis"})
    permanova = _permanova(topology)
    stability = _scale_stability(topology)
    _write_csv(TESTS, tests)
    _write_csv(PAIRWISE, pairwise)
    _write_csv(PERMANOVA, permanova)
    _write_csv(STABILITY, stability)

    representative = _select_representative(topology)
    feature = _read_npz(ROOT / Path(representative.feature_relative_path))
    graph = _read_npz(ROOT / Path(representative.graph_relative_path))
    sensitivity_persistence = _read_npz(
        ROOT / Path(representative.sensitivity_persistence_relative_path)
    )
    figures: list[Path] = []
    figures.extend(plot_chromagram_ssm(feature, str(representative.segment_id)))
    figures.extend(plot_directed_graph(graph, str(representative.segment_id)))
    figures.extend(plot_filtration(graph, sensitivity_persistence))
    figures.extend(
        plot_persistence_diagram(sensitivity_persistence, str(representative.segment_id))
    )
    figures.extend(plot_barcode(sensitivity_persistence, str(representative.segment_id)))
    figures.extend(plot_group_summary(topology))
    figures.extend(plot_betti_curves(pd.read_csv(SENSITIVITY)))
    figures.extend(plot_effect_sizes(topology, tests))
    _write_report(
        topology,
        tests,
        pairwise,
        permanova,
        stability,
        representative,
        feature,
        sensitivity_persistence,
        figures,
    )
    artifacts = [
        SEGMENTS,
        FILTRATION,
        SENSITIVITY,
        TESTS,
        PAIRWISE,
        PERMANOVA,
        STABILITY,
        REPORT,
        *figures,
    ]
    primary = tests[tests.analysis_set == "primary_validation_180"]
    duration = tests[tests.analysis_set == "sensitivity_validation_300"]
    primary_sig = primary[primary.p_fdr_bh <= 0.10].metric.tolist()
    duration_sig = duration[duration.p_fdr_bh <= 0.10].metric.tolist()
    payload = {
        "generated_at": "2026-08-02",
        "scope": "standard pitch-view Path Homology on current Open Focus/Classical data",
        "evidence_status": "frozen-method post-migration observational reanalysis",
        "ok": True,
        "segments": int(len(topology)),
        "tracks": int(topology.track_id.nunique()),
        "groups": sorted(topology.group.unique().tolist()),
        "representative_segment_id": str(representative.segment_id),
        "primary_validation_fdr_q": 0.10,
        "primary_validation_discoveries": primary_sig,
        "duration_sensitivity_discoveries": duration_sig,
        "permanova": permanova.to_dict("records"),
        "inputs_sha256": {
            "metadata/feature_segments.csv": _sha256(METADATA / "feature_segments.csv"),
            "configs/pipeline.toml": _sha256(ROOT / "configs" / "pipeline.toml"),
        },
        "outputs": [path.relative_to(ROOT).as_posix() for path in artifacts],
        "output_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in artifacts},
    }
    _write_json_atomic(SUMMARY, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
