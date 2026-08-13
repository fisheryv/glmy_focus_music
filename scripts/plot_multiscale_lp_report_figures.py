from __future__ import annotations

# ruff: noqa: I001

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
FIGURES = ROOT / "runs" / "multiscale_hierarchical_fusion" / "figures"

BLUE = "#4472C4"
ORANGE = "#D95F02"
GREEN = "#70AD47"
GRAY = "#666666"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(METADATA / name)


def _save(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _annotate_bars(ax: plt.Axes, bars: object, values: np.ndarray) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_validation_permanova(frame: pd.DataFrame) -> None:
    order = ["L", "P", "LP"]
    labels = ["Local L", "Phase P", "L + P"]
    figure, ax = plt.subplots(figsize=(7.8, 4.7), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.34
    for offset, scale, color, hatch in (
        (-width / 2, 180.0, BLUE, ""),
        (width / 2, 300.0, GREEN, "//"),
    ):
        subset = frame[
            (frame["scale_seconds"] == scale) & frame["feature_set"].isin(order)
        ].set_index("feature_set")
        values = np.array([subset.loc[name, "pseudo_f"] for name in order])
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=color,
            hatch=hatch,
            edgecolor="white",
            label=f"{int(scale)} s",
        )
        _annotate_bars(ax, bars, values)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Permutation pseudo-F")
    ax.set_title("Two-scale validation group separation")
    ax.set_ylim(0, 34)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(figure, "lp_validation_permanova")


def plot_incremental_and_residual(
    incremental: pd.DataFrame, residuals: pd.DataFrame
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

    inc = incremental[incremental["comparison"] == "LP_minus_L"].sort_values(
        "scale_seconds"
    )
    colors = {180.0: BLUE, 300.0: GREEN}
    for y, row in enumerate(inc.itertuples(index=False)):
        color = colors[row.scale_seconds]
        axes[0].plot(
            [row.null_ci_low, row.null_ci_high],
            [y, y],
            color=color,
            linewidth=7,
            alpha=0.28,
            solid_capstyle="round",
        )
        axes[0].scatter(
            row.delta_pseudo_f,
            y,
            color=color,
            marker="D",
            s=65,
            zorder=3,
        )
        axes[0].text(
            row.delta_pseudo_f + 0.25,
            y,
            f"Δ={row.delta_pseudo_f:.2f}\np={row.p_value_one_sided:.3f}",
            va="center",
            fontsize=8,
        )
    axes[0].axvline(0.0, color=GRAY, linewidth=0.9, linestyle="--")
    axes[0].set_yticks([0, 1], ["180 s", "300 s"])
    axes[0].set_xlabel("Observed Δpseudo-F\n(line: permutation null 95% interval)")
    axes[0].set_title("Increment of L + P over L")
    axes[0].set_xlim(-1.5, 12.5)
    axes[0].grid(axis="x", alpha=0.2)

    res = residuals[residuals["conditional_test"] == "P_given_L"].sort_values(
        "scale_seconds"
    )
    x = np.arange(len(res))
    values = res["pseudo_f"].to_numpy(float)
    bars = axes[1].bar(
        x,
        values,
        color=[colors[scale] for scale in res["scale_seconds"]],
        width=0.55,
    )
    for bar, row in zip(bars, res.itertuples(index=False), strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"F={row.pseudo_f:.2f}\np={row.p_value:.3f}, q={row.p_fdr_bh:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(x, [f"{int(scale)} s" for scale in res["scale_seconds"]])
    axes[1].set_ylabel("Residual-space pseudo-F")
    axes[1].set_title("Conditional phase signal P | L")
    axes[1].set_ylim(0, 5.8)
    axes[1].grid(axis="y", alpha=0.2)
    _save(figure, "lp_incremental_and_residual")


def plot_validation_classification(frame: pd.DataFrame) -> None:
    order = ["L", "P", "LP"]
    labels = ["Local L", "Phase P", "L + P"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharey=True)
    figure.subplots_adjust(bottom=0.13, top=0.78, wspace=0.08)
    metrics = (
        (
            "balanced_accuracy",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
            "Balanced accuracy",
            "o",
            BLUE,
        ),
        (
            "macro_auroc_ovr",
            "macro_auroc_ovr_ci_low",
            "macro_auroc_ovr_ci_high",
            "Macro AUROC",
            "s",
            ORANGE,
        ),
    )
    for ax, scale in zip(axes, (180.0, 300.0), strict=True):
        subset = frame[
            (frame["scale_seconds"] == scale) & frame["feature_set"].isin(order)
        ].set_index("feature_set")
        y = np.arange(len(order))
        for offset, (metric, low, high, label, marker, color) in zip(
            (-0.10, 0.10), metrics, strict=True
        ):
            values = np.array([subset.loc[name, metric] for name in order])
            lows = np.array([subset.loc[name, low] for name in order])
            highs = np.array([subset.loc[name, high] for name in order])
            ax.errorbar(
                values,
                y + offset,
                xerr=np.vstack([values - lows, highs - values]),
                fmt=marker,
                color=color,
                capsize=3,
                linewidth=1.2,
                markersize=6,
                label=label,
            )
            for row, value in enumerate(values):
                ax.text(value + 0.012, row + offset, f"{value:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color=GRAY, linewidth=0.8, linestyle="--")
        ax.set_xlim(0.48, 1.05)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Validation score with bootstrap 95% CI")
        ax.set_title(f"{int(scale)} s")
        ax.grid(axis="x", alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
    )
    figure.suptitle("Discovery-trained classification by two-scale stage", y=0.97)
    _save(figure, "lp_validation_classification")


def plot_phase_sensitivity(frame: pd.DataFrame) -> None:
    subset = frame[frame["fusion_stage"].isin(["P", "LP"])].copy()
    keys = [(180.0, "P"), (180.0, "LP"), (300.0, "P"), (300.0, "LP")]
    labels = ["P\n180 s", "L + P\n180 s", "P\n300 s", "L + P\n300 s"]
    figure, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    x = np.arange(len(keys))
    width = 0.34
    definitions = (
        ("primary_acoustic_chroma", "Acoustic + Chroma", BLUE, ""),
        ("all_three", "+ Rhythm sensitivity", ORANGE, "//"),
    )
    for offset, (definition, label, color, hatch) in zip(
        (-width / 2, width / 2), definitions, strict=True
    ):
        values = []
        for scale, stage in keys:
            row = subset[
                (subset["scale_seconds"] == scale)
                & (subset["fusion_stage"] == stage)
                & (subset["phase_definition"] == definition)
            ]
            if len(row) != 1:
                raise RuntimeError(f"missing phase sensitivity row: {scale}, {stage}, {definition}")
            values.append(float(row.iloc[0]["pseudo_f"]))
        values_array = np.asarray(values)
        bars = ax.bar(
            x + offset,
            values_array,
            width,
            color=color,
            hatch=hatch,
            edgecolor="white",
            label=label,
        )
        _annotate_bars(ax, bars, values_array)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Permutation pseudo-F")
    ax.set_title("Sensitivity to the composition of phase block P")
    ax.set_ylim(0, 34)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(figure, "lp_phase_sensitivity")


def plot_holdout_descriptive(frame: pd.DataFrame) -> None:
    order = ["L", "P", "LP"]
    labels = ["Local L", "Phase P", "L + P"]
    subset = frame[
        (frame["scale_seconds"] == 180.0) & frame["feature_set"].isin(order)
    ].set_index("feature_set")
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)

    x = np.arange(len(order))
    pseudo_f = np.array([subset.loc[name, "pseudo_f_descriptive"] for name in order])
    bars = axes[0].bar(x, pseudo_f, color=[BLUE, ORANGE, GREEN], width=0.6)
    _annotate_bars(axes[0], bars, pseudo_f)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Descriptive pseudo-F")
    axes[0].set_title("Opened holdout: distance geometry")
    axes[0].set_ylim(0, 18)
    axes[0].grid(axis="y", alpha=0.2)

    y = np.arange(len(order))
    balanced = np.array([subset.loc[name, "balanced_accuracy"] for name in order])
    auroc = np.array([subset.loc[name, "macro_auroc_ovr"] for name in order])
    axes[1].scatter(balanced, y - 0.10, marker="o", color=BLUE, label="Balanced accuracy")
    axes[1].scatter(auroc, y + 0.10, marker="s", color=ORANGE, label="Macro AUROC")
    for row, value in enumerate(balanced):
        axes[1].text(value + 0.012, row - 0.10, f"{value:.3f}", va="center", fontsize=8)
    for row, value in enumerate(auroc):
        axes[1].text(value + 0.012, row + 0.10, f"{value:.3f}", va="center", fontsize=8)
    axes[1].axvline(0.5, color=GRAY, linewidth=0.8, linestyle="--")
    axes[1].set_xlim(0.48, 1.05)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Descriptive score")
    axes[1].set_title("Opened holdout: classification")
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("Opened holdout / 180 s (descriptive only; no new inference)")
    _save(figure, "lp_holdout_descriptive")


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    plot_validation_permanova(_read("multiscale_hierarchical_fusion_permanova.csv"))
    plot_incremental_and_residual(
        _read("multiscale_hierarchical_fusion_incremental.csv"),
        _read("multiscale_hierarchical_fusion_residuals.csv"),
    )
    plot_validation_classification(
        _read("multiscale_hierarchical_fusion_classification.csv")
    )
    plot_phase_sensitivity(
        _read("multiscale_hierarchical_fusion_phase_sensitivity.csv")
    )
    plot_holdout_descriptive(
        _read("multiscale_hierarchical_fusion_holdout_descriptive.csv")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
