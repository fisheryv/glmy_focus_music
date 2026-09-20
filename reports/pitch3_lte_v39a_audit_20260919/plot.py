"""Plot verified archived metrics; this does not run or select models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
a = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
colors = {"baseline": "#34699a", "transition": "#c6572c"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(14, 9), layout="constrained")
grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
family_ax = fig.add_subplot(grid[:, 1])
top = fig.add_subplot(grid[0, 0])
bottom = fig.add_subplot(grid[1, 0])
for v in colors:
    curve = a["training_aggregates"]["cv"][v]["curves"]
    top.plot(
        [x["epoch"] for x in curve],
        [x["family_mean"] for x in curve],
        "o-",
        ms=4,
        color=colors[v],
        label=v,
    )
    bottom.plot(
        [x["epoch"] for x in curve],
        [x["low_rho"] for x in curve],
        "o-",
        ms=4,
        color=colors[v],
        label=v,
    )
    outer = a["cv"][v]["ensemble_aggregates"]["selection"]["zero_and_low_positive_rho"]["mean"]
    bottom.scatter([14], [outer], marker="D", s=65, color=colors[v])
    bottom.annotate(
        f"{v}: {outer:.3f}",
        (14, outer),
        xytext=(-7, 10 if v == "baseline" else -18),
        textcoords="offset points",
        ha="right",
        color=colors[v],
        fontsize=9,
    )
top.axhline(0.5, color="#777777", ls="--", lw=1)
top.set(
    xlabel="Epoch",
    ylabel="Mean within-family Spearman",
    title="Fit curve: average of 15 runs per variant",
    ylim=(-0.02, 0.62),
)
top.text(
    0.02,
    0.95,
    "Dashed: 0.50 reference; the gate requires EVERY family",
    transform=top.transAxes,
    fontsize=9,
    va="top",
    color="#555555",
)
top.legend(loc="lower right")
bottom.axhline(0, color="#888888", lw=1)
bottom.set(
    xlabel="Epoch (0-12); diamond = held-out ensemble",
    ylabel="Mean zero + low-positive Spearman",
    title="Low-value ordering remains near zero",
    ylim=(-0.17, 0.10),
    xlim=(-0.3, 14.8),
)
bottom.set_xticks([0, 2, 4, 6, 8, 10, 12, 14], ["0", "2", "4", "6", "8", "10", "12", "Outer"])
b = a["summaries"]["20260941_20260942_20260943"]["report"]["variants"]["baseline"][
    "family_spearman"
]
t = a["summaries"]["20260941_20260942_20260943"]["report"]["variants"]["transition"][
    "family_spearman"
]
names = sorted(b)
y = np.arange(len(names))
for i, f in enumerate(names):
    family_ax.plot([b[f], t[f]], [i, i], color="#b7bcc2", lw=2, zorder=1)
family_ax.scatter(
    [b[f] for f in names], y, s=45, color=colors["baseline"], label="baseline: 7/20 pass", zorder=2
)
family_ax.scatter(
    [t[f] for f in names],
    y,
    s=45,
    marker="D",
    color=colors["transition"],
    label="transition: 8/20 pass",
    zorder=3,
)
family_ax.axvline(0.5, color="#777777", ls="--", lw=1)
family_ax.axvline(0, color="#cccccc", lw=1)
family_ax.set_yticks(y, [f.replace("_", " ") for f in names])
family_ax.invert_yaxis()
family_ax.set(
    xlabel="Held-out family Spearman",
    title="Three-seed energy ensemble: 20 train families",
    xlim=(-0.38, 0.82),
)
family_ax.legend(loc="lower left", fontsize=9)
fig.suptitle("Pitch3-LTE V3.9-A: small average gain; boundary ranking unresolved", fontsize=16)
fig.savefig(OUT / "v39a_evidence.png", dpi=160)
fig.savefig(OUT / "v39a_evidence.pdf")
plt.close(fig)
print(OUT / "v39a_evidence.png")
