"""Generate deterministic, explicitly synthetic LTCH-P3 placeholder results.

These artifacts validate the paper's analysis and layout only.  They are not
model outputs, qualification evidence, or production authorization.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

_ROOT_FOR_MPL = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT_FOR_MPL / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
DATA_DIR = PAPERS / "mock_data"
FIG_DIR = PAPERS / "fig"
GENERATED_DIR = PAPERS / "generated"
MOCK_SEED = 20260915
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_REPLICATES = 2000

FAMILY_PLANS = {
    "p04_piano_strings": [4, 5, 5, 5, 5, 6, 6, 6],
    "p16_bell_texture": [4, 5, 5, 6, 6, 6, 6, 6],
    "p23_rhodes_bass": [5, 5, 6, 6, 6, 6, 6, 6],
    "p28_flowing_texture": [5, 5, 6, 6, 6, 6, 7, 7],
}


def _paired_hodges_lehmann(values: np.ndarray) -> float:
    walsh = (values[:, None] + values[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(values.size)]))


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(low), float(high)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    radius /= denominator
    return center - radius, center + radius


def _build_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(MOCK_SEED)
    rows: list[dict[str, object]] = []
    prompt_index = 0

    for family, improvement_counts in FAMILY_PLANS.items():
        for variant, improved_count in enumerate(improvement_counts, start=1):
            prompt_id = f"{family}__v{variant:02d}"
            improved_seeds = set(
                int(value) for value in rng.choice(8, size=improved_count, replace=False)
            )
            non_improved = [seed for seed in range(8) if seed not in improved_seeds]
            baseline_hit_seed = non_improved[0] if prompt_index < 28 else None

            for seed in range(8):
                improved = seed in improved_seeds
                baseline_hit = seed == baseline_hit_seed
                baseline = 0.0 if baseline_hit else float(
                    np.clip(rng.lognormal(mean=-0.65, sigma=0.45), 0.16, 1.35)
                )
                rows.append(
                    {
                        "pair_id": f"{prompt_id}__s{seed:02d}",
                        "prompt_id": prompt_id,
                        "prompt_family": family,
                        "seed_index": seed,
                        "baseline_band_distance": baseline,
                        "planned_improvement": improved,
                        "baseline_target_band_hit": baseline_hit,
                    }
                )
            prompt_index += 1

    improved_indices = [
        index for index, row in enumerate(rows) if bool(row["planned_improvement"])
    ]
    guided_hit_indices = set(
        int(value) for value in rng.choice(improved_indices, size=30, replace=False)
    )

    for index, row in enumerate(rows):
        baseline = float(row["baseline_band_distance"])
        improved = bool(row.pop("planned_improvement"))
        baseline_hit = bool(row["baseline_target_band_hit"])
        if improved:
            if index in guided_hit_indices:
                guided = 0.0
            else:
                reduction = min(
                    baseline * 0.65,
                    float(rng.lognormal(mean=-2.15, sigma=0.42)),
                )
                guided = max(0.0, baseline - reduction)
        elif baseline_hit:
            guided = 0.0
        else:
            guided = baseline + float(rng.lognormal(mean=-2.50, sigma=0.50))

        delta = guided - baseline
        row.update(
            {
                "guided_band_distance": guided,
                "delta_band_distance": delta,
                "positive_guidance": delta < -1e-12,
                "neutral_guidance": abs(delta) <= 1e-12,
                "guided_target_band_hit": guided <= 1e-12,
                "synthetic": True,
                "scientific_evidence": False,
                "qualification_result": False,
                "production_authorization": False,
                "mock_seed": MOCK_SEED,
            }
        )
    return rows


def _analyse(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = np.asarray([float(row["delta_band_distance"]) for row in rows])
    baseline_hits = np.asarray([bool(row["baseline_target_band_hit"]) for row in rows])
    guided_hits = np.asarray([bool(row["guided_target_band_hit"]) for row in rows])
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    by_prompt = {
        prompt_id: [index for index, row in enumerate(rows) if row["prompt_id"] == prompt_id]
        for prompt_id in prompt_ids
    }
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rate: list[float] = []
    boot_median: list[float] = []
    boot_hl: list[float] = []
    boot_hit_gain: list[float] = []

    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_prompts = bootstrap_rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        indices = np.asarray(
            [index for prompt_id in sampled_prompts for index in by_prompt[str(prompt_id)]],
            dtype=int,
        )
        sampled_delta = deltas[indices]
        boot_rate.append(float(np.mean(sampled_delta < -1e-12)))
        boot_median.append(float(np.median(sampled_delta)))
        boot_hl.append(_paired_hodges_lehmann(sampled_delta))
        boot_hit_gain.append(
            float(np.mean(guided_hits[indices]) - np.mean(baseline_hits[indices]))
        )

    family_metrics: dict[str, dict[str, float | int | list[float]]] = {}
    for family in FAMILY_PLANS:
        family_rows = [row for row in rows if row["prompt_family"] == family]
        successes = sum(bool(row["positive_guidance"]) for row in family_rows)
        low, high = _wilson_interval(successes, len(family_rows))
        family_metrics[family] = {
            "pairs": len(family_rows),
            "positive_pairs": successes,
            "positive_rate": successes / len(family_rows),
            "wilson_95_ci": [low, high],
        }

    positive_count = int(np.sum(deltas < -1e-12))
    neutral_count = int(np.sum(np.abs(deltas) <= 1e-12))
    negative_count = len(rows) - positive_count - neutral_count
    return {
        "artifact_role": "synthetic_layout_and_analysis_placeholder_only",
        "synthetic": True,
        "scientific_evidence": False,
        "qualification_result": False,
        "production_authorization": False,
        "mock_seed": MOCK_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "pairs": len(rows),
        "prompts": len(prompt_ids),
        "seeds_per_prompt": 8,
        "positive_pairs": positive_count,
        "positive_rate": positive_count / len(rows),
        "positive_rate_cluster_bootstrap_95_ci": list(_percentile_interval(boot_rate)),
        "neutral_pairs": neutral_count,
        "neutral_rate": neutral_count / len(rows),
        "worsened_pairs": negative_count,
        "worsened_rate": negative_count / len(rows),
        "median_delta": float(np.median(deltas)),
        "median_delta_cluster_bootstrap_95_ci": list(_percentile_interval(boot_median)),
        "hodges_lehmann_delta": _paired_hodges_lehmann(deltas),
        "hodges_lehmann_cluster_bootstrap_95_ci": list(_percentile_interval(boot_hl)),
        "baseline_hit_rate": float(np.mean(baseline_hits)),
        "guided_hit_rate": float(np.mean(guided_hits)),
        "hit_rate_gain": float(np.mean(guided_hits) - np.mean(baseline_hits)),
        "hit_rate_gain_cluster_bootstrap_95_ci": list(_percentile_interval(boot_hit_gain)),
        "family_metrics": family_metrics,
    }


def _write_csv(rows: list[dict[str, object]]) -> None:
    path = DATA_DIR / "ltch_p3_mock_final_evaluation.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(stats: dict[str, object]) -> None:
    path = DATA_DIR / "ltch_p3_mock_final_evaluation_summary.json"
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_ci(values: list[float], scale: float = 1.0, digits: int = 3) -> str:
    return f"{values[0] * scale:.{digits}f}--{values[1] * scale:.{digits}f}"


def _write_tex(stats: dict[str, object]) -> None:
    family_rates = [
        float(item["positive_rate"])
        for item in dict(stats["family_metrics"]).values()
    ]
    lines = [
        "% Generated synthetic placeholder values. Do not edit by hand.",
        f"\\newcommand{{\\MockPairCount}}{{{stats['pairs']}}}",
        f"\\newcommand{{\\MockPositiveCount}}{{{stats['positive_pairs']}}}",
        f"\\newcommand{{\\MockPositiveRate}}{{{float(stats['positive_rate']) * 100:.1f}\\%}}",
        "\\newcommand{\\MockPositiveRateCI}{"
        + _format_ci(list(stats["positive_rate_cluster_bootstrap_95_ci"]), 100.0, 1)
        + "\\%}",
        f"\\newcommand{{\\MockNeutralCount}}{{{stats['neutral_pairs']}}}",
        f"\\newcommand{{\\MockNeutralRate}}{{{float(stats['neutral_rate']) * 100:.1f}\\%}}",
        f"\\newcommand{{\\MockWorsenedCount}}{{{stats['worsened_pairs']}}}",
        f"\\newcommand{{\\MockWorsenedRate}}{{{float(stats['worsened_rate']) * 100:.1f}\\%}}",
        f"\\newcommand{{\\MockMedianDelta}}{{{float(stats['median_delta']):.3f}}}",
        "\\newcommand{\\MockMedianDeltaCI}{"
        + _format_ci(list(stats["median_delta_cluster_bootstrap_95_ci"]), 1.0, 3)
        + "}",
        f"\\newcommand{{\\MockHLDelta}}{{{float(stats['hodges_lehmann_delta']):.3f}}}",
        "\\newcommand{\\MockHLDeltaCI}{"
        + _format_ci(list(stats["hodges_lehmann_cluster_bootstrap_95_ci"]), 1.0, 3)
        + "}",
        f"\\newcommand{{\\MockBaselineHitRate}}{{{float(stats['baseline_hit_rate']) * 100:.1f}\\%}}",
        f"\\newcommand{{\\MockGuidedHitRate}}{{{float(stats['guided_hit_rate']) * 100:.1f}\\%}}",
        f"\\newcommand{{\\MockHitGain}}{{{float(stats['hit_rate_gain']) * 100:.1f}}}",
        "\\newcommand{\\MockHitGainCI}{"
        + _format_ci(list(stats["hit_rate_gain_cluster_bootstrap_95_ci"]), 100.0, 1)
        + "}",
        f"\\newcommand{{\\MockFamilyRateRange}}{{{min(family_rates) * 100:.1f}\\%--{max(family_rates) * 100:.1f}\\%}}",
    ]
    (GENERATED_DIR / "ltch_p3_mock_stats.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot(rows: list[dict[str, object]], stats: dict[str, object]) -> None:
    baseline = np.asarray([float(row["baseline_band_distance"]) for row in rows])
    guided = np.asarray([float(row["guided_band_distance"]) for row in rows])
    delta = guided - baseline
    positive = delta < -1e-12
    neutral = np.abs(delta) <= 1e-12
    worsened = delta > 1e-12

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.55), constrained_layout=True)

    ax = axes[0]
    limit = max(float(np.max(baseline)), float(np.max(guided))) * 1.04
    ax.plot([0, limit], [0, limit], color="#666666", lw=1.0, ls="--")
    ax.scatter(baseline[positive], guided[positive], s=15, alpha=0.75, color="#18864b", label="Improved")
    ax.scatter(baseline[neutral], guided[neutral], s=16, alpha=0.8, color="#777777", label="Neutral")
    ax.scatter(baseline[worsened], guided[worsened], s=15, alpha=0.75, color="#d97818", label="Worsened")
    ax.set(xlabel="Baseline exact band distance", ylabel="Guided exact band distance", title="A  Paired exact audit")
    ax.set_xlim(-0.02, limit)
    ax.set_ylim(-0.02, limit)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    ax.hist(delta, bins=np.linspace(float(np.min(delta)), float(np.max(delta)), 24), color="#4c78a8", alpha=0.88)
    ax.axvline(0.0, color="#b22222", lw=1.2, ls="--")
    ax.axvline(float(stats["median_delta"]), color="#111111", lw=1.2)
    ax.set(xlabel=r"Paired $\Delta=L_{guided}-L_{baseline}$", ylabel="Pairs", title="B  Paired shift distribution")
    ax.text(0.03, 0.95, f"positive: {float(stats['positive_rate']) * 100:.1f}%", transform=ax.transAxes, va="top")

    ax = axes[2]
    family_metrics = dict(stats["family_metrics"])
    labels = ["Piano/strings", "Bell texture", "Rhodes/bass", "Flowing texture"]
    rates = np.asarray([float(item["positive_rate"]) for item in family_metrics.values()])
    intervals = np.asarray([item["wilson_95_ci"] for item in family_metrics.values()], dtype=float)
    errors = np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates))
    positions = np.arange(len(labels))
    ax.errorbar(positions, rates * 100.0, yerr=errors * 100.0, fmt="o", color="#6f4c9b", capsize=3, lw=1.2)
    ax.axhspan(65.0, 75.0, color="#18864b", alpha=0.08)
    ax.axhline(float(stats["positive_rate"]) * 100.0, color="#18864b", ls="--", lw=1.1, label="Overall")
    ax.set_xticks(positions, labels, rotation=22, ha="right")
    ax.set_ylim(45.0, 90.0)
    ax.set(ylabel="Positive-guidance rate (%)", title="C  Prompt-family consistency")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle("LTCH-P3 mock final evaluation (synthetic placeholder)", fontsize=12, weight="bold")
    fig.text(
        0.5,
        0.50,
        "MOCK / SYNTHETIC DATA - NOT EVIDENCE",
        ha="center",
        va="center",
        color="#b22222",
        alpha=0.12,
        fontsize=23,
        weight="bold",
        rotation=12,
    )
    fig.text(0.995, 0.005, f"fixed mock seed: {MOCK_SEED}", ha="right", va="bottom", fontsize=6, color="#555555")
    fig.savefig(FIG_DIR / "ltch_p3_mock_final_evaluation.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ltch_p3_mock_final_evaluation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    rows = _build_rows()
    stats = _analyse(rows)
    if stats["pairs"] != 256 or stats["positive_pairs"] != 180:
        raise RuntimeError("mock contract changed unexpectedly")
    _write_csv(rows)
    _write_json(stats)
    _write_tex(stats)
    _plot(rows, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
