"""Reproduce a read-only audit of synchronized LTE experiments without Torch.

Writes diagnostic artifacts only. Does not fit, select, or promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rho(x: np.ndarray, y: np.ndarray) -> float:
    a, b = rankdata(x), rankdata(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(positive: np.ndarray, scores: np.ndarray) -> float:
    ranks = rankdata(scores)
    n = int(positive.sum())
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(scores) - n)))


def pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    local = frame[frame.source_kind == "local_finite_difference"]
    for direction, group in local.groupby("direction_id"):
        assert set(group.direction_sign) == {-1, 1} and len(group) == 2
        minus = group[group.direction_sign == -1].iloc[0]
        plus = group[group.direction_sign == 1].iloc[0]
        assert minus.epsilon == plus.epsilon
        exact = (plus.exact_energy - minus.exact_energy) / (2 * minus.epsilon)
        predicted = (plus.predicted_energy - minus.predicted_energy) / (2 * minus.epsilon)
        rows.append(
            {
                "direction_id": direction,
                "prompt_id": minus.prompt_id,
                "prompt_family": minus.prompt_family,
                "exact_derivative": exact,
                "predicted_derivative": predicted,
                "nontrivial": abs(exact) > 1e-6,
                "correct": math.copysign(1, exact) == math.copysign(1, predicted),
                "exact_delta": plus.exact_energy - minus.exact_energy,
                "predicted_delta": plus.predicted_energy - minus.predicted_energy,
                "global_predicted_derivative": (
                    (plus.predicted_global_energy - minus.predicted_global_energy)
                    / (2 * minus.epsilon)
                    if "predicted_global_energy" in group
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_accuracy(frame: pd.DataFrame, same_prompt: bool = True) -> tuple[int, int]:
    correct = total = 0
    groups = frame.groupby("prompt_id") if same_prompt else [("all", frame)]
    for _, group in groups:
        records = list(group.itertuples())
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                if not same_prompt and left.prompt_id == right.prompt_id:
                    continue
                delta = left.exact_band - right.exact_band
                if abs(delta) <= 1e-6:
                    continue
                estimate = left.predicted_energy - right.predicted_energy
                correct += math.copysign(1, delta) == math.copysign(1, estimate)
                total += 1
    return correct, total


def bootstrap_rho(frame: pd.DataFrame, rng: np.random.Generator) -> list[float]:
    # Resample prompt clusters, retaining all four base seeds in each draw.
    groups = [g for _, g in frame.groupby("prompt_id")]
    values = []
    for _ in range(2000):
        sampled = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        y = np.concatenate([g.exact_band.to_numpy() for g in sampled])
        p = np.concatenate([g.predicted_energy.to_numpy() for g in sampled])
        values.append(rho(y, p))
    return np.quantile(values, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir or root / "analysis_archive/pitch3_lte_v37_dte_audit"
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv"
    data = pd.read_csv(dataset_path, keep_default_na=False)
    assert data.sample_id.is_unique
    dataset_hash = sha256(dataset_path)
    fingerprint_path = root / "metadata/focus_pitch3_fingerprint_v1.json"
    config_path = root / "configs/pitch3_lte_v37_dte.toml"
    fingerprint_hash = sha256(fingerprint_path)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    inputs = {
        str(p.relative_to(root)): sha256(p) for p in [dataset_path, fingerprint_path, config_path]
    }
    lower = np.array([-1.265414901960784, 0.040998909488645215, 0.11669086558746891])
    upper = np.array([-0.04232156862745084, 1.1840803951227092, 3.1333906608457447])
    assert np.allclose(lower, fingerprint["focus_target"]["coordinate_lower"], atol=1e-12)
    assert np.allclose(upper, fingerprint["focus_target"]["coordinate_upper"], atol=1e-12)
    coordinates = np.array([json.loads(x) for x in data.coordinates_json])
    band = (
        (np.maximum(lower - coordinates, 0) ** 2 + np.maximum(coordinates - upper, 0) ** 2) / 3
    ).sum(1)
    assert np.allclose(band, data.exact_band, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.log1p(band), data.energy_target, atol=1e-10, rtol=1e-10)
    split_families = {s: sorted(g.prompt_family.unique()) for s, g in data.groupby("split")}
    assert not set(split_families["train"]) & set(split_families["development"])
    summary = {
        "diagnostic_only": True,
        "production_authorization": False,
        "guidance_promotion_eligible": False,
        "fingerprint": {
            "local_sha256": fingerprint_hash,
            "training_recorded_sha256": str(data.fingerprint_json_sha256.iloc[0]),
            "local_file_matches_training": fingerprint_hash == data.fingerprint_json_sha256.iloc[0],
            "local_band_bounds_match_training_config": True,
            "full_server_fingerprint_contents_available": (
                fingerprint_hash == data.fingerprint_json_sha256.iloc[0]
            ),
        },
        "bootstrap": {
            "unit": "prompt_id",
            "replicates": 2000,
            "seed": 20260916,
            "selection_adjusted": False,
        },
        "dataset": {
            "sha256": dataset_hash,
            "split_families": split_families,
            "split_counts": data.groupby(["split", "source_kind"]).size().to_string(),
            "local_latents_present": sum(
                (dataset_path.parent / p).exists() for p in data.latent_path
            ),
            "labels_recomputed_from_coordinates": True,
            "latent_content_audit_performed": False,
        },
    }
    history_rows, version_rows, family_rows = [], [], []
    predictions = {}
    current = None
    rng = np.random.default_rng(20260916)
    for screen_path in sorted(
        (root / "runs").glob(
            "pitch3_lte_v*/development_screen_model_only_fp32/pitch3_lte_development_screen.json"
        )
    ):
        version = screen_path.parts[-3]
        screen = json.loads(screen_path.read_text(encoding="utf-8"))
        assert screen["dataset_manifest_sha256"] == dataset_hash
        prediction_path = screen_path.with_name("pitch3_lte_development_predictions.csv")
        assert sha256(prediction_path) == screen["predictions_sha256"]
        for path in [screen_path, prediction_path]:
            inputs[str(path.relative_to(root))] = sha256(path)
        frame = pd.read_csv(prediction_path, keep_default_na=False)
        assert frame.sample_id.is_unique
        assert set(frame.sample_id) == set(data[data.split == "development"].sample_id)
        frame = frame.merge(
            data[["sample_id", "coordinates_json", "energy_target"]], on="sample_id", validate="1:1"
        )
        assert np.allclose(frame.exact_energy, frame.energy_target, atol=2e-7, rtol=2e-7)
        base = frame[frame.source_kind == "base_step4_seed"].copy()
        predictions[version] = base.set_index("sample_id").predicted_energy
        local = pairs(frame)
        nontrivial = local[local.nontrivial]
        m = screen["metrics"]
        correct, total = rank_accuracy(base)
        assert np.isclose(
            rho(base.exact_band, base.predicted_energy), m["direct_energy_spearman"], atol=1e-12
        )
        assert np.isclose(correct / total, m["same_prompt_ranking_accuracy"], atol=1e-12)
        assert np.isclose(nontrivial.correct.mean(), m["local_direction_sign_accuracy"], atol=1e-12)
        version_rows.append(
            {
                "version": version,
                **{
                    k: m.get(k)
                    for k in [
                        "direct_energy_spearman",
                        "minimum_prompt_family_spearman",
                        "same_prompt_ranking_accuracy",
                        "local_direction_sign_accuracy",
                        "local_derivative_spearman",
                        "collapsed_prompt_count",
                        "total_gate_deficit",
                    ]
                },
                **m["prompt_family_spearman"],
            }
        )
        if "predicted_global_energy" in frame:
            global_correct = np.sign(nontrivial.exact_derivative) == np.sign(
                nontrivial.global_predicted_derivative
            )
            version_rows[-1].update(
                {
                    "global_only_direction_accuracy": float(global_correct.mean()),
                    "global_only_derivative_rho": rho(
                        nontrivial.exact_derivative, nontrivial.global_predicted_derivative
                    ),
                    "local_residual_fixed_pairs": int((nontrivial.correct & ~global_correct).sum()),
                    "local_residual_broke_pairs": int((~nontrivial.correct & global_correct).sum()),
                }
            )
        for family, group in base.groupby("prompt_family"):
            y, p = group.exact_energy.to_numpy(), group.predicted_energy.to_numpy()
            positive = y > 0
            means = group.groupby("prompt_id")[["exact_energy", "predicted_energy"]].mean()
            centered = group[["exact_energy", "predicted_energy"]] - group.groupby("prompt_id")[
                ["exact_energy", "predicted_energy"]
            ].transform("mean")
            q = np.array([json.loads(x) for x in group.coordinates_json])
            components = (np.maximum(lower - q, 0) ** 2 + np.maximum(q - upper, 0) ** 2) / 3
            within_c, within_n = rank_accuracy(group)
            cross_c, cross_n = rank_accuracy(group, False)
            near = y <= 0.1  # Descriptive diagnostic only; not a changed gate or label.
            item = {
                "version": version,
                "family": family,
                "n": len(group),
                "zero_count": int((~positive).sum()),
                "rho": rho(y, p),
                "positive_only_rho": rho(y[positive], p[positive]),
                "positive_auc": auc(positive, p),
                "within_prompt_centered_rho": rho(centered.exact_energy, centered.predicted_energy),
                "prompt_mean_rho": rho(means.exact_energy, means.predicted_energy),
                "within_prompt_rank": within_c / within_n,
                "cross_prompt_rank": cross_c / cross_n,
                "mae": float(np.mean(abs(y - p))),
                "near_band_count": int(near.sum()),
                "near_band_rho": rho(y[near], p[near]),
                "q3_band_mass_share": float(components[:, 2].sum() / components.sum()),
                **{
                    f"near_band_q{k + 1}_mass_share": float(
                        components[near, k].sum() / components[near].sum()
                    )
                    for k in range(3)
                },
                "exact_energy_median": float(np.median(y)),
                "predicted_energy_median": float(np.median(p)),
            }
            if version == "pitch3_lte_v37_dte":
                item["rho_cluster_ci_low"], item["rho_cluster_ci_high"] = bootstrap_rho(group, rng)
            family_rows.append(item)
        if version == "pitch3_lte_v37_dte":
            current = frame
            local.to_csv(out / "local_pairs.csv", index=False)
            base["absolute_error"] = abs(base.exact_energy - base.predicted_energy)
            base["rank_error"] = (
                base.groupby("prompt_family").predicted_energy.rank()
                - base.groupby("prompt_family").exact_energy.rank()
            )
            base.sort_values("absolute_error", ascending=False).to_csv(
                out / "base_errors.csv", index=False
            )
    assert current is not None
    seeds = []
    seed_predictions = []
    run_root = root / "runs/pitch3_lte_v37_dte"
    ensemble_path = run_root / "pitch3_lte_ensemble.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    inputs[str(ensemble_path.relative_to(root))] = sha256(ensemble_path)
    ensemble_members = {m["seed"]: m for m in ensemble["members"]}
    for manifest_path in sorted(run_root.glob("seed_*/models/pitch3_lte_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = manifest["seed"]
        inputs[str(manifest_path.relative_to(root))] = sha256(manifest_path)
        assert manifest["fingerprint_json_sha256"] == data.fingerprint_json_sha256.iloc[0]
        assert manifest["training_manifest_sha256"] == dataset_hash
        assert manifest["source_training_config_sha256"] == sha256(config_path)
        assert sha256(manifest_path) == ensemble_members[seed]["manifest_sha256"]
        effective = manifest_path.with_name("pitch3_lte_effective_config.json")
        assert sha256(effective) == manifest["training_config_sha256"]
        inputs[str(effective.relative_to(root))] = sha256(effective)
        for candidate in manifest["checkpoint_candidates"].values():
            checkpoint = manifest_path.parent / Path(candidate["checkpoint"]).name
            assert sha256(checkpoint) == candidate["checkpoint_sha256"]
            inputs[str(checkpoint.relative_to(root))] = sha256(checkpoint)
        assert manifest["checkpoint_sha256"] == ensemble_members[seed]["checkpoint_sha256"]
        prediction_path = (
            manifest_path.parent.parent
            / "development_screen_model_only_fp32/pitch3_lte_development_predictions.csv"
        )
        seed_frame = pd.read_csv(prediction_path, keep_default_na=False).set_index("sample_id")
        seed_predictions.append(seed_frame.predicted_energy.reindex(current.sample_id).to_numpy())
        inputs[str(prediction_path.relative_to(root))] = sha256(prediction_path)
        for row in manifest["training_history"]:
            for source, variant in row["development_variants"].items():
                m = variant["metrics"]
                history_rows.append(
                    {
                        "seed": seed,
                        "epoch": row["epoch"],
                        "stage": row["training_stage"],
                        "source": source,
                        "direct_rho": m["direct_energy_spearman"],
                        "min_family_rho": m["minimum_prompt_family_spearman"],
                        "p04_rho": m["prompt_family_spearman"]["p04_piano_strings"],
                        "p28_rho": m["prompt_family_spearman"]["p28_flowing_texture"],
                        "direction": m["local_direction_sign_accuracy"],
                        "all_passed": m["all_gates_passed"],
                    }
                )
        history = pd.DataFrame([r for r in history_rows if r["seed"] == seed])
        seeds.append(
            {
                "seed": seed,
                "best_epoch": manifest["best_epoch"],
                "global_epochs": int(history[history.stage == "global"].epoch.nunique()),
                "local_epochs": int(history[history.stage == "local"].epoch.nunique()),
                "selected_global_epoch": manifest["checkpoint_candidates"]["global_stage"]["epoch"],
                "max_p04_rho_any_epoch_or_ema": float(history.p04_rho.max()),
                "all_gates_passed_any_epoch": bool(history.all_passed.any()),
                "selected_metrics": manifest["best_development_metrics"],
            }
        )
    averaged = np.mean(seed_predictions, axis=0)
    assert np.allclose(averaged, current.predicted_energy, atol=3e-7, rtol=3e-7)
    summary["seeds"] = seeds
    summary["ensemble_recomputed_max_absolute_error"] = float(
        np.max(abs(averaged - current.predicted_energy))
    )
    base = current[current.source_kind == "base_step4_seed"]
    summary["base_local_residual_max_absolute"] = float(base.predicted_local_energy.abs().max())
    # Quantify the smoothing in the currently implemented family target ranks.
    smoothing = []
    for (split, family), group in data[data.source_kind == "base_step4_seed"].groupby(
        ["split", "prompt_family"]
    ):
        y = group.energy_target.to_numpy()
        normalized = (y - y.mean()) / max(y.std(), 1e-6)
        soft_rank = expit((normalized[:, None] - normalized[None, :]) / 0.1).sum(1)
        hard_rank = rankdata(y)
        near = y <= 0.1
        smoothing.append(
            {
                "split": split,
                "family": family,
                "energy_std": float(y.std()),
                "effective_target_temperature_energy_units": float(0.1 * y.std()),
                "near_band_n": int(near.sum()),
                "near_soft_rank_span": float(np.ptp(soft_rank[near])),
                "near_hard_rank_span": float(np.ptp(hard_rank[near])),
            }
        )
    pd.DataFrame(smoothing).to_csv(out / "target_rank_smoothing.csv", index=False)
    pd.DataFrame(version_rows).to_csv(out / "version_metrics.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out / "family_diagnostics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out / "v37_training_history.csv", index=False)
    prediction_matrix = pd.DataFrame(predictions)
    prediction_matrix.corr(method="spearman").to_csv(out / "version_prediction_correlations.csv")
    p04_ids = base[base.prompt_family == "p04_piano_strings"].sample_id
    prediction_matrix.loc[p04_ids].corr(method="spearman").to_csv(
        out / "p04_prediction_correlations.csv"
    )
    summary["inputs_sha256"] = inputs
    (out / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    versions = pd.DataFrame(version_rows)
    diagnostic = pd.DataFrame(family_rows)
    selected = diagnostic[diagnostic.version == "pitch3_lte_v37_dte"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    colors = ["#b74132", "#29826b", "#4169a1", "#cb8929"]
    labels = [
        v.replace("pitch3_lte_", "").replace("_dte", "").replace("_tcr", "")
        for v in versions.version
    ]
    for family, color in zip(sorted(base.prompt_family.unique()), colors, strict=True):
        axes[0].plot(labels, versions[family], "o-", label=family[:3], color=color, linewidth=1.7)
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0].set(
        title="Family ranking across experiments", ylabel="Spearman correlation", ylim=(0.15, 0.8)
    )
    axes[0].tick_params(axis="x", rotation=50)
    axes[0].legend(ncol=2, frameon=False)
    indices = np.arange(4)
    axes[1].bar(indices - 0.19, selected.rho, 0.38, label="All 64 base samples", color="#4169a1")
    axes[1].bar(
        indices + 0.19, selected.near_band_rho, 0.38, label="Exact energy <= 0.1", color="#b74132"
    )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(indices, [f[:3] for f in selected.family])
    axes[1].set(
        title="V3.7 loses low-energy ordering", ylabel="Spearman correlation", ylim=(-0.16, 0.83)
    )
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    p04 = selected[selected.family == "p04_piano_strings"].iloc[0]
    components = np.array([p04[f"near_band_q{k}_mass_share"] for k in (1, 2, 3)])
    bars = axes[2].bar(
        ["q1: persistence", "q2: self-transition", "q3: recurrence"],
        components,
        color=["#4169a1", "#29826b", "#cb8929"],
    )
    axes[2].bar_label(bars, labels=[f"{100 * v:.1f}%" for v in components], padding=4)
    axes[2].set(
        title="p04: low-energy Band contributions",
        ylabel="Share of exact Band mass",
        ylim=(0, 0.62),
    )
    axes[2].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Diagnostic audit | frozen gates unchanged | no new model inference", fontsize=12)
    fig.savefig(out / "diagnostic_summary.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "synced_model_config_dataset_hash_checks": "passed",
                "local_fingerprint_matches_training": summary["fingerprint"][
                    "local_file_matches_training"
                ],
                "reported_metrics_reproduced": True,
                "model_inference_performed": False,
                "production_authorization": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
