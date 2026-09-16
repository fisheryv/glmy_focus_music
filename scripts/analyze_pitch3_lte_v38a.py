"""Audit archived V3.8-A predictions, provenance and CV without Torch or inference.

Only diagnostic outputs are written. No fitting, checkpoint selection or promotion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_pitch3_lte_v37_dte import auc, pairs, rank_accuracy, rho, sha256
from summarize_pitch3_lte_cv import summarize


def family_diagnostics(frame: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> list[dict]:
    rows = []
    for family, group in frame[frame.source_kind == "base_step4_seed"].groupby("prompt_family"):
        y, p = group.exact_energy.to_numpy(), group.predicted_energy.to_numpy()
        positive, near = y > 0, y <= 0.1  # Descriptive slice, not a new gate.
        q = np.array([json.loads(x) for x in group.coordinates_json])
        qp = np.array([json.loads(x) for x in group.predicted_coordinates_json])
        components = (np.maximum(lower - q, 0) ** 2 + np.maximum(q - upper, 0) ** 2) / 3
        correct, total = rank_accuracy(group)
        row = {
            "family": family,
            "n": len(group),
            "rho": rho(y, p),
            "zero_n": int((~positive).sum()),
            "positive_only_rho": rho(y[positive], p[positive]),
            "zero_positive_auc": auc(positive, p) if positive.any() and (~positive).any() else None,
            "near_n": int(near.sum()),
            "near_rho": rho(y[near], p[near]),
            "exact_energy_median": float(np.median(y)),
            "predicted_energy_q25": float(np.quantile(p, 0.25)),
            "predicted_energy_median": float(np.median(p)),
            "predicted_energy_q75": float(np.quantile(p, 0.75)),
            "energy_mae": float(np.mean(abs(y - p))),
            "same_prompt_correct": correct,
            "same_prompt_total": total,
        }
        for i in range(3):
            row.update(
                {
                    f"q{i + 1}_rho": rho(q[:, i], qp[:, i]),
                    f"near_q{i + 1}_rho": rho(q[near, i], qp[near, i]),
                    f"q{i + 1}_mae_width": float(
                        np.mean(abs(q[:, i] - qp[:, i])) / (upper[i] - lower[i])
                    ),
                    f"near_q{i + 1}_band_mass_share": float(
                        components[near, i].sum() / components[near].sum()
                    ),
                }
            )
        rows.append(row)
    return rows


def verify_metrics(frame: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    base = frame[frame.source_kind == "base_step4_seed"]
    pair_frame = pairs(frame)
    nontrivial = pair_frame[pair_frame.nontrivial]
    correct, total = rank_accuracy(base)
    assert total == metrics["same_prompt_rank_pairs"]
    assert len(nontrivial) == metrics["local_direction_pairs"]
    checks = {
        "direct_energy_spearman": rho(base.exact_band, base.predicted_energy),
        "same_prompt_ranking_accuracy": correct / total,
        "local_direction_sign_accuracy": float(nontrivial.correct.mean()),
        "local_derivative_spearman": rho(
            nontrivial.exact_derivative, nontrivial.predicted_derivative
        ),
    }
    for key, value in checks.items():
        assert np.isclose(value, metrics[key], atol=1e-12, rtol=0), (key, value, metrics[key])
    for family, group in base.groupby("prompt_family"):
        assert np.isclose(
            rho(group.exact_band, group.predicted_energy),
            metrics["prompt_family_spearman"][family],
            atol=1e-12,
            rtol=0,
        )
    if "collapsed_prompt_count" in metrics:
        collapsed = sum(
            np.ptp(g.predicted_energy) <= 1e-3 and np.ptp(g.exact_energy) >= 0.1
            for _, g in base.groupby("prompt_id")
        )
        assert collapsed == metrics["collapsed_prompt_count"]
    return pair_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output_dir or root / "analysis_archive/pitch3_lte_v38a_audit").resolve()
    # Audit output must not overwrite synchronized runs or source/configuration files.
    if not out.is_relative_to(root / "analysis_archive"):
        raise ValueError("Diagnostic output must be under the project's analysis_archive")
    out.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, str] = {}

    def digest(path: Path, expected: str | None = None) -> str:
        key = path.relative_to(root).as_posix()
        if key not in inputs:
            inputs[key] = sha256(path)
        if expected is not None:
            assert inputs[key] == expected, f"SHA-256 mismatch: {key}"
        return inputs[key]

    def read_json(path: Path) -> dict:
        digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    dataset_path = root / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv"
    dataset_hash = digest(dataset_path)
    data = pd.read_csv(dataset_path, keep_default_na=False)
    assert data.sample_id.is_unique
    assert set(data.split) == {"train", "development"}
    fp_path = root / "metadata/focus_pitch3_fingerprint_v1.json"
    fingerprint = read_json(fp_path)
    fp_hash = digest(fp_path)
    assert set(data.fingerprint_json_sha256) == {fp_hash}
    lower = np.array(fingerprint["focus_target"]["coordinate_lower"])
    upper = np.array(fingerprint["focus_target"]["coordinate_upper"])
    q = np.array([json.loads(x) for x in data.coordinates_json])
    band = ((np.maximum(lower - q, 0) ** 2 + np.maximum(q - upper, 0) ** 2) / 3).sum(1)
    assert np.allclose(band, data.exact_band, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.log1p(band), data.energy_target, atol=1e-10, rtol=1e-10)
    plan_hash = digest(dataset_path.with_name("pitch3_lte_dataset_plan.json"))
    train = data[data.split == "train"]
    dev = data[data.split == "development"]
    assert not set(train.prompt_family) & set(dev.prompt_family)

    def read_predictions(path: Path, expected_ids: set[str]) -> pd.DataFrame:
        digest(path)
        frame = pd.read_csv(path, keep_default_na=False)
        assert frame.sample_id.is_unique and set(frame.sample_id) == expected_ids
        truth = data.set_index("sample_id").loc[frame.sample_id]
        for col in ["prompt_id", "prompt_family", "source_kind", "direction_id", "direction_sign"]:
            assert np.array_equal(frame[col], truth[col]), (path, col)
        assert np.allclose(frame.exact_band, truth.exact_band, atol=2e-6, rtol=2e-7)
        assert np.allclose(frame.exact_energy, truth.energy_target, atol=2e-7, rtol=2e-7)
        if "predicted_global_energy" in frame:
            assert np.allclose(
                frame.predicted_energy,
                frame.predicted_global_energy + frame.predicted_local_energy,
                atol=4e-7,
                rtol=4e-7,
            )
            assert (
                frame.loc[frame.source_kind == "base_step4_seed", "predicted_local_energy"] == 0
            ).all()
        return frame.merge(data[["sample_id", "coordinates_json"]], on="sample_id", validate="1:1")

    versions, families, pair_rows = [], [], []
    current = None
    for screen_path in sorted(
        (root / "runs").glob(
            "pitch3_lte_v*/development_screen_model_only_fp32/pitch3_lte_development_screen.json"
        )
    ):
        version = screen_path.parts[-3]
        screen = read_json(screen_path)
        assert screen["dataset_manifest_sha256"] == dataset_hash
        path = screen_path.with_name("pitch3_lte_development_predictions.csv")
        digest(path, screen["predictions_sha256"])
        frame = read_predictions(path, set(dev.sample_id))
        m = screen["metrics"]
        local = verify_metrics(frame, m)
        row = {
            "version": version,
            **{k: v for k, v in m.items() if isinstance(v, (int, float))},
            **m["prompt_family_spearman"],
        }
        if "predicted_global_energy" in frame:
            nt = local[local.nontrivial]
            gc = np.sign(nt.exact_derivative) == np.sign(nt.global_predicted_derivative)
            row.update(
                {
                    "global_direction_correct": int(gc.sum()),
                    "total_direction_correct": int(nt.correct.sum()),
                    "local_fixed": int((nt.correct & ~gc).sum()),
                    "local_broke": int((~nt.correct & gc).sum()),
                    "global_derivative_rho": rho(
                        nt.exact_derivative, nt.global_predicted_derivative
                    ),
                }
            )
        versions.append(row)
        if version in {"pitch3_lte_v37_dte", "pitch3_lte_v38a"}:
            families.extend(
                {"version": version, **r} for r in family_diagnostics(frame, lower, upper)
            )
            pair_rows.append(local.assign(version=version))
        if version == "pitch3_lte_v38a":
            current = frame
    assert current is not None

    run_root = root / "runs/pitch3_lte_v38a"
    ensemble_path = run_root / "pitch3_lte_ensemble.json"
    ensemble = read_json(ensemble_path)
    screen = read_json(
        run_root / "development_screen_model_only_fp32/pitch3_lte_development_screen.json"
    )
    digest(ensemble_path, screen["ensemble_manifest_sha256"])
    config_hash = digest(root / "configs/pitch3_lte_v38a.toml")
    assert ensemble["training_manifest_sha256"] == dataset_hash
    assert ensemble["fingerprint_json_sha256"] == fp_hash
    assert ensemble["dataset_plan_sha256"] == plan_hash
    assert ensemble["source_training_config_sha256"] == config_hash
    assert ensemble["member_count"] == 3 and len(ensemble["members"]) == 3
    members = {m["seed"]: m for m in ensemble["members"]}
    assert len(members) == 3
    assert all(np.isclose(m["weight"], 1 / 3) for m in members.values())
    histories, seeds, cv_rows, cv_families, seed_frames = [], [], [], [], []
    manifest_paths = sorted(run_root.glob("seed_*/models/pitch3_lte_manifest.json"))
    manifest_paths += sorted(run_root.glob("cv/fold_*/models/pitch3_lte_manifest.json"))
    assert len(manifest_paths) == 8
    for path in manifest_paths:
        m = read_json(path)
        folder = path.parent
        cv = m["validation_scope"] == "train_family_cv"
        name = f"cv_fold_{m['cv_fold']}" if cv else f"seed_{m['seed']}"
        assert m["training_manifest_sha256"] == dataset_hash
        assert m["fingerprint_json_sha256"] == fp_hash
        assert m["dataset_plan_sha256"] == plan_hash
        assert m["source_training_config_sha256"] == config_hash
        for filename, key in [
            ("pitch3_lte_effective_config.json", "training_config_sha256"),
            ("pitch3_lte_run_protocol.json", "run_protocol_sha256"),
            ("pitch3_lte_training_statistics.json", "training_statistics_sha256"),
            ("pitch3_lte_selection_predictions.csv", "selection_predictions_sha256"),
        ]:
            digest(folder / filename, m[key])
        digest(folder / Path(m["checkpoint"]).name, m["checkpoint_sha256"])
        for c in m["checkpoint_candidates"].values():
            digest(folder / Path(c["checkpoint"]).name, c["checkpoint_sha256"])
        protocol = read_json(folder / "pitch3_lte_run_protocol.json")
        fit_ids, selection_ids = (
            set(protocol["train_sample_ids"]),
            set(protocol["selection_sample_ids"]),
        )
        assert not fit_ids & selection_ids
        if cv:
            assert m["local_residual_mode"] == "zero_global_only"
            expected_families = sorted(train.prompt_family.unique())[m["cv_fold"] :: 5]
            expected_selection = train[train.prompt_family.isin(expected_families)]
            expected_fit = train[~train.prompt_family.isin(expected_families)]
        else:
            expected_fit, expected_selection = train, dev
            digest(path, members[m["seed"]]["manifest_sha256"])
            assert m["checkpoint_sha256"] == members[m["seed"]]["checkpoint_sha256"]
        assert fit_ids == set(expected_fit.sample_id)
        assert selection_ids == set(expected_selection.sample_id)
        assert set(protocol["selection_families"]) == set(expected_selection.prompt_family)
        assert set(protocol["train_families"]) == set(expected_fit.prompt_family)
        stats = read_json(folder / "pitch3_lte_training_statistics.json")
        assert stats["run_protocol_sha256"] == m["run_protocol_sha256"]
        y = expected_fit.loc[
            expected_fit.source_kind == "base_step4_seed", "energy_target"
        ].to_numpy()
        thresholds = [0.0, *np.quantile(y[y > 0], [1 / 3, 2 / 3])]
        assert np.allclose(thresholds, stats["energy_stratification"]["thresholds"], atol=1e-12)
        frame = read_predictions(folder / "pitch3_lte_selection_predictions.csv", selection_ids)
        verify_metrics(frame, m["best_development_metrics"])
        for epoch in m["training_history"]:
            for source, variant in epoch["development_variants"].items():
                metrics = variant["metrics"]
                histories.append(
                    {
                        "run": name,
                        "epoch": epoch["epoch"],
                        "stage": epoch["training_stage"],
                        "source": source,
                        "minimum_family_rho": metrics["minimum_prompt_family_spearman"],
                        "direction": metrics["local_direction_sign_accuracy"],
                        "all_gates_passed": metrics["all_gates_passed"],
                        **{
                            f"train_contribution_{k}": v
                            for k, v in epoch["active_normalized_training_contributions"].items()
                        },
                    }
                )
        history = [h for h in histories if h["run"] == name]
        selected = {
            "run": name,
            "seed": m["seed"],
            "selected_epoch": m["best_epoch"],
            "epochs_completed": m["epochs_completed"],
            "global_epochs": len({h["epoch"] for h in history if h["stage"] == "global"}),
            "local_epochs": len({h["epoch"] for h in history if h["stage"] == "local"}),
            "selected_global_epoch": m["checkpoint_candidates"]["global_stage"]["epoch"],
            "any_history_passed_all_gates": any(h["all_gates_passed"] for h in history),
            "max_historical_minimum_family_rho": max(h["minimum_family_rho"] for h in history),
            "metrics": m["best_development_metrics"],
        }
        if cv:
            cv_rows.append(selected)
            cv_families.extend(
                {"fold": m["cv_fold"], **r} for r in family_diagnostics(frame, lower, upper)
            )
        else:
            seeds.append(selected)
            seed_frames.append(frame.set_index("sample_id").loc[current.sample_id].reset_index())

    ensemble_errors = {}
    for column in ["predicted_energy", "predicted_global_energy", "predicted_local_energy"]:
        mean = np.mean([f[column].to_numpy() for f in seed_frames], axis=0)
        ensemble_errors[column] = float(np.max(abs(mean - current[column])))
        assert np.allclose(mean, current[column], atol=4e-7, rtol=4e-7)
    seed_pairs = [pairs(f).set_index("direction_id").sort_index() for f in seed_frames]
    seed_pairs = [f[f.nontrivial] for f in seed_pairs]
    exact = seed_pairs[0].exact_derivative.to_numpy()
    for p in seed_pairs[1:]:
        assert p.index.equals(seed_pairs[0].index)
        assert np.array_equal(exact, p.exact_derivative)
    predicted = np.stack([p.predicted_derivative.to_numpy() for p in seed_pairs], axis=1)
    correct = np.sign(predicted) == np.sign(exact[:, None])
    averaged_correct = np.sign(predicted.mean(1)) == np.sign(exact)
    votes = correct.sum(1)
    diagnostic_pairs = seed_pairs[0][["prompt_id", "prompt_family", "exact_derivative"]].copy()
    diagnostic_pairs["correct_member_count"] = votes
    diagnostic_pairs["energy_mean_correct"] = averaged_correct
    diagnostic_pairs["majority_correct_mean_wrong"] = (votes >= 2) & ~averaged_correct
    diagnostic_pairs["largest_absolute_member"] = np.array([s["seed"] for s in seeds])[
        abs(predicted).argmax(1)
    ]
    for i, seed in enumerate(seeds):
        diagnostic_pairs[f"derivative_{seed['seed']}"] = predicted[:, i]
        seed["direction_correct"] = int(correct[:, i].sum())
        seed["absolute_derivative_median"] = float(np.median(abs(predicted[:, i])))
        seed["absolute_derivative_p90"] = float(np.quantile(abs(predicted[:, i]), 0.9))
    ensemble_directions = {
        "nontrivial_pairs": len(exact),
        "energy_mean_correct": int(averaged_correct.sum()),
        "majority_sign_correct_diagnostic_only": int((votes >= 2).sum()),
        "majority_correct_mean_wrong": int(((votes >= 2) & ~averaged_correct).sum()),
        "majority_wrong_mean_correct": int(((votes < 2) & averaged_correct).sum()),
        "wrong_minority_member_counts": {
            str(s["seed"]): int(
                (
                    diagnostic_pairs.majority_correct_mean_wrong
                    & (diagnostic_pairs.largest_absolute_member == s["seed"])
                ).sum()
            )
            for s in seeds
        },
        "pairwise_derivative_spearman": [[rho(a, b) for b in predicted.T] for a in predicted.T],
        "majority_is_not_a_deployable_energy_or_guidance_rule": True,
    }
    cv_summary = summarize(run_root / "cv")
    assert cv_summary == read_json(run_root / "cv/pitch3_lte_cv_summary.json")
    for p in [
        Path(__file__),
        root / "scripts/analyze_pitch3_lte_v37_dte.py",
        root / "scripts/summarize_pitch3_lte_cv.py",
        root / "src/generation/pitch3_lte.py",
        root / "src/generation/pitch3_lte_training.py",
    ]:
        digest(p.resolve())
    summary = {
        "diagnostic_only": True,
        "model_inference_performed": False,
        "checkpoint_selection_performed": False,
        "production_authorization": False,
        "guidance_promotion_eligible": False,
        "dataset": {
            "sha256": dataset_hash,
            "labels_recomputed_from_coordinates": True,
            "train_base_samples": int(
                ((data.split == "train") & (data.source_kind == "base_step4_seed")).sum()
            ),
            "latent_references_present_at_manifest_paths": sum(
                (dataset_path.parent / p).is_file() for p in data.latent_path
            ),
            "latent_content_verified": False,
        },
        "fingerprint_sha256": fp_hash,
        "local_fingerprint_matches_training": True,
        "all_eight_manifests_and_candidate_checkpoint_hashes_verified": True,
        "protocol_splits_verified_against_original_dataset": True,
        "cv_summary_reproduced": True,
        "cv_summary": cv_summary,
        "matched_v37_cv_present": (run_root / "cv_v37_control").exists(),
        "local_cv_present": (run_root / "cv_with_local").exists(),
        "ensemble_mean_max_absolute_errors": ensemble_errors,
        "ensemble_direction_diagnostics": ensemble_directions,
        "seeds": seeds,
        "cv_runs": cv_rows,
        "inputs_sha256": inputs,
    }
    pd.DataFrame(versions).to_csv(out / "version_metrics.csv", index=False)
    pd.DataFrame(families).to_csv(out / "family_diagnostics.csv", index=False)
    pd.DataFrame(cv_families).to_csv(out / "cv_family_diagnostics.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(out / "local_pairs.csv", index=False)
    diagnostic_pairs.to_csv(out / "ensemble_direction_pairs.csv")
    current.to_csv(out / "base_and_local_predictions_with_truth.csv", index=False)
    (out / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    diag = pd.DataFrame(families)
    indices = np.arange(4)
    for i, (version, color) in enumerate(
        [("pitch3_lte_v37_dte", "#5277a3"), ("pitch3_lte_v38a", "#c75c42")]
    ):
        d = diag[diag.version == version]
        for ax, metric in [(axes[0, 0], "rho"), (axes[0, 1], "near_rho")]:
            ax.bar(
                indices + (i - 0.5) * 0.36,
                d[metric],
                width=0.36,
                color=color,
                label=version.replace("pitch3_lte_", ""),
            )
            ax.set_xticks(indices, [f[:3] for f in d.family])
    axes[0, 0].axhline(0.5, ls="--", color="#444", lw=1)
    axes[0, 0].set(
        title="Development family ranking (64 base samples each)",
        ylabel="Spearman rho",
        ylim=(0, 0.8),
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 1].axhline(0, color="#444", lw=1)
    axes[0, 1].set(
        title="Low-energy ordering (exact energy <= 0.1)", ylabel="Spearman rho", ylim=(-0.3, 0.5)
    )
    cvd = pd.DataFrame(cv_families).sort_values("rho")
    axes[1, 0].barh(
        [f[:3] for f in cvd.family],
        cvd.rho,
        color=["#428878" if r >= 0.5 else "#c75c42" for r in cvd.rho],
    )
    axes[1, 0].axvline(0.5, ls="--", color="#444", lw=1)
    axes[1, 0].set(
        title="Train-family CV: 8/20 pass rho >= 0.50",
        xlabel="Selection-fold Spearman rho",
        xlim=(0, 0.75),
    )
    counts = [s["direction_correct"] for s in seeds] + [
        int((votes >= 2).sum()),
        int(averaged_correct.sum()),
    ]
    bars = axes[1, 1].bar(
        ["Seed 41", "Seed 42", "Seed 43", "Sign majority*", "Energy mean"],
        counts,
        color=["#5277a3"] * 3 + ["#999999", "#c75c42"],
    )
    axes[1, 1].bar_label(bars, padding=3)
    axes[1, 1].axhline(62, ls="--", color="#444", lw=1)
    axes[1, 1].set(
        title="Local direction: magnitude disagreement hurts averaging",
        ylabel="Correct nontrivial pairs / 94",
        ylim=(0, 75),
    )
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].text(
        0.02,
        0.04,
        "*Diagnostic only; not a scalar energy or promotion rule.",
        transform=axes[1, 1].transAxes,
        fontsize=8,
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "V3.8-A archived-output audit | Frozen gates unchanged | No new inference", fontsize=13
    )
    fig.savefig(out / "diagnostic_summary.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "hashes_checked": len(inputs),
                "metrics_reproduced": True,
                "cv_summary_reproduced": True,
                "ensemble_direction_diagnostics": ensemble_directions,
                "ensemble_mean_errors": ensemble_errors,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
