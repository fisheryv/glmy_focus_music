"""Audit archived V3.8-B runs without fitting, inference, or modifying source artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from generation.pitch3_lte_metrics import pitch3_lte_metrics  # noqa: E402
from generation.pitch3_lte_v38b_protocol import (  # noqa: E402
    correlation,
    describe_predictions,
    file_hash,
    ranks,
    read_csv,
    read_json,
    validate_protocol,
    verify_file,
)
from scripts.summarize_pitch3_lte_v38b import (  # noqa: E402
    _validate_prediction_labels,
    average_rows,
    direction_disagreement,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    items = [v for v in values if v is not None]
    return float(np.mean(items)) if items else None


def auc(target, score):
    positive = np.asarray(target, dtype=bool)
    p, n = int(positive.sum()), int((~positive).sum())
    return (
        float((ranks(np.asarray(score))[positive].sum() - p * (p - 1) / 2) / (p * n))
        if p and n
        else None
    )


def pairs(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["direction_id"]:
            sign = int(row["direction_sign"])
            assert sign not in groups[row["direction_id"]], "Duplicate direction sign"
            groups[row["direction_id"]][sign] = row
    result = []
    for did, group in sorted(groups.items()):
        assert set(group) == {-1, 1}
        minus, plus = group[-1], group[1]
        eps = float(minus["epsilon"])
        assert eps > 0 and math.isclose(eps, float(plus["epsilon"]), abs_tol=1e-9)
        item = {"direction_id": did, "family": minus["prompt_family"]}
        for label, field in [
            ("exact", "exact_energy"),
            ("predicted", "predicted_energy"),
            ("global", "predicted_global_energy"),
        ]:
            item[label] = (float(plus[field]) - float(minus[field])) / (2 * eps)
        item["nontrivial"] = abs(item["exact"]) > 1e-6
        for key in ["predicted", "global"]:
            item[key + "_correct"] = math.copysign(1, item[key]) == math.copysign(1, item["exact"])
        result.append(item)
    return result


def family_slices(rows, truth, threshold):
    groups = defaultdict(list)
    for row in rows:
        if row["source_kind"] == "base_step4_seed":
            groups[row["prompt_family"]].append(row)
    result = []
    for family, group in sorted(groups.items()):
        y = np.array([float(r["exact_energy"]) for r in group])
        p = np.array([float(r["predicted_energy"]) for r in group])
        q = np.array([json.loads(truth[r["sample_id"]]["coordinates_json"]) for r in group])
        qp = np.array([json.loads(r["predicted_coordinates_json"]) for r in group])
        low = y <= threshold
        item = dict(
            family=family,
            n=len(group),
            rho=correlation(y, p),
            zero_positive_auc=auc(y > 0, p),
            positive_rho=correlation(y[y > 0], p[y > 0]),
            low_threshold=threshold,
            low_n=int(low.sum()),
            low_rho=correlation(y[low], p[low]),
            energy_mae=float(np.mean(abs(y - p))),
        )
        for i in range(3):
            item[f"q{i + 1}_rho"] = correlation(q[:, i], qp[:, i])
            item[f"low_q{i + 1}_rho"] = correlation(q[low, i], qp[low, i])
        if "predicted_ordinal_probabilities_json" in group[0]:
            ordinal = np.array(
                [json.loads(r["predicted_ordinal_probabilities_json"]) for r in group]
            )
            assert np.isfinite(ordinal).all() and np.all((0 <= ordinal) & (ordinal <= 1))
            assert np.all(np.diff(ordinal, axis=1) <= 0)
            item["ordinal_zero_auc"] = auc(y > 0, ordinal[:, 0])
        result.append(item)
    return result


def summary_row(label, report):
    metrics = [
        f["ensemble_metrics"] if "ensemble_metrics" in f else f["metrics"] for f in report["folds"]
    ]
    n = sum(m["local_direction_pairs"] for m in metrics)
    correct = sum(
        round(m["local_direction_sign_accuracy"] * m["local_direction_pairs"]) for m in metrics
    )
    return dict(
        label=label,
        passing_families=report["families_passing_original_rho_gate"],
        min_family_rho=report["minimum_family_spearman"],
        mean_family_rho=report["mean_family_spearman"],
        passing_folds=report["folds_passing_all_original_model_gates"],
        direction_correct=correct,
        direction_pairs=n,
        direction_accuracy=correct / n,
        mean_fold_derivative_rho=mean(m["local_derivative_spearman"] for m in metrics),
        majority_correct_mean_wrong=sum(
            f.get("directions", {}).get("majority_correct_mean_wrong", 0) for f in report["folds"]
        ),
    )


def verify_prediction_archive(path, data, checkpoint_checks):
    """Verify CSV/JSON provenance; record bad model bytes without qualifying them.

    This diagnostic does not replace or relax the production verify_manifest.
    No inference/qualification is possible when the checkpoint check fails.
    """
    m = read_json(path)
    checks = {
        "pitch3_lte_run_protocol.json": m["run_protocol_sha256"],
        "pitch3_lte_effective_config.json": m["training_config_sha256"],
        "pitch3_lte_training_statistics.json": m.get("training_statistics_sha256"),
        "pitch3_lte_train_predictions.csv": m.get("train_predictions_sha256"),
        "pitch3_lte_selection_predictions.csv": m.get("selection_predictions_sha256"),
        **m.get("diagnostic_artifacts_sha256", {}),
    }
    for filename, expected in checks.items():
        if expected is not None:
            verify_file(path.parent / filename, expected)
    checkpoint = path.parent / Path(m["checkpoint"]).name
    actual = file_hash(checkpoint) if checkpoint.is_file() else None
    checkpoint_checks.append(
        dict(
            path=str(checkpoint),
            expected=m["checkpoint_sha256"],
            actual=actual,
            hash_matches=actual == m["checkpoint_sha256"],
            bytes=checkpoint.stat().st_size if checkpoint.exists() else 0,
            valid_zip=zipfile.is_zipfile(checkpoint) if checkpoint.exists() else False,
        )
    )
    protocol = read_json(path.parent / "pitch3_lte_run_protocol.json")
    if m.get("architecture_revision") == "v3.8b_ordinal_calibrated_direction":
        validate_protocol(protocol, data)
        for key in ["cv_fold", "validation_scope", "experiment_contract", "checkpoint_selection"]:
            assert protocol[key] == m[key]
        stats = read_json(path.parent / "pitch3_lte_training_statistics.json")
        assert stats["run_protocol_sha256"] == m["run_protocol_sha256"]
        assert protocol["dataset_manifest_sha256"] == m["training_manifest_sha256"]
        assert protocol["outer_evaluation_count"] == (1 if m["cv_fold"] is not None else 0)
        assert m["validation_scope"] == (
            "train_family_cv" if m["cv_fold"] is not None else "development"
        )
    for split in ["train", "selection"]:
        p = path.parent / f"pitch3_lte_{split}_predictions.csv"
        if p.exists():
            ids = [r["sample_id"] for r in read_csv(p)]
            assert len(set(ids)) == len(ids) and set(ids) == set(protocol[f"{split}_sample_ids"])
        elif m.get("architecture_revision") == "v3.8b_ordinal_calibrated_direction":
            assert split == "selection" and m["cv_fold"] is None
    return m, protocol


def archive_summary(prefix, chosen, rows_by_run, records):
    folds, families, shared = [], {}, None
    for fold in range(5):
        members, member_metrics = [], []
        for seed in chosen:
            name = f"{prefix}/fold_{fold}/seed_{seed}/models"
            m = records[name]
            contract = {
                key: m[key]
                for key in [
                    "source_training_config_sha256",
                    "training_manifest_sha256",
                    "fingerprint_json_sha256",
                    "experiment_contract",
                    "checkpoint_selection",
                    "implementation_sha256",
                ]
            }
            assert shared is None or shared == contract
            shared = contract
            rows = rows_by_run[name]["selection"]
            members.append(rows)
            member_metrics.append(dict(seed=seed, metrics=pitch3_lte_metrics(rows)))
        metrics = pitch3_lte_metrics(average_rows(members))
        assert not set(families) & set(metrics["prompt_family_spearman"])
        families.update(metrics["prompt_family_spearman"])
        folds.append(
            dict(
                fold=fold,
                members=member_metrics,
                ensemble_metrics=metrics,
                directions=direction_disagreement(members),
            )
        )
    assert len(families) == 20
    return dict(
        seeds=chosen,
        contract=shared,
        family_spearman=families,
        families_passing_original_rho_gate=sum(v >= 0.5 for v in families.values()),
        minimum_family_spearman=min(families.values()),
        mean_family_spearman=mean(families.values()),
        folds_passing_all_original_model_gates=sum(
            f["ensemble_metrics"]["all_gates_passed"] for f in folds
        ),
        folds=folds,
    )


def plot_audit(output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    history = read_json(output / "historical_development.json")
    integrity = read_json(output / "audit.json")["inventory"]
    checkpoint_note = (
        f"{integrity['checkpoint_hash_failures']} local checkpoints fail SHA-256 checks"
        if integrity["checkpoint_hash_failures"]
        else "Checkpoint file hashes match the manifests"
    )
    cv = {r["label"]: r for r in read_csv(output / "cv_comparison.csv")}
    fit = [
        r
        for r in read_csv(output / "fit_vs_heldout.csv")
        if r["global_variant"] == "g1" and r["local_variant"] == "l0" and r["fold"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("V3.8-B archived experiment audit", fontsize=17, fontweight="bold")
    a, b = history[-2]["prompt_family_spearman"], history[-1]["prompt_family_spearman"]
    x = np.arange(4)
    ax = axes[0, 0]
    ax.bar(x - 0.18, list(a.values()), 0.36, label="V3.8-A", color="#93a8bd")
    ax.bar(x + 0.18, list(b.values()), 0.36, label="V3.8-B G1/L2", color="#176b87")
    ax.axhline(0.5, color="#ad403d", linestyle="--", label="Frozen gate")
    ax.set(
        xticks=x,
        xticklabels=[k[:3] for k in a],
        ylabel="Spearman rho",
        ylim=(0, 0.8),
        title="A  Development: family energy ranking",
    )
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    labels = ["G0", "G1", "G37"]
    values = [int(cv[f"{v.lower()}_l0_20260941"]["passing_families"]) for v in labels]
    ax.bar(labels, values, color=["#93a8bd", "#176b87", "#8c7aab"])
    ax.set(
        ylim=(0, 20),
        yticks=[0, 5, 10, 15, 20],
        ylabel="Families passing rho >= 0.50 (of 20)",
        title="B  Matched global CV: seed 20260941",
    )
    for i, value in enumerate(values):
        ax.text(i, value + 0.4, f"{value}/20", ha="center")
    ax = axes[1, 0]
    values = [
        float(cv[f"g1_{lv}_20260941_20260942_20260943"]["direction_accuracy"])
        for lv in ["l0", "l1", "l2"]
    ]
    ax.bar(["L0", "L1", "L2"], values, color=["#93a8bd", "#549b8d", "#176b87"])
    ax.axhline(0.65, color="#ad403d", linestyle="--")
    ax.set(
        ylim=(0, 0.75),
        ylabel="Direction accuracy",
        title="C  G1 CV: three-seed equal-weight energies",
    )
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{round(value * 476)}/476", ha="center")
    ax = axes[1, 1]
    for i, (split, label, color) in enumerate(
        [("train", "Fit", "#93a8bd"), ("selection", "Held-out families", "#176b87")]
    ):
        group = [r for r in fit if r["split"] == split]
        values = [mean(float(r[k]) for r in group) for k in ["family_rho", "low_rho"]]
        ax.bar(np.arange(2) + (i - 0.5) * 0.34, values, 0.34, label=label, color=color)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set(
        xticks=[0, 1],
        xticklabels=["All base energies", "Zero + low positive"],
        ylim=(-0.08, 0.6),
        ylabel="Mean family Spearman rho",
        title="D  G1/L0: weak low-energy fit as well as transfer",
    )
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "CV and development are separate scopes. Low-energy cutoffs use fit-only tertiles.\n"
        f"CSV/JSON metrics reproduced; {checkpoint_note}. No new inference.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output / "diagnostic_summary.png", dpi=180)
    fig.savefig(output / "diagnostic_summary.svg")
    plt.close(fig)


def audit(root, output):
    runs = root / "runs/pitch3_lte_v38b"
    if output.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("Audit output must be outside the source runs directory")
    output.mkdir(parents=True, exist_ok=True)
    dataset = root / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv"
    fingerprint = root / "metadata/focus_pitch3_fingerprint_v1.json"
    config = root / "configs/pitch3_lte_v38b.toml"
    plan = dataset.parent / "pitch3_lte_dataset_plan.json"
    data = read_csv(dataset)
    truth = {r["sample_id"]: r for r in data}
    objects = {
        sid: SimpleNamespace(coordinates=json.loads(r["coordinates_json"]))
        for sid, r in truth.items()
    }
    assert len(truth) == len(data)
    tracked = {p: file_hash(p) for p in runs.rglob("*") if p.is_file()}
    for p in [dataset, fingerprint, config, plan]:
        tracked[p] = file_hash(p)
    manifests = sorted((runs / "cv").rglob("pitch3_lte_manifest.json"))
    manifests += sorted((runs / "final").rglob("pitch3_lte_manifest.json"))
    records, rows_by_run, histories, gradients, families, fit_rows, source_checks = (
        {},
        {},
        [],
        [],
        [],
        [],
        [],
    )
    label_count = 0
    checkpoint_checks = []
    for path in manifests:
        name = path.parent.relative_to(runs).as_posix()
        m, protocol = verify_prediction_archive(path, data, checkpoint_checks)
        records[name] = m
        assert m["fingerprint_json_sha256"] == tracked[fingerprint]
        assert m["training_manifest_sha256"] == tracked[dataset]
        assert m["dataset_plan_sha256"] == tracked[plan]
        config_lf_hash = hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert m["source_training_config_sha256"] in {tracked[config], config_lf_hash}
        assert m["epochs_completed"] == 12 and m["checkpoint_parameter_source"] == "online"
        assert m["production_authorization"] is False
        for filename, expected in m["implementation_sha256"].items():
            raw = (root / "src/generation" / filename).read_bytes()
            source_checks.append(
                dict(
                    run=name,
                    file=filename,
                    expected=expected,
                    exact_match=hashlib.sha256(raw).hexdigest() == expected,
                    lf_match=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected,
                )
            )
        fit = [truth[sid] for sid in protocol["train_sample_ids"]]
        stats = read_json(path.parent / "pitch3_lte_training_statistics.json")
        y = np.array(
            [float(r["energy_target"]) for r in fit if r["source_kind"] == "base_step4_seed"]
        )
        threshold = [0.0, *np.quantile(y[y > 0], [1 / 3, 2 / 3])]
        assert np.allclose(
            threshold, stats["energy_stratification"]["thresholds"], rtol=0, atol=1e-12
        )
        fit_pairs = pairs(
            [
                {
                    **r,
                    "exact_energy": r["energy_target"],
                    "predicted_energy": r["energy_target"],
                    "predicted_global_energy": r["energy_target"],
                    "direction_sign": str(int(float(r["direction_sign"]))),
                }
                for r in fit
            ]
        )
        nonzero = [abs(p["exact"]) for p in fit_pairs if p["nontrivial"]]
        assert math.isclose(
            np.median(nonzero),
            stats["local_training_scales"]["local_derivative_scale"],
            rel_tol=0,
            abs_tol=1e-12,
        )
        info = dict(
            run=name,
            global_variant=m["experiment_contract"]["global_variant"],
            local_variant=m["experiment_contract"]["local_variant"],
            fold=m["cv_fold"],
            seed=m["seed"],
        )
        rows_by_run[name] = {}
        for split in ["train", "selection"]:
            pred_path = path.parent / f"pitch3_lte_{split}_predictions.csv"
            if not pred_path.exists():
                assert split == "selection" and m["cv_fold"] is None
                continue
            rows = read_csv(pred_path)
            for row in rows:
                t = truth[row["sample_id"]]
                _validate_prediction_labels(row, t)
                expected_anchor = (
                    t["sample_id"] if t["source_kind"] == "base_step4_seed" else t["trajectory_id"]
                )
                assert row["anchor_sample_id"] == expected_anchor
                assert math.isclose(
                    float(row["epsilon"]), float(t["epsilon"]), rel_tol=2e-7, abs_tol=2e-7
                )
            label_count += len(rows)
            metrics = pitch3_lte_metrics(rows)
            assert (
                metrics == m["train_metrics" if split == "train" else "best_development_metrics"]
            ), name
            rows_by_run[name][split] = rows
            fam = family_slices(rows, truth, threshold[1])
            families.extend([{**info, "split": split, **f} for f in fam])
            diag = read_json(path.parent / f"pitch3_lte_{split}_diagnostics.json")
            assert metrics == diag["metrics"]
            ps = [p for p in pairs(rows) if p["nontrivial"]]
            assert sum(p["predicted_correct"] for p in ps) == diag["local"]["correct"]
            fit_rows.append(
                {
                    **info,
                    "split": split,
                    "family_rho": mean(f["rho"] for f in fam),
                    "passing_families": sum(f["rho"] >= 0.5 for f in fam),
                    "families": len(fam),
                    "low_rho": mean(f["low_rho"] for f in fam),
                    "zero_auc": mean(f["zero_positive_auc"] for f in fam),
                    "ordinal_zero_auc": mean(f.get("ordinal_zero_auc") for f in fam),
                    "direction_accuracy": metrics["local_direction_sign_accuracy"],
                    **diag["local"],
                }
            )
        history = read_json(path.parent / "pitch3_lte_training_history.json")["history"]
        assert history == m["training_history"] and len(history) == 12
        for h in history:
            assert h["selection_evaluated"] is False
            histories.append(
                {
                    **info,
                    **{k: v for k, v in h.items() if k != "normalized_training_contributions"},
                    **{f"loss_{k}": v for k, v in h["normalized_training_contributions"].items()},
                }
            )
        for when in ["initial", "final"]:
            g = read_json(path.parent / f"pitch3_lte_gradient_{when}.json")
            assert set(g["sample_ids"]) <= set(protocol["train_sample_ids"])
            for component, norm in g["weighted_normalized_gradient_norms"].items():
                gradients.append(
                    {
                        **info,
                        "when": when,
                        "component": component,
                        "norm": norm,
                        **{f"cos_{k}": v for k, v in g["gradient_cosine"][component].items()},
                    }
                )
    # Bind every local branch to its actual L0 parent and check unchanged base outputs.
    for name, m in records.items():
        if m["experiment_contract"]["local_variant"] == "l0":
            continue
        parts = name.split("/")
        parts[2] = "l0"
        parent = "/".join(parts)
        parent_file = runs / parent / "pitch3_lte_manifest.json"
        assert m["source_global_manifest_sha256"] == tracked[parent_file]
        assert m["source_global_checkpoint_sha256"] == records[parent]["checkpoint_sha256"]
        assert m["global_parameters_unchanged_during_local"] is True
        for split, rows in rows_by_run[name].items():
            base = {r["sample_id"]: r for r in rows_by_run[parent][split]}
            for row in rows:
                assert (
                    row["predicted_global_energy"]
                    == base[row["sample_id"]]["predicted_global_energy"]
                )
                if row["source_kind"] == "base_step4_seed":
                    assert float(row["predicted_local_energy"]) == 0
                    assert row["predicted_energy"] == base[row["sample_id"]]["predicted_energy"]
    summaries, cv_rows, ensemble_pairs = {}, [], []
    for variant in sorted((runs / "cv").iterdir()):
        if not variant.is_dir():
            continue
        for local in sorted(variant.iterdir()):
            if not local.is_dir():
                continue
            seeds = sorted(
                {
                    m["seed"]
                    for name, m in records.items()
                    if name.startswith(local.relative_to(runs).as_posix() + "/")
                }
            )
            for chosen in [[s] for s in seeds] + ([seeds] if len(seeds) > 1 else []):
                key = f"{variant.name}_{local.name}_" + "_".join(map(str, chosen))
                report = archive_summary(
                    local.relative_to(runs).as_posix(), chosen, rows_by_run, records
                )
                summaries[key] = report
                saved = local / (
                    "pitch3_lte_v38b_cv_summary_" + "_".join(map(str, chosen)) + ".json"
                )
                if saved.exists():
                    original = read_json(saved)
                    assert all(original[k] == v for k, v in report.items() if k != "folds")
                    for a, b in zip(report["folds"], original["folds"], strict=True):
                        assert {k: v for k, v in a.items() if k != "manifests"} == {
                            k: v for k, v in b.items() if k != "manifests"
                        }
                write_json(output / "summaries" / f"{key}.json", report)
                cv_rows.append(summary_row(key, report))
                for fold in range(5):
                    members = [
                        rows_by_run[f"cv/{variant.name}/{local.name}/fold_{fold}/seed_{s}/models"][
                            "selection"
                        ]
                        for s in chosen
                    ]
                    for pair in pairs(average_rows(members)):
                        ensemble_pairs.append(dict(summary=key, fold=fold, **pair))
    control = read_json(runs / "cv_v37_control/pitch3_lte_cv_summary.json")
    train_families = sorted({r["prompt_family"] for r in data if r["split"] == "train"})
    for fold in range(5):
        p = runs / f"cv_v37_control/fold_{fold}/models/pitch3_lte_manifest.json"
        m, protocol = verify_prediction_archive(p, data, checkpoint_checks)
        expected_eval = {
            r["sample_id"]
            for r in data
            if r["split"] == "train" and r["prompt_family"] in train_families[fold::5]
        }
        expected_fit = {r["sample_id"] for r in data if r["split"] == "train"} - expected_eval
        assert set(protocol["selection_sample_ids"]) == expected_eval
        assert set(protocol["train_sample_ids"]) == expected_fit
        rows = read_csv(p.parent / "pitch3_lte_selection_predictions.csv")
        for row in rows:
            _validate_prediction_labels(row, truth[row["sample_id"]])
        metrics = pitch3_lte_metrics(rows)
        assert metrics == m["best_development_metrics"] == control["folds"][fold]["metrics"]
    cv_rows.append(summary_row("legacy_v37_control_selected", control))
    # Final report has real server ensemble inference; global-only below is descriptive reuse.
    screen_dir = runs / "final/g1/l2/development_screen_model_only_fp32"
    screen = read_json(screen_dir / "pitch3_lte_development_screen.json")
    ensemble_path = runs / "final/g1/l2/pitch3_lte_ensemble.json"
    ensemble = read_json(ensemble_path)
    assert screen["ensemble_manifest_sha256"] == tracked[ensemble_path]
    assert screen["dataset_manifest_sha256"] == tracked[dataset]
    verify_file(screen_dir / "pitch3_lte_development_predictions.csv", screen["predictions_sha256"])
    for member in ensemble["members"]:
        folder = runs / f"final/g1/l2/seed_{member['seed']}/models"
        verify_file(folder / "pitch3_lte_manifest.json", member["manifest_sha256"])
        assert (
            member["checkpoint_sha256"]
            == records[f"final/g1/l2/seed_{member['seed']}/models"]["checkpoint_sha256"]
        )
    final_rows = read_csv(screen_dir / "pitch3_lte_development_predictions.csv")
    assert {r["sample_id"] for r in final_rows} == {
        r["sample_id"] for r in data if r["split"] == "development"
    }
    for row in final_rows:
        _validate_prediction_labels(row, truth[row["sample_id"]])
    assert pitch3_lte_metrics(final_rows) == screen["metrics"]
    global_rows = [
        {**r, "predicted_energy": r["predicted_global_energy"], "predicted_local_energy": "0"}
        for r in final_rows
    ]
    final_global = pitch3_lte_metrics(global_rows)
    stats = read_json(runs / "final/g1/l0/seed_20260941/models/pitch3_lte_training_statistics.json")
    final_diag = describe_predictions(
        final_rows,
        objects,
        stats["energy_stratification"]["thresholds"],
        stats["local_training_scales"]["local_derivative_scale"],
        stats["coordinate_auxiliary"]["coordinate_widths"],
    )
    final_families = family_slices(
        final_rows, truth, stats["energy_stratification"]["thresholds"][1]
    )
    fpairs = pairs(final_rows)
    changed = dict(
        global_correct=sum(p["global_correct"] for p in fpairs if p["nontrivial"]),
        total_correct=sum(p["predicted_correct"] for p in fpairs if p["nontrivial"]),
        fixed=sum(
            not p["global_correct"] and p["predicted_correct"] for p in fpairs if p["nontrivial"]
        ),
        broke=sum(
            p["global_correct"] and not p["predicted_correct"] for p in fpairs if p["nontrivial"]
        ),
    )
    write_json(
        output / "final_development.json",
        dict(
            screen=screen,
            global_component_diagnostic=final_global,
            diagnostics=final_diag,
            local_changes=changed,
        ),
    )
    historical = []
    for label in ["pitch3_lte_v35r", "pitch3_lte_v37_dte", "pitch3_lte_v38a"]:
        folder = root / "runs" / label / "development_screen_model_only_fp32"
        old = read_json(folder / "pitch3_lte_development_screen.json")
        old_rows = read_csv(folder / "pitch3_lte_development_predictions.csv")
        assert pitch3_lte_metrics(old_rows) == old["metrics"]
        historical.append(dict(version=label, **old["metrics"]))
        for p in [
            folder / "pitch3_lte_development_screen.json",
            folder / "pitch3_lte_development_predictions.csv",
        ]:
            tracked[p] = file_hash(p)
    historical.append(dict(version="pitch3_lte_v38b", **screen["metrics"]))
    write_json(output / "historical_development.json", historical)
    write_csv(output / "cv_comparison.csv", cv_rows)
    write_csv(output / "fit_vs_heldout.csv", fit_rows)
    write_csv(output / "family_diagnostics.csv", families)
    write_csv(output / "training_history.csv", histories)
    write_csv(output / "gradient_components.csv", gradients)
    write_csv(output / "cv_direction_pairs.csv", ensemble_pairs)
    write_csv(output / "final_family_diagnostics.csv", final_families)
    write_csv(output / "final_direction_pairs.csv", fpairs)
    write_csv(output / "implementation_checks.csv", source_checks)
    write_csv(output / "checkpoint_integrity.csv", checkpoint_checks)
    diag_root = runs / "diagnostics_v38a"
    inventory = dict(
        manifests=len(manifests),
        legacy_control_manifests=5,
        cv_manifests=sum(m["cv_fold"] is not None for m in records.values()),
        final_manifests=sum(m["cv_fold"] is None for m in records.values()),
        summaries_recomputed=len(summaries),
        prediction_rows_verified=label_count,
        archived_diagnostic_folders=[p.name for p in diag_root.iterdir() if p.is_dir()],
        archived_diagnostic_completion_reports=len(
            list(diag_root.rglob("pitch3_lte_checkpoint_diagnostics.json"))
        ),
        source_exact_mismatches=sum(not c["exact_match"] for c in source_checks),
        source_lf_mismatches=sum(not c["lf_match"] for c in source_checks),
        config_source_sha256=records[next(iter(records))]["source_training_config_sha256"],
        config_local_raw_sha256=tracked[config],
        config_local_lf_sha256=hashlib.sha256(
            config.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        checkpoint_hash_failures=sum(not c["hash_matches"] for c in checkpoint_checks),
        checkpoint_invalid_zip=sum(not c["valid_zip"] for c in checkpoint_checks),
    )
    # Include optional old diagnostic CSVs as incomplete evidence, not completed diagnostics.
    partial_diagnostics = []
    for folder in sorted(diag_root.iterdir()):
        if folder.is_dir():
            for split in ["train", "selection"]:
                p = folder / f"pitch3_lte_{split}_predictions.csv"
                if p.exists():
                    rows = read_csv(p)
                    for row in rows:
                        _validate_prediction_labels(row, truth[row["sample_id"]])
                    metrics = pitch3_lte_metrics(rows)
                    partial_diagnostics.append(dict(run=folder.name, split=split, **metrics))
    write_json(output / "incomplete_v38a_diagnostics.json", partial_diagnostics)
    assert all(file_hash(p) == digest for p, digest in tracked.items()), (
        "Source artifact changed during audit"
    )
    write_csv(
        output / "input_sha256.csv",
        [{"path": str(p.relative_to(root)), "sha256": h} for p, h in sorted(tracked.items())],
    )
    report = dict(
        stage="pitch3_lte_v38b_archived_audit",
        inventory=inventory,
        fitting_performed=False,
        inference_performed=False,
        checkpoint_selection_performed=False,
        production_authorization=False,
        source_artifacts_unchanged=True,
        analysis_scope="verified_csv_json_only_not_checkpoint_qualification",
        input_files_hashed=len(tracked),
        final_metrics=screen["metrics"],
        local_changes=changed,
        cv=cv_rows,
    )
    write_json(output / "audit.json", report)
    plot_audit(output)
    print(
        json.dumps(
            dict(
                output=str(output),
                inventory=inventory,
                final=screen["metrics"],
                local_changes=changed,
            ),
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output or args.root / "analysis_archive/pitch3_lte_v38b_audit")
