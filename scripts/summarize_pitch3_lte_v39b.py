"""Recompute paired internal experiments without model or checkpoint selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from generation.pitch3_lte_metrics import pitch3_lte_metrics  # noqa: E402
from generation.pitch3_lte_v38b_protocol import (  # noqa: E402
    describe_predictions,
    read_csv,
    read_json,
)
from generation.pitch3_lte_v39a_protocol import content_hash, file_hash  # noqa: E402
from generation.pitch3_lte_v39b_protocol import variants_for, verify_run  # noqa: E402
from scripts.summarize_pitch3_lte_v38b import (  # noqa: E402
    _validate_prediction_labels,
    average_rows,
)


def summarize(run_root, dataset, mode, folds, seeds):
    if not folds or len(set(folds)) != len(folds) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique nonempty folds and seeds required")
    rows = read_csv(dataset)
    truth = {r["sample_id"]: r for r in rows}
    coordinates = {
        k: SimpleNamespace(coordinates=json.loads(v["coordinates_json"])) for k, v in truth.items()
    }
    matched, reports, source_contract = {}, {}, None
    for variant in variants_for(mode):
        fold_reports = []
        for fold in folds:
            members, info = [], []
            for seed in seeds:
                folder = run_root / mode / variant / f"fold_{fold}" / f"seed_{seed}"
                path = folder / "pitch3_lte_v39b_complete.json"
                m, p = verify_run(path, rows)
                if (m["mode"], m["variant"], m["cv_fold"], m["seed"]) != (
                    mode,
                    variant,
                    fold,
                    seed,
                ):
                    raise ValueError("Path differs from run identity")
                if p["dataset_manifest_sha256"] != file_hash(dataset):
                    raise ValueError("Dataset differs from run")
                shared = {
                    k: p[k]
                    for k in (
                        "dataset_manifest_sha256",
                        "source_config_sha256",
                        "fingerprint_json_sha256",
                        "teacher_sha256",
                        "experiment",
                    )
                }
                shared["implementation"] = m["implementation_sha256"]
                if source_contract is not None and source_contract != shared:
                    raise ValueError("Mixed training/source contracts")
                source_contract = shared
                identity = {
                    k: m[k]
                    for k in (
                        "shared_initial_state_sha256",
                        "initial_normalizers_sha256",
                        "initial_teacher_head_sha256",
                    )
                }
                identity["fit_ids"] = p["train_sample_ids"]
                identity["outer_ids"] = p["selection_sample_ids"]
                initialization = read_json(folder / "pitch3_lte_initialization.json")
                identity["transition_initial_state"] = initialization.get(
                    "transition_initial_state_sha256"
                )
                key = (fold, seed)
                if key in matched and matched[key] != identity:
                    raise ValueError("Unmatched initialization, normalizers or data")
                matched[key] = identity
                split = "selection" if mode == "cv" else "train"
                predictions = read_csv(folder / f"pitch3_lte_{split}_predictions.csv")
                for r in predictions:
                    _validate_prediction_labels(r, truth[r["sample_id"]])
                metrics = pitch3_lte_metrics(predictions)
                if metrics != m[f"{split}_metrics"]:
                    raise ValueError("Metrics do not reproduce")
                members.append(predictions)
                h = read_json(folder / "pitch3_lte_training_history.json")["history"]
                checkpoints = []
                for epoch in (12, 24, 48):
                    if epoch <= len(h):
                        item = h[epoch - 1]
                        checkpoints.append(
                            {
                                "epoch": epoch,
                                "fit_evaluation": item["fit_evaluation"],
                                "clipping_fraction": item["clipping_fraction"],
                                "block_parameter_updates": item["block_parameter_updates"],
                            }
                        )
                info.append(
                    {
                        "seed": seed,
                        "completion_sha256": file_hash(path),
                        "metrics": metrics,
                        "fixed_epoch_observations_no_selection": checkpoints,
                    }
                )
            stats = read_json(folder / "pitch3_lte_training_statistics.json")
            averaged = average_rows(members)
            metrics = pitch3_lte_metrics(averaged)
            diag = describe_predictions(
                averaged,
                coordinates,
                stats["energy_stratification"]["thresholds"],
                stats["local_training_scales"]["local_derivative_scale"],
                stats["coordinate_auxiliary"]["coordinate_widths"],
            )
            fold_reports.append(
                {"fold": fold, "members": info, "metrics": metrics, "diagnostics": diag}
            )
        family_values = [
            rho for f in fold_reports for rho in f["metrics"]["prompt_family_spearman"].values()
        ]
        low = [
            d["zero_and_low_positive_rho"]
            for f in fold_reports
            for d in f["diagnostics"]["families"].values()
            if d["zero_and_low_positive_rho"] is not None
        ]
        correct = sum(
            round(
                f["metrics"]["local_direction_pairs"]
                * f["metrics"]["local_direction_sign_accuracy"]
            )
            for f in fold_reports
        )
        pairs = sum(f["metrics"]["local_direction_pairs"] for f in fold_reports)
        reports[variant] = {
            "family_count": len(family_values),
            "families_passing_rho": sum(r >= 0.5 for r in family_values),
            "mean_family_rho": float(np.mean(family_values)),
            "min_family_rho": min(family_values),
            "mean_low_rho": float(np.mean(low)) if low else None,
            "direction_correct": correct,
            "direction_pairs": pairs,
            "folds_passing_all_gates": sum(f["metrics"]["all_gates_passed"] for f in fold_reports),
            "folds": fold_reports,
        }
    left, right = variants_for(mode)
    comparison = {}
    for a, b in zip(reports[left]["folds"], reports[right]["folds"], strict=True):
        comparison[f"fold_{a['fold']}"] = {
            family: b["metrics"]["prompt_family_spearman"][family] - rho
            for family, rho in a["metrics"]["prompt_family_spearman"].items()
        }
    return {
        "stage": "pitch3_lte_v39b_internal_summary",
        "mode": mode,
        "folds": folds,
        "seeds": seeds,
        "complete_five_fold": set(folds) == set(range(5)),
        "diagnostic_only": True,
        "development_used": False,
        "checkpoint_selection_performed": False,
        "next_stage_automatically_authorized": False,
        "production_authorization": False,
        "matched_initialization_and_normalizers": True,
        "contract": source_contract,
        "contract_sha256": content_hash(source_contract),
        "variants": reports,
        "paired_family_delta": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["budget", "distill-diagnose", "cv"], required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260941])
    args = parser.parse_args()
    result = summarize(args.run_root, args.dataset_manifest, args.mode, args.folds, args.seeds)
    output = (
        args.run_root
        / "summary"
        / (
            f"{args.mode}_folds_{'_'.join(map(str, args.folds))}"
            f"_seeds_{'_'.join(map(str, args.seeds))}.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "variants": {
                    k: {a: b for a, b in v.items() if a != "folds"}
                    for k, v in result["variants"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
