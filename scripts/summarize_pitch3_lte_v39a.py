"""Verify paired V3.9-A CV artifacts and compare frozen-gate archived predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from generation.pitch3_lte_metrics import pitch3_lte_metrics  # noqa: E402
from generation.pitch3_lte_v39a_protocol import (  # noqa: E402
    VARIANTS,
    file_hash,
    read_csv,
    verify_file,
    verify_manifest,
)
from scripts.summarize_pitch3_lte_v38b import (  # noqa: E402
    _validate_prediction_labels,
    average_rows,
    direction_disagreement,
)


def summarize(run_root: Path, dataset: Path, seeds: list[int], variants=VARIANTS):
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError("Summary needs unique nonnegative seeds")
    if not variants or len(variants) != len(set(variants)) or not set(variants) <= set(VARIANTS):
        raise ValueError("Summary needs unique known variants")
    data = read_csv(dataset)
    truth = {r["sample_id"]: r for r in data}
    reports, matched, shared_contract = {}, {}, None
    for variant in variants:
        folds, family_rho = [], {}
        for fold in range(5):
            members, member_info = [], []
            for seed in seeds:
                path = (
                    run_root
                    / f"cv/{variant}/fold_{fold}/seed_{seed}/models/pitch3_lte_manifest.json"
                )
                m, protocol = verify_manifest(path, data)
                if (
                    m["cv_fold"],
                    m["seed"],
                    m["run_mode"],
                    m["experiment_contract"]["variant"],
                ) != (fold, seed, "cv", variant):
                    raise ValueError("CV path differs from fold/seed/variant metadata")
                verify_file(dataset, m["training_manifest_sha256"])
                contract = {
                    k: m[k]
                    for k in (
                        "source_training_config_sha256",
                        "training_manifest_sha256",
                        "fingerprint_json_sha256",
                        "checkpoint_selection",
                        "implementation_sha256",
                    )
                }
                contract["experiment"] = {
                    k: v for k, v in m["experiment_contract"].items() if k != "variant"
                }
                if shared_contract is not None and shared_contract != contract:
                    raise ValueError("Cannot compare mixed training/implementation contracts")
                shared_contract = contract
                identity = {
                    "shared_initial_state_sha256": m["shared_initial_state_sha256"],
                    "initial_normalizers_sha256": m["initial_normalizers_sha256"],
                    "fit_ids": protocol["train_sample_ids"],
                    "outer_ids": protocol["selection_sample_ids"],
                }
                key = (fold, seed)
                if key in matched and matched[key] != identity:
                    raise ValueError(
                        f"Unmatched baseline/transition initialization or statistics: {key}"
                    )
                matched[key] = identity
                predictions = read_csv(path.parent / "pitch3_lte_selection_predictions.csv")
                for row in predictions:
                    _validate_prediction_labels(row, truth[row["sample_id"]])
                metrics = pitch3_lte_metrics(predictions)
                if metrics != m["best_development_metrics"]:
                    raise ValueError("Saved CV metrics do not reproduce")
                members.append(predictions)
                member_info.append(
                    {
                        "seed": seed,
                        "manifest": str(path),
                        "manifest_sha256": file_hash(path),
                        "metrics": metrics,
                    }
                )
            metrics = pitch3_lte_metrics(average_rows(members))
            if set(family_rho) & set(metrics["prompt_family_spearman"]):
                raise ValueError("A family appears in multiple outer folds")
            family_rho.update(metrics["prompt_family_spearman"])
            folds.append(
                {
                    "fold": fold,
                    "members": member_info,
                    "ensemble_metrics": metrics,
                    "directions": direction_disagreement(members),
                }
            )
        if len(family_rho) != 20:
            raise ValueError("Summary must cover all 20 training families")
        correct = sum(f["directions"].get("mean_correct", 0) for f in folds)
        total = sum(f["directions"]["pairs"] for f in folds)
        reports[variant] = {
            "family_spearman": family_rho,
            "families_passing_original_rho_gate": sum(r >= 0.5 for r in family_rho.values()),
            "minimum_family_spearman": min(family_rho.values()),
            "mean_family_spearman": float(np.mean(list(family_rho.values()))),
            "folds_passing_all_original_model_gates": sum(
                f["ensemble_metrics"]["all_gates_passed"] for f in folds
            ),
            "direction_correct": correct,
            "direction_pairs": total,
            "direction_accuracy": correct / total if total else None,
            "folds": folds,
        }
    comparison = None
    if set(variants) == set(VARIANTS):
        base, candidate = (reports[v]["family_spearman"] for v in VARIANTS)
        comparison = {
            "family_rho_delta": {f: candidate[f] - base[f] for f in base},
            "families_improved": sum(candidate[f] > base[f] for f in base),
            "families_worsened": sum(candidate[f] < base[f] for f in base),
            "new_passing_families": [f for f in base if base[f] < 0.5 <= candidate[f]],
            "lost_passing_families": [f for f in base if candidate[f] < 0.5 <= base[f]],
            "direction_net_correct": reports["transition"]["direction_correct"]
            - reports["baseline"]["direction_correct"],
            "matched_initialization_and_normalizers": True,
        }
    return {
        "stage": "pitch3_lte_v39a_train_family_cv",
        "diagnostic_only": True,
        "development_used": False,
        "checkpoint_selection_performed": False,
        "ensemble_inference_performed": False,
        "production_authorization": False,
        "ensemble_prediction_source": "float64_arithmetic_mean_of_archived_member_energies",
        "seeds": seeds,
        "shared_contract": shared_contract,
        "variants": reports,
        "paired_comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260941])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    report = summarize(args.run_root, args.dataset_manifest, args.seeds, args.variants)
    output = (
        args.run_root
        / "summary"
        / (
            "pitch3_lte_v39a_cv_"
            + "_".join(args.variants)
            + "_"
            + "_".join(map(str, args.seeds))
            + ".json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "variants": {
                    k: {
                        name: v[name]
                        for name in (
                            "families_passing_original_rho_gate",
                            "minimum_family_spearman",
                            "direction_accuracy",
                        )
                    }
                    for k, v in report["variants"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
