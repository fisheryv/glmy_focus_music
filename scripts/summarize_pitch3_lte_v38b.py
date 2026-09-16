"""Verify five V3.8-B folds and evaluate equal-weight archived member predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation.pitch3_lte_metrics import pitch3_lte_metrics  # noqa: E402
from generation.pitch3_lte_v38b_protocol import (  # noqa: E402
    file_hash,
    read_csv,
    verify_file,
    verify_manifest,
)


def average_rows(members):
    indexed = [{r["sample_id"]: r for r in rows} for rows in members]
    if not indexed or any(len(d) != len(rows) for d, rows in zip(indexed, members, strict=True)):
        raise ValueError("Empty ensemble or duplicate prediction IDs")
    ids = sorted(indexed[0])
    if any(set(d) != set(ids) for d in indexed):
        raise ValueError("Member prediction IDs differ")
    output = []
    for sid in ids:
        rows = [d[sid] for d in indexed]
        for key in [
            "prompt_id",
            "prompt_family",
            "source_kind",
            "direction_id",
            "direction_sign",
            "epsilon",
            "exact_energy",
            "exact_band",
            "anchor_sample_id",
        ]:
            if any(r[key] != rows[0][key] for r in rows):
                raise ValueError(f"Member labels differ: {sid} / {key}")
        result = {
            k: v
            for k, v in rows[0].items()
            if k not in {"predicted_global_logit", "predicted_ordinal_probabilities_json"}
        }
        for key in ["predicted_energy", "predicted_global_energy", "predicted_local_energy"]:
            values = np.array([float(r[key]) for r in rows])
            if not np.isfinite(values).all():
                raise ValueError("Non-finite member energy")
            result[key] = float(values.mean())
        result["predicted_coordinates_json"] = json.dumps(
            np.mean([json.loads(r["predicted_coordinates_json"]) for r in rows], axis=0).tolist()
        )
        output.append(result)
    return output


def direction_disagreement(members):
    values, truth = [], None
    for rows in members:
        by_direction = {}
        for r in rows:
            if r["direction_id"]:
                by_direction.setdefault(r["direction_id"], {})[int(r["direction_sign"])] = r
        pred, exact = [], []
        for pair in [by_direction[k] for k in sorted(by_direction)]:
            if set(pair) != {-1, 1}:
                raise ValueError("Incomplete direction pair")
            eps = float(pair[-1]["epsilon"])
            if eps <= 0 or eps != float(pair[1]["epsilon"]):
                raise ValueError("Direction epsilon mismatch")
            d = (float(pair[1]["exact_energy"]) - float(pair[-1]["exact_energy"])) / (2 * eps)
            if abs(d) <= 1e-6:
                continue
            exact.append(d)
            pred.append(
                (float(pair[1]["predicted_energy"]) - float(pair[-1]["predicted_energy"]))
                / (2 * eps)
            )
        if truth is not None and not np.array_equal(truth, exact):
            raise ValueError("Direction labels differ")
        truth = np.array(exact)
        values.append(pred)
    p = np.array(values)
    if truth is None or not len(truth):
        return {"pairs": 0}
    # Match the existing gate's copysign treatment of exactly-zero predictions.
    signs = np.where(p < 0, -1, 1)
    correct = signs == np.where(truth < 0, -1, 1)
    majority = correct.sum(0) > len(values) / 2
    mean_correct = np.where(p.mean(0) < 0, -1, 1) == np.where(truth < 0, -1, 1)
    return {
        "pairs": len(truth),
        "member_correct": correct.sum(1).tolist(),
        "member_abs_derivative_median": np.median(abs(p), axis=1).tolist(),
        "majority_correct_diagnostic_only": int(majority.sum()),
        "mean_correct": int(mean_correct.sum()),
        "majority_correct_mean_wrong": int((majority & ~mean_correct).sum()),
        "majority_wrong_mean_correct": int((~majority & mean_correct).sum()),
    }


def summarize(root: Path, dataset: Path, seeds: list[int]):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Summary seeds must be nonempty and unique")
    data = read_csv(dataset)
    by_id = {r["sample_id"]: r for r in data}
    shared = None
    folds = []
    family_rho = {}
    for fold in range(5):
        members, manifests, member_metrics = [], [], []
        for seed in seeds:
            path = root / f"fold_{fold}/seed_{seed}/models/pitch3_lte_manifest.json"
            m, protocol = verify_manifest(path, data)
            if (
                m["cv_fold"] != fold
                or m["seed"] != seed
                or m["validation_scope"] != "train_family_cv"
            ):
                raise ValueError("CV manifest fold/seed/scope differs")
            verify_file(dataset, m["training_manifest_sha256"])
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
            if shared is not None and contract != shared:
                raise ValueError("Cannot compare mixed experiment contracts within a CV summary")
            shared = contract
            rows = read_csv(path.parent / "pitch3_lte_selection_predictions.csv")
            if len(rows) != len(protocol["selection_sample_ids"]) or {
                r["sample_id"] for r in rows
            } != set(protocol["selection_sample_ids"]):
                raise ValueError("Selection predictions do not cover the held-out fold")
            for row in rows:
                truth = by_id[row["sample_id"]]
                for key in [
                    "prompt_id",
                    "prompt_family",
                    "source_kind",
                    "direction_id",
                    "direction_sign",
                ]:
                    if row[key] != truth[key]:
                        raise ValueError(f"Prediction detached from true {key}")
                for key, target in [
                    ("exact_band", "exact_band"),
                    ("exact_energy", "energy_target"),
                ]:
                    if not math.isclose(
                        float(row[key]), float(truth[target]), rel_tol=2e-7, abs_tol=2e-7
                    ):
                        raise ValueError("Prediction exact label mismatch")
            metrics = pitch3_lte_metrics(rows)
            if metrics != m["best_development_metrics"]:
                raise ValueError("Saved CV metrics do not reproduce")
            member_metrics.append({"seed": seed, "metrics": metrics})
            members.append(rows)
            manifests.append({"path": str(path), "sha256": file_hash(path)})
        averaged = average_rows(members)
        metrics = pitch3_lte_metrics(averaged)
        if set(metrics["prompt_family_spearman"]) & set(family_rho):
            raise ValueError("Repeated held-out family")
        family_rho.update(metrics["prompt_family_spearman"])
        folds.append(
            {
                "fold": fold,
                "manifests": manifests,
                "members": member_metrics,
                "ensemble_metrics": metrics,
                "directions": direction_disagreement(members),
            }
        )
    if len(family_rho) != 20:
        raise ValueError("CV must cover all 20 training families")
    return {
        "stage": "pitch3_lte_v38b_train_family_cv",
        "diagnostic_only": True,
        "production_authorization": False,
        "development_used": False,
        "checkpoint_selection_performed": False,
        "ensemble_inference_performed": False,
        "ensemble_prediction_source": "float64_arithmetic_mean_of_archived_member_outputs",
        "seeds": seeds,
        "contract": shared,
        "family_spearman": family_rho,
        "families_passing_original_rho_gate": sum(v >= 0.5 for v in family_rho.values()),
        "minimum_family_spearman": min(family_rho.values()),
        "mean_family_spearman": float(np.mean(list(family_rho.values()))),
        "folds_passing_all_original_model_gates": sum(
            f["ensemble_metrics"]["all_gates_passed"] for f in folds
        ),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260941])
    args = parser.parse_args()
    result = summarize(args.cv_root, args.dataset_manifest, args.seeds)
    output = args.cv_root / (
        "pitch3_lte_v38b_cv_summary_" + "_".join(map(str, args.seeds)) + ".json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passing_families": result["families_passing_original_rho_gate"],
            }
        )
    )


if __name__ == "__main__":
    main()
