from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from generation.ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from generation.ltsn_dataset import LTSNSnapshotDataset, collate_ltsn_batch, read_ltsn_manifest
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.ltsn_training import (
    load_checkpoint_model,
    predict_dataset,
    spearman_correlation,
)
from generation.ltsn_v51c import wilson_interval
from generation.ltsn_v52a import mcnemar_exact


def _load_predictions(
    *,
    model: Any,
    records: list[Any],
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        LTSNSnapshotDataset(records, split),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ltsn_batch,
        pin_memory=device.type == "cuda",
    )
    return predict_dataset(model, loader, device)


def _direction_outcomes(
    prediction: dict[str, Any], *, focus_band_threshold: float, exact_margin: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(prediction["local_direction_group_id"]):
        if group_id:
            groups[str(group_id)].append(index)
    exact_focus = np.asarray(prediction["focus_logit"], dtype=float)
    predicted_focus = np.asarray(prediction["predicted_focus_logit"], dtype=float)
    exact_loss = np.maximum(0.0, focus_band_threshold - exact_focus) ** 2
    signs = np.asarray(prediction["local_direction_sign"], dtype=float)
    rms_values = np.asarray(prediction["local_direction_rms_ratio"], dtype=float)
    steps = np.asarray(prediction["step_number"], dtype=int)
    ood = np.asarray(prediction["ood_label"], dtype=float)
    outcomes = []
    for group_id, indices in sorted(groups.items()):
        by_sign = {float(signs[index]): index for index in indices}
        if set(by_sign) != {-1.0, 1.0} or len(indices) != 2:
            raise LTSNContractError(f"V5.2a evaluation pair is incomplete: {group_id}")
        minus, plus = by_sign[-1.0], by_sign[1.0]
        if ood[minus] >= 0.5 or ood[plus] >= 0.5:
            continue
        difference = float(exact_loss[minus] - exact_loss[plus])
        if abs(difference) < exact_margin:
            continue
        rms = float(rms_values[minus])
        predicted = float(predicted_focus[plus] - predicted_focus[minus])
        outcomes.append(
            {
                "direction_group_id": group_id,
                "step_number": int(steps[minus]),
                "rms_ratio": rms,
                "exact_derivative": difference / (2.0 * rms),
                "predicted_derivative": predicted / (2.0 * rms),
                "correct": bool(np.sign(difference) == np.sign(predicted)),
            }
        )
    if not outcomes:
        raise LTSNContractError("V5.2a evaluation produced no informative direction pairs")
    exact = np.asarray([row["exact_derivative"] for row in outcomes], dtype=float)
    predicted = np.asarray([row["predicted_derivative"] for row in outcomes], dtype=float)
    successes = sum(bool(row["correct"]) for row in outcomes)
    low, high = wilson_interval(successes, len(outcomes))
    spearman = spearman_correlation(exact, predicted)
    return {
        "pairs": len(outcomes),
        "successes": successes,
        "direction_agreement": successes / len(outcomes),
        "direction_agreement_wilson95": [low, high],
        "derivative_spearman": spearman,
        "derivative_mae": float(np.mean(np.abs(exact - predicted))),
        "by_step": {
            str(step): {
                "pairs": sum(row["step_number"] == step for row in outcomes),
                "agreement": (
                    sum(row["step_number"] == step and row["correct"] for row in outcomes)
                    / max(1, sum(row["step_number"] == step for row in outcomes))
                ),
            }
            for step in (4, 5, 6)
        },
    }, outcomes


def _compare_outcomes(
    true_outcomes: list[dict[str, Any]], control_outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    true_by_group = {row["direction_group_id"]: bool(row["correct"]) for row in true_outcomes}
    control_by_group = {row["direction_group_id"]: bool(row["correct"]) for row in control_outcomes}
    if set(true_by_group) != set(control_by_group):
        raise LTSNContractError("V5.2a true/control outcomes use different evaluation pairs")
    return mcnemar_exact(true_by_group, control_by_group)


def _load_ensemble(
    *,
    name: str,
    ensemble_path: Path,
    expected_manifest_sha256: str,
    expected_config_sha256: str,
    contract: Any,
    device: torch.device,
    own_train_records: list[Any],
    unseen_records: list[Any],
    seen_records: list[Any],
    exact_margin: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, list[dict[str, Any]]]]]:
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    if (
        ensemble.get("status") != "engineering_smoke_only"
        or ensemble.get("qualification_eligible") is not False
        or ensemble.get("precision") != "fp32"
    ):
        raise LTSNContractError(f"V5.2a {name} ensemble violates diagnostic boundaries")
    metadata = ensemble.get("metadata", {})
    if metadata.get("training_manifest_sha256") != expected_manifest_sha256:
        raise LTSNContractError(f"V5.2a {name} ensemble uses the wrong view")
    if metadata.get("ltsn_config_sha256") != expected_config_sha256:
        raise LTSNContractError(f"V5.2a {name} ensemble uses the wrong config")
    checkpoints = ensemble.get("checkpoints", [])
    if len(checkpoints) != 3 or len({row["seed"] for row in checkpoints}) != 3:
        raise LTSNContractError(f"V5.2a {name} requires exactly three seeds")
    reports = []
    outcome_sets: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for checkpoint_info in checkpoints:
        checkpoint_path = ensemble_path.parent / checkpoint_info["path"]
        if sha256_file(checkpoint_path) != checkpoint_info["sha256"]:
            raise LTSNContractError(f"V5.2a {name} checkpoint hash mismatch")
        model, payload = load_checkpoint_model(checkpoint_path, contract, device)
        seed = int(payload["seed"])
        target = payload["training_target_contract"]
        training = payload["training_config"]
        weights = payload["loss_weights"]
        if (
            training.get("central_direction_grouped_batches") is not True
            or training.get("central_direction_classification_only") is not True
            or training.get("central_direction_overfit_diagnostic") is not True
            or training.get("use_bf16") is not False
            or list(training.get("seeds", [])) != [20260716, 20260717, 20260718]
            or int(training.get("minimum_epochs", 0)) != int(training.get("max_epochs", -1))
            or payload["model_config"].get("dropout") != 0.0
            or payload["model_config"].get("stem_dropout") != 0.0
        ):
            raise LTSNContractError(f"V5.2a {name} checkpoint violates the frozen recipe")
        if weights.get("central_direction", 0) <= 0 or any(
            value != 0 for key, value in weights.items() if key != "central_direction"
        ):
            raise LTSNContractError(f"V5.2a {name} checkpoint is not central-only")
        predictions = {
            "train": _load_predictions(
                model=model, records=own_train_records, split="train", device=device
            ),
            "unseen_anchor": _load_predictions(
                model=model, records=unseen_records, split="development", device=device
            ),
            "seen_anchor_heldout_direction": _load_predictions(
                model=model, records=seen_records, split="development", device=device
            ),
        }
        metrics = {}
        outcome_sets[seed] = {}
        for split_name, prediction in predictions.items():
            metric, outcomes = _direction_outcomes(
                prediction,
                focus_band_threshold=float(target["focus_band_threshold"]),
                exact_margin=exact_margin,
            )
            metrics[split_name] = metric
            outcome_sets[seed][split_name] = outcomes
        reports.append(
            {
                "seed": seed,
                "checkpoint_sha256": checkpoint_info["sha256"],
                "best_epoch": int(payload["best_epoch"]),
                "epochs_completed": len(payload["history"]),
                "metrics": metrics,
            }
        )
    return sorted(reports, key=lambda row: row["seed"]), outcome_sets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate V5.2a identifiability.")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--views-summary", type=Path, required=True)
    parser.add_argument("--true-ensemble", type=Path, required=True)
    parser.add_argument("--control-ensemble", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    views_path = args.views_summary.resolve()
    views = json.loads(views_path.read_text(encoding="utf-8"))
    if (
        views.get("experiment") != "ltsn_v5_2a_multi_direction_identifiability"
        or views.get("diagnostic_only") is not True
        or views.get("qualification_eligible") is not False
        or views.get("guidance_promotion_eligible") is not False
    ):
        raise LTSNContractError("unexpected or unbounded V5.2a views summary")
    contract = load_fingerprint_contract(args.fingerprint.resolve())
    device = torch.device(args.device)
    true_view = views["views"]["true_pairs"]
    control_view = views["views"]["permuted_pair_control"]
    seen_view = views["views"]["seen_anchor_heldout_direction"]
    true_records = read_ltsn_manifest(Path(true_view["manifest"]), contract)
    control_records = read_ltsn_manifest(Path(control_view["manifest"]), contract)
    seen_records = read_ltsn_manifest(Path(seen_view["manifest"]), contract)
    config_sha256 = sha256_file(args.config.resolve())
    true_reports, true_outcomes = _load_ensemble(
        name="true_pairs",
        ensemble_path=args.true_ensemble.resolve(),
        expected_manifest_sha256=true_view["training_manifest_sha256"],
        expected_config_sha256=config_sha256,
        contract=contract,
        device=device,
        own_train_records=true_records,
        unseen_records=true_records,
        seen_records=seen_records,
        exact_margin=float(views["exact_margin"]),
    )
    control_reports, control_outcomes = _load_ensemble(
        name="permuted_pair_control",
        ensemble_path=args.control_ensemble.resolve(),
        expected_manifest_sha256=control_view["training_manifest_sha256"],
        expected_config_sha256=config_sha256,
        contract=contract,
        device=device,
        own_train_records=control_records,
        unseen_records=true_records,
        seen_records=seen_records,
        exact_margin=float(views["exact_margin"]),
    )
    control_by_seed = {row["seed"]: row for row in control_reports}
    comparisons = []
    outcome_rows = []
    for true_report in true_reports:
        seed = true_report["seed"]
        control_report = control_by_seed[seed]
        unseen_true = true_report["metrics"]["unseen_anchor"]
        unseen_control = control_report["metrics"]["unseen_anchor"]
        mcnemar = _compare_outcomes(
            true_outcomes[seed]["unseen_anchor"],
            control_outcomes[seed]["unseen_anchor"],
        )
        comparisons.append(
            {
                "seed": seed,
                "unseen_anchor_agreement_delta": (
                    unseen_true["direction_agreement"] - unseen_control["direction_agreement"]
                ),
                "unseen_anchor_spearman_delta": (
                    unseen_true["derivative_spearman"] - unseen_control["derivative_spearman"]
                ),
                "mcnemar": mcnemar,
            }
        )
        for model_name, source in (
            ("true_pairs", true_outcomes[seed]),
            ("permuted_pair_control", control_outcomes[seed]),
        ):
            for evaluation_split, outcomes in source.items():
                for row in outcomes:
                    outcome_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "evaluation_split": evaluation_split,
                            **row,
                        }
                    )
    outcome_path = args.output.resolve().with_name("v52a_pair_outcomes.csv")
    write_csv_atomic(outcome_path, outcome_rows)
    criteria = {
        "all_expected_pair_counts_present": all(
            row["metrics"]["train"]["pairs"]
            == int(views["informative_pairs_by_partition"]["train_direction"])
            and row["metrics"]["seen_anchor_heldout_direction"]["pairs"]
            == int(views["informative_pairs_by_partition"]["heldout_direction"])
            and row["metrics"]["unseen_anchor"]["pairs"]
            == int(views["informative_pairs_by_partition"]["unseen_anchor"])
            for row in true_reports
        )
        and all(
            row["metrics"]["train"]["pairs"]
            == int(control_view["informative_permuted_train_pairs"])
            and row["metrics"]["seen_anchor_heldout_direction"]["pairs"]
            == int(views["informative_pairs_by_partition"]["heldout_direction"])
            and row["metrics"]["unseen_anchor"]["pairs"]
            == int(views["informative_pairs_by_partition"]["unseen_anchor"])
            for row in control_reports
        ),
        "true_train_agreement_each_seed_at_least_0_95": all(
            row["metrics"]["train"]["direction_agreement"] >= 0.95 for row in true_reports
        ),
        "control_train_agreement_each_seed_at_least_0_95": all(
            row["metrics"]["train"]["direction_agreement"] >= 0.95 for row in control_reports
        ),
        "seen_direction_agreement_each_seed_at_least_0_60": all(
            row["metrics"]["seen_anchor_heldout_direction"]["direction_agreement"] >= 0.60
            for row in true_reports
        ),
        "seen_direction_wilson_lower_each_seed_above_0_50": all(
            row["metrics"]["seen_anchor_heldout_direction"]["direction_agreement_wilson95"][0]
            > 0.50
            for row in true_reports
        ),
        "unseen_anchor_agreement_each_seed_at_least_0_60": all(
            row["metrics"]["unseen_anchor"]["direction_agreement"] >= 0.60 for row in true_reports
        ),
        "unseen_anchor_wilson_lower_each_seed_above_0_50": all(
            row["metrics"]["unseen_anchor"]["direction_agreement_wilson95"][0] > 0.50
            for row in true_reports
        ),
        "unseen_anchor_spearman_positive_each_seed": all(
            row["metrics"]["unseen_anchor"]["derivative_spearman"] > 0.0 for row in true_reports
        ),
        "true_minus_control_unseen_agreement_each_seed_at_least_0_05": all(
            row["unseen_anchor_agreement_delta"] >= 0.05 for row in comparisons
        ),
        "mcnemar_true_better_p_lt_0_05_at_least_two_seeds": sum(
            row["mcnemar"]["true_only_correct"] > row["mcnemar"]["control_only_correct"]
            and row["mcnemar"]["two_sided_exact_p"] < 0.05
            for row in comparisons
        )
        >= 2,
    }
    supported = all(criteria.values())
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_2a_multi_direction_identifiability",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "views_summary_sha256": sha256_file(views_path),
        "config_sha256": config_sha256,
        "true_pairs": true_reports,
        "permuted_pair_control": control_reports,
        "paired_unseen_anchor_comparisons": comparisons,
        "pair_outcomes": str(outcome_path),
        "pair_outcomes_sha256": sha256_file(outcome_path),
        "thresholds": {
            "minimum_train_agreement": 0.95,
            "minimum_seen_direction_agreement": 0.60,
            "minimum_unseen_anchor_agreement": 0.60,
            "minimum_true_minus_control_agreement": 0.05,
            "wilson_lower_must_exceed": 0.50,
            "mcnemar_p": 0.05,
            "minimum_significant_seeds": 2,
        },
        "criteria": criteria,
        "identifiability_supported": supported,
        "status": (
            "multi_direction_identifiability_supported"
            if supported
            else "multi_direction_identifiability_not_supported"
        ),
        "interpretation": (
            "held-out directions and unseen anchors support a topology-conditioned local field; "
            "this remains diagnostic and does not authorize generation guidance"
            if supported
            else "the critic does not distinguish the frozen topology direction field from the "
            "permuted-pair control on held-out directions and unseen anchors"
        ),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
