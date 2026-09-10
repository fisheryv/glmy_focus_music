from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import write_json_atomic
from generation.ltsn_v51b import select_peak_memorization_epoch
from generation.ltsn_v51c import wilson_interval


def _metric_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    train_pairs = int(row.get("overfit_train_central_direction_pairs", 0))
    train_agreement = float(row.get("overfit_train_central_direction_agreement", 0.0))
    development_pairs = int(row.get("central_direction_pairs", 0))
    development_agreement = float(row.get("central_direction_agreement", 0.0))
    development_successes = round(development_pairs * development_agreement)
    interval = (
        wilson_interval(development_successes, development_pairs)
        if development_pairs
        else (None, None)
    )
    return {
        "epoch": int(row["epoch"]),
        "train_loss": float(row["train_loss"]),
        "train_direction_pairs": train_pairs,
        "train_direction_agreement": train_agreement,
        "train_derivative_spearman": float(
            row.get("overfit_train_central_derivative_spearman", 0.0)
        ),
        "train_derivative_mae": row.get("overfit_train_central_derivative_mae"),
        "development_direction_pairs": development_pairs,
        "development_direction_successes": development_successes,
        "development_direction_agreement": development_agreement,
        "development_direction_agreement_wilson95": list(interval),
        "development_derivative_spearman": float(row.get("central_derivative_spearman", 0.0)),
        "development_derivative_mae": row.get("central_derivative_mae"),
        "development_by_step": {
            str(step): {
                "pairs": int(row.get(f"central_direction_step_{step}_pairs", 0)),
                "agreement": float(row.get(f"central_direction_step_{step}_agreement", 0.0)),
            }
            for step in (4, 5, 6)
        },
    }


def _history_epoch(history: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    row = next((item for item in history if int(item.get("epoch", -1)) == epoch), None)
    return None if row is None else _metric_snapshot(row)


def _load_variant(
    *,
    variant: dict[str, Any],
    suite: dict[str, Any],
    models_root: Path,
) -> dict[str, Any]:
    name = str(variant["name"])
    view_name = str(variant["view"])
    view = dict(suite["views"][view_name])
    ensemble_path = models_root / name / "ensemble_manifest.json"
    if not ensemble_path.is_file():
        raise LTSNContractError(f"V5.1c ensemble is missing for {name}: {ensemble_path}")
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    checkpoints = ensemble.get("checkpoints", [])
    if ensemble.get("status") != "engineering_smoke_only" or len(checkpoints) != 1:
        raise LTSNContractError(f"V5.1c {name} requires one diagnostic checkpoint")
    if ensemble.get("qualification_eligible") is not False or ensemble.get("precision") != "fp32":
        raise LTSNContractError(f"V5.1c {name} must be FP32 and qualification-ineligible")
    metadata = ensemble.get("metadata", {})
    if metadata.get("training_manifest_sha256") != view["training_manifest_sha256"]:
        raise LTSNContractError(f"V5.1c {name} checkpoint uses the wrong data view")
    if metadata.get("ltsn_config_sha256") != variant["config_sha256"]:
        raise LTSNContractError(f"V5.1c {name} checkpoint uses the wrong config")
    checkpoint_info = checkpoints[0]
    checkpoint_path = ensemble_path.parent / checkpoint_info["path"]
    if sha256_file(checkpoint_path) != checkpoint_info["sha256"]:
        raise LTSNContractError(f"V5.1c {name} checkpoint is hash-mismatched")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = checkpoint.get("training_config", {})
    if (
        training.get("use_bf16") is not False
        or training.get("central_direction_overfit_diagnostic") is not True
        or training.get("central_direction_classification_only") is not True
        or list(training.get("seeds", [])) != [20260716]
    ):
        raise LTSNContractError(f"V5.1c {name} violates the single-seed diagnostic contract")
    weights = checkpoint.get("loss_weights", {})
    if weights.get("central_direction", 0) <= 0 or any(
        value != 0 for key, value in weights.items() if key != "central_direction"
    ):
        raise LTSNContractError(f"V5.1c {name} is not central-direction-only")
    history = checkpoint.get("history", [])
    if not isinstance(history, list) or not history:
        raise LTSNContractError(f"V5.1c {name} history is empty")
    peak = dict(select_peak_memorization_epoch(history))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best = next((row for row in history if int(row.get("epoch", -1)) == best_epoch), None)
    if best is None:
        raise LTSNContractError(f"V5.1c {name} best epoch is absent from history")
    expected_train_pairs = (
        int(view["permutation"]["informative_pairs"])
        if view_name == "permuted_pair_control"
        else int(suite["train_direction_pairs"])
    )
    peak_snapshot = _metric_snapshot(peak)
    final_snapshot = _metric_snapshot(history[-1])
    integrity = {
        "expected_train_direction_pairs": (
            peak_snapshot["train_direction_pairs"] == expected_train_pairs
        ),
        "full_development_direction_pairs": (
            peak_snapshot["development_direction_pairs"]
            == int(suite["development_direction_pairs"])
        ),
        "history_within_configured_epoch_budget": (
            len(history) <= int(training["max_epochs"])
            and len(history) >= int(training["minimum_epochs"])
        ),
    }
    if not all(integrity.values()):
        raise LTSNContractError(f"V5.1c {name} failed data/history integrity checks")
    return {
        "name": name,
        "view": view_name,
        "restored_factor": variant["restored_factor"],
        "config_sha256": variant["config_sha256"],
        "checkpoint_sha256": checkpoint_info["sha256"],
        "ensemble_manifest_sha256": sha256_file(ensemble_path),
        "epochs_completed": len(history),
        "configured_max_epochs": int(training["max_epochs"]),
        "stopped_early": len(history) < int(training["max_epochs"]),
        "minimum_train_loss": min(float(row["train_loss"]) for row in history),
        "train_selected_peak": peak_snapshot,
        "checkpoint_best_epoch": _metric_snapshot(best),
        "epoch_100": _history_epoch(history, 100),
        "epoch_300": _history_epoch(history, 300),
        "final_epoch": final_snapshot,
        "integrity": integrity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the V5.1c training-factor suite.")
    parser.add_argument("--suite-summary", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    suite_path = args.suite_summary.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("experiment") != "ltsn_v5_1c_one_factor_training_ablation":
        raise LTSNContractError("unexpected V5.1c suite summary")
    if (
        suite.get("diagnostic_only") is not True
        or suite.get("qualification_eligible") is not False
        or suite.get("guidance_promotion_eligible") is not False
        or suite.get("single_seed_screen") is not True
    ):
        raise LTSNContractError("V5.1c suite is not bounded to single-seed diagnosis")
    reports = [
        _load_variant(
            variant=dict(variant),
            suite=suite,
            models_root=args.models_root.resolve(),
        )
        for variant in suite.get("variants", [])
    ]
    if not reports:
        raise LTSNContractError("V5.1c suite contains no variants")
    by_name = {report["name"]: report for report in reports}
    baseline = by_name.get("baseline")
    if baseline is None:
        raise LTSNContractError("V5.1c baseline report is missing")
    baseline_peak = baseline["train_selected_peak"]
    factor_effects = []
    for report in reports:
        if report["view"] != "true_pairs" or report["name"] == "baseline":
            continue
        peak = report["train_selected_peak"]
        factor_effects.append(
            {
                "name": report["name"],
                "restored_factor": report["restored_factor"],
                "delta_train_direction_agreement_vs_baseline": (
                    peak["train_direction_agreement"] - baseline_peak["train_direction_agreement"]
                ),
                "delta_development_direction_agreement_vs_baseline": (
                    peak["development_direction_agreement"]
                    - baseline_peak["development_direction_agreement"]
                ),
                "delta_development_derivative_spearman_vs_baseline": (
                    peak["development_derivative_spearman"]
                    - baseline_peak["development_derivative_spearman"]
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_1c_one_factor_training_ablation",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "single_seed_screen": True,
        "multi_seed_confirmation_required": True,
        "suite_summary_sha256": sha256_file(suite_path),
        "variants": reports,
        "factor_effects_at_train_selected_peak": factor_effects,
        "generalization_supported": False,
        "status": "ablation_complete",
        "interpretation": (
            "single-seed factor effects are descriptive only; inspect full-development "
            "agreement, Wilson intervals, Spearman, and the permuted-pair control before "
            "pre-registering a three-seed confirmation"
        ),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
