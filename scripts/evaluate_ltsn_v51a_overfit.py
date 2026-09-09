from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the frozen V5.1a overfit diagnostic.")
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-view-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-train-direction-pairs", type=int, default=128)
    parser.add_argument("--minimum-train-direction-agreement", type=float, default=0.90)
    args = parser.parse_args(argv)

    ensemble_path = args.ensemble_manifest.resolve()
    summary_path = args.diagnostic_view_summary.resolve()
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoints = ensemble.get("checkpoints", [])
    if ensemble.get("status") != "engineering_smoke_only" or len(checkpoints) != 1:
        raise LTSNContractError("V5.1a requires one diagnostic-only checkpoint")
    if ensemble.get("precision") != "fp32":
        raise LTSNContractError("V5.1a must run in FP32")
    target = ensemble.get("training_target_contract", {})
    if (
        target.get("central_direction_overfit_diagnostic") is not True
        or target.get("central_direction_classification_only") is not True
        or target.get("normalize_central_direction_by_rms") is not False
        or target.get("central_direction_exact_margin") != 1e-5
    ):
        raise LTSNContractError("checkpoint does not use the frozen V5.1a target contract")
    if ensemble.get("metadata", {}).get("training_manifest_sha256") != summary.get(
        "training_manifest_sha256"
    ):
        raise LTSNContractError("V5.1a checkpoint uses a different diagnostic view")
    checkpoint_path = ensemble_path.parent / checkpoints[0]["path"]
    if sha256_file(checkpoint_path) != checkpoints[0]["sha256"]:
        raise LTSNContractError("V5.1a checkpoint is hash-mismatched")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("loss_weights", {})
    if weights.get("central_direction", 0) <= 0 or any(
        value != 0 for name, value in weights.items() if name != "central_direction"
    ):
        raise LTSNContractError("V5.1a checkpoint is not central-direction-only")
    history = checkpoint.get("history", [])
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best = next((row for row in history if int(row.get("epoch", -1)) == best_epoch), None)
    if not isinstance(best, dict):
        raise LTSNContractError("V5.1a checkpoint is missing best-epoch history")
    train_pairs = int(best.get("overfit_train_central_direction_pairs", 0))
    train_agreement = float(best.get("overfit_train_central_direction_agreement", 0.0))
    train_spearman = float(best.get("overfit_train_central_derivative_spearman", 0.0))
    criteria = {
        "fp32_forward": ensemble.get("precision") == "fp32",
        "central_direction_only": True,
        "minimum_train_direction_pairs": train_pairs >= args.minimum_train_direction_pairs,
        "minimum_train_direction_agreement": (
            train_agreement >= args.minimum_train_direction_agreement
        ),
    }
    supported = all(criteria.values())
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_1a_central_direction_overfit_diagnostic",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "ensemble_manifest_sha256": sha256_file(ensemble_path),
        "diagnostic_view_summary_sha256": sha256_file(summary_path),
        "seed": checkpoint.get("seed"),
        "best_epoch": best_epoch,
        "train_direction_pairs": train_pairs,
        "train_direction_agreement": train_agreement,
        "train_derivative_spearman": train_spearman,
        "train_derivative_mae": best.get("overfit_train_central_derivative_mae"),
        "development_direction_pairs": best.get("central_direction_pairs"),
        "development_direction_agreement": best.get("central_direction_agreement"),
        "development_derivative_spearman": best.get("central_derivative_spearman"),
        "development_derivative_mae": best.get("central_derivative_mae"),
        "thresholds": {
            "minimum_train_direction_pairs": args.minimum_train_direction_pairs,
            "minimum_train_direction_agreement": args.minimum_train_direction_agreement,
        },
        "criteria": criteria,
        "overfit_signal_supported": supported,
        "status": "overfit_supported" if supported else "overfit_not_supported",
        "interpretation": (
            "the unchanged LTSN can fit the bounded central-direction task; investigate "
            "full-data sampling and generalization"
            if supported
            else "the unchanged LTSN cannot fit the bounded central-direction task; "
            "investigate loss resolution or representation before collecting more data"
        ),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
