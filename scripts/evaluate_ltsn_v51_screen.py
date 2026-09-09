from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the frozen V5.1 single-seed screen.")
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--training-view-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-direction-agreement", type=float, default=0.60)
    parser.add_argument("--minimum-direction-spearman", type=float, default=0.15)
    parser.add_argument("--minimum-direction-pairs", type=int, default=128)
    args = parser.parse_args(argv)

    ensemble_path = args.ensemble_manifest.resolve()
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    view = json.loads(args.training_view_summary.resolve().read_text(encoding="utf-8"))
    checkpoints = ensemble.get("checkpoints", [])
    if ensemble.get("status") != "engineering_smoke_only" or len(checkpoints) != 1:
        raise LTSNContractError("V5.1 screen requires one engineering-smoke checkpoint")
    if ensemble.get("precision") != "fp32":
        raise LTSNContractError("V5.1 screen must run in FP32")
    target_contract = ensemble.get("training_target_contract", {})
    if (
        target_contract.get("normalize_central_direction_by_rms") is not False
        or target_contract.get("central_direction_exact_margin") != 1e-5
        or target_contract.get("central_direction_primary_early_stopping") is not True
    ):
        raise LTSNContractError(
            "V5.1 screen checkpoint does not use the frozen raw-difference training contract"
        )
    if ensemble.get("metadata", {}).get("training_manifest_sha256") != view.get(
        "training_manifest_sha256"
    ):
        raise LTSNContractError("V5.1 screen checkpoint uses a different training view")
    checkpoint_path = ensemble_path.parent / checkpoints[0]["path"]
    if sha256_file(checkpoint_path) != checkpoints[0]["sha256"]:
        raise LTSNContractError("V5.1 screen checkpoint is hash-mismatched")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best = next((row for row in history if int(row.get("epoch", -1)) == best_epoch), None)
    if not isinstance(best, dict):
        raise LTSNContractError("V5.1 screen checkpoint is missing its best-epoch history")
    agreement = float(best.get("central_direction_agreement", 0.0))
    spearman = float(best.get("central_derivative_spearman", 0.0))
    pairs = int(best.get("central_direction_pairs", 0))
    criteria = {
        "fp32_forward": ensemble.get("precision") == "fp32",
        "minimum_central_direction_pairs": pairs >= args.minimum_direction_pairs,
        "central_direction_agreement": agreement >= args.minimum_direction_agreement,
        "central_derivative_spearman": spearman > args.minimum_direction_spearman,
    }
    supported = all(criteria.values())
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_1_stable_finite_difference_screen",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "ensemble_manifest_sha256": sha256_file(ensemble_path),
        "training_view_summary_sha256": sha256_file(args.training_view_summary.resolve()),
        "seed": checkpoint.get("seed"),
        "best_epoch": best_epoch,
        "central_direction_pairs": pairs,
        "central_direction_agreement": agreement,
        "central_derivative_spearman": spearman,
        "central_derivative_mae": best.get("central_derivative_mae"),
        "focus_logit_spearman": best.get("focus_logit_spearman"),
        "score_mae": best.get("score_mae"),
        "thresholds": {
            "minimum_direction_pairs": args.minimum_direction_pairs,
            "minimum_direction_agreement": args.minimum_direction_agreement,
            "minimum_direction_spearman": args.minimum_direction_spearman,
        },
        "criteria": criteria,
        "formal_v51_training_recommended": supported,
        "status": "signal_supported" if supported else "signal_not_supported",
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
