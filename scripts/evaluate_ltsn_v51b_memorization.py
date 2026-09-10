from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import write_json_atomic
from generation.ltsn_v51b import (
    memorization_agreement_threshold,
    select_peak_memorization_epoch,
)


def _load_rung(
    *,
    rung: dict[str, Any],
    models_root: Path,
    maximum_train_loss: float,
) -> dict[str, Any]:
    count = int(rung["train_anchors"])
    name = str(rung["name"])
    if (
        rung.get("diagnostic_only") is not True
        or rung.get("qualification_eligible") is not False
        or rung.get("guidance_promotion_eligible") is not False
    ):
        raise LTSNContractError(f"V5.1b {name} is not bounded to diagnostic-only use")
    ensemble_path = models_root / name / "ensemble_manifest.json"
    if not ensemble_path.is_file():
        raise LTSNContractError(f"V5.1b ensemble is missing for {name}: {ensemble_path}")
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    checkpoints = ensemble.get("checkpoints", [])
    if ensemble.get("status") != "engineering_smoke_only" or len(checkpoints) != 1:
        raise LTSNContractError(f"V5.1b {name} requires one engineering-smoke checkpoint")
    if ensemble.get("qualification_eligible") is not False:
        raise LTSNContractError(f"V5.1b {name} checkpoint must be qualification-ineligible")
    if ensemble.get("precision") != "fp32":
        raise LTSNContractError(f"V5.1b {name} must use FP32")
    model = ensemble.get("checkpoints", [])[0]
    metadata = ensemble.get("metadata", {})
    if metadata.get("training_manifest_sha256") != rung.get("training_manifest_sha256"):
        raise LTSNContractError(f"V5.1b {name} checkpoint uses a different rung manifest")
    checkpoint_path = ensemble_path.parent / model["path"]
    if sha256_file(checkpoint_path) != model["sha256"]:
        raise LTSNContractError(f"V5.1b {name} checkpoint is hash-mismatched")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    training = checkpoint.get("training_config", {})
    if config.get("dropout") != 0.0 or config.get("stem_dropout") != 0.0:
        raise LTSNContractError(f"V5.1b {name} did not disable all model dropout")
    if (
        training.get("weight_decay") != 0.0
        or training.get("use_bf16") is not False
        or training.get("central_direction_overfit_diagnostic") is not True
        or training.get("central_direction_classification_only") is not True
        or int(training.get("minimum_epochs", 0)) != int(training.get("max_epochs", -1))
    ):
        raise LTSNContractError(f"V5.1b {name} is not deterministic memorization training")
    weights = checkpoint.get("loss_weights", {})
    if weights.get("central_direction", 0) <= 0 or any(
        value != 0 for key, value in weights.items() if key != "central_direction"
    ):
        raise LTSNContractError(f"V5.1b {name} is not central-direction-only")
    history = checkpoint.get("history", [])
    peak = select_peak_memorization_epoch(history)
    minimum_loss = min(float(row["train_loss"]) for row in history)
    expected_pairs = count * 2
    agreement = float(peak["overfit_train_central_direction_agreement"])
    threshold = memorization_agreement_threshold(count)
    criteria = {
        "expected_train_direction_pairs": int(peak.get("overfit_train_central_direction_pairs", 0))
        == expected_pairs,
        "minimum_train_direction_agreement": agreement >= threshold,
        "maximum_train_loss": minimum_loss <= maximum_train_loss,
        "completed_fixed_epochs": len(history) == int(training["max_epochs"]),
    }
    return {
        "name": name,
        "train_anchors": count,
        "train_direction_pairs": expected_pairs,
        "checkpoint_sha256": model["sha256"],
        "ensemble_manifest_sha256": sha256_file(ensemble_path),
        "epochs_completed": len(history),
        "peak_epoch": int(peak["epoch"]),
        "peak_train_direction_agreement": agreement,
        "peak_train_derivative_spearman": float(peak["overfit_train_central_derivative_spearman"]),
        "peak_train_derivative_mae": peak.get("overfit_train_central_derivative_mae"),
        "minimum_train_loss": minimum_loss,
        "development_direction_agreement_at_peak": peak.get("central_direction_agreement"),
        "development_derivative_spearman_at_peak": peak.get("central_derivative_spearman"),
        "thresholds": {
            "minimum_train_direction_agreement": threshold,
            "maximum_train_loss": maximum_train_loss,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate all deterministic V5.1b memorization rungs."
    )
    parser.add_argument("--ladder-summary", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-train-loss", type=float, default=0.1)
    args = parser.parse_args(argv)
    ladder_path = args.ladder_summary.resolve()
    models_root = args.models_root.resolve()
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    if ladder.get("experiment") != "ltsn_v5_1b_deterministic_memorization_ladder":
        raise LTSNContractError("unexpected V5.1b ladder summary")
    if (
        ladder.get("diagnostic_only") is not True
        or ladder.get("qualification_eligible") is not False
        or ladder.get("guidance_promotion_eligible") is not False
    ):
        raise LTSNContractError("V5.1b ladder is not bounded to diagnostic-only use")
    rung_reports = [
        _load_rung(
            rung=dict(rung),
            models_root=models_root,
            maximum_train_loss=args.maximum_train_loss,
        )
        for rung in ladder.get("rungs", [])
    ]
    if not rung_reports:
        raise LTSNContractError("V5.1b ladder summary contains no rungs")
    passed = all(report["passed"] for report in rung_reports)
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_1b_deterministic_memorization_ladder",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "ladder_summary_sha256": sha256_file(ladder_path),
        "rungs": rung_reports,
        "memorization_supported": passed,
        "status": "memorization_supported" if passed else "memorization_not_supported",
        "interpretation": (
            "the current encoder can deterministically memorize the bounded direction task"
            if passed
            else "the current representation/training path fails at least one deterministic "
            "memorization rung; do not collect more direction data yet"
        ),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
