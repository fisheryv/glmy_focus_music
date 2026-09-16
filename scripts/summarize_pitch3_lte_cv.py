"""Verify and summarize five train-family LTE folds without loading Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cv_root: Path) -> dict:
    folds = []
    seen_families: set[str] = set()
    shared = None
    for fold in range(5):
        folder = cv_root / f"fold_{fold}" / "models"
        manifest_path = folder / "pitch3_lte_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("validation_scope") != "train_family_cv" or manifest.get("cv_fold") != fold:
            raise ValueError("CV summary requires all five correctly labeled train-family folds")
        protocol_path = folder / "pitch3_lte_run_protocol.json"
        if _hash(protocol_path) != manifest["run_protocol_sha256"]:
            raise ValueError("CV split protocol hash mismatch")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        families = set(protocol["selection_families"])
        if len(families) != 4 or families & seen_families:
            raise ValueError("CV validation families overlap or are incomplete")
        if families & set(protocol["train_families"]):
            raise ValueError("CV train/validation family leakage")
        if set(protocol["train_sample_ids"]) & set(protocol["selection_sample_ids"]):
            raise ValueError("CV train/validation sample leakage")
        seen_families.update(families)
        contract = {
            key: manifest[key]
            for key in (
                "training_manifest_sha256",
                "fingerprint_json_sha256",
                "source_training_config_sha256",
                "architecture_revision",
                "seed",
                "local_residual_mode",
            )
        }
        if shared is not None and shared != contract:
            raise ValueError("CV folds must use the same data, configuration, seed and local mode")
        shared = contract
        checkpoint = folder / Path(manifest["checkpoint"]).name
        if _hash(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("CV checkpoint hash mismatch")
        if (
            _hash(folder / "pitch3_lte_selection_predictions.csv")
            != manifest["selection_predictions_sha256"]
        ):
            raise ValueError("CV selection predictions hash mismatch")
        metrics = manifest["best_development_metrics"]
        if set(metrics["prompt_family_spearman"]) != families:
            raise ValueError("CV metrics do not correspond to the selected families")
        folds.append(
            {
                "fold": fold,
                "manifest_sha256": _hash(manifest_path),
                "best_epoch": manifest["best_epoch"],
                "metrics": metrics,
            }
        )
    family_rho = {
        family: value
        for fold in folds
        for family, value in fold["metrics"]["prompt_family_spearman"].items()
    }
    if len(family_rho) != 20:
        raise ValueError("CV must cover exactly 20 training families")
    return {
        "stage": "pitch3_lte_train_family_cv",
        "diagnostic_only": True,
        "development_used": False,
        "production_authorization": False,
        "guidance_promotion_eligible": False,
        "contract": shared,
        "family_spearman": family_rho,
        "families_passing_original_rho_gate": sum(value >= 0.5 for value in family_rho.values()),
        "minimum_family_spearman": min(family_rho.values()),
        "mean_family_spearman": sum(family_rho.values()) / 20,
        "folds_passing_all_original_model_gates": sum(
            fold["metrics"]["all_gates_passed"] for fold in folds
        ),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.cv_root)
    output = args.cv_root / "pitch3_lte_cv_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"summary": str(output), "minimum_family_spearman": report["minimum_family_spearman"]}
        )
    )


if __name__ == "__main__":
    main()
