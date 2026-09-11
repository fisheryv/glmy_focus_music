from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from generation.ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.ltsn_v51c import wilson_interval
from generation.ltsn_v52a import mcnemar_exact
from generation.ltsn_v52b import (
    PRIMARY_VARIANT,
    V52B_EXPERIMENT,
    V52B_TARGET_MODES,
    V52B_VARIANTS,
)
from generation.ltsn_v52b_training import (
    V52BPairDataset,
    collate_v52b_pairs,
    load_v52b_model,
    predict_v52b,
    read_v52b_pairs,
)

OPERATIONAL_SPLITS = ("train", "seen_anchor_heldout_direction", "unseen_anchor")
ALL_SPLITS = (
    *OPERATIONAL_SPLITS,
    "train_rms_sensitivity",
    "seen_anchor_heldout_direction_rms_sensitivity",
    "unseen_anchor_rms_sensitivity",
)


def _spearman(rows: list[dict[str, Any]]) -> float:
    from generation.ltsn_v52b_training import _spearman as calculate

    return calculate(
        [float(row["exact_derivative"]) for row in rows],
        [float(row["logit"]) / (2.0 * float(row["rms_ratio"])) for row in rows],
    )


def _metrics(rows: list[dict[str, Any]], *, target_key: str = "target") -> dict[str, Any]:
    successes = sum((float(row["logit"]) > 0.0) == (float(row[target_key]) > 0.5) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    return {
        "pairs": len(rows),
        "successes": successes,
        "direction_agreement": successes / len(rows),
        "direction_agreement_wilson95": [low, high],
        "derivative_spearman": _spearman(rows),
        "by_step": {
            str(step): {
                "pairs": sum(int(row["step_number"]) == step for row in rows),
                "agreement": (
                    sum(
                        int(row["step_number"]) == step
                        and (float(row["logit"]) > 0.0) == (float(row[target_key]) > 0.5)
                        for row in rows
                    )
                    / max(1, sum(int(row["step_number"]) == step for row in rows))
                ),
            }
            for step in (4, 5, 6)
        },
    }


def _loader(records: list[Any], split: str, target_mode: str) -> DataLoader[Any]:
    return DataLoader(
        V52BPairDataset(records, split, target_mode=target_mode),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v52b_pairs,
    )


def _evaluate_ensemble(
    *,
    ensemble_path: Path,
    expected_variant: str,
    expected_target_mode: str,
    expected_hashes: dict[str, str],
    contract: Any,
    records: list[Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, list[dict[str, Any]]]]]:
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    if (
        ensemble.get("experiment") != V52B_EXPERIMENT
        or ensemble.get("diagnostic_only") is not True
        or ensemble.get("qualification_eligible") is not False
        or ensemble.get("guidance_promotion_eligible") is not False
        or ensemble.get("precision") != "fp32"
        or ensemble.get("variant") != expected_variant
        or ensemble.get("target_mode") != expected_target_mode
    ):
        raise LTSNContractError("unexpected or unbounded V5.2b ensemble")
    for name, expected in expected_hashes.items():
        if ensemble.get(name) != expected:
            raise LTSNContractError(f"V5.2b ensemble {name} mismatch")
    checkpoints = ensemble.get("checkpoints", [])
    if len(checkpoints) != 3 or len({row["seed"] for row in checkpoints}) != 3:
        raise LTSNContractError("V5.2b ensemble requires exactly three seeds")
    reports = []
    outcomes: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for checkpoint in checkpoints:
        checkpoint_path = ensemble_path.parent / checkpoint["path"]
        if sha256_file(checkpoint_path) != checkpoint["sha256"]:
            raise LTSNContractError("V5.2b checkpoint hash mismatch")
        model, payload = load_v52b_model(checkpoint_path, contract, device)
        if (
            payload.get("variant") != expected_variant
            or payload.get("target_mode") != expected_target_mode
            or payload.get("pair_manifest_sha256") != expected_hashes["pair_manifest_sha256"]
            or payload.get("preparation_sha256") != expected_hashes["preparation_sha256"]
            or payload.get("config_sha256") != expected_hashes["config_sha256"]
        ):
            raise LTSNContractError("V5.2b checkpoint metadata mismatch")
        seed = int(payload["seed"])
        split_rows = {
            split: predict_v52b(model, _loader(records, split, "true"), device)
            for split in ALL_SPLITS
        }
        train_fit_rows = predict_v52b(
            model, _loader(records, "train", expected_target_mode), device
        )
        metrics = {split: _metrics(rows) for split, rows in split_rows.items()}
        metrics["train_target_fit"] = _metrics(train_fit_rows, target_key="training_target")
        reports.append(
            {
                "seed": seed,
                "checkpoint_sha256": checkpoint["sha256"],
                "best_epoch": int(payload["best_epoch"]),
                "epochs_completed": len(payload["history"]),
                "metrics": metrics,
            }
        )
        outcomes[seed] = split_rows
    return sorted(reports, key=lambda row: row["seed"]), outcomes


def _compare(true_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_correct = {
        row["pair_id"]: (float(row["logit"]) > 0.0) == (float(row["target"]) > 0.5)
        for row in true_rows
    }
    control_correct = {
        row["pair_id"]: (float(row["logit"]) > 0.0) == (float(row["target"]) > 0.5)
        for row in control_rows
    }
    return mcnemar_exact(true_correct, control_correct)


def _criteria(
    true_reports: list[dict[str, Any]],
    control_reports: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    expected_counts: dict[str, int],
) -> dict[str, bool]:
    control_by_seed = {row["seed"]: row for row in control_reports}
    return {
        "all_expected_pair_counts_present": all(
            all(
                int(row["metrics"][split]["pairs"]) == int(expected_counts[split])
                for split in ALL_SPLITS
            )
            for row in true_reports + control_reports
        ),
        "true_train_agreement_each_seed_at_least_0_95": all(
            row["metrics"]["train_target_fit"]["direction_agreement"] >= 0.95
            for row in true_reports
        ),
        "control_train_target_fit_each_seed_at_least_0_95": all(
            control_by_seed[row["seed"]]["metrics"]["train_target_fit"]["direction_agreement"]
            >= 0.95
            for row in true_reports
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the V5.2b probe suite.")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    preparation_path = args.preparation.resolve()
    pair_path = args.pair_manifest.resolve()
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if (
        preparation.get("experiment") != V52B_EXPERIMENT
        or preparation.get("diagnostic_only") is not True
        or preparation.get("qualification_eligible") is not False
        or preparation.get("guidance_promotion_eligible") is not False
        or preparation.get("primary_variant") != PRIMARY_VARIANT
        or preparation.get("pair_manifest_sha256") != sha256_file(pair_path)
    ):
        raise LTSNContractError("invalid or stale V5.2b preparation")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    contract = load_fingerprint_contract(args.fingerprint.resolve())
    records = read_v52b_pairs(pair_path)
    expected_hashes = {
        "fingerprint_sha256": sha256_file(args.fingerprint.resolve()),
        "pair_manifest_sha256": sha256_file(pair_path),
        "preparation_sha256": sha256_file(preparation_path),
        "config_sha256": sha256_file(args.config.resolve()),
    }
    reports: dict[str, Any] = {}
    outcome_rows = []
    support_by_variant = {}
    for variant in V52B_VARIANTS:
        loaded = {}
        for target_mode in V52B_TARGET_MODES:
            ensemble_path = (
                args.models_root.resolve() / variant / target_mode / "ensemble_manifest.json"
            )
            loaded[target_mode] = _evaluate_ensemble(
                ensemble_path=ensemble_path,
                expected_variant=variant,
                expected_target_mode=target_mode,
                expected_hashes=expected_hashes,
                contract=contract,
                records=records,
                device=device,
            )
        true_reports, true_outcomes = loaded["true"]
        control_reports, control_outcomes = loaded["matched_control"]
        control_by_seed = {row["seed"]: row for row in control_reports}
        comparisons = []
        for true_report in true_reports:
            seed = true_report["seed"]
            true_unseen = true_report["metrics"]["unseen_anchor"]
            control_unseen = control_by_seed[seed]["metrics"]["unseen_anchor"]
            comparisons.append(
                {
                    "seed": seed,
                    "unseen_anchor_agreement_delta": (
                        true_unseen["direction_agreement"] - control_unseen["direction_agreement"]
                    ),
                    "unseen_anchor_spearman_delta": (
                        true_unseen["derivative_spearman"] - control_unseen["derivative_spearman"]
                    ),
                    "mcnemar": _compare(
                        true_outcomes[seed]["unseen_anchor"],
                        control_outcomes[seed]["unseen_anchor"],
                    ),
                }
            )
        criteria = _criteria(
            true_reports,
            control_reports,
            comparisons,
            preparation["pair_counts"],
        )
        support_by_variant[variant] = all(criteria.values())
        reports[variant] = {
            "true": true_reports,
            "matched_control": control_reports,
            "paired_unseen_anchor_comparisons": comparisons,
            "criteria": criteria,
            "operational_rms_identifiability_supported": all(criteria.values()),
        }
        for model_name, source in (
            ("true", true_outcomes),
            ("matched_control", control_outcomes),
        ):
            for seed, split_rows in source.items():
                for split, rows in split_rows.items():
                    for row in rows:
                        outcome_rows.append(
                            {
                                "variant": variant,
                                "model": model_name,
                                "seed": seed,
                                "evaluation_split": split,
                                **row,
                            }
                        )
    outcome_path = args.output.resolve().with_name("v52b_pair_outcomes.csv")
    write_csv_atomic(outcome_path, outcome_rows)
    primary_supported = bool(support_by_variant[PRIMARY_VARIANT])
    cross_rms_supported = bool(preparation["cross_rms_direction_field_supported"])
    payload = {
        "schema_version": 1,
        "experiment": V52B_EXPERIMENT,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "primary_variant": PRIMARY_VARIANT,
        "preparation_sha256": sha256_file(preparation_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "config_sha256": sha256_file(args.config.resolve()),
        "thresholds": {
            "minimum_train_agreement": 0.95,
            "minimum_seen_direction_agreement": 0.60,
            "minimum_unseen_anchor_agreement": 0.60,
            "minimum_true_minus_control_agreement": 0.05,
            "wilson_lower_must_exceed": 0.50,
            "mcnemar_p": 0.05,
            "minimum_significant_seeds": 2,
        },
        "scale_consistency_audit": preparation["scale_consistency_audit"],
        "variants": reports,
        "support_by_variant": support_by_variant,
        "operational_rms_identifiability_supported": primary_supported,
        "cross_rms_direction_field_supported": cross_rms_supported,
        "infinitesimal_latent_guidance_supported": (primary_supported and cross_rms_supported),
        "pair_outcomes": str(outcome_path),
        "pair_outcomes_sha256": sha256_file(outcome_path),
        "status": (
            "operational_probe_supported_but_cross_rms_unstable"
            if primary_supported and not cross_rms_supported
            else (
                "operational_and_cross_rms_direction_supported"
                if primary_supported and cross_rms_supported
                else "v52b_direction_identifiability_not_supported"
            )
        ),
        "interpretation": (
            "only the preregistered direction-field head can support the operational-RMS claim; "
            "cross-RMS consistency remains independently required for infinitesimal guidance, "
            "and no V5.2b result authorizes qualification or generation guidance"
        ),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
