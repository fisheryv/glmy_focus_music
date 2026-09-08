"""Calibration, independent qualification, and exact guidance verification."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_dataset import LTSNSnapshotDataset, collate_ltsn_batch, read_ltsn_manifest
from .ltsn_pipeline import canonical_json_sha256, write_json_atomic
from .ltsn_training import load_checkpoint_model, predict_dataset, spearman_correlation

LEGACY_GUIDANCE_PROMOTION_GATE_NAME = "latent_guidance_promotion_v1"
GUIDANCE_PROMOTION_GATE_NAME = "latent_guidance_promotion_v2"


def _load_ensemble(
    ensemble_manifest: Path,
    fingerprint_path: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], list[Any]]:
    contract = load_fingerprint_contract(fingerprint_path)
    payload = json.loads(ensemble_manifest.read_text(encoding="utf-8"))
    target_contract = payload.get("training_target_contract")
    target_contract_sha256 = payload.get("metadata", {}).get("training_target_contract_sha256")
    if target_contract_sha256 is not None and (
        not isinstance(target_contract, dict)
        or canonical_json_sha256(target_contract) != target_contract_sha256
    ):
        raise LTSNContractError("ensemble training target contract is hash-mismatched")
    models = []
    for item in payload.get("checkpoints", []):
        path = ensemble_manifest.parent / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise LTSNContractError(f"checkpoint SHA-256 mismatch: {path}")
        model, checkpoint = load_checkpoint_model(path, contract, device)
        if checkpoint["metadata"] != payload["metadata"]:
            raise LTSNContractError("ensemble checkpoint metadata differs from its manifest")
        models.append(model)
    if not models:
        raise LTSNContractError("ensemble manifest contains no checkpoints")
    return contract, payload, models


def _loader(records: Sequence[Any], split: str, batch_size: int) -> DataLoader[Any]:
    return DataLoader(
        LTSNSnapshotDataset(records, split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ltsn_batch,
    )


def _ensemble_predictions(
    models: Sequence[Any], loader: DataLoader[Any], device: torch.device
) -> dict[str, Any]:
    members = [predict_dataset(model, loader, device) for model in models]
    sample_ids = members[0]["sample_id"]
    if any(member["sample_id"] != sample_ids for member in members[1:]):
        raise RuntimeError("ensemble member prediction order differs")
    means = np.stack([member["coordinate_mean"] for member in members])
    logvars = np.stack([member["coordinate_logvar"] for member in members])
    scores = np.stack([member["predicted_focus_logit"] for member in members])
    ood_logits = np.stack([member["ood_logit"] for member in members])
    base = members[0]
    active_masks = [
        model.coordinate_active_mask.detach().cpu().numpy().astype(bool) for model in models
    ]
    if any(not np.array_equal(active_masks[0], mask) for mask in active_masks[1:]):
        raise LTSNContractError("ensemble members use different active-coordinate masks")
    return {
        **{
            name: base[name]
            for name in (
                "sample_id",
                "prompt_id",
                "trajectory_id",
                "coordinates",
                "focus_logit",
                "ood_label",
                "step_number",
            )
        },
        "coordinate_mean": means.mean(axis=0),
        "member_coordinate_mean": means,
        "member_logvar": logvars,
        "predicted_focus_logit": scores.mean(axis=0),
        "ood_probability": (1.0 / (1.0 + np.exp(-ood_logits))).max(axis=0),
        "aleatoric_variance": np.exp(logvars).mean(axis=(0, 2)),
        "epistemic_variance": means.var(axis=0).mean(axis=1),
        "total_variance": np.exp(logvars).mean(axis=0) + means.var(axis=0),
        "active_coordinate_mask": active_masks[0],
    }


def calibrate_ensemble(
    *,
    fingerprint_path: Path,
    manifest_path: Path,
    ensemble_manifest: Path,
    output_path: Path,
    batch_size: int = 8,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Freeze 90% interval scaling and no-op thresholds on calibration only."""

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    contract, ensemble, models = _load_ensemble(ensemble_manifest, fingerprint_path, device)
    if sha256_file(manifest_path) != ensemble["metadata"]["training_manifest_sha256"]:
        raise LTSNContractError("calibration manifest differs from the trained ensemble")
    records = read_ltsn_manifest(manifest_path, contract)
    prediction = _ensemble_predictions(models, _loader(records, "calibration", batch_size), device)
    labels = prediction["ood_label"]
    in_distribution = labels < 0.5
    if not np.any(in_distribution):
        raise LTSNContractError("calibration requires in-distribution samples")
    residual = np.abs(prediction["coordinates"] - prediction["coordinate_mean"])
    standard = np.sqrt(np.maximum(prediction["total_variance"], 1e-12))
    ratio = residual / standard
    quantile = np.quantile(ratio[in_distribution], 0.90, axis=0)
    variance_scale = np.maximum((quantile / 1.6448536269514722) ** 2, 1e-8)
    calibrated_variance = prediction["total_variance"] * variance_scale[None, :]
    calibrated_aleatoric = (
        np.exp(prediction["member_logvar"]).mean(axis=0) * variance_scale[None, :]
    ).mean(axis=1)
    calibrated_epistemic = (
        prediction["member_coordinate_mean"].var(axis=0) * variance_scale[None, :]
    ).mean(axis=1)
    interval_width = np.mean(2.0 * 1.6448536269514722 * np.sqrt(calibrated_variance), axis=1)
    ood = prediction["ood_probability"]
    calibration_steps = np.asarray(prediction["step_number"], dtype=int)
    matched_steps = sorted(
        int(step)
        for step in np.unique(calibration_steps)
        if np.any(labels[calibration_steps == step] >= 0.5)
        and np.any(labels[calibration_steps == step] < 0.5)
    )
    matched_balanced: dict[str, float] = {}
    if np.any(labels >= 0.5):
        candidates = np.unique(ood)
        best_threshold = float(candidates[0])
        best_objective = -math.inf
        best_balanced = -math.inf
        for threshold in candidates:
            predicted = ood >= threshold
            sensitivity = np.mean(predicted[labels >= 0.5])
            specificity = np.mean(~predicted[in_distribution])
            balanced = 0.5 * (sensitivity + specificity)
            step_scores = []
            for step in matched_steps:
                step_mask = calibration_steps == step
                step_positive = step_mask & (labels >= 0.5)
                step_negative = step_mask & (labels < 0.5)
                step_scores.append(
                    0.5 * (np.mean(predicted[step_positive]) + np.mean(~predicted[step_negative]))
                )
            # The minimum matched-step score prevents a threshold from exploiting
            # a step/OOD distribution mismatch.  Global balanced accuracy is the
            # deterministic tie-breaker for older engineering-smoke manifests.
            objective = min(step_scores) if step_scores else float(balanced)
            if (objective, balanced) > (best_objective, best_balanced):
                best_objective = float(objective)
                best_balanced = float(balanced)
                best_threshold = float(threshold)
        selected = ood >= best_threshold
        for step in matched_steps:
            step_mask = calibration_steps == step
            step_positive = step_mask & (labels >= 0.5)
            step_negative = step_mask & (labels < 0.5)
            matched_balanced[str(step)] = float(
                0.5 * (np.mean(selected[step_positive]) + np.mean(~selected[step_negative]))
            )
    else:
        best_threshold = float(np.quantile(ood[in_distribution], 0.95))
        best_balanced = None
    payload = {
        "schema_version": 2,
        "status": "engineering_smoke_only" if not ensemble["qualification_eligible"] else "frozen",
        "qualification_eligible": bool(ensemble["qualification_eligible"]),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(prediction["sample_id"]),
        "variance_scale": variance_scale.tolist(),
        "ood_probability_threshold": best_threshold,
        "ood_calibration_balanced_accuracy": best_balanced,
        "ood_calibration_matched_step_balanced_accuracy": matched_balanced,
        "ood_calibration_worst_step_balanced_accuracy": (
            min(matched_balanced.values()) if matched_balanced else None
        ),
        "max_aleatoric_variance": float(np.quantile(calibrated_aleatoric[in_distribution], 0.95)),
        "max_epistemic_variance": float(np.quantile(calibrated_epistemic[in_distribution], 0.95)),
        "max_interval_width": float(np.quantile(interval_width[in_distribution], 0.95)),
        "policy": (
            "95th percentile ID no-op thresholds; OOD threshold maximizes worst "
            "matched-step balanced accuracy on calibration only"
        ),
    }
    write_json_atomic(output_path, payload)
    payload["calibration_sha256"] = sha256_file(output_path)
    return payload


def _quartile_ranking_accuracy(exact: np.ndarray, predicted: np.ndarray) -> float:
    low = np.flatnonzero(exact <= np.quantile(exact, 0.25))
    high = np.flatnonzero(exact >= np.quantile(exact, 0.75))
    if not len(low) or not len(high):
        return 0.0
    comparisons = predicted[high, None] > predicted[None, low]
    return float(np.mean(comparisons))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels >= 0.5]
    negative = scores[labels < 0.5]
    if not len(positive) or not len(negative):
        return None
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _metrics(prediction: Mapping[str, Any], variance_scale: np.ndarray) -> dict[str, Any]:
    ood_label = prediction["ood_label"]
    in_distribution = ood_label < 0.5
    if not np.any(in_distribution):
        raise LTSNContractError("qualification metrics require in-distribution samples")
    exact = prediction["coordinates"][in_distribution]
    mean = prediction["coordinate_mean"][in_distribution]
    coordinate_rhos = [spearman_correlation(mean[:, index], exact[:, index]) for index in range(18)]
    active_mask = np.asarray(
        prediction.get("active_coordinate_mask", np.ones(18, dtype=bool)), dtype=bool
    )
    if active_mask.shape != (18,) or not active_mask.any():
        raise LTSNContractError("qualification active-coordinate mask is malformed")
    score_rho = spearman_correlation(
        prediction["predicted_focus_logit"][in_distribution],
        prediction["focus_logit"][in_distribution],
    )
    pitch_rho = spearman_correlation(
        np.linalg.norm(mean[:, :16], axis=1), np.linalg.norm(exact[:, :16], axis=1)
    )
    phase_rho = spearman_correlation(
        np.linalg.norm(mean[:, 16:], axis=1), np.linalg.norm(exact[:, 16:], axis=1)
    )
    variance = prediction["total_variance"][in_distribution] * variance_scale[None, :]
    half_width = 1.6448536269514722 * np.sqrt(np.maximum(variance, 0.0))
    covered = (exact >= mean - half_width) & (exact <= mean + half_width)
    coverage = float(np.mean(covered[:, active_mask]))
    ood_prediction = prediction["ood_probability"]
    return {
        "n": len(prediction["coordinates"]),
        "n_in_distribution": len(exact),
        "focus_logit_spearman": score_rho,
        "coordinate_spearman": coordinate_rhos,
        "active_coordinate_count": int(np.count_nonzero(active_mask)),
        "coordinate_median_spearman": float(np.median(np.asarray(coordinate_rhos)[active_mask])),
        "pitch_block_distance_spearman": pitch_rho,
        "phase_block_distance_spearman": phase_rho,
        "acoustic_loop_coordinate_spearman": coordinate_rhos[16],
        "chroma_loop_coordinate_spearman": coordinate_rhos[17],
        "quartile_ranking_accuracy": _quartile_ranking_accuracy(
            prediction["focus_logit"][in_distribution],
            prediction["predicted_focus_logit"][in_distribution],
        ),
        "interval_90_coverage": coverage,
        "coordinate_mae": np.mean(np.abs(mean - exact), axis=0).tolist(),
        "coordinate_rmse": np.sqrt(np.mean((mean - exact) ** 2, axis=0)).tolist(),
        "focus_logit_mae": float(
            np.mean(
                np.abs(
                    prediction["predicted_focus_logit"][in_distribution]
                    - prediction["focus_logit"][in_distribution]
                )
            )
        ),
        "ood_auroc": _auc(ood_label, ood_prediction),
    }


def _static_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "focus_logit_spearman": metrics["focus_logit_spearman"] >= 0.70,
        "coordinate_median_spearman": metrics["coordinate_median_spearman"] >= 0.50,
        "pitch_block_distance_spearman": metrics["pitch_block_distance_spearman"] >= 0.50,
        "phase_block_distance_spearman": metrics["phase_block_distance_spearman"] >= 0.50,
        "acoustic_loop_coordinate_spearman": metrics["acoustic_loop_coordinate_spearman"] >= 0.50,
        "chroma_loop_coordinate_spearman": metrics["chroma_loop_coordinate_spearman"] >= 0.50,
        "quartile_ranking_accuracy": metrics["quartile_ranking_accuracy"] >= 0.65,
        "interval_90_coverage": 0.85 <= metrics["interval_90_coverage"] <= 0.95,
        "ood_noop_reliability": isinstance(metrics["ood_auroc"], (int, float))
        and math.isfinite(metrics["ood_auroc"])
        and metrics["ood_auroc"] >= 0.80,
    }


def _matched_step_ood_summary(metrics_by_step: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = {
        str(step): float(metrics["ood_auroc"])
        for step, metrics in metrics_by_step.items()
        if isinstance(metrics.get("ood_auroc"), (int, float))
        and math.isfinite(float(metrics["ood_auroc"]))
    }
    return {
        "by_step": values,
        "macro_auroc": None if not values else float(np.mean(list(values.values()))),
        "worst_step_auroc": None if not values else min(values.values()),
        "required_steps_present": {4, 5, 6, 8}.issubset({int(step) for step in values}),
    }


def qualify_ensemble(
    *,
    fingerprint_path: Path,
    manifest_path: Path,
    ensemble_manifest: Path,
    calibration_path: Path,
    output_path: Path,
    guidance_development_report: Path | None = None,
    batch_size: int = 8,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the one-shot independent, step-stratified LTSN qualification."""

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    contract, ensemble, models = _load_ensemble(ensemble_manifest, fingerprint_path, device)
    if sha256_file(manifest_path) != ensemble["metadata"]["training_manifest_sha256"]:
        raise LTSNContractError("qualification manifest differs from the trained ensemble")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("ensemble_manifest_sha256") != sha256_file(ensemble_manifest):
        raise LTSNContractError("calibration belongs to a different ensemble")
    variance_scale = np.asarray(calibration["variance_scale"], dtype=float)
    if variance_scale.shape != (18,) or not np.isfinite(variance_scale).all():
        raise LTSNContractError("calibration variance_scale must contain 18 finite values")
    records = read_ltsn_manifest(manifest_path, contract)
    prediction = _ensemble_predictions(
        models, _loader(records, "qualification", batch_size), device
    )
    overall = _metrics(prediction, variance_scale)
    by_step: dict[str, Any] = {}
    for step in sorted(np.unique(prediction["step_number"])):
        mask = prediction["step_number"] == step
        subset = {
            name: value
            if name == "active_coordinate_mask"
            else value[:, mask]
            if name in {"member_coordinate_mean", "member_logvar"}
            else value[mask]
            if isinstance(value, np.ndarray)
            else [item for item, keep in zip(value, mask, strict=True) if keep]
            for name, value in prediction.items()
        }
        by_step[str(int(step))] = _metrics(subset, variance_scale)
    matched_ood = _matched_step_ood_summary(by_step)
    overall["ood_step_matched"] = matched_ood
    gates = _static_gates(overall)
    gates["ood_noop_reliability"] = bool(
        matched_ood["required_steps_present"]
        and isinstance(matched_ood["worst_step_auroc"], (int, float))
        and matched_ood["worst_step_auroc"] >= 0.80
    )
    gates["all_step_strata_reported"] = {4, 5, 6}.issubset(
        {int(value) for value in by_step}
    ) and any(record.is_final for record in records if record.split == "qualification")
    direction_payload = None
    if guidance_development_report is not None:
        direction_payload = json.loads(guidance_development_report.read_text(encoding="utf-8"))
        if direction_payload.get("mode") != "development":
            raise LTSNContractError("qualification requires a development guidance report")
        if direction_payload.get("authorization_scope") != "development_only":
            raise LTSNContractError("qualification requires authorization_scope=development_only")
        if (
            direction_payload.get("schema_version") != 2
            or direction_payload.get("gate") != GUIDANCE_PROMOTION_GATE_NAME
        ):
            raise LTSNContractError(
                "qualification requires the V2 latent-guidance promotion contract"
            )
        if direction_payload.get("fingerprint_json_sha256") != contract.artifact_sha256:
            raise LTSNContractError("development guidance report uses a different exact scorer")
        if direction_payload.get("guidance_promotion_eligible") is not False:
            raise LTSNContractError(
                "development guidance evidence must not claim promotion eligibility"
            )
        gates["decoded_exact_direction_agreement"] = bool(
            direction_payload.get("proxy_exact_direction_gate_passed")
        )
    else:
        gates["decoded_exact_direction_agreement"] = False
    eligible = bool(ensemble["qualification_eligible"] and calibration["qualification_eligible"])
    passed = eligible and all(gates.values())
    payload = {
        "schema_version": 2,
        "status": "passed" if passed else ("engineering_smoke_only" if not eligible else "failed"),
        "qualification_passed": passed,
        "qualification_eligible": eligible,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest),
        "calibration_sha256": sha256_file(calibration_path),
        "qualification_manifest_sha256": sha256_file(manifest_path),
        "metrics": overall,
        "metrics_by_step": by_step,
        "gates": gates,
        "guidance_development_report_sha256": (
            "" if guidance_development_report is None else sha256_file(guidance_development_report)
        ),
        "safety_thresholds": {
            name: calibration[name]
            for name in (
                "ood_probability_threshold",
                "max_aleatoric_variance",
                "max_epistemic_variance",
                "max_interval_width",
            )
        },
        "variance_scale": calibration["variance_scale"],
    }
    write_json_atomic(output_path, payload)
    payload["qualification_sha256"] = sha256_file(output_path)
    return payload


def _cluster_bootstrap_interval(
    values: np.ndarray,
    prompt_ids: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    prompts = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)
    medians = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.choice(prompts, len(prompts), replace=True)
        sample = np.concatenate([values[prompt_ids == prompt] for prompt in selected])
        medians[index] = np.median(sample)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def _optional_noninferiority_flag(
    rows: Sequence[Mapping[str, str]], name: str
) -> tuple[bool | None, bool]:
    values = {(row.get(name) or "").strip().lower() for row in rows}
    if values <= {"", "not_evaluated"}:
        return None, False
    if values == {"true"}:
        return True, True
    if values == {"false"}:
        return False, True
    raise LTSNContractError(f"guidance pair table has inconsistent {name} values")


def evaluate_guidance_pairs(
    *,
    pair_table: Path,
    output_path: Path,
    fingerprint_sha256: str,
    mode: str,
    qualification_report: Path | None = None,
    bootstrap_resamples: int = 2000,
    seed: int = 20260716,
    minimum_optimized_pairs: int = 1,
    minimum_optimized_prompts: int = 1,
) -> dict[str, Any]:
    """Verify surrogate changes using decoded exact scores and non-inferiority flags."""

    if mode not in {"development", "confirmation"}:
        raise ValueError("mode must be development or confirmation")
    if minimum_optimized_pairs < 1 or minimum_optimized_prompts < 1:
        raise ValueError("minimum optimization coverage must be positive")
    expected_scope = "development_only" if mode == "development" else "qualified_confirmation"
    qualification_sha256 = ""
    qualification_passed = False
    if mode == "confirmation":
        if qualification_report is None:
            raise LTSNContractError(
                "confirmation requires a passed independent qualification report"
            )
        qualification = json.loads(qualification_report.read_text(encoding="utf-8"))
        if qualification.get("fingerprint_json_sha256") != fingerprint_sha256:
            raise LTSNContractError("qualification report uses a different exact scorer")
        qualification_passed = qualification.get("qualification_passed") is True
        if not qualification_passed:
            raise LTSNContractError("independent LTSN qualification has not passed")
        qualification_sha256 = sha256_file(qualification_report)
    with pair_table.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError("guidance pair table is empty")
    if any(row.get("authorization_scope") != expected_scope for row in rows):
        raise LTSNContractError(
            f"{mode} guidance pairs require authorization_scope={expected_scope}"
        )
    if any(row.get("fingerprint_json_sha256") != fingerprint_sha256 for row in rows):
        raise LTSNContractError("guidance pair table uses a different exact scorer")
    prompt_ids = np.asarray([row["prompt_id"] for row in rows])
    exact_before = np.asarray([float(row["exact_focus_band_loss_before"]) for row in rows])
    exact_after = np.asarray([float(row["exact_focus_band_loss_after"]) for row in rows])
    proxy_before = np.asarray([float(row["proxy_focus_band_loss_before"]) for row in rows])
    proxy_after = np.asarray([float(row["proxy_focus_band_loss_after"]) for row in rows])
    combined = np.concatenate((exact_before, exact_after, proxy_before, proxy_after))
    if not np.isfinite(combined).all():
        raise LTSNContractError("guidance score table contains NaN or Inf")
    exact_improvement = exact_before - exact_after
    proxy_improvement = proxy_before - proxy_after
    optimized = proxy_improvement > 0
    optimized_prompt_count = len(np.unique(prompt_ids[optimized]))
    optimized_exact = exact_improvement[optimized]
    improved = int(np.count_nonzero(optimized_exact > 0))
    tied = int(np.count_nonzero(optimized_exact == 0))
    worsened = int(np.count_nonzero(optimized_exact < 0))
    direction_agreement = (
        0.0 if not len(optimized_exact) else float(improved / len(optimized_exact))
    )
    optimized_exact_median = 0.0 if not len(optimized_exact) else float(np.median(optimized_exact))
    optimized_exact_mean = 0.0 if not len(optimized_exact) else float(np.mean(optimized_exact))
    optimized_correlation = (
        None
        if len(optimized_exact) < 2
        else spearman_correlation(proxy_improvement[optimized], optimized_exact)
    )
    ci_low, ci_high = _cluster_bootstrap_interval(
        exact_improvement, prompt_ids, resamples=bootstrap_resamples, seed=seed
    )
    quality_noninferior, blind_quality_evidence_available = _optional_noninferiority_flag(
        rows, "quality_noninferior"
    )
    prompt_noninferior, prompt_evidence_available = _optional_noninferiority_flag(
        rows, "prompt_noninferior"
    )
    diversity_preserved, diversity_evidence_available = _optional_noninferiority_flag(
        rows, "diversity_preserved"
    )
    latent_changed_values = {(row.get("latent_changed") or "").strip().lower() for row in rows}
    if latent_changed_values == {""}:
        latent_identity_evidence_available = False
    elif latent_changed_values <= {"true", "false"}:
        latent_identity_evidence_available = True
    else:
        raise LTSNContractError("guidance pair table has malformed latent_changed values")
    latent_changed_pairs = (
        sum(row.get("latent_changed", "").lower() == "true" for row in rows)
        if latent_identity_evidence_available
        else None
    )
    if not prompt_evidence_available:
        raise LTSNContractError("prompt non-inferiority evidence is required")
    technical_quality_eligible = all(
        row.get("guided_technical_quality_eligible", "").lower() == "true" for row in rows
    )
    proxy_exact_passed = (
        direction_agreement >= 0.65
        and optimized_exact_median > 0
        and int(np.count_nonzero(optimized)) >= minimum_optimized_pairs
        and optimized_prompt_count >= minimum_optimized_prompts
        and prompt_noninferior is True
        and technical_quality_eligible
    )
    if mode == "confirmation":
        if not diversity_evidence_available:
            raise LTSNContractError("confirmation requires diversity evidence")
        proxy_exact_passed = (
            proxy_exact_passed and diversity_preserved is True and qualification_passed
        )
    guidance_promotion_eligible = mode == "confirmation" and proxy_exact_passed
    payload = {
        "schema_version": 2,
        "gate": GUIDANCE_PROMOTION_GATE_NAME,
        "mode": mode,
        "authorization_scope": expected_scope,
        "fingerprint_json_sha256": fingerprint_sha256,
        "pair_table_sha256": sha256_file(pair_table),
        "pairs": len(rows),
        "prompts": len(np.unique(prompt_ids)),
        "median_exact_loss_improvement": float(np.median(exact_improvement)),
        "cluster_bootstrap_ci95": [ci_low, ci_high],
        "proxy_optimized_pairs": int(np.count_nonzero(optimized)),
        "proxy_optimized_prompts": optimized_prompt_count,
        "minimum_optimized_pairs": minimum_optimized_pairs,
        "minimum_optimized_prompts": minimum_optimized_prompts,
        "optimization_coverage_gate_passed": bool(
            int(np.count_nonzero(optimized)) >= minimum_optimized_pairs
            and optimized_prompt_count >= minimum_optimized_prompts
        ),
        "proxy_exact_direction_agreement": direction_agreement,
        "optimized_exact_improved_pairs": improved,
        "optimized_exact_tied_pairs": tied,
        "optimized_exact_worsened_pairs": worsened,
        "optimized_exact_median_improvement": optimized_exact_median,
        "optimized_exact_mean_improvement": optimized_exact_mean,
        "optimized_proxy_exact_spearman": optimized_correlation,
        "exact_zero_loss_before_pairs": int(np.count_nonzero(exact_before == 0)),
        "exact_zero_loss_after_pairs": int(np.count_nonzero(exact_after == 0)),
        "optimized_exact_zero_loss_before_pairs": int(
            np.count_nonzero(exact_before[optimized] == 0)
        ),
        "optimized_exact_zero_loss_after_pairs": int(np.count_nonzero(exact_after[optimized] == 0)),
        "latent_identity_evidence_available": latent_identity_evidence_available,
        "latent_changed_pairs": latent_changed_pairs,
        "latent_noop_pairs": (
            None if latent_changed_pairs is None else len(rows) - latent_changed_pairs
        ),
        "blind_quality_is_gate": False,
        "blind_quality_evidence_available": blind_quality_evidence_available,
        "quality_noninferior": quality_noninferior,
        "prompt_noninferior": prompt_noninferior,
        "diversity_preserved": diversity_preserved,
        "all_guided_technical_quality_eligible": technical_quality_eligible,
        "proxy_exact_direction_gate_passed": proxy_exact_passed,
        "qualification_report_sha256": qualification_sha256,
        "guidance_promotion_eligible": guidance_promotion_eligible,
        "status": "passed" if proxy_exact_passed else "failed",
    }
    write_json_atomic(output_path, payload)
    payload["report_sha256"] = sha256_file(output_path)
    return payload
