#!/usr/bin/env python3
"""Build a synthetic, reference-only gate-passing counterpart to V5 outputs.

The generated artifacts exercise the frozen gate calculations but are explicitly
barred from production qualification and promotion. They are useful for comparing
the observed V5 values with a structurally realistic passing example.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_FLAGS: dict[str, Any] = {
    "reference_only": True,
    "synthetic": True,
    "scientific_evidence": False,
    "production_authorization": False,
}
ACTIVE_COORDINATE_MASK = np.asarray([False, False, False] + [True] * 15)
INTERVAL_HALF_WIDTH = 0.02
INTERVAL_VARIANCE = (INTERVAL_HALF_WIDTH / 1.6448536269514722) ** 2
OOD_CALIBRATION_REQUIRED_STEPS = (4, 5, 6, 8)
OOD_CORRECTION_STEPS = (4, 5, 6)
OOD_ID_ACCEPTANCE_MINIMUM = 0.95
OOD_AUROC_MINIMUM = 0.80
OOD_SENSITIVITY_MINIMUM = 0.80


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels >= 0.5]
    negative = scores[labels < 0.5]
    if not len(positive) or not len(negative):
        return None
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return 0.0
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _ood_calibration_summary(
    labels: np.ndarray, probabilities: np.ndarray, steps: np.ndarray
) -> dict[str, Any]:
    id_quantiles = {
        step: float(np.quantile(probabilities[(steps == step) & (labels < 0.5)], 0.95))
        for step in OOD_CORRECTION_STEPS
    }
    threshold = max(id_quantiles.values())
    by_step: dict[str, dict[str, Any]] = {}
    predicted_ood = probabilities > threshold
    for step in OOD_CALIBRATION_REQUIRED_STEPS:
        step_mask = steps == step
        step_id = step_mask & (labels < 0.5)
        step_ood = step_mask & (labels >= 0.5)
        acceptance = float(np.mean(~predicted_ood[step_id]))
        sensitivity = float(np.mean(predicted_ood[step_ood]))
        by_step[str(step)] = {
            "id_samples": int(np.count_nonzero(step_id)),
            "ood_samples": int(np.count_nonzero(step_ood)),
            "id_probability_q95": float(np.quantile(probabilities[step_id], 0.95)),
            "id_acceptance_rate": acceptance,
            "ood_sensitivity": sensitivity,
            "balanced_accuracy": 0.5 * (acceptance + sensitivity),
            "ood_auroc": _auc(labels[step_mask], probabilities[step_mask]),
        }
    gate = {
        "required_steps_present": all(
            by_step[str(step)]["id_samples"] > 0 and by_step[str(step)]["ood_samples"] > 0
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
        "correction_step_id_acceptance": all(
            by_step[str(step)]["id_acceptance_rate"] >= OOD_ID_ACCEPTANCE_MINIMUM
            for step in OOD_CORRECTION_STEPS
        ),
        "matched_step_ood_auroc": all(
            by_step[str(step)]["ood_auroc"] >= OOD_AUROC_MINIMUM
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
        "matched_step_ood_sensitivity": all(
            by_step[str(step)]["ood_sensitivity"] >= OOD_SENSITIVITY_MINIMUM
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
    }
    in_distribution = labels < 0.5
    out_of_distribution = labels >= 0.5
    global_sensitivity = float(np.mean(predicted_ood[out_of_distribution]))
    global_specificity = float(np.mean(~predicted_ood[in_distribution]))
    return {
        "threshold": threshold,
        "by_step": by_step,
        "gate": gate,
        "passed": all(gate.values()),
        "global_balanced_accuracy": 0.5 * (global_sensitivity + global_specificity),
        "policy": {
            "threshold": "maximum correction-step calibration-ID OOD probability q95",
            "required_steps": list(OOD_CALIBRATION_REQUIRED_STEPS),
            "correction_steps": list(OOD_CORRECTION_STEPS),
            "minimum_id_acceptance": OOD_ID_ACCEPTANCE_MINIMUM,
            "minimum_matched_step_ood_auroc": OOD_AUROC_MINIMUM,
            "minimum_matched_step_ood_sensitivity": OOD_SENSITIVITY_MINIMUM,
        },
    }


def _quartile_ranking_accuracy(exact: np.ndarray, predicted: np.ndarray) -> float:
    low = np.flatnonzero(exact <= np.quantile(exact, 0.25))
    high = np.flatnonzero(exact >= np.quantile(exact, 0.75))
    if not len(low) or not len(high):
        return 0.0
    return float(np.mean(predicted[high, None] > predicted[None, low]))


def _metrics(prediction: dict[str, Any], variance_scale: np.ndarray) -> dict[str, Any]:
    in_distribution = prediction["ood_label"] < 0.5
    exact = prediction["coordinates"][in_distribution]
    mean = prediction["coordinate_mean"][in_distribution]
    coordinate_rhos = [_spearman(mean[:, index], exact[:, index]) for index in range(18)]
    active_mask = prediction["active_coordinate_mask"]
    score_rho = _spearman(
        prediction["predicted_focus_logit"][in_distribution],
        prediction["focus_logit"][in_distribution],
    )
    pitch_rho = _spearman(
        np.linalg.norm(mean[:, :16], axis=1), np.linalg.norm(exact[:, :16], axis=1)
    )
    phase_rho = _spearman(
        np.linalg.norm(mean[:, 16:], axis=1), np.linalg.norm(exact[:, 16:], axis=1)
    )
    variance = prediction["total_variance"][in_distribution] * variance_scale[None, :]
    half_width = 1.6448536269514722 * np.sqrt(np.maximum(variance, 0.0))
    covered = (exact >= mean - half_width) & (exact <= mean + half_width)
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
        "interval_90_coverage": float(np.mean(covered[:, active_mask])),
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
        "ood_auroc": _auc(prediction["ood_label"], prediction["ood_probability"]),
    }


def _static_gates(metrics: dict[str, Any]) -> dict[str, bool]:
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


def _matched_step_ood_summary(
    metrics_by_step: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    values = {
        str(step): float(metrics["ood_auroc"])
        for step, metrics in metrics_by_step.items()
        if isinstance(metrics.get("ood_auroc"), (int, float))
        and math.isfinite(float(metrics["ood_auroc"]))
    }
    return {
        "by_step": values,
        "macro_auroc": float(np.mean(list(values.values()))),
        "worst_step_auroc": min(values.values()),
        "required_steps_present": set(OOD_CALIBRATION_REQUIRED_STEPS).issubset(
            {int(step) for step in values}
        ),
    }


def _cluster_bootstrap_interval(
    values: np.ndarray, prompt_ids: np.ndarray, resamples: int, seed: int
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_compact(values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _build_calibration_reference(
    manifest_rows: list[dict[str, str]], output_dir: Path, source_manifest: Path
) -> dict[str, Any]:
    rows = [row for row in manifest_rows if row["split"] == "calibration"]
    if not rows:
        raise ValueError("V5 manifest contains no calibration rows")
    table_rows: list[dict[str, Any]] = []
    labels: list[float] = []
    probabilities: list[float] = []
    steps: list[int] = []
    for row in rows:
        label = float(row["ood_label"])
        probability = 0.90 if label >= 0.5 else 0.10
        step = int(row["step_number"])
        labels.append(label)
        probabilities.append(probability)
        steps.append(step)
        table_rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt_id": row["prompt_id"],
                "step_number": step,
                "ood_label": label,
                "ood_probability": probability,
                "reference_only": "true",
                "synthetic": "true",
            }
        )
    table_path = output_dir / "calibration_predictions_reference.csv"
    _write_csv(table_path, list(table_rows[0]), table_rows)
    summary = _ood_calibration_summary(
        np.asarray(labels), np.asarray(probabilities), np.asarray(steps)
    )
    payload = {
        "schema_version": 3,
        "status": "counterfactual_pass",
        "qualification_eligible": True,
        **REFERENCE_FLAGS,
        "source_v5_manifest_sha256": _sha256(source_manifest),
        "prediction_table": table_path.name,
        "prediction_table_sha256": _sha256(table_path),
        "sample_count": len(rows),
        "ood_probability_threshold": summary["threshold"],
        "ood_calibration_balanced_accuracy": summary["global_balanced_accuracy"],
        "ood_calibration_by_step": summary["by_step"],
        "ood_calibration_gate": summary["gate"],
        "ood_calibration_gate_passed": summary["passed"],
        "ood_policy": summary["policy"],
    }
    _write_json_atomic(output_dir / "calibration_reference.json", payload)
    return payload


def _build_development_reference(
    source_pairs: Path, output_dir: Path, fingerprint_sha256: str
) -> dict[str, Any]:
    source_rows = _read_csv(source_pairs)
    if not source_rows:
        raise ValueError("V5 development pair table is empty")
    reference_rows: list[dict[str, Any]] = []
    for source in source_rows:
        row: dict[str, Any] = dict(source)
        exact_before = float(source["exact_focus_band_loss_before"])
        proxy_before = float(source["proxy_focus_band_loss_before"])
        optimized = exact_before > 0.0 and proxy_before > 0.0
        row.update(
            {
                "source_exact_focus_band_loss_after": source["exact_focus_band_loss_after"],
                "source_proxy_focus_band_loss_after": source["proxy_focus_band_loss_after"],
                "source_prompt_noninferior": source["prompt_noninferior"],
                "source_diversity_preserved": source["diversity_preserved"],
                "exact_focus_band_loss_after": exact_before * 0.80 if optimized else exact_before,
                "proxy_focus_band_loss_after": proxy_before * 0.90 if optimized else proxy_before,
                "latent_changed": "true" if optimized else "false",
                "authorization_scope": "development_only",
                "ood_ablation_only": "false",
                "quality_noninferior": "not_evaluated",
                "prompt_noninferior": "true",
                "diversity_preserved": "true",
                "guided_technical_quality_eligible": "true",
                "reference_only": "true",
                "synthetic": "true",
            }
        )
        reference_rows.append(row)
    pair_path = output_dir / "development_pairs_reference.csv"
    fieldnames = list(source_rows[0]) + [
        "source_exact_focus_band_loss_after",
        "source_proxy_focus_band_loss_after",
        "source_prompt_noninferior",
        "source_diversity_preserved",
        "reference_only",
        "synthetic",
    ]
    _write_csv(pair_path, fieldnames, reference_rows)
    prompt_ids = np.asarray([row["prompt_id"] for row in reference_rows])
    exact_before = np.asarray(
        [float(row["exact_focus_band_loss_before"]) for row in reference_rows]
    )
    exact_after = np.asarray([float(row["exact_focus_band_loss_after"]) for row in reference_rows])
    proxy_before = np.asarray(
        [float(row["proxy_focus_band_loss_before"]) for row in reference_rows]
    )
    proxy_after = np.asarray([float(row["proxy_focus_band_loss_after"]) for row in reference_rows])
    exact_improvement = exact_before - exact_after
    proxy_improvement = proxy_before - proxy_after
    optimized = proxy_improvement > 0
    optimized_exact = exact_improvement[optimized]
    optimized_prompts = len(np.unique(prompt_ids[optimized]))
    ci_low, ci_high = _cluster_bootstrap_interval(exact_improvement, prompt_ids, 2000, 20260716)
    direction_agreement = float(np.mean(optimized_exact > 0))
    report = {
        "schema_version": 2,
        "gate": "latent_guidance_promotion_v2",
        "mode": "development",
        "authorization_scope": "development_only",
        "ood_ablation_only": False,
        "fingerprint_json_sha256": fingerprint_sha256,
        "pair_table_sha256": _sha256(pair_path),
        "pairs": len(reference_rows),
        "prompts": len(np.unique(prompt_ids)),
        "median_exact_loss_improvement": float(np.median(exact_improvement)),
        "cluster_bootstrap_ci95": [ci_low, ci_high],
        "proxy_optimized_pairs": int(np.count_nonzero(optimized)),
        "proxy_optimized_prompts": optimized_prompts,
        "minimum_optimized_pairs": 64,
        "minimum_optimized_prompts": 16,
        "optimization_coverage_gate_passed": bool(
            np.count_nonzero(optimized) >= 64 and optimized_prompts >= 16
        ),
        "proxy_exact_direction_agreement": direction_agreement,
        "optimized_exact_improved_pairs": int(np.count_nonzero(optimized_exact > 0)),
        "optimized_exact_tied_pairs": int(np.count_nonzero(optimized_exact == 0)),
        "optimized_exact_worsened_pairs": int(np.count_nonzero(optimized_exact < 0)),
        "optimized_exact_median_improvement": float(np.median(optimized_exact)),
        "optimized_exact_mean_improvement": float(np.mean(optimized_exact)),
        "optimized_proxy_exact_spearman": _spearman(proxy_improvement[optimized], optimized_exact),
        "exact_zero_loss_before_pairs": int(np.count_nonzero(exact_before == 0)),
        "exact_zero_loss_after_pairs": int(np.count_nonzero(exact_after == 0)),
        "optimized_exact_zero_loss_before_pairs": int(
            np.count_nonzero(exact_before[optimized] == 0)
        ),
        "optimized_exact_zero_loss_after_pairs": int(np.count_nonzero(exact_after[optimized] == 0)),
        "latent_identity_evidence_available": True,
        "latent_changed_pairs": int(np.count_nonzero(optimized)),
        "latent_noop_pairs": int(np.count_nonzero(~optimized)),
        "blind_quality_is_gate": False,
        "blind_quality_evidence_available": False,
        "quality_noninferior": None,
        "prompt_noninferior": True,
        "diversity_preserved": True,
        "all_guided_technical_quality_eligible": True,
        "proxy_exact_direction_gate_passed": bool(
            direction_agreement >= 0.65
            and np.median(optimized_exact) > 0
            and np.count_nonzero(optimized) >= 64
            and optimized_prompts >= 16
        ),
        "qualification_report_sha256": "",
        "guidance_promotion_eligible": False,
        "status": "counterfactual_pass",
        **REFERENCE_FLAGS,
        "source_v5_pair_table_sha256": _sha256(source_pairs),
    }
    report_path = output_dir / "guidance_development_reference.json"
    _write_json_atomic(report_path, report)
    return report


def _prediction_payload(
    exact: np.ndarray,
    predicted: np.ndarray,
    exact_focus: np.ndarray,
    predicted_focus: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    variances: np.ndarray,
) -> dict[str, Any]:
    return {
        "coordinates": exact,
        "coordinate_mean": predicted,
        "focus_logit": exact_focus,
        "predicted_focus_logit": predicted_focus,
        "ood_label": labels,
        "ood_probability": probabilities,
        "total_variance": variances,
        "active_coordinate_mask": ACTIVE_COORDINATE_MASK,
    }


def _build_qualification_reference(
    manifest_rows: list[dict[str, str]],
    output_dir: Path,
    source_manifest: Path,
    development_report: dict[str, Any],
) -> dict[str, Any]:
    rows = [row for row in manifest_rows if row["split"] == "qualification"]
    if not rows:
        raise ValueError("V5 manifest contains no qualification rows")
    table_rows: list[dict[str, Any]] = []
    exact_rows: list[list[float]] = []
    predicted_rows: list[list[float]] = []
    focus_rows: list[float] = []
    predicted_focus_rows: list[float] = []
    labels: list[float] = []
    probabilities: list[float] = []
    variance_rows: list[list[float]] = []
    steps: list[int] = []
    id_row_index = 0
    active_count = int(np.count_nonzero(ACTIVE_COORDINATE_MASK))
    for row in rows:
        exact = np.asarray(json.loads(row["coordinates_json"]), dtype=float)
        if exact.shape != (18,):
            raise ValueError(f"invalid 18-D coordinates for {row['sample_id']}")
        label = float(row["ood_label"])
        predicted = exact.copy()
        if label < 0.5:
            active_index = 0
            for coordinate_index, active in enumerate(ACTIVE_COORDINATE_MASK):
                if not active:
                    continue
                linear_index = id_row_index * active_count + active_index
                magnitude = 0.03 if linear_index % 10 == 0 else 0.005
                sign = -1.0 if (id_row_index + coordinate_index) % 2 else 1.0
                predicted[coordinate_index] += sign * magnitude
                active_index += 1
            id_row_index += 1
        exact_focus = float(row["focus_logit"])
        predicted_focus = exact_focus + (0.001 if len(focus_rows) % 2 else -0.001)
        probability = 0.90 if label >= 0.5 else 0.10
        variance = np.full(18, INTERVAL_VARIANCE, dtype=float)
        exact_rows.append(exact.tolist())
        predicted_rows.append(predicted.tolist())
        focus_rows.append(exact_focus)
        predicted_focus_rows.append(predicted_focus)
        labels.append(label)
        probabilities.append(probability)
        variance_rows.append(variance.tolist())
        steps.append(int(row["step_number"]))
        table_rows.append(
            {
                "sample_id": row["sample_id"],
                "prompt_id": row["prompt_id"],
                "step_number": row["step_number"],
                "ood_label": label,
                "exact_coordinates_json": _json_compact(exact.tolist()),
                "predicted_coordinates_json": _json_compact(predicted.tolist()),
                "exact_focus_logit": exact_focus,
                "predicted_focus_logit": predicted_focus,
                "total_variance_json": _json_compact(variance.tolist()),
                "ood_probability": probability,
                "active_coordinate_mask_json": _json_compact(ACTIVE_COORDINATE_MASK.tolist()),
                "reference_only": "true",
                "synthetic": "true",
            }
        )
    table_path = output_dir / "qualification_predictions_reference.csv"
    _write_csv(table_path, list(table_rows[0]), table_rows)
    exact_array = np.asarray(exact_rows)
    predicted_array = np.asarray(predicted_rows)
    focus_array = np.asarray(focus_rows)
    predicted_focus_array = np.asarray(predicted_focus_rows)
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    variance_array = np.asarray(variance_rows)
    steps_array = np.asarray(steps)
    prediction = _prediction_payload(
        exact_array,
        predicted_array,
        focus_array,
        predicted_focus_array,
        labels_array,
        probabilities_array,
        variance_array,
    )
    overall = _metrics(prediction, np.ones(18, dtype=float))
    by_step: dict[str, Any] = {}
    for step in sorted(np.unique(steps_array)):
        mask = steps_array == step
        subset = _prediction_payload(
            exact_array[mask],
            predicted_array[mask],
            focus_array[mask],
            predicted_focus_array[mask],
            labels_array[mask],
            probabilities_array[mask],
            variance_array[mask],
        )
        by_step[str(int(step))] = _metrics(subset, np.ones(18, dtype=float))
    matched_ood = _matched_step_ood_summary(by_step)
    overall["ood_step_matched"] = matched_ood
    gates = _static_gates(overall)
    gates["ood_noop_reliability"] = bool(
        matched_ood["required_steps_present"]
        and isinstance(matched_ood["worst_step_auroc"], (int, float))
        and matched_ood["worst_step_auroc"] >= OOD_AUROC_MINIMUM
    )
    gates["all_step_strata_reported"] = set(OOD_CALIBRATION_REQUIRED_STEPS).issubset(
        {int(step) for step in by_step}
    )
    gates["decoded_exact_direction_agreement"] = bool(
        development_report["proxy_exact_direction_gate_passed"]
    )
    passed = all(gates.values())
    payload = {
        "schema_version": 2,
        "status": "counterfactual_pass" if passed else "counterfactual_fail",
        "qualification_passed": passed,
        "qualification_eligible": True,
        **REFERENCE_FLAGS,
        "fingerprint_json_sha256": rows[0]["fingerprint_json_sha256"],
        "source_v5_manifest_sha256": _sha256(source_manifest),
        "prediction_table": table_path.name,
        "prediction_table_sha256": _sha256(table_path),
        "metrics": overall,
        "metrics_by_step": by_step,
        "gates": gates,
    }
    _write_json_atomic(output_dir / "qualification_reference.json", payload)
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"generated qualification reference failed gates: {failed}")
    return payload


def _rule_pass(value: Any, rule: str) -> bool:
    if rule.startswith(">="):
        return float(value) >= float(rule[2:])
    if rule.startswith(">"):
        return float(value) > float(rule[1:])
    if rule.startswith("["):
        lower, upper = (float(item) for item in rule.strip("[]").split(","))
        return lower <= float(value) <= upper
    if rule == "true":
        return value is True
    raise ValueError(f"unsupported rule: {rule}")


def _build_comparison(
    source_calibration: dict[str, Any],
    calibration_reference: dict[str, Any],
    source_development: dict[str, Any],
    development_reference: dict[str, Any],
    source_qualification: dict[str, Any],
    qualification_reference: dict[str, Any],
    output_dir: Path,
) -> None:
    source_calibration_steps = source_calibration["ood_calibration_by_step"]
    reference_calibration_steps = calibration_reference["ood_calibration_by_step"]
    rows: list[dict[str, Any]] = []

    def add(stage: str, metric: str, rule: str, source: Any, reference: Any) -> None:
        rows.append(
            {
                "stage": stage,
                "metric": metric,
                "gate_rule": rule,
                "source_v5": source,
                "passing_reference": reference,
                "source_passed": str(_rule_pass(source, rule)).lower(),
                "reference_passed": str(_rule_pass(reference, rule)).lower(),
            }
        )

    add(
        "calibration",
        "minimum_correction_step_id_acceptance",
        f">={OOD_ID_ACCEPTANCE_MINIMUM}",
        min(
            source_calibration_steps[str(step)]["id_acceptance_rate"]
            for step in OOD_CORRECTION_STEPS
        ),
        min(
            reference_calibration_steps[str(step)]["id_acceptance_rate"]
            for step in OOD_CORRECTION_STEPS
        ),
    )
    add(
        "calibration",
        "minimum_matched_step_ood_auroc",
        f">={OOD_AUROC_MINIMUM}",
        min(
            source_calibration_steps[str(step)]["ood_auroc"]
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
        min(
            reference_calibration_steps[str(step)]["ood_auroc"]
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
    )
    add(
        "calibration",
        "minimum_matched_step_ood_sensitivity",
        f">={OOD_SENSITIVITY_MINIMUM}",
        min(
            source_calibration_steps[str(step)]["ood_sensitivity"]
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
        min(
            reference_calibration_steps[str(step)]["ood_sensitivity"]
            for step in OOD_CALIBRATION_REQUIRED_STEPS
        ),
    )
    add(
        "development",
        "proxy_optimized_pairs",
        f">={development_reference['minimum_optimized_pairs']}",
        source_development["proxy_optimized_pairs"],
        development_reference["proxy_optimized_pairs"],
    )
    add(
        "development",
        "proxy_optimized_prompts",
        f">={development_reference['minimum_optimized_prompts']}",
        source_development["proxy_optimized_prompts"],
        development_reference["proxy_optimized_prompts"],
    )
    add(
        "development",
        "proxy_exact_direction_agreement",
        ">=0.65",
        source_development["proxy_exact_direction_agreement"],
        development_reference["proxy_exact_direction_agreement"],
    )
    add(
        "development",
        "optimized_exact_median_improvement",
        ">0",
        source_development["optimized_exact_median_improvement"],
        development_reference["optimized_exact_median_improvement"],
    )
    source_metrics = source_qualification["metrics"]
    reference_metrics = qualification_reference["metrics"]
    qualification_rules = {
        "focus_logit_spearman": ">=0.70",
        "coordinate_median_spearman": ">=0.50",
        "pitch_block_distance_spearman": ">=0.50",
        "phase_block_distance_spearman": ">=0.50",
        "acoustic_loop_coordinate_spearman": ">=0.50",
        "chroma_loop_coordinate_spearman": ">=0.50",
        "quartile_ranking_accuracy": ">=0.65",
        "interval_90_coverage": "[0.85,0.95]",
        "ood_auroc": ">=0.80",
    }
    for metric, rule in qualification_rules.items():
        add(
            "qualification",
            metric,
            rule,
            source_metrics[metric],
            reference_metrics[metric],
        )
    _write_csv(output_dir / "v5_vs_passing_reference.csv", list(rows[0]), rows)


def _write_readme(output_dir: Path) -> None:
    content = """# LTSN V5 gate-passing comparison reference

This directory is a deterministic **synthetic counterfactual**, built from the row
identity, split structure, exact labels, and schemas of the checked-in V5 outputs.
It is not a model result, qualification result, or scientific result.

The reference tables are designed to pass the frozen calculations without changing
their thresholds:

- calibration uses perfectly separated synthetic ID/OOD probabilities;
- qualification predictions stay close to V5 exact labels, with exactly 90% active-
  coordinate interval coverage and step-matched OOD separation;
- development pairs preserve V5 identities but make the eligible nonzero-loss subset
  improve in both proxy and decoded exact loss while preserving prompt/diversity flags.

Every JSON report contains `reference_only=true`, `synthetic=true`,
`scientific_evidence=false`, and `production_authorization=false`. Production
qualification and confirmation code rejects these markers. Use
`v5_vs_passing_reference.csv` for the direct comparison.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8", newline="\n")


def _write_manifest(
    root: Path, output_dir: Path, source_paths: list[Path], generated_paths: list[Path]
) -> None:
    payload = {
        "schema_version": 1,
        **REFERENCE_FLAGS,
        "generator": "scripts/build_ltsn_v5_gate_passing_reference.py",
        "sources": {path.relative_to(root).as_posix(): _sha256(path) for path in source_paths},
        "generated": {path.name: _sha256(path) for path in generated_paths},
    }
    manifest_path = output_dir / "reference_manifest.json"
    _write_json_atomic(manifest_path, payload)
    checksummed = sorted(generated_paths + [manifest_path], key=lambda path: path.name)
    lines = [f"{_sha256(path)}  {path.name}" for path in checksummed]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def build_reference(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = root / "runs/ltsn_turbo/training_augmentation_v5/ltsn_manifest_v5.csv"
    source_pairs = root / "runs/ltsn_turbo/development_pairs_v5/development_pairs.csv"
    source_calibration_path = root / "runs/ltsn_turbo/calibration_v5.json"
    source_development_path = root / "runs/ltsn_turbo/guidance_development_v5.json"
    source_qualification_path = root / "runs/ltsn_turbo/qualification.json"
    source_paths = [
        source_manifest,
        source_pairs,
        source_calibration_path,
        source_development_path,
        source_qualification_path,
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V5 source artifacts: {missing}")
    manifest_rows = _read_csv(source_manifest)
    source_calibration = json.loads(source_calibration_path.read_text(encoding="utf-8"))
    source_development = json.loads(source_development_path.read_text(encoding="utf-8"))
    source_qualification = json.loads(source_qualification_path.read_text(encoding="utf-8"))
    fingerprint_values = {row["fingerprint_json_sha256"] for row in manifest_rows}
    if len(fingerprint_values) != 1:
        raise ValueError("V5 manifest does not use exactly one scorer fingerprint")
    fingerprint_sha256 = next(iter(fingerprint_values))
    calibration_reference = _build_calibration_reference(manifest_rows, output_dir, source_manifest)
    development_reference = _build_development_reference(
        source_pairs, output_dir, fingerprint_sha256
    )
    qualification_reference = _build_qualification_reference(
        manifest_rows,
        output_dir,
        source_manifest,
        development_reference,
    )
    _build_comparison(
        source_calibration,
        calibration_reference,
        source_development,
        development_reference,
        source_qualification,
        qualification_reference,
        output_dir,
    )
    _write_readme(output_dir)
    generated_paths = [
        output_dir / "README.md",
        output_dir / "calibration_predictions_reference.csv",
        output_dir / "calibration_reference.json",
        output_dir / "development_pairs_reference.csv",
        output_dir / "guidance_development_reference.json",
        output_dir / "qualification_predictions_reference.csv",
        output_dir / "qualification_reference.json",
        output_dir / "v5_vs_passing_reference.csv",
    ]
    _write_manifest(root, output_dir, source_paths, generated_paths)
    return {
        "ok": True,
        "output_dir": str(output_dir),
        "calibration_gate_passed": calibration_reference["ood_calibration_gate_passed"],
        "development_gate_passed": development_reference["proxy_exact_direction_gate_passed"],
        "qualification_gate_passed": qualification_reference["qualification_passed"],
        "reference_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/ltsn_v5_gate_passing_reference"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    result = build_reference(root, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
