"""Exact, non-promotable evaluation for the step-4 LTSN gradient diagnostic."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_json_atomic


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2:
        return None
    left, right = _rank(first), _rank(second)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _cluster_interval(
    values: np.ndarray,
    prompt_ids: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int,
    seed: int,
) -> list[float] | None:
    prompts = np.unique(prompt_ids)
    if not len(values) or not len(prompts):
        return None
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.choice(prompts, len(prompts), replace=True)
        sample = np.concatenate([values[prompt_ids == prompt] for prompt in selected])
        samples[index] = statistic(sample)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def evaluate_step4_gradient_diagnostic(
    *,
    pair_table: Path,
    generation_plan: Path,
    output_path: Path,
    bootstrap_resamples: int = 2000,
    seed: int = 20260716,
) -> dict[str, Any]:
    """Evaluate proxy/exact agreement without issuing any promotion authorization."""

    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    plan = json.loads(generation_plan.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != 3
        or plan.get("experiment_mode") != "step4_gradient_diagnostic"
        or plan.get("diagnostic_only") is not True
        or plan.get("guidance_promotion_eligible") is not False
        or plan.get("ood_ablation_only") is not True
        or plan.get("diagnostic_prompt_limit") != 16
    ):
        raise LTSNContractError("generation plan is not a frozen step-4 diagnostic")
    corrector = plan.get("corrector")
    if not isinstance(corrector, dict) or corrector.get("step_weights") != {"4": 0.5}:
        raise LTSNContractError("step-4 diagnostic must disable correction steps 5 and 6")
    if not math.isclose(float(corrector.get("rms_clip_ratio", -1.0)), 0.005):
        raise LTSNContractError("step-4 diagnostic requires RMS clip ratio 0.005")
    if (
        corrector.get("require_all_members_out_of_band") is not False
        or corrector.get("require_all_member_improvement") is not False
        or not math.isclose(float(corrector.get("minimum_member_gradient_cosine", 0.0)), -1.0)
    ):
        raise LTSNContractError("step-4 diagnostic contains an unexpected consensus filter")

    rows = _read_csv(pair_table)
    prompt_counts: dict[str, int] = {}
    for row in rows:
        prompt_id = row.get("prompt_id", "")
        prompt_counts[prompt_id] = prompt_counts.get(prompt_id, 0) + 1
    if len(rows) != 64 or len(prompt_counts) != 16 or set(prompt_counts.values()) != {4}:
        raise LTSNContractError("step-4 diagnostic requires 16 prompts x 4 paired seeds")
    if any(
        row.get("authorization_scope") != "development_only"
        or row.get("ood_ablation_only", "").lower() != "true"
        or row.get("diagnostic_only", "").lower() != "true"
        for row in rows
    ):
        raise LTSNContractError("step-4 pair table has an invalid diagnostic scope")
    plan_sha256 = sha256_file(generation_plan)
    if {row.get("generation_plan_sha256") for row in rows} != {plan_sha256}:
        raise LTSNContractError("step-4 pair table belongs to a different generation plan")

    prompt_ids = np.asarray([row["prompt_id"] for row in rows])
    proxy_before = np.asarray([float(row["proxy_focus_band_loss_before"]) for row in rows])
    proxy_after = np.asarray([float(row["proxy_focus_band_loss_after"]) for row in rows])
    exact_before = np.asarray([float(row["exact_focus_band_loss_before"]) for row in rows])
    exact_after = np.asarray([float(row["exact_focus_band_loss_after"]) for row in rows])
    values = np.concatenate((proxy_before, proxy_after, exact_before, exact_after))
    if not np.isfinite(values).all():
        raise LTSNContractError("step-4 diagnostic contains NaN or Inf")

    proxy_improvement = proxy_before - proxy_after
    exact_improvement = exact_before - exact_after
    proxy_optimized = proxy_improvement > 1e-12
    both_out_of_band = (proxy_before > 1e-12) & (exact_before > 1e-12)
    selected = proxy_optimized & both_out_of_band
    informative = selected & (np.abs(exact_improvement) > 1e-12)
    selected_exact = exact_improvement[selected]
    selected_prompts = prompt_ids[selected]
    informative_exact = exact_improvement[informative]
    direction_agreement = (
        0.0 if not np.any(informative) else float(np.mean(informative_exact > 0.0))
    )
    median_ci = _cluster_interval(
        selected_exact,
        selected_prompts,
        lambda sample: float(np.median(sample)),
        resamples=bootstrap_resamples,
        seed=seed,
    )
    mean_ci = _cluster_interval(
        selected_exact,
        selected_prompts,
        lambda sample: float(np.mean(sample)),
        resamples=bootstrap_resamples,
        seed=seed + 1,
    )
    criteria = {
        "minimum_proxy_optimized_both_oob_pairs": int(np.count_nonzero(selected)) >= 16,
        "minimum_proxy_optimized_both_oob_prompts": len(np.unique(selected_prompts)) >= 8,
        "non_tied_exact_direction_agreement": direction_agreement >= 0.65,
        "median_exact_improvement_positive": bool(
            len(selected_exact) and float(np.median(selected_exact)) > 0.0
        ),
        "median_exact_improvement_ci_lower_nonnegative": bool(
            median_ci is not None and median_ci[0] >= 0.0
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_step4_gradient_diagnostic_v1",
        "mode": "development_only_diagnostic",
        "diagnostic_only": True,
        "guidance_promotion_eligible": False,
        "status": "signal_supported" if all(criteria.values()) else "signal_not_supported",
        "pair_table_sha256": sha256_file(pair_table),
        "generation_plan_sha256": plan_sha256,
        "pairs": len(rows),
        "prompts": len(prompt_counts),
        "latent_changed_pairs": sum(
            row.get("latent_changed", "").lower() == "true" for row in rows
        ),
        "proxy_optimized_pairs": int(np.count_nonzero(proxy_optimized)),
        "proxy_optimized_prompts": len(np.unique(prompt_ids[proxy_optimized])),
        "proxy_optimized_both_oob_pairs": int(np.count_nonzero(selected)),
        "proxy_optimized_both_oob_prompts": len(np.unique(selected_prompts)),
        "exact_improved_pairs": int(np.count_nonzero(exact_improvement[selected] > 1e-12)),
        "exact_tied_pairs": int(np.count_nonzero(np.abs(exact_improvement[selected]) <= 1e-12)),
        "exact_worsened_pairs": int(np.count_nonzero(exact_improvement[selected] < -1e-12)),
        "non_tied_exact_direction_agreement": direction_agreement,
        "median_exact_improvement": (
            None if not len(selected_exact) else float(np.median(selected_exact))
        ),
        "median_exact_improvement_cluster_ci95": median_ci,
        "mean_exact_improvement": (
            None if not len(selected_exact) else float(np.mean(selected_exact))
        ),
        "mean_exact_improvement_cluster_ci95": mean_ci,
        "proxy_exact_spearman": _spearman(proxy_improvement[selected], exact_improvement[selected]),
        "criteria": criteria,
    }
    write_json_atomic(output_path, payload)
    payload["report_sha256"] = sha256_file(output_path)
    return payload
