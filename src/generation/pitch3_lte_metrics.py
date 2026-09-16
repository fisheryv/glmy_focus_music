"""Frozen LTE model gates, shared by training and Torch-free archived CV reports."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError

LTE_COLLAPSE_PREDICTED_RANGE = 1e-3

LTE_COLLAPSE_EXACT_RANGE = 0.1

LTE_MAX_COLLAPSED_PROMPT_FRACTION = 0.05


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return 0.0
    left, right = _rank(np.asarray(first, dtype=float)), _rank(np.asarray(second, dtype=float))
    if np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def pitch3_lte_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = [row for row in rows if row["source_kind"] == "base_step4_seed"]
    if not base:
        raise LTSNContractError("V3-LTE metrics require base Step-4 rows")
    exact_band = np.asarray([float(row["exact_band"]) for row in base])
    predicted = np.asarray([float(row["predicted_energy"]) for row in base])
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base:
        by_prompt[str(row["prompt_id"])].append(row)
        by_family[str(row["prompt_family"])].append(row)
    collapsed_prompt_ids = []
    for prompt_id, values in by_prompt.items():
        exact_values = [float(row["exact_energy"]) for row in values]
        predicted_values = [float(row["predicted_energy"]) for row in values]
        exact_range = max(exact_values) - min(exact_values)
        predicted_range = max(predicted_values) - min(predicted_values)
        if (
            exact_range > LTE_COLLAPSE_EXACT_RANGE
            and predicted_range < LTE_COLLAPSE_PREDICTED_RANGE
        ):
            collapsed_prompt_ids.append(prompt_id)
    collapsed_fraction = len(collapsed_prompt_ids) / len(by_prompt)
    rank_correct = 0
    rank_total = 0
    for values in by_prompt.values():
        for offset, left in enumerate(values):
            for right in values[offset + 1 :]:
                exact_delta = float(left["exact_band"]) - float(right["exact_band"])
                if abs(exact_delta) <= 1e-6:
                    continue
                predicted_delta = float(left["predicted_energy"]) - float(right["predicted_energy"])
                rank_correct += int(
                    math.copysign(1, exact_delta) == math.copysign(1, predicted_delta)
                )
                rank_total += 1
    family_rho = {
        family: spearman(
            np.asarray([float(row["exact_band"]) for row in values]),
            np.asarray([float(row["predicted_energy"]) for row in values]),
        )
        for family, values in sorted(by_family.items())
    }
    local: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["direction_id"]:
            local[str(row["direction_id"])][int(row["direction_sign"])] = row
    direction_correct = 0
    direction_total = 0
    derivative_exact: list[float] = []
    derivative_predicted: list[float] = []
    for signed in local.values():
        if set(signed) != {-1, 1}:
            continue
        minus, plus = signed[-1], signed[1]
        epsilon = float(minus["epsilon"])
        exact = (float(plus["exact_energy"]) - float(minus["exact_energy"])) / (2 * epsilon)
        estimate = (float(plus["predicted_energy"]) - float(minus["predicted_energy"])) / (
            2 * epsilon
        )
        if abs(exact) <= 1e-6:
            continue
        direction_correct += int(math.copysign(1, exact) == math.copysign(1, estimate))
        direction_total += 1
        derivative_exact.append(exact)
        derivative_predicted.append(estimate)
    pooled = spearman(exact_band, predicted)
    ranking = rank_correct / rank_total if rank_total else 0.0
    direction = direction_correct / direction_total if direction_total else 0.0
    gates = {
        "direct_energy_spearman": pooled >= 0.50,
        "every_prompt_family_spearman": bool(family_rho) and min(family_rho.values()) >= 0.50,
        "same_prompt_ranking_accuracy": rank_total > 0 and ranking >= 0.65,
        "local_direction_sign_accuracy": direction_total > 0 and direction >= 0.65,
        "prompt_latent_sensitivity": collapsed_fraction <= LTE_MAX_COLLAPSED_PROMPT_FRACTION,
    }
    deficits = {
        "direct_energy_spearman": max(0.0, 0.50 - pooled),
        "every_prompt_family_spearman": max(0.0, 0.50 - min(family_rho.values(), default=0.0)),
        "same_prompt_ranking_accuracy": max(0.0, 0.65 - ranking),
        "local_direction_sign_accuracy": max(0.0, 0.65 - direction) if direction_total else 0.65,
        "prompt_latent_sensitivity": max(
            0.0, collapsed_fraction - LTE_MAX_COLLAPSED_PROMPT_FRACTION
        ),
    }
    return {
        "samples": len(rows),
        "base_samples": len(base),
        "direct_energy_spearman": pooled,
        "prompt_family_spearman": family_rho,
        "minimum_prompt_family_spearman": min(family_rho.values(), default=0.0),
        "same_prompt_rank_pairs": rank_total,
        "same_prompt_ranking_accuracy": ranking,
        "local_direction_pairs": direction_total,
        "local_direction_sign_accuracy": direction,
        "local_derivative_spearman": spearman(
            np.asarray(derivative_exact), np.asarray(derivative_predicted)
        )
        if direction_total >= 2
        else 0.0,
        "collapsed_prompt_count": len(collapsed_prompt_ids),
        "collapsed_prompt_fraction": collapsed_fraction,
        "collapsed_prompt_ids": sorted(collapsed_prompt_ids),
        "prompt_collapse_definition": {
            "maximum_predicted_range": LTE_COLLAPSE_PREDICTED_RANGE,
            "minimum_exact_range": LTE_COLLAPSE_EXACT_RANGE,
            "maximum_fraction": LTE_MAX_COLLAPSED_PROMPT_FRACTION,
        },
        "gates": gates,
        "gate_deficits": deficits,
        "total_gate_deficit": sum(deficits.values()),
        "all_gates_passed": all(gates.values()),
    }
