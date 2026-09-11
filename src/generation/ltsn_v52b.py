"""V5.2b scale-consistency audit and matched-label probe preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .ltsn_v51c import wilson_interval

V52B_EXPERIMENT = "ltsn_v5_2b_scale_consistency_and_matched_probe"
V52B_VARIANTS = ("scalar", "pair", "direction_field")
V52B_TARGET_MODES = ("true", "matched_control")
PRIMARY_VARIANT = "direction_field"
MINIMUM_CROSS_RMS_SIGN_AGREEMENT = 0.80
WILSON_LOWER_MUST_EXCEED = 0.50


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            result[index] = average
        start = end
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _scale_audit(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rms_values = sorted({float(row["rms_ratio"]) for row in rows})
    if len(rms_values) != 2:
        raise LTSNContractError("V5.2b requires exactly two frozen V5.2a RMS ratios")
    grouped: dict[tuple[str, int], dict[float, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("informative", "").lower() != "true":
            raise LTSNContractError("V5.2b requires every V5.2a pair to be informative")
        if row.get("pair_in_distribution", "").lower() != "true":
            raise LTSNContractError("V5.2b requires every V5.2a pair to be in-distribution")
        key = (row["anchor_sample_id"], int(row["direction_index"]))
        rms = float(row["rms_ratio"])
        if rms in grouped[key]:
            raise LTSNContractError("duplicate V5.2a anchor/direction/RMS evidence")
        grouped[key][rms] = row
    if any(set(group) != set(rms_values) for group in grouped.values()):
        raise LTSNContractError("V5.2a evidence is not rectangular across the two RMS ratios")

    def summarize(groups: Sequence[dict[float, Mapping[str, str]]]) -> dict[str, Any]:
        small_values = [float(group[rms_values[0]]["exact_derivative"]) for group in groups]
        large_values = [float(group[rms_values[1]]["exact_derivative"]) for group in groups]
        stable = sum(
            left * right > 0.0 for left, right in zip(small_values, large_values, strict=True)
        )
        total = len(groups)
        low, high = wilson_interval(stable, total)
        pair_count = total * 2
        invariant_success_ceiling = stable * 2 + (total - stable)
        return {
            "direction_groups": total,
            "stable_sign_groups": stable,
            "conflicting_sign_groups": total - stable,
            "cross_rms_sign_agreement": stable / total,
            "cross_rms_sign_agreement_wilson95": [low, high],
            "cross_rms_derivative_spearman": _correlation(_rank(small_values), _rank(large_values)),
            "rms_invariant_pair_count": pair_count,
            "rms_invariant_success_ceiling": invariant_success_ceiling,
            "rms_invariant_agreement_ceiling": invariant_success_ceiling / pair_count,
        }

    by_partition: dict[str, list[dict[float, Mapping[str, str]]]] = defaultdict(list)
    by_partition_step: dict[tuple[str, int], list[dict[float, Mapping[str, str]]]] = defaultdict(
        list
    )
    for group in grouped.values():
        first = group[rms_values[0]]
        partition = first["direction_partition"]
        step = int(first["step_number"])
        if any(
            row["direction_partition"] != partition or int(row["step_number"]) != step
            for row in group.values()
        ):
            raise LTSNContractError("V5.2a RMS group changes partition or diffusion step")
        by_partition[partition].append(group)
        by_partition_step[(partition, step)].append(group)
    expected = {"train_direction", "heldout_direction", "unseen_anchor"}
    if set(by_partition) != expected:
        raise LTSNContractError("V5.2b received unexpected V5.2a direction partitions")
    partitions = {
        partition: {
            **summarize(groups),
            "by_step": {
                str(step): summarize(by_partition_step[(partition, step)]) for step in (4, 5, 6)
            },
        }
        for partition, groups in sorted(by_partition.items())
    }
    criteria = {
        "cross_rms_sign_agreement_each_partition_at_least_0_80": all(
            summary["cross_rms_sign_agreement"] >= MINIMUM_CROSS_RMS_SIGN_AGREEMENT
            for summary in partitions.values()
        ),
        "cross_rms_wilson_lower_each_partition_above_0_50": all(
            summary["cross_rms_sign_agreement_wilson95"][0] > WILSON_LOWER_MUST_EXCEED
            for summary in partitions.values()
        ),
    }
    return {
        "rms_ratios": rms_values,
        "partitions": partitions,
        "thresholds": {
            "minimum_cross_rms_sign_agreement": MINIMUM_CROSS_RMS_SIGN_AGREEMENT,
            "wilson_lower_must_exceed": WILSON_LOWER_MUST_EXCEED,
        },
        "criteria": criteria,
        "cross_rms_direction_field_supported": all(criteria.values()),
    }


def _choose_label_permutation(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[dict[str, tuple[int, str]], dict[str, Any]]:
    """Permute only labels while preserving every endpoint pair and stratum count."""

    output: dict[str, tuple[int, str]] = {}
    strata: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[int(row["step_number"])].append(row)
    summaries = {}
    for step, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest(),
        )
        if len(ordered) < 2:
            raise LTSNContractError("matched-label control stratum is too small")
        targets = {
            value: [row for row in ordered if int(row["true_target"]) == value] for value in (0, 1)
        }
        sources = {value: list(values) for value, values in targets.items()}
        assignments: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        opposite_count = min(len(targets[0]), len(targets[1]))
        assignments.extend(
            zip(
                targets[0][:opposite_count],
                sources[1][:opposite_count],
                strict=True,
            )
        )
        assignments.extend(
            zip(
                targets[1][:opposite_count],
                sources[0][:opposite_count],
                strict=True,
            )
        )
        for value in (0, 1):
            remaining_targets = targets[value][opposite_count:]
            remaining_sources = sources[value][opposite_count:]
            if len(remaining_sources) > 1:
                remaining_sources = remaining_sources[1:] + remaining_sources[:1]
            assignments.extend(zip(remaining_targets, remaining_sources, strict=True))
        for target, source in assignments:
            output[str(target["pair_id"])] = (
                int(source["true_target"]),
                str(source["pair_id"]),
            )
        source_ids = [str(source["pair_id"]) for _, source in assignments]
        if len(assignments) != len(ordered) or len(set(source_ids)) != len(ordered):
            raise LTSNContractError("matched-label control is not a complete permutation")
        if any(str(target["pair_id"]) == str(source["pair_id"]) for target, source in assignments):
            raise LTSNContractError("matched-label control permutation contains a fixed point")
        best_changed = 2 * opposite_count
        summaries[str(step)] = {
            "pairs": len(ordered),
            "positive_targets": sum(int(row["true_target"]) for row in ordered),
            "changed_targets": best_changed,
            "changed_fraction": best_changed / len(ordered),
            "theoretical_maximum_changed_targets": best_changed,
            "theoretical_maximum_attained": True,
        }
    changed = sum(int(row["true_target"]) != output[str(row["pair_id"])][0] for row in rows)
    return output, {
        "seed": seed,
        "grouping": "step_number",
        "endpoint_pairs_preserved": True,
        "latent_distances_preserved": True,
        "class_counts_preserved_within_stratum": True,
        "source_pair_derangement": True,
        "pairs": len(rows),
        "changed_targets": changed,
        "changed_fraction": changed / len(rows),
        "theoretical_maximum_changed_targets": sum(
            row["theoretical_maximum_changed_targets"] for row in summaries.values()
        ),
        "theoretical_maximum_attained": changed
        == sum(row["theoretical_maximum_changed_targets"] for row in summaries.values()),
        "by_step": summaries,
    }


def prepare_v52b_probe(
    *,
    master_manifest_path: Path,
    evidence_path: Path,
    views_summary_path: Path,
    v52a_report_path: Path,
    output_dir: Path,
    operational_rms: float = 0.005,
    control_seed: int = 20260911,
) -> dict[str, Any]:
    """Build the frozen V5.2b pair table without decoding or exact rescoring."""

    master_manifest_path = master_manifest_path.resolve()
    evidence_path = evidence_path.resolve()
    views_summary_path = views_summary_path.resolve()
    v52a_report_path = v52a_report_path.resolve()
    output_dir = output_dir.resolve()
    views = json.loads(views_summary_path.read_text(encoding="utf-8"))
    report = json.loads(v52a_report_path.read_text(encoding="utf-8"))
    if (
        views.get("experiment") != "ltsn_v5_2a_multi_direction_identifiability"
        or views.get("diagnostic_only") is not True
        or views.get("qualification_eligible") is not False
        or views.get("guidance_promotion_eligible") is not False
    ):
        raise LTSNContractError("V5.2b requires bounded V5.2a views")
    if (
        report.get("experiment") != "ltsn_v5_2a_multi_direction_identifiability"
        or report.get("status") != "multi_direction_identifiability_not_supported"
        or report.get("identifiability_supported") is not False
    ):
        raise LTSNContractError("V5.2b requires the failed frozen V5.2a report")
    if views.get("master_manifest_sha256") != sha256_file(master_manifest_path):
        raise LTSNContractError("V5.2a master manifest hash mismatch")
    if views.get("central_direction_evidence_sha256") != sha256_file(evidence_path):
        raise LTSNContractError("V5.2a evidence hash mismatch")
    if report.get("views_summary_sha256") != sha256_file(views_summary_path):
        raise LTSNContractError("V5.2a report/views hash mismatch")

    master_rows = _read_csv(master_manifest_path)
    evidence_rows = _read_csv(evidence_path)
    scale_audit = _scale_audit(evidence_rows)
    if operational_rms not in scale_audit["rms_ratios"]:
        raise LTSNContractError("operational RMS is absent from V5.2a evidence")
    sample_by_id = {row["sample_id"]: row for row in master_rows}
    if len(sample_by_id) != len(master_rows):
        raise LTSNContractError("V5.2a master manifest contains duplicate sample IDs")
    pairs: list[dict[str, Any]] = []
    split_by_partition = {
        "train_direction": "train",
        "heldout_direction": "seen_anchor_heldout_direction",
        "unseen_anchor": "unseen_anchor",
    }
    for evidence in evidence_rows:
        minus = sample_by_id.get(evidence["minus_sample_id"])
        plus = sample_by_id.get(evidence["plus_sample_id"])
        if minus is None or plus is None:
            raise LTSNContractError("V5.2a evidence references an absent manifest sample")
        if (
            minus["prompt_id"] != plus["prompt_id"]
            or minus["step_number"] != plus["step_number"]
            or minus["timestep"] != plus["timestep"]
        ):
            raise LTSNContractError("V5.2a pair endpoint metadata differs")
        rms = float(evidence["rms_ratio"])
        partition = evidence["direction_partition"]
        base_split = split_by_partition[partition]
        evaluation_split = (
            base_split if math.isclose(rms, operational_rms) else f"{base_split}_rms_sensitivity"
        )
        difference = float(evidence["exact_loss_minus"]) - float(evidence["exact_loss_plus"])
        if difference == 0.0:
            raise LTSNContractError("V5.2b cannot use a tied exact direction target")
        pairs.append(
            {
                "pair_id": evidence["direction_group_id"],
                "anchor_sample_id": evidence["anchor_sample_id"],
                "prompt_id": minus["prompt_id"],
                "direction_partition": partition,
                "evaluation_split": evaluation_split,
                "step_number": int(evidence["step_number"]),
                "timestep": float(minus["timestep"]),
                "direction_index": int(evidence["direction_index"]),
                "rms_ratio": rms,
                "minus_sample_id": minus["sample_id"],
                "plus_sample_id": plus["sample_id"],
                "minus_latent_path": os.path.relpath(
                    (master_manifest_path.parent / minus["latent_path"]).resolve(), output_dir
                ).replace("\\", "/"),
                "plus_latent_path": os.path.relpath(
                    (master_manifest_path.parent / plus["latent_path"]).resolve(), output_dir
                ).replace("\\", "/"),
                "minus_latent_sha256": minus["latent_sha256"],
                "plus_latent_sha256": plus["latent_sha256"],
                "exact_derivative": float(evidence["exact_derivative"]),
                "true_target": int(difference > 0.0),
                "matched_control_target": "",
                "matched_control_source_pair_id": "",
                "operational_rms": math.isclose(rms, operational_rms),
            }
        )
    if len({row["pair_id"] for row in pairs}) != len(pairs):
        raise LTSNContractError("V5.2b pair IDs are not unique")
    train_pairs = [
        row for row in pairs if row["evaluation_split"] == "train" and row["operational_rms"]
    ]
    permutation, control_audit = _choose_label_permutation(train_pairs, seed=control_seed)
    if not control_audit["theoretical_maximum_attained"]:
        raise LTSNContractError("matched-label control did not attain the stratum-wise optimum")
    for row in pairs:
        if row["evaluation_split"] == "train":
            target, source_id = permutation[row["pair_id"]]
            row["matched_control_target"] = target
            row["matched_control_source_pair_id"] = source_id
        else:
            row["matched_control_target"] = row["true_target"]
            row["matched_control_source_pair_id"] = row["pair_id"]

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "v52b_pair_manifest.csv"
    write_csv_atomic(pair_path, pairs)
    v52a_train = report["true_pairs"]
    observed_successes = sorted(
        {int(seed_report["metrics"]["train"]["successes"]) for seed_report in v52a_train}
    )
    ceiling = scale_audit["partitions"]["train_direction"]
    ceiling_explains_v52a = observed_successes == [ceiling["rms_invariant_success_ceiling"]]
    split_counts = {
        split: sum(row["evaluation_split"] == split for row in pairs)
        for split in sorted({str(row["evaluation_split"]) for row in pairs})
    }
    payload = {
        "schema_version": 1,
        "experiment": V52B_EXPERIMENT,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "reuses_v52a_decodes": True,
        "new_audio_files": 0,
        "primary_variant": PRIMARY_VARIANT,
        "variants": list(V52B_VARIANTS),
        "target_modes": list(V52B_TARGET_MODES),
        "operational_rms": operational_rms,
        "sensitivity_rms": next(
            value for value in scale_audit["rms_ratios"] if value != operational_rms
        ),
        "source": {
            "master_manifest_sha256": sha256_file(master_manifest_path),
            "central_direction_evidence_sha256": sha256_file(evidence_path),
            "views_summary_sha256": sha256_file(views_summary_path),
            "v52a_report_sha256": sha256_file(v52a_report_path),
        },
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "pair_counts": split_counts,
        "scale_consistency_audit": scale_audit,
        "v52a_ceiling_check": {
            "observed_train_successes_each_seed": observed_successes,
            "rms_invariant_success_ceiling": ceiling["rms_invariant_success_ceiling"],
            "rms_invariant_agreement_ceiling": ceiling["rms_invariant_agreement_ceiling"],
            "observed_train_agreement_equals_invariant_ceiling": ceiling_explains_v52a,
        },
        "matched_label_control": control_audit,
        "operational_probe_allowed": True,
        "cross_rms_direction_field_supported": scale_audit["cross_rms_direction_field_supported"],
        "status": (
            "prepared_cross_rms_field_supported"
            if scale_audit["cross_rms_direction_field_supported"]
            else "prepared_cross_rms_field_not_supported"
        ),
        "interpretation": (
            "V5.2b isolates the frozen operational RMS and uses a pair-preserving label null; "
            "cross-RMS support remains an independent prerequisite for infinitesimal guidance"
        ),
    }
    summary_path = output_dir / "v52b_preparation.json"
    write_json_atomic(summary_path, payload)
    payload["preparation_sha256"] = sha256_file(summary_path)
    return payload
