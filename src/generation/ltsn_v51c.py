"""Build the V5.1c one-factor training ablation suite."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .ltsn_v51b import nested_stratified_anchor_order

V51C_SCOPE = "central_direction_training_factor_ablation_v5_1c"
V51C_CONTROL_SCOPE = "central_direction_permuted_pair_control_v5_1c"
EXPECTED_TRAIN_ANCHORS = 64
EXPECTED_DEVELOPMENT_ANCHORS = 78
PERMUTATION_SEED = 20260910
EXACT_MARGIN = 1e-5

V51C_VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {
        "config": "ltsn_training_v51c_baseline.toml",
        "view": "true_pairs",
        "restored_factor": "none_v51b_recipe",
    },
    "restore_dropout": {
        "config": "ltsn_training_v51c_restore_dropout.toml",
        "view": "true_pairs",
        "restored_factor": "model_dropout_and_stem_dropout",
    },
    "restore_weight_decay": {
        "config": "ltsn_training_v51c_restore_weight_decay.toml",
        "view": "true_pairs",
        "restored_factor": "optimizer_weight_decay",
    },
    "restore_lr_schedule": {
        "config": "ltsn_training_v51c_restore_lr_schedule.toml",
        "view": "true_pairs",
        "restored_factor": "learning_rate_warmup_and_cosine_schedule",
    },
    "restore_batch32": {
        "config": "ltsn_training_v51c_restore_batch32.toml",
        "view": "true_pairs",
        "restored_factor": "effective_batch_size",
    },
    "restore_short_early_stop": {
        "config": "ltsn_training_v51c_restore_short_early_stop.toml",
        "view": "true_pairs",
        "restored_factor": "v51a_epoch_budget_and_early_stopping",
    },
    "permuted_pair_control": {
        "config": "ltsn_training_v51c_permuted_pair_control.toml",
        "view": "permuted_pair_control",
        "restored_factor": "none_control_only",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _relative_artifact_path(source_dir: Path, value: str, output_dir: Path) -> str:
    path = (source_dir / value).resolve()
    if not path.is_file():
        raise LTSNContractError(f"V5.1c source artifact is missing: {path}")
    return os.path.relpath(path, output_dir).replace("\\", "/")


def _rewrite_view_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_dir: Path,
    output_dir: Path,
) -> list[dict[str, str]]:
    rewritten = []
    for source in rows:
        row = dict(source)
        row["latent_path"] = _relative_artifact_path(source_dir, row["latent_path"], output_dir)
        row["exact_label_table_path"] = _relative_artifact_path(
            source_dir, row["exact_label_table_path"], output_dir
        )
        rewritten.append(row)
    return rewritten


def _validate_full_v51a_view(
    manifest_rows: Sequence[Mapping[str, str]],
    evidence_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    train_order, train_audit = nested_stratified_anchor_order(evidence_rows, split="train")
    development_order, development_audit = nested_stratified_anchor_order(
        evidence_rows, split="development"
    )
    if len(train_order) != EXPECTED_TRAIN_ANCHORS:
        raise LTSNContractError(
            f"V5.1c requires {EXPECTED_TRAIN_ANCHORS} train anchors, got {len(train_order)}"
        )
    if len(development_order) != EXPECTED_DEVELOPMENT_ANCHORS:
        raise LTSNContractError(
            f"V5.1c requires all 78 V5.1a development anchors, got {len(development_order)}"
        )
    counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in ("train", "development")
    }
    expected = {
        "train": EXPECTED_TRAIN_ANCHORS * 5,
        "development": EXPECTED_DEVELOPMENT_ANCHORS * 5,
    }
    if counts != expected:
        raise LTSNContractError(
            f"V5.1c requires one anchor and four perturbations per anchor: {counts}"
        )
    return {
        "train": train_audit,
        "development": development_audit,
        "rows_by_split": counts,
    }


def _permutation_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def _choose_shift(rows: Sequence[Mapping[str, str]]) -> tuple[int, dict[str, int]]:
    if len(rows) < 2:
        raise LTSNContractError("V5.1c permutation stratum needs at least two pairs")
    best: tuple[tuple[int, int, int, int], int, dict[str, int]] | None = None
    for shift in range(1, len(rows)):
        differences = [
            float(left["exact_loss_minus"])
            - float(rows[(index + shift) % len(rows)]["exact_loss_plus"])
            for index, left in enumerate(rows)
        ]
        informative = [value for value in differences if abs(value) >= EXACT_MARGIN]
        positive = sum(value > 0 for value in informative)
        negative = sum(value < 0 for value in informative)
        stats = {
            "pairs": len(rows),
            "informative_pairs": len(informative),
            "positive_targets": positive,
            "negative_targets": negative,
        }
        score = (
            len(informative),
            min(positive, negative),
            -abs(positive - negative),
            -shift,
        )
        candidate = (score, shift, stats)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def build_permuted_pair_control(
    manifest_rows: Sequence[Mapping[str, str]],
    evidence_rows: Sequence[Mapping[str, str]],
    *,
    seed: int = PERMUTATION_SEED,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Re-pair train minus/plus samples while retaining their exact labels."""

    rows = [dict(row) for row in manifest_rows]
    by_sample = {row["sample_id"]: row for row in rows}
    strata: dict[tuple[int, float], list[Mapping[str, str]]] = defaultdict(list)
    for evidence in evidence_rows:
        if evidence["split"] == "train":
            strata[(int(evidence["step_number"]), float(evidence["rms_ratio"]))].append(evidence)
    pair_map: list[dict[str, Any]] = []
    stratum_audit: dict[str, Any] = {}
    assigned_samples: set[str] = set()
    for (step, rms), evidence in sorted(strata.items()):
        ordered = sorted(
            evidence,
            key=lambda row: _permutation_key(row["direction_group_id"], seed),
        )
        shift, stats = _choose_shift(ordered)
        stratum_name = f"step_{step}_rms_{rms:g}"
        stratum_audit[stratum_name] = {"cyclic_shift": shift, **stats}
        for index, minus_source in enumerate(ordered):
            plus_source = ordered[(index + shift) % len(ordered)]
            minus_id = minus_source["minus_sample_id"]
            plus_id = plus_source["plus_sample_id"]
            if minus_id not in by_sample or plus_id not in by_sample:
                raise LTSNContractError("V5.1c evidence references a missing manifest sample")
            group_id = f"v51c_perm_s{step}_r{rms:g}_{index:03d}"
            for sample_id, sign in ((minus_id, -1), (plus_id, 1)):
                row = by_sample[sample_id]
                row["local_direction_group_id"] = group_id
                row["local_direction_sign"] = str(sign)
                row["local_direction_rms_ratio"] = str(rms)
                row["local_anchor_sample_id"] = ""
                assigned_samples.add(sample_id)
            difference = float(minus_source["exact_loss_minus"]) - float(
                plus_source["exact_loss_plus"]
            )
            pair_map.append(
                {
                    "permuted_group_id": group_id,
                    "step_number": step,
                    "rms_ratio": rms,
                    "minus_sample_id": minus_id,
                    "minus_source_group_id": minus_source["direction_group_id"],
                    "plus_sample_id": plus_id,
                    "plus_source_group_id": plus_source["direction_group_id"],
                    "exact_loss_difference": difference,
                    "informative": abs(difference) >= EXACT_MARGIN,
                    "target_sign": 1 if difference > 0 else -1,
                }
            )
    expected_samples = {
        row["sample_id"]
        for row in rows
        if row["split"] == "train" and row.get("local_direction_group_id", "")
    }
    if assigned_samples != expected_samples:
        raise LTSNContractError("V5.1c permutation did not assign every train direction sample")
    if any(row["minus_source_group_id"] == row["plus_source_group_id"] for row in pair_map):
        raise LTSNContractError("V5.1c permutation retained an original train pair")
    summary = {
        "seed": seed,
        "method": "within_step_and_rms_cyclic_plus_member_permutation",
        "preserves_exact_sample_labels": True,
        "original_train_pairs_retained": 0,
        "pairs": len(pair_map),
        "informative_pairs": sum(bool(row["informative"]) for row in pair_map),
        "positive_targets": sum(row["target_sign"] > 0 for row in pair_map),
        "negative_targets": sum(row["target_sign"] < 0 for row in pair_map),
        "strata": stratum_audit,
    }
    return rows, pair_map, summary


def _write_view(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    central_evidence_path: Path,
    scope: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ltsn_manifest_v51c.csv"
    evidence_path = output_dir / "central_direction_exact_evidence_v51c.csv"
    write_csv_atomic(manifest_path, rows)
    write_csv_atomic(evidence_path, evidence_rows)
    split_payload = {
        "schema_version": 513,
        "grouping": "prompt_id+trajectory_id",
        "assignments": dict(assignments),
        "training_augmentation_scope": scope,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "training_manifest_sha256": sha256_file(manifest_path),
    }
    split_path = output_dir / "split_manifest_v51c.json"
    write_json_atomic(split_path, split_payload)
    return {
        "scope": scope,
        "manifest": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "central_direction_evidence": str(evidence_path),
        "central_direction_evidence_sha256": sha256_file(evidence_path),
    }


def build_v51c_ablation_suite(
    *,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    central_evidence_path: Path,
    config_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create full-development true and permuted-control views plus a frozen plan."""

    source_manifest_path = source_manifest_path.resolve()
    source_split_manifest_path = source_split_manifest_path.resolve()
    central_evidence_path = central_evidence_path.resolve()
    config_root = config_root.resolve()
    output_root = output_root.resolve()
    source_rows = _read_csv(source_manifest_path)
    evidence_rows = _read_csv(central_evidence_path)
    audit = _validate_full_v51a_view(source_rows, evidence_rows)
    assignments = dict(sorted({row["prompt_id"]: row["split"] for row in source_rows}.items()))

    true_dir = output_root / "true_pairs"
    true_rows = _rewrite_view_rows(
        source_rows, source_dir=source_manifest_path.parent, output_dir=true_dir
    )
    true_view = _write_view(
        output_dir=true_dir,
        rows=true_rows,
        evidence_rows=evidence_rows,
        assignments=assignments,
        source_manifest_path=source_manifest_path,
        source_split_manifest_path=source_split_manifest_path,
        central_evidence_path=central_evidence_path,
        scope=V51C_SCOPE,
    )

    control_dir = output_root / "permuted_pair_control"
    control_source_rows, pair_map, control_summary = build_permuted_pair_control(
        source_rows, evidence_rows
    )
    control_rows = _rewrite_view_rows(
        control_source_rows,
        source_dir=source_manifest_path.parent,
        output_dir=control_dir,
    )
    control_view = _write_view(
        output_dir=control_dir,
        rows=control_rows,
        evidence_rows=evidence_rows,
        assignments=assignments,
        source_manifest_path=source_manifest_path,
        source_split_manifest_path=source_split_manifest_path,
        central_evidence_path=central_evidence_path,
        scope=V51C_CONTROL_SCOPE,
    )
    pair_map_path = control_dir / "permuted_pair_map_v51c.csv"
    write_csv_atomic(pair_map_path, pair_map)
    control_view["permutation"] = {
        **control_summary,
        "pair_map": str(pair_map_path),
        "pair_map_sha256": sha256_file(pair_map_path),
    }

    variants = []
    for name, specification in V51C_VARIANTS.items():
        config_path = config_root / specification["config"]
        if not config_path.is_file():
            raise LTSNContractError(f"V5.1c config is missing: {config_path}")
        variants.append(
            {
                "name": name,
                **specification,
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_1c_one_factor_training_ablation",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "single_seed_screen": True,
        "multi_seed_confirmation_required": True,
        "train_anchors": EXPECTED_TRAIN_ANCHORS,
        "train_direction_pairs": EXPECTED_TRAIN_ANCHORS * 2,
        "development_anchors": EXPECTED_DEVELOPMENT_ANCHORS,
        "development_direction_pairs": EXPECTED_DEVELOPMENT_ANCHORS * 2,
        "selection_audit": audit,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "views": {
            "true_pairs": true_view,
            "permuted_pair_control": control_view,
        },
        "variants": variants,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "ablation_suite.json", payload)
    return payload


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval."""

    if trials <= 0 or successes < 0 or successes > trials or not math.isfinite(z) or z <= 0:
        raise ValueError("invalid Wilson interval inputs")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width
