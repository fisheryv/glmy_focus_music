"""Build the bounded V5.1a central-direction overfit diagnostic view."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic

V51A_SCOPE = "central_direction_overfit_diagnostic_v5_1a"
SYMMETRIC_KIND = "on_policy_symmetric"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _anchor_key(anchor_id: str) -> str:
    return hashlib.sha256(anchor_id.encode("utf-8")).hexdigest()


def select_v51a_overfit_anchors(
    evidence_rows: list[dict[str, str]],
    *,
    step_quotas: dict[int, int],
) -> tuple[set[str], dict[str, Any]]:
    """Select deterministic, step- and sign-balanced train anchors."""

    if not step_quotas or any(step not in {4, 5, 6} for step in step_quotas):
        raise ValueError("V5.1a step quotas must cover only steps 4, 5, and 6")
    if any(quota <= 0 or quota % 2 for quota in step_quotas.values()):
        raise ValueError("V5.1a step quotas must be positive even integers")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evidence_rows:
        if row["split"] == "train":
            grouped[row["anchor_sample_id"]].append(row)
    candidates: dict[tuple[int, int], list[str]] = defaultdict(list)
    anchor_metadata: dict[str, tuple[int, int]] = {}
    for anchor_id, rows in sorted(grouped.items()):
        if len(rows) != 2:
            raise LTSNContractError(f"V5.1 anchor does not contain two RMS pairs: {anchor_id}")
        steps = {int(row["step_number"]) for row in rows}
        if len(steps) != 1:
            raise LTSNContractError(f"V5.1 anchor crosses denoising steps: {anchor_id}")
        differences = [
            float(row["exact_loss_minus"]) - float(row["exact_loss_plus"]) for row in rows
        ]
        if any(not math.isfinite(value) or value == 0 for value in differences):
            raise LTSNContractError(f"V5.1a received a tied or non-finite anchor: {anchor_id}")
        signs = {1 if value > 0 else -1 for value in differences}
        if len(signs) != 1:
            raise LTSNContractError(f"V5.1a received a cross-RMS sign reversal: {anchor_id}")
        step = steps.pop()
        sign = signs.pop()
        anchor_metadata[anchor_id] = (step, sign)
        candidates[(step, sign)].append(anchor_id)

    selected: set[str] = set()
    strata: dict[str, dict[str, int]] = {}
    for step, quota in sorted(step_quotas.items()):
        per_sign = quota // 2
        strata[str(step)] = {}
        for sign in (-1, 1):
            available = sorted(candidates[(step, sign)], key=_anchor_key)
            if len(available) < per_sign:
                raise LTSNContractError(
                    f"V5.1a cannot fill step {step}, sign {sign:+d}: "
                    f"required {per_sign}, available {len(available)}"
                )
            chosen = available[:per_sign]
            selected.update(chosen)
            strata[str(step)]["negative" if sign < 0 else "positive"] = len(chosen)
    if len(selected) != sum(step_quotas.values()):
        raise LTSNContractError("V5.1a selected anchor count differs from the frozen quota")
    return selected, {
        "step_quotas": {str(step): quota for step, quota in sorted(step_quotas.items())},
        "selected_by_step_and_sign": strata,
        "candidate_anchors": len(anchor_metadata),
        "selected_anchors": len(selected),
    }


def build_v51a_overfit_view(
    *,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    central_evidence_path: Path,
    output_dir: Path,
    step_quotas: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Create a central-only diagnostic view without copying latent or audio files."""

    quotas = {4: 32, 5: 16, 6: 16} if step_quotas is None else dict(step_quotas)
    source_manifest_path = source_manifest_path.resolve()
    source_split_manifest_path = source_split_manifest_path.resolve()
    central_evidence_path = central_evidence_path.resolve()
    output_dir = output_dir.resolve()
    source_rows = _read_csv(source_manifest_path)
    evidence_rows = _read_csv(central_evidence_path)
    selected_train, selection = select_v51a_overfit_anchors(evidence_rows, step_quotas=quotas)
    development_anchors = {
        row["anchor_sample_id"] for row in evidence_rows if row["split"] == "development"
    }
    if not development_anchors:
        raise LTSNContractError("V5.1a requires held-out development direction pairs")

    retained: list[dict[str, str]] = []
    retained_ids: set[str] = set()
    wanted_anchors = selected_train | development_anchors
    for source in source_rows:
        anchor_id = source.get("local_anchor_sample_id", "")
        keep_symmetric = (
            source.get("training_augmentation_kind") == SYMMETRIC_KIND
            and anchor_id in wanted_anchors
        )
        keep_anchor = source["sample_id"] in wanted_anchors
        if not (keep_symmetric or keep_anchor):
            continue
        row = dict(source)
        latent_path = (source_manifest_path.parent / row["latent_path"]).resolve()
        label_path = (source_manifest_path.parent / row["exact_label_table_path"]).resolve()
        if not latent_path.is_file() or not label_path.is_file():
            raise LTSNContractError(f"V5.1a source artifact is missing: {row['sample_id']}")
        row["latent_path"] = os.path.relpath(latent_path, output_dir).replace("\\", "/")
        row["exact_label_table_path"] = os.path.relpath(label_path, output_dir).replace(
            "\\", "/"
        )
        retained.append(row)
        retained_ids.add(row["sample_id"])

    missing_anchors = wanted_anchors - retained_ids
    if missing_anchors:
        raise LTSNContractError(
            f"V5.1a source manifest is missing {len(missing_anchors)} anchor rows"
        )
    expected_train_rows = len(selected_train) * 5
    expected_development_rows = len(development_anchors) * 5
    split_counts = {
        split: sum(row["split"] == split for row in retained)
        for split in ("train", "development")
    }
    if split_counts != {
        "train": expected_train_rows,
        "development": expected_development_rows,
    }:
        raise LTSNContractError(
            "V5.1a requires one anchor and four symmetric rows for every selected anchor"
        )

    retained_evidence = [
        row
        for row in evidence_rows
        if row["anchor_sample_id"] in selected_train | development_anchors
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ltsn_manifest_v51a.csv"
    write_csv_atomic(manifest_path, retained)
    evidence_path = output_dir / "central_direction_exact_evidence_v51a.csv"
    write_csv_atomic(evidence_path, retained_evidence)
    assignments = dict(sorted({row["prompt_id"]: row["split"] for row in retained}.items()))
    split_payload = {
        "schema_version": 511,
        "grouping": "prompt_id+trajectory_id",
        "assignments": assignments,
        "training_augmentation_scope": V51A_SCOPE,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "training_manifest_sha256": sha256_file(manifest_path),
    }
    split_path = output_dir / "split_manifest_v51a.json"
    write_json_atomic(split_path, split_payload)
    summary = {
        "schema_version": 511,
        "scope": V51A_SCOPE,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "selection": selection,
        "train_anchors": len(selected_train),
        "train_direction_pairs": len(selected_train) * 2,
        "development_anchors": len(development_anchors),
        "development_direction_pairs": len(development_anchors) * 2,
        "rows_by_split": split_counts,
        "reused_audio_and_latents": True,
        "new_audio_files": 0,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "central_direction_evidence": str(evidence_path),
        "central_direction_evidence_sha256": sha256_file(evidence_path),
    }
    write_json_atomic(output_dir / "diagnostic_view_summary.json", summary)
    return summary
