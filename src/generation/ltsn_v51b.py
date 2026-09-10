"""Build and evaluate the deterministic V5.1b memorization ladder."""

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

V51B_SCOPE = "deterministic_central_direction_memorization_v5_1b"
SYMMETRIC_KIND = "on_policy_symmetric"
DEFAULT_RUNGS = (1, 4, 16, 64)
DEFAULT_DEVELOPMENT_ANCHORS = 6


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _anchor_key(anchor_id: str) -> str:
    return hashlib.sha256(anchor_id.encode("utf-8")).hexdigest()


def _anchor_metadata(
    evidence_rows: Sequence[Mapping[str, str]],
    *,
    split: str,
) -> dict[str, tuple[int, int]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in evidence_rows:
        if row["split"] == split:
            grouped[row["anchor_sample_id"]].append(row)
    metadata: dict[str, tuple[int, int]] = {}
    for anchor_id, rows in sorted(grouped.items()):
        if len(rows) != 2:
            raise LTSNContractError(
                f"V5.1b anchor does not contain exactly two RMS pairs: {anchor_id}"
            )
        steps = {int(row["step_number"]) for row in rows}
        if len(steps) != 1 or not steps.issubset({4, 5, 6}):
            raise LTSNContractError(f"V5.1b anchor has an invalid step: {anchor_id}")
        differences = [
            float(row["exact_loss_minus"]) - float(row["exact_loss_plus"]) for row in rows
        ]
        if any(not math.isfinite(value) or value == 0.0 for value in differences):
            raise LTSNContractError(f"V5.1b anchor has a tied/non-finite direction: {anchor_id}")
        signs = {1 if value > 0 else -1 for value in differences}
        if len(signs) != 1:
            raise LTSNContractError(f"V5.1b anchor reverses direction across RMS: {anchor_id}")
        metadata[anchor_id] = (steps.pop(), signs.pop())
    if not metadata:
        raise LTSNContractError(f"V5.1b found no {split} anchors")
    return metadata


def nested_stratified_anchor_order(
    evidence_rows: Sequence[Mapping[str, str]],
    *,
    split: str,
) -> tuple[list[str], dict[str, Any]]:
    """Return a deterministic nested order that exposes all strata early."""

    metadata = _anchor_metadata(evidence_rows, split=split)
    strata_order = ((4, -1), (5, 1), (6, -1), (4, 1), (5, -1), (6, 1))
    strata: dict[tuple[int, int], list[str]] = {
        key: sorted((anchor for anchor, value in metadata.items() if value == key), key=_anchor_key)
        for key in strata_order
    }
    if any(not values for values in strata.values()):
        missing = [key for key, values in strata.items() if not values]
        raise LTSNContractError(f"V5.1b is missing step/sign strata for {split}: {missing}")
    ordered: list[str] = []
    index = 0
    while len(ordered) < len(metadata):
        added = False
        for key in strata_order:
            values = strata[key]
            if index < len(values):
                ordered.append(values[index])
                added = True
        if not added:
            break
        index += 1
    if len(ordered) != len(metadata) or len(set(ordered)) != len(ordered):
        raise LTSNContractError("V5.1b nested anchor ordering is incomplete")
    return ordered, {
        "anchors": len(ordered),
        "by_step_and_sign": {
            str(step): {
                "negative": sum(value == (step, -1) for value in metadata.values()),
                "positive": sum(value == (step, 1) for value in metadata.values()),
            }
            for step in (4, 5, 6)
        },
        "ordering": "sha256_with_step_sign_round_robin",
    }


def _relative_artifact_path(source_dir: Path, value: str, output_dir: Path) -> str:
    path = (source_dir / value).resolve()
    if not path.is_file():
        raise LTSNContractError(f"V5.1b source artifact is missing: {path}")
    return os.path.relpath(path, output_dir).replace("\\", "/")


def _build_rung(
    *,
    source_rows: Sequence[Mapping[str, str]],
    evidence_rows: Sequence[Mapping[str, str]],
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    central_evidence_path: Path,
    output_dir: Path,
    train_anchors: Sequence[str],
    development_anchors: Sequence[str],
) -> dict[str, Any]:
    selected_train = set(train_anchors)
    selected_development = set(development_anchors)
    wanted = selected_train | selected_development
    retained: list[dict[str, str]] = []
    retained_ids: set[str] = set()
    for source in source_rows:
        anchor_id = source.get("local_anchor_sample_id", "")
        keep_symmetric = (
            source.get("training_augmentation_kind") == SYMMETRIC_KIND and anchor_id in wanted
        )
        keep_anchor = source["sample_id"] in wanted
        if not (keep_symmetric or keep_anchor):
            continue
        row = dict(source)
        row["latent_path"] = _relative_artifact_path(
            source_manifest_path.parent, row["latent_path"], output_dir
        )
        row["exact_label_table_path"] = _relative_artifact_path(
            source_manifest_path.parent, row["exact_label_table_path"], output_dir
        )
        retained.append(row)
        retained_ids.add(row["sample_id"])
    missing = wanted - retained_ids
    if missing:
        raise LTSNContractError(f"V5.1b source manifest is missing anchors: {sorted(missing)}")
    split_counts = {
        split: sum(row["split"] == split for row in retained) for split in ("train", "development")
    }
    expected = {
        "train": len(selected_train) * 5,
        "development": len(selected_development) * 5,
    }
    if split_counts != expected:
        raise LTSNContractError(
            f"V5.1b requires one anchor and four symmetric rows per anchor: {split_counts}"
        )
    retained_evidence = [dict(row) for row in evidence_rows if row["anchor_sample_id"] in wanted]
    if len(retained_evidence) != 2 * len(wanted):
        raise LTSNContractError("V5.1b retained evidence does not contain two RMS pairs per anchor")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ltsn_manifest_v51b.csv"
    evidence_path = output_dir / "central_direction_exact_evidence_v51b.csv"
    write_csv_atomic(manifest_path, retained)
    write_csv_atomic(evidence_path, retained_evidence)
    assignments = dict(sorted({row["prompt_id"]: row["split"] for row in retained}.items()))
    split_payload = {
        "schema_version": 512,
        "grouping": "prompt_id+trajectory_id",
        "assignments": assignments,
        "training_augmentation_scope": V51B_SCOPE,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "training_manifest_sha256": sha256_file(manifest_path),
    }
    split_path = output_dir / "split_manifest_v51b.json"
    write_json_atomic(split_path, split_payload)
    summary = {
        "schema_version": 512,
        "scope": V51B_SCOPE,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "train_anchors": len(selected_train),
        "train_direction_pairs": len(selected_train) * 2,
        "development_anchors": len(selected_development),
        "development_direction_pairs": len(selected_development) * 2,
        "rows_by_split": split_counts,
        "selected_train_anchor_ids": list(train_anchors),
        "selected_development_anchor_ids": list(development_anchors),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "central_direction_evidence": str(evidence_path),
        "central_direction_evidence_sha256": sha256_file(evidence_path),
    }
    write_json_atomic(output_dir / "rung_summary.json", summary)
    return summary


def build_v51b_memorization_ladder(
    *,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    central_evidence_path: Path,
    output_root: Path,
    rungs: Sequence[int] = DEFAULT_RUNGS,
    development_anchor_count: int = DEFAULT_DEVELOPMENT_ANCHORS,
) -> dict[str, Any]:
    """Create nested 1/4/16/64-anchor views without copying latent or audio files."""

    counts = tuple(int(value) for value in rungs)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("V5.1b rung sizes must be positive")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("V5.1b rung sizes must be strictly increasing and unique")
    if development_anchor_count < 1:
        raise ValueError("V5.1b requires at least one development anchor")
    source_manifest_path = source_manifest_path.resolve()
    source_split_manifest_path = source_split_manifest_path.resolve()
    central_evidence_path = central_evidence_path.resolve()
    output_root = output_root.resolve()
    source_rows = _read_csv(source_manifest_path)
    evidence_rows = _read_csv(central_evidence_path)
    train_order, train_audit = nested_stratified_anchor_order(evidence_rows, split="train")
    development_order, development_audit = nested_stratified_anchor_order(
        evidence_rows, split="development"
    )
    if counts[-1] > len(train_order):
        raise LTSNContractError(
            f"V5.1b largest rung requests {counts[-1]} of {len(train_order)} train anchors"
        )
    if development_anchor_count > len(development_order):
        raise LTSNContractError(
            "V5.1b development anchor request exceeds the available stable anchors"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    rung_summaries = []
    for count in counts:
        name = f"rung_{count:03d}"
        summary = _build_rung(
            source_rows=source_rows,
            evidence_rows=evidence_rows,
            source_manifest_path=source_manifest_path,
            source_split_manifest_path=source_split_manifest_path,
            central_evidence_path=central_evidence_path,
            output_dir=output_root / name,
            train_anchors=train_order[:count],
            development_anchors=development_order[:development_anchor_count],
        )
        rung_summaries.append({"name": name, **summary})
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_1b_deterministic_memorization_ladder",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "rung_sizes": list(counts),
        "development_anchor_count": development_anchor_count,
        "train_selection_audit": train_audit,
        "development_selection_audit": development_audit,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "rungs": rung_summaries,
    }
    write_json_atomic(output_root / "memorization_ladder.json", payload)
    return payload


def memorization_agreement_threshold(anchor_count: int) -> float:
    """Return the preregistered agreement gate for one ladder rung."""

    thresholds = {1: 1.0, 4: 1.0, 16: 0.99, 64: 0.98}
    if anchor_count not in thresholds:
        raise ValueError(f"no frozen V5.1b threshold for {anchor_count} anchors")
    return thresholds[anchor_count]


def select_peak_memorization_epoch(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select peak train agreement, then lower loss, correlation, and earlier epoch."""

    if not history:
        raise LTSNContractError("V5.1b checkpoint history is empty")
    required = {
        "epoch",
        "train_loss",
        "overfit_train_central_direction_agreement",
        "overfit_train_central_derivative_spearman",
    }
    if any(not required.issubset(row) for row in history):
        raise LTSNContractError("V5.1b checkpoint history lacks memorization metrics")
    return max(
        history,
        key=lambda row: (
            float(row["overfit_train_central_direction_agreement"]),
            -float(row["train_loss"]),
            float(row["overfit_train_central_derivative_spearman"]),
            -int(row["epoch"]),
        ),
    )
