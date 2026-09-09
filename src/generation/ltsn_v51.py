"""Build the V5.1 stable finite-difference view without decoding new audio."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic

V51_SCOPE = "stable_symmetric_finite_difference_v5_1"
SYMMETRIC_KIND = "on_policy_symmetric"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _stable_anchor_selection(
    evidence_rows: list[dict[str, str]],
    *,
    expected_rms_ratios: tuple[float, ...],
    minimum_abs_loss_separation: float,
) -> tuple[set[str], list[dict[str, str]], dict[str, Any]]:
    if minimum_abs_loss_separation < 0 or not math.isfinite(minimum_abs_loss_separation):
        raise ValueError("minimum_abs_loss_separation must be finite and non-negative")
    expected_rms = tuple(sorted(float(value) for value in expected_rms_ratios))
    if not expected_rms or any(value <= 0 or not math.isfinite(value) for value in expected_rms):
        raise ValueError("expected_rms_ratios must be finite and positive")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in evidence_rows:
        grouped[(row["split"], row["anchor_sample_id"])].append(row)

    selected: set[str] = set()
    retained_evidence: list[dict[str, str]] = []
    audit: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "candidate_anchors": 0,
            "stable_anchors": 0,
            "opposite_sign_anchors": 0,
            "tied_anchors": 0,
            "weak_effect_anchors": 0,
        }
    )
    for (split, anchor_id), rows in sorted(grouped.items()):
        audit[split]["candidate_anchors"] += 1
        if len(rows) != len(expected_rms):
            raise LTSNContractError(
                f"V5 anchor does not contain one central group per RMS: {anchor_id}"
            )
        rms_values = tuple(sorted(float(row["rms_ratio"]) for row in rows))
        if rms_values != expected_rms:
            raise LTSNContractError(f"V5 anchor RMS set differs from the contract: {anchor_id}")
        derivatives = [float(row["exact_derivative"]) for row in rows]
        separations = [
            abs(float(row["exact_loss_minus"]) - float(row["exact_loss_plus"])) for row in rows
        ]
        if any(not math.isfinite(value) for value in (*derivatives, *separations)):
            raise LTSNContractError(f"V5 central evidence contains NaN or Inf: {anchor_id}")
        signs = {1 if value > 0 else -1 if value < 0 else 0 for value in derivatives}
        if 0 in signs:
            audit[split]["tied_anchors"] += 1
            continue
        if len(signs) != 1:
            audit[split]["opposite_sign_anchors"] += 1
            continue
        if min(separations) < minimum_abs_loss_separation:
            audit[split]["weak_effect_anchors"] += 1
            continue
        selected.add(anchor_id)
        retained_evidence.extend(sorted(rows, key=lambda row: float(row["rms_ratio"])))
        audit[split]["stable_anchors"] += 1
    if not selected:
        raise LTSNContractError("V5.1 stable-direction filtering retained no anchors")
    return selected, retained_evidence, {key: dict(value) for key, value in sorted(audit.items())}


def build_v51_training_view(
    *,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    central_evidence_path: Path,
    output_dir: Path,
    minimum_abs_loss_separation: float = 1e-5,
    expected_rms_ratios: tuple[float, ...] = (0.0025, 0.005),
) -> dict[str, Any]:
    """Filter unstable V5 anchors and write a path-rebased V5.1 manifest."""

    source_manifest_path = source_manifest_path.resolve()
    source_split_manifest_path = source_split_manifest_path.resolve()
    central_evidence_path = central_evidence_path.resolve()
    output_dir = output_dir.resolve()
    source_rows = _read_csv(source_manifest_path)
    evidence_rows = _read_csv(central_evidence_path)
    selected, retained_evidence, selection_audit = _stable_anchor_selection(
        evidence_rows,
        expected_rms_ratios=expected_rms_ratios,
        minimum_abs_loss_separation=minimum_abs_loss_separation,
    )

    retained_rows: list[dict[str, str]] = []
    removed_symmetric_rows = 0
    for source in source_rows:
        row = dict(source)
        if row.get("training_augmentation_kind") == SYMMETRIC_KIND:
            anchor_id = row.get("local_anchor_sample_id", "")
            if anchor_id not in selected:
                removed_symmetric_rows += 1
                continue
        latent_path = (source_manifest_path.parent / row["latent_path"]).resolve()
        label_path = (source_manifest_path.parent / row["exact_label_table_path"]).resolve()
        if not latent_path.is_file() or not label_path.is_file():
            raise LTSNContractError(f"V5.1 source artifact is missing: {row['sample_id']}")
        row["latent_path"] = os.path.relpath(latent_path, output_dir).replace("\\", "/")
        row["exact_label_table_path"] = os.path.relpath(label_path, output_dir).replace("\\", "/")
        retained_rows.append(row)

    retained_symmetric = [
        row for row in retained_rows if row.get("training_augmentation_kind") == SYMMETRIC_KIND
    ]
    expected_symmetric = len(selected) * len(expected_rms_ratios) * 2
    if len(retained_symmetric) != expected_symmetric:
        raise LTSNContractError(
            "V5.1 manifest does not contain exactly two signs for every retained RMS group"
        )
    selected_by_split = {
        split: len(
            {row["local_anchor_sample_id"] for row in retained_symmetric if row["split"] == split}
        )
        for split in ("train", "development")
    }
    if not all(selected_by_split.values()):
        raise LTSNContractError("V5.1 requires stable train and development anchors")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ltsn_manifest_v51.csv"
    write_csv_atomic(manifest_path, retained_rows)
    evidence_path = output_dir / "central_direction_exact_evidence_v51.csv"
    write_csv_atomic(evidence_path, retained_evidence)

    split_payload = json.loads(source_split_manifest_path.read_text(encoding="utf-8"))
    split_payload.update(
        {
            "schema_version": 51,
            "training_augmentation_scope": V51_SCOPE,
            "source_training_manifest_sha256": sha256_file(source_manifest_path),
            "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
            "central_direction_evidence_sha256": sha256_file(central_evidence_path),
            "training_manifest_sha256": sha256_file(manifest_path),
        }
    )
    split_path = output_dir / "split_manifest_v51.json"
    write_json_atomic(split_path, split_payload)
    summary = {
        "schema_version": 51,
        "scope": V51_SCOPE,
        "source_training_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_central_direction_evidence_sha256": sha256_file(central_evidence_path),
        "expected_rms_ratios": list(expected_rms_ratios),
        "minimum_abs_loss_separation": minimum_abs_loss_separation,
        "selection_audit": selection_audit,
        "stable_anchors_by_split": selected_by_split,
        "source_samples": len(source_rows),
        "retained_samples": len(retained_rows),
        "retained_symmetric_samples": len(retained_symmetric),
        "removed_symmetric_samples": removed_symmetric_rows,
        "reused_audio_and_latents": True,
        "new_audio_files": 0,
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "central_direction_evidence": str(evidence_path),
        "central_direction_evidence_sha256": sha256_file(evidence_path),
    }
    write_json_atomic(output_dir / "training_view_summary.json", summary)
    return summary
