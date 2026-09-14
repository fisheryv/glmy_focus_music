"""Exact-label manifest builder for the frozen Pitch-3 teacher."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import (
    read_trajectory_manifest,
    validate_snapshot_coverage,
    write_csv_atomic,
    write_json_atomic,
)
from .pitch3_exact_scorer import ExactPitch3Scorer


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise LTSNContractError(f"{name} must be finite")
    return number


def build_pitch3_label_tables(
    *,
    trajectory_manifest: Path,
    descriptor_table: Path,
    output_manifest: Path,
    exact_label_table: Path,
    split_manifest: Path,
    scorer: ExactPitch3Scorer,
    engineering_smoke: bool,
) -> dict[str, Any]:
    """Join decoded per-snapshot Pitch descriptors with trajectory latents."""

    trajectories = read_trajectory_manifest(trajectory_manifest)
    if not engineering_smoke:
        validate_snapshot_coverage(trajectories)
    with descriptor_table.open("r", encoding="utf-8-sig", newline="") as handle:
        descriptors = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if len(descriptors) != len(trajectories):
        raise LTSNContractError("Pitch-3 descriptors must contain one row per snapshot")
    label_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        sample_id = trajectory["sample_id"]
        descriptor = descriptors.get(sample_id)
        if descriptor is None:
            raise LTSNContractError(f"missing Pitch-3 descriptor row for {sample_id}")
        if not engineering_smoke:
            if descriptor.get("label_source") != "decoded_snapshot_exact_v1":
                raise LTSNContractError(
                    "formal Pitch-3 labels require decoded_snapshot_exact_v1"
                )
            if descriptor.get("audio_sha256") != trajectory.get("audio_sha256"):
                raise LTSNContractError("Pitch-3 descriptor audio hash mismatch")
        pitch = json.loads(descriptor["pitch_descriptors_json"])
        score = scorer.score(pitch)
        coordinates = score.coordinates[0].tolist()
        ood_label = _finite(descriptor.get("ood_label", 0.0), "ood_label")
        if ood_label not in {0.0, 1.0}:
            raise LTSNContractError("Pitch-3 ood_label must be zero or one")
        label_rows.append(
            {
                "sample_id": sample_id,
                "pitch_descriptors_json": json.dumps(pitch, separators=(",", ":")),
                "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                "focus_logit": float(score.focus_logit[0]),
                "focus_probability": float(score.focus_probability[0]),
                "focus_target_band_loss": float(score.target_band_loss[0]),
                "target_center_distance": float(score.target_center_distance[0]),
                "ood_label": ood_label,
                "label_scope": "per_snapshot_exact",
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
            }
        )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": trajectory["prompt_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "split": trajectory["split"],
                "model_family": trajectory["model_family"],
                "step_number": int(trajectory["step_number"]),
                "timestep": float(trajectory["timestep"]),
                "latent_path": os.path.relpath(
                    (trajectory_manifest.parent / trajectory["latent_path"]).resolve(),
                    output_manifest.parent.resolve(),
                ).replace("\\", "/"),
                "latent_sha256": trajectory["latent_sha256"],
                "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                "focus_logit": float(score.focus_logit[0]),
                "ood_label": ood_label,
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "feature_order_json": json.dumps(list(scorer.contract.feature_order)),
                "label_scope": "per_snapshot_exact",
                "exact_label_table_sha256": "",
                "exact_label_table_path": os.path.relpath(
                    exact_label_table.resolve(), output_manifest.parent.resolve()
                ).replace("\\", "/"),
                "is_final": trajectory["is_final"],
                "ace_model_sha256": trajectory["ace_model_sha256"],
                "vae_sha256": trajectory["vae_sha256"],
                "qualification_eligible": str(not engineering_smoke).lower(),
                "guidance_promotion_eligible": "false",
            }
        )
    write_csv_atomic(exact_label_table, label_rows)
    label_sha256 = sha256_file(exact_label_table)
    for row in manifest_rows:
        row["exact_label_table_sha256"] = label_sha256
    split_payload = {
        "schema_version": 1,
        "fingerprint_json_sha256": scorer.contract.artifact_sha256,
        "grouping": "prompt_id+trajectory_id",
        "assignments": dict(
            sorted({row["prompt_id"]: row["split"] for row in manifest_rows}.items())
        ),
        "qualification_eligible": not engineering_smoke,
        "guidance_promotion_eligible": False,
    }
    write_json_atomic(split_manifest, split_payload)
    write_csv_atomic(output_manifest, manifest_rows)
    return {
        "samples": len(manifest_rows),
        "prompts": len({row["prompt_id"] for row in manifest_rows}),
        "fingerprint_json_sha256": scorer.contract.artifact_sha256,
        "exact_label_table_sha256": label_sha256,
        "qualification_eligible": not engineering_smoke,
        "guidance_promotion_eligible": False,
    }
