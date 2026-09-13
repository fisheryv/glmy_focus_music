"""No-new-audio TAC pair preparation for the V6.1 multitask screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .tac_target import TACTopologyTarget, robust_center_scale

V61_EXPERIMENT = "ltsn_v6_1_final_target_derivative_multitask_screen"
V61_PAIR_SPLITS = (
    "direction_fit",
    "direction_validation",
    "heldout_seen_anchor",
    "heldout_unseen_anchor",
)
_EVALUATION_TO_V61 = {
    "seen_anchor_heldout_direction": "heldout_seen_anchor",
    "unseen_anchor": "heldout_unseen_anchor",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"empty V6.1 input table: {path}")
    return rows


def _validation_anchors(
    train_pairs: list[dict[str, str]], *, seed: int, per_step: int
) -> set[str]:
    by_step: dict[int, set[str]] = defaultdict(set)
    for row in train_pairs:
        by_step[int(row["step_number"])].add(row["anchor_sample_id"])
    if set(by_step) != {4, 5, 6}:
        raise LTSNContractError("V6.1 train-direction anchors must cover steps 4/5/6")
    selected: set[str] = set()
    for step, anchors in sorted(by_step.items()):
        ordered = sorted(
            anchors,
            key=lambda value: hashlib.sha256(f"{seed}:{step}:{value}".encode()).hexdigest(),
        )
        if len(ordered) <= per_step:
            raise LTSNContractError("V6.1 lacks anchors for a disjoint validation split")
        selected.update(ordered[:per_step])
    return selected


def prepare_v61_pair_view(
    *,
    fingerprint_path: Path,
    tac_target_path: Path,
    v6_preparation_path: Path,
    v6_view_path: Path,
    v6_checkpoint_path: Path,
    pair_manifest_path: Path,
    coordinate_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    validation_anchors_per_step: int = 2,
    split_seed: int = 20260914,
) -> dict[str, Any]:
    """Build TAC derivative targets from existing coordinates and antithetic latents."""

    fingerprint_path = fingerprint_path.resolve()
    tac_target_path = tac_target_path.resolve()
    v6_preparation_path = v6_preparation_path.resolve()
    v6_view_path = v6_view_path.resolve()
    v6_checkpoint_path = v6_checkpoint_path.resolve()
    pair_manifest_path = pair_manifest_path.resolve()
    coordinate_manifest_path = coordinate_manifest_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    target = TACTopologyTarget.from_json(tac_target_path, verify_sources=False)
    if target.payload.get("source_sha256", {}).get(
        "metadata/focus_path_homology_fingerprint_v2.json"
    ) != sha256_file(fingerprint_path):
        raise LTSNContractError("V6.1 TAC target uses a different fingerprint")
    v6_preparation = json.loads(v6_preparation_path.read_text(encoding="utf-8"))
    expected_v6 = {
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "view_sha256": sha256_file(v6_view_path),
    }
    if (
        v6_preparation.get("experiment") != "ltsn_v6_final_target_screen"
        or v6_preparation.get("diagnostic_only") is not True
        or v6_preparation.get("new_audio_files") != 0
    ):
        raise LTSNContractError("V6.1 requires a bounded V6 preparation")
    for name, expected in expected_v6.items():
        if v6_preparation.get(name) != expected:
            raise LTSNContractError(f"V6.1 V6 preparation {name} mismatch")

    pair_rows = _read_csv(pair_manifest_path)
    coordinate_rows = _read_csv(coordinate_manifest_path)
    coordinates_by_id = {row["sample_id"]: row for row in coordinate_rows}
    if len(coordinates_by_id) != len(coordinate_rows):
        raise LTSNContractError("V6.1 coordinate manifest contains duplicate sample IDs")
    operational = [
        row
        for row in pair_rows
        if row["evaluation_split"]
        in {"train", "seen_anchor_heldout_direction", "unseen_anchor"}
    ]
    train_pairs = [row for row in operational if row["evaluation_split"] == "train"]
    validation_anchors = _validation_anchors(
        train_pairs, seed=split_seed, per_step=validation_anchors_per_step
    )
    output_rows: list[dict[str, Any]] = []
    fingerprint_hashes: set[str] = set()
    for pair in sorted(operational, key=lambda row: row["pair_id"]):
        minus = coordinates_by_id.get(pair["minus_sample_id"])
        plus = coordinates_by_id.get(pair["plus_sample_id"])
        if minus is None or plus is None:
            raise LTSNContractError(f"V6.1 pair coordinates are missing: {pair['pair_id']}")
        if (
            minus.get("latent_sha256") != pair.get("minus_latent_sha256")
            or plus.get("latent_sha256") != pair.get("plus_latent_sha256")
        ):
            raise LTSNContractError(f"V6.1 pair latent hash mismatch: {pair['pair_id']}")
        fingerprint_hashes.update(
            (minus.get("fingerprint_json_sha256", ""), plus.get("fingerprint_json_sha256", ""))
        )
        evaluation_split = pair["evaluation_split"]
        if evaluation_split == "train":
            split = (
                "direction_validation"
                if pair["anchor_sample_id"] in validation_anchors
                else "direction_fit"
            )
        else:
            split = _EVALUATION_TO_V61[evaluation_split]
        minus_coordinates = json.loads(minus["coordinates_json"])
        plus_coordinates = json.loads(plus["coordinates_json"])
        minus_blocks = target.block_distances(minus_coordinates)
        plus_blocks = target.block_distances(plus_coordinates)
        rms = float(pair["rms_ratio"])
        minus_distance = float(target.distance(minus_coordinates)[0])
        plus_distance = float(target.distance(plus_coordinates)[0])
        derivative = (minus_distance - plus_distance) / (2.0 * rms)
        if not math.isfinite(derivative) or abs(derivative) <= 1e-12:
            raise LTSNContractError(f"V6.1 TAC derivative is tied or non-finite: {pair['pair_id']}")
        minus_path = (pair_manifest_path.parent / pair["minus_latent_path"]).resolve()
        plus_path = (pair_manifest_path.parent / pair["plus_latent_path"]).resolve()
        output_rows.append(
            {
                "pair_id": pair["pair_id"],
                "anchor_sample_id": pair["anchor_sample_id"],
                "prompt_id": pair["prompt_id"],
                "v61_split": split,
                "source_evaluation_split": evaluation_split,
                "step_number": int(pair["step_number"]),
                "timestep": pair["timestep"],
                "rms_ratio": pair["rms_ratio"],
                "minus_latent_path": os.path.relpath(minus_path, output_dir),
                "plus_latent_path": os.path.relpath(plus_path, output_dir),
                "minus_latent_sha256": pair["minus_latent_sha256"],
                "plus_latent_sha256": pair["plus_latent_sha256"],
                "exact_tac_distance_minus": format(minus_distance, ".17g"),
                "exact_tac_distance_plus": format(plus_distance, ".17g"),
                "exact_tac_derivative": format(derivative, ".17g"),
                "exact_pitch_derivative": format(
                    (float(minus_blocks["pitch"][0]) - float(plus_blocks["pitch"][0]))
                    / (2.0 * rms),
                    ".17g",
                ),
                "exact_acoustic_phase_derivative": format(
                    (
                        float(minus_blocks["path_acoustic_phase"][0])
                        - float(plus_blocks["path_acoustic_phase"][0])
                    )
                    / (2.0 * rms),
                    ".17g",
                ),
                "exact_chroma_phase_derivative": format(
                    (
                        float(minus_blocks["path_chroma_phase"][0])
                        - float(plus_blocks["path_chroma_phase"][0])
                    )
                    / (2.0 * rms),
                    ".17g",
                ),
            }
        )
    if fingerprint_hashes != {sha256_file(fingerprint_path)}:
        raise LTSNContractError("V6.1 coordinate manifest uses a different fingerprint")

    transforms: dict[str, dict[str, float]] = {}
    for step in (4, 5, 6):
        values = np.asarray(
            [
                [float(row["exact_tac_derivative"])]
                for row in output_rows
                if row["v61_split"] == "direction_fit" and row["step_number"] == step
            ],
            dtype=np.float64,
        )
        center, scale = robust_center_scale(values)
        transforms[str(step)] = {"center": float(center[0]), "scale": float(scale[0])}
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_view_path = output_dir / "v61_tac_pair_targets.csv"
    write_csv_atomic(pair_view_path, output_rows)
    counts = {
        split: sum(row["v61_split"] == split for row in output_rows)
        for split in V61_PAIR_SPLITS
    }
    payload = {
        "schema_version": 1,
        "experiment": V61_EXPERIMENT,
        "mode": "diagnostic_only",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "new_audio_files": 0,
        "copied_latent_files": 0,
        "reused_pair_latents": len(output_rows) * 2,
        "pair_counts": counts,
        "validation_anchors_per_step": validation_anchors_per_step,
        "validation_anchors": sorted(validation_anchors),
        "split_seed": split_seed,
        "direction_target": "tac_topology_distance_v1",
        "direction_formula": "(D_TAC(minus)-D_TAC(plus))/(2*rms_ratio)",
        "direction_transforms_by_step": transforms,
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "v6_preparation_sha256": sha256_file(v6_preparation_path),
        "v6_view_sha256": sha256_file(v6_view_path),
        "v6_checkpoint_sha256": sha256_file(v6_checkpoint_path),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "coordinate_manifest_sha256": sha256_file(coordinate_manifest_path),
        "config_sha256": sha256_file(config_path),
        "pair_view": str(pair_view_path),
        "pair_view_sha256": sha256_file(pair_view_path),
    }
    preparation_path = output_dir / "v61_preparation.json"
    write_json_atomic(preparation_path, payload)
    payload["preparation_sha256"] = sha256_file(preparation_path)
    return payload
