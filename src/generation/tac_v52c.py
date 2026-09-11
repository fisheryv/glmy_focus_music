"""Stage-1 local response-curve audit for the TAC route."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import canonical_json_sha256, write_csv_atomic, write_json_atomic
from .ltsn_v52a import orthogonal_smooth_directions
from .path_homology_exact_scorer import ExactPathHomologyScorer
from .tac_target import TACTopologyTarget

V52C_RADII = (0.00125, 0.0025, 0.00375, 0.005, 0.0075)
V52C_NEW_RADII = (0.00125, 0.00375, 0.0075)
V52C_REFERENCE_RADIUS = 0.005
V52C_EXACT_MARGIN = 1e-8
V52C_KIND = "tac_v52c_local_response_curve"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def _bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary, path)


def _response_row(
    *,
    sample_id: str,
    anchor_id: str,
    direction_index: int,
    step_number: int,
    radius: float,
    sign: float,
    coordinates: Sequence[float],
    ood_label: Any,
    source: str,
    target: TACTopologyTarget,
) -> dict[str, Any]:
    matrix = np.asarray(coordinates, dtype=np.float64).reshape(1, 18)
    blocks = target.block_distances(matrix)
    return {
        "sample_id": sample_id,
        "anchor_sample_id": anchor_id,
        "direction_index": direction_index,
        "step_number": step_number,
        "rms_ratio": radius,
        "sign": sign,
        "target_distance": float(target.distance(matrix)[0]),
        "pitch_distance": float(blocks["pitch"][0]),
        "path_acoustic_phase_distance": float(blocks["path_acoustic_phase"][0]),
        "path_chroma_phase_distance": float(blocks["path_chroma_phase"][0]),
        "coordinates_json": json.dumps(list(coordinates), separators=(",", ":")),
        "ood_label": str(bool(float(ood_label))).lower(),
        "source": source,
    }


def _validate_completed_batch(
    batch_dir: Path, *, plan_sha256: str, expected_ids: list[str]
) -> list[dict[str, str]] | None:
    receipt_path = batch_dir / "completion.json"
    descriptor_path = batch_dir / "descriptors.csv"
    if not receipt_path.exists() and not descriptor_path.exists():
        return None
    if descriptor_path.is_file() and not receipt_path.exists():
        rows = read_csv(descriptor_path)
        if sorted(row["sample_id"] for row in rows) != sorted(expected_ids):
            raise LTSNContractError(f"incomplete V5.2c descriptor checkpoint: {batch_dir}")
        audio_dir = batch_dir / "audio"
        for audio_path in audio_dir.glob("*.wav") if audio_dir.is_dir() else ():
            audio_path.unlink()
        write_json_atomic(
            receipt_path,
            {
                "schema_version": 1,
                "plan_sha256": plan_sha256,
                "sample_ids": expected_ids,
                "descriptor_sha256": sha256_file(descriptor_path),
                "wav_retained": 0,
                "recovered_after_descriptor_commit": True,
            },
        )
    if not receipt_path.is_file() or not descriptor_path.is_file():
        raise LTSNContractError(f"incomplete V5.2c batch checkpoint: {batch_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("plan_sha256") != plan_sha256
        or receipt.get("descriptor_sha256") != sha256_file(descriptor_path)
        or receipt.get("sample_ids") != expected_ids
        or receipt.get("wav_retained") != 0
    ):
        raise LTSNContractError(f"mismatched V5.2c batch checkpoint: {batch_dir}")
    rows = read_csv(descriptor_path)
    if sorted(row["sample_id"] for row in rows) != sorted(expected_ids):
        raise LTSNContractError(f"V5.2c descriptor identities changed: {batch_dir}")
    if any(batch_dir.joinpath("audio", f"{sample_id}.wav").exists() for sample_id in expected_ids):
        raise LTSNContractError(f"completed V5.2c batch retained WAV: {batch_dir}")
    return rows


def collect_v52c_response_curve(
    *,
    root: Path,
    source_manifest_path: Path,
    v52a_plan_path: Path,
    v52a_master_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    target_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    workers: int = 8,
    batch_size: int = 64,
    materialize_mode: str = "auto",
    device_name: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    """Collect only the three new radii and delete decoded WAV batch-by-batch."""

    if workers <= 0 or batch_size <= 0:
        raise ValueError("workers and batch_size must be positive")
    root = root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    source_manifest_path = resolve(source_manifest_path)
    v52a_plan_path = resolve(v52a_plan_path)
    v52a_master_path = resolve(v52a_master_path)
    ace_config_path = resolve(ace_config_path)
    fingerprint_path = resolve(fingerprint_path)
    target_path = resolve(target_path)
    output_dir = resolve(output_dir)
    source_rows = read_csv(source_manifest_path)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    if {row.get("ace_model_sha256", "") for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("V5.2c ACE model hash differs from source manifest")
    if {row.get("vae_sha256", "") for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("V5.2c VAE hash differs from source manifest")
    v52a_plan = json.loads(v52a_plan_path.read_text(encoding="utf-8"))
    if (
        v52a_plan.get("direction_count") != 8
        or v52a_plan.get("rms_ratios") != [0.0025, 0.005]
        or v52a_plan.get("diagnostic_only") is not True
    ):
        raise LTSNContractError("V5.2c requires the frozen V5.2a 8-direction plan")
    anchor_ids = list(v52a_plan.get("selection", {}).get("unseen_anchor_ids", []))
    if len(anchor_ids) != 16 or len(set(anchor_ids)) != 16:
        raise LTSNContractError("V5.2c requires exactly 16 V5.2a unseen anchors")
    missing = set(anchor_ids) - set(source_by_id)
    if missing:
        raise LTSNContractError(f"V5.2c source anchors are missing: {sorted(missing)}")
    target = TACTopologyTarget.from_json(target_path, root=root)
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    contract = load_fingerprint_contract(fingerprint_path)
    if target.payload["fingerprint_sha256"] != contract.artifact_sha256:
        raise LTSNContractError("V5.2c target is bound to a different fingerprint")

    anchors: dict[str, dict[str, Any]] = {}
    planned: list[dict[str, Any]] = []
    direction_seed = int(v52a_plan["seed"])
    for anchor_id in anchor_ids:
        source = source_by_id[anchor_id]
        latent_path = (source_manifest_path.parent / source["latent_path"]).resolve()
        if not latent_path.is_file() or sha256_file(latent_path) != source["latent_sha256"]:
            raise LTSNContractError(f"V5.2c anchor latent is missing: {anchor_id}")
        anchors[anchor_id] = {**source, "_latent_path": str(latent_path)}
        for direction_index in range(8):
            expected_seed = int(
                v52a_plan["direction_audit"][anchor_id]["direction_seeds"][direction_index]
            )
            for radius in V52C_NEW_RADII:
                radius_tag = int(round(radius * 1_000_000))
                for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                    sample_id = (
                        f"{anchor_id}__v52c_d{direction_index:02d}_r{radius_tag:05d}__{sign_name}"
                    )
                    planned.append(
                        {
                            "sample_id": sample_id,
                            "anchor_sample_id": anchor_id,
                            "direction_index": direction_index,
                            "direction_seed": expected_seed,
                            "rms_ratio": radius,
                            "sign": sign,
                            "step_number": int(source["step_number"]),
                        }
                    )
    planned.sort(key=lambda row: row["sample_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "experiment": "tac_v5_2c_local_response_curve",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "v52a_plan_sha256": sha256_file(v52a_plan_path),
        "v52a_master_sha256": sha256_file(v52a_master_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "anchor_ids": anchor_ids,
        "direction_count": 8,
        "reused_radii": [0.0025, 0.005],
        "new_radii": list(V52C_NEW_RADII),
        "all_radii": list(V52C_RADII),
        "planned_new_samples": len(planned),
        "wav_policy": "ephemeral_delete_after_each_exact_batch",
        "planned": planned,
    }
    plan_path = output_dir / "v52c_generation_plan.json"
    if plan_path.is_file():
        if not resume or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("V5.2c plan changed; use a new output directory")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)

    from .ace_adapter import AceStepAdapter
    from .experiment import load_experiment_config

    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    all_descriptors: list[dict[str, str]] = []
    batches_root = output_dir / "batches"
    direction_cache: dict[str, tuple[np.ndarray, list[np.ndarray], dict[str, Any]]] = {}
    for batch_index, start in enumerate(range(0, len(planned), batch_size)):
        items = planned[start : start + batch_size]
        expected_ids = [str(item["sample_id"]) for item in items]
        batch_dir = batches_root / f"batch_{batch_index:05d}"
        completed = _validate_completed_batch(
            batch_dir, plan_sha256=plan_sha256, expected_ids=expected_ids
        )
        if completed is not None:
            all_descriptors.extend(completed)
            continue
        batch_dir.mkdir(parents=True, exist_ok=True)
        trajectory_rows = []
        audio_paths: list[Path] = []
        for item in items:
            anchor = anchors[str(item["anchor_sample_id"])]
            anchor_id = str(item["anchor_sample_id"])
            if anchor_id not in direction_cache:
                latent = np.load(anchor["_latent_path"], allow_pickle=False).astype(
                    np.float32, copy=False
                )
                directions, audit = orthogonal_smooth_directions(
                    latent, anchor_id=anchor_id, count=8, seed=direction_seed
                )
                direction_cache[anchor_id] = (latent, directions, audit)
            latent, directions, audit = direction_cache[anchor_id]
            direction_index = int(item["direction_index"])
            if int(audit["direction_seeds"][direction_index]) != int(item["direction_seed"]):
                raise LTSNContractError("V5.2c reconstructed direction seed changed")
            latent_rms = float(np.sqrt(np.mean(np.square(latent), dtype=np.float64)))
            augmented = latent + (
                float(item["sign"])
                * float(item["rms_ratio"])
                * latent_rms
                * directions[direction_index]
            )
            latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
            if latent_path.is_file():
                existing = np.load(latent_path, allow_pickle=False)
                if existing.dtype != np.float32 or not np.array_equal(
                    existing, augmented.astype(np.float32)
                ):
                    raise LTSNContractError("V5.2c resumed latent is mismatched")
            else:
                _save_npy_atomic(latent_path, augmented)
            audio_path = batch_dir / "audio" / f"{item['sample_id']}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.unlink(missing_ok=True)
            adapter.decode_latent_to_audio(augmented.astype(np.float32), audio_path)
            audio_paths.append(audio_path)
            trajectory_rows.append(
                {
                    "sample_id": item["sample_id"],
                    "prompt_id": anchor["prompt_id"],
                    "trajectory_id": item["sample_id"],
                    "split": "development",
                    "model_family": anchor["model_family"],
                    "step_number": anchor["step_number"],
                    "timestep": anchor["timestep"],
                    "latent_path": os.path.relpath(latent_path, batch_dir).replace("\\", "/"),
                    "latent_sha256": sha256_file(latent_path),
                    "audio_path": os.path.relpath(audio_path, batch_dir).replace("\\", "/"),
                    "audio_sha256": sha256_file(audio_path),
                    "is_final": "false",
                    "ace_model_sha256": ace_model_sha256,
                    "vae_sha256": vae_sha256,
                    "training_augmentation_kind": V52C_KIND,
                    "local_anchor_sample_id": "",
                }
            )
        trajectory_path = batch_dir / "trajectories.csv"
        write_csv_atomic(trajectory_path, trajectory_rows)
        descriptor_path = batch_dir / "descriptors.csv"
        try:
            build_exact_snapshot_descriptors(
                project_root=root,
                trajectory_manifest=trajectory_path,
                work_dir=batch_dir / "exact_work",
                output_path=descriptor_path,
                workers=workers,
                batch_size=len(items),
                materialize_mode=materialize_mode,
                cleanup_batches=True,
                resume=resume,
            )
            descriptors = read_csv(descriptor_path)
            if sorted(row["sample_id"] for row in descriptors) != sorted(expected_ids):
                raise LTSNContractError("V5.2c exact batch identities changed")
        finally:
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)
        write_json_atomic(
            batch_dir / "completion.json",
            {
                "schema_version": 1,
                "plan_sha256": plan_sha256,
                "sample_ids": expected_ids,
                "descriptor_sha256": sha256_file(descriptor_path),
                "wav_retained": 0,
            },
        )
        all_descriptors.extend(descriptors)

    if len(all_descriptors) != len(planned):
        raise LTSNContractError("V5.2c did not produce every new exact descriptor")
    descriptor_by_id = {row["sample_id"]: row for row in all_descriptors}
    response_rows: list[dict[str, Any]] = []
    for anchor_id in anchor_ids:
        anchor = anchors[anchor_id]
        response_rows.append(
            _response_row(
                sample_id=anchor_id,
                anchor_id=anchor_id,
                direction_index=-1,
                step_number=int(anchor["step_number"]),
                radius=0.0,
                sign=0.0,
                coordinates=json.loads(anchor["coordinates_json"]),
                ood_label=anchor["ood_label"],
                source="source_anchor",
                target=target,
            )
        )
    v52a_rows = read_csv(v52a_master_path)
    reused = 0
    for row in v52a_rows:
        anchor_id = row.get("anchor_sample_id") or row.get("v52a_anchor_sample_id", "")
        if not anchor_id:
            sample_id = row["sample_id"]
            anchor_id = next((value for value in anchor_ids if sample_id.startswith(value)), "")
        if anchor_id not in anchors or row.get("v52a_anchor_partition") != "unseen_anchor":
            continue
        radius = float(row["local_direction_rms_ratio"])
        if radius not in {0.0025, 0.005}:
            continue
        response_rows.append(
            _response_row(
                sample_id=row["sample_id"],
                anchor_id=anchor_id,
                direction_index=int(row["v52a_direction_index"]),
                step_number=int(row["step_number"]),
                radius=radius,
                sign=float(row["local_direction_sign"]),
                coordinates=json.loads(row["coordinates_json"]),
                ood_label=row["ood_label"],
                source="reused_v52a_exact",
                target=target,
            )
        )
        reused += 1
    if reused != 16 * 8 * 2 * 2:
        raise LTSNContractError(f"expected 512 reused V5.2a points, observed {reused}")
    for item in planned:
        descriptor = descriptor_by_id[str(item["sample_id"])]
        score = scorer.score(
            json.loads(descriptor["pitch_descriptors_json"]),
            [float(descriptor["acoustic_loop_score"])],
            [float(descriptor["chroma_loop_score"])],
        )
        response_rows.append(
            _response_row(
                sample_id=str(item["sample_id"]),
                anchor_id=str(item["anchor_sample_id"]),
                direction_index=int(item["direction_index"]),
                step_number=int(item["step_number"]),
                radius=float(item["rms_ratio"]),
                sign=float(item["sign"]),
                coordinates=score.coordinates[0].tolist(),
                ood_label=descriptor["ood_label"],
                source="new_ephemeral_wav_exact",
                target=target,
            )
        )
    response_rows.sort(
        key=lambda row: (
            row["anchor_sample_id"],
            int(row["direction_index"]),
            float(row["rms_ratio"]),
            float(row["sign"]),
        )
    )
    response_path = output_dir / "v52c_response_points.csv"
    write_csv_atomic(response_path, response_rows)
    report, outcomes = analyze_response_points(response_rows)
    outcomes_path = output_dir / "v52c_response_outcomes.csv"
    write_csv_atomic(outcomes_path, outcomes)
    report.update(
        {
            "generation_plan_sha256": plan_sha256,
            "target_sha256": sha256_file(target_path),
            "response_points_sha256": sha256_file(response_path),
            "response_outcomes_sha256": sha256_file(outcomes_path),
            "new_samples": len(planned),
            "reused_v52a_samples": reused,
            "retained_wav_files": len(list(output_dir.rglob("*.wav"))),
            "retained_latent_files": len(list((output_dir / "latents").glob("*.npy"))),
            "configuration_sha256": canonical_json_sha256(
                {
                    "anchors": 16,
                    "directions": 8,
                    "new_radii": list(V52C_NEW_RADII),
                    "batch_size": batch_size,
                }
            ),
        }
    )
    if report["retained_wav_files"] != 0:
        raise LTSNContractError("V5.2c ephemeral WAV cleanup failed")
    report_path = output_dir / "v52c_response_report.json"
    write_json_atomic(report_path, report)
    return report


def analyze_response_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    exact_margin: float = V52C_EXACT_MARGIN,
    reference_radius: float = V52C_REFERENCE_RADIUS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute pre-registered cross-radius stability and curvature diagnostics."""

    centers: dict[str, Mapping[str, Any]] = {}
    pairs: dict[tuple[str, int, float], dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        anchor_id = str(row["anchor_sample_id"])
        radius = float(row["rms_ratio"])
        if radius == 0.0:
            centers[anchor_id] = row
            continue
        key = (anchor_id, int(row["direction_index"]), radius)
        sign = float(row["sign"])
        if sign in pairs[key]:
            raise LTSNContractError(f"duplicate V5.2c response point: {key}/{sign}")
        pairs[key][sign] = row
    if not centers or not pairs:
        raise LTSNContractError("V5.2c response table has no centers or pairs")

    outcomes: list[dict[str, Any]] = []
    by_direction: dict[tuple[str, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    block_names = ("pitch", "path_acoustic_phase", "path_chroma_phase")
    block_weights = (0.5, 0.25, 0.25)
    for (anchor_id, direction_index, radius), signed in sorted(pairs.items()):
        if set(signed) != {-1.0, 1.0}:
            raise LTSNContractError(
                f"incomplete V5.2c pair: {anchor_id}/{direction_index}/{radius}"
            )
        if anchor_id not in centers:
            raise LTSNContractError(f"missing V5.2c center: {anchor_id}")
        minus, plus = signed[-1.0], signed[1.0]
        center = centers[anchor_id]
        minus_distance = float(minus["target_distance"])
        plus_distance = float(plus["target_distance"])
        center_distance = float(center["target_distance"])
        odd = (plus_distance - minus_distance) / 2.0
        even = (plus_distance + minus_distance) / 2.0 - center_distance
        response = (minus_distance - plus_distance) / (2.0 * radius)
        block_response: dict[str, float] = {}
        block_odd: dict[str, float] = {}
        for name, weight in zip(block_names, block_weights, strict=True):
            minus_block = float(minus[f"{name}_distance"])
            plus_block = float(plus[f"{name}_distance"])
            block_response[name] = weight * (minus_block - plus_block) / (2.0 * radius)
            block_odd[name] = weight * (plus_block - minus_block) / 2.0
        outcome = {
            "anchor_sample_id": anchor_id,
            "direction_index": direction_index,
            "step_number": int(minus["step_number"]),
            "rms_ratio": radius,
            "minus_sample_id": minus["sample_id"],
            "plus_sample_id": plus["sample_id"],
            "pair_in_distribution": not _bool(minus["ood_label"]) and not _bool(plus["ood_label"]),
            "target_distance_minus": minus_distance,
            "target_distance_plus": plus_distance,
            "target_response": response,
            "response_sign": int(np.sign(response)) if abs(response) >= exact_margin else 0,
            "odd_component": odd,
            "even_component": even,
            "absolute_even_to_odd_ratio": abs(even) / max(abs(odd), exact_margin),
            **{f"{name}_response": value for name, value in block_response.items()},
            **{f"{name}_odd_component": value for name, value in block_odd.items()},
        }
        outcomes.append(outcome)
        by_direction[(anchor_id, direction_index)][radius] = outcome

    expected_radii = set(V52C_RADII)
    comparisons = 0
    successes = 0
    reference_ties = 0
    tested_ties = 0
    direction_monotonic = 0
    direction_count = 0
    for key, radius_rows in sorted(by_direction.items()):
        if set(radius_rows) != expected_radii:
            raise LTSNContractError(f"V5.2c radius grid changed for {key}: {sorted(radius_rows)}")
        reference_sign = int(radius_rows[reference_radius]["response_sign"])
        if reference_sign == 0:
            reference_ties += 1
        absolute_responses = [abs(float(radius_rows[r]["target_response"])) for r in V52C_RADII]
        direction_monotonic += all(
            right + exact_margin >= left
            for left, right in zip(absolute_responses, absolute_responses[1:], strict=False)
        )
        direction_count += 1
        for radius in V52C_RADII:
            if radius == reference_radius:
                continue
            comparisons += 1
            tested_sign = int(radius_rows[radius]["response_sign"])
            tested_ties += tested_sign == 0
            successes += reference_sign != 0 and tested_sign == reference_sign
    interval = wilson_interval(successes, comparisons)
    agreement = successes / comparisons
    gate = agreement >= 0.80 and interval[0] > 0.50

    weighted_attribution = {}
    absolute_total = sum(abs(float(row["target_response"])) for row in outcomes)
    for name in block_names:
        absolute = sum(abs(float(row[f"{name}_response"])) for row in outcomes)
        weighted_attribution[name] = {
            "absolute_response_sum": absolute,
            "fraction_of_total_absolute_response": absolute / max(absolute_total, exact_margin),
        }
    report = {
        "schema_version": 1,
        "experiment": "tac_v5_2c_local_response_curve",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "radii": list(V52C_RADII),
        "reference_radius": reference_radius,
        "exact_margin": exact_margin,
        "anchors": len(centers),
        "directions": direction_count,
        "pairs": len(outcomes),
        "cross_radius_sign": {
            "successes": successes,
            "comparisons": comparisons,
            "agreement": agreement,
            "wilson95": interval,
            "reference_ties": reference_ties,
            "tested_ties": tested_ties,
        },
        "response_magnitude_monotonicity": {
            "directions_monotonic": direction_monotonic,
            "directions_total": direction_count,
            "fraction": direction_monotonic / direction_count,
            "role": "descriptive_not_gating",
        },
        "weighted_block_attribution": weighted_attribution,
        "gates": {
            "cross_radius_sign_agreement_at_least_0_80": agreement >= 0.80,
            "cross_radius_wilson_lower_above_0_50": interval[0] > 0.50,
        },
        "stage1_status": "supported_for_critic_collection" if gate else "not_supported",
        "next_stage_authorized": gate,
    }
    return report, outcomes


def write_response_report(*, response_points_path: Path, output_path: Path) -> dict[str, Any]:
    rows = read_csv(response_points_path)
    report, outcomes = analyze_response_points(rows)
    outcomes_path = output_path.with_name("v52c_response_outcomes.csv")
    write_csv_atomic(outcomes_path, outcomes)
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        expected = previous.get("response_points_sha256")
        if expected is not None and expected != sha256_file(response_points_path):
            raise LTSNContractError("V5.2c response points changed after collection")
        report = {**previous, **report}
    report["response_points"] = str(response_points_path.resolve())
    report["response_outcomes"] = str(outcomes_path.resolve())
    report["response_points_sha256"] = sha256_file(response_points_path)
    report["response_outcomes_sha256"] = sha256_file(outcomes_path)
    write_json_atomic(output_path, report)
    return report
