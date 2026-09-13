"""Native-duration causal diagnostic for latent topology response stability.

This module intentionally cannot authorize LTSN qualification or sampling
guidance.  It compares matched 180/60/30 s generations and keeps every
duration's exact scorer/target identity explicit.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from features.batch import SegmentJob, extract_batch
from features.batch import _load_config as load_feature_config
from topology.metrics import TOPOLOGY_METRICS
from topology.multiview_fusion import DiscoveryMahalanobisBlock

from .exact_features import _copy_frozen_model
from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_exact_labeling import (
    _compute_exact_descriptor,
    _initialize_exact_descriptor_worker,
    build_exact_snapshot_descriptors,
)
from .ltsn_pipeline import canonical_json_sha256, write_csv_atomic, write_json_atomic
from .ltsn_storage import materialize_file
from .ltsn_v52a import orthogonal_smooth_directions
from .path_homology_exact_scorer import ExactPathHomologyScorer
from .tac_target import TACTopologyTarget, fit_target_payload
from .tac_v52c import wilson_interval

DURATIONS = (180, 60, 30)
ANCHOR_COUNT = 16
DIRECTION_COUNT = 4
RADII = (0.0025, 0.005, 0.0075)
REFERENCE_RADIUS = 0.005
ANCHOR_STEP = 5
EXACT_MARGIN = 1e-8
BOOTSTRAP_SEED = 20260912
PHASE_BLOCK_FRAMES = 2
PHASE_MIN_RAW_STEPS = 36
REPETITION_OVERRIDES = {
    "block_frames": PHASE_BLOCK_FRAMES,
    "min_raw_steps": PHASE_MIN_RAW_STEPS,
}
FEATURE_ORDER = [
    *[f"pitch_whitened_{index:02d}" for index in range(16)],
    "path_acoustic_phase__loop_score",
    "path_chroma_phase__loop_score",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mapping_sha256(values: Mapping[str, str]) -> str:
    return canonical_json_sha256(dict(sorted(values.items())))


def _classifier_sha256(coef: Sequence[float], intercept: float) -> str:
    return canonical_json_sha256({"coef": list(coef), "intercept": intercept})


def _serialize_block(
    transformer: DiscoveryMahalanobisBlock,
    input_features: list[str],
    fusion_scale: float,
) -> dict[str, Any]:
    if any(
        value is None
        for value in (
            transformer.imputer,
            transformer.keep,
            transformer.mean,
            transformer.whitening,
        )
    ):
        raise RuntimeError("duration block transformer is not fitted")
    assert transformer.imputer is not None
    assert transformer.keep is not None
    assert transformer.mean is not None
    assert transformer.whitening is not None
    return {
        "input_features": input_features,
        "imputer_median": [float(value) for value in transformer.imputer.statistics_],
        "keep_mask": [bool(value) for value in transformer.keep],
        "retained_mean": [float(value) for value in transformer.mean],
        "whitening": transformer.whitening.astype(float).tolist(),
        "effective_rank": int(transformer.effective_rank),
        "output_dimensions": int(transformer.whitening.shape[1]),
        "fusion_scale": float(fusion_scale),
    }


def prepare_duration_plan(
    *,
    prompt_manifest_path: Path,
    output_dir: Path,
    protocol_path: Path | None = None,
    seed_start: int = 2026091200,
) -> dict[str, Any]:
    """Freeze 16 development prompts, balanced four-per base family."""

    source = _read_csv(prompt_manifest_path)
    if protocol_path is not None:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = {
            "experiment": "ltsn_duration_causality_v1",
            "durations_seconds": list(DURATIONS),
            "anchor_step": ANCHOR_STEP,
            "anchors": ANCHOR_COUNT,
            "directions_per_anchor": DIRECTION_COUNT,
            "rms_ratios": list(RADII),
            "reference_rms_ratio": REFERENCE_RADIUS,
            "phase_block_frames": PHASE_BLOCK_FRAMES,
            "phase_min_raw_steps": PHASE_MIN_RAW_STEPS,
        }
        if any(protocol.get(name) != value for name, value in expected.items()):
            raise LTSNContractError("duration protocol differs from the implemented contract")
    development = [row for row in source if row.get("split") == "development"]
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in development:
        prompt_id = row["prompt_id"]
        family = prompt_id.split("__v", 1)[0]
        by_family[family].append(row)
    if len(by_family) != 4 or any(len(rows) < 4 for rows in by_family.values()):
        raise LTSNContractError(
            "duration diagnostic requires four development families with >=4 variants each"
        )
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        for row in sorted(by_family[family], key=lambda item: item["prompt_id"])[:4]:
            selected.append(
                {
                    **row,
                    "seed": seed_start + len(selected),
                    "duration_anchor_index": len(selected),
                    "base_family": family,
                }
            )
    if len(selected) != ANCHOR_COUNT:
        raise LTSNContractError("duration diagnostic prompt selection changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "duration_prompts.csv"
    write_csv_atomic(prompt_path, selected)
    plan = {
        "schema_version": 1,
        "experiment": "ltsn_duration_causality_v1",
        "mode": "development_only_diagnostic",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "native_duration_generation_required": True,
        "durations_seconds": list(DURATIONS),
        "anchor_step": ANCHOR_STEP,
        "anchors": ANCHOR_COUNT,
        "directions_per_anchor": DIRECTION_COUNT,
        "rms_ratios": list(RADII),
        "reference_rms_ratio": REFERENCE_RADIUS,
        "phase_block_frames": PHASE_BLOCK_FRAMES,
        "phase_min_raw_steps": PHASE_MIN_RAW_STEPS,
        "antithetic_signs": [-1, 1],
        "perturbation_decodes_per_duration": ANCHOR_COUNT
        * DIRECTION_COUNT
        * len(RADII)
        * 2,
        "center_decodes_per_duration": ANCHOR_COUNT,
        "prompt_source_sha256": sha256_file(prompt_manifest_path),
        "protocol_sha256": sha256_file(protocol_path) if protocol_path is not None else "",
        "duration_prompt_sha256": sha256_file(prompt_path),
        "seed_start": seed_start,
        "prompt_ids": [row["prompt_id"] for row in selected],
        "seeds": [int(row["seed"]) for row in selected],
        "gates": {
            "cross_radius_sign_agreement": 0.80,
            "wilson95_lower_strictly_above": 0.50,
            "minimum_absolute_improvement_over_180s": 0.10,
            "cluster_bootstrap_difference_lower_strictly_above": 0.0,
        },
    }
    plan_path = output_dir / "duration_causality_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise LTSNContractError("duration plan changed; use a new output directory")
    write_json_atomic(plan_path, plan)
    return {**plan, "plan_sha256": sha256_file(plan_path)}


def build_duration_reference_descriptors(
    *,
    project_root: Path,
    preprocess_manifest_path: Path,
    work_dir: Path,
    output_path: Path,
    duration_seconds: int,
    workers: int = 8,
    materialize_mode: str = "auto",
) -> dict[str, Any]:
    """Extract raw Pitch/phase descriptors from isolated native-duration references."""

    if duration_seconds not in DURATIONS:
        raise ValueError(f"native reference extraction is restricted to {DURATIONS}")
    rows = [
        row
        for row in _read_csv(preprocess_manifest_path)
        if row.get("status") != "failed"
        and math.isclose(float(row["scale_seconds"]), duration_seconds, abs_tol=1e-6)
    ]
    if len(rows) != 600:
        raise LTSNContractError(
            f"duration reference requires 600 rows at {duration_seconds}s; observed {len(rows)}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, tuple[Path, str]] = {}
    jobs: list[SegmentJob] = []
    for row in sorted(rows, key=lambda item: item["segment_id"]):
        source = (project_root / row["output_relative_path"]).resolve()
        expected = row["sha256"]
        target = work_dir / "data_raw" / "reference" / f"{row['segment_id']}.wav"
        materialize_file(source, target, mode=materialize_mode, expected_sha256=expected)
        copied[row["segment_id"]] = (target, expected)
        jobs.append(
            SegmentJob(
                segment_id=row["segment_id"],
                track_id=row["track_id"],
                group=row["group"],
                split=row["split"],
                scale_seconds=float(duration_seconds),
                input_relative_path=_relative(target, work_dir),
                input_sha256=expected,
            )
        )
    _copy_frozen_model(project_root, work_dir)
    feature_rows = extract_batch(
        jobs,
        root=work_dir,
        config=load_feature_config(project_root),
        workers=workers,
        overwrite=False,
        manifest_path=work_dir / "manifests" / "reference_features.csv",
    )
    failures = [row for row in feature_rows if row.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"duration reference feature extraction failed for {len(failures)} rows")
    insufficient = [
        row
        for row in feature_rows
        if int(row["acoustic_windows"]) < PHASE_MIN_RAW_STEPS
        or int(row["pitch_steps"]) < PHASE_MIN_RAW_STEPS
    ]
    if insufficient:
        raise LTSNContractError(
            f"{len(insufficient)} native references cannot support the frozen short-duration "
            "phase contract"
        )
    identity = {row["segment_id"]: row for row in rows}
    tasks = [
        (
            feature_row,
            {"audio_sha256": copied[str(feature_row["segment_id"])][1]},
            _relative(copied[str(feature_row["segment_id"])][0], work_dir),
        )
        for feature_row in feature_rows
    ]
    codebook_path = project_root / "features" / "models" / "pitch_v2_codebook.npz"
    with np.load(codebook_path, allow_pickle=False) as archive:
        centers = np.asarray(archive["centers"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        mp_context=context,
        initializer=_initialize_exact_descriptor_worker,
        initargs=(
            str(project_root),
            str(work_dir),
            centers,
            sha256_file(codebook_path),
            REPETITION_OVERRIDES,
        ),
    ) as executor:
        futures = {executor.submit(_compute_exact_descriptor, task): task for task in tasks}
        for future in as_completed(futures):
            raw = future.result()
            row = identity[str(raw["sample_id"])]
            output.append(
                {
                    "segment_id": raw["sample_id"],
                    "track_id": row["track_id"],
                    "group": row["group"],
                    "split": row["split"],
                    "scale_seconds": float(duration_seconds),
                    "pitch_descriptors_json": raw["pitch_descriptors_json"],
                    "acoustic_loop_score": raw["acoustic_loop_score"],
                    "chroma_loop_score": raw["chroma_loop_score"],
                    "ood_label": raw["ood_label"],
                    "audio_sha256": raw["audio_sha256"],
                    "pitch_v2_codebook_sha256": raw["pitch_v2_codebook_sha256"],
                }
            )
    output.sort(key=lambda item: item["segment_id"])
    write_csv_atomic(output_path, output)
    report = {
        "schema_version": 1,
        "duration_seconds": duration_seconds,
        "samples": len(output),
        "descriptor_sha256": sha256_file(output_path),
        "preprocess_manifest_sha256": sha256_file(preprocess_manifest_path),
        "diagnostic_only": True,
        "scientific_evidence": False,
    }
    write_json_atomic(output_path.with_suffix(".json"), report)
    return report


def build_duration_topology_contract(
    *,
    root: Path,
    descriptor_path: Path,
    output_dir: Path,
    duration_seconds: int,
    random_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fit a duration-native schema-v3 exact scorer and diagnostic TAC target."""

    if duration_seconds not in DURATIONS:
        raise ValueError(f"duration-native contracts are only built for {DURATIONS}")
    rows = _read_csv(descriptor_path)
    identity = rows
    pitch = np.asarray([json.loads(row["pitch_descriptors_json"]) for row in rows], dtype=float)
    acoustic = np.asarray([[float(row["acoustic_loop_score"])] for row in rows])
    chroma = np.asarray([[float(row["chroma_loop_score"])] for row in rows])
    discovery = np.asarray([row["split"] == "discovery" for row in rows])
    labels = np.asarray([row["group"] for row in rows])
    if pitch.shape != (600, len(TOPOLOGY_METRICS)) or discovery.sum() != 390:
        raise LTSNContractError("duration reference identity or discovery split changed")
    pitch_transform = DiscoveryMahalanobisBlock(output_dimensions=16).fit(pitch[discovery])
    acoustic_transform = DiscoveryMahalanobisBlock().fit(acoustic[discovery])
    chroma_transform = DiscoveryMahalanobisBlock().fit(chroma[discovery])
    coordinates = np.concatenate(
        (
            pitch_transform.transform(pitch) / math.sqrt(2.0),
            acoustic_transform.transform(acoustic) / 2.0,
            chroma_transform.transform(chroma) / 2.0,
        ),
        axis=1,
    )
    if coordinates.shape != (600, 18):
        raise LTSNContractError("duration-native scorer did not produce 18 coordinates")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=random_seed,
    ).fit(coordinates[discovery], labels[discovery])
    if classifier.classes_.tolist() != ["classical", "focus"]:
        raise LTSNContractError("duration classifier class order changed")
    logits = classifier.decision_function(coordinates)
    focus_reference = discovery & (labels == "focus")
    threshold = float(np.quantile(logits[focus_reference], 0.10))
    output_dir.mkdir(parents=True, exist_ok=True)
    module_path = Path(__file__).resolve()
    config_identity = {
        "experiment": "ltsn_duration_causality_v1",
        "duration_seconds": duration_seconds,
        "output_dimensions": 18,
        "distance_weights": [0.5, 0.25, 0.25],
        "classifier_c": 1.0,
        "focus_target_logit_quantile": 0.10,
        "seed": random_seed,
        "phase_block_frames": PHASE_BLOCK_FRAMES,
        "phase_min_raw_steps": PHASE_MIN_RAW_STEPS,
    }
    input_sources = {_relative(descriptor_path, root): sha256_file(descriptor_path)}
    coef = classifier.coef_[0].astype(float).tolist()
    intercept = float(classifier.intercept_[0])
    fingerprint = {
        "schema_version": 3,
        "fingerprint_id": "focus_path_homology_fingerprint_v2",
        "spec_revision": f"duration-causality-{duration_seconds}s-v1",
        "dimensions": 18,
        "feature_order": FEATURE_ORDER,
        "distance_weights": [0.5, 0.25, 0.25],
        "scope": "duration-native diagnostic Pitch 16-D plus Acoustic/Chroma phase",
        "reference_split": "discovery",
        "reference_scale_seconds": float(duration_seconds),
        "reference_sample_count": int(discovery.sum()),
        "reference_focus_count": int(focus_reference.sum()),
        "contains_tda_features": False,
        "block_transforms": {
            "pitch": _serialize_block(
                pitch_transform, list(TOPOLOGY_METRICS), 1.0 / math.sqrt(2.0)
            ),
            "path_acoustic_phase": _serialize_block(
                acoustic_transform, ["loop_score"], 0.5
            ),
            "path_chroma_phase": _serialize_block(chroma_transform, ["loop_score"], 0.5),
        },
        "fusion": {
            "formula": "concat(Pitch/sqrt(2), Acoustic/2, Chroma/2)",
            "squared_distance_weights": {
                "pitch": 0.5,
                "path_acoustic_phase": 0.25,
                "path_chroma_phase": 0.25,
            },
        },
        "classifier_coef": coef,
        "classifier_intercept": intercept,
        "classifier_sha256": _classifier_sha256(coef, intercept),
        "focus_band_threshold": threshold,
        "classifier": {
            "kind": "logistic_regression",
            "classes": classifier.classes_.tolist(),
            "positive_class": "focus",
            "c": 1.0,
            "decision_threshold_probability": 0.5,
            "focus_target_logit_quantile": 0.10,
            "control_loss": "max(0, focus_band_threshold - focus_logit)^2",
        },
        "input_sha256": _mapping_sha256(input_sources),
        "config_sha256": canonical_json_sha256(config_identity),
        "code_sha256": sha256_file(module_path),
        "source_sha256": {
            **input_sources,
            _relative(module_path, root): sha256_file(module_path),
        },
        "evidence": {
            "role": "development_duration_diagnostic",
            "scientific_evidence": False,
            "qualification_eligible": False,
            "guidance_promotion_eligible": False,
            "production_authorization": False,
        },
        "duration_native_analysis": {
            "phase_block_frames": PHASE_BLOCK_FRAMES,
            "phase_min_raw_steps": PHASE_MIN_RAW_STEPS,
            "same_across_all_durations": True,
        },
        "runtime_status": {
            "exact_scoring": "diagnostic_only",
            "sampling_guidance": "disabled",
        },
    }
    fingerprint_path = output_dir / "fingerprint.json"
    write_json_atomic(fingerprint_path, fingerprint)
    contract = load_fingerprint_contract(fingerprint_path)
    scores = []
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    rescored = scorer.score(pitch, acoustic, chroma)
    np.testing.assert_allclose(rescored.coordinates, coordinates, rtol=0.0, atol=1e-12)
    for index, row in enumerate(identity):
        scores.append(
            {
                "segment_id": row["segment_id"],
                "track_id": row["track_id"],
                "group": row["group"],
                "split": row["split"],
                "scale_seconds": float(duration_seconds),
                "coordinates_json": json.dumps(coordinates[index].tolist(), separators=(",", ":")),
                "focus_logit": float(logits[index]),
                "focus_band_loss": float(max(0.0, threshold - logits[index]) ** 2),
            }
        )
    scores_path = output_dir / "reference_scores.csv"
    write_csv_atomic(scores_path, scores)
    target_payload = fit_target_payload(
        coordinates[focus_reference],
        active_pitch_indices=tuple(range(3, 16)),
        acoustic_index=16,
        chroma_index=17,
        distance_weights=(0.5, 0.25, 0.25),
        pitch_shrinkage=0.1,
        reference_quantiles=(0.5, 0.9, 0.95),
        metadata={
            "target_id": "tac_topology_target_v1",
            "spec_revision": f"duration-causality-{duration_seconds}s-v1",
            "fingerprint_id": contract.fingerprint_id,
            "fingerprint_sha256": contract.artifact_sha256,
            "feature_order": FEATURE_ORDER,
            "reference_split": "discovery",
            "reference_group": "focus",
            "reference_scale_seconds": float(duration_seconds),
            "source_sha256": {
                _relative(fingerprint_path, root): sha256_file(fingerprint_path),
                _relative(descriptor_path, root): sha256_file(descriptor_path),
            },
            "source_bundle_sha256": _mapping_sha256(
                {
                    _relative(fingerprint_path, root): sha256_file(fingerprint_path),
                    _relative(descriptor_path, root): sha256_file(descriptor_path),
                }
            ),
            "evidence": fingerprint["evidence"],
        },
    )
    target_path = output_dir / "target.json"
    write_json_atomic(target_path, target_payload)
    TACTopologyTarget.from_json(target_path, root=root)
    report = {
        "schema_version": 1,
        "duration_seconds": duration_seconds,
        "reference_samples": len(rows),
        "discovery_samples": int(discovery.sum()),
        "focus_references": int(focus_reference.sum()),
        "fingerprint": _relative(fingerprint_path, root),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target": _relative(target_path, root),
        "target_sha256": sha256_file(target_path),
        "diagnostic_only": True,
        "scientific_evidence": False,
    }
    write_json_atomic(output_dir / "contract_report.json", report)
    return report


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary, path)


def _response_row(
    *,
    item: Mapping[str, Any],
    coordinates: Sequence[float],
    ood_label: Any,
    target: TACTopologyTarget,
) -> dict[str, Any]:
    matrix = np.asarray(coordinates, dtype=np.float64).reshape(1, 18)
    blocks = target.block_distances(matrix)
    return {
        "duration_seconds": int(item["duration_seconds"]),
        "sample_id": item["sample_id"],
        "anchor_sample_id": item["anchor_sample_id"],
        "matched_anchor_index": int(item["matched_anchor_index"]),
        "prompt_id": item["prompt_id"],
        "seed": int(item["seed"]),
        "direction_index": int(item["direction_index"]),
        "step_number": int(item["step_number"]),
        "rms_ratio": float(item["rms_ratio"]),
        "sign": float(item["sign"]),
        "target_distance": float(target.distance(matrix)[0]),
        "pitch_distance": float(blocks["pitch"][0]),
        "path_acoustic_phase_distance": float(blocks["path_acoustic_phase"][0]),
        "path_chroma_phase_distance": float(blocks["path_chroma_phase"][0]),
        "coordinates_json": json.dumps(list(coordinates), separators=(",", ":")),
        "ood_label": str(bool(float(ood_label))).lower(),
    }


def collect_duration_responses(
    *,
    root: Path,
    source_manifest_path: Path,
    duration_plan_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    target_path: Path,
    output_dir: Path,
    duration_seconds: int,
    ace_model_sha256: str,
    vae_sha256: str,
    workers: int = 8,
    batch_size: int = 64,
    materialize_mode: str = "auto",
    device_name: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    """Decode centers and matched antithetic perturbations for one native duration."""

    if duration_seconds not in DURATIONS:
        raise ValueError(f"duration_seconds must be one of {DURATIONS}")
    root = root.resolve()
    source_rows = _read_csv(source_manifest_path)
    plan = json.loads(duration_plan_path.read_text(encoding="utf-8"))
    if plan.get("experiment") != "ltsn_duration_causality_v1":
        raise LTSNContractError("unexpected duration diagnostic plan")
    prompt_index = {value: index for index, value in enumerate(plan["prompt_ids"])}
    anchors = [
        row
        for row in source_rows
        if int(row["step_number"]) == ANCHOR_STEP and row["prompt_id"] in prompt_index
    ]
    anchors.sort(key=lambda row: prompt_index[row["prompt_id"]])
    if len(anchors) != ANCHOR_COUNT or len({row["prompt_id"] for row in anchors}) != ANCHOR_COUNT:
        raise LTSNContractError("native-duration collection must contain 16 matched step-5 anchors")
    if {row.get("ace_model_sha256") for row in anchors} != {ace_model_sha256}:
        raise LTSNContractError("duration anchors use a different ACE model")
    if {row.get("vae_sha256") for row in anchors} != {vae_sha256}:
        raise LTSNContractError("duration anchors use a different VAE")
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    target = TACTopologyTarget.from_json(target_path, root=root)
    if target.payload["fingerprint_sha256"] != scorer.contract.artifact_sha256:
        raise LTSNContractError("duration target is bound to another fingerprint")
    planned: list[dict[str, Any]] = []
    anchor_state: dict[str, dict[str, Any]] = {}
    for matched_index, anchor in enumerate(anchors):
        latent_path = (source_manifest_path.parent / anchor["latent_path"]).resolve()
        if not latent_path.is_file() or sha256_file(latent_path) != anchor["latent_sha256"]:
            raise LTSNContractError(f"duration anchor latent is missing: {anchor['sample_id']}")
        latent = np.load(latent_path, allow_pickle=False).astype(np.float32, copy=False)
        directions, audit = orthogonal_smooth_directions(
            latent,
            anchor_id=f"matched_anchor_{matched_index:02d}",
            count=DIRECTION_COUNT,
            seed=BOOTSTRAP_SEED,
        )
        anchor_state[anchor["sample_id"]] = {
            "row": anchor,
            "latent": latent,
            "latent_path": latent_path,
            "directions": directions,
            "audit": audit,
        }
        common = {
            "duration_seconds": duration_seconds,
            "anchor_sample_id": anchor["sample_id"],
            "matched_anchor_index": matched_index,
            "prompt_id": anchor["prompt_id"],
            "seed": int(plan["seeds"][matched_index]),
            "step_number": ANCHOR_STEP,
        }
        planned.append(
            {
                **common,
                "sample_id": f"{anchor['sample_id']}__duration_center",
                "direction_index": -1,
                "direction_seed": -1,
                "rms_ratio": 0.0,
                "sign": 0.0,
            }
        )
        for direction_index, direction_seed in enumerate(audit["direction_seeds"]):
            for radius in RADII:
                radius_tag = int(round(radius * 1_000_000))
                for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                    planned.append(
                        {
                            **common,
                            "sample_id": (
                                f"{anchor['sample_id']}__duration_d{direction_index:02d}_"
                                f"r{radius_tag:05d}__{sign_name}"
                            ),
                            "direction_index": direction_index,
                            "direction_seed": int(direction_seed),
                            "rms_ratio": radius,
                            "sign": sign,
                        }
                    )
    planned.sort(key=lambda item: item["sample_id"])
    generation_plan = {
        "schema_version": 1,
        "experiment": "ltsn_duration_causality_v1",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "duration_seconds": duration_seconds,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "duration_plan_sha256": sha256_file(duration_plan_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "fingerprint_spec_revision": scorer.contract.spec_revision,
        "target_sha256": sha256_file(target_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "anchors": ANCHOR_COUNT,
        "directions": DIRECTION_COUNT,
        "radii": list(RADII),
        "planned_decodes": len(planned),
        "wav_policy": "ephemeral_delete_after_each_exact_batch",
        "planned": planned,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_plan_path = output_dir / "generation_plan.json"
    if generation_plan_path.is_file():
        existing_plan = json.loads(generation_plan_path.read_text(encoding="utf-8"))
        if not resume or existing_plan != generation_plan:
            raise LTSNContractError("duration response plan changed; use a new output directory")
    else:
        write_json_atomic(generation_plan_path, generation_plan)
    generation_plan_sha256 = sha256_file(generation_plan_path)

    from .ace_adapter import AceStepAdapter
    from .experiment import load_experiment_config

    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    descriptors: list[dict[str, str]] = []
    for batch_index, start in enumerate(range(0, len(planned), batch_size)):
        items = planned[start : start + batch_size]
        batch_dir = output_dir / "batches" / f"batch_{batch_index:05d}"
        descriptor_path = batch_dir / "descriptors.csv"
        receipt_path = batch_dir / "completion.json"
        expected_ids = [item["sample_id"] for item in items]
        if descriptor_path.is_file() and not receipt_path.exists():
            recovered_rows = _read_csv(descriptor_path)
            if sorted(row["sample_id"] for row in recovered_rows) != sorted(expected_ids):
                raise LTSNContractError(
                    f"incomplete duration descriptor checkpoint: {batch_dir}"
                )
            audio_dir = batch_dir / "audio"
            for audio_path in audio_dir.glob("*.wav") if audio_dir.is_dir() else ():
                audio_path.unlink()
            write_json_atomic(
                receipt_path,
                {
                    "schema_version": 1,
                    "generation_plan_sha256": generation_plan_sha256,
                    "sample_ids": expected_ids,
                    "descriptor_sha256": sha256_file(descriptor_path),
                    "wav_retained": 0,
                    "recovered_after_descriptor_commit": True,
                },
            )
        if receipt_path.is_file() and descriptor_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            resumed_rows = _read_csv(descriptor_path)
            if (
                receipt.get("generation_plan_sha256") != generation_plan_sha256
                or receipt.get("sample_ids") != expected_ids
                or receipt.get("descriptor_sha256") != sha256_file(descriptor_path)
                or sorted(row["sample_id"] for row in resumed_rows) != sorted(expected_ids)
                or receipt.get("wav_retained") != 0
            ):
                raise LTSNContractError(f"duration batch receipt mismatch: {batch_dir}")
            descriptors.extend(resumed_rows)
            continue
        if receipt_path.exists() or descriptor_path.exists():
            raise LTSNContractError(f"incomplete duration response checkpoint: {batch_dir}")
        trajectory_rows = []
        audio_paths = []
        for item in items:
            state = anchor_state[str(item["anchor_sample_id"])]
            latent = state["latent"]
            if int(item["direction_index"]) < 0:
                generated_latent = latent
                latent_path = state["latent_path"]
            else:
                direction = state["directions"][int(item["direction_index"])]
                latent_rms = float(np.sqrt(np.mean(np.square(latent), dtype=np.float64)))
                generated_latent = latent + (
                    float(item["sign"])
                    * float(item["rms_ratio"])
                    * latent_rms
                    * direction
                )
                latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
                if latent_path.is_file():
                    existing = np.load(latent_path, allow_pickle=False)
                    if existing.dtype != np.float32 or not np.array_equal(
                        existing, generated_latent.astype(np.float32)
                    ):
                        raise LTSNContractError("resumed duration perturbation latent changed")
                else:
                    _save_npy_atomic(latent_path, generated_latent)
            audio_path = batch_dir / "audio" / f"{item['sample_id']}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.unlink(missing_ok=True)
            adapter.decode_latent_to_audio(generated_latent.astype(np.float32), audio_path)
            audio_paths.append(audio_path)
            anchor = state["row"]
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
                }
            )
        batch_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = batch_dir / "trajectories.csv"
        write_csv_atomic(trajectory_path, trajectory_rows)
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
                duration_seconds=float(duration_seconds),
                repetition_config_overrides=REPETITION_OVERRIDES,
            )
        finally:
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)
        batch_rows = _read_csv(descriptor_path)
        write_json_atomic(
            receipt_path,
            {
                "schema_version": 1,
                "generation_plan_sha256": generation_plan_sha256,
                "sample_ids": expected_ids,
                "descriptor_sha256": sha256_file(descriptor_path),
                "wav_retained": 0,
            },
        )
        descriptors.extend(batch_rows)
    descriptor_by_id = {row["sample_id"]: row for row in descriptors}
    if set(descriptor_by_id) != {item["sample_id"] for item in planned}:
        raise LTSNContractError("duration response descriptors are incomplete")
    response_rows = []
    for item in planned:
        descriptor = descriptor_by_id[item["sample_id"]]
        score = scorer.score(
            json.loads(descriptor["pitch_descriptors_json"]),
            [float(descriptor["acoustic_loop_score"])],
            [float(descriptor["chroma_loop_score"])],
        )
        response_rows.append(
            _response_row(
                item=item,
                coordinates=score.coordinates[0].tolist(),
                ood_label=descriptor["ood_label"],
                target=target,
            )
        )
    response_rows.sort(
        key=lambda row: (
            int(row["matched_anchor_index"]),
            int(row["direction_index"]),
            float(row["rms_ratio"]),
            float(row["sign"]),
        )
    )
    response_path = output_dir / "response_points.csv"
    write_csv_atomic(response_path, response_rows)
    report, outcomes = analyze_duration_response_points(response_rows)
    outcomes_path = output_dir / "response_outcomes.csv"
    write_csv_atomic(outcomes_path, outcomes)
    report.update(
        {
            "generation_plan_sha256": generation_plan_sha256,
            "response_points_sha256": sha256_file(response_path),
            "response_outcomes_sha256": sha256_file(outcomes_path),
            "retained_wav_files": len(list(output_dir.rglob("*.wav"))),
            "retained_perturbation_latents": len(list((output_dir / "latents").glob("*.npy"))),
        }
    )
    if report["retained_wav_files"] != 0:
        raise LTSNContractError("duration diagnostic ephemeral WAV cleanup failed")
    write_json_atomic(output_dir / "duration_report.json", report)
    return report


def analyze_duration_response_points(
    rows: Sequence[Mapping[str, Any]], *, exact_margin: float = EXACT_MARGIN
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize one duration using independent anchor/direction units."""

    durations = {int(row["duration_seconds"]) for row in rows}
    if len(durations) != 1:
        raise LTSNContractError("one response table must contain exactly one duration")
    duration = durations.pop()
    centers: dict[int, Mapping[str, Any]] = {}
    signed: dict[tuple[int, int, float], dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        anchor = int(row["matched_anchor_index"])
        radius = float(row["rms_ratio"])
        if radius == 0.0:
            centers[anchor] = row
        else:
            signed[(anchor, int(row["direction_index"]), radius)][float(row["sign"])] = row
    outcomes: list[dict[str, Any]] = []
    by_direction: dict[tuple[int, int], dict[float, dict[str, Any]]] = defaultdict(dict)
    block_names = ("pitch", "path_acoustic_phase", "path_chroma_phase")
    weights = (0.5, 0.25, 0.25)
    for (anchor, direction, radius), pair in sorted(signed.items()):
        if set(pair) != {-1.0, 1.0} or anchor not in centers:
            raise LTSNContractError("duration response contains an incomplete antithetic pair")
        minus, plus = pair[-1.0], pair[1.0]
        center = centers[anchor]
        minus_distance = float(minus["target_distance"])
        plus_distance = float(plus["target_distance"])
        center_distance = float(center["target_distance"])
        response = (minus_distance - plus_distance) / (2.0 * radius)
        odd = (plus_distance - minus_distance) / 2.0
        even = (plus_distance + minus_distance) / 2.0 - center_distance
        outcome = {
            "duration_seconds": duration,
            "matched_anchor_index": anchor,
            "prompt_id": minus["prompt_id"],
            "seed": int(minus["seed"]),
            "direction_index": direction,
            "rms_ratio": radius,
            "pair_in_distribution": (
                str(minus["ood_label"]).lower() != "true"
                and str(plus["ood_label"]).lower() != "true"
            ),
            "target_response": response,
            "response_sign": int(np.sign(response)) if abs(response) >= exact_margin else 0,
            "odd_component": odd,
            "even_component": even,
            "absolute_even_to_odd_ratio": abs(even) / max(abs(odd), exact_margin),
        }
        for name, weight in zip(block_names, weights, strict=True):
            outcome[f"{name}_response"] = weight * (
                float(minus[f"{name}_distance"]) - float(plus[f"{name}_distance"])
            ) / (2.0 * radius)
        outcomes.append(outcome)
        by_direction[(anchor, direction)][radius] = outcome
    successes = comparisons = reference_ties = tested_ties = monotonic = 0
    direction_scores: dict[str, float] = {}
    for (anchor, direction), radius_rows in sorted(by_direction.items()):
        if set(radius_rows) != set(RADII):
            raise LTSNContractError("duration response radius grid changed")
        reference_sign = int(radius_rows[REFERENCE_RADIUS]["response_sign"])
        reference_ties += reference_sign == 0
        per_direction_success = 0
        for radius in RADII:
            if radius == REFERENCE_RADIUS:
                continue
            tested_sign = int(radius_rows[radius]["response_sign"])
            comparisons += 1
            tested_ties += tested_sign == 0
            success = reference_sign != 0 and tested_sign == reference_sign
            successes += success
            per_direction_success += success
        direction_scores[f"{anchor}:{direction}"] = per_direction_success / 2.0
        magnitudes = [abs(float(radius_rows[radius]["odd_component"])) for radius in RADII]
        monotonic += all(
            right + exact_margin >= left
            for left, right in zip(magnitudes, magnitudes[1:], strict=False)
        )
    interval = wilson_interval(successes, comparisons)
    agreement = successes / comparisons
    ood_pairs = sum(not bool(row["pair_in_distribution"]) for row in outcomes)
    tie_responses = sum(int(row["response_sign"]) == 0 for row in outcomes)
    total_absolute = sum(abs(float(row["target_response"])) for row in outcomes)
    attribution = {
        name: sum(abs(float(row[f"{name}_response"])) for row in outcomes)
        / max(total_absolute, exact_margin)
        for name in block_names
    }
    report = {
        "schema_version": 1,
        "experiment": "ltsn_duration_causality_v1",
        "mode": "development_only_diagnostic",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "duration_seconds": duration,
        "anchors": len(centers),
        "directions": len(by_direction),
        "pairs": len(outcomes),
        "cross_radius_sign": {
            "successes": successes,
            "comparisons": comparisons,
            "agreement": agreement,
            "wilson95": interval,
            "reference_ties": reference_ties,
            "tested_ties": tested_ties,
        },
        "tie_response_fraction": tie_responses / len(outcomes),
        "ood_pair_fraction": ood_pairs / len(outcomes),
        "response_magnitude_monotonicity": monotonic / len(by_direction),
        "median_absolute_target_response": float(
            np.median([abs(float(row["target_response"])) for row in outcomes])
        ),
        "weighted_block_attribution_fraction": attribution,
        "direction_stability_scores": direction_scores,
        "within_duration_gates": {
            "agreement_at_least_0_80": agreement >= 0.80,
            "wilson_lower_above_0_50": interval[0] > 0.50,
        },
    }
    return report, outcomes


def _cluster_bootstrap_difference(
    shorter: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if set(shorter) != set(baseline):
        raise LTSNContractError("duration direction identities are not matched")
    by_anchor: dict[int, list[float]] = defaultdict(list)
    for key in sorted(shorter):
        anchor = int(key.split(":", 1)[0])
        by_anchor[anchor].append(float(shorter[key]) - float(baseline[key]))
    anchors = sorted(by_anchor)
    observed = float(np.mean([value for values in by_anchor.values() for value in values]))
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.choice(anchors, size=len(anchors), replace=True)
        draws[index] = np.mean([value for anchor in sampled for value in by_anchor[int(anchor)]])
    return {
        "estimate": observed,
        "cluster_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "clusters": len(anchors),
        "resamples": resamples,
        "seed": seed,
    }


def build_duration_causality_report(
    *,
    duration_report_paths: Mapping[int, Path],
    output_path: Path,
    bootstrap_resamples: int = 5000,
) -> dict[str, Any]:
    """Compare 60/30 s with the matched 180 s baseline without best-of-two selection."""

    if set(duration_report_paths) != set(DURATIONS):
        raise ValueError(f"reports must cover exactly {DURATIONS}")
    reports = {
        duration: json.loads(path.read_text(encoding="utf-8"))
        for duration, path in duration_report_paths.items()
    }
    baseline = reports[180]
    comparisons: dict[str, Any] = {}
    supported = []
    for duration in (60, 30):
        current = reports[duration]
        difference = _cluster_bootstrap_difference(
            current["direction_stability_scores"],
            baseline["direction_stability_scores"],
            resamples=bootstrap_resamples,
            seed=BOOTSTRAP_SEED + duration,
        )
        agreement = float(current["cross_radius_sign"]["agreement"])
        wilson_low = float(current["cross_radius_sign"]["wilson95"][0])
        gates = {
            "agreement_at_least_0_80": agreement >= 0.80,
            "wilson_lower_above_0_50": wilson_low > 0.50,
            "improvement_over_180s_at_least_0_10": difference["estimate"] >= 0.10,
            "cluster_bootstrap_lower_above_0": difference["cluster_bootstrap_ci95"][0] > 0.0,
        }
        status = "duration_signal_supported" if all(gates.values()) else "not_supported"
        supported.append(status == "duration_signal_supported")
        comparisons[f"{duration}s_vs_180s"] = {
            **difference,
            "shorter_agreement": agreement,
            "baseline_180s_agreement": baseline["cross_radius_sign"]["agreement"],
            "gates": gates,
            "status": status,
        }
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_duration_causality_v1",
        "mode": "development_only_diagnostic",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "native_duration_generation": True,
        "durations_seconds": list(DURATIONS),
        "duration_reports": {str(key): reports[key] for key in DURATIONS},
        "matched_cluster_comparisons": comparisons,
        "status": "duration_causality_supported" if any(supported) else "signal_not_supported",
        "interpretation": (
            "60s and 30s were both pre-registered and are reported independently; "
            "the result is not selected post hoc from the better short duration"
        ),
        "source_sha256": {
            str(duration): sha256_file(path)
            for duration, path in duration_report_paths.items()
        },
    }
    write_json_atomic(output_path, payload)
    return payload
