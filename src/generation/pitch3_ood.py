"""Deterministic, exact-decoded OOD augmentation for the Pitch-3 control head."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import AceStepAdapter
from .experiment import load_experiment_config
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .pitch3_contract import Pitch3Contract, load_pitch3_contract
from .pitch3_exact_scorer import ExactPitch3Scorer

PITCH3_OOD_TRANSFORM_VERSION = "pitch3_latent_ood_v1"
PITCH3_OOD_LABEL_SOURCE = "deterministic_latent_transform_v1"
TRAIN_OOD_KINDS = ("ood_zero", "ood_scale_high")
EVALUATION_OOD_KINDS = ("ood_time_reverse", "ood_channel_roll")
CORRECTION_STEPS = (4, 5, 6)
EVALUATION_STEPS = (4, 5, 6, 8)
SPLITS = ("train", "development", "calibration", "qualification")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _rectangular(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    ordered = list(columns or ())
    for row in rows:
        for name in row:
            if name not in ordered:
                ordered.append(name)
    return [{name: row.get(name, "") for name in ordered} for row in rows]


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve()).replace("\\", "/")


def _rooted(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_source_manifest(
    path: Path, rows: Sequence[Mapping[str, str]], contract: Pitch3Contract
) -> None:
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("fingerprint_json_sha256", "").lower() != contract.artifact_sha256:
            raise LTSNContractError("Pitch-3 source fingerprint hash mismatch")
        if row.get("label_scope") != "per_snapshot_exact":
            raise LTSNContractError("Pitch-3 OOD augmentation requires exact source labels")
        if json.loads(row.get("feature_order_json", "null")) != list(
            contract.feature_order
        ):
            raise LTSNContractError("Pitch-3 source feature order mismatch")
        coordinates = [float(value) for value in json.loads(row["coordinates_json"])]
        if len(coordinates) != 3 or not all(math.isfinite(value) for value in coordinates):
            raise LTSNContractError("Pitch-3 source coordinates are malformed")
        split = row.get("split", "")
        if split not in SPLITS:
            raise LTSNContractError(f"unsupported Pitch-3 OOD split: {split}")
        latent_path = (path.parent / row["latent_path"]).resolve()
        if not latent_path.is_file() or sha256_file(latent_path) != row["latent_sha256"]:
            raise LTSNContractError("Pitch-3 source latent is missing or hash-mismatched")
        prompt_splits[row["prompt_id"]].add(split)
        trajectory_splits[row["trajectory_id"]].add(split)
    if any(len(values) != 1 for values in prompt_splits.values()):
        raise LTSNContractError("Pitch-3 source prompt leakage detected")
    if any(len(values) != 1 for values in trajectory_splits.values()):
        raise LTSNContractError("Pitch-3 source trajectory leakage detected")


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary, path)


def apply_pitch3_ood_transform(latent: np.ndarray, kind: str) -> np.ndarray:
    """Apply one frozen OOD transform without changing the latent shape."""

    values = np.asarray(latent, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 64 or not np.isfinite(values).all():
        raise LTSNContractError("Pitch-3 OOD anchors must have finite shape [T,64]")
    if kind == "ood_zero":
        transformed = np.zeros_like(values)
    elif kind == "ood_scale_high":
        transformed = values * np.float32(4.0)
    elif kind == "ood_time_reverse":
        transformed = np.ascontiguousarray(values[::-1])
    elif kind == "ood_channel_roll":
        transformed = np.roll(values, shift=17, axis=1).copy()
    else:
        raise LTSNContractError(f"unsupported Pitch-3 OOD transform: {kind}")
    if transformed.shape != values.shape or not np.isfinite(transformed).all():
        raise LTSNContractError("Pitch-3 OOD transform produced an invalid latent")
    return transformed.astype(np.float32, copy=False)


def build_pitch3_ood_plan(
    source_rows: Sequence[Mapping[str, str]],
    *,
    ood_per_prompt: int = 1,
    evaluation_ood_per_prompt: int = 1,
) -> list[dict[str, Any]]:
    """Select split-local anchors and assign train/evaluation transform families."""

    if ood_per_prompt < 1 or evaluation_ood_per_prompt < 1:
        raise ValueError("Pitch-3 OOD counts per prompt must be positive")
    grouped: dict[tuple[str, str, int], list[Mapping[str, str]]] = defaultdict(list)
    prompts: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        split = str(row.get("split", ""))
        if split not in SPLITS:
            raise LTSNContractError(f"unsupported Pitch-3 OOD split: {split}")
        if float(row.get("ood_label", 0.0)) >= 0.5:
            continue
        prompt_id = str(row.get("prompt_id", ""))
        step = int(row.get("step_number", 0))
        prompts[split].add(prompt_id)
        expected = EVALUATION_STEPS if split in {"calibration", "qualification"} else (
            CORRECTION_STEPS
        )
        if step in expected:
            grouped[(split, prompt_id, step)].append(row)

    planned: list[dict[str, Any]] = []
    for split in SPLITS:
        expected_steps = (
            EVALUATION_STEPS if split in {"calibration", "qualification"} else CORRECTION_STEPS
        )
        count = (
            evaluation_ood_per_prompt
            if split in {"calibration", "qualification"}
            else ood_per_prompt
        )
        families = (
            EVALUATION_OOD_KINDS
            if split in {"calibration", "qualification"}
            else TRAIN_OOD_KINDS
        )
        for prompt_rank, prompt_id in enumerate(sorted(prompts.get(split, set()))):
            for step_rank, step in enumerate(expected_steps):
                candidates = sorted(
                    grouped.get((split, prompt_id, step), ()),
                    key=lambda row: (str(row.get("trajectory_id", "")), row["sample_id"]),
                )
                if len(candidates) < count:
                    raise LTSNContractError(
                        f"{split} prompt {prompt_id} lacks {count} ID anchors at step {step}"
                    )
                for index, anchor in enumerate(candidates[:count]):
                    kind = families[(prompt_rank + step_rank + index) % len(families)]
                    sample_id = f"{anchor['sample_id']}__p3ood_{kind[4:]}_{index:02d}"
                    planned.append(
                        {
                            "sample_id": sample_id,
                            "source_sample_id": anchor["sample_id"],
                            "source_latent_sha256": anchor["latent_sha256"],
                            "prompt_id": prompt_id,
                            "trajectory_id": sample_id,
                            "split": split,
                            "step_number": step,
                            "timestep": float(anchor["timestep"]),
                            "is_final": str(anchor.get("is_final", "false")).lower()
                            == "true",
                            "ood_kind": kind,
                            "ood_transform_version": PITCH3_OOD_TRANSFORM_VERSION,
                            "ood_label_source": PITCH3_OOD_LABEL_SOURCE,
                        }
                    )
    if not planned:
        raise LTSNContractError("Pitch-3 OOD plan is empty")
    sample_ids = [str(item["sample_id"]) for item in planned]
    if len(sample_ids) != len(set(sample_ids)):
        raise LTSNContractError("Pitch-3 OOD plan contains duplicate sample IDs")
    return planned


def _validate_receipt(path: Path, output_dir: Path, plan_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_sha256") != plan_sha256:
        raise LTSNContractError("Pitch-3 OOD receipt belongs to a different plan")
    for name in ("latent", "audio"):
        artifact = output_dir / payload[f"{name}_path"]
        if not artifact.is_file() or sha256_file(artifact) != payload[f"{name}_sha256"]:
            raise LTSNContractError(f"Pitch-3 OOD receipt has mismatched {name}")
    return payload


def _materialize_ood(
    *,
    item: Mapping[str, Any],
    source: Mapping[str, str],
    source_manifest: Path,
    output_dir: Path,
    plan_sha256: str,
    adapter: AceStepAdapter,
) -> dict[str, Any]:
    receipt_path = output_dir / "receipts" / f"{item['sample_id']}.json"
    if receipt_path.is_file():
        receipt = _validate_receipt(receipt_path, output_dir, plan_sha256)
        if receipt.get("sample_id") != item["sample_id"]:
            raise LTSNContractError("Pitch-3 OOD receipt sample binding mismatch")
        return receipt
    source_latent = (source_manifest.parent / source["latent_path"]).resolve()
    if sha256_file(source_latent) != item["source_latent_sha256"]:
        raise LTSNContractError("Pitch-3 OOD source latent hash changed")
    transformed = apply_pitch3_ood_transform(
        np.load(source_latent, allow_pickle=False), str(item["ood_kind"])
    )
    latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
    audio_path = output_dir / "data_raw" / "pitch3_ood" / f"{item['sample_id']}.wav"
    if latent_path.is_file():
        existing = np.load(latent_path, allow_pickle=False)
        if existing.dtype != np.float32 or not np.array_equal(existing, transformed):
            raise LTSNContractError("unreceipted Pitch-3 OOD latent is mismatched")
    else:
        _save_npy_atomic(latent_path, transformed)
    adapter.decode_latent_to_audio(transformed, audio_path)
    receipt = {
        "schema_version": 1,
        "sample_id": item["sample_id"],
        "source_sample_id": item["source_sample_id"],
        "source_latent_sha256": item["source_latent_sha256"],
        "ood_kind": item["ood_kind"],
        "ood_transform_version": PITCH3_OOD_TRANSFORM_VERSION,
        "latent_path": latent_path.relative_to(output_dir).as_posix(),
        "latent_sha256": sha256_file(latent_path),
        "audio_path": audio_path.relative_to(output_dir).as_posix(),
        "audio_sha256": sha256_file(audio_path),
        "plan_sha256": plan_sha256,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _load_source_exact_labels(
    source_manifest: Path, source_rows: Sequence[Mapping[str, str]]
) -> tuple[Path, list[dict[str, str]]]:
    locations = {
        (row.get("exact_label_table_path", ""), row.get("exact_label_table_sha256", ""))
        for row in source_rows
    }
    if len(locations) != 1:
        raise LTSNContractError("Pitch-3 source rows must share one exact label table")
    relative, expected_sha256 = locations.pop()
    exact_path = (source_manifest.parent / relative).resolve()
    if not relative or not exact_path.is_file() or sha256_file(exact_path) != expected_sha256:
        raise LTSNContractError("Pitch-3 source exact label table is missing or mismatched")
    labels = _read_csv(exact_path)
    if {row["sample_id"] for row in labels} != {row["sample_id"] for row in source_rows}:
        raise LTSNContractError("Pitch-3 source exact labels do not match the manifest")
    return exact_path, labels


def _score_ood_rows(
    *,
    trajectories: Sequence[Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, str]],
    scorer: ExactPitch3Scorer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    descriptor_by_id = {row["sample_id"]: row for row in descriptors}
    label_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        sample_id = str(trajectory["sample_id"])
        descriptor = descriptor_by_id.get(sample_id)
        if descriptor is None or descriptor.get("label_source") != "decoded_snapshot_exact_v1":
            raise LTSNContractError(f"missing exact Pitch-3 OOD descriptor: {sample_id}")
        if descriptor.get("audio_sha256") != trajectory["audio_sha256"]:
            raise LTSNContractError("Pitch-3 OOD descriptor audio hash mismatch")
        score = scorer.score(json.loads(descriptor["pitch_descriptors_json"]))
        coordinates = score.coordinates[0].tolist()
        technical_ood = int(float(descriptor.get("ood_label", 0.0)) >= 0.5)
        provenance = {
            "source_sample_id": trajectory["source_sample_id"],
            "source_latent_sha256": trajectory["source_latent_sha256"],
            "ood_kind": trajectory["ood_kind"],
            "ood_transform_version": PITCH3_OOD_TRANSFORM_VERSION,
            "ood_label_source": PITCH3_OOD_LABEL_SOURCE,
            "technical_ood_label": technical_ood,
        }
        label_rows.append(
            {
                "sample_id": sample_id,
                "pitch_descriptors_json": descriptor["pitch_descriptors_json"],
                "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                "focus_logit": float(score.focus_logit[0]),
                "focus_probability": float(score.focus_probability[0]),
                "focus_target_band_loss": float(score.target_band_loss[0]),
                "target_center_distance": float(score.target_center_distance[0]),
                "ood_label": 1.0,
                "label_scope": "per_snapshot_exact",
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                **provenance,
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
                "latent_path": trajectory["latent_path"],
                "latent_sha256": trajectory["latent_sha256"],
                "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                "focus_logit": float(score.focus_logit[0]),
                "ood_label": 1.0,
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "feature_order_json": json.dumps(list(scorer.contract.feature_order)),
                "label_scope": "per_snapshot_exact",
                "exact_label_table_sha256": "",
                "exact_label_table_path": "",
                "is_final": str(bool(trajectory["is_final"])).lower(),
                "ace_model_sha256": trajectory["ace_model_sha256"],
                "vae_sha256": trajectory["vae_sha256"],
                "qualification_eligible": "true",
                "guidance_promotion_eligible": "false",
                **provenance,
            }
        )
    return label_rows, manifest_rows


def _validate_split_manifest(
    path: Path, source_rows: Sequence[Mapping[str, str]], contract: Pitch3Contract
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint_json_sha256") != contract.artifact_sha256:
        raise LTSNContractError("Pitch-3 split manifest fingerprint hash mismatch")
    expected = {row["prompt_id"]: row["split"] for row in source_rows}
    if payload.get("assignments") != dict(sorted(expected.items())):
        raise LTSNContractError("Pitch-3 split assignments do not match the source manifest")
    return payload


def build_pitch3_ood_augmentation(
    *,
    root: Path,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    output_dir: Path,
    ood_per_prompt: int = 1,
    evaluation_ood_per_prompt: int = 1,
    duration_seconds: float = 180.0,
    workers: int = 8,
    exact_batch_size: int = 256,
    materialize_mode: str = "auto",
    cleanup_exact_batches: bool = True,
    device_name: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    """Build exact-decoded Pitch-3 OOD rows and a merged training manifest."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive and finite")
    root = root.resolve()
    source_manifest_path = _rooted(source_manifest_path, root)
    source_split_manifest_path = _rooted(source_split_manifest_path, root)
    ace_config_path = _rooted(ace_config_path, root)
    fingerprint_path = _rooted(fingerprint_path, root)
    output_dir = _rooted(output_dir, root)
    contract = load_pitch3_contract(fingerprint_path)
    source_rows = _read_csv(source_manifest_path)
    _validate_source_manifest(source_manifest_path, source_rows, contract)
    if any(float(row.get("ood_label", 0.0)) >= 0.5 for row in source_rows):
        raise LTSNContractError("Pitch-3 OOD augmentation requires an ID-only source manifest")
    split_payload = _validate_split_manifest(
        source_split_manifest_path, source_rows, contract
    )
    source_exact_path, source_exact_rows = _load_source_exact_labels(
        source_manifest_path, source_rows
    )
    source_by_id = {row["sample_id"]: row for row in source_rows}
    planned = build_pitch3_ood_plan(
        source_rows,
        ood_per_prompt=ood_per_prompt,
        evaluation_ood_per_prompt=evaluation_ood_per_prompt,
    )
    plan = {
        "schema_version": 1,
        "scope": "pitch3_exact_decoded_ood_augmentation",
        "transform_version": PITCH3_OOD_TRANSFORM_VERSION,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_exact_label_table_sha256": sha256_file(source_exact_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "duration_seconds": float(duration_seconds),
        "ood_per_prompt": ood_per_prompt,
        "evaluation_ood_per_prompt": evaluation_ood_per_prompt,
        "items": planned,
    }
    plan_path = output_dir / "pitch3_ood_plan.json"
    if plan_path.is_file():
        if not resume:
            raise FileExistsError("Pitch-3 OOD plan exists; pass --resume or use a new output")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("Pitch-3 OOD plan changed; use a new output directory")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)

    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    receipts = [
        _materialize_ood(
            item=item,
            source=source_by_id[str(item["source_sample_id"])],
            source_manifest=source_manifest_path,
            output_dir=output_dir,
            plan_sha256=plan_sha256,
            adapter=adapter,
        )
        for item in planned
    ]
    receipt_by_id = {row["sample_id"]: row for row in receipts}
    trajectory_rows: list[dict[str, Any]] = []
    for item in planned:
        source = source_by_id[str(item["source_sample_id"])]
        receipt = receipt_by_id[str(item["sample_id"])]
        trajectory_rows.append(
            {
                "sample_id": item["sample_id"],
                "prompt_id": item["prompt_id"],
                "trajectory_id": item["trajectory_id"],
                "split": item["split"],
                "model_family": source["model_family"],
                "step_number": item["step_number"],
                "timestep": item["timestep"],
                "is_final": str(bool(item["is_final"])).lower(),
                "latent_path": receipt["latent_path"],
                "latent_sha256": receipt["latent_sha256"],
                "audio_path": receipt["audio_path"],
                "audio_sha256": receipt["audio_sha256"],
                "ace_model_sha256": source["ace_model_sha256"],
                "vae_sha256": source["vae_sha256"],
                "engineering_smoke": "false",
                "schema_version": 1,
                "source_sample_id": item["source_sample_id"],
                "source_latent_sha256": item["source_latent_sha256"],
                "ood_kind": item["ood_kind"],
                "ood_transform_version": PITCH3_OOD_TRANSFORM_VERSION,
                "ood_label_source": PITCH3_OOD_LABEL_SOURCE,
            }
        )
    trajectory_path = output_dir / "pitch3_ood_trajectories.csv"
    write_csv_atomic(trajectory_path, trajectory_rows)
    descriptor_path = output_dir / "pitch3_ood_exact_descriptors.csv"
    exact_storage = build_exact_snapshot_descriptors(
        project_root=root,
        trajectory_manifest=trajectory_path,
        work_dir=output_dir / "exact_work",
        output_path=descriptor_path,
        workers=workers,
        batch_size=exact_batch_size,
        materialize_mode=materialize_mode,
        cleanup_batches=cleanup_exact_batches,
        resume=resume,
        duration_seconds=duration_seconds,
    )
    descriptors = _read_csv(descriptor_path)
    scorer = ExactPitch3Scorer(contract)
    ood_labels, ood_manifest = _score_ood_rows(
        trajectories=trajectory_rows,
        descriptors=descriptors,
        scorer=scorer,
    )

    ood_exact_path = output_dir / "pitch3_ood_exact_labels.csv"
    write_csv_atomic(ood_exact_path, ood_labels)
    ood_exact_sha256 = sha256_file(ood_exact_path)
    for row in ood_manifest:
        row["exact_label_table_sha256"] = ood_exact_sha256
        row["exact_label_table_path"] = ood_exact_path.name
    ood_manifest_path = output_dir / "pitch3_ood_manifest.csv"
    write_csv_atomic(ood_manifest_path, ood_manifest)

    combined_exact_path = output_dir / "pitch3_exact_labels_augmented.csv"
    combined_exact = _rectangular([*source_exact_rows, *ood_labels])
    write_csv_atomic(combined_exact_path, combined_exact)
    combined_exact_sha256 = sha256_file(combined_exact_path)
    combined_manifest_path = output_dir / "pitch3_training_manifest_augmented.csv"
    source_rebased: list[dict[str, Any]] = []
    for source in source_rows:
        row: dict[str, Any] = dict(source)
        row["latent_path"] = _relative(
            source_manifest_path.parent / source["latent_path"], output_dir
        )
        source_rebased.append(row)
    combined_manifest = _rectangular([*source_rebased, *ood_manifest])
    for row in combined_manifest:
        row["exact_label_table_path"] = combined_exact_path.name
        row["exact_label_table_sha256"] = combined_exact_sha256
    write_csv_atomic(combined_manifest_path, combined_manifest)
    _validate_source_manifest(combined_manifest_path, combined_manifest, contract)

    counts = Counter((str(row["split"]), int(float(row["ood_label"]))) for row in combined_manifest)
    split_payload.update(
        {
            "schema_version": 2,
            "ood_transform_version": PITCH3_OOD_TRANSFORM_VERSION,
            "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
            "source_training_manifest_sha256": sha256_file(source_manifest_path),
            "pitch3_ood_plan_sha256": plan_sha256,
            "training_manifest_sha256": sha256_file(combined_manifest_path),
            "exact_label_table_sha256": combined_exact_sha256,
            "ood_counts_by_split": {
                split: counts[(split, 1)] for split in SPLITS
            },
            "qualification_eligible": True,
            "guidance_promotion_eligible": False,
        }
    )
    combined_split_path = output_dir / "pitch3_split_manifest_augmented.json"
    write_json_atomic(combined_split_path, split_payload)
    summary = {
        "schema_version": 1,
        "scope": "pitch3_exact_decoded_ood_augmentation",
        "source_samples": len(source_rows),
        "ood_samples": len(ood_manifest),
        "combined_samples": len(combined_manifest),
        "counts_by_split": {
            split: {"id": counts[(split, 0)], "ood": counts[(split, 1)]}
            for split in SPLITS
        },
        "ood_counts_by_kind": dict(
            sorted(Counter(str(row["ood_kind"]) for row in ood_manifest).items())
        ),
        "technical_ood_samples": sum(int(row["technical_ood_label"]) for row in ood_manifest),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "plan_sha256": plan_sha256,
        "trajectory_manifest_sha256": sha256_file(trajectory_path),
        "descriptor_table_sha256": sha256_file(descriptor_path),
        "ood_exact_label_table_sha256": ood_exact_sha256,
        "ood_manifest_sha256": sha256_file(ood_manifest_path),
        "combined_exact_label_table_sha256": combined_exact_sha256,
        "combined_training_manifest_sha256": sha256_file(combined_manifest_path),
        "combined_split_manifest_sha256": sha256_file(combined_split_path),
        "exact_storage": exact_storage,
        "qualification_eligible": True,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    write_json_atomic(output_dir / "pitch3_ood_summary.json", summary)
    return summary
