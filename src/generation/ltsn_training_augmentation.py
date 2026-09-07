"""Exact-labelled local and held-out OOD augmentation for LTSN V3."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import AceStepAdapter
from .experiment import load_experiment_config
from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_dataset import LTSNSnapshot, read_ltsn_manifest
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import canonical_json_sha256, write_csv_atomic, write_json_atomic
from .path_homology_exact_scorer import ExactPathHomologyScorer


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary, path)


def _smooth_direction(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape, dtype=np.float32)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(raw, ((2, 2), (0, 0)), mode="edge")
    smooth = sum(
        weight * padded[offset : offset + shape[0]]
        for offset, weight in enumerate(kernel)
    )
    return np.asarray(smooth, dtype=np.float32)


def _local_perturbation(
    latent: np.ndarray, *, seed: int, rms_ratio: float, sign: float
) -> np.ndarray:
    direction = _smooth_direction(latent.shape, seed)
    latent_rms = float(np.sqrt(np.mean(np.square(latent, dtype=np.float64))))
    direction_rms = float(np.sqrt(np.mean(np.square(direction, dtype=np.float64))))
    if latent_rms <= 0 or direction_rms <= 0:
        raise LTSNContractError("local perturbation requires non-zero latent and direction RMS")
    update = direction * (sign * rms_ratio * latent_rms / direction_rms)
    result = latent + update
    if not np.isfinite(result).all():
        raise LTSNContractError("local perturbation produced NaN or Inf")
    return result.astype(np.float32, copy=False)


def _select_anchors(
    records: list[LTSNSnapshot], trajectories_per_prompt: int
) -> list[LTSNSnapshot]:
    grouped: dict[str, dict[str, list[LTSNSnapshot]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record.split == "train" and not record.is_final and record.step_number in {4, 5, 6}:
            grouped[record.prompt_id][record.trajectory_id].append(record)
    selected: list[LTSNSnapshot] = []
    for prompt_id in sorted(grouped):
        trajectories = grouped[prompt_id]
        for trajectory_id in sorted(trajectories)[:trajectories_per_prompt]:
            rows = sorted(trajectories[trajectory_id], key=lambda item: item.step_number)
            if {row.step_number for row in rows} != {4, 5, 6}:
                raise LTSNContractError(
                    f"selected augmentation trajectory lacks steps 4/5/6: {trajectory_id}"
                )
            selected.extend(rows)
    if not selected:
        raise LTSNContractError("no train step 4/5/6 anchors are available for augmentation")
    return selected


def _augmentation_plan(
    anchors: list[LTSNSnapshot],
    *,
    perturbations_per_anchor: int,
    rms_ratio: float,
    ood_per_prompt: int,
    seed: int,
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    prompt_ood: dict[str, int] = defaultdict(int)
    prompt_rank = {
        prompt_id: index
        for index, prompt_id in enumerate(sorted({anchor.prompt_id for anchor in anchors}))
    }
    for anchor_index, anchor in enumerate(anchors):
        for perturbation_index in range(perturbations_per_anchor):
            sign = -1.0 if perturbation_index % 2 else 1.0
            item_seed = seed + anchor_index * max(perturbations_per_anchor, 1) + perturbation_index
            sample_id = f"{anchor.sample_id}__local{perturbation_index:02d}"
            planned.append(
                {
                    "sample_id": sample_id,
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": anchor.prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": "local_direction",
                    "split": "train",
                    "seed": item_seed,
                    "rms_ratio": rms_ratio,
                    "sign": sign,
                }
            )
        if (
            anchor.step_number == 5
            and prompt_ood[anchor.prompt_id] < ood_per_prompt
        ):
            ood_index = prompt_ood[anchor.prompt_id]
            prompt_ood[anchor.prompt_id] += 1
            planned.append(
                {
                    "sample_id": f"{anchor.sample_id}__ood{ood_index:02d}",
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": anchor.prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": (
                        "ood_zero"
                        if (prompt_rank[anchor.prompt_id] + ood_index) % 2 == 0
                        else "ood_scale_high"
                    ),
                    "split": "train",
                    "seed": seed + 10_000_000 + anchor_index,
                    "rms_ratio": 0.0,
                    "sign": 0.0,
                }
            )
    sample_ids = [item["sample_id"] for item in planned]
    if len(set(sample_ids)) != len(sample_ids):
        raise LTSNContractError("augmentation plan contains duplicate sample IDs")
    return planned


def _evaluation_ood_plan(
    records: list[LTSNSnapshot],
    *,
    ood_per_prompt: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create prompt-held-out OOD examples with transforms unseen in training."""

    if ood_per_prompt < 0:
        raise ValueError("evaluation OOD count must be non-negative")
    grouped: dict[tuple[str, str], list[LTSNSnapshot]] = defaultdict(list)
    for record in records:
        if (
            record.split in {"calibration", "qualification"}
            and record.step_number == 5
            and not record.is_final
        ):
            grouped[(record.split, record.prompt_id)].append(record)
    planned: list[dict[str, Any]] = []
    for prompt_rank, ((split, prompt_id), candidates) in enumerate(sorted(grouped.items())):
        ordered = sorted(candidates, key=lambda item: (item.trajectory_id, item.sample_id))
        if len(ordered) < ood_per_prompt:
            raise LTSNContractError(
                f"{split} prompt lacks {ood_per_prompt} step-5 OOD anchors: {prompt_id}"
            )
        for ood_index, anchor in enumerate(ordered[:ood_per_prompt]):
            kind = (
                "ood_time_reverse"
                if (prompt_rank + ood_index) % 2 == 0
                else "ood_channel_roll"
            )
            planned.append(
                {
                    "sample_id": f"{anchor.sample_id}__heldout_ood{ood_index:02d}",
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": kind,
                    "split": split,
                    "seed": seed + 20_000_000 + prompt_rank * max(ood_per_prompt, 1) + ood_index,
                    "rms_ratio": 0.0,
                    "sign": 0.0,
                }
            )
    expected_prompts = {
        (record.split, record.prompt_id)
        for record in records
        if record.split in {"calibration", "qualification"}
    }
    if ood_per_prompt and set(grouped) != expected_prompts:
        raise LTSNContractError(
            "every calibration/qualification prompt must provide a step-5 OOD anchor"
        )
    return planned


def _validate_receipt(path: Path, output_dir: Path, plan_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_sha256") != plan_sha256:
        raise LTSNContractError("augmentation receipt belongs to a different plan")
    for name in ("latent", "audio"):
        artifact = output_dir / payload[f"{name}_path"]
        if not artifact.is_file() or sha256_file(artifact) != payload[f"{name}_sha256"]:
            raise LTSNContractError(f"augmentation receipt {name} is hash-mismatched")
    return payload


def _materialize_augmentation(
    *,
    item: dict[str, Any],
    anchor: LTSNSnapshot,
    adapter: AceStepAdapter,
    output_dir: Path,
    plan_sha256: str,
) -> dict[str, Any]:
    receipt_path = output_dir / "receipts" / f"{item['sample_id']}.json"
    if receipt_path.is_file():
        receipt = _validate_receipt(receipt_path, output_dir, plan_sha256)
        if receipt.get("sample_id") != item["sample_id"]:
            raise LTSNContractError("augmentation receipt sample binding mismatch")
        return receipt
    latent = np.load(anchor.latent_path, allow_pickle=False).astype(np.float32, copy=False)
    if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
        raise LTSNContractError(f"invalid augmentation anchor latent: {anchor.sample_id}")
    if item["kind"] == "local_direction":
        augmented = _local_perturbation(
            latent,
            seed=int(item["seed"]),
            rms_ratio=float(item["rms_ratio"]),
            sign=float(item["sign"]),
        )
    elif item["kind"] == "ood_zero":
        augmented = np.zeros_like(latent)
    elif item["kind"] == "ood_scale_high":
        augmented = latent * 4.0
    elif item["kind"] == "ood_time_reverse":
        augmented = np.ascontiguousarray(latent[::-1])
    elif item["kind"] == "ood_channel_roll":
        augmented = np.roll(latent, shift=17, axis=1).copy()
    else:
        raise LTSNContractError(f"unknown augmentation kind: {item['kind']}")
    latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
    audio_path = output_dir / "data_raw" / "local_augmentation" / f"{item['sample_id']}.wav"
    if latent_path.is_file():
        existing = np.load(latent_path, allow_pickle=False)
        if existing.dtype != np.float32 or not np.array_equal(existing, augmented):
            raise LTSNContractError(
                f"unreceipted augmentation latent is mismatched: {item['sample_id']}"
            )
    elif latent_path.exists():
        raise LTSNContractError(
            f"augmentation latent path is not a file: {item['sample_id']}"
        )
    else:
        _save_npy_atomic(latent_path, augmented)
    if audio_path.exists() and not audio_path.is_file():
        raise LTSNContractError(
            f"augmentation audio path is not a file: {item['sample_id']}"
        )
    # Re-decode any unreceipted audio atomically. This recovers a crash between
    # derived-artifact creation and the receipt without trusting stale audio.
    adapter.decode_latent_to_audio(augmented, audio_path)
    receipt = {
        "schema_version": 1,
        "sample_id": item["sample_id"],
        "anchor_sample_id": anchor.sample_id,
        "kind": item["kind"],
        "latent_path": latent_path.relative_to(output_dir).as_posix(),
        "latent_sha256": sha256_file(latent_path),
        "audio_path": audio_path.relative_to(output_dir).as_posix(),
        "audio_sha256": sha256_file(audio_path),
        "plan_sha256": plan_sha256,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _rectangular(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def _write_augmentation_trajectory_manifest(
    *,
    path: Path,
    planned: list[dict[str, Any]],
    source_by_sample: dict[str, dict[str, str]],
    receipt_by_id: dict[str, dict[str, Any]],
) -> None:
    """Issue the minimal hash-bound manifest consumed by exact batch labeling."""

    rows: list[dict[str, Any]] = []
    for item in planned:
        anchor = source_by_sample[item["anchor_sample_id"]]
        receipt = receipt_by_id[item["sample_id"]]
        rows.append(
            {
                "sample_id": item["sample_id"],
                "prompt_id": item["prompt_id"],
                "trajectory_id": item["sample_id"],
                "split": item["split"],
                "model_family": anchor["model_family"],
                "step_number": item["step_number"],
                "timestep": item["timestep"],
                "latent_path": receipt["latent_path"],
                "latent_sha256": receipt["latent_sha256"],
                "audio_path": receipt["audio_path"],
                "audio_sha256": receipt["audio_sha256"],
                "is_final": "false",
                "ace_model_sha256": anchor["ace_model_sha256"],
                "vae_sha256": anchor["vae_sha256"],
                "training_augmentation_kind": item["kind"],
                "local_anchor_sample_id": (
                    item["anchor_sample_id"]
                    if item["kind"] == "local_direction"
                    else ""
                ),
            }
        )
    write_csv_atomic(path, rows)


def build_ltsn_training_augmentation(
    *,
    root: Path,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    trajectories_per_prompt: int = 1,
    perturbations_per_anchor: int = 2,
    rms_ratio: float = 0.005,
    ood_per_prompt: int = 1,
    evaluation_ood_per_prompt: int = 1,
    seed: int = 2026071600,
    duration_seconds: float = 180.0,
    workers: int = 8,
    exact_batch_size: int = 256,
    materialize_mode: str = "auto",
    cleanup_exact_batches: bool = True,
    device_name: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    """Append train-only local/OOD and prompt-held-out evaluation OOD records."""

    if (
        trajectories_per_prompt < 1
        or perturbations_per_anchor < 1
        or ood_per_prompt < 0
        or evaluation_ood_per_prompt < 0
    ):
        raise ValueError("augmentation counts are invalid")
    if rms_ratio not in {0.0025, 0.005, 0.01}:
        raise ValueError("rms_ratio must be 0.0025, 0.005, or 0.01")
    if duration_seconds != 180.0:
        raise ValueError("exact LTSN training augmentation is frozen to 180 seconds")
    root = root.resolve()
    output_dir = output_dir.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    source_manifest_path = resolve(source_manifest_path)
    source_split_manifest_path = resolve(source_split_manifest_path)
    ace_config_path = resolve(ace_config_path)
    fingerprint_path = resolve(fingerprint_path)
    contract = load_fingerprint_contract(fingerprint_path)
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    source_records = read_ltsn_manifest(source_manifest_path, contract)
    source_rows = _read_csv(source_manifest_path)
    if {row.get("ace_model_sha256", "") for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("augmentation ACE model hash differs from the source manifest")
    if {row.get("vae_sha256", "") for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("augmentation VAE hash differs from the source manifest")
    anchors = _select_anchors(source_records, trajectories_per_prompt)
    planned = _augmentation_plan(
        anchors,
        perturbations_per_anchor=perturbations_per_anchor,
        rms_ratio=rms_ratio,
        ood_per_prompt=ood_per_prompt,
        seed=seed,
    )
    planned.extend(
        _evaluation_ood_plan(
            source_records,
            ood_per_prompt=evaluation_ood_per_prompt,
            seed=seed,
        )
    )
    plan_path = output_dir / "training_augmentation_plan.json"
    plan = {
        "schema_version": 3,
        "scope": "train_local_ood_and_heldout_evaluation_ood_v3",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "trajectories_per_prompt": trajectories_per_prompt,
        "perturbations_per_anchor": perturbations_per_anchor,
        "rms_ratio": rms_ratio,
        "ood_per_prompt": ood_per_prompt,
        "evaluation_ood_per_prompt": evaluation_ood_per_prompt,
        "seed": seed,
        "duration_seconds": duration_seconds,
        "exact_batch_size": exact_batch_size,
        "planned": planned,
    }
    if plan_path.is_file():
        if not resume:
            raise FileExistsError("augmentation plan exists; pass --resume or use a new output dir")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("training augmentation plan changed; use a new output dir")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    anchor_by_id = {record.sample_id: record for record in source_records}
    receipts = [
        _materialize_augmentation(
            item=item,
            anchor=anchor_by_id[item["anchor_sample_id"]],
            adapter=adapter,
            output_dir=output_dir,
            plan_sha256=plan_sha256,
        )
        for item in planned
    ]
    source_by_sample = {row["sample_id"]: row for row in source_rows}
    receipt_by_id = {receipt["sample_id"]: receipt for receipt in receipts}
    trajectory_manifest_path = output_dir / "training_augmentation_trajectories.csv"
    _write_augmentation_trajectory_manifest(
        path=trajectory_manifest_path,
        planned=planned,
        source_by_sample=source_by_sample,
        receipt_by_id=receipt_by_id,
    )
    descriptor_path = output_dir / "training_augmentation_descriptors.csv"
    storage_summary = build_exact_snapshot_descriptors(
        project_root=root,
        trajectory_manifest=trajectory_manifest_path,
        work_dir=output_dir / "exact_work",
        output_path=descriptor_path,
        workers=workers,
        batch_size=exact_batch_size,
        materialize_mode=materialize_mode,
        cleanup_batches=cleanup_exact_batches,
        resume=resume,
    )
    descriptors = _read_csv(descriptor_path)
    descriptor_by_id = {row["sample_id"]: row for row in descriptors}

    old_label_path = (
        source_manifest_path.parent / source_rows[0]["exact_label_table_path"]
    ).resolve()
    old_labels = _read_csv(old_label_path)
    label_columns = list(old_labels[0])
    new_labels: list[dict[str, Any]] = []
    augmented_rows: list[dict[str, Any]] = []
    for item in planned:
        descriptor = descriptor_by_id[item["sample_id"]]
        receipt = receipt_by_id[item["sample_id"]]
        if descriptor.get("label_source") != "decoded_snapshot_exact_v1":
            raise LTSNContractError("training augmentation requires decoded exact labels")
        if descriptor.get("audio_sha256") != receipt["audio_sha256"]:
            raise LTSNContractError(
                f"augmentation descriptor audio hash mismatch: {item['sample_id']}"
            )
        forced_ood = item["kind"].startswith("ood_")
        technical_ood = float(descriptor["ood_label"]) >= 0.5
        ood_label = int(forced_ood or technical_ood)
        pitch = json.loads(descriptor["pitch_descriptors_json"])
        score = scorer.score(
            pitch,
            [float(descriptor["acoustic_loop_score"])],
            [float(descriptor["chroma_loop_score"])],
        )
        coordinates_json = json.dumps(score.coordinates[0].tolist(), separators=(",", ":"))
        new_labels.append(
            {
                "sample_id": item["sample_id"],
                "pitch_descriptors_json": json.dumps(pitch, separators=(",", ":")),
                "acoustic_loop_score": descriptor["acoustic_loop_score"],
                "chroma_loop_score": descriptor["chroma_loop_score"],
                "coordinates_json": coordinates_json,
                "focus_logit": float(score.focus_logit[0]),
                "focus_probability": float(score.focus_probability[0]),
                "focus_band_loss": float(score.focus_band_loss[0]),
                "pitch_block_l2_norm": float(score.pitch_block_l2_norm[0]),
                "phase_block_l2_norm": float(score.phase_block_l2_norm[0]),
                "ood_label": ood_label,
                "label_scope": "per_snapshot_exact",
                "fingerprint_json_sha256": contract.artifact_sha256,
            }
        )
        anchor = source_by_sample[item["anchor_sample_id"]]
        augmented = dict(anchor)
        augmented.update(
            {
                "sample_id": item["sample_id"],
                "trajectory_id": item["sample_id"],
                "split": item["split"],
                "step_number": item["step_number"],
                "timestep": item["timestep"],
                "latent_path": os.path.relpath(
                    output_dir / receipt["latent_path"], output_dir
                ).replace("\\", "/"),
                "latent_sha256": receipt["latent_sha256"],
                "coordinates_json": coordinates_json,
                "focus_logit": float(score.focus_logit[0]),
                "ood_label": ood_label,
                "is_final": "false",
                "local_anchor_sample_id": (
                    item["anchor_sample_id"] if item["kind"] == "local_direction" else ""
                ),
                "training_augmentation_kind": item["kind"],
            }
        )
        augmented_rows.append(augmented)

    exact_label_path = output_dir / "exact_snapshot_labels_v3.csv"
    write_csv_atomic(
        exact_label_path,
        _rectangular([*old_labels, *new_labels], label_columns),
    )
    label_sha256 = sha256_file(exact_label_path)
    manifest_path = output_dir / "ltsn_manifest_v3.csv"
    manifest_columns = list(source_rows[0])
    for column in ("local_anchor_sample_id", "training_augmentation_kind"):
        if column not in manifest_columns:
            manifest_columns.append(column)
    for row in source_rows:
        source_latent = (source_manifest_path.parent / row["latent_path"]).resolve()
        row["latent_path"] = os.path.relpath(source_latent, output_dir).replace("\\", "/")
    combined = [*source_rows, *augmented_rows]
    for row in combined:
        row["exact_label_table_path"] = exact_label_path.name
        row["exact_label_table_sha256"] = label_sha256
    write_csv_atomic(manifest_path, _rectangular(combined, manifest_columns))
    split_payload = json.loads(source_split_manifest_path.read_text(encoding="utf-8"))
    split_payload.update(
        {
            "schema_version": 3,
            "training_augmentation_scope": "train_local_ood_and_heldout_evaluation_ood",
            "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
            "augmentation_plan_sha256": plan_sha256,
            "training_manifest_sha256": sha256_file(manifest_path),
        }
    )
    split_path = output_dir / "split_manifest_v3.json"
    write_json_atomic(split_path, split_payload)
    summary = {
        "schema_version": 3,
        "scope": "train_local_ood_and_heldout_evaluation_ood_v3",
        "source_samples": len(source_rows),
        "augmentation_samples": len(augmented_rows),
        "local_direction_samples": sum(
            item["kind"] == "local_direction" for item in planned
        ),
        "ood_samples": sum(item["kind"].startswith("ood_") for item in planned),
        "evaluation_ood_samples": sum(
            item["split"] in {"calibration", "qualification"} for item in planned
        ),
        "ood_samples_by_split": {
            split: sum(
                item["split"] == split and item["kind"].startswith("ood_")
                for item in planned
            )
            for split in ("train", "calibration", "qualification")
        },
        "prompts": len({item["prompt_id"] for item in planned}),
        "plan_sha256": plan_sha256,
        "trajectory_manifest_sha256": sha256_file(trajectory_manifest_path),
        "descriptor_table_sha256": sha256_file(descriptor_path),
        "exact_storage": storage_summary,
        "exact_label_table_sha256": label_sha256,
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "configuration_sha256": canonical_json_sha256(
            {
                "trajectories_per_prompt": trajectories_per_prompt,
                "perturbations_per_anchor": perturbations_per_anchor,
                "rms_ratio": rms_ratio,
                "ood_per_prompt": ood_per_prompt,
                "evaluation_ood_per_prompt": evaluation_ood_per_prompt,
                "seed": seed,
                "exact_batch_size": exact_batch_size,
            }
        ),
    }
    write_json_atomic(output_dir / "training_augmentation_summary.json", summary)
    return summary
