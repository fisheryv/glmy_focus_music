"""V5.2a multi-direction identifiability data collection and frozen views."""

from __future__ import annotations

import csv
import hashlib
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
from .ltsn_v51b import nested_stratified_anchor_order
from .ltsn_v51c import _choose_shift
from .path_homology_exact_scorer import ExactPathHomologyScorer

V52A_SCOPE = "multi_direction_identifiability_v5_2a"
V52A_KIND = "v52a_multi_direction_symmetric"
DEFAULT_DIRECTION_COUNT = 8
DEFAULT_TRAIN_DIRECTION_COUNT = 6
DEFAULT_TRAIN_ANCHORS = 32
DEFAULT_UNSEEN_ANCHORS = 16
DEFAULT_RMS_RATIOS = (0.0025, 0.005)
EXACT_MARGIN = 1e-5


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
    return np.asarray(
        sum(weight * padded[offset : offset + shape[0]] for offset, weight in enumerate(kernel)),
        dtype=np.float32,
    )


def _direction_seed(seed: int, anchor_id: str, attempt: int) -> int:
    digest = hashlib.sha256(f"{seed}:{anchor_id}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def mcnemar_exact(
    true_correct: Mapping[str, bool], control_correct: Mapping[str, bool]
) -> dict[str, Any]:
    """Compare paired decisions with the two-sided exact McNemar test."""

    if set(true_correct) != set(control_correct) or not true_correct:
        raise ValueError("McNemar inputs must contain the same non-empty pair IDs")
    true_only = sum(true_correct[key] and not control_correct[key] for key in true_correct)
    control_only = sum(control_correct[key] and not true_correct[key] for key in true_correct)
    discordant = true_only + control_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value) for value in range(0, min(true_only, control_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "true_only_correct": true_only,
        "control_only_correct": control_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def orthogonal_smooth_directions(
    latent: np.ndarray,
    *,
    anchor_id: str,
    count: int = DEFAULT_DIRECTION_COUNT,
    seed: int = 20260910,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Return deterministic unit-RMS smooth directions using two-pass Gram-Schmidt."""

    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
        raise LTSNContractError("V5.2a anchor latent must be finite with shape [T,64]")
    if count < 2:
        raise ValueError("V5.2a requires at least two directions")
    directions: list[np.ndarray] = []
    direction_seeds: list[int] = []
    attempts = 0
    while len(directions) < count and attempts < count * 32:
        candidate_seed = _direction_seed(seed, anchor_id, attempts)
        candidate = _smooth_direction(latent.shape, candidate_seed).astype(np.float64)
        attempts += 1
        for _ in range(2):
            for previous in directions:
                previous64 = previous.astype(np.float64, copy=False)
                denominator = float(np.sum(previous64 * previous64, dtype=np.float64))
                candidate -= previous64 * (
                    float(np.sum(candidate * previous64, dtype=np.float64)) / denominator
                )
        rms = float(np.sqrt(np.mean(np.square(candidate), dtype=np.float64)))
        if not math.isfinite(rms) or rms <= 1e-8:
            continue
        directions.append((candidate / rms).astype(np.float32))
        direction_seeds.append(candidate_seed)
    if len(directions) != count:
        raise LTSNContractError("V5.2a could not construct the frozen direction basis")
    flattened = [value.astype(np.float64, copy=False).reshape(-1) for value in directions]
    cosines = [
        float(
            np.dot(flattened[left], flattened[right])
            / math.sqrt(
                float(np.dot(flattened[left], flattened[left]))
                * float(np.dot(flattened[right], flattened[right]))
            )
        )
        for left in range(len(flattened))
        for right in range(left + 1, len(flattened))
    ]
    maximum_cosine = max((abs(value) for value in cosines), default=0.0)
    if maximum_cosine > 1e-5:
        raise LTSNContractError(f"V5.2a direction basis is not orthogonal enough: {maximum_cosine}")
    return directions, {
        "direction_count": count,
        "construction": "smooth_gaussian_two_pass_gram_schmidt",
        "attempts": attempts,
        "direction_seeds": direction_seeds,
        "maximum_absolute_pairwise_cosine": maximum_cosine,
        "direction_sha256": [
            hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in directions
        ],
    }


def _select_anchor_rows(
    source_rows: Sequence[Mapping[str, str]],
    evidence_rows: Sequence[Mapping[str, str]],
    *,
    train_anchor_count: int,
    unseen_anchor_count: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    train_order, train_audit = nested_stratified_anchor_order(evidence_rows, split="train")
    unseen_order, unseen_audit = nested_stratified_anchor_order(evidence_rows, split="development")
    if train_anchor_count > len(train_order) or unseen_anchor_count > len(unseen_order):
        raise LTSNContractError("V5.2a requested more anchors than V5.1a provides")
    selected = {
        **{anchor_id: "train_anchor" for anchor_id in train_order[:train_anchor_count]},
        **{anchor_id: "unseen_anchor" for anchor_id in unseen_order[:unseen_anchor_count]},
    }
    by_sample = {row["sample_id"]: dict(row) for row in source_rows}
    missing = set(selected) - set(by_sample)
    if missing:
        raise LTSNContractError(f"V5.2a source manifest is missing anchors: {sorted(missing)}")
    rows = []
    for anchor_id, partition in selected.items():
        row = by_sample[anchor_id]
        if row.get("local_direction_group_id", ""):
            raise LTSNContractError(f"V5.2a selected a perturbation as anchor: {anchor_id}")
        row["v52a_anchor_partition"] = partition
        rows.append(row)
    return rows, {
        "train_anchor_ids": train_order[:train_anchor_count],
        "unseen_anchor_ids": unseen_order[:unseen_anchor_count],
        "source_train_audit": train_audit,
        "source_unseen_audit": unseen_audit,
    }


def _plan_items(
    anchor_rows: Sequence[Mapping[str, str]],
    *,
    direction_count: int,
    train_direction_count: int,
    rms_ratios: Sequence[float],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    direction_audit: dict[str, Any] = {}
    for anchor in anchor_rows:
        latent_path = Path(anchor["_resolved_latent_path"])
        latent = np.load(latent_path, allow_pickle=False).astype(np.float32, copy=False)
        _, audit = orthogonal_smooth_directions(
            latent,
            anchor_id=anchor["sample_id"],
            count=direction_count,
            seed=seed,
        )
        direction_audit[anchor["sample_id"]] = audit
        anchor_partition = anchor["v52a_anchor_partition"]
        for direction_index in range(direction_count):
            if anchor_partition == "unseen_anchor":
                direction_partition = "unseen_anchor"
            elif direction_index < train_direction_count:
                direction_partition = "train_direction"
            else:
                direction_partition = "heldout_direction"
            for rms_ratio in rms_ratios:
                rms_tag = int(round(float(rms_ratio) * 10_000))
                group_id = f"{anchor['sample_id']}__v52a_d{direction_index:02d}_r{rms_tag:04d}"
                for sign, sign_tag in ((-1.0, "minus"), (1.0, "plus")):
                    planned.append(
                        {
                            "sample_id": f"{group_id}__{sign_tag}",
                            "anchor_sample_id": anchor["sample_id"],
                            "prompt_id": anchor["prompt_id"],
                            "step_number": int(anchor["step_number"]),
                            "timestep": float(anchor["timestep"]),
                            "anchor_partition": anchor_partition,
                            "direction_partition": direction_partition,
                            "direction_index": direction_index,
                            "direction_seed": audit["direction_seeds"][direction_index],
                            "rms_ratio": float(rms_ratio),
                            "sign": sign,
                            "direction_group_id": group_id,
                            "kind": V52A_KIND,
                        }
                    )
    return planned, direction_audit


def _validate_receipt(path: Path, output_dir: Path, plan_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_sha256") != plan_sha256:
        raise LTSNContractError("V5.2a receipt belongs to another plan")
    for name in ("latent", "audio"):
        artifact = output_dir / payload[f"{name}_path"]
        if not artifact.is_file() or sha256_file(artifact) != payload[f"{name}_sha256"]:
            raise LTSNContractError(f"V5.2a receipt {name} is hash-mismatched")
    return payload


def _materialize(
    *,
    anchor_rows: Sequence[Mapping[str, str]],
    planned: Sequence[Mapping[str, Any]],
    adapter: Any,
    output_dir: Path,
    plan_sha256: str,
    direction_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    items_by_anchor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in planned:
        items_by_anchor[str(item["anchor_sample_id"])].append(item)
    receipts = []
    for anchor in anchor_rows:
        anchor_id = anchor["sample_id"]
        latent = np.load(anchor["_resolved_latent_path"], allow_pickle=False).astype(
            np.float32, copy=False
        )
        directions, _ = orthogonal_smooth_directions(
            latent, anchor_id=anchor_id, count=direction_count, seed=seed
        )
        latent_rms = float(np.sqrt(np.mean(np.square(latent), dtype=np.float64)))
        if latent_rms <= 0:
            raise LTSNContractError(f"V5.2a anchor has zero latent RMS: {anchor_id}")
        for item in items_by_anchor[anchor_id]:
            receipt_path = output_dir / "receipts" / f"{item['sample_id']}.json"
            if receipt_path.is_file():
                receipt = _validate_receipt(receipt_path, output_dir, plan_sha256)
                if receipt.get("sample_id") != item["sample_id"]:
                    raise LTSNContractError("V5.2a receipt sample binding mismatch")
                receipts.append(receipt)
                continue
            direction = directions[int(item["direction_index"])]
            augmented = latent + (
                float(item["sign"]) * float(item["rms_ratio"]) * latent_rms * direction
            )
            if not np.isfinite(augmented).all():
                raise LTSNContractError("V5.2a perturbation produced NaN or Inf")
            latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
            audio_path = output_dir / "data_raw" / "v52a" / f"{item['sample_id']}.wav"
            if latent_path.is_file():
                existing = np.load(latent_path, allow_pickle=False)
                if existing.dtype != np.float32 or not np.array_equal(
                    existing, augmented.astype(np.float32)
                ):
                    raise LTSNContractError("V5.2a unreceipted latent is mismatched")
            else:
                _save_npy_atomic(latent_path, augmented)
            adapter.decode_latent_to_audio(augmented.astype(np.float32), audio_path)
            receipt = {
                "schema_version": 1,
                "sample_id": item["sample_id"],
                "anchor_sample_id": anchor_id,
                "direction_index": item["direction_index"],
                "direction_seed": item["direction_seed"],
                "rms_ratio": item["rms_ratio"],
                "sign": item["sign"],
                "latent_path": latent_path.relative_to(output_dir).as_posix(),
                "latent_sha256": sha256_file(latent_path),
                "audio_path": audio_path.relative_to(output_dir).as_posix(),
                "audio_sha256": sha256_file(audio_path),
                "plan_sha256": plan_sha256,
            }
            write_json_atomic(receipt_path, receipt)
            receipts.append(receipt)
    return receipts


def _write_trajectory_manifest(
    *,
    path: Path,
    planned: Sequence[Mapping[str, Any]],
    anchor_by_id: Mapping[str, Mapping[str, str]],
    receipt_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    rows = []
    for item in planned:
        anchor = anchor_by_id[str(item["anchor_sample_id"])]
        receipt = receipt_by_id[str(item["sample_id"])]
        rows.append(
            {
                "sample_id": item["sample_id"],
                "prompt_id": item["prompt_id"],
                "trajectory_id": item["sample_id"],
                "split": "train"
                if item["direction_partition"] == "train_direction"
                else "development",
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
                "training_augmentation_kind": V52A_KIND,
                "local_anchor_sample_id": "",
            }
        )
    write_csv_atomic(path, rows)


def _rectangular(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def _build_exact_artifacts(
    *,
    source_manifest_path: Path,
    anchor_rows: Sequence[Mapping[str, str]],
    planned: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    fingerprint_path: Path,
    output_dir: Path,
    project_root: Path,
    workers: int,
    exact_batch_size: int,
    materialize_mode: str,
    cleanup_exact_batches: bool,
    resume: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    anchor_by_id = {row["sample_id"]: row for row in anchor_rows}
    receipt_by_id = {str(row["sample_id"]): row for row in receipts}
    trajectory_path = output_dir / "v52a_trajectories.csv"
    _write_trajectory_manifest(
        path=trajectory_path,
        planned=planned,
        anchor_by_id=anchor_by_id,
        receipt_by_id=receipt_by_id,
    )
    descriptor_path = output_dir / "v52a_descriptors.csv"
    storage = build_exact_snapshot_descriptors(
        project_root=project_root,
        trajectory_manifest=trajectory_path,
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
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    contract = load_fingerprint_contract(fingerprint_path)
    source_rows = _read_csv(source_manifest_path)
    old_label_path = (
        source_manifest_path.parent / source_rows[0]["exact_label_table_path"]
    ).resolve()
    label_columns = list(_read_csv(old_label_path)[0])
    new_labels: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    evidence_inputs: list[dict[str, Any]] = []
    for item in planned:
        descriptor = descriptor_by_id[str(item["sample_id"])]
        receipt = receipt_by_id[str(item["sample_id"])]
        if descriptor.get("audio_sha256") != receipt["audio_sha256"]:
            raise LTSNContractError("V5.2a descriptor audio hash mismatch")
        pitch = json.loads(descriptor["pitch_descriptors_json"])
        score = scorer.score(
            pitch,
            [float(descriptor["acoustic_loop_score"])],
            [float(descriptor["chroma_loop_score"])],
        )
        coordinates_json = json.dumps(score.coordinates[0].tolist(), separators=(",", ":"))
        label = {
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
            "ood_label": int(float(descriptor["ood_label"]) >= 0.5),
            "label_scope": "per_snapshot_exact",
            "fingerprint_json_sha256": contract.artifact_sha256,
        }
        new_labels.append(label)
        anchor = anchor_by_id[str(item["anchor_sample_id"])]
        row = {name: value for name, value in anchor.items() if not name.startswith("_")}
        row.update(
            {
                "sample_id": item["sample_id"],
                "trajectory_id": item["sample_id"],
                "split": "train"
                if item["direction_partition"] == "train_direction"
                else "development",
                "step_number": item["step_number"],
                "timestep": item["timestep"],
                "latent_path": receipt["latent_path"],
                "latent_sha256": receipt["latent_sha256"],
                "coordinates_json": coordinates_json,
                "focus_logit": label["focus_logit"],
                "ood_label": label["ood_label"],
                "is_final": "false",
                "local_anchor_sample_id": "",
                "local_direction_group_id": item["direction_group_id"],
                "local_direction_sign": item["sign"],
                "local_direction_rms_ratio": item["rms_ratio"],
                "training_augmentation_kind": V52A_KIND,
                "v52a_anchor_partition": item["anchor_partition"],
                "v52a_direction_partition": item["direction_partition"],
                "v52a_direction_index": item["direction_index"],
                "v52a_direction_seed": item["direction_seed"],
            }
        )
        manifest_rows.append(row)
        evidence_inputs.append(
            {
                **item,
                "exact_band_loss": label["focus_band_loss"],
                "ood_label": label["ood_label"],
            }
        )
    label_path = output_dir / "exact_snapshot_labels_v52a.csv"
    write_csv_atomic(label_path, _rectangular(new_labels, label_columns))
    label_sha256 = sha256_file(label_path)
    manifest_columns = list(manifest_rows[0])
    for row in manifest_rows:
        row["exact_label_table_path"] = label_path.name
        row["exact_label_table_sha256"] = label_sha256
    master_path = output_dir / "ltsn_manifest_v52a_master.csv"
    write_csv_atomic(master_path, _rectangular(manifest_rows, manifest_columns))
    evidence: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_inputs:
        grouped[str(row["direction_group_id"])].append(row)
    for group_id, rows in sorted(grouped.items()):
        by_sign = {float(row["sign"]): row for row in rows}
        if set(by_sign) != {-1.0, 1.0} or len(rows) != 2:
            raise LTSNContractError(f"V5.2a incomplete central pair: {group_id}")
        minus, plus = by_sign[-1.0], by_sign[1.0]
        difference = float(minus["exact_band_loss"]) - float(plus["exact_band_loss"])
        evidence.append(
            {
                "direction_group_id": group_id,
                "anchor_sample_id": minus["anchor_sample_id"],
                "anchor_partition": minus["anchor_partition"],
                "direction_partition": minus["direction_partition"],
                "direction_index": minus["direction_index"],
                "direction_seed": minus["direction_seed"],
                "step_number": minus["step_number"],
                "rms_ratio": minus["rms_ratio"],
                "minus_sample_id": minus["sample_id"],
                "plus_sample_id": plus["sample_id"],
                "exact_loss_minus": minus["exact_band_loss"],
                "exact_loss_plus": plus["exact_band_loss"],
                "exact_derivative": difference / (2.0 * float(minus["rms_ratio"])),
                "informative": abs(difference) >= EXACT_MARGIN,
                "pair_in_distribution": (
                    int(minus["ood_label"]) == 0 and int(plus["ood_label"]) == 0
                ),
            }
        )
    evidence_path = output_dir / "central_direction_exact_evidence_v52a.csv"
    write_csv_atomic(evidence_path, evidence)
    return (
        master_path,
        evidence_path,
        label_path,
        {
            "trajectory_manifest_sha256": sha256_file(trajectory_path),
            "descriptor_table_sha256": sha256_file(descriptor_path),
            "master_manifest_sha256": sha256_file(master_path),
            "central_direction_evidence_sha256": sha256_file(evidence_path),
            "exact_label_table_sha256": label_sha256,
            "exact_storage": storage,
        },
    )


def _relative_rows(
    rows: Sequence[Mapping[str, Any]], *, source_dir: Path, output_dir: Path
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        for name in ("latent_path", "exact_label_table_path"):
            path = (source_dir / str(row[name])).resolve()
            if not path.is_file():
                raise LTSNContractError(f"V5.2a view source is missing: {path}")
            row[name] = os.path.relpath(path, output_dir).replace("\\", "/")
        output.append(row)
    return output


def _write_view(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    source_dir: Path,
    scope: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relative = _relative_rows(rows, source_dir=source_dir, output_dir=output_dir)
    manifest_path = output_dir / "ltsn_manifest_v52a.csv"
    write_csv_atomic(manifest_path, relative)
    assignments = dict(sorted({row["prompt_id"]: row["split"] for row in relative}.items()))
    split = {
        "schema_version": 520,
        "grouping": "prompt_id+trajectory_id",
        "assignments": assignments,
        "training_augmentation_scope": scope,
        "training_manifest_sha256": sha256_file(manifest_path),
    }
    split_path = output_dir / "split_manifest_v52a.json"
    write_json_atomic(split_path, split)
    return {
        "scope": scope,
        "manifest": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "rows": len(relative),
    }


def build_v52a_views(
    *, master_manifest_path: Path, evidence_path: Path, output_root: Path
) -> dict[str, Any]:
    """Freeze true/control training views and the seen-anchor direction holdout."""

    master_manifest_path = master_manifest_path.resolve()
    evidence_path = evidence_path.resolve()
    output_root = output_root.resolve()
    master_rows = _read_csv(master_manifest_path)
    evidence = _read_csv(evidence_path)
    by_sample = {row["sample_id"]: row for row in master_rows}
    informative = [
        row
        for row in evidence
        if row["informative"].lower() == "true" and row["pair_in_distribution"].lower() == "true"
    ]
    by_partition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in informative:
        by_partition[row["direction_partition"]].append(row)

    def rows_for(pairs: Sequence[Mapping[str, str]], split: str) -> list[dict[str, Any]]:
        rows = []
        for pair in sorted(pairs, key=lambda value: value["direction_group_id"]):
            for sample_id in (pair["minus_sample_id"], pair["plus_sample_id"]):
                row = dict(by_sample[sample_id])
                row["split"] = split
                rows.append(row)
        return rows

    train_rows = rows_for(by_partition["train_direction"], "train")
    unseen_rows = rows_for(by_partition["unseen_anchor"], "development")
    seen_rows = rows_for(by_partition["heldout_direction"], "development")
    true_rows = [*train_rows, *unseen_rows]
    true_view = _write_view(
        output_dir=output_root / "true_pairs",
        rows=true_rows,
        source_dir=master_manifest_path.parent,
        scope=V52A_SCOPE,
    )
    seen_view = _write_view(
        output_dir=output_root / "seen_anchor_heldout_direction",
        rows=seen_rows,
        source_dir=master_manifest_path.parent,
        scope=f"{V52A_SCOPE}_seen_anchor_heldout_direction",
    )

    control_rows = [dict(row) for row in true_rows]
    control_by_sample = {row["sample_id"]: row for row in control_rows}
    strata: dict[tuple[int, float, int], list[dict[str, str]]] = defaultdict(list)
    for pair in by_partition["train_direction"]:
        strata[
            (
                int(pair["step_number"]),
                float(pair["rms_ratio"]),
                int(pair["direction_index"]),
            )
        ].append(pair)
    pair_map = []
    for (step, rms, direction_index), pairs in sorted(strata.items()):
        ordered = sorted(
            pairs,
            key=lambda row: hashlib.sha256(
                f"20260910:{row['direction_group_id']}".encode()
            ).hexdigest(),
        )
        shift, _ = _choose_shift(ordered)
        for index, minus_source in enumerate(ordered):
            plus_source = ordered[(index + shift) % len(ordered)]
            group_id = f"v52a_perm_s{step}_d{direction_index:02d}_r{rms:g}_{index:03d}"
            minus_id = minus_source["minus_sample_id"]
            plus_id = plus_source["plus_sample_id"]
            for sample_id, sign in ((minus_id, -1), (plus_id, 1)):
                row = control_by_sample[sample_id]
                row["local_direction_group_id"] = group_id
                row["local_direction_sign"] = str(sign)
                row["local_direction_rms_ratio"] = str(rms)
                row["local_anchor_sample_id"] = ""
            difference = float(minus_source["exact_loss_minus"]) - float(
                plus_source["exact_loss_plus"]
            )
            pair_map.append(
                {
                    "permuted_group_id": group_id,
                    "step_number": step,
                    "direction_index": direction_index,
                    "rms_ratio": rms,
                    "minus_sample_id": minus_id,
                    "minus_source_group_id": minus_source["direction_group_id"],
                    "plus_sample_id": plus_id,
                    "plus_source_group_id": plus_source["direction_group_id"],
                    "exact_loss_difference": difference,
                    "informative": abs(difference) >= EXACT_MARGIN,
                }
            )
    control_view = _write_view(
        output_dir=output_root / "permuted_pair_control",
        rows=control_rows,
        source_dir=master_manifest_path.parent,
        scope=f"{V52A_SCOPE}_permuted_pair_control",
    )
    pair_map_path = output_root / "permuted_pair_control" / "permuted_pair_map_v52a.csv"
    write_csv_atomic(pair_map_path, pair_map)
    control_view["pair_map_sha256"] = sha256_file(pair_map_path)
    control_view["permuted_train_pairs"] = len(pair_map)
    control_view["informative_permuted_train_pairs"] = sum(
        bool(row["informative"]) for row in pair_map
    )
    if control_view["informative_permuted_train_pairs"] != len(pair_map):
        raise LTSNContractError("V5.2a control permutation introduced tied pairs")
    payload = {
        "schema_version": 1,
        "experiment": "ltsn_v5_2a_multi_direction_identifiability",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "exact_margin": EXACT_MARGIN,
        "master_manifest_sha256": sha256_file(master_manifest_path),
        "central_direction_evidence_sha256": sha256_file(evidence_path),
        "planned_pairs_by_partition": {
            partition: sum(row["direction_partition"] == partition for row in evidence)
            for partition in ("train_direction", "heldout_direction", "unseen_anchor")
        },
        "informative_pairs_by_partition": {
            partition: len(by_partition[partition])
            for partition in ("train_direction", "heldout_direction", "unseen_anchor")
        },
        "views": {
            "true_pairs": true_view,
            "seen_anchor_heldout_direction": seen_view,
            "permuted_pair_control": control_view,
        },
    }
    write_json_atomic(output_root / "v52a_views.json", payload)
    return payload


def collect_v52a_multidirection(
    *,
    root: Path,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    source_evidence_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    train_anchor_count: int = DEFAULT_TRAIN_ANCHORS,
    unseen_anchor_count: int = DEFAULT_UNSEEN_ANCHORS,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    train_direction_count: int = DEFAULT_TRAIN_DIRECTION_COUNT,
    rms_ratios: Sequence[float] = DEFAULT_RMS_RATIOS,
    seed: int = 20260910,
    workers: int = 8,
    exact_batch_size: int = 256,
    materialize_mode: str = "auto",
    cleanup_exact_batches: bool = True,
    device_name: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    """Decode and exact-score the frozen V5.2a multi-direction experiment."""

    if not 1 < train_direction_count < direction_count:
        raise ValueError("V5.2a train direction count must be between 1 and total count")
    ratios = tuple(sorted(set(float(value) for value in rms_ratios)))
    if ratios != DEFAULT_RMS_RATIOS:
        raise ValueError("V5.2a RMS ratios are frozen to 0.0025 and 0.005")
    root = root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    source_manifest_path = resolve(source_manifest_path)
    source_split_manifest_path = resolve(source_split_manifest_path)
    source_evidence_path = resolve(source_evidence_path)
    ace_config_path = resolve(ace_config_path)
    fingerprint_path = resolve(fingerprint_path)
    output_dir = resolve(output_dir)
    source_rows = _read_csv(source_manifest_path)
    evidence_rows = _read_csv(source_evidence_path)
    if {row.get("ace_model_sha256", "") for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("V5.2a ACE model hash differs from the source manifest")
    if {row.get("vae_sha256", "") for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("V5.2a VAE hash differs from the source manifest")
    anchor_rows, selection = _select_anchor_rows(
        source_rows,
        evidence_rows,
        train_anchor_count=train_anchor_count,
        unseen_anchor_count=unseen_anchor_count,
    )
    for row in anchor_rows:
        latent_path = (source_manifest_path.parent / row["latent_path"]).resolve()
        if not latent_path.is_file() or sha256_file(latent_path) != row["latent_sha256"]:
            raise LTSNContractError(f"V5.2a anchor latent is missing: {row['sample_id']}")
        row["_resolved_latent_path"] = str(latent_path)
    planned, direction_audit = _plan_items(
        anchor_rows,
        direction_count=direction_count,
        train_direction_count=train_direction_count,
        rms_ratios=ratios,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "scope": V52A_SCOPE,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "source_evidence_sha256": sha256_file(source_evidence_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_json_sha256": sha256_file(fingerprint_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "train_anchor_count": train_anchor_count,
        "unseen_anchor_count": unseen_anchor_count,
        "direction_count": direction_count,
        "train_direction_count": train_direction_count,
        "heldout_direction_count": direction_count - train_direction_count,
        "rms_ratios": list(ratios),
        "seed": seed,
        "selection": selection,
        "direction_audit": direction_audit,
        "planned": planned,
    }
    plan_path = output_dir / "v52a_generation_plan.json"
    if plan_path.is_file():
        if not resume or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("V5.2a plan changed; use a new output directory")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    from .ace_adapter import AceStepAdapter
    from .experiment import load_experiment_config

    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    receipts = _materialize(
        anchor_rows=anchor_rows,
        planned=planned,
        adapter=adapter,
        output_dir=output_dir,
        plan_sha256=plan_sha256,
        direction_count=direction_count,
        seed=seed,
    )
    master_path, central_path, label_path, exact_summary = _build_exact_artifacts(
        source_manifest_path=source_manifest_path,
        anchor_rows=anchor_rows,
        planned=planned,
        receipts=receipts,
        fingerprint_path=fingerprint_path,
        output_dir=output_dir,
        project_root=root,
        workers=workers,
        exact_batch_size=exact_batch_size,
        materialize_mode=materialize_mode,
        cleanup_exact_batches=cleanup_exact_batches,
        resume=resume,
    )
    views = build_v52a_views(
        master_manifest_path=master_path,
        evidence_path=central_path,
        output_root=output_dir / "views",
    )
    summary = {
        "schema_version": 1,
        "scope": V52A_SCOPE,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "generation_plan_sha256": plan_sha256,
        "configuration_sha256": canonical_json_sha256(
            {
                "train_anchor_count": train_anchor_count,
                "unseen_anchor_count": unseen_anchor_count,
                "direction_count": direction_count,
                "train_direction_count": train_direction_count,
                "rms_ratios": list(ratios),
                "seed": seed,
            }
        ),
        "samples": len(planned),
        "exact_artifacts": exact_summary,
        "exact_label_table": str(label_path),
        "views_summary_sha256": sha256_file(output_dir / "views" / "v52a_views.json"),
        "informative_pairs_by_partition": views["informative_pairs_by_partition"],
    }
    write_json_atomic(output_dir / "v52a_collection_summary.json", summary)
    return summary
