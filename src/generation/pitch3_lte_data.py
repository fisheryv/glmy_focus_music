"""Auditable prompt states and local finite-difference data for V3-LTE."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import AceStepAdapter
from .experiment import load_experiment_config
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import canonical_json_sha256, write_csv_atomic, write_json_atomic
from .pitch3_contract import Pitch3Contract, load_pitch3_contract
from .pitch3_exact_scorer import ExactPitch3Scorer

LTE_SCHEMA_VERSION = 1
LTE_MODEL_FAMILY = "pitch3_prompt_conditioned_local_energy_v3"
LTE_RADIUS_RATIO = 0.05
LTE_DIRECTIONS_PER_PROMPT = 2


def pitch3_lte_prompt_shard(prompt_id: str, shard_count: int) -> int:
    """Assign one complete prompt group to a deterministic process shard."""

    if shard_count < 1:
        raise ValueError("V3-LTE shard_count must be positive")
    digest = hashlib.sha256(f"v3-lte-data-shard|{prompt_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _rooted(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def _save_npz_atomic(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path, parent).replace("\\", "/")


def _caption_sha256(caption: str) -> str:
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def build_pitch3_lte_prompt_embeddings(
    *,
    root: Path,
    prompt_manifest_path: Path,
    source_manifest_path: Path,
    ace_config_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    duration_seconds: float = 180.0,
    device_name: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    """Cache frozen ACE text-token states for every formal prompt."""

    root = root.resolve()
    prompt_manifest_path = _rooted(prompt_manifest_path, root)
    source_manifest_path = _rooted(source_manifest_path, root)
    ace_config_path = _rooted(ace_config_path, root)
    output_dir = _rooted(output_dir, root)
    prompts = _read_csv(prompt_manifest_path)
    source = _read_csv(source_manifest_path)
    if not prompts or not source:
        raise LTSNContractError("V3-LTE prompt/source manifest is empty")
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise LTSNContractError("V3-LTE prompt manifest has duplicate prompt IDs")
    prompt_ids = {row["prompt_id"] for row in source if float(row.get("ood_label", 0.0)) < 0.5}
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    if not prompt_ids.issubset(prompt_by_id):
        raise LTSNContractError("V3-LTE source manifest references unknown prompts")
    observed_model_hashes = {row.get("ace_model_sha256", "").lower() for row in source}
    if observed_model_hashes != {ace_model_sha256.lower()}:
        raise LTSNContractError("V3-LTE ACE model hash differs from the latent source")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("V3-LTE duration must be positive and finite")
    plan = {
        "schema_version": LTE_SCHEMA_VERSION,
        "stage": "pitch3_lte_prompt_embeddings",
        "model_family": LTE_MODEL_FAMILY,
        "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "ace_model_sha256": ace_model_sha256.lower(),
        "duration_seconds": float(duration_seconds),
        "prompt_ids": sorted(prompt_ids),
        "encoder_contract": "ace_step_1_5_build_dit_inputs_get_text_hidden_states_v1",
        "prompt_id_embedding_forbidden": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "pitch3_lte_prompt_embedding_plan.json"
    if plan_path.is_file():
        if not resume or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("V3-LTE prompt plan changed; use a new output directory")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    rows: list[dict[str, Any]] = []
    for prompt_id in sorted(prompt_ids):
        prompt = prompt_by_id[prompt_id]
        path = output_dir / "embeddings" / f"{prompt_id}.npz"
        caption = prompt.get("caption", "").strip()
        if not caption:
            raise LTSNContractError(f"V3-LTE prompt has empty caption: {prompt_id}")
        if not path.is_file():
            hidden, mask, formatted = adapter.encode_generation_prompt(
                caption,
                duration_seconds=duration_seconds,
                bpm=int(prompt["bpm"]) if prompt.get("bpm", "").strip() else None,
                keyscale=prompt.get("keyscale", ""),
                timesignature=prompt.get("timesignature", ""),
            )
            hidden_np = hidden.squeeze(0).numpy().astype(np.float32, copy=False)
            mask_np = mask.squeeze(0).numpy().astype(np.bool_, copy=False)
            if hidden_np.ndim != 2 or hidden_np.shape[1] != 1024:
                raise LTSNContractError("ACE text encoder returned an unexpected hidden shape")
            if mask_np.shape != hidden_np.shape[:1] or not mask_np.any():
                raise LTSNContractError("ACE text encoder returned an invalid attention mask")
            _save_npz_atomic(
                path,
                hidden=hidden_np,
                mask=mask_np,
                formatted_prompt=np.asarray(formatted),
            )
        with np.load(path, allow_pickle=False) as payload:
            hidden_np = payload["hidden"]
            mask_np = payload["mask"]
            formatted = str(payload["formatted_prompt"].item())
        if (
            hidden_np.ndim != 2
            or hidden_np.shape[1] != 1024
            or mask_np.shape != hidden_np.shape[:1]
        ):
            raise LTSNContractError(f"invalid resumed V3-LTE prompt embedding: {prompt_id}")
        rows.append(
            {
                "prompt_id": prompt_id,
                "split": prompt["split"],
                "caption_sha256": _caption_sha256(caption),
                "formatted_prompt_sha256": _caption_sha256(formatted),
                "embedding_path": _relative(path, output_dir),
                "embedding_sha256": sha256_file(path),
                "token_count": int(mask_np.sum()),
                "hidden_dimension": int(hidden_np.shape[1]),
                "ace_model_sha256": ace_model_sha256.lower(),
                "plan_sha256": plan_sha256,
            }
        )
    manifest_path = output_dir / "pitch3_lte_prompt_embeddings.csv"
    write_csv_atomic(manifest_path, rows)
    result = {
        "schema_version": LTE_SCHEMA_VERSION,
        "stage": "pitch3_lte_prompt_embeddings",
        "status": "complete",
        "prompts": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "ace_model_sha256": ace_model_sha256.lower(),
        "plan_sha256": plan_sha256,
        "prompt_embedding_manifest": str(manifest_path),
        "prompt_embedding_manifest_sha256": sha256_file(manifest_path),
        "prompt_id_embedding_used": False,
    }
    write_json_atomic(output_dir / "pitch3_lte_prompt_embedding_summary.json", result)
    return result


def _band_loss(coordinates: Sequence[float], contract: Pitch3Contract) -> float:
    values = np.asarray(coordinates, dtype=float)
    lower = np.asarray(contract.target_lower, dtype=float)
    upper = np.asarray(contract.target_upper, dtype=float)
    weights = np.asarray(contract.distance_weights, dtype=float)
    return float(
        ((np.maximum(lower - values, 0) ** 2 + np.maximum(values - upper, 0) ** 2) * weights).sum()
    )


def _validate_prompt_embeddings(
    manifest_path: Path, source_prompts: set[str], ace_model_sha256: str
) -> dict[str, dict[str, str]]:
    rows = _read_csv(manifest_path)
    by_id = {row["prompt_id"]: row for row in rows}
    if not source_prompts.issubset(by_id):
        raise LTSNContractError("V3-LTE prompt embeddings are incomplete")
    for prompt_id in source_prompts:
        row = by_id[prompt_id]
        path = (manifest_path.parent / row["embedding_path"]).resolve()
        if row["ace_model_sha256"].lower() != ace_model_sha256.lower():
            raise LTSNContractError("V3-LTE prompt embedding ACE model hash mismatch")
        if not path.is_file() or sha256_file(path) != row["embedding_sha256"]:
            raise LTSNContractError("V3-LTE prompt embedding is missing or hash-mismatched")
        row["_path"] = str(path)
    return by_id


def _step4_id_groups(
    rows: Sequence[Mapping[str, str]], splits: set[str]
) -> dict[str, list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("split") in splits
            and int(row["step_number"]) == 4
            and float(row.get("ood_label", 0.0)) < 0.5
            and row.get("is_final", "false").lower() != "true"
        ):
            grouped[row["prompt_id"]].append(row)
    for prompt_id, values in grouped.items():
        values.sort(key=lambda row: row["trajectory_id"])
        if len(values) != 4 or len({row["trajectory_id"] for row in values}) != 4:
            raise LTSNContractError(
                f"V3-LTE requires exactly four Step-4 seeds per prompt: {prompt_id}"
            )
        shapes = {
            np.load(Path(str(row["_latent_path"])), mmap_mode="r", allow_pickle=False).shape
            for row in values
        }
        if len(shapes) != 1:
            raise LTSNContractError(
                f"V3-LTE prompt seeds have mismatched latent shapes: {prompt_id}"
            )
    if not grouped:
        raise LTSNContractError("V3-LTE found no Step-4 ID prompt groups")
    return dict(sorted(grouped.items()))


def _anchor_index(prompt_id: str) -> int:
    return (
        int.from_bytes(hashlib.sha256(f"v3-lte-anchor|{prompt_id}".encode()).digest()[:4], "big")
        % 4
    )


def _direction(values: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise LTSNContractError("V3-LTE seed-difference direction is degenerate")
    return np.asarray(values / rms, dtype=np.float32)


def _local_plan(
    groups: Mapping[str, Sequence[Mapping[str, str]]], local_splits: set[str]
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for prompt_id, rows in groups.items():
        split = str(rows[0]["split"])
        if split not in local_splits:
            continue
        anchor = rows[_anchor_index(prompt_id)]
        for direction_index, (left, right) in enumerate(((0, 1), (2, 3))):
            direction_id = f"{prompt_id}__d{direction_index + 1}"
            for sign, name in ((-1.0, "minus"), (1.0, "plus")):
                planned.append(
                    {
                        "sample_id": f"{direction_id}__{name}",
                        "prompt_id": prompt_id,
                        "split": split,
                        "anchor_sample_id": anchor["sample_id"],
                        "anchor_latent_sha256": anchor["latent_sha256"],
                        "anchor_index": _anchor_index(prompt_id),
                        "direction_id": direction_id,
                        "direction_index": direction_index,
                        "direction_left_sample_id": rows[left]["sample_id"],
                        "direction_right_sample_id": rows[right]["sample_id"],
                        "direction_sign": sign,
                        "radius_ratio": LTE_RADIUS_RATIO,
                        "step_number": 4,
                        "timestep": float(anchor["timestep"]),
                    }
                )
    planned.sort(key=lambda row: row["sample_id"])
    return planned


def build_pitch3_lte_dataset(
    *,
    root: Path,
    source_manifest_path: Path,
    prompt_embedding_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    output_dir: Path,
    include_splits: Sequence[str] = ("train", "development"),
    local_splits: Sequence[str] = ("train",),
    workers: int = 8,
    exact_batch_size: int = 64,
    materialize_mode: str = "auto",
    device_name: str = "cuda:0",
    resume: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Build base values plus exact ``z +/- epsilon*d`` finite differences."""

    root = root.resolve()
    source_manifest_path = _rooted(source_manifest_path, root)
    prompt_embedding_manifest_path = _rooted(prompt_embedding_manifest_path, root)
    ace_config_path = _rooted(ace_config_path, root)
    fingerprint_path = _rooted(fingerprint_path, root)
    output_dir = _rooted(output_dir, root)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("V3-LTE shard index/count is invalid")
    include = set(include_splits)
    local = set(local_splits)
    if not include or not local.issubset(include) or not include.issubset({"train", "development"}):
        raise ValueError(
            "V3-LTE splits must be train/development and local splits must be included"
        )
    if workers < 1 or exact_batch_size < 1:
        raise ValueError("V3-LTE exact workers and batch size must be positive")
    contract = load_pitch3_contract(fingerprint_path)
    scorer = ExactPitch3Scorer(contract)
    source_rows = _read_csv(source_manifest_path)
    if not source_rows:
        raise LTSNContractError("V3-LTE source manifest is empty")
    model_hashes = {row.get("ace_model_sha256", "").lower() for row in source_rows}
    vae_hashes = {row.get("vae_sha256", "").lower() for row in source_rows}
    if len(model_hashes) != 1 or len(vae_hashes) != 1:
        raise LTSNContractError("V3-LTE source model/VAE hashes are not frozen")
    ace_model_sha256 = model_hashes.pop()
    vae_sha256 = vae_hashes.pop()
    for row in source_rows:
        if row.get("fingerprint_json_sha256", "").lower() != contract.artifact_sha256:
            raise LTSNContractError("V3-LTE source fingerprint hash mismatch")
        path = (source_manifest_path.parent / row["latent_path"]).resolve()
        if not path.is_file() or sha256_file(path) != row["latent_sha256"]:
            raise LTSNContractError("V3-LTE source latent is missing or hash-mismatched")
        row["_latent_path"] = str(path)
    all_groups = _step4_id_groups(source_rows, include)
    groups = {
        prompt_id: values
        for prompt_id, values in all_groups.items()
        if pitch3_lte_prompt_shard(prompt_id, shard_count) == shard_index
    }
    if not groups:
        raise LTSNContractError(f"V3-LTE data shard {shard_index}/{shard_count} is empty")
    embeddings = _validate_prompt_embeddings(
        prompt_embedding_manifest_path, set(groups), ace_model_sha256
    )
    planned = _local_plan(groups, local)
    expected = (
        sum(
            len(groups[prompt_id]) == 4
            for prompt_id in groups
            if groups[prompt_id][0]["split"] in local
        )
        * 4
    )
    if len(planned) != expected:
        raise LTSNContractError("V3-LTE local plan count changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": LTE_SCHEMA_VERSION,
        "stage": "pitch3_lte_exact_local_dataset",
        "model_family": LTE_MODEL_FAMILY,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "prompt_embedding_manifest_sha256": sha256_file(prompt_embedding_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "include_splits": sorted(include),
        "local_splits": sorted(local),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "prompt_assignment": "sha256(v3-lte-data-shard|prompt_id)-mod-shard_count",
        "prompt_ids": sorted(groups),
        "anchor_rule": "sha256(v3-lte-anchor|prompt_id)-mod-4",
        "direction_rule": ["normalize_rms(z1-z0)", "normalize_rms(z3-z2)"],
        "directions_per_prompt": LTE_DIRECTIONS_PER_PROMPT,
        "radius_ratio": LTE_RADIUS_RATIO,
        "planned_local_samples": len(planned),
        "items_sha256": canonical_json_sha256(planned),
        "wav_policy": "ephemeral_delete_after_each_exact_batch",
    }
    plan_path = output_dir / "pitch3_lte_dataset_plan.json"
    if plan_path.is_file():
        if not resume or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("V3-LTE dataset plan changed; use a new output directory")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    exact_by_id: dict[str, dict[str, str]] = {}
    for batch_index, start in enumerate(range(0, len(planned), exact_batch_size)):
        items = planned[start : start + exact_batch_size]
        batch_dir = output_dir / "exact_batches" / f"batch_{batch_index:05d}"
        descriptor_path = batch_dir / "descriptors.csv"
        completion_path = batch_dir / "completion.json"
        expected_ids = [str(item["sample_id"]) for item in items]
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                completion.get("plan_sha256") != plan_sha256
                or completion.get("sample_ids") != expected_ids
            ):
                raise LTSNContractError("V3-LTE resumed exact batch belongs to another plan")
            if (
                not descriptor_path.is_file()
                or sha256_file(descriptor_path) != completion["descriptor_sha256"]
            ):
                raise LTSNContractError("V3-LTE resumed descriptor hash mismatch")
            rows = _read_csv(descriptor_path)
            if any(
                (batch_dir / "audio" / f"{sample_id}.wav").exists() for sample_id in expected_ids
            ):
                raise LTSNContractError("completed V3-LTE batch retained ephemeral WAV")
            exact_by_id.update({row["sample_id"]: row for row in rows})
            continue
        trajectory_rows: list[dict[str, Any]] = []
        audio_paths: list[Path] = []
        latent_cache: dict[str, np.ndarray] = {}
        for item in items:
            prompt_rows = groups[str(item["prompt_id"])]
            anchor = source_by_id[str(item["anchor_sample_id"])]
            anchor_values = latent_cache.setdefault(
                anchor["sample_id"],
                np.load(anchor["_latent_path"], allow_pickle=False).astype(np.float32, copy=False),
            )
            left = source_by_id[str(item["direction_left_sample_id"])]
            right = source_by_id[str(item["direction_right_sample_id"])]
            left_values = latent_cache.setdefault(
                left["sample_id"],
                np.load(left["_latent_path"], allow_pickle=False).astype(np.float32, copy=False),
            )
            right_values = latent_cache.setdefault(
                right["sample_id"],
                np.load(right["_latent_path"], allow_pickle=False).astype(np.float32, copy=False),
            )
            direction = _direction(right_values - left_values)
            anchor_rms = float(np.sqrt(np.mean(np.square(anchor_values), dtype=np.float64)))
            epsilon = LTE_RADIUS_RATIO * anchor_rms
            perturbed = anchor_values + float(item["direction_sign"]) * epsilon * direction
            latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
            if latent_path.is_file():
                existing = np.load(latent_path, allow_pickle=False)
                if existing.dtype != np.float32 or not np.array_equal(
                    existing, perturbed.astype(np.float32)
                ):
                    raise LTSNContractError("V3-LTE resumed local latent changed")
            else:
                _save_npy_atomic(latent_path, perturbed)
            batch_dir.mkdir(parents=True, exist_ok=True)
            audio_path = batch_dir / "audio" / f"{item['sample_id']}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.unlink(missing_ok=True)
            adapter.decode_latent_to_audio(perturbed, audio_path)
            audio_paths.append(audio_path)
            trajectory_rows.append(
                {
                    "sample_id": item["sample_id"],
                    "prompt_id": item["prompt_id"],
                    "trajectory_id": item["sample_id"],
                    "split": item["split"],
                    "model_family": anchor["model_family"],
                    "step_number": 4,
                    "timestep": item["timestep"],
                    "latent_path": _relative(latent_path, batch_dir),
                    "latent_sha256": sha256_file(latent_path),
                    "audio_path": _relative(audio_path, batch_dir),
                    "audio_sha256": sha256_file(audio_path),
                    "is_final": "false",
                    "ace_model_sha256": ace_model_sha256,
                    "vae_sha256": vae_sha256,
                    "local_anchor_sample_id": anchor["sample_id"],
                    "engineering_smoke": "false",
                    "schema_version": LTE_SCHEMA_VERSION,
                }
            )
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
            )
            rows = _read_csv(descriptor_path)
            if {row["sample_id"] for row in rows} != set(expected_ids):
                raise LTSNContractError("V3-LTE exact descriptor identities changed")
        finally:
            for audio_path in audio_paths:
                audio_path.unlink(missing_ok=True)
        write_json_atomic(
            completion_path,
            {
                "schema_version": LTE_SCHEMA_VERSION,
                "plan_sha256": plan_sha256,
                "sample_ids": expected_ids,
                "descriptor_sha256": sha256_file(descriptor_path),
                "wav_retained": 0,
            },
        )
        exact_by_id.update({row["sample_id"]: row for row in rows})
    if len(exact_by_id) != len(planned):
        raise LTSNContractError("V3-LTE exact local dataset is incomplete")

    examples: list[dict[str, Any]] = []
    for prompt_id, prompt_rows in groups.items():
        embedding = embeddings[prompt_id]
        for row in prompt_rows:
            coordinates = json.loads(row["coordinates_json"])
            band = _band_loss(coordinates, contract)
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt_id": prompt_id,
                    "prompt_family": prompt_id.rsplit("__v", 1)[0],
                    "trajectory_id": row["trajectory_id"],
                    "split": row["split"],
                    "source_kind": "base_step4_seed",
                    "latent_path": _relative(Path(str(row["_latent_path"])), output_dir),
                    "latent_sha256": row["latent_sha256"],
                    "prompt_embedding_path": _relative(Path(embedding["_path"]), output_dir),
                    "prompt_embedding_sha256": embedding["embedding_sha256"],
                    "exact_band": band,
                    "energy_target": math.log1p(band),
                    "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                    "direction_id": "",
                    "direction_sign": 0,
                    "epsilon": 0,
                    "radius_ratio": 0,
                    "step_number": 4,
                    "timestep": row["timestep"],
                    "fingerprint_json_sha256": contract.artifact_sha256,
                    "ace_model_sha256": ace_model_sha256,
                    "vae_sha256": vae_sha256,
                    "dataset_plan_sha256": plan_sha256,
                }
            )
    item_by_id = {str(item["sample_id"]): item for item in planned}
    for sample_id, descriptor in exact_by_id.items():
        item = item_by_id[sample_id]
        score = scorer.score(json.loads(descriptor["pitch_descriptors_json"]))
        band = float(score.target_band_loss[0])
        latent_path = output_dir / "latents" / f"{sample_id}.npy"
        embedding = embeddings[str(item["prompt_id"])]
        anchor = source_by_id[str(item["anchor_sample_id"])]
        anchor_values = np.load(anchor["_latent_path"], mmap_mode="r", allow_pickle=False)
        anchor_rms = float(np.sqrt(np.mean(np.square(anchor_values), dtype=np.float64)))
        examples.append(
            {
                "sample_id": sample_id,
                "prompt_id": item["prompt_id"],
                "prompt_family": str(item["prompt_id"]).rsplit("__v", 1)[0],
                "trajectory_id": item["anchor_sample_id"],
                "split": item["split"],
                "source_kind": "local_finite_difference",
                "latent_path": _relative(latent_path, output_dir),
                "latent_sha256": sha256_file(latent_path),
                "prompt_embedding_path": _relative(Path(embedding["_path"]), output_dir),
                "prompt_embedding_sha256": embedding["embedding_sha256"],
                "exact_band": band,
                "energy_target": math.log1p(band),
                "coordinates_json": json.dumps(
                    score.coordinates[0].tolist(), separators=(",", ":")
                ),
                "direction_id": item["direction_id"],
                "direction_sign": item["direction_sign"],
                "epsilon": LTE_RADIUS_RATIO * anchor_rms,
                "radius_ratio": LTE_RADIUS_RATIO,
                "step_number": 4,
                "timestep": item["timestep"],
                "fingerprint_json_sha256": contract.artifact_sha256,
                "ace_model_sha256": ace_model_sha256,
                "vae_sha256": vae_sha256,
                "dataset_plan_sha256": plan_sha256,
            }
        )
    examples.sort(
        key=lambda row: (row["split"], row["prompt_id"], row["source_kind"], row["sample_id"])
    )
    manifest_path = output_dir / "pitch3_lte_examples.csv"
    write_csv_atomic(manifest_path, examples)
    local_pairs = len({row["direction_id"] for row in examples if row["direction_id"]})
    nontrivial = 0
    for direction_id in {row["direction_id"] for row in examples if row["direction_id"]}:
        pair = [row for row in examples if row["direction_id"] == direction_id]
        if (
            len(pair) == 2
            and abs(float(pair[0]["energy_target"]) - float(pair[1]["energy_target"])) > 1e-6
        ):
            nontrivial += 1
    base_by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in examples:
        if row["source_kind"] == "base_step4_seed":
            base_by_prompt[str(row["prompt_id"])].append(float(row["energy_target"]))
    sortable_prompts = sum(
        any(
            abs(left - right) > 1e-6
            for offset, left in enumerate(values)
            for right in values[offset + 1 :]
        )
        for values in base_by_prompt.values()
    )
    sortable_fraction = sortable_prompts / len(base_by_prompt) if base_by_prompt else 0.0
    local_fraction = nontrivial / local_pairs if local_pairs else 0.0
    preflight_passed = bool(local_pairs and local_fraction >= 0.5 and sortable_fraction >= 0.5)
    summary = {
        "schema_version": LTE_SCHEMA_VERSION,
        "stage": "pitch3_lte_exact_local_dataset",
        "status": "complete" if preflight_passed else "failed",
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "base_samples": sum(row["source_kind"] == "base_step4_seed" for row in examples),
        "local_samples": sum(row["source_kind"] == "local_finite_difference" for row in examples),
        "local_pairs": local_pairs,
        "nontrivial_local_pairs": nontrivial,
        "nontrivial_local_pair_fraction": local_fraction,
        "sortable_prompts": sortable_prompts,
        "same_prompt_sortable_fraction": sortable_fraction,
        "local_preflight_passed": preflight_passed,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_plan_sha256": plan_sha256,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device": device_name,
        "retained_wav_files": len(list(output_dir.rglob("*.wav"))),
    }
    if summary["retained_wav_files"]:
        raise LTSNContractError("V3-LTE ephemeral WAV cleanup failed")
    write_json_atomic(output_dir / "pitch3_lte_dataset_summary.json", summary)
    return summary


def _merged_dataset_preflight(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    direction_ids = {str(row["direction_id"]) for row in examples if row["direction_id"]}
    nontrivial = 0
    for direction_id in direction_ids:
        pair = [row for row in examples if row["direction_id"] == direction_id]
        if (
            len(pair) == 2
            and {int(float(row["direction_sign"])) for row in pair} == {-1, 1}
            and abs(float(pair[0]["energy_target"]) - float(pair[1]["energy_target"])) > 1e-6
        ):
            nontrivial += 1
    base_by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in examples:
        if row["source_kind"] == "base_step4_seed":
            base_by_prompt[str(row["prompt_id"])].append(float(row["energy_target"]))
    if any(len(values) != 4 for values in base_by_prompt.values()):
        raise LTSNContractError("merged V3-LTE prompts do not contain four base seeds")
    sortable = sum(
        any(
            abs(left - right) > 1e-6
            for offset, left in enumerate(values)
            for right in values[offset + 1 :]
        )
        for values in base_by_prompt.values()
    )
    local_fraction = nontrivial / len(direction_ids) if direction_ids else 0.0
    sortable_fraction = sortable / len(base_by_prompt) if base_by_prompt else 0.0
    return {
        "local_pairs": len(direction_ids),
        "nontrivial_local_pairs": nontrivial,
        "nontrivial_local_pair_fraction": local_fraction,
        "sortable_prompts": sortable,
        "same_prompt_sortable_fraction": sortable_fraction,
        "local_preflight_passed": bool(
            direction_ids and local_fraction >= 0.5 and sortable_fraction >= 0.5
        ),
    }


def _merged_dataset_items_sha256(
    shard_rows: Sequence[tuple[Path, Sequence[Mapping[str, str]]]],
) -> str:
    """Hash logical dataset content without binding it to an execution layout."""

    ignored = {"latent_path", "prompt_embedding_path", "dataset_plan_sha256"}
    items = [
        {name: value for name, value in row.items() if name not in ignored}
        for _, rows in shard_rows
        for row in rows
    ]
    items.sort(key=lambda row: str(row["sample_id"]))
    return canonical_json_sha256(items)


def _publish_merged_dataset_plan(
    path: Path,
    plan: Mapping[str, Any],
    *,
    compatibility_fields: Sequence[str],
) -> str | None:
    """Publish a canonical merged plan and preserve a compatible legacy plan.

    Early V3-LTE multi-GPU plans included shard hashes and shard counts in the
    final dataset identity.  That made a scientifically identical resume fail
    when users switched from one GPU to several GPUs, or changed only the
    number of execution shards.  The final plan is now layout-independent;
    shard provenance remains in the dataset summary.
    """

    if not path.is_file():
        write_json_atomic(path, plan)
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == plan:
        return None
    if any(existing.get(name) != plan.get(name) for name in compatibility_fields):
        raise LTSNContractError(
            "merged V3-LTE scientific plan changed; use a new output directory"
        )
    if existing.get("publication_kind") == "canonical_merged_dataset":
        raise LTSNContractError(
            "merged V3-LTE dataset content changed; use a new output directory"
        )
    previous_sha256 = sha256_file(path)
    archived = path.with_name(f"{path.stem}_superseded_{previous_sha256[:12]}.json")
    if archived.is_file():
        if json.loads(archived.read_text(encoding="utf-8")) != existing:
            raise LTSNContractError("V3-LTE superseded plan archive hash collision")
    else:
        write_json_atomic(archived, existing)
    write_json_atomic(path, plan)
    return previous_sha256


def merge_pitch3_lte_dataset_shards(
    *,
    output_dir: Path,
    shard_dirs: Sequence[Path],
    devices: Sequence[str],
) -> dict[str, Any]:
    """Hash-verify disjoint GPU shards and publish one formal dataset manifest."""

    output_dir = output_dir.resolve()
    shard_dirs = tuple(path.resolve() for path in shard_dirs)
    shard_count = len(shard_dirs)
    if shard_count < 2 or len(devices) != shard_count:
        raise ValueError("multi-GPU V3-LTE merge requires one device per shard")
    plans: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    shard_rows: list[tuple[Path, list[dict[str, str]]]] = []
    common_fields = (
        "source_manifest_sha256",
        "prompt_embedding_manifest_sha256",
        "ace_config_sha256",
        "fingerprint_json_sha256",
        "ace_model_sha256",
        "vae_sha256",
        "include_splits",
        "local_splits",
        "radius_ratio",
        "anchor_rule",
        "direction_rule",
        "directions_per_prompt",
    )
    for expected_index, shard_dir in enumerate(shard_dirs):
        plan_path = shard_dir / "pitch3_lte_dataset_plan.json"
        summary_path = shard_dir / "pitch3_lte_dataset_summary.json"
        manifest_path = shard_dir / "pitch3_lte_examples.csv"
        if not all(path.is_file() for path in (plan_path, summary_path, manifest_path)):
            raise LTSNContractError(f"V3-LTE shard is incomplete: {shard_dir}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            plan.get("shard_index") != expected_index
            or plan.get("shard_count") != shard_count
            or summary.get("shard_index") != expected_index
            or summary.get("shard_count") != shard_count
            or summary.get("dataset_manifest_sha256") != sha256_file(manifest_path)
            or summary.get("local_preflight_passed") is not True
        ):
            raise LTSNContractError(f"V3-LTE shard contract failed: {shard_dir}")
        if plans and any(plan.get(name) != plans[0].get(name) for name in common_fields):
            raise LTSNContractError("V3-LTE shards were built from different frozen inputs")
        rows = _read_csv(manifest_path)
        if any(
            pitch3_lte_prompt_shard(row["prompt_id"], shard_count) != expected_index for row in rows
        ):
            raise LTSNContractError("V3-LTE shard contains a foreign prompt group")
        plans.append(plan)
        summaries.append(summary)
        shard_rows.append((shard_dir, rows))

    prompt_sets = [set(plan["prompt_ids"]) for plan in plans]
    if any(
        prompt_sets[left] & prompt_sets[right]
        for left in range(shard_count)
        for right in range(left + 1, shard_count)
    ):
        raise LTSNContractError("V3-LTE prompt groups overlap across GPU shards")
    shard_plan_hashes = [
        sha256_file(shard_dir / "pitch3_lte_dataset_plan.json") for shard_dir in shard_dirs
    ]
    merged_plan = {
        "schema_version": LTE_SCHEMA_VERSION,
        "stage": "pitch3_lte_exact_local_dataset",
        "model_family": LTE_MODEL_FAMILY,
        **{name: plans[0][name] for name in common_fields},
        "publication_kind": "canonical_merged_dataset",
        "planned_local_samples": sum(int(plan["planned_local_samples"]) for plan in plans),
        "items_sha256": _merged_dataset_items_sha256(shard_rows),
        "prompt_ids": sorted(set().union(*prompt_sets)),
        "wav_policy": "ephemeral_delete_after_each_exact_batch",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_plan_path = output_dir / "pitch3_lte_dataset_plan.json"
    replaced_plan_sha256 = _publish_merged_dataset_plan(
        merged_plan_path,
        merged_plan,
        compatibility_fields=(
            "schema_version",
            "stage",
            "model_family",
            *common_fields,
            "planned_local_samples",
            "prompt_ids",
            "wav_policy",
        ),
    )
    merged_plan_sha256 = sha256_file(merged_plan_path)

    examples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for shard_dir, rows in shard_rows:
        for raw in rows:
            if raw["sample_id"] in sample_ids:
                raise LTSNContractError("V3-LTE shard merge found a duplicate sample")
            sample_ids.add(raw["sample_id"])
            row: dict[str, Any] = dict(raw)
            for name in ("latent_path", "prompt_embedding_path"):
                artifact = (shard_dir / raw[name]).resolve()
                if not artifact.is_file():
                    raise LTSNContractError(f"V3-LTE merged artifact is missing: {artifact}")
                row[name] = _relative(artifact, output_dir)
            row["dataset_plan_sha256"] = merged_plan_sha256
            examples.append(row)
    examples.sort(
        key=lambda row: (row["split"], row["prompt_id"], row["source_kind"], row["sample_id"])
    )
    prompt_counts = Counter(
        row["split"] for row in examples if row["source_kind"] == "base_step4_seed"
    )
    if prompt_counts != {"train": 1280, "development": 256}:
        raise LTSNContractError(
            "formal merged V3-LTE data requires 1280 train and 256 development base rows"
        )
    local_counts = Counter(
        row["split"] for row in examples if row["source_kind"] == "local_finite_difference"
    )
    if local_counts != {"train": 1280, "development": 256}:
        raise LTSNContractError(
            "formal merged V3-LTE data requires 1280 train and 256 development local rows"
        )
    preflight = _merged_dataset_preflight(examples)
    manifest_path = output_dir / "pitch3_lte_examples.csv"
    write_csv_atomic(manifest_path, examples)
    retained_wav = len(list((output_dir / "shards").rglob("*.wav")))
    summary = {
        "schema_version": LTE_SCHEMA_VERSION,
        "stage": "pitch3_lte_exact_local_dataset",
        "status": "complete" if preflight["local_preflight_passed"] else "failed",
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "multi_gpu": True,
        "devices": list(devices),
        "shard_count": shard_count,
        "base_samples": sum(row["source_kind"] == "base_step4_seed" for row in examples),
        "local_samples": sum(row["source_kind"] == "local_finite_difference" for row in examples),
        **preflight,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_plan_sha256": merged_plan_sha256,
        "shard_summary_sha256": [
            sha256_file(shard_dir / "pitch3_lte_dataset_summary.json") for shard_dir in shard_dirs
        ],
        "shard_plan_sha256": shard_plan_hashes,
        "replaced_compatible_plan_sha256": replaced_plan_sha256,
        "retained_wav_files": retained_wav,
    }
    if retained_wav:
        raise LTSNContractError("multi-GPU V3-LTE shards retained ephemeral WAV files")
    write_json_atomic(output_dir / "pitch3_lte_dataset_summary.json", summary)
    return summary


def _build_pitch3_lte_shard(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return build_pitch3_lte_dataset(**dict(kwargs))


def build_pitch3_lte_dataset_multigpu(
    *,
    root: Path,
    source_manifest_path: Path,
    prompt_embedding_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    output_dir: Path,
    devices: Sequence[str],
    include_splits: Sequence[str] = ("train", "development"),
    local_splits: Sequence[str] = ("train", "development"),
    workers_per_device: int = 4,
    exact_batch_size: int = 64,
    materialize_mode: str = "auto",
    resume: bool = True,
) -> dict[str, Any]:
    """Run deterministic, process-isolated VAE/exact shards on several GPUs."""

    devices = tuple(str(device).strip() for device in devices if str(device).strip())
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("V3-LTE multi-GPU mode requires at least two unique devices")
    if workers_per_device < 1:
        raise ValueError("workers_per_device must be positive")
    if set(include_splits) != {"train", "development"} or set(local_splits) != {
        "train",
        "development",
    }:
        raise ValueError("formal V3-LTE multi-GPU mode requires train+development local data")
    root = root.resolve()
    output_dir = _rooted(output_dir, root)
    shard_root = output_dir / "shards"
    shard_dirs = tuple(shard_root / f"shard_{index:02d}" for index in range(len(devices)))
    payloads = [
        {
            "root": root,
            "source_manifest_path": source_manifest_path,
            "prompt_embedding_manifest_path": prompt_embedding_manifest_path,
            "ace_config_path": ace_config_path,
            "fingerprint_path": fingerprint_path,
            "output_dir": shard_dirs[index],
            "include_splits": tuple(include_splits),
            "local_splits": tuple(local_splits),
            "workers": workers_per_device,
            "exact_batch_size": exact_batch_size,
            "materialize_mode": materialize_mode,
            "device_name": device,
            "resume": resume,
            "shard_index": index,
            "shard_count": len(devices),
        }
        for index, device in enumerate(devices)
    ]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
        shard_summaries = list(executor.map(_build_pitch3_lte_shard, payloads))
    if any(summary.get("local_preflight_passed") is not True for summary in shard_summaries):
        raise LTSNContractError("at least one V3-LTE GPU shard failed its local preflight")
    return merge_pitch3_lte_dataset_shards(
        output_dir=output_dir,
        shard_dirs=shard_dirs,
        devices=devices,
    )
