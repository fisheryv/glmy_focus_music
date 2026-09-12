from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ace_adapter import AceStepAdapter, GenerationRequest
from .experiment import (
    CandidateRecord,
    ExperimentConfig,
    config_fingerprint,
    read_candidate_manifest,
    write_candidate_manifest,
)
from .ltsn_contract import sha256_file
from .rerank_experiment import _invalidate_scoring_artifacts, ensure_experiment


def validate_generation_devices(devices: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(device).strip() for device in devices)
    if not normalized or any(not device for device in normalized):
        raise ValueError("--devices requires at least one non-empty CUDA device")
    if len(set(normalized)) != len(normalized):
        raise ValueError("--devices must not contain duplicates")
    for device in normalized:
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError(
                f"multi-GPU generation requires explicit cuda:N devices, got {device!r}"
            )
    return normalized


def assign_prompt_shards(
    records: list[CandidateRecord], shard_count: int
) -> list[list[CandidateRecord]]:
    """Assign complete prompt pools to deterministic, balanced round-robin shards."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    prompt_order: list[str] = []
    pools: dict[str, list[CandidateRecord]] = {}
    for record in records:
        if record.prompt_id not in pools:
            prompt_order.append(record.prompt_id)
            pools[record.prompt_id] = []
        pools[record.prompt_id].append(record)
    shards: list[list[CandidateRecord]] = [[] for _ in range(shard_count)]
    for prompt_index, prompt_id in enumerate(prompt_order):
        shards[prompt_index % shard_count].extend(pools[prompt_id])
    return shards


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _planned_copy(record: CandidateRecord) -> CandidateRecord:
    return replace(
        record,
        status="planned",
        audio_relative_path="",
        audio_sha256="",
        latent_relative_path="",
        latent_sha256="",
        metadata_relative_path="",
        generated_at="",
        error="",
    )


def _identity(record: CandidateRecord) -> tuple[Any, ...]:
    return (
        record.experiment_id,
        record.prompt_id,
        record.caption,
        record.candidate_index,
        record.candidate_id,
        record.seed,
        record.duration_seconds,
        record.bpm,
        record.keyscale,
        record.timesignature,
    )


def _resolved_artifact(root: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid {label} path in candidate manifest")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its shard root") from exc
    return path


def _validate_generated_record(
    root: Path,
    record: CandidateRecord,
    expected: CandidateRecord,
    *,
    save_latents: bool,
) -> None:
    if _identity(record) != _identity(expected):
        raise ValueError(f"candidate identity mismatch: {expected.candidate_id}")
    if record.status not in {"generated", "scored"}:
        raise ValueError(f"candidate is not generated: {expected.candidate_id}")
    audio_path = _resolved_artifact(root, record.audio_relative_path, "audio")
    if not audio_path.is_file() or not record.audio_sha256:
        raise ValueError(f"candidate audio is missing: {expected.candidate_id}")
    if sha256_file(audio_path) != record.audio_sha256:
        raise ValueError(f"candidate audio hash mismatch: {expected.candidate_id}")
    metadata_path = _resolved_artifact(root, record.metadata_relative_path, "metadata")
    if not metadata_path.is_file():
        raise ValueError(f"candidate metadata is missing: {expected.candidate_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_identity = (
        metadata.get("candidate_id"),
        metadata.get("prompt_id"),
        metadata.get("caption"),
        int(metadata.get("seed", -1)),
        float(metadata.get("duration_seconds", -1.0)),
    )
    expected_metadata_identity = (
        expected.candidate_id,
        expected.prompt_id,
        expected.caption,
        expected.seed,
        expected.duration_seconds,
    )
    if metadata_identity != expected_metadata_identity:
        raise ValueError(f"candidate metadata identity mismatch: {expected.candidate_id}")
    if metadata.get("audio_sha256") != record.audio_sha256:
        raise ValueError(f"candidate metadata audio hash mismatch: {expected.candidate_id}")
    if save_latents:
        latent_path = _resolved_artifact(root, record.latent_relative_path, "latent")
        if not latent_path.is_file() or not record.latent_sha256:
            raise ValueError(f"candidate latent is missing: {expected.candidate_id}")
        if sha256_file(latent_path) != record.latent_sha256:
            raise ValueError(f"candidate latent hash mismatch: {expected.candidate_id}")
        if metadata.get("latent_sha256") != record.latent_sha256:
            raise ValueError(f"candidate metadata latent hash mismatch: {expected.candidate_id}")


def _record_is_valid(
    root: Path,
    record: CandidateRecord,
    expected: CandidateRecord,
    *,
    save_latents: bool,
) -> bool:
    try:
        _validate_generated_record(root, record, expected, save_latents=save_latents)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _load_shard_records(
    shard_root: Path, expected: list[CandidateRecord]
) -> tuple[Path, list[CandidateRecord]]:
    manifest_path = shard_root / "manifest.csv"
    if manifest_path.is_file():
        records = read_candidate_manifest(manifest_path)
        if [_identity(record) for record in records] != [_identity(record) for record in expected]:
            raise ValueError(f"existing shard manifest does not match frozen plan: {manifest_path}")
    else:
        records = [_planned_copy(record) for record in expected]
        write_candidate_manifest(manifest_path, records)
    return manifest_path, records


def _generate_shard(
    project_root: Path,
    config: ExperimentConfig,
    device: str,
    shard_index: int,
    expected: list[CandidateRecord],
    skip_candidate_ids: set[str],
    retry_failed: bool,
) -> dict[str, Any]:
    run_root = project_root / config.run_directory / config.run_id
    shard_root = run_root / "generation_shards" / f"shard_{shard_index:03d}"
    report_path = shard_root / "report.json"
    manifest_path, records = _load_shard_records(shard_root, expected)
    expected_by_id = {record.candidate_id: record for record in expected}
    pending: list[CandidateRecord] = []
    for record in records:
        if record.candidate_id in skip_candidate_ids:
            continue
        if _record_is_valid(
            shard_root,
            record,
            expected_by_id[record.candidate_id],
            save_latents=config.save_latents,
        ):
            continue
        if record.status == "failed" and not retry_failed:
            continue
        pending.append(record)

    backend = AceStepAdapter(project_root / config.ace.checkout, config.ace)
    if pending:
        os.environ["ACESTEP_DEVICE"] = device
        backend.initialize()
    audio_dir = shard_root / "data_raw" / "candidates"
    latent_dir = shard_root / "latents"
    metadata_dir = shard_root / "metadata" / "candidates"
    temporary_dir = shard_root / "tmp" / "generation"
    for record in pending:
        try:
            result = backend.generate(
                GenerationRequest(
                    prompt=record.caption,
                    seed=record.seed,
                    duration_seconds=record.duration_seconds,
                    output_dir=temporary_dir / record.candidate_id,
                    inference_steps=config.ace.inference_steps,
                    bpm=record.bpm,
                    keyscale=record.keyscale,
                    timesignature=record.timesignature,
                )
            )
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{record.candidate_id}.wav"
            if audio_path.is_file():
                audio_path.unlink()
            shutil.move(str(result.audio_path), audio_path)
            record.audio_relative_path = audio_path.relative_to(shard_root).as_posix()
            record.audio_sha256 = sha256_file(audio_path)
            if config.save_latents and result.final_latent is not None:
                import numpy as np

                value = result.final_latent
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "float"):
                    value = value.float()
                if hasattr(value, "numpy"):
                    value = value.numpy()
                latent_dir.mkdir(parents=True, exist_ok=True)
                latent_path = latent_dir / f"{record.candidate_id}.npz"
                np.savez_compressed(latent_path, latent=np.asarray(value, dtype=np.float32))
                record.latent_relative_path = latent_path.relative_to(shard_root).as_posix()
                record.latent_sha256 = sha256_file(latent_path)
            metadata_path = metadata_dir / f"{record.candidate_id}.json"
            _write_json(
                metadata_path,
                {
                    "candidate_id": record.candidate_id,
                    "prompt_id": record.prompt_id,
                    "caption": record.caption,
                    "seed": record.seed,
                    "duration_seconds": record.duration_seconds,
                    "audio_sha256": record.audio_sha256,
                    "latent_sha256": record.latent_sha256,
                    "backend": result.metadata,
                    "generation_shard": shard_index,
                    "generation_device": device,
                },
            )
            record.metadata_relative_path = metadata_path.relative_to(shard_root).as_posix()
            record.status = "generated"
            record.generated_at = datetime.now(UTC).isoformat()
            record.error = ""
        except Exception as exc:  # Keep completed candidates resumable after one bad request.
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        write_candidate_manifest(manifest_path, records)

    failed = [record for record in records if record.status == "failed"]
    unresolved = [
        record
        for record in records
        if record.candidate_id not in skip_candidate_ids
        and not _record_is_valid(
            shard_root,
            record,
            expected_by_id[record.candidate_id],
            save_latents=config.save_latents,
        )
    ]
    payload = {
        "schema_version": 1,
        "status": "complete" if not unresolved else "incomplete",
        "shard_index": shard_index,
        "device": device,
        "assigned_prompts": len({record.prompt_id for record in records}),
        "assigned_candidates": len(records),
        "skipped_from_published_manifest": len(skip_candidate_ids),
        "generated_or_resumed_in_shard": len(records) - len(unresolved) - len(skip_candidate_ids),
        "failed_candidates": len(failed),
        "unresolved_candidates": len(unresolved),
        "manifest_sha256": sha256_file(manifest_path),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    _write_json(report_path, payload)
    if unresolved:
        raise RuntimeError(
            f"generation shard {shard_index} is incomplete: {len(unresolved)} candidates"
        )
    return payload


def _worker_entry(
    project_root: Path,
    config: ExperimentConfig,
    device: str,
    shard_index: int,
    expected: list[CandidateRecord],
    skip_candidate_ids: set[str],
    retry_failed: bool,
) -> None:
    shard_root = (
        project_root
        / config.run_directory
        / config.run_id
        / "generation_shards"
        / f"shard_{shard_index:03d}"
    )
    try:
        _generate_shard(
            project_root,
            config,
            device,
            shard_index,
            expected,
            skip_candidate_ids,
            retry_failed,
        )
    except BaseException as exc:
        report_path = shard_root / "report.json"
        prior: dict[str, Any] = {}
        if report_path.is_file():
            try:
                prior = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
        _write_json(
            report_path,
            {
                **prior,
                "schema_version": 1,
                "status": "failed",
                "shard_index": shard_index,
                "device": device,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise


def _copy_atomic(source: Path, destination: Path, expected_sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        expected_sha256
        and destination.is_file()
        and sha256_file(destination) == expected_sha256
    ):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    if expected_sha256 and sha256_file(temporary) != expected_sha256:
        temporary.unlink()
        raise ValueError(f"copied artifact hash mismatch: {source}")
    os.replace(temporary, destination)


def merge_generation_shards(
    project_root: Path,
    config: ExperimentConfig,
    official_records: list[CandidateRecord],
    assignments: list[list[CandidateRecord]],
) -> list[CandidateRecord]:
    """Validate all records before atomically publishing the complete formal manifest."""

    run_root = project_root / config.run_directory / config.run_id
    official_by_id = {record.candidate_id: record for record in official_records}
    shard_sources: dict[str, tuple[Path, CandidateRecord]] = {}
    expected_by_id: dict[str, CandidateRecord] = {}
    for shard_index, expected_records in enumerate(assignments):
        shard_root = run_root / "generation_shards" / f"shard_{shard_index:03d}"
        manifest_path = shard_root / "manifest.csv"
        if not manifest_path.is_file():
            raise ValueError(f"generation shard manifest is missing: {manifest_path}")
        shard_records = read_candidate_manifest(manifest_path)
        if [_identity(record) for record in shard_records] != [
            _identity(record) for record in expected_records
        ]:
            raise ValueError(
                f"generation shard manifest differs from its assignment: {manifest_path}"
            )
        for expected, record in zip(expected_records, shard_records, strict=True):
            if expected.candidate_id in expected_by_id:
                raise ValueError(f"candidate assigned to multiple shards: {expected.candidate_id}")
            expected_by_id[expected.candidate_id] = expected
            shard_sources[expected.candidate_id] = (shard_root, record)

    if set(expected_by_id) != set(official_by_id):
        raise ValueError("generation shard assignments do not cover the frozen candidate plan")

    selected_sources: dict[str, tuple[Path, CandidateRecord]] = {}
    for candidate_id, expected in expected_by_id.items():
        official = official_by_id[candidate_id]
        if _record_is_valid(
            run_root, official, expected, save_latents=config.save_latents
        ):
            selected_sources[candidate_id] = (run_root, official)
            continue
        shard_root, shard_record = shard_sources[candidate_id]
        _validate_generated_record(
            shard_root,
            shard_record,
            expected,
            save_latents=config.save_latents,
        )
        selected_sources[candidate_id] = (shard_root, shard_record)

    merged: list[CandidateRecord] = []
    for official in official_records:
        expected = expected_by_id[official.candidate_id]
        source_root, source = selected_sources[official.candidate_id]
        if source_root == run_root:
            merged.append(source)
            continue
        audio_source = _resolved_artifact(source_root, source.audio_relative_path, "audio")
        audio_destination = run_root / "data_raw" / "candidates" / f"{source.candidate_id}.wav"
        _copy_atomic(audio_source, audio_destination, source.audio_sha256)
        metadata_source = _resolved_artifact(
            source_root, source.metadata_relative_path, "metadata"
        )
        metadata_destination = (
            run_root / "metadata" / "candidates" / f"{source.candidate_id}.json"
        )
        _copy_atomic(metadata_source, metadata_destination, sha256_file(metadata_source))
        latent_relative_path = ""
        if config.save_latents:
            latent_source = _resolved_artifact(source_root, source.latent_relative_path, "latent")
            latent_destination = run_root / "latents" / f"{source.candidate_id}.npz"
            _copy_atomic(latent_source, latent_destination, source.latent_sha256)
            latent_relative_path = latent_destination.relative_to(run_root).as_posix()
        merged_record = replace(
            expected,
            status="generated",
            audio_relative_path=audio_destination.relative_to(run_root).as_posix(),
            audio_sha256=source.audio_sha256,
            latent_relative_path=latent_relative_path,
            latent_sha256=source.latent_sha256 if config.save_latents else "",
            metadata_relative_path=metadata_destination.relative_to(run_root).as_posix(),
            generated_at=source.generated_at,
            error="",
        )
        _validate_generated_record(
            run_root, merged_record, expected, save_latents=config.save_latents
        )
        merged.append(merged_record)
    return merged


def generate_candidates_multi_device(
    project_root: Path,
    config: ExperimentConfig,
    devices: list[str] | tuple[str, ...],
    *,
    retry_failed: bool = False,
) -> list[CandidateRecord]:
    """Generate prompt-pool shards with spawn workers, then publish one checked manifest."""

    normalized_devices = validate_generation_devices(devices)
    run_root, official_records = ensure_experiment(project_root, config)
    expected = [_planned_copy(record) for record in official_records]
    assignments = assign_prompt_shards(expected, len(normalized_devices))
    official_by_id = {record.candidate_id: record for record in official_records}
    valid_official_ids = {
        record.candidate_id
        for record in expected
        if _record_is_valid(
            run_root,
            official_by_id[record.candidate_id],
            record,
            save_latents=config.save_latents,
        )
    }
    audit_path = run_root / "manifests" / "multigpu_generation_audit.json"
    started_at = datetime.now(UTC).isoformat()
    base_audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": config.run_id,
        "config_sha256": config_fingerprint(config),
        "devices": list(normalized_devices),
        "shard_count": len(assignments),
        "prompt_pools": len({record.prompt_id for record in expected}),
        "candidates": len(expected),
        "valid_published_candidates_at_start": len(valid_official_ids),
        "started_at": started_at,
        "assignments": [
            {
                "shard_index": index,
                "device": normalized_devices[index],
                "prompt_ids": list(dict.fromkeys(record.prompt_id for record in records)),
                "candidate_count": len(records),
            }
            for index, records in enumerate(assignments)
        ],
    }
    _write_json(audit_path, base_audit)
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for shard_index, (device, shard_records) in enumerate(
        zip(normalized_devices, assignments, strict=True)
    ):
        shard_skip = {
            record.candidate_id
            for record in shard_records
            if record.candidate_id in valid_official_ids
        }
        process = context.Process(
            target=_worker_entry,
            args=(
                project_root,
                config,
                device,
                shard_index,
                shard_records,
                shard_skip,
                retry_failed,
            ),
            name=f"ace-generation-shard-{shard_index:03d}",
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    exit_codes = [process.exitcode for process in processes]
    if any(code != 0 for code in exit_codes):
        _write_json(
            audit_path,
            {
                **base_audit,
                "status": "failed",
                "worker_exit_codes": exit_codes,
                "formal_manifest_published": False,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise RuntimeError(
            "one or more generation shards failed; formal candidate manifest was not published: "
            f"exit_codes={exit_codes}"
        )

    try:
        merged = merge_generation_shards(
            project_root, config, official_records, assignments
        )
    except Exception as exc:
        _write_json(
            audit_path,
            {
                **base_audit,
                "status": "merge_failed",
                "worker_exit_codes": exit_codes,
                "formal_manifest_published": False,
                "merge_error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise

    manifest_path = run_root / "manifests" / "candidates.csv"
    manifest_updated = merged != official_records
    if manifest_updated:
        write_candidate_manifest(manifest_path, merged)
        _invalidate_scoring_artifacts(run_root)
    _write_json(
        audit_path,
        {
            **base_audit,
            "status": "complete",
            "worker_exit_codes": exit_codes,
            "formal_manifest_published": True,
            "formal_manifest_updated": manifest_updated,
            "published_candidates": len(merged),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    return merged
