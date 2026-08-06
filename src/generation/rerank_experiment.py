from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import GenerationBackend, GenerationRequest
from .exact_features import (
    compute_exact_descriptors,
    extract_candidate_features,
    preprocess_candidates,
    write_descriptor_csv,
)
from .experiment import (
    CandidateRecord,
    ExperimentConfig,
    build_candidate_plan,
    config_fingerprint,
    read_candidate_manifest,
    read_prompts,
    write_candidate_manifest,
)
from .target_profile import (
    CORE_FEATURES,
    TargetProfile,
    fit_target_profile,
    read_target_profile,
    write_target_profile,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def experiment_root(project_root: Path, config: ExperimentConfig) -> Path:
    return project_root / config.run_directory / config.run_id


def _state_model_hashes(project_root: Path) -> dict[str, str]:
    model_root = project_root / "features" / "models"
    hashes: dict[str, str] = {}
    for name in ("state_model.npz", "state_model.json"):
        path = model_root / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen state model is missing: {path}")
        hashes[name] = _sha256(path)
    return hashes


def ensure_experiment(
    project_root: Path, config: ExperimentConfig
) -> tuple[Path, list[CandidateRecord]]:
    run_root = experiment_root(project_root, config)
    run_root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config)
    state_model_hashes = _state_model_hashes(project_root)
    freeze_path = run_root / "experiment.json"
    if freeze_path.is_file():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("config_sha256") != fingerprint:
            raise ValueError(
                f"run {config.run_id!r} already exists with a different configuration"
            )
        frozen_hashes = frozen.get("state_model_sha256")
        if frozen_hashes is not None and frozen_hashes != state_model_hashes:
            raise ValueError(
                "state model changed after the experiment was frozen; start a new run_id"
            )
        if frozen_hashes is None:
            frozen["schema_version"] = 2
            frozen["state_model_sha256"] = state_model_hashes
            _write_json(freeze_path, frozen)
    else:
        _write_json(
            freeze_path,
            {
                "schema_version": 2,
                "config_sha256": fingerprint,
                "config": asdict(config),
                "state_model_sha256": state_model_hashes,
                "ace_step_commit": _git_commit(project_root / config.ace.checkout),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    expected = build_candidate_plan(config, read_prompts(project_root, config.prompt_manifest))
    manifest_path = run_root / "manifests" / "candidates.csv"
    if manifest_path.is_file():
        records = read_candidate_manifest(manifest_path)
        expected_keys = [(item.candidate_id, item.seed, item.caption) for item in expected]
        observed_keys = [(item.candidate_id, item.seed, item.caption) for item in records]
        if observed_keys != expected_keys:
            raise ValueError("existing candidate manifest does not match the frozen plan")
    else:
        records = expected
        write_candidate_manifest(manifest_path, records)
    target_path = run_root / "target_profile.json"
    fresh_profile = fit_target_profile(
        project_root,
        group=config.scoring.target_group,
        split=config.scoring.target_split,
        scale_seconds=config.scoring.target_scale_seconds,
        covariance_shrinkage=config.scoring.covariance_shrinkage,
    )
    if target_path.is_file():
        frozen_profile = read_target_profile(target_path)
        if frozen_profile.source_sha256 != fresh_profile.source_sha256:
            raise ValueError("target source files changed after the experiment was frozen")
    else:
        write_target_profile(target_path, fresh_profile)
    return run_root, records


def _latent_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _invalidate_scoring_artifacts(run_root: Path) -> None:
    for name in ("descriptors.csv", "scores.csv", "pool_summary.csv", "summary.json"):
        path = run_root / name
        if path.is_file():
            path.unlink()


def generate_candidates(
    project_root: Path,
    config: ExperimentConfig,
    backend: GenerationBackend,
    *,
    retry_failed: bool = False,
) -> list[CandidateRecord]:
    run_root, records = ensure_experiment(project_root, config)
    manifest_path = run_root / "manifests" / "candidates.csv"
    audio_dir = run_root / "data_raw" / "candidates"
    latent_dir = run_root / "latents"
    metadata_dir = run_root / "metadata" / "candidates"
    temporary_dir = run_root / "tmp" / "generation"
    for record in records:
        existing_audio = (
            run_root / record.audio_relative_path if record.audio_relative_path else None
        )
        existing_latent = (
            run_root / record.latent_relative_path if record.latent_relative_path else None
        )
        existing_metadata = (
            run_root / record.metadata_relative_path if record.metadata_relative_path else None
        )
        audio_valid = bool(
            existing_audio
            and existing_audio.is_file()
            and record.audio_sha256
            and _sha256(existing_audio) == record.audio_sha256
        )
        latent_valid = not config.save_latents or bool(
            existing_latent
            and existing_latent.is_file()
            and record.latent_sha256
            and _sha256(existing_latent) == record.latent_sha256
        )
        metadata_valid = bool(existing_metadata and existing_metadata.is_file())
        if (
            record.status in {"generated", "scored"}
            and audio_valid
            and latent_valid
            and metadata_valid
        ):
            continue
        if record.status == "failed" and not retry_failed:
            continue
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
            if audio_path.exists():
                audio_path.unlink()
            shutil.move(str(result.audio_path), audio_path)
            record.audio_relative_path = audio_path.relative_to(run_root).as_posix()
            record.audio_sha256 = _sha256(audio_path)
            if config.save_latents and result.final_latent is not None:
                latent_dir.mkdir(parents=True, exist_ok=True)
                latent_path = latent_dir / f"{record.candidate_id}.npz"
                np.savez_compressed(latent_path, latent=_latent_array(result.final_latent))
                record.latent_relative_path = latent_path.relative_to(run_root).as_posix()
                record.latent_sha256 = _sha256(latent_path)
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
                },
            )
            record.metadata_relative_path = metadata_path.relative_to(run_root).as_posix()
            record.status = "generated"
            record.generated_at = datetime.now(UTC).isoformat()
            record.error = ""
            _invalidate_scoring_artifacts(run_root)
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        write_candidate_manifest(manifest_path, records)
    return records


def _read_descriptor_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "candidate_index",
        "seed",
        *CORE_FEATURES,
        "raw_sample_rate",
        "raw_duration_seconds",
        "raw_peak",
        "raw_rms",
        "raw_clip_fraction",
        "raw_dc_offset",
    }
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row["candidate_index"] = int(row["candidate_index"])
        row["seed"] = int(row["seed"])
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _random_selection_pvalue(
    pools: list[list[dict[str, Any]]], observed_mean: float, permutations: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        random_mean = float(
            np.mean(
                [
                    pool[int(rng.integers(0, len(pool)))]["topology_distance"]
                    for pool in pools
                ]
            )
        )
        exceed += random_mean <= observed_mean
    return (exceed + 1) / (permutations + 1)


def _bootstrap_median(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [np.median(rng.choice(values, size=values.size, replace=True)) for _ in range(resamples)]
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def rank_and_summarize(
    project_root: Path,
    config: ExperimentConfig,
    records: list[CandidateRecord],
    descriptors: list[dict[str, Any]],
) -> dict[str, Any]:
    run_root = experiment_root(project_root, config)
    profile: TargetProfile = read_target_profile(run_root / "target_profile.json")
    pools_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in descriptors:
        descriptor = [float(row[name]) for name in CORE_FEATURES]
        topology_distance = profile.distance(descriptor)
        quality_penalty = float(row["raw_clip_fraction"]) + max(0.0, 1e-4 - float(row["raw_rms"]))
        row["topology_distance"] = topology_distance
        row["technical_quality_penalty"] = quality_penalty
        row["total_score"] = (
            topology_distance + config.scoring.technical_quality_weight * quality_penalty
        )
        pools_by_prompt.setdefault(str(row["prompt_id"]), []).append(row)
    ranked_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    complete_pools: list[list[dict[str, Any]]] = []
    for prompt_id, pool in sorted(pools_by_prompt.items()):
        if len(pool) != config.candidate_count:
            continue
        ranked = sorted(pool, key=lambda row: (row["total_score"], row["candidate_id"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            row["selected"] = int(rank == 1)
            row["baseline"] = int(row["candidate_index"] == 0)
            ranked_rows.append(row)
        baseline = next(row for row in pool if row["candidate_index"] == 0)
        winner = ranked[0]
        relative = (baseline["topology_distance"] - winner["topology_distance"]) / max(
            baseline["topology_distance"], 1e-12
        )
        pool_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_candidate_id": baseline["candidate_id"],
                "selected_candidate_id": winner["candidate_id"],
                "baseline_distance": baseline["topology_distance"],
                "selected_distance": winner["topology_distance"],
                "relative_improvement": relative,
                "selected_clip_fraction": winner["raw_clip_fraction"],
                "selected_rms": winner["raw_rms"],
            }
        )
        complete_pools.append(pool)
    if not pool_rows:
        raise ValueError("no complete candidate pools are available for ranking")
    improvements = np.asarray([row["relative_improvement"] for row in pool_rows], dtype=float)
    selected_mean = float(np.mean([row["selected_distance"] for row in pool_rows]))
    p_value = _random_selection_pvalue(
        complete_pools, selected_mean, config.scoring.permutations, config.seed_start
    )
    ci_low, ci_high = _bootstrap_median(
        improvements, config.scoring.bootstrap_resamples, config.seed_start + 1
    )
    scale_matches = np.isclose(config.duration_seconds, profile.scale_seconds)
    sufficient = len(pool_rows) >= config.scoring.minimum_prompt_pools
    passes = (
        scale_matches
        and sufficient
        and float(np.median(improvements)) >= config.scoring.success_min_median_improvement
        and p_value < 0.05
    )
    verdict = "pass" if passes else "fail"
    if not scale_matches:
        verdict = "smoke_only_scale_mismatch"
    elif not sufficient:
        verdict = "insufficient_prompt_pools"
    _write_rows(run_root / "scores.csv", ranked_rows)
    _write_rows(run_root / "pool_summary.csv", pool_rows)
    selected_ids = {row["selected_candidate_id"] for row in pool_rows}
    for record in records:
        if record.candidate_id in {row["candidate_id"] for row in ranked_rows}:
            record.status = "scored"
    write_candidate_manifest(run_root / "manifests" / "candidates.csv", records)
    summary = {
        "schema_version": 1,
        "experiment_id": config.run_id,
        "verdict": verdict,
        "prompt_pools": len(pool_rows),
        "candidates_scored": len(ranked_rows),
        "selected_candidates": sorted(selected_ids),
        "target_scale_seconds": profile.scale_seconds,
        "generation_duration_seconds": config.duration_seconds,
        "scale_matches_target": bool(scale_matches),
        "median_relative_improvement": float(np.median(improvements)),
        "median_improvement_bootstrap_95_ci": [ci_low, ci_high],
        "mean_relative_improvement": float(np.mean(improvements)),
        "random_selection_p_value": p_value,
        "success_threshold": config.scoring.success_min_median_improvement,
        "minimum_prompt_pools": config.scoring.minimum_prompt_pools,
        "quality_scope": (
            "technical clipping/RMS penalty only; perceptual and prompt-alignment "
            "non-inferiority require a separate blinded evaluation"
        ),
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def score_candidates(
    project_root: Path, config: ExperimentConfig, records: list[CandidateRecord]
) -> dict[str, Any]:
    run_root = experiment_root(project_root, config)
    descriptor_path = run_root / "descriptors.csv"
    if descriptor_path.is_file():
        descriptors = _read_descriptor_csv(descriptor_path)
    else:
        processed = preprocess_candidates(
            project_root, run_root, records, workers=config.workers
        )
        feature_rows = extract_candidate_features(
            project_root, run_root, processed, workers=config.workers
        )
        descriptors = compute_exact_descriptors(
            project_root, run_root, records, feature_rows
        )
        write_descriptor_csv(descriptor_path, descriptors)
    return rank_and_summarize(project_root, config, records, descriptors)
