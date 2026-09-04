from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import GenerationBackend, GenerationRequest
from .exact_features import (
    compute_frozen_18d_descriptors,
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
from .ltsn_contract import sha256_file
from .ltsn_pipeline import RERANKING_GATE_NAME, load_reranking_gate
from .path_homology_exact_scorer import ExactPathHomologyScorer

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DESCRIPTOR_NUMERIC = {
    "candidate_index",
    "seed",
    "acoustic_loop_score",
    "chroma_loop_score",
    "focus_logit",
    "focus_probability",
    "focus_band_loss",
    "pitch_block_l2_norm",
    "phase_block_l2_norm",
    "raw_sample_rate",
    "raw_duration_seconds",
    "raw_peak",
    "raw_rms",
    "raw_clip_fraction",
    "raw_dc_offset",
}


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


def _exact_input_hashes(
    project_root: Path, config: ExperimentConfig
) -> tuple[dict[str, str], ExactPathHomologyScorer]:
    model_root = project_root / "features" / "models"
    hashes: dict[str, str] = {}
    for name in ("state_model.npz", "state_model.json", "pitch_v2_codebook.npz"):
        path = model_root / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen exact-scoring input is missing: {path}")
        hashes[path.relative_to(project_root).as_posix()] = _sha256(path)
    fingerprint_path = project_root / config.scoring.fingerprint_path
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    hashes[fingerprint_path.relative_to(project_root).as_posix()] = (
        scorer.contract.artifact_sha256
    )
    protocol_path = project_root / config.scoring.noninferiority_protocol_path
    _load_noninferiority_protocol(protocol_path)
    hashes[protocol_path.relative_to(project_root).as_posix()] = _sha256(protocol_path)
    return hashes, scorer


def _load_noninferiority_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen non-inferiority protocol is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != (
        "frozen_before_generation"
    ):
        raise ValueError("non-inferiority protocol is not frozen")
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != {"quality", "prompt", "diversity"}:
        raise ValueError("non-inferiority protocol criteria are incomplete")
    for name, criterion in criteria.items():
        if not isinstance(criterion, dict) or not str(criterion.get("metric", "")).strip():
            raise ValueError(f"non-inferiority protocol metric is missing: {name}")
        if criterion.get("direction") not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"non-inferiority protocol direction is invalid: {name}")
        margin = float(criterion.get("margin"))
        if margin < 0 or not math.isfinite(margin):
            raise ValueError(f"non-inferiority protocol margin is invalid: {name}")
    return payload


def ensure_experiment(
    project_root: Path, config: ExperimentConfig
) -> tuple[Path, list[CandidateRecord]]:
    run_root = experiment_root(project_root, config)
    run_root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config)
    exact_input_hashes, scorer = _exact_input_hashes(project_root, config)
    freeze_path = run_root / "experiment.json"
    if freeze_path.is_file():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen.get("config_sha256") != fingerprint:
            raise ValueError(
                f"run {config.run_id!r} already exists with a different configuration"
            )
        frozen_hashes = frozen.get("exact_input_sha256")
        if frozen_hashes != exact_input_hashes:
            raise ValueError(
                "exact scorer inputs changed after the experiment was frozen; start a new run_id"
            )
        if frozen.get("fingerprint_json_sha256") != scorer.contract.artifact_sha256:
            raise ValueError("frozen experiment uses a different 18-D scorer")
    else:
        _write_json(
            freeze_path,
            {
                "schema_version": 3,
                "config_sha256": fingerprint,
                "config": asdict(config),
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "feature_order": list(scorer.contract.feature_order),
                "exact_input_sha256": exact_input_hashes,
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
    for name in (
        "descriptors_18d.csv",
        "scores.csv",
        "pool_summary.csv",
        "summary.json",
        "noninferiority_template.json",
    ):
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
    return _normalize_descriptor_rows(rows)


def _normalize_descriptor_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    for row in normalized:
        for key in _DESCRIPTOR_NUMERIC:
            row[key] = float(row[key])
        row["candidate_index"] = int(row["candidate_index"])
        row["seed"] = int(row["seed"])
    return normalized


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
                    pool[int(rng.integers(0, len(pool)))]["focus_band_loss"]
                    for pool in pools
                ]
            )
        )
        exceed += random_mean <= observed_mean
    return (exceed + 1) / (permutations + 1)


def _bootstrap_median(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty prompt set")
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [np.median(rng.choice(values, size=values.size, replace=True)) for _ in range(resamples)]
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty prompt set")
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(resamples)]
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _quality_penalty(row: dict[str, Any], config: ExperimentConfig) -> tuple[float, bool]:
    clip = float(row["raw_clip_fraction"])
    rms = float(row["raw_rms"])
    dc = float(row["raw_dc_offset"])
    limits = config.scoring
    penalty = (
        max(0.0, clip - limits.maximum_clip_fraction)
        / max(limits.maximum_clip_fraction, 1e-12)
        + max(0.0, limits.minimum_rms - rms) / limits.minimum_rms
        + max(0.0, dc - limits.maximum_dc_offset) / limits.maximum_dc_offset
    )
    eligible = (
        clip <= limits.maximum_clip_fraction
        and rms >= limits.minimum_rms
        and dc <= limits.maximum_dc_offset
    )
    return float(penalty), eligible


def _validate_exact_descriptor_rows(
    descriptors: list[dict[str, Any]],
    scorer: ExactPathHomologyScorer,
    records: list[CandidateRecord],
    codebook_sha256: str,
) -> None:
    if not descriptors:
        raise ValueError("exact descriptor table is empty")
    candidate_ids: set[str] = set()
    record_by_id = {record.candidate_id: record for record in records}
    for row in descriptors:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("exact descriptor candidate IDs are empty or duplicated")
        candidate_ids.add(candidate_id)
        if row.get("fingerprint_json_sha256") != scorer.contract.artifact_sha256:
            raise ValueError("exact descriptor table uses a different frozen scorer")
        if row.get("label_source") != "decoded_candidate_exact_18d_v1":
            raise ValueError("reranking requires decoded-candidate exact 18-D descriptors")
        if row.get("pitch_v2_codebook_sha256") != codebook_sha256:
            raise ValueError("exact descriptor table uses a different Pitch codebook")
        record = record_by_id.get(candidate_id)
        if record is None or row.get("audio_sha256") != record.audio_sha256:
            raise ValueError("exact descriptor audio identity differs from candidate manifest")
        if json.loads(str(row["feature_order_json"])) != list(scorer.contract.feature_order):
            raise ValueError("exact descriptor feature order differs from the frozen scorer")
        coordinates = json.loads(str(row["coordinates_json"]))
        if len(coordinates) != 18 or not all(math.isfinite(float(value)) for value in coordinates):
            raise ValueError("exact descriptor row has malformed 18-D coordinates")
        exact = scorer.score(
            json.loads(str(row["pitch_descriptors_json"])),
            [float(row["acoustic_loop_score"])],
            [float(row["chroma_loop_score"])],
        )
        checks = {
            "coordinates": (np.asarray(coordinates), exact.coordinates[0]),
            "focus_logit": (float(row["focus_logit"]), exact.focus_logit[0]),
            "focus_probability": (
                float(row["focus_probability"]),
                exact.focus_probability[0],
            ),
            "focus_band_loss": (
                float(row["focus_band_loss"]),
                exact.focus_band_loss[0],
            ),
            "pitch_block_l2_norm": (
                float(row["pitch_block_l2_norm"]),
                exact.pitch_block_l2_norm[0],
            ),
            "phase_block_l2_norm": (
                float(row["phase_block_l2_norm"]),
                exact.phase_block_l2_norm[0],
            ),
        }
        if any(
            not np.allclose(actual, expected, rtol=0.0, atol=1e-12)
            for actual, expected in checks.values()
        ):
            raise ValueError("cached exact descriptor scores do not reproduce from raw inputs")
    if candidate_ids != set(record_by_id):
        raise ValueError("exact descriptor table does not cover the candidate manifest")


def rank_and_summarize(
    project_root: Path,
    config: ExperimentConfig,
    records: list[CandidateRecord],
    descriptors: list[dict[str, Any]],
) -> dict[str, Any]:
    run_root = experiment_root(project_root, config)
    descriptors = _normalize_descriptor_rows(descriptors)
    scorer = ExactPathHomologyScorer.from_json(
        project_root / config.scoring.fingerprint_path
    )
    codebook_sha256 = sha256_file(
        project_root / "features" / "models" / "pitch_v2_codebook.npz"
    )
    _validate_exact_descriptor_rows(descriptors, scorer, records, codebook_sha256)
    pools_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in descriptors:
        quality_penalty, quality_eligible = _quality_penalty(row, config)
        row["technical_quality_penalty"] = quality_penalty
        row["technical_quality_eligible"] = int(quality_eligible)
        pools_by_prompt.setdefault(str(row["prompt_id"]), []).append(row)
    ranked_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    complete_pools: list[list[dict[str, Any]]] = []
    for prompt_id, pool in sorted(pools_by_prompt.items()):
        if len(pool) != config.candidate_count:
            continue
        indices = [int(row["candidate_index"]) for row in pool]
        if sorted(indices) != list(range(config.candidate_count)):
            raise ValueError(f"candidate indices are incomplete for prompt {prompt_id}")
        ranked = sorted(
            pool,
            key=lambda row: (
                -int(row["technical_quality_eligible"]),
                float(row["focus_band_loss"]),
                float(row["technical_quality_penalty"]),
                str(row["candidate_id"]),
            ),
        )
        quality_only = min(
            pool,
            key=lambda row: (
                float(row["technical_quality_penalty"]), str(row["candidate_id"])
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            row["selected"] = int(rank == 1)
            row["baseline"] = int(row["candidate_index"] == 0)
            ranked_rows.append(row)
        baseline = next(row for row in pool if row["candidate_index"] == 0)
        winner = ranked[0]
        baseline_loss = float(baseline["focus_band_loss"])
        selected_loss = float(winner["focus_band_loss"])
        evaluable = baseline_loss > 0.0
        relative = (
            (baseline_loss - selected_loss) / baseline_loss if evaluable else float("nan")
        )
        pool_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_candidate_id": baseline["candidate_id"],
                "quality_only_candidate_id": quality_only["candidate_id"],
                "selected_candidate_id": winner["candidate_id"],
                "baseline_focus_band_loss": baseline_loss,
                "selected_focus_band_loss": selected_loss,
                "absolute_loss_change": selected_loss - baseline_loss,
                "relative_improvement": relative,
                "evaluable_positive_baseline_loss": int(evaluable),
                "baseline_target_band_hit": int(baseline_loss == 0.0),
                "selected_target_band_hit": int(selected_loss == 0.0),
                "baseline_quality_eligible": baseline["technical_quality_eligible"],
                "selected_quality_eligible": winner["technical_quality_eligible"],
                "selected_clip_fraction": winner["raw_clip_fraction"],
                "selected_rms": winner["raw_rms"],
                "selected_dc_offset": winner["raw_dc_offset"],
            }
        )
        complete_pools.append(pool)
    if not pool_rows:
        raise ValueError("no complete candidate pools are available for ranking")
    evaluable_rows = [
        row for row in pool_rows if int(row["evaluable_positive_baseline_loss"]) == 1
    ]
    if not evaluable_rows:
        raise ValueError("no prompt pool has positive baseline focus_band_loss")
    improvements = np.asarray(
        [float(row["relative_improvement"]) for row in evaluable_rows], dtype=float
    )
    selected_mean = float(
        np.mean([float(row["selected_focus_band_loss"]) for row in pool_rows])
    )
    p_value = _random_selection_pvalue(
        complete_pools, selected_mean, config.scoring.permutations, config.seed_start
    )
    ci_low, ci_high = _bootstrap_median(
        improvements, config.scoring.bootstrap_resamples, config.seed_start + 1
    )
    baseline_hit_rate = float(
        np.mean([int(row["baseline_target_band_hit"]) for row in pool_rows])
    )
    selected_hit_rate = float(
        np.mean([int(row["selected_target_band_hit"]) for row in pool_rows])
    )
    target_band_hit_rate_improved = selected_hit_rate > baseline_hit_rate
    selected_technical_quality_eligible = all(
        int(row["selected_quality_eligible"]) == 1 for row in pool_rows
    )
    median_improvement = float(np.median(improvements))
    sufficient = len(pool_rows) >= config.scoring.minimum_prompt_pools
    formal_design = bool(
        len(pool_rows) == 32
        and len(ranked_rows) == 256
        and config.candidate_count == 8
        and np.isclose(config.duration_seconds, 180.0)
    )
    topology_passed = bool(
        formal_design
        and sufficient
        and median_improvement >= config.scoring.success_min_median_improvement
        and ci_low > 0.0
        and target_band_hit_rate_improved
        and selected_technical_quality_eligible
    )
    topology_blockers: list[str] = []
    if not formal_design:
        topology_blockers.append("formal_design_failed")
    if not sufficient:
        topology_blockers.append(
            f"complete_prompt_pools={len(pool_rows)}<{config.scoring.minimum_prompt_pools}"
        )
    if median_improvement < config.scoring.success_min_median_improvement:
        topology_blockers.append("median_loss_improvement_below_threshold")
    if ci_low <= 0.0:
        topology_blockers.append("bootstrap_ci95_low_not_positive")
    if not target_band_hit_rate_improved:
        topology_blockers.append("target_band_hit_rate_not_improved")
    if not selected_technical_quality_eligible:
        topology_blockers.append("selected_candidate_failed_technical_quality")
    _write_rows(run_root / "scores.csv", ranked_rows)
    _write_rows(run_root / "pool_summary.csv", pool_rows)
    selected_ids = {row["selected_candidate_id"] for row in pool_rows}
    for record in records:
        if record.candidate_id in {row["candidate_id"] for row in ranked_rows}:
            record.status = "scored"
    write_candidate_manifest(run_root / "manifests" / "candidates.csv", records)
    descriptor_path = run_root / "descriptors_18d.csv"
    score_path = run_root / "scores.csv"
    pool_path = run_root / "pool_summary.csv"
    candidate_manifest = run_root / "manifests" / "candidates.csv"
    summary = {
        "schema_version": 2,
        "gate": RERANKING_GATE_NAME,
        "experiment_id": config.run_id,
        "verdict": (
            "topology_pass_noninferiority_pending" if topology_passed else "fail"
        ),
        "topology_passed": topology_passed,
        "formal_design": formal_design,
        "prompt_pools": len(pool_rows),
        "complete_prompt_pools_sufficient": sufficient,
        "evaluable_positive_loss_prompt_pools": len(evaluable_rows),
        "candidates_scored": len(ranked_rows),
        "selected_candidates": sorted(selected_ids),
        "generation_duration_seconds": config.duration_seconds,
        "candidate_count_per_prompt": config.candidate_count,
        "fingerprint_json_sha256": scorer.contract.artifact_sha256,
        "feature_order": list(scorer.contract.feature_order),
        "median_loss_improvement_fraction": median_improvement,
        "median_improvement_bootstrap_95_ci": [ci_low, ci_high],
        "mean_relative_improvement": float(np.mean(improvements)),
        "baseline_target_band_hit_rate": baseline_hit_rate,
        "selected_target_band_hit_rate": selected_hit_rate,
        "target_band_hit_rate_improved": target_band_hit_rate_improved,
        "all_selected_technical_quality_eligible": selected_technical_quality_eligible,
        "random_selection_p_value": p_value,
        "success_threshold": config.scoring.success_min_median_improvement,
        "minimum_prompt_pools": config.scoring.minimum_prompt_pools,
        "experiment_sha256": sha256_file(run_root / "experiment.json"),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "descriptor_table_sha256": sha256_file(descriptor_path),
        "selection_table_sha256": sha256_file(score_path),
        "pool_summary_sha256": sha256_file(pool_path),
        "gate_issued": False,
        "topology_blockers": topology_blockers,
        "gate_blocker": (
            "quality, prompt, and diversity non-inferiority evidence required"
            if topology_passed
            else "; ".join(topology_blockers)
        ),
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def score_candidates(
    project_root: Path, config: ExperimentConfig, records: list[CandidateRecord]
) -> dict[str, Any]:
    run_root = experiment_root(project_root, config)
    scorer = ExactPathHomologyScorer.from_json(
        project_root / config.scoring.fingerprint_path
    )
    descriptor_path = run_root / "descriptors_18d.csv"
    if descriptor_path.is_file():
        descriptors = _read_descriptor_csv(descriptor_path)
    else:
        processed = preprocess_candidates(
            project_root, run_root, records, workers=config.workers
        )
        feature_rows = extract_candidate_features(
            project_root, run_root, processed, workers=config.workers
        )
        descriptors = compute_frozen_18d_descriptors(
            project_root, run_root, records, feature_rows, scorer
        )
        write_descriptor_csv(descriptor_path, descriptors)
    return rank_and_summarize(project_root, config, records, descriptors)


def initialize_noninferiority_report(
    project_root: Path, config: ExperimentConfig, output_path: Path
) -> dict[str, Any]:
    """Create a run-bound, deliberately failing template for external evaluation."""

    run_root = experiment_root(project_root, config)
    summary_path = run_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("run exact score before initializing non-inferiority evidence")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "experiment_id": config.run_id,
        "fingerprint_json_sha256": summary["fingerprint_json_sha256"],
        "candidate_manifest_sha256": summary["candidate_manifest_sha256"],
        "selection_table_sha256": summary["selection_table_sha256"],
        "protocol_path": "REPLACE_WITH_FROZEN_PROTOCOL_PATH",
        "protocol_sha256": "REPLACE_WITH_FROZEN_PROTOCOL_SHA256",
        "criteria": {
            name: {
                "passed": False,
                "metric": "REPLACE_WITH_FROZEN_METRIC",
                "direction": "higher_is_better",
                "margin": None,
                "estimate": None,
                "ci95": [None, None],
                "evidence_path": "REPLACE_WITH_EVIDENCE_PATH",
                "evidence_sha256": "REPLACE_WITH_EVIDENCE_SHA256",
            }
            for name in ("quality", "prompt", "diversity")
        },
    }
    _write_json(output_path, payload)
    return payload


def evaluate_noninferiority_table(
    project_root: Path,
    config: ExperimentConfig,
    table_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate frozen paired quality/prompt/diversity metrics by prompt bootstrap."""

    run_root = experiment_root(project_root, config)
    summary_path = run_root / "summary.json"
    pool_path = run_root / "pool_summary.csv"
    if not summary_path.is_file() or not pool_path.is_file():
        raise FileNotFoundError("run exact reranking score before non-inferiority evaluation")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with pool_path.open("r", encoding="utf-8-sig", newline="") as handle:
        pool_rows = list(csv.DictReader(handle))
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    expected = {row["prompt_id"]: row for row in pool_rows}
    observed = {row.get("prompt_id", ""): row for row in evidence_rows}
    if len(observed) != len(evidence_rows) or set(observed) != set(expected):
        raise ValueError("non-inferiority table must contain exactly one row per reranking prompt")
    for prompt_id, row in observed.items():
        pool = expected[prompt_id]
        if row.get("baseline_candidate_id") != pool["baseline_candidate_id"] or row.get(
            "selected_candidate_id"
        ) != pool["selected_candidate_id"]:
            raise ValueError(f"non-inferiority candidate binding mismatch: {prompt_id}")

    protocol_path = project_root / config.scoring.noninferiority_protocol_path
    protocol = _load_noninferiority_protocol(protocol_path)
    criteria: dict[str, Any] = {}
    for index, (name, specification) in enumerate(protocol["criteria"].items()):
        baseline_column = f"{name}_baseline"
        selected_column = f"{name}_selected"
        if any(
            baseline_column not in row or selected_column not in row
            for row in evidence_rows
        ):
            raise ValueError(f"non-inferiority table is missing {name} columns")
        baseline = np.asarray(
            [float(observed[prompt_id][baseline_column]) for prompt_id in sorted(observed)],
            dtype=float,
        )
        selected = np.asarray(
            [float(observed[prompt_id][selected_column]) for prompt_id in sorted(observed)],
            dtype=float,
        )
        if not np.isfinite(baseline).all() or not np.isfinite(selected).all():
            raise ValueError(f"non-inferiority table contains non-finite {name} values")
        direction = specification["direction"]
        difference = selected - baseline
        if direction == "lower_is_better":
            difference = -difference
        estimate = float(np.mean(difference))
        low, high = _bootstrap_mean(
            difference,
            config.scoring.bootstrap_resamples,
            config.seed_start + 100 + index,
        )
        margin = float(specification["margin"])
        criteria[name] = {
            "passed": low >= -margin,
            "metric": specification["metric"],
            "direction": direction,
            "margin": margin,
            "estimate": estimate,
            "ci95": [low, high],
            "evidence_path": os.path.relpath(
                table_path.resolve(), output_path.parent.resolve()
            ).replace("\\", "/"),
            "evidence_sha256": sha256_file(table_path),
        }
    payload = {
        "schema_version": 1,
        "experiment_id": config.run_id,
        "fingerprint_json_sha256": summary["fingerprint_json_sha256"],
        "candidate_manifest_sha256": summary["candidate_manifest_sha256"],
        "selection_table_sha256": summary["selection_table_sha256"],
        "protocol_path": os.path.relpath(
            protocol_path.resolve(), output_path.parent.resolve()
        ).replace("\\", "/"),
        "protocol_sha256": sha256_file(protocol_path),
        "criteria": criteria,
    }
    _write_json(output_path, payload)
    return payload


def _validated_noninferiority(
    path: Path, summary: dict[str, Any], config: ExperimentConfig
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bindings = {
        "schema_version": 1,
        "experiment_id": config.run_id,
        "fingerprint_json_sha256": summary["fingerprint_json_sha256"],
        "candidate_manifest_sha256": summary["candidate_manifest_sha256"],
        "selection_table_sha256": summary["selection_table_sha256"],
    }
    for name, expected in bindings.items():
        if payload.get(name) != expected:
            raise ValueError(f"non-inferiority report binding mismatch: {name}")
    protocol_sha256 = str(payload.get("protocol_sha256", ""))
    protocol_path = Path(str(payload.get("protocol_path", "")))
    if not _SHA256.fullmatch(protocol_sha256):
        raise ValueError("non-inferiority protocol_sha256 is missing or malformed")
    if not protocol_path.is_absolute():
        protocol_path = path.parent / protocol_path
    if not protocol_path.is_file() or sha256_file(protocol_path) != protocol_sha256:
        raise ValueError("frozen non-inferiority protocol is missing or hash-mismatched")
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != {"quality", "prompt", "diversity"}:
        raise ValueError("non-inferiority report must contain quality, prompt, and diversity")
    for name, criterion in criteria.items():
        if not isinstance(criterion, dict) or criterion.get("passed") is not True:
            raise ValueError(f"{name} non-inferiority has not passed")
        if not str(criterion.get("metric", "")).strip():
            raise ValueError(f"{name} non-inferiority metric is missing")
        evidence_sha256 = str(criterion.get("evidence_sha256", ""))
        if not _SHA256.fullmatch(evidence_sha256):
            raise ValueError(f"{name} evidence_sha256 is missing or malformed")
        evidence_path = Path(str(criterion.get("evidence_path", "")))
        if not evidence_path.is_absolute():
            evidence_path = path.parent / evidence_path
        if not evidence_path.is_file() or sha256_file(evidence_path) != evidence_sha256:
            raise ValueError(f"{name} evidence file is missing or hash-mismatched")
        direction = criterion.get("direction")
        if direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"{name} direction is invalid")
        margin = float(criterion.get("margin"))
        estimate = float(criterion.get("estimate"))
        ci95 = criterion.get("ci95")
        if (
            margin < 0
            or not math.isfinite(margin)
            or not math.isfinite(estimate)
            or not isinstance(ci95, list)
            or len(ci95) != 2
        ):
            raise ValueError(f"{name} non-inferiority statistics are malformed")
        low, high = (float(value) for value in ci95)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"{name} confidence interval is malformed")
        calculated_pass = low >= -margin
        if not calculated_pass:
            raise ValueError(f"{name} confidence interval crosses the frozen margin")
    return payload


def issue_reranking_gate(
    project_root: Path,
    config: ExperimentConfig,
    noninferiority_report: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Issue a passed gate only from exact topology and bound non-inferiority evidence."""

    run_root = experiment_root(project_root, config)
    _, records = ensure_experiment(project_root, config)
    descriptor_path = run_root / "descriptors_18d.csv"
    if not descriptor_path.is_file():
        raise FileNotFoundError("exact 18-D descriptor table is missing")
    summary = rank_and_summarize(
        project_root, config, records, _read_descriptor_csv(descriptor_path)
    )
    summary_path = run_root / "summary.json"
    if summary.get("topology_passed") is not True:
        raise ValueError("exact 18-D topology reranking gate has not passed")
    evidence = _validated_noninferiority(noninferiority_report, summary, config)
    scorer = ExactPathHomologyScorer.from_json(
        project_root / config.scoring.fingerprint_path
    )
    if summary["fingerprint_json_sha256"] != scorer.contract.artifact_sha256:
        raise ValueError("reranking summary uses a different frozen 18-D scorer")
    payload = {
        "schema_version": 1,
        "gate": RERANKING_GATE_NAME,
        "status": "passed",
        "experiment_id": config.run_id,
        "fingerprint_json_sha256": summary["fingerprint_json_sha256"],
        "median_loss_improvement_fraction": summary[
            "median_loss_improvement_fraction"
        ],
        "bootstrap_ci95_low": summary["median_improvement_bootstrap_95_ci"][0],
        "target_band_hit_rate_improved": summary["target_band_hit_rate_improved"],
        "quality_noninferior": True,
        "prompt_noninferior": True,
        "diversity_preserved": True,
        "reranking_summary_sha256": sha256_file(summary_path),
        "candidate_manifest_sha256": summary["candidate_manifest_sha256"],
        "descriptor_table_sha256": summary["descriptor_table_sha256"],
        "selection_table_sha256": summary["selection_table_sha256"],
        "noninferiority_report_sha256": sha256_file(noninferiority_report),
        "protocol_sha256": evidence["protocol_sha256"],
    }
    _write_json(output_path, payload)
    load_reranking_gate(output_path, scorer.contract)
    return payload
