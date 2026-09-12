"""Paired Base-versus-LoRA generation and frozen exact-score validation."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import AceStepAdapter, GenerationRequest
from .artifact_hash import sha256_directory
from .exact_features import (
    compute_frozen_18d_descriptors,
    extract_candidate_features,
    preprocess_candidates,
    write_descriptor_csv,
)
from .experiment import CandidateRecord, ExperimentConfig, read_prompts, write_candidate_manifest
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import canonical_json_sha256, write_json_atomic
from .path_homology_exact_scorer import ExactPathHomologyScorer
from .topology_lora import TOPOLOGY_LORA_TAG
from .topology_lora_training import load_lora_config

VALIDATION_SPLITS = ("development", "qualification")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _wilson_low(wins: int, total: int) -> float:
    if total < 1:
        raise ValueError("paired validation has no prompts")
    z = 1.959963984540054
    rate = wins / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center - radius) / denominator


def _freeze_validation(
    *,
    run_dir: Path,
    experiment_config: ExperimentConfig,
    prompt_manifest: Path,
    split: str,
    lora_path: Path,
    lora_sha256: str,
    lora_experiment: str,
    scale: float,
    seed_start: int,
) -> None:
    payload = {
        "schema_version": 1,
        "experiment": lora_experiment,
        "split": split,
        "prompt_manifest": str(prompt_manifest.resolve()),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
        "lora_path": str(lora_path.resolve()),
        "lora_bundle_sha256": lora_sha256,
        "lora_scale": scale,
        "seed_start": seed_start,
        "ace": asdict(experiment_config.ace),
        "scoring": asdict(experiment_config.scoring),
    }
    payload["freeze_sha256"] = canonical_json_sha256(payload)
    path = run_dir / "validation_experiment.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("freeze_sha256") != payload["freeze_sha256"]:
            raise LTSNContractError("validation run already exists with different inputs")
    else:
        write_json_atomic(path, payload)


def _planned_records(
    project_root: Path,
    experiment_config: ExperimentConfig,
    seed_start: int,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for prompt_index, prompt in enumerate(
        read_prompts(project_root, experiment_config.prompt_manifest)
    ):
        seed = seed_start + prompt_index
        for candidate_index, arm in enumerate(("base", "lora")):
            records.append(
                CandidateRecord(
                    experiment_id=f"topology_lora_{arm}",
                    prompt_id=prompt.prompt_id,
                    caption=prompt.caption,
                    candidate_index=candidate_index,
                    candidate_id=f"{prompt.prompt_id}__{arm}__s{seed}",
                    seed=seed,
                    duration_seconds=experiment_config.duration_seconds,
                    bpm=prompt.bpm,
                    keyscale=prompt.keyscale,
                    timesignature=prompt.timesignature,
                )
            )
    return records


def _generate_pairs(
    *,
    project_root: Path,
    run_dir: Path,
    config: ExperimentConfig,
    records: list[CandidateRecord],
    lora_path: Path,
    lora_sha256: str,
    scale: float,
    activation_tag: str,
) -> None:
    manifest = run_dir / "manifests" / "candidates.csv"
    adapter: AceStepAdapter | None = None
    for record in records:
        audio_path = run_dir / record.audio_relative_path if record.audio_relative_path else None
        if (
            record.status in {"generated", "scored"}
            and audio_path is not None
            and audio_path.is_file()
            and sha256_file(audio_path) == record.audio_sha256
        ):
            continue
        if adapter is None:
            adapter = AceStepAdapter(project_root / config.ace.checkout, config.ace)
            adapter.load_lora(lora_path, scale=scale, expected_sha256=lora_sha256)
        lora_arm = record.candidate_index == 1
        adapter.set_lora_enabled(lora_arm)
        prompt = f"{activation_tag}, {record.caption}" if lora_arm else record.caption
        result = adapter.generate(
            GenerationRequest(
                prompt=prompt,
                seed=record.seed,
                duration_seconds=record.duration_seconds,
                output_dir=run_dir / "tmp" / record.candidate_id,
                inference_steps=config.ace.inference_steps,
                bpm=record.bpm,
                keyscale=record.keyscale,
                timesignature=record.timesignature,
            )
        )
        destination = run_dir / "data_raw" / "candidates" / f"{record.candidate_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.move(str(result.audio_path), destination)
        record.audio_relative_path = destination.relative_to(run_dir).as_posix()
        record.audio_sha256 = sha256_file(destination)
        record.status = "generated"
        record.metadata_relative_path = f"metadata/candidates/{record.candidate_id}.json"
        write_json_atomic(
            run_dir / record.metadata_relative_path,
            {
                "candidate_id": record.candidate_id,
                "arm": "lora" if lora_arm else "base",
                "semantic_caption": record.caption,
                "generation_prompt": prompt,
                "seed": record.seed,
                "audio_sha256": record.audio_sha256,
                "backend": result.metadata,
            },
        )
        write_candidate_manifest(manifest, records)


def _score_pairs(
    project_root: Path,
    run_dir: Path,
    config: ExperimentConfig,
    records: list[CandidateRecord],
) -> list[dict[str, Any]]:
    descriptor_path = run_dir / "descriptors_18d.csv"
    if descriptor_path.is_file():
        rows: list[dict[str, Any]] = _read_rows(descriptor_path)
        by_id = {record.candidate_id: record for record in records}
        scorer = ExactPathHomologyScorer.from_json(
            project_root / config.scoring.fingerprint_path
        )
        if set(by_id) != {row["candidate_id"] for row in rows}:
            raise LTSNContractError("cached validation descriptors do not cover the plan")
        for row in rows:
            if row["audio_sha256"] != by_id[row["candidate_id"]].audio_sha256:
                raise LTSNContractError("validation audio changed after exact scoring")
            if row["fingerprint_json_sha256"] != scorer.contract.artifact_sha256:
                raise LTSNContractError("validation descriptors use a different exact scorer")
        return rows
    processed = preprocess_candidates(project_root, run_dir, records, workers=config.workers)
    features = extract_candidate_features(
        project_root, run_dir, processed, workers=config.workers
    )
    scorer = ExactPathHomologyScorer.from_json(project_root / config.scoring.fingerprint_path)
    rows = compute_frozen_18d_descriptors(project_root, run_dir, records, features, scorer)
    write_descriptor_csv(descriptor_path, rows)
    return rows


def summarize_paired_validation(
    *,
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    validation_config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    """Summarize paired exact loss and technical quality under frozen thresholds."""

    pools: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        pools.setdefault(str(row["prompt_id"]), {})[int(row["candidate_index"])] = row
    if not pools or any(set(pool) != {0, 1} for pool in pools.values()):
        raise LTSNContractError("paired validation requires one Base and one LoRA row per prompt")
    pair_rows = []
    for prompt_id, pool in sorted(pools.items()):
        base, lora = pool[0], pool[1]
        base_loss = float(base["focus_band_loss"])
        lora_loss = float(lora["focus_band_loss"])
        quality = (
            float(lora["raw_clip_fraction"]) <= config.scoring.maximum_clip_fraction
            and float(lora["raw_rms"]) >= config.scoring.minimum_rms
            and float(lora["raw_dc_offset"]) <= config.scoring.maximum_dc_offset
        )
        pair_rows.append(
            {
                "prompt_id": prompt_id,
                "seed": int(lora["seed"]),
                "base_focus_band_loss": base_loss,
                "lora_focus_band_loss": lora_loss,
                "lora_minus_base_loss": lora_loss - base_loss,
                "lora_topology_win": int(lora_loss < base_loss),
                "base_target_band_hit": int(base_loss == 0.0),
                "lora_target_band_hit": int(lora_loss == 0.0),
                "lora_technical_quality_eligible": int(quality),
            }
        )
    wins = sum(row["lora_topology_win"] for row in pair_rows)
    changes = np.asarray([row["lora_minus_base_loss"] for row in pair_rows], dtype=float)
    base_hits = np.mean([row["base_target_band_hit"] for row in pair_rows])
    lora_hits = np.mean([row["lora_target_band_hit"] for row in pair_rows])
    thresholds = validation_config["validation"]
    win_rate = wins / len(pair_rows)
    ci_low = _wilson_low(wins, len(pair_rows))
    topology_supported = bool(
        win_rate >= float(thresholds["paired_topology_win_rate_minimum"])
        and ci_low > float(thresholds["paired_topology_wilson_ci95_low_must_exceed"])
        and float(np.median(changes)) < 0.0
        and lora_hits > base_hits
        and all(row["lora_technical_quality_eligible"] for row in pair_rows)
    )
    return {
        "pair_rows": pair_rows,
        "summary": {
            "schema_version": 1,
            "experiment": validation_config["experiment"],
            "split": split,
            "prompt_pairs": len(pair_rows),
            "paired_topology_win_rate": win_rate,
            "paired_topology_wilson_ci95_low": ci_low,
            "median_lora_minus_base_exact_loss": float(np.median(changes)),
            "base_target_band_hit_rate": float(base_hits),
            "lora_target_band_hit_rate": float(lora_hits),
            "all_lora_technical_quality_eligible": all(
                row["lora_technical_quality_eligible"] for row in pair_rows
            ),
            "topology_supported": topology_supported,
            "development_scale_selection_eligible": split == "development" and topology_supported,
            "qualification_topology_confirmed": split == "qualification" and topology_supported,
            "production_authorization": False,
            "production_blocker": "blind_quality_prompt_diversity_noninferiority_required",
        },
    }


def run_paired_validation(
    *,
    project_root: Path,
    config: ExperimentConfig,
    lora_config_path: Path,
    prompt_manifest: Path,
    split: str,
    run_dir: Path,
    lora_path: Path,
    lora_sha256: str,
    scale: float,
    seed_start: int,
    activation_tag: str = TOPOLOGY_LORA_TAG,
) -> dict[str, Any]:
    """Generate resumable Base/LoRA pairs, exact-score them, and write an audit report."""

    if split not in VALIDATION_SPLITS:
        raise ValueError(f"validation split must be one of {VALIDATION_SPLITS}")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("LoRA validation scale must lie in [0, 1]")
    if sha256_directory(lora_path) != lora_sha256:
        raise LTSNContractError("LoRA artifact changed before paired validation")
    prompt_rows = _read_rows(prompt_manifest)
    if {row.get("split") for row in prompt_rows} != {split}:
        raise LTSNContractError("validation prompt manifest has the wrong split")
    run_dir.mkdir(parents=True, exist_ok=True)
    lora_config = load_lora_config(lora_config_path)
    _freeze_validation(
        run_dir=run_dir,
        experiment_config=config,
        prompt_manifest=prompt_manifest,
        split=split,
        lora_path=lora_path,
        lora_sha256=lora_sha256,
        lora_experiment=lora_config["experiment"],
        scale=scale,
        seed_start=seed_start,
    )
    expected_records = _planned_records(project_root, config, seed_start)
    records = expected_records
    manifest = run_dir / "manifests" / "candidates.csv"
    if manifest.is_file():
        from .experiment import read_candidate_manifest

        records = read_candidate_manifest(manifest)
        expected_identity = [
            (item.candidate_id, item.prompt_id, item.seed, item.caption)
            for item in expected_records
        ]
        observed_identity = [
            (item.candidate_id, item.prompt_id, item.seed, item.caption) for item in records
        ]
        if observed_identity != expected_identity:
            raise LTSNContractError("validation candidate manifest differs from the frozen plan")
    else:
        write_candidate_manifest(manifest, records)
    _generate_pairs(
        project_root=project_root,
        run_dir=run_dir,
        config=config,
        records=records,
        lora_path=lora_path,
        lora_sha256=lora_sha256,
        scale=scale,
        activation_tag=activation_tag,
    )
    rows = _score_pairs(project_root, run_dir, config, records)
    result = summarize_paired_validation(
        rows=rows,
        config=config,
        validation_config=lora_config,
        split=split,
    )
    from .ltsn_pipeline import write_csv_atomic

    write_csv_atomic(run_dir / "paired_results.csv", result["pair_rows"])
    summary = result["summary"]
    summary.update(
        {
            "lora_bundle_sha256": lora_sha256,
            "lora_scale": scale,
            "prompt_manifest_sha256": sha256_file(prompt_manifest),
            "candidate_manifest_sha256": sha256_file(manifest),
            "descriptor_table_sha256": sha256_file(run_dir / "descriptors_18d.csv"),
            "paired_results_sha256": sha256_file(run_dir / "paired_results.csv"),
        }
    )
    write_json_atomic(run_dir / "validation_report.json", summary)
    return summary


def select_development_scale(
    *,
    validation_root: Path,
    lora_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select one predeclared scale using development reports only."""

    config = load_lora_config(lora_config_path)
    expected_scales = [float(value) for value in config["validation"]["development_scales"]]
    reports = []
    for scale in expected_scales:
        path = validation_root / f"development_scale_{scale}" / "validation_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("split") != "development" or not math.isclose(
            float(report.get("lora_scale")), scale, rel_tol=0.0, abs_tol=1e-12
        ):
            raise LTSNContractError(f"development report has wrong split or scale: {path}")
        reports.append((scale, path, report))
    if len({report["lora_bundle_sha256"] for _, _, report in reports}) != 1:
        raise LTSNContractError("development scale reports use different LoRA artifacts")
    if len({report["prompt_manifest_sha256"] for _, _, report in reports}) != 1:
        raise LTSNContractError("development scale reports use different prompt manifests")
    eligible = [item for item in reports if item[2]["development_scale_selection_eligible"]]
    selected = min(
        eligible,
        key=lambda item: (
            float(item[2]["median_lora_minus_base_exact_loss"]),
            -float(item[2]["paired_topology_win_rate"]),
            item[0],
        ),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "selection_split": "development",
        "status": "selected" if selected else "not_supported",
        "selected_scale": selected[0] if selected else None,
        "lora_bundle_sha256": reports[0][2]["lora_bundle_sha256"],
        "reports": [
            {
                "scale": scale,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "eligible": bool(report["development_scale_selection_eligible"]),
            }
            for scale, path, report in reports
        ],
        "qualification_authorized": selected is not None,
    }
    write_json_atomic(output_path, payload)
    return payload
