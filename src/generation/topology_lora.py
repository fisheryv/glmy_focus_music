"""Exact-reranking teacher export for topology-distilled ACE-Step LoRA."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment import read_candidate_manifest
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import load_reranking_gate, write_csv_atomic, write_json_atomic
from .path_homology_exact_scorer import ExactPathHomologyScorer

TOPOLOGY_LORA_EXPERIMENT = "exact_reranking_distilled_topology_lora_v1"
TOPOLOGY_LORA_TAG = "topology_focus"
ALLOWED_PROMPT_SPLITS = ("train", "development", "calibration", "qualification")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _family_id(prompt_id: str) -> str:
    return prompt_id.split("__v", 1)[0]


def build_reranking_prompt_splits(
    source_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Freeze family-disjoint reranking manifests from the formal prompt table."""

    rows = _read_csv(source_path)
    required = {"prompt_id", "caption", "split", "bpm", "keyscale", "timesignature"}
    if not required.issubset(rows[0]):
        raise LTSNContractError("topology LoRA prompt manifest is missing required columns")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise LTSNContractError("topology LoRA prompt IDs are duplicated")
    observed_splits = {row["split"] for row in rows}
    if observed_splits != set(ALLOWED_PROMPT_SPLITS):
        raise LTSNContractError(f"unexpected prompt splits: {sorted(observed_splits)}")
    family_splits: dict[str, set[str]] = {}
    for row in rows:
        family_splits.setdefault(_family_id(row["prompt_id"]), set()).add(row["split"])
    leaked = sorted(family for family, splits in family_splits.items() if len(splits) != 1)
    if leaked:
        raise LTSNContractError(f"prompt-family leakage detected: {leaked[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    columns = ("prompt_id", "caption", "split", "bpm", "keyscale", "timesignature")
    for split in ALLOWED_PROMPT_SPLITS:
        selected = [
            {name: row.get(name, "") for name in columns}
            for row in rows
            if row["split"] == split
        ]
        path = output_dir / f"{split}.csv"
        write_csv_atomic(path, selected)
        outputs[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "prompts": len(selected),
            "families": len({_family_id(row["prompt_id"]) for row in selected}),
        }
    payload = {
        "schema_version": 1,
        "experiment": TOPOLOGY_LORA_EXPERIMENT,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "family_disjoint": True,
        "outputs": outputs,
    }
    write_json_atomic(output_dir / "prompt_split_manifest.json", payload)
    return payload


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LTSNContractError(f"{label} must be a JSON object")
    return payload


def _validate_reranking_run(
    *,
    run_dir: Path,
    fingerprint_path: Path,
    reranking_gate_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    summary_path = run_dir / "summary.json"
    scores_path = run_dir / "scores.csv"
    pool_path = run_dir / "pool_summary.csv"
    candidate_path = run_dir / "manifests" / "candidates.csv"
    for path in (summary_path, scores_path, pool_path, candidate_path):
        if not path.is_file():
            raise LTSNContractError(f"reranking artifact is missing: {path}")
    summary = _load_json(summary_path, "reranking summary")
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    load_reranking_gate(reranking_gate_path, scorer.contract)
    expected = {
        "fingerprint_json_sha256": scorer.contract.artifact_sha256,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "selection_table_sha256": sha256_file(scores_path),
        "pool_summary_sha256": sha256_file(pool_path),
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise LTSNContractError(f"reranking run binding changed: {name}")
    return summary, _read_csv(scores_path), _read_csv(pool_path)


def _sample_payload(
    *,
    record: Any,
    run_dir: Path,
    role: str,
    activation_tag: str,
    score: Mapping[str, str],
) -> dict[str, Any]:
    audio_path = run_dir / record.audio_relative_path
    return {
        "filename": audio_path.name,
        "audio_path": str(audio_path.resolve()),
        "caption": record.caption,
        "lyrics": "[Instrumental]",
        "bpm": record.bpm,
        "keyscale": record.keyscale,
        "timesignature": record.timesignature,
        "duration": record.duration_seconds,
        "is_instrumental": True,
        "custom_tag": activation_tag if role == "winner" else "",
        "prompt_override": "caption",
        "teacher_role": role,
        "candidate_id": record.candidate_id,
        "prompt_id": record.prompt_id,
        "seed": record.seed,
        "audio_sha256": record.audio_sha256,
        "focus_band_loss": float(score["focus_band_loss"]),
    }


def export_lora_teacher_dataset(
    *,
    reranking_run_dir: Path,
    prompt_manifest_path: Path,
    fingerprint_path: Path,
    reranking_gate_path: Path,
    output_dir: Path,
    activation_tag: str = TOPOLOGY_LORA_TAG,
    include_baseline_replay: bool = True,
) -> dict[str, Any]:
    """Export exact-selected winners and optional untagged matched replay samples."""

    if not activation_tag.strip() or "," in activation_tag:
        raise ValueError("activation tag must be non-empty and contain no comma")
    run_dir = reranking_run_dir.resolve()
    summary, score_rows, pool_rows = _validate_reranking_run(
        run_dir=run_dir,
        fingerprint_path=fingerprint_path,
        reranking_gate_path=reranking_gate_path,
    )
    prompt_rows = _read_csv(prompt_manifest_path)
    if {row.get("split") for row in prompt_rows} != {"train"}:
        raise LTSNContractError("LoRA teacher export accepts only the train prompt split")
    prompt_ids = {row["prompt_id"] for row in prompt_rows}
    if len(prompt_ids) != len(prompt_rows):
        raise LTSNContractError("LoRA teacher prompt IDs are duplicated")

    records = read_candidate_manifest(run_dir / "manifests" / "candidates.csv")
    record_by_id = {record.candidate_id: record for record in records}
    score_by_id = {row["candidate_id"]: row for row in score_rows}
    if set(record_by_id) != set(score_by_id):
        raise LTSNContractError("reranking scores do not cover the candidate manifest")
    pools = {row["prompt_id"]: row for row in pool_rows}
    if set(pools) != prompt_ids:
        raise LTSNContractError("reranking pools do not match the train prompt split")

    samples: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for prompt_id in sorted(prompt_ids):
        pool = pools[prompt_id]
        winner_id = pool["selected_candidate_id"]
        baseline_id = pool["baseline_candidate_id"]
        chosen = [(winner_id, "winner")]
        if include_baseline_replay and baseline_id != winner_id:
            chosen.append((baseline_id, "baseline_replay"))
        for candidate_id, role in chosen:
            record = record_by_id[candidate_id]
            audio_path = run_dir / record.audio_relative_path
            if (
                not audio_path.is_file()
                or sha256_file(audio_path) != record.audio_sha256
                or score_by_id[candidate_id].get("technical_quality_eligible") != "1"
            ):
                raise LTSNContractError(f"ineligible or changed teacher audio: {candidate_id}")
            sample = _sample_payload(
                record=record,
                run_dir=run_dir,
                role=role,
                activation_tag=activation_tag,
                score=score_by_id[candidate_id],
            )
            samples.append(sample)
            audit_rows.append(
                {
                    "prompt_id": prompt_id,
                    "candidate_id": candidate_id,
                    "teacher_role": role,
                    "activation_tag": sample["custom_tag"],
                    "audio_path": sample["audio_path"],
                    "audio_sha256": sample["audio_sha256"],
                    "focus_band_loss": sample["focus_band_loss"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "metadata": {
            "name": TOPOLOGY_LORA_EXPERIMENT,
            "custom_tag": "",
            "tag_position": "prepend",
            "genre_ratio": 0,
            "all_instrumental": True,
            "source": "exact_18d_best_of_n_winners",
        },
        "samples": samples,
    }
    dataset_path = output_dir / "ace_lora_dataset.json"
    manifest_path = output_dir / "teacher_manifest.csv"
    write_json_atomic(dataset_path, dataset)
    write_csv_atomic(manifest_path, audit_rows)
    report = {
        "schema_version": 1,
        "experiment": TOPOLOGY_LORA_EXPERIMENT,
        "teacher_dataset_only": True,
        "qualification_authority": False,
        "production_authorization": False,
        "teacher_source": "frozen_exact_18d_best_of_n",
        "activation_tag": activation_tag,
        "train_prompts": len(prompt_ids),
        "winner_samples": sum(row["teacher_role"] == "winner" for row in audit_rows),
        "baseline_replay_samples": sum(
            row["teacher_role"] == "baseline_replay" for row in audit_rows
        ),
        "dataset_samples": len(samples),
        "fingerprint_json_sha256": summary["fingerprint_json_sha256"],
        "reranking_run_dir": str(run_dir),
        "reranking_summary_sha256": sha256_file(run_dir / "summary.json"),
        "reranking_gate_sha256": sha256_file(reranking_gate_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
        "dataset_json_sha256": sha256_file(dataset_path),
        "teacher_manifest_sha256": sha256_file(manifest_path),
        "next_stage": "ace_step_lora_preprocess",
    }
    write_json_atomic(output_dir / "teacher_report.json", report)
    return report
