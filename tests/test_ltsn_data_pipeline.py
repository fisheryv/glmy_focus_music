from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from generation.ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from generation.ltsn_pipeline import (
    RERANKING_GATE_NAME,
    SURROGATE_TRAINING_GATE_NAME,
    TrajectoryRecorder,
    TrajectorySnapshotRecord,
    build_exact_label_tables,
    load_reranking_gate,
    load_surrogate_training_gate,
    merge_trajectory_shard_manifests,
    synthetic_descriptor_rows,
    validate_snapshot_coverage,
    write_csv_atomic,
)
from generation.path_homology_exact_scorer import ExactPathHomologyScorer

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"


def _recorder(tmp_path: Path) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        tmp_path / "collection",
        model_family="acestep-v15-xl-turbo",
        ace_model_sha256="a" * 64,
        vae_sha256="b" * 64,
        engineering_smoke=True,
    )


def _write_complete_trajectory(
    collection: Path,
    *,
    trajectory_id: str,
    prompt_id: str,
    split: str = "train",
) -> Path:
    latents = collection / "latents"
    latents.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in (4, 5, 6, 8):
        sample_id = f"{trajectory_id}__step{step:02d}__b00"
        latent_path = latents / f"{sample_id}.npy"
        np.save(
            latent_path,
            np.full((16, 64), step, dtype=np.float32),
            allow_pickle=False,
        )
        rows.append(
            asdict(
                TrajectorySnapshotRecord(
                    sample_id=sample_id,
                    prompt_id=prompt_id,
                    trajectory_id=trajectory_id,
                    split=split,
                    model_family="acestep-v15-xl-turbo",
                    step_number=step,
                    timestep=0.0 if step == 8 else 1.0 - step / 8.0,
                    is_final=step == 8,
                    latent_path=latent_path.relative_to(collection).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path="",
                    audio_sha256="",
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                    engineering_smoke=False,
                )
            )
        )
    manifest = collection / "trajectory_manifest.csv"
    write_csv_atomic(manifest, rows)
    return manifest


@pytest.mark.parametrize("shape", [(32, 64), (2, 32, 64)])
def test_record_final_latent_accepts_unbatched_and_batched_arrays(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    recorder = _recorder(tmp_path)
    recorder.begin(prompt_id="prompt", trajectory_id="trajectory", split="train")
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    sample_ids = recorder.record_final_latent(values)

    assert len(sample_ids) == (shape[0] if len(shape) == 3 else 1)
    assert all(record.step_number == 8 for record in recorder.records)
    assert all(record.timestep == 0.0 and record.is_final for record in recorder.records)
    for batch_index, record in enumerate(recorder.records):
        stored = np.load(recorder.output_dir / record.latent_path, allow_pickle=False)
        expected = values[batch_index] if values.ndim == 3 else values
        np.testing.assert_array_equal(stored, expected)


def test_record_final_latent_rejects_missing_invalid_and_duplicate_values(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.begin(prompt_id="prompt", trajectory_id="trajectory", split="train")
    with pytest.raises(LTSNContractError, match=r"\[B,T,64\]"):
        recorder.record_final_latent(None)
    with pytest.raises(LTSNContractError, match=r"\[B,T,64\]"):
        recorder.record_final_latent(np.zeros((32, 63), dtype=np.float32))
    recorder.record_final_latent(np.zeros((32, 64), dtype=np.float32))
    with pytest.raises(LTSNContractError, match="already contains a final latent"):
        recorder.record_final_latent(np.zeros((32, 64), dtype=np.float32))


def test_formal_snapshot_coverage_requires_steps_4_5_6_and_8() -> None:
    complete = [
        {
            "sample_id": f"trajectory__step{step:02d}__b00",
            "trajectory_id": "trajectory",
            "step_number": step,
            "is_final": step == 8,
        }
        for step in (4, 5, 6, 8)
    ]
    validate_snapshot_coverage(complete)
    with pytest.raises(LTSNContractError, match="exactly steps"):
        validate_snapshot_coverage(complete[:-1])
    complete[-1]["is_final"] = False
    with pytest.raises(LTSNContractError, match="final-step metadata"):
        validate_snapshot_coverage(complete)


def test_recorder_resumes_only_complete_hashed_trajectories(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    manifest = _write_complete_trajectory(
        collection,
        trajectory_id="prompt__seed7",
        prompt_id="prompt",
    )
    recorder = TrajectoryRecorder(
        collection,
        model_family="acestep-v15-xl-turbo",
        ace_model_sha256="a" * 64,
        vae_sha256="b" * 64,
    )

    completed = recorder.resume_from_manifest(manifest, require_audio=False)

    assert completed == frozenset({"prompt__seed7"})
    assert len(recorder.records) == 4
    validate_snapshot_coverage(recorder.records)


def test_merge_shards_rewrites_paths_and_validates_frozen_plan(tmp_path: Path) -> None:
    output_dir = tmp_path / "trajectories"
    first = _write_complete_trajectory(
        output_dir / "shards" / "shard_00",
        trajectory_id="prompt_a__seed7",
        prompt_id="prompt_a",
    )
    second = _write_complete_trajectory(
        output_dir / "shards" / "shard_01",
        trajectory_id="prompt_b__seed8",
        prompt_id="prompt_b",
        split="development",
    )
    merged = output_dir / "trajectory_manifest.csv"

    summary = merge_trajectory_shard_manifests(
        [first, second],
        output_manifest=merged,
        expected_trajectory_plan={
            "prompt_a__seed7": ("prompt_a", "train"),
            "prompt_b__seed8": ("prompt_b", "development"),
        },
        require_audio=False,
    )

    assert summary["shards"] == 2
    assert summary["trajectories"] == 2
    assert summary["snapshots"] == 8
    with merged.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["trajectory_id"] for row in rows} == {
        "prompt_a__seed7",
        "prompt_b__seed8",
    }
    assert all(row["latent_path"].startswith("shards/shard_") for row in rows)
    validate_snapshot_coverage(rows)


def test_merge_shards_does_not_issue_manifest_for_incomplete_plan(tmp_path: Path) -> None:
    output_dir = tmp_path / "trajectories"
    shard = _write_complete_trajectory(
        output_dir / "shards" / "shard_00",
        trajectory_id="prompt_a__seed7",
        prompt_id="prompt_a",
    )
    merged = output_dir / "trajectory_manifest.csv"

    with pytest.raises(LTSNContractError, match="frozen plan"):
        merge_trajectory_shard_manifests(
            [shard],
            output_manifest=merged,
            expected_trajectory_plan={
                "prompt_a__seed7": ("prompt_a", "train"),
                "prompt_b__seed8": ("prompt_b", "train"),
            },
            require_audio=False,
        )
    assert not merged.exists()


def test_reranking_gate_requires_all_frozen_effect_conditions(tmp_path: Path) -> None:
    contract = load_fingerprint_contract(FINGERPRINT)
    payload = {
        "gate": RERANKING_GATE_NAME,
        "status": "passed",
        "fingerprint_json_sha256": contract.artifact_sha256,
        "median_loss_improvement_fraction": 0.11,
        "bootstrap_ci95_low": 0.01,
        "target_band_hit_rate_improved": True,
        "quality_noninferior": True,
        "prompt_noninferior": True,
        "diversity_preserved": True,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gate = load_reranking_gate(path, contract)
    assert gate.median_loss_improvement_fraction == pytest.approx(0.11)

    payload["quality_noninferior"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LTSNContractError, match="quality"):
        load_reranking_gate(path, contract)


def test_surrogate_training_gate_records_but_does_not_require_prompt_diversity(
    tmp_path: Path,
) -> None:
    contract = load_fingerprint_contract(FINGERPRINT)
    payload = {
        "schema_version": 2,
        "gate": SURROGATE_TRAINING_GATE_NAME,
        "status": "passed",
        "scope": "exact_labeling_and_surrogate_training_only",
        "fingerprint_json_sha256": contract.artifact_sha256,
        "median_loss_improvement_fraction": 0.11,
        "bootstrap_ci95_low": 0.01,
        "target_band_hit_rate_improved": True,
        "all_selected_technical_quality_eligible": True,
        "quality_noninferior": True,
        "prompt_noninferior": False,
        "diversity_preserved": False,
        "guidance_promotion_eligible": False,
    }
    path = tmp_path / "training_gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gate = load_surrogate_training_gate(path, contract)
    assert gate.prompt_noninferior is False
    assert gate.diversity_preserved is False

    payload["guidance_promotion_eligible"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LTSNContractError, match="must not promote"):
        load_surrogate_training_gate(path, contract)


def test_smoke_label_builder_issues_hashed_nonqualifying_manifest(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    latents = collection / "latents"
    latents.mkdir(parents=True)
    records = []
    for index, split in enumerate(("train", "development", "calibration", "qualification")):
        latent = np.full((32, 64), index / 10.0, dtype=np.float32)
        latent_path = latents / f"sample_{index}.npy"
        np.save(latent_path, latent, allow_pickle=False)
        records.append(
            asdict(
                TrajectorySnapshotRecord(
                    sample_id=f"sample_{index}",
                    prompt_id=f"prompt_{index}",
                    trajectory_id=f"trajectory_{index}",
                    split=split,
                    model_family="acestep-v15-xl-turbo",
                    step_number=4,
                    timestep=0.75,
                    is_final=False,
                    latent_path=latent_path.relative_to(collection).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path="",
                    audio_sha256="",
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                    engineering_smoke=True,
                )
            )
        )
    trajectory_manifest = collection / "trajectory_manifest.csv"
    write_csv_atomic(trajectory_manifest, records)
    scorer = ExactPathHomologyScorer.from_json(FINGERPRINT)
    descriptor_table = tmp_path / "descriptors.csv"
    write_csv_atomic(
        descriptor_table,
        synthetic_descriptor_rows(
            trajectory_manifest,
            pitch_dimensions=len(scorer.transforms["pitch"]["input_features"]),
        ),
    )

    summary = build_exact_label_tables(
        trajectory_manifest=trajectory_manifest,
        descriptor_table=descriptor_table,
        output_manifest=tmp_path / "labels" / "manifest.csv",
        exact_label_table=tmp_path / "labels" / "exact.csv",
        split_manifest=tmp_path / "labels" / "splits.json",
        scorer=scorer,
        gate=None,
        engineering_smoke=True,
    )

    assert summary["samples"] == 4
    assert not summary["qualification_eligible"]
    assert len(summary["exact_label_table_sha256"]) == 64


def test_formal_label_builder_rejects_three_step_trajectory_manifest(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    latents = collection / "latents"
    latents.mkdir(parents=True)
    records = []
    for step in (4, 5, 6):
        latent_path = latents / f"step_{step:02d}.npy"
        np.save(latent_path, np.zeros((32, 64), dtype=np.float32), allow_pickle=False)
        records.append(
            asdict(
                TrajectorySnapshotRecord(
                    sample_id=f"trajectory__step{step:02d}__b00",
                    prompt_id="prompt",
                    trajectory_id="trajectory",
                    split="train",
                    model_family="acestep-v15-xl-turbo",
                    step_number=step,
                    timestep=1.0 - step / 8.0,
                    is_final=False,
                    latent_path=latent_path.relative_to(collection).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path="",
                    audio_sha256="",
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                    engineering_smoke=False,
                )
            )
        )
    trajectory_manifest = collection / "trajectory_manifest.csv"
    write_csv_atomic(trajectory_manifest, records)

    scorer = ExactPathHomologyScorer.from_json(FINGERPRINT)
    with pytest.raises(LTSNContractError, match="exactly steps"):
        build_exact_label_tables(
            trajectory_manifest=trajectory_manifest,
            descriptor_table=tmp_path / "unused.csv",
            output_manifest=tmp_path / "labels" / "manifest.csv",
            exact_label_table=tmp_path / "labels" / "exact.csv",
            split_manifest=tmp_path / "labels" / "splits.json",
            scorer=scorer,
            gate=None,
            engineering_smoke=False,
        )
