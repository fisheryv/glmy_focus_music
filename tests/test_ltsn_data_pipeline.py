from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from generation.ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from generation.ltsn_pipeline import (
    RERANKING_GATE_NAME,
    SURROGATE_TRAINING_GATE_NAME,
    TrajectorySnapshotRecord,
    build_exact_label_tables,
    load_reranking_gate,
    load_surrogate_training_gate,
    synthetic_descriptor_rows,
    write_csv_atomic,
)
from generation.path_homology_exact_scorer import ExactPathHomologyScorer

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"


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
