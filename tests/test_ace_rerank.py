from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from generation.exact_features import write_descriptor_csv
from generation.experiment import (
    AceConfig,
    CandidateRecord,
    ExperimentConfig,
    ScoringConfig,
    build_candidate_plan,
    load_experiment_config,
    read_candidate_manifest,
    read_prompts,
    write_candidate_manifest,
)
from generation.fake_backend import FakeMusicBackend
from generation.ltsn_contract import sha256_file
from generation.ltsn_pipeline import load_reranking_gate
from generation.path_homology_exact_scorer import ExactPathHomologyScorer
from generation.rerank_experiment import (
    _validated_noninferiority,
    ensure_experiment,
    evaluate_noninferiority_table,
    generate_candidates,
    initialize_noninferiority_report,
    issue_reranking_gate,
    rank_and_summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_target_inputs(target: Path) -> None:
    (target / "metadata").mkdir(parents=True)
    model_dir = target / "features" / "models"
    model_dir.mkdir(parents=True)
    for name in ("state_model.npz", "state_model.json"):
        shutil.copy2(ROOT / "features" / "models" / name, model_dir / name)
    shutil.copy2(
        ROOT / "features" / "models" / "pitch_v2_codebook.npz",
        model_dir / "pitch_v2_codebook.npz",
    )
    shutil.copy2(
        ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json",
        target / "metadata" / "focus_path_homology_fingerprint_v2.json",
    )
    (target / "configs").mkdir(parents=True)
    shutil.copy2(
        ROOT / "configs" / "ace_reranking_noninferiority.json",
        target / "configs" / "ace_reranking_noninferiority.json",
    )


def test_formal_plan_has_32_complete_eight_candidate_pools() -> None:
    config = load_experiment_config(ROOT, Path("configs/ace_rerank_180s.toml"))
    prompts = read_prompts(ROOT, config.prompt_manifest)
    records = build_candidate_plan(config, prompts)

    assert len(prompts) == 32
    assert len(records) == 256
    assert len({record.seed for record in records}) == 256
    pool_sizes = [sum(item.prompt_id == prompt.prompt_id for item in records) for prompt in prompts]
    assert pool_sizes == [8] * 32


def test_candidate_manifest_round_trip_preserves_audit_fields(tmp_path: Path) -> None:
    record = CandidateRecord(
        experiment_id="test",
        prompt_id="p1",
        caption="steady instrumental",
        candidate_index=0,
        candidate_id="p1__c00__s7",
        seed=7,
        duration_seconds=180,
        status="generated",
        audio_relative_path="data_raw/candidates/test.wav",
        audio_sha256="abc",
        latent_relative_path="latents/test.npz",
        latent_sha256="def",
        metadata_relative_path="metadata/candidates/test.json",
    )
    path = tmp_path / "candidates.csv"
    write_candidate_manifest(path, [record])

    assert read_candidate_manifest(path) == [record]


def test_fake_generation_is_resumable_and_hashed(tmp_path: Path) -> None:
    _copy_target_inputs(tmp_path)
    prompt_dir = tmp_path / "generation" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "test.csv").write_text(
        "prompt_id,caption,bpm,keyscale,timesignature\n"
        "p1,steady instrumental focus texture,84,,4\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        run_id="fake_test",
        prompt_manifest="generation/prompts/test.csv",
        candidate_count=2,
        duration_seconds=10,
        workers=1,
        save_latents=False,
        run_directory="runs",
        ace=AceConfig(),
        scoring=ScoringConfig(
            minimum_prompt_pools=2,
            permutations=99,
            bootstrap_resamples=100,
        ),
    )
    _, planned = ensure_experiment(tmp_path, config)
    assert len(planned) == 2

    generated = generate_candidates(tmp_path, config, FakeMusicBackend())
    assert all(record.status == "generated" for record in generated)
    assert all(record.audio_sha256 for record in generated)
    paths = [tmp_path / "runs" / "fake_test" / record.audio_relative_path for record in generated]
    mtimes = [path.stat().st_mtime_ns for path in paths]

    resumed = generate_candidates(tmp_path, config, FakeMusicBackend())
    assert [path.stat().st_mtime_ns for path in paths] == mtimes
    assert resumed == generated

    descriptor_path = tmp_path / "runs" / "fake_test" / "descriptors_18d.csv"
    descriptor_path.write_text("stale\n", encoding="utf-8")
    paths[0].write_bytes(b"corrupt")
    repaired = generate_candidates(tmp_path, config, FakeMusicBackend())
    assert repaired[0].audio_sha256
    assert not descriptor_path.exists()


def test_configuration_change_cannot_reuse_run_id(tmp_path: Path) -> None:
    _copy_target_inputs(tmp_path)
    prompt_dir = tmp_path / "generation" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "test.csv").write_text(
        "prompt_id,caption,bpm,keyscale,timesignature\np1,steady instrumental,84,,4\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        run_id="frozen",
        prompt_manifest="generation/prompts/test.csv",
        candidate_count=2,
        duration_seconds=10,
        run_directory="runs",
        scoring=ScoringConfig(
            minimum_prompt_pools=2,
            permutations=99,
            bootstrap_resamples=100,
        ),
    )
    ensure_experiment(tmp_path, config)

    with pytest.raises(ValueError, match="different configuration"):
        ensure_experiment(tmp_path, replace(config, duration_seconds=20))


def test_exact_18d_reranking_and_gate_issuance_are_separate(tmp_path: Path) -> None:
    _copy_target_inputs(tmp_path)
    prompt_dir = tmp_path / "generation" / "prompts"
    prompt_dir.mkdir(parents=True)
    prompt_manifest = prompt_dir / "formal.csv"
    prompt_manifest.write_text(
        "prompt_id,caption,bpm,keyscale,timesignature\n"
        + "".join(
            f"prompt_{index:02d},steady instrumental focus music,80,,4\n"
            for index in range(32)
        ),
        encoding="utf-8",
    )
    scorer = ExactPathHomologyScorer.from_json(
        ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
    )
    config = ExperimentConfig(
        run_id="formal_18d",
        prompt_manifest="generation/prompts/formal.csv",
        candidate_count=8,
        duration_seconds=180.0,
        run_directory="runs",
        scoring=ScoringConfig(
            minimum_prompt_pools=20,
            permutations=99,
            bootstrap_resamples=100,
        ),
    )
    run_root, records = ensure_experiment(tmp_path, config)
    codebook_sha256 = sha256_file(
        tmp_path / "features" / "models" / "pitch_v2_codebook.npz"
    )
    descriptors = []
    for record in records:
        record.status = "generated"
        record.audio_relative_path = f"data_raw/candidates/{record.candidate_id}.wav"
        record.audio_sha256 = "a" * 64
        pitch = [0.0] * len(scorer.transforms["pitch"]["input_features"])
        prompt_index = int(record.prompt_id.rsplit("_", 1)[1])
        if record.candidate_index == 0 and prompt_index < 9:
            pitch[7] = 1.0
        elif record.candidate_index > 1:
            pitch[7] = 0.2 + 0.01 * record.candidate_index
        exact = scorer.score(pitch, [0.0], [0.0])
        descriptors.append(
                {
                    "candidate_id": record.candidate_id,
                    "prompt_id": record.prompt_id,
                    "candidate_index": record.candidate_index,
                    "seed": record.seed,
                    "audio_sha256": "a" * 64,
                    "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                    "feature_order_json": json.dumps(list(scorer.contract.feature_order)),
                    "pitch_descriptors_json": json.dumps(pitch),
                    "acoustic_loop_score": 0.0,
                    "chroma_loop_score": 0.0,
                    "coordinates_json": json.dumps(exact.coordinates[0].tolist()),
                    "focus_logit": float(exact.focus_logit[0]),
                    "focus_probability": float(exact.focus_probability[0]),
                    "focus_band_loss": float(exact.focus_band_loss[0]),
                    "pitch_block_l2_norm": float(exact.pitch_block_l2_norm[0]),
                    "phase_block_l2_norm": float(exact.phase_block_l2_norm[0]),
                    "pitch_v2_codebook_sha256": codebook_sha256,
                    "label_source": "decoded_candidate_exact_18d_v1",
                    "raw_sample_rate": 48_000,
                    "raw_duration_seconds": 180.0,
                    "raw_peak": 0.8,
                    "raw_rms": 0.1,
                    "raw_clip_fraction": 0.0,
                    "raw_dc_offset": 0.0,
                }
            )
    write_candidate_manifest(run_root / "manifests" / "candidates.csv", records)
    write_descriptor_csv(run_root / "descriptors_18d.csv", descriptors)

    summary = rank_and_summarize(tmp_path, config, records, descriptors)

    assert summary["topology_passed"] is True
    assert summary["verdict"] == "topology_pass_noninferiority_pending"
    assert summary["prompt_pools"] == 32
    assert summary["evaluable_positive_loss_prompt_pools"] == 9
    assert summary["complete_prompt_pools_sufficient"] is True
    assert summary["topology_blockers"] == []
    assert summary["median_loss_improvement_fraction"] == pytest.approx(1.0)
    assert not (tmp_path / "gate.json").exists()

    report_path = run_root / "noninferiority_report.json"
    report = initialize_noninferiority_report(tmp_path, config, report_path)
    evidence_path = run_root / "noninferiority_results.csv"
    assert not any(item["passed"] for item in report["criteria"].values())
    with (run_root / "pool_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        pool_rows = list(csv.DictReader(handle))
    evidence_rows = [
        {
            "prompt_id": row["prompt_id"],
            "baseline_candidate_id": row["baseline_candidate_id"],
            "selected_candidate_id": row["selected_candidate_id"],
            "quality_baseline": 0.5,
            "quality_selected": 0.6,
            "prompt_baseline": 0.5,
            "prompt_selected": 0.6,
            "diversity_baseline": 0.5,
            "diversity_selected": 0.6,
        }
        for row in pool_rows
    ]
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(evidence_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)
    report = evaluate_noninferiority_table(
        tmp_path, config, evidence_path, report_path
    )
    assert all(item["passed"] for item in report["criteria"].values())
    gate_path = tmp_path / "gate.json"
    issue_reranking_gate(tmp_path, config, report_path, gate_path)

    gate = load_reranking_gate(gate_path, scorer.contract)
    assert gate.median_loss_improvement_fraction == pytest.approx(1.0)


def test_noninferiority_report_rejects_wrong_selection_hash(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        run_id="blocked",
        prompt_manifest="unused.csv",
        run_directory="runs",
    )
    summary = {
        "topology_passed": True,
        "fingerprint_json_sha256": "a" * 64,
        "candidate_manifest_sha256": "b" * 64,
        "selection_table_sha256": "c" * 64,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config.run_id,
        "fingerprint_json_sha256": "a" * 64,
        "candidate_manifest_sha256": "b" * 64,
        "selection_table_sha256": "wrong",
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="selection_table_sha256"):
        _validated_noninferiority(report_path, summary, config)
