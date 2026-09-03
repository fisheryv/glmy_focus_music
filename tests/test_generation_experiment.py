from __future__ import annotations

from pathlib import Path

from generation.experiment import (
    CandidateRecord,
    build_candidate_plan,
    load_experiment_config,
    read_candidate_manifest,
    read_prompts,
    write_candidate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_formal_plan_has_32_complete_eight_candidate_pools() -> None:
    config = load_experiment_config(ROOT, Path("configs/ace_rerank_180s.toml"))
    prompts = read_prompts(ROOT, config.prompt_manifest)
    records = build_candidate_plan(config, prompts)

    assert config.ace.model == "acestep-v15-xl-turbo"
    assert config.ace.model_repository == "ACE-Step/acestep-v15-xl-turbo"
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
