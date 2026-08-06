from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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
from generation.rerank_experiment import ensure_experiment, generate_candidates
from generation.target_profile import CORE_FEATURES, fit_target_profile

ROOT = Path(__file__).resolve().parents[1]


def _copy_target_inputs(target: Path) -> None:
    (target / "metadata").mkdir(parents=True)
    for name in ("tda_features.csv", "repetition_homology_features.csv"):
        shutil.copy2(ROOT / "metadata" / name, target / "metadata" / name)
    model_dir = target / "features" / "models"
    model_dir.mkdir(parents=True)
    for name in ("state_model.npz", "state_model.json"):
        shutil.copy2(ROOT / "features" / "models" / name, model_dir / name)


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


def test_target_profile_uses_complete_discovery_focus_rows(tmp_path: Path) -> None:
    _copy_target_inputs(tmp_path)
    profile = fit_target_profile(tmp_path)

    assert profile.feature_names == CORE_FEATURES
    assert profile.sample_count == 130
    assert profile.center[0] == pytest.approx(1.0004947)
    assert profile.center[3] == pytest.approx(0.4116638)
    assert np.all(np.linalg.eigvalsh(np.asarray(profile.precision)) > 0)


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

    descriptor_path = tmp_path / "runs" / "fake_test" / "descriptors.csv"
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
