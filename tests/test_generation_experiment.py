from __future__ import annotations

import csv
from collections import Counter
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


def test_ltsn_prompt_manifest_is_unique_grouped_and_balanced() -> None:
    path = ROOT / "metadata" / "ltsn_prompts.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 512
    assert Counter(row["split"] for row in rows) == {
        "train": 320,
        "development": 64,
        "calibration": 64,
        "qualification": 64,
    }
    assert len({row["prompt_id"] for row in rows}) == len(rows)
    assert len({row["caption"] for row in rows}) == len(rows)
    assert all(not row["seed"] for row in rows)
    assert all(row["bpm"].isdigit() for row in rows)
    assert all(row["timesignature"] == "4" for row in rows)

    family_splits: dict[str, set[str]] = {}
    family_counts: Counter[str] = Counter()
    for row in rows:
        family = row["prompt_id"].rsplit("__v", 1)[0]
        family_splits.setdefault(family, set()).add(row["split"])
        family_counts[family] += 1
    assert len(family_splits) == 32
    assert all(splits and len(splits) == 1 for splits in family_splits.values())
    assert set(family_counts.values()) == {16}
