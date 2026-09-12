from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from generation.experiment import (
    CandidateRecord,
    ExperimentConfig,
    read_candidate_manifest,
    write_candidate_manifest,
)
from generation.ltsn_contract import sha256_file
from generation.rerank_cli import build_parser
from generation.rerank_multigpu import (
    assign_prompt_shards,
    generate_candidates_multi_device,
    merge_generation_shards,
    validate_generation_devices,
)


def _plan(prompt_count: int = 4, candidate_count: int = 3) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for prompt_index in range(prompt_count):
        for candidate_index in range(candidate_count):
            seed = 1000 + prompt_index * candidate_count + candidate_index
            records.append(
                CandidateRecord(
                    experiment_id="multi",
                    prompt_id=f"p{prompt_index}",
                    caption=f"prompt {prompt_index}",
                    candidate_index=candidate_index,
                    candidate_id=f"p{prompt_index}__c{candidate_index:02d}__s{seed}",
                    seed=seed,
                    duration_seconds=180.0,
                )
            )
    return records


def _write_complete_shards(
    project_root: Path,
    config: ExperimentConfig,
    assignments: list[list[CandidateRecord]],
) -> None:
    run_root = project_root / config.run_directory / config.run_id
    for shard_index, expected_records in enumerate(assignments):
        shard_root = run_root / "generation_shards" / f"shard_{shard_index:03d}"
        records: list[CandidateRecord] = []
        for expected in expected_records:
            audio_path = shard_root / "data_raw" / "candidates" / f"{expected.candidate_id}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(f"audio:{expected.candidate_id}".encode())
            audio_sha256 = sha256_file(audio_path)
            metadata_path = (
                shard_root / "metadata" / "candidates" / f"{expected.candidate_id}.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "candidate_id": expected.candidate_id,
                        "prompt_id": expected.prompt_id,
                        "caption": expected.caption,
                        "seed": expected.seed,
                        "duration_seconds": expected.duration_seconds,
                        "audio_sha256": audio_sha256,
                        "latent_sha256": "",
                        "backend": {"device": f"cuda:{shard_index}"},
                    }
                ),
                encoding="utf-8",
            )
            records.append(
                replace(
                    expected,
                    status="generated",
                    audio_relative_path=audio_path.relative_to(shard_root).as_posix(),
                    audio_sha256=audio_sha256,
                    metadata_relative_path=metadata_path.relative_to(shard_root).as_posix(),
                    generated_at="2026-09-12T00:00:00+00:00",
                )
            )
        write_candidate_manifest(shard_root / "manifest.csv", records)


def test_prompt_shards_are_balanced_and_never_split_pools() -> None:
    records = _plan(prompt_count=9, candidate_count=4)

    shards = assign_prompt_shards(records, 4)

    prompt_to_shards: dict[str, set[int]] = {}
    for shard_index, shard in enumerate(shards):
        for record in shard:
            prompt_to_shards.setdefault(record.prompt_id, set()).add(shard_index)
    assert all(len(indices) == 1 for indices in prompt_to_shards.values())
    assert [len({record.prompt_id for record in shard}) for shard in shards] == [3, 2, 2, 2]
    assert [record.candidate_id for shard in shards for record in shard] != [
        record.candidate_id for record in records
    ]


def test_device_contract_requires_unique_explicit_cuda_indices() -> None:
    assert validate_generation_devices(["cuda:0", "cuda:3"]) == ("cuda:0", "cuda:3")
    with pytest.raises(ValueError, match="duplicates"):
        validate_generation_devices(["cuda:0", "cuda:0"])
    with pytest.raises(ValueError, match="explicit cuda:N"):
        validate_generation_devices(["cuda"])


def test_cli_parses_multi_device_generation() -> None:
    args = build_parser().parse_args(
        ["run", "--backend", "ace", "--devices", "cuda:0", "cuda:1"]
    )
    assert args.devices == ["cuda:0", "cuda:1"]


def test_merge_validates_shards_and_restores_frozen_plan_order(tmp_path: Path) -> None:
    config = ExperimentConfig(
        run_id="multi",
        prompt_manifest="unused.csv",
        candidate_count=3,
        save_latents=False,
        run_directory="runs",
    )
    official = _plan()
    assignments = assign_prompt_shards(official, 2)
    _write_complete_shards(tmp_path, config, assignments)

    merged = merge_generation_shards(tmp_path, config, official, assignments)

    assert [record.candidate_id for record in merged] == [
        record.candidate_id for record in official
    ]
    assert all(record.status == "generated" for record in merged)
    run_root = tmp_path / "runs" / "multi"
    assert all((run_root / record.audio_relative_path).is_file() for record in merged)
    assert not (run_root / "manifests" / "candidates.csv").exists()


@pytest.mark.parametrize("damage", ["missing", "hash"])
def test_merge_rejects_missing_or_hash_changed_audio(tmp_path: Path, damage: str) -> None:
    config = ExperimentConfig(
        run_id="multi",
        prompt_manifest="unused.csv",
        candidate_count=3,
        save_latents=False,
        run_directory="runs",
    )
    official = _plan()
    assignments = assign_prompt_shards(official, 2)
    _write_complete_shards(tmp_path, config, assignments)
    shard_root = tmp_path / "runs" / "multi" / "generation_shards" / "shard_000"
    records = read_candidate_manifest(shard_root / "manifest.csv")
    audio_path = shard_root / records[0].audio_relative_path
    if damage == "missing":
        audio_path.unlink()
    else:
        audio_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="audio is missing|audio hash mismatch"):
        merge_generation_shards(tmp_path, config, official, assignments)


def test_merge_rejects_duplicate_cross_shard_assignment(tmp_path: Path) -> None:
    config = ExperimentConfig(
        run_id="multi",
        prompt_manifest="unused.csv",
        candidate_count=3,
        save_latents=False,
        run_directory="runs",
    )
    official = _plan()
    assignments = assign_prompt_shards(official, 2)
    assignments[1] = [assignments[0][0], *assignments[1]]
    _write_complete_shards(tmp_path, config, assignments)

    with pytest.raises(ValueError, match="multiple shards"):
        merge_generation_shards(tmp_path, config, official, assignments)


def test_worker_failure_does_not_publish_formal_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ExperimentConfig(
        run_id="multi",
        prompt_manifest="unused.csv",
        candidate_count=3,
        save_latents=False,
        run_directory="runs",
    )
    official = _plan()
    run_root = tmp_path / "runs" / "multi"
    manifest_path = run_root / "manifests" / "candidates.csv"
    write_candidate_manifest(manifest_path, official)
    initial_manifest = manifest_path.read_bytes()

    class FailedProcess:
        exitcode = 1

        def __init__(self, **_: object) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FailedContext:
        Process = FailedProcess

    monkeypatch.setattr(
        "generation.rerank_multigpu.ensure_experiment",
        lambda _root, _config: (run_root, official),
    )
    monkeypatch.setattr(
        "generation.rerank_multigpu.mp.get_context", lambda _method: FailedContext()
    )

    with pytest.raises(RuntimeError, match="formal candidate manifest was not published"):
        generate_candidates_multi_device(
            tmp_path, config, ["cuda:0", "cuda:1"], retry_failed=True
        )

    assert manifest_path.read_bytes() == initial_manifest
    audit = json.loads(
        (run_root / "manifests" / "multigpu_generation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "failed"
    assert audit["formal_manifest_published"] is False
