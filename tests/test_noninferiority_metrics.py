from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from generation.experiment import CandidateRecord, write_candidate_manifest
from generation.ltsn_contract import sha256_file
from generation.noninferiority_metrics import (
    OUTPUT_COLUMNS,
    PromptPair,
    TransformersClapBackend,
    build_metric_rows,
    generate_noninferiority_metrics,
    nearest_neighbor_diversity,
    split_audio,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_nearest_neighbor_diversity_is_leave_one_out_cosine_distance() -> None:
    root_half = np.sqrt(0.5)
    embeddings = np.asarray(
        [[1.0, 0.0], [root_half, root_half], [-1.0, 0.0]], dtype=float
    )

    result = nearest_neighbor_diversity(embeddings)

    assert result == pytest.approx([1.0 - root_half, 1.0 - root_half, 1.0 + root_half])


def test_split_audio_zero_pads_final_window_and_retains_weights() -> None:
    segments, weights = split_audio(np.arange(5, dtype=np.float32), 3)

    assert len(segments) == 2
    assert segments[0].tolist() == [0.0, 1.0, 2.0]
    assert segments[1].tolist() == [3.0, 4.0, 0.0]
    assert weights.tolist() == [3.0, 2.0]


def test_build_metric_rows_calculates_requested_columns() -> None:
    pairs = (
        PromptPair("p1", "one", "a", "a"),
        PromptPair("p2", "two", "b", "c"),
    )
    candidate_embeddings = {
        "a": np.asarray([1.0, 0.0]),
        "b": np.asarray([0.0, 1.0]),
        "c": np.asarray([-1.0, 0.0]),
    }
    text_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    rows = build_metric_rows(pairs, candidate_embeddings, text_embeddings)

    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert float(rows[0]["prompt_baseline"]) == pytest.approx(1.0)
    assert float(rows[0]["prompt_selected"]) == pytest.approx(1.0)
    assert float(rows[1]["prompt_baseline"]) == pytest.approx(1.0)
    assert float(rows[1]["prompt_selected"]) == pytest.approx(0.0)
    assert [float(row["diversity_baseline"]) for row in rows] == pytest.approx([1.0, 1.0])
    assert [float(row["diversity_selected"]) for row in rows] == pytest.approx([2.0, 2.0])
    assert rows[0]["quality_baseline"] == ""
    assert rows[0]["quality_selected"] == ""


class _FakeBackend:
    sampling_rate = 4
    maximum_audio_samples = 4
    model_commit = "a" * 40

    def embed_audio(self, waveforms: list[np.ndarray]) -> np.ndarray:
        lookup = {
            1: np.asarray([1.0, 0.0]),
            2: np.asarray([0.0, 1.0]),
            3: np.asarray([-1.0, 0.0]),
        }
        return np.stack([lookup[int(round(float(item[0])))] for item in waveforms])

    def embed_text(self, captions: list[str]) -> np.ndarray:
        lookup = {"one": np.asarray([1.0, 0.0]), "two": np.asarray([0.0, 1.0])}
        return np.stack([lookup[item] for item in captions])


def test_generate_metrics_verifies_bindings_and_writes_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    audio_dir = run_root / "data_raw" / "candidates"
    audio_dir.mkdir(parents=True)
    candidate_specs = (("a", "p1", "one"), ("b", "p2", "two"), ("c", "p2", "two"))
    records: list[CandidateRecord] = []
    for index, (candidate_id, prompt_id, caption) in enumerate(candidate_specs):
        audio_path = audio_dir / f"{candidate_id}.wav"
        audio_path.write_bytes(f"audio-{candidate_id}".encode())
        records.append(
            CandidateRecord(
                experiment_id="test",
                prompt_id=prompt_id,
                caption=caption,
                candidate_index=index,
                candidate_id=candidate_id,
                seed=index,
                duration_seconds=2.0,
                status="scored",
                audio_relative_path=audio_path.relative_to(run_root).as_posix(),
                audio_sha256=sha256_file(audio_path),
            )
        )
    write_candidate_manifest(run_root / "manifests" / "candidates.csv", records)
    _write_rows(
        run_root / "pool_summary.csv",
        [
            {"prompt_id": "p1", "baseline_candidate_id": "a", "selected_candidate_id": "a"},
            {"prompt_id": "p2", "baseline_candidate_id": "b", "selected_candidate_id": "c"},
        ],
    )
    prompt_manifest = tmp_path / "prompts.csv"
    _write_rows(
        prompt_manifest,
        [{"prompt_id": "p1", "caption": "one"}, {"prompt_id": "p2", "caption": "two"}],
    )

    audio_values = {"a": 1.0, "b": 2.0, "c": 3.0}
    monkeypatch.setattr(
        "generation.noninferiority_metrics._load_resampled_mono",
        lambda path, target_rate: np.full(8, audio_values[path.stem], dtype=np.float32),
    )
    output = run_root / "noninferiority_metrics.csv"
    audit_path = run_root / "noninferiority_metrics.audit.json"

    audit = generate_noninferiority_metrics(
        run_root=run_root,
        prompt_manifest_path=prompt_manifest,
        output_path=output,
        audit_path=audit_path,
        backend=_FakeBackend(),
        model_id="fake/clap",
        model_revision="a" * 40,
        device="cpu",
        batch_size=2,
        segment_seconds=1.0,
        expected_prompt_count=2,
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert float(rows[1]["prompt_baseline"]) == pytest.approx(1.0)
    assert float(rows[1]["prompt_selected"]) == pytest.approx(0.0)
    assert audit["output"]["quality_columns_complete"] is False
    assert audit["output"]["sha256"] == sha256_file(output)
    assert json.loads(audit_path.read_text(encoding="utf-8"))["model"]["resolved_commit"] == (
        "a" * 40
    )


def test_clap_backend_rejects_moving_revision_before_importing_model() -> None:
    with pytest.raises(ValueError, match="immutable 40-hex"):
        TransformersClapBackend("laion/clap-htsat-fused", "main", "cpu")
