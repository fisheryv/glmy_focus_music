from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import generation.ltsn_exact_labeling as exact_labeling
from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import TrajectorySnapshotRecord, write_csv_atomic
from generation.ltsn_storage import (
    materialize_file,
    remove_generated_audio,
    remove_tree_within,
)


def test_materialize_file_auto_preserves_content_and_reuses_target(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "work" / "target.wav"
    source.write_bytes(b"immutable snapshot" * 1024)
    expected = sha256_file(source)

    method = materialize_file(source, target, mode="auto", expected_sha256=expected)

    assert method in {"reflink", "hardlink", "copy"}
    assert target.read_bytes() == source.read_bytes()
    assert materialize_file(source, target, mode="auto", expected_sha256=expected) == "existing"


def test_materialize_file_rejects_existing_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    source.write_bytes(b"source")
    target.write_bytes(b"different")

    with pytest.raises(LTSNContractError, match="hash-mismatched"):
        materialize_file(source, target, mode="copy", expected_sha256=sha256_file(source))


def test_cleanup_helpers_are_confined_to_declared_roots(tmp_path: Path) -> None:
    root = tmp_path / "exact_work"
    batch = root / "batches" / "batch_00000"
    batch.mkdir(parents=True)
    (batch / "scratch.bin").write_bytes(b"scratch")

    remove_tree_within(batch, root)
    assert not batch.exists()
    with pytest.raises(LTSNContractError, match="refusing cleanup"):
        remove_tree_within(root, root)

    output_root = tmp_path / "collection" / "final_audio"
    audio = output_root / "trajectory" / "generated.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    remove_generated_audio(audio, output_root)
    assert not audio.exists()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"keep")
    with pytest.raises(LTSNContractError, match="refusing to delete"):
        remove_generated_audio(outside, output_root)
    assert outside.exists()


def _trajectory_manifest(tmp_path: Path, count: int = 5) -> Path:
    collection = tmp_path / "collection"
    latents = collection / "latents"
    latents.mkdir(parents=True)
    records = []
    for index in range(count):
        latent_path = latents / f"sample_{index}.npy"
        np.save(latent_path, np.full((8, 64), index, dtype=np.float32), allow_pickle=False)
        records.append(
            asdict(
                TrajectorySnapshotRecord(
                    sample_id=f"sample_{index}",
                    prompt_id=f"prompt_{index}",
                    trajectory_id=f"trajectory_{index}",
                    split="train",
                    model_family="acestep-v15-xl-turbo",
                    step_number=4,
                    timestep=0.5,
                    is_final=False,
                    latent_path=latent_path.relative_to(collection).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path=f"snapshot_audio/sample_{index}.wav",
                    audio_sha256=f"{index:064x}",
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                    engineering_smoke=False,
                )
            )
        )
    manifest = collection / "trajectory_manifest.csv"
    write_csv_atomic(manifest, records)
    return manifest


def test_exact_label_batches_resume_and_cleanup_intermediates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    model_dir = project_root / "features" / "models"
    model_dir.mkdir(parents=True)
    np.savez(model_dir / "pitch_v2_codebook.npz", centers=np.zeros((16, 6)))
    manifest = _trajectory_manifest(tmp_path)
    calls: list[list[str]] = []

    def fake_extract_batch(**kwargs: object) -> tuple[list[dict[str, object]], Counter[str]]:
        rows = kwargs["rows"]
        assert isinstance(rows, list)
        work_dir = kwargs["work_dir"]
        assert isinstance(work_dir, Path)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "large-reconstructible-file.bin").write_bytes(b"temporary")
        calls.append([str(row["sample_id"]) for row in rows])
        output = [
            {
                "sample_id": str(row["sample_id"]),
                "pitch_descriptors_json": "[]",
                "acoustic_loop_score": 0.1,
                "chroma_loop_score": 0.2,
                "ood_label": 0,
                "label_source": "decoded_snapshot_exact_v1",
                "audio_sha256": str(row["audio_sha256"]),
                "pitch_v2_codebook_sha256": "c" * 64,
            }
            for row in rows
        ]
        return output, Counter({"hardlink": len(rows)})

    monkeypatch.setattr(exact_labeling, "_extract_batch", fake_extract_batch)
    output = tmp_path / "labels" / "descriptors.csv"
    work_dir = tmp_path / "labels" / "exact_work"
    first = exact_labeling.build_exact_snapshot_descriptors(
        project_root=project_root,
        trajectory_manifest=manifest,
        work_dir=work_dir,
        output_path=output,
        workers=1,
        batch_size=2,
    )

    assert first["batches"] == 3
    assert first["resumed_batches"] == 0
    assert first["materialization_counts"] == {"hardlink": 5}
    assert len(calls) == 3
    assert not any((work_dir / "batches").glob("*/large-reconstructible-file.bin"))

    second = exact_labeling.build_exact_snapshot_descriptors(
        project_root=project_root,
        trajectory_manifest=manifest,
        work_dir=work_dir,
        output_path=output,
        workers=1,
        batch_size=2,
    )
    assert second["resumed_batches"] == 3
    assert len(calls) == 3

