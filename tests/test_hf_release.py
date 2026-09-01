from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from data.hf_release import DatasetReleaseError, prepare_release_dataset, verify_release_dataset


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_release_dataset_verifies_and_materializes(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    audio = snapshot / "data" / "focus" / "discovery" / "one.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audited audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    row = {
        "file_name": "data/focus/discovery/one.mp3",
        "track_id": "one",
        "group": "focus",
        "split": "discovery",
        "relative_path": "focus_music/one.mp3",
        "sha256": digest,
    }
    _write_csv(snapshot / "metadata" / "tracks.csv", [row])
    _write_csv(snapshot / "metadata" / "licenses.csv", [{"track_id": "one"}])
    (snapshot / "SHA256SUMS").write_text(
        f"{digest}  data/focus/discovery/one.mp3\n", encoding="utf-8"
    )
    project_metadata = tmp_path / "metadata"
    _write_csv(
        project_metadata / "track_index.csv",
        [
            {
                "track_id": "one",
                "group": "focus",
                "relative_path": "focus_music/one.mp3",
                "sha256": digest,
            }
        ],
    )

    result = prepare_release_dataset(
        snapshot_dir=snapshot,
        data_root=tmp_path / "data_raw",
        project_metadata=project_metadata,
        expected_count=1,
        expected_sha256s_sha256=None,
        expected_tracks_sha256=None,
        expected_licenses_sha256=None,
        materialize_mode="copy",
    )

    assert result["status"] == "verified"
    assert (tmp_path / "data_raw" / "focus_music" / "one.mp3").read_bytes() == b"audited audio"


def test_prepare_release_dataset_rejects_tampered_audio(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    audio = snapshot / "data" / "focus" / "discovery" / "one.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"tampered")
    expected = hashlib.sha256(b"expected").hexdigest()
    row = {
        "file_name": "data/focus/discovery/one.mp3",
        "track_id": "one",
        "group": "focus",
        "split": "discovery",
        "relative_path": "focus_music/one.mp3",
        "sha256": expected,
    }
    _write_csv(snapshot / "metadata" / "tracks.csv", [row])
    _write_csv(snapshot / "metadata" / "licenses.csv", [{"track_id": "one"}])
    (snapshot / "SHA256SUMS").write_text(
        f"{expected}  data/focus/discovery/one.mp3\n", encoding="utf-8"
    )
    project_metadata = tmp_path / "metadata"
    _write_csv(
        project_metadata / "track_index.csv",
        [
            {
                "track_id": "one",
                "group": "focus",
                "relative_path": "focus_music/one.mp3",
                "sha256": expected,
            }
        ],
    )

    with pytest.raises(DatasetReleaseError, match="downloaded audio hash mismatch"):
        prepare_release_dataset(
            snapshot_dir=snapshot,
            data_root=tmp_path / "data_raw",
            project_metadata=project_metadata,
            expected_count=1,
            expected_sha256s_sha256=None,
            expected_tracks_sha256=None,
            expected_licenses_sha256=None,
            materialize_mode="copy",
        )


def test_verify_release_dataset_returns_direct_source_map(tmp_path: Path) -> None:
    snapshot = tmp_path / "dataset"
    audio = snapshot / "data" / "classical" / "holdout" / "one.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"original audio")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    row = {
        "file_name": "data/classical/holdout/one.wav",
        "track_id": "one",
        "group": "classical",
        "split": "holdout",
        "relative_path": "classical_music/one.wav",
        "sha256": digest,
    }
    _write_csv(snapshot / "metadata" / "tracks.csv", [row])
    _write_csv(snapshot / "metadata" / "licenses.csv", [{"track_id": "one"}])
    (snapshot / "SHA256SUMS").write_text(
        f"{digest}  data/classical/holdout/one.wav\n", encoding="utf-8"
    )

    result = verify_release_dataset(
        dataset_root=snapshot,
        expected_count=1,
        expected_sha256s_sha256=None,
        expected_tracks_sha256=None,
        expected_licenses_sha256=None,
    )

    assert result.source_by_track == {"one": audio.resolve()}
    assert result.summary["group_split_counts"] == {"classical/holdout": 1}
