from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from data.analysis_inputs import AnalysisInputError, audit_analysis_inputs


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset" / "open-focus-classical-600"
    audio = dataset / "data" / "focus" / "validation" / "one.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"source")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    release_row = {
        "file_name": "data/focus/validation/one.mp3",
        "track_id": "one",
        "group": "focus",
        "split": "validation",
        "relative_path": "focus_music/one.mp3",
        "sha256": digest,
    }
    _write_csv(dataset / "metadata" / "tracks.csv", [release_row])
    _write_csv(dataset / "metadata" / "licenses.csv", [{"track_id": "one"}])
    (dataset / "SHA256SUMS").write_text(
        f"{digest}  data/focus/validation/one.mp3\n", encoding="utf-8"
    )
    _write_csv(
        root / "metadata" / "track_index.csv",
        [
            {
                "track_id": "one",
                "group": "focus",
                "relative_path": "focus_music/one.mp3",
                "sha256": digest,
            }
        ],
    )
    preprocessed = []
    features = []
    for scale in (180.0, 300.0):
        segment_id = f"one__{int(scale)}s"
        output = f"features/audio/{int(scale)}s/focus/validation/{segment_id}.wav"
        output_hash = hashlib.sha256(segment_id.encode()).hexdigest()
        preprocessed.append(
            {
                "segment_id": segment_id,
                "track_id": "one",
                "group": "focus",
                "split": "validation",
                "scale_seconds": str(scale),
                "source_relative_path": "focus_music/one.mp3",
                "output_relative_path": output,
                "sha256": output_hash,
                "status": "verified",
            }
        )
        features.append(
            {
                "segment_id": segment_id,
                "track_id": "one",
                "group": "focus",
                "split": "validation",
                "scale_seconds": str(scale),
                "input_relative_path": output,
                "input_sha256": output_hash,
                "status": "transformed",
            }
        )
    preprocess_path = root / "metadata" / "preprocessed_segments.csv"
    feature_path = root / "metadata" / "feature_segments.csv"
    _write_csv(preprocess_path, preprocessed)
    _write_csv(feature_path, features)
    return dataset, feature_path


def _audit(root: Path, dataset: Path) -> dict[str, object]:
    return audit_analysis_inputs(
        root=root,
        dataset_root=dataset,
        expected_tracks=1,
        expected_sha256s_sha256=None,
        expected_tracks_sha256=None,
        expected_licenses_sha256=None,
    )


def test_analysis_audit_binds_release_preprocessing_and_features(tmp_path: Path) -> None:
    dataset, _ = _fixture(tmp_path)

    result = _audit(tmp_path, dataset)

    assert result["tracks"] == 1
    assert result["segments"] == 2
    assert len(str(result["provenance_chain_sha256"])) == 64


def test_analysis_audit_rejects_feature_hash_not_from_preprocessing(tmp_path: Path) -> None:
    dataset, feature_path = _fixture(tmp_path)
    rows = list(csv.DictReader(feature_path.open(encoding="utf-8")))
    rows[0]["input_sha256"] = "0" * 64
    _write_csv(feature_path, rows)

    with pytest.raises(AnalysisInputError, match="SHA-256 mismatch"):
        _audit(tmp_path, dataset)
