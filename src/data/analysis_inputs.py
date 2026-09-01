"""Audit the immutable dataset -> preprocessing -> feature-analysis provenance chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from data.hf_release import (
    DEFAULT_DATASET_ROOT,
    FROZEN_LICENSES_SHA256,
    FROZEN_SHA256SUMS_SHA256,
    FROZEN_TRACKS_SHA256,
    DatasetReleaseError,
    sha256_file,
    verify_release_dataset,
)


class AnalysisInputError(RuntimeError):
    """Raised when analysis inputs do not derive from the frozen HF release."""


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisInputError(f"missing provenance manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]


def _unique_by(rows: list[dict[str, str]], key: str, path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            raise AnalysisInputError(f"{path} contains an empty {key}")
        if value in result:
            raise AnalysisInputError(f"{path} contains duplicate {key}={value!r}")
        result[value] = row
    return result


def audit_analysis_inputs(
    *,
    root: Path,
    dataset_root: Path | None = None,
    feature_manifest: Path = Path("metadata/feature_segments.csv"),
    preprocess_manifest: Path = Path("metadata/preprocessed_segments.csv"),
    expected_scales: tuple[float, ...] = (180.0, 300.0),
    expected_tracks: int = 600,
    verify_dataset_audio: bool = False,
    expected_sha256s_sha256: str | None = FROZEN_SHA256SUMS_SHA256,
    expected_tracks_sha256: str | None = FROZEN_TRACKS_SHA256,
    expected_licenses_sha256: str | None = FROZEN_LICENSES_SHA256,
) -> dict[str, Any]:
    """Prove that every analyzed segment descends from one frozen release track."""

    root = root.resolve()
    selected_dataset = dataset_root or Path(
        os.environ.get("FOCUS_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    )
    if not selected_dataset.is_absolute():
        selected_dataset = root / selected_dataset
    feature_path = feature_manifest if feature_manifest.is_absolute() else root / feature_manifest
    preprocess_path = (
        preprocess_manifest if preprocess_manifest.is_absolute() else root / preprocess_manifest
    )
    try:
        release = verify_release_dataset(
            dataset_root=selected_dataset,
            project_metadata=root / "metadata",
            expected_count=expected_tracks,
            verify_audio=verify_dataset_audio,
            expected_sha256s_sha256=expected_sha256s_sha256,
            expected_tracks_sha256=expected_tracks_sha256,
            expected_licenses_sha256=expected_licenses_sha256,
        )
    except DatasetReleaseError as exc:
        raise AnalysisInputError(f"HF dataset validation failed: {exc}") from exc

    feature_rows = _read_rows(feature_path)
    preprocess_rows = _read_rows(preprocess_path)
    if any(row.get("status") == "failed" for row in feature_rows):
        raise AnalysisInputError("feature manifest contains failed rows")
    if any(row.get("status") == "failed" for row in preprocess_rows):
        raise AnalysisInputError("preprocessing manifest contains failed rows")
    features = _unique_by(feature_rows, "segment_id", feature_path)
    preprocessed = _unique_by(preprocess_rows, "segment_id", preprocess_path)
    dataset = {row["track_id"]: row for row in release.rows}

    expected_segments = len(dataset) * len(expected_scales)
    if len(features) != expected_segments or len(preprocessed) != expected_segments:
        raise AnalysisInputError(
            "canonical manifest size mismatch: "
            f"dataset={len(dataset)}, preprocessed={len(preprocessed)}, features={len(features)}"
        )
    if set(features) != set(preprocessed):
        raise AnalysisInputError("feature and preprocessing segment IDs differ")
    if {row["track_id"] for row in feature_rows} != set(dataset):
        raise AnalysisInputError("feature and HF dataset track IDs differ")

    seen_track_scales: Counter[tuple[str, float]] = Counter()
    for segment_id, feature in features.items():
        preprocessed_row = preprocessed[segment_id]
        track_id = feature["track_id"]
        dataset_row = dataset.get(track_id)
        if dataset_row is None:
            raise AnalysisInputError(f"unknown HF track in feature manifest: {track_id}")
        scale = float(feature["scale_seconds"])
        seen_track_scales[(track_id, scale)] += 1
        for field in ("track_id", "group", "split", "scale_seconds"):
            if feature[field] != preprocessed_row[field]:
                raise AnalysisInputError(f"{segment_id}: feature/preprocessing {field} mismatch")
        if feature["input_relative_path"] != preprocessed_row["output_relative_path"]:
            raise AnalysisInputError(
                f"{segment_id}: preprocessing output path is not feature input"
            )
        if feature["input_sha256"].lower() != preprocessed_row["sha256"].lower():
            raise AnalysisInputError(f"{segment_id}: preprocessing/feature SHA-256 mismatch")
        if preprocessed_row["source_relative_path"] != dataset_row["relative_path"]:
            raise AnalysisInputError(f"{segment_id}: HF source path mismatch")
        if feature["group"] != dataset_row["group"] or feature["split"] != dataset_row["split"]:
            raise AnalysisInputError(f"{segment_id}: HF group/split mismatch")

    expected_scale_set = set(expected_scales)
    for track_id in dataset:
        actual = {
            scale
            for candidate_track, scale in seen_track_scales
            if candidate_track == track_id
        }
        if actual != expected_scale_set:
            raise AnalysisInputError(
                f"{track_id}: expected scales {sorted(expected_scale_set)}, got {sorted(actual)}"
            )
        if any(seen_track_scales[(track_id, scale)] != 1 for scale in expected_scale_set):
            raise AnalysisInputError(f"{track_id}: duplicate track/scale feature row")

    feature_sha256 = sha256_file(feature_path)
    preprocess_sha256 = sha256_file(preprocess_path)
    identity = {
        key: release.summary[key]
        for key in (
            "audio_files",
            "sha256s_sha256",
            "tracks_csv_sha256",
            "licenses_csv_sha256",
            "group_counts",
            "split_counts",
            "group_split_counts",
        )
    }
    audit_payload = {
        "dataset_identity": identity,
        "preprocess_manifest_sha256": preprocess_sha256,
        "feature_manifest_sha256": feature_sha256,
        "segments": len(features),
        "tracks": len(dataset),
        "group_counts": dict(sorted(Counter(row["group"] for row in feature_rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in feature_rows).items())),
        "scale_counts": {
            str(key): value
            for key, value in sorted(
                Counter(float(row["scale_seconds"]) for row in feature_rows).items()
            )
        },
    }
    encoded = json.dumps(audit_payload, sort_keys=True, separators=(",", ":"))
    audit_payload["provenance_chain_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return audit_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit-analysis-inputs")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("metadata/feature_segments.csv"),
    )
    parser.add_argument(
        "--preprocess-manifest",
        type=Path,
        default=Path("metadata/preprocessed_segments.csv"),
    )
    parser.add_argument("--verify-audio", action="store_true")
    args = parser.parse_args(argv)
    summary = audit_analysis_inputs(
        root=args.root,
        dataset_root=args.dataset_root,
        feature_manifest=args.feature_manifest,
        preprocess_manifest=args.preprocess_manifest,
        verify_dataset_audio=args.verify_audio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
