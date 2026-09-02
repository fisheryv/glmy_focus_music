"""Download, verify, and optionally materialize the canonical HF audio release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ROOT = Path("datasets/open-focus-classical-600")
FROZEN_SHA256SUMS_SHA256 = "8b767b8d0d85fb3ef9ba5340ff6b5288d1e7681a5f37b71e2230f30c825ada20"
FROZEN_TRACKS_SHA256 = "0636ddf6cb5b4ee418829dcd24578d7abe49ac2479719d36874b3b1ae5fd2e97"
FROZEN_LICENSES_SHA256 = "4dc811a6fa31c772903cbf1178a478b56fc6f0896bb30096397426f116a9d66c"


class DatasetReleaseError(RuntimeError):
    """Raised when the downloaded release differs from the frozen contract."""


@dataclass(frozen=True, slots=True)
class ReleaseVerification:
    """Verified release metadata and the physical source path for every track."""

    dataset_root: Path
    rows: tuple[dict[str, str], ...]
    source_by_track: dict[str, Path]
    summary: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_sha256s(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise DatasetReleaseError(f"{path}:{line_number}: malformed SHA256SUMS line") from exc
        relative = relative.lstrip(" *").replace("\\", "/")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise DatasetReleaseError(f"{path}:{line_number}: invalid SHA-256")
        if relative in result:
            raise DatasetReleaseError(f"{path}:{line_number}: duplicate path {relative!r}")
        result[relative] = digest.lower()
    return result


def _materialize(source: Path, target: Path, digest: str, mode: str) -> str:
    if target.is_file():
        if sha256_file(target) != digest:
            raise DatasetReleaseError(f"existing target has the wrong hash: {target}")
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists():
        temporary.unlink()
    selected = mode
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, temporary)
            selected = "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
            selected = "copy"
    if selected == "copy":
        shutil.copy2(source, temporary)
    if sha256_file(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise DatasetReleaseError(f"materialized file has the wrong hash: {target}")
    os.replace(temporary, target)
    return selected


def _validate_project_index(project_metadata: Path, dataset_rows: list[dict[str, str]]) -> None:
    if project_metadata.is_file():
        index_path = project_metadata
    elif (project_metadata / "track_index.csv").is_file():
        index_path = project_metadata / "track_index.csv"
    else:
        index_path = project_metadata / "tracks.csv"
    if not index_path.is_file():
        raise DatasetReleaseError(f"missing project track index: {index_path}")
    project_rows = _read_csv(index_path)
    project = {row["track_id"]: row for row in project_rows}
    dataset = {row["track_id"]: row for row in dataset_rows}
    if len(project) != len(project_rows):
        raise DatasetReleaseError(f"duplicate track IDs in project index: {index_path}")
    if set(project) != set(dataset):
        raise DatasetReleaseError("project and Hugging Face track IDs differ")
    for track_id, row in dataset.items():
        local = project[track_id]
        if local["relative_path"] != row["relative_path"]:
            raise DatasetReleaseError(f"relative path mismatch for {track_id}")
        if local["sha256"].lower() != row["sha256"].lower():
            raise DatasetReleaseError(f"SHA-256 mismatch in project index for {track_id}")
        if local.get("group") and local["group"] != row["group"]:
            raise DatasetReleaseError(f"group mismatch in project index for {track_id}")


def verify_release_dataset(
    *,
    dataset_root: Path,
    project_metadata: Path | None = None,
    expected_count: int = 600,
    expected_sha256s_sha256: str | None = FROZEN_SHA256SUMS_SHA256,
    expected_tracks_sha256: str | None = FROZEN_TRACKS_SHA256,
    expected_licenses_sha256: str | None = FROZEN_LICENSES_SHA256,
    verify_audio: bool = True,
) -> ReleaseVerification:
    """Validate the immutable HF layout without copying audio into ``data_raw``."""

    dataset_root = dataset_root.resolve()
    sums_path = dataset_root / "SHA256SUMS"
    tracks_path = dataset_root / "metadata" / "tracks.csv"
    licenses_path = dataset_root / "metadata" / "licenses.csv"
    for path in (sums_path, tracks_path, licenses_path):
        if not path.is_file():
            raise DatasetReleaseError(f"release file is missing: {path}")
    frozen = (
        (sums_path, expected_sha256s_sha256),
        (tracks_path, expected_tracks_sha256),
        (licenses_path, expected_licenses_sha256),
    )
    for path, expected in frozen:
        if expected and sha256_file(path) != expected.lower():
            raise DatasetReleaseError(f"frozen release hash mismatch: {path}")

    sums = read_sha256s(sums_path)
    rows = _read_csv(tracks_path)
    required = {"file_name", "track_id", "group", "split", "relative_path", "sha256"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise DatasetReleaseError(f"tracks.csv is missing columns: {sorted(missing)}")
    if len(rows) != expected_count or len(sums) != expected_count:
        raise DatasetReleaseError(
            f"expected {expected_count} audio files, got tracks={len(rows)} sums={len(sums)}"
        )
    track_ids = [row["track_id"] for row in rows]
    published_paths = [row["file_name"].replace("\\", "/") for row in rows]
    if len(set(track_ids)) != len(track_ids):
        raise DatasetReleaseError("tracks.csv contains duplicate track IDs")
    if len(set(published_paths)) != len(published_paths):
        raise DatasetReleaseError("tracks.csv contains duplicate published paths")
    if project_metadata is not None:
        _validate_project_index(project_metadata.resolve(), rows)

    source_by_track: dict[str, Path] = {}
    for row, published_path in zip(rows, published_paths, strict=True):
        if published_path not in sums:
            raise DatasetReleaseError(f"SHA256SUMS has no entry for {published_path}")
        digest = sums[published_path]
        if row["sha256"].lower() != digest:
            raise DatasetReleaseError(f"tracks.csv hash differs for {row['track_id']}")
        relative = Path(published_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetReleaseError(f"unsafe published path for {row['track_id']}")
        expected_prefix = f"data/{row['group']}/{row['split']}/"
        if not published_path.startswith(expected_prefix):
            raise DatasetReleaseError(f"group/split path mismatch for {row['track_id']}")
        canonical = Path(row["relative_path"])
        if canonical.is_absolute() or ".." in canonical.parts:
            raise DatasetReleaseError(f"unsafe canonical path for {row['track_id']}")
        source = (dataset_root / relative).resolve()
        if not source.is_relative_to(dataset_root) or not source.is_file():
            raise DatasetReleaseError(
                f"audio file is missing or escapes dataset root: {published_path}"
            )
        if verify_audio and sha256_file(source) != digest:
            raise DatasetReleaseError(f"downloaded audio hash mismatch: {published_path}")
        source_by_track[row["track_id"]] = source

    groups = Counter(row["group"] for row in rows)
    splits = Counter(row["split"] for row in rows)
    group_splits = Counter(f"{row['group']}/{row['split']}" for row in rows)
    summary = {
        "schema_version": 2,
        "status": "verified" if verify_audio else "metadata_verified",
        "dataset_root": str(dataset_root),
        "audio_files": len(rows),
        "audio_hashes_verified": verify_audio,
        "sha256s_sha256": sha256_file(sums_path),
        "tracks_csv_sha256": sha256_file(tracks_path),
        "licenses_csv_sha256": sha256_file(licenses_path),
        "group_counts": dict(sorted(groups.items())),
        "split_counts": dict(sorted(splits.items())),
        "group_split_counts": dict(sorted(group_splits.items())),
    }
    return ReleaseVerification(
        dataset_root=dataset_root,
        rows=tuple(rows),
        source_by_track=source_by_track,
        summary=summary,
    )


def prepare_release_dataset(
    *,
    snapshot_dir: Path,
    data_root: Path,
    project_metadata: Path,
    expected_count: int = 600,
    expected_sha256s_sha256: str | None = FROZEN_SHA256SUMS_SHA256,
    expected_tracks_sha256: str | None = FROZEN_TRACKS_SHA256,
    expected_licenses_sha256: str | None = FROZEN_LICENSES_SHA256,
    materialize_mode: str = "auto",
) -> dict[str, Any]:
    """Verify all release bytes and expose them under the project's canonical paths."""

    verification = verify_release_dataset(
        dataset_root=snapshot_dir,
        project_metadata=project_metadata,
        expected_count=expected_count,
        expected_sha256s_sha256=expected_sha256s_sha256,
        expected_tracks_sha256=expected_tracks_sha256,
        expected_licenses_sha256=expected_licenses_sha256,
        verify_audio=True,
    )
    snapshot_dir = verification.dataset_root
    data_root = data_root.resolve()
    methods: dict[str, int] = {}
    for row in verification.rows:
        digest = row["sha256"].lower()
        source = verification.source_by_track[row["track_id"]]
        relative = Path(row["relative_path"])
        method = _materialize(source, data_root / relative, digest, materialize_mode)
        methods[method] = methods.get(method, 0) + 1
    return {
        **verification.summary,
        "data_root": str(data_root),
        "materialization": methods,
    }


def _download(repo_id: str, revision: str, destination: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DatasetReleaseError(
            "install the 'repro' extra to download Hugging Face data"
        ) from exc
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare-release-dataset")
    parser.add_argument("--repo-id", default="fisheryv/open-focus-classical-600")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="optional legacy materialization destination; omitted means verify in place",
    )
    parser.add_argument("--project-metadata", type=Path, default=Path("metadata"))
    parser.add_argument("--receipt", type=Path, default=Path("runs/reproducibility/dataset.json"))
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--materialize-mode", choices=("auto", "hardlink", "copy"), default="auto")
    args = parser.parse_args(argv)
    snapshot = args.snapshot_dir or _download(args.repo_id, args.revision, args.download_dir)
    if args.data_root is None:
        summary = verify_release_dataset(
            dataset_root=snapshot,
            project_metadata=args.project_metadata,
            expected_count=args.expected_count,
            verify_audio=True,
        ).summary
    else:
        summary = prepare_release_dataset(
            snapshot_dir=snapshot,
            data_root=args.data_root,
            project_metadata=args.project_metadata,
            expected_count=args.expected_count,
            materialize_mode=args.materialize_mode,
        )
    summary.update({"repo_id": args.repo_id, "revision": args.revision})
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
