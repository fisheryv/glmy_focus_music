"""Download and materialize the canonical Hugging Face audio release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


class DatasetReleaseError(RuntimeError):
    """Raised when the downloaded release differs from the frozen contract."""


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
    index_path = project_metadata / "track_index.csv"
    if not index_path.is_file():
        raise DatasetReleaseError(f"missing project track index: {index_path}")
    project = {row["track_id"]: row for row in _read_csv(index_path)}
    dataset = {row["track_id"]: row for row in dataset_rows}
    if set(project) != set(dataset):
        raise DatasetReleaseError("project and Hugging Face track IDs differ")
    for track_id, row in dataset.items():
        local = project[track_id]
        if local["relative_path"] != row["relative_path"]:
            raise DatasetReleaseError(f"relative path mismatch for {track_id}")
        if local["sha256"].lower() != row["sha256"].lower():
            raise DatasetReleaseError(f"SHA-256 mismatch in project index for {track_id}")


def prepare_release_dataset(
    *,
    snapshot_dir: Path,
    data_root: Path,
    project_metadata: Path,
    expected_count: int = 600,
    expected_sha256s_sha256: str | None = None,
    expected_tracks_sha256: str | None = None,
    expected_licenses_sha256: str | None = None,
    materialize_mode: str = "auto",
) -> dict[str, Any]:
    """Verify all release bytes and expose them under the project's canonical paths."""

    snapshot_dir = snapshot_dir.resolve()
    data_root = data_root.resolve()
    sums_path = snapshot_dir / "SHA256SUMS"
    tracks_path = snapshot_dir / "metadata" / "tracks.csv"
    licenses_path = snapshot_dir / "metadata" / "licenses.csv"
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
    if len(rows) != expected_count or len(sums) != expected_count:
        raise DatasetReleaseError(
            f"expected {expected_count} audio files, got tracks={len(rows)} sums={len(sums)}"
        )
    _validate_project_index(project_metadata.resolve(), rows)
    methods: dict[str, int] = {}
    for row in rows:
        published_path = row["file_name"].replace("\\", "/")
        if published_path not in sums:
            raise DatasetReleaseError(f"SHA256SUMS has no entry for {published_path}")
        digest = sums[published_path]
        if row["sha256"].lower() != digest:
            raise DatasetReleaseError(f"tracks.csv hash differs for {row['track_id']}")
        source = (snapshot_dir / published_path).resolve()
        if not source.is_relative_to(snapshot_dir) or not source.is_file():
            raise DatasetReleaseError(
                f"audio file is missing or escapes snapshot: {published_path}"
            )
        if sha256_file(source) != digest:
            raise DatasetReleaseError(f"downloaded audio hash mismatch: {published_path}")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetReleaseError(f"unsafe canonical path for {row['track_id']}")
        method = _materialize(source, data_root / relative, digest, materialize_mode)
        methods[method] = methods.get(method, 0) + 1
    return {
        "schema_version": 1,
        "status": "verified",
        "snapshot_dir": str(snapshot_dir),
        "data_root": str(data_root),
        "audio_files": len(rows),
        "sha256s_sha256": sha256_file(sums_path),
        "tracks_csv_sha256": sha256_file(tracks_path),
        "licenses_csv_sha256": sha256_file(licenses_path),
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
    parser.add_argument(
        "--download-dir", type=Path, default=Path("datasets/open-focus-classical-600")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--project-metadata", type=Path, default=Path("metadata"))
    parser.add_argument("--receipt", type=Path, default=Path("runs/reproducibility/dataset.json"))
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--materialize-mode", choices=("auto", "hardlink", "copy"), default="auto")
    args = parser.parse_args(argv)
    snapshot = args.snapshot_dir or _download(args.repo_id, args.revision, args.download_dir)
    summary = prepare_release_dataset(
        snapshot_dir=snapshot,
        data_root=args.data_root,
        project_metadata=args.project_metadata,
        expected_count=args.expected_count,
        expected_sha256s_sha256="8b767b8d0d85fb3ef9ba5340ff6b5288d1e7681a5f37b71e2230f30c825ada20",
        expected_tracks_sha256="0636ddf6cb5b4ee418829dcd24578d7abe49ac2479719d36874b3b1ae5fd2e97",
        expected_licenses_sha256="4dc811a6fa31c772903cbf1178a478b56fc6f0896bb30096397426f116a9d66c",
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
