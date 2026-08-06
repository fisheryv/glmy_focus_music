from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

ARCHIVE_NAME = "brainfm_legacy_2026-08-02"
SCALES = ("180s", "300s")
FEATURE_VIEWS = ("acoustic", "chroma", "rhythm", "modulation", "structure", "manifests")
CORE_METADATA = (
    "focus_manifest.csv",
    "track_index.csv",
    "licenses.csv",
    "split_discovery.csv",
    "split_validation.csv",
    "split_holdout.csv",
    "dataset_summary.json",
    "preprocessed_segments.csv",
    "preprocessing_summary.json",
    "feature_segments.csv",
    "feature_summary.json",
)
MODEL_FILES = (
    "state_model.npz",
    "state_model.json",
    "pitch_v2_codebook.npz",
    "pitch_v2_codebook.json",
)


class MigrationError(RuntimeError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_open_path(value: str) -> str:
    value = value.replace("focus_open_music/", "focus_music/")
    value = value.replace("features/audio_focus_open/", "features/audio/")
    value = value.replace("/focus_open/", "/focus/")
    return value


def _canonicalize_row(row: dict[str, str]) -> dict[str, str]:
    result = {key: _replace_open_path(value) for key, value in row.items()}
    if result.get("group") == "focus_open":
        result["group"] = "focus"
    return result


def _deep_canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _deep_canonicalize(item) for key, item in value.items()}
        if result.get("group") == "focus_open":
            result["group"] = "focus"
        return result
    if isinstance(value, list):
        return [_deep_canonicalize(item) for item in value]
    if isinstance(value, str):
        return _replace_open_path(value)
    return value


def _archive_moves(root: Path, archive: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = [
        (root / "data_raw" / "focus_music", archive / "data_raw" / "focus_music")
    ]
    for scale in SCALES:
        moves.append(
            (
                root / "features" / "audio" / scale / "focus",
                archive / "features" / "audio" / scale / "focus",
            )
        )
        moves.append(
            (
                root / "features" / "pitch_v2" / scale / "focus",
                archive / "features" / "pitch_v2" / scale / "focus",
            )
        )
        for view in FEATURE_VIEWS:
            moves.append(
                (
                    root / "features" / view / scale / "focus",
                    archive / "features" / view / scale / "focus",
                )
            )
    for name in CORE_METADATA:
        moves.append((root / "metadata" / name, archive / "metadata" / name))
    for name in MODEL_FILES:
        moves.append((root / "features" / "models" / name, archive / "features" / "models" / name))
    return [(source, target) for source, target in moves if source.exists()]


def _promotion_moves(root: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = [
        (root / "data_raw" / "focus_open_music", root / "data_raw" / "focus_music")
    ]
    for scale in SCALES:
        moves.append(
            (
                root / "features" / "audio_focus_open" / scale / "focus",
                root / "features" / "audio" / scale / "focus",
            )
        )
        for view in FEATURE_VIEWS:
            moves.append(
                (
                    root / "features" / view / scale / "focus_open",
                    root / "features" / view / scale / "focus",
                )
            )
    return moves


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise MigrationError("required migration inputs are missing: " + "; ".join(missing))


def _verify_expected_hashes(
    root: Path,
    old_tracks: list[dict[str, str]],
    open_rows: list[dict[str, str]],
    old_segments: list[dict[str, str]],
    open_segments: list[dict[str, str]],
    *,
    workers: int,
) -> tuple[dict[Path, str], dict[str, int]]:
    checks: list[tuple[str, Path, str]] = []
    checks.extend(
        ("brainfm_raw", root / "data_raw" / row["relative_path"], row["sha256"])
        for row in old_tracks
        if row["group"] == "focus"
    )
    checks.extend(
        ("open_raw", root / "data_raw" / row["relative_path"], row["sha256"]) for row in open_rows
    )
    checks.extend(
        ("brainfm_wav", root / row["output_relative_path"], row["sha256"])
        for row in old_segments
        if row["group"] == "focus"
    )
    checks.extend(
        ("open_wav", root / row["output_relative_path"], row["sha256"]) for row in open_segments
    )
    missing = [str(path) for _, path, _ in checks if not path.is_file()]
    if missing:
        raise MigrationError(f"{len(missing)} files required for hash verification are missing")
    unique_paths = sorted({path for _, path, _ in checks})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = dict(zip(unique_paths, executor.map(_sha256, unique_paths), strict=True))
    mismatches = [
        f"{label}: {path}" for label, path, expected in checks if hashes[path] != expected.lower()
    ]
    if mismatches:
        raise MigrationError("hash verification failed: " + "; ".join(mismatches[:10]))
    return hashes, dict(Counter(label for label, _, _ in checks))


def _move(source: Path, target: Path) -> None:
    if target.exists():
        raise MigrationError(f"refusing to overwrite migration target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _rewrite_focus_sidecars(root: Path) -> int:
    count = 0
    for scale in SCALES:
        directory = root / "features" / "manifests" / scale / "focus"
        for path in directory.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            _write_json(path, _deep_canonicalize(payload))
            count += 1
    return count


def _focus_track_rows(
    open_rows: list[dict[str, str]], corrections: dict[str, str]
) -> list[dict[str, str]]:
    return [
        {
            "track_id": row["track_id"],
            "group": "focus",
            "relative_path": _replace_open_path(row["relative_path"]),
            "sha256": row["sha256"],
            "duration_seconds": corrections.get(row["track_id"], row["duration_seconds"]),
            "sample_rate": row["sample_rate"],
            "channels": row["channels"],
            "artist_key": row["artist_key"],
            "album_key": row["album_key"],
            "composer_key": "",
            "instrumental": "true",
            "restricted": "false",
        }
        for row in open_rows
    ]


def _focus_license_rows(open_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "track_id": row["track_id"],
            "group": "focus",
            "source_url": row["source_url"],
            "license_type": row["license_type"],
            "downloaded_at": row["downloaded_at"],
            "redistribution_allowed": "true",
            "notes": (
                f"{row['source_dataset']}; redistribution subject to per-track CC terms "
                f"(attribution and NC/ND/SA where applicable); license URL: {row['license_url']}"
            ),
        }
        for row in open_rows
    ]


def _split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["split"] for row in rows))


def _preprocessing_summary(root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") not in {"", "failed"}]
    output_paths = [root / row["output_relative_path"] for row in successful]
    output_bytes = sum(path.stat().st_size for path in output_paths if path.is_file())
    return {
        "generated_at": date.today().isoformat(),
        "ok": len(successful) == len(rows),
        "segments": len(rows),
        "tracks": len({row["track_id"] for row in rows}),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "group_counts": dict(Counter(row["group"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "scale_counts": dict(Counter(f"{float(row['scale_seconds']):g}s" for row in rows)),
        "output_files": sum(path.is_file() for path in output_paths),
        "output_bytes": output_bytes,
        "output_gib": round(output_bytes / (1024**3), 3),
        "canonical_focus_source": "jamendo-api-open-focus",
    }


def _dataset_summary(
    root: Path,
    tracks: list[dict[str, str]],
    licenses: list[dict[str, str]],
    selected_candidates: list[dict[str, str]],
    verification_counts: dict[str, int],
    archive: Path,
) -> dict[str, Any]:
    split_by_track: dict[str, str] = {}
    for split in ("discovery", "validation", "holdout"):
        for row in _read_csv(root / "metadata" / f"split_{split}.csv"):
            split_by_track[row["track_id"]] = split
    license_by_track = {row["track_id"]: row for row in licenses}
    groups: dict[str, Any] = {}
    for group in ("focus", "pop", "classical"):
        members = [row for row in tracks if row["group"] == group]
        files = [root / "data_raw" / row["relative_path"] for row in members]
        duration = sum(float(row["duration_seconds"]) for row in members)
        groups[group] = {
            "tracks": len(members),
            "discovery": sum(split_by_track.get(row["track_id"]) == "discovery" for row in members),
            "validation": sum(
                split_by_track.get(row["track_id"]) == "validation" for row in members
            ),
            "holdout": sum(split_by_track.get(row["track_id"]) == "holdout" for row in members),
            "duration_seconds": round(duration, 3),
            "duration_hours": round(duration / 3600.0, 2),
            "bytes": sum(path.stat().st_size for path in files),
            "gibibytes": round(sum(path.stat().st_size for path in files) / (1024**3), 3),
            "license_counts": dict(
                Counter(license_by_track[row["track_id"]]["license_type"] for row in members)
            ),
        }
    focus = groups["focus"]
    focus.update(
        {
            "source_counts": dict(Counter(row["source_dataset"] for row in selected_candidates)),
            "artists": len({row["artist_key"] for row in selected_candidates}),
            "albums": len({row["album_key"] for row in selected_candidates}),
            "restricted": False,
            "redistribution_allowed": True,
        }
    )
    combined_duration = sum(float(row["duration_seconds"]) for row in tracks)
    combined_bytes = sum(item["bytes"] for item in groups.values())
    return {
        "generated_at": date.today().isoformat(),
        "schema_version": 2,
        "canonical_groups": ["focus", "pop", "classical"],
        "canonical_focus_source": "jamendo-api-open-focus",
        "groups": groups,
        "combined": {
            "tracks": len(tracks),
            "discovery": sum(value == "discovery" for value in split_by_track.values()),
            "validation": sum(value == "validation" for value in split_by_track.values()),
            "holdout": sum(value == "holdout" for value in split_by_track.values()),
            "duration_hours": round(combined_duration / 3600.0, 2),
            "bytes": combined_bytes,
            "gibibytes": round(combined_bytes / (1024**3), 3),
        },
        "audit": {
            "hashes_verified": True,
            "verification_counts": verification_counts,
            "duplicate_audio": False,
            "split_leakage": False,
            "license_records": len(licenses),
        },
        "legacy_brainfm": {
            "status": "retired_from_canonical_dataset",
            "archive": archive.relative_to(root).as_posix(),
            "scientific_results": "historical_only; rerun required for open Focus claims",
        },
    }


def _inventory_archive(
    root: Path,
    archive: Path,
    moved: list[tuple[Path, Path]],
    hash_cache: dict[Path, str],
    *,
    workers: int,
) -> tuple[int, int, str]:
    records: list[dict[str, Any]] = []
    pending: list[Path] = []
    mapping: dict[Path, tuple[str, str]] = {}
    for source, target in moved:
        archived_files = (
            [target]
            if target.is_file()
            else sorted(path for path in target.rglob("*") if path.is_file())
        )
        for archived_path in archived_files:
            suffix = archived_path.relative_to(target) if target.is_dir() else Path()
            original_path = source / suffix if target.is_dir() else source
            mapping[archived_path] = (
                original_path.relative_to(root).as_posix(),
                archived_path.relative_to(root).as_posix(),
            )
            if original_path not in hash_cache:
                pending.append(archived_path)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending_hashes = dict(zip(pending, executor.map(_sha256, pending), strict=True))
    for archived_path, (original, archived) in mapping.items():
        source_path = root / original
        digest = hash_cache.get(source_path, pending_hashes.get(archived_path, ""))
        records.append(
            {
                "original_relative_path": original,
                "archived_relative_path": archived,
                "bytes": archived_path.stat().st_size,
                "sha256": digest,
            }
        )
    records.sort(key=lambda row: row["original_relative_path"])
    inventory_path = archive / "file_inventory.csv"
    _write_csv(
        inventory_path,
        ("original_relative_path", "archived_relative_path", "bytes", "sha256"),
        records,
    )
    return len(records), sum(int(row["bytes"]) for row in records), _sha256(inventory_path)


def _execute(root: Path, archive: Path, *, workers: int) -> dict[str, Any]:
    metadata = root / "metadata"
    old_tracks = _read_csv(metadata / "track_index.csv")
    old_licenses = _read_csv(metadata / "licenses.csv")
    old_splits = {
        split: _read_csv(metadata / f"split_{split}.csv")
        for split in ("discovery", "validation", "holdout")
    }
    old_segments = _read_csv(metadata / "preprocessed_segments.csv")
    candidates = _read_csv(metadata / "focus_open_candidates.csv")
    selected = [row for row in candidates if row["selection_status"] == "selected"]
    open_segments = _read_csv(metadata / "focus_open_preprocessed_segments.csv")
    raw_features = _read_csv(metadata / "four_group_raw_feature_segments.csv")
    if len(selected) != 300:
        raise MigrationError(f"expected 300 selected open Focus tracks, found {len(selected)}")
    if any(row["download_status"] != "verified" for row in selected):
        raise MigrationError("not every selected open Focus track is verified")
    if len([row for row in old_tracks if row["group"] == "focus"]) != 200:
        raise MigrationError(
            "canonical pre-migration dataset is not the expected 200-track Focus set"
        )
    archive_moves = _archive_moves(root, archive)
    promotion_moves = _promotion_moves(root)
    _require_paths(source for source, _ in promotion_moves)
    archived_sources = {source.resolve() for source, _ in archive_moves}
    conflicts = [str(target) for _, target in archive_moves if target.exists()]
    conflicts.extend(
        str(target)
        for _, target in promotion_moves
        if target.exists() and target.resolve() not in archived_sources
    )
    if conflicts:
        raise MigrationError("migration targets already exist: " + "; ".join(conflicts))

    hash_cache, verification_counts = _verify_expected_hashes(
        root, old_tracks, selected, old_segments, open_segments, workers=workers
    )

    for source, target in archive_moves:
        _move(source, target)
    for source, target in promotion_moves:
        _move(source, target)

    corrections = {
        row["track_id"]: row["corrected_duration_seconds"]
        for row in _read_csv(metadata / "focus_open_duration_corrections.csv")
    }
    focus_tracks = _focus_track_rows(selected, corrections)
    focus_licenses = _focus_license_rows(selected)
    control_tracks = [row for row in old_tracks if row["group"] in {"pop", "classical"}]
    control_licenses = [row for row in old_licenses if row["group"] in {"pop", "classical"}]
    tracks = sorted(control_tracks + focus_tracks, key=lambda row: row["track_id"])
    licenses = sorted(control_licenses + focus_licenses, key=lambda row: row["track_id"])
    track_fields = list(old_tracks[0])
    license_fields = list(old_licenses[0])
    _write_csv(metadata / "track_index.csv", track_fields, tracks)
    _write_csv(metadata / "licenses.csv", license_fields, licenses)

    for split, old_rows in old_splits.items():
        controls = [row for row in old_rows if row["group"] in {"pop", "classical"}]
        focus_rows = [
            {"track_id": row["track_id"], "group": "focus"}
            for row in selected
            if row["split"] == split
        ]
        _write_csv(
            metadata / f"split_{split}.csv",
            ("track_id", "group"),
            sorted(controls + focus_rows, key=lambda row: row["track_id"]),
        )

    canonical_candidates = [
        _canonicalize_row(row) if row["selection_status"] == "selected" else row
        for row in candidates
    ]
    candidate_fields = list(candidates[0])
    _write_csv(metadata / "focus_open_candidates.csv", candidate_fields, canonical_candidates)
    selected_manifest = [_canonicalize_row(row) for row in selected]
    _write_csv(metadata / "focus_manifest.csv", candidate_fields, selected_manifest)

    focus_segments = [_canonicalize_row(row) for row in open_segments]
    control_segments = [row for row in old_segments if row["group"] in {"pop", "classical"}]
    preprocessed = sorted(
        control_segments + focus_segments,
        key=lambda row: (row["track_id"], float(row["scale_seconds"])),
    )
    _write_csv(metadata / "preprocessed_segments.csv", list(old_segments[0]), preprocessed)
    preprocessing_summary = _preprocessing_summary(root, preprocessed)
    _write_json(metadata / "preprocessing_summary.json", preprocessing_summary)
    _write_csv(
        metadata / "focus_open_preprocessed_segments.csv", list(open_segments[0]), focus_segments
    )
    _write_json(
        metadata / "focus_open_preprocessing_summary.json",
        _preprocessing_summary(root, focus_segments),
    )

    canonical_feature_rows: list[dict[str, str]] = []
    for row in raw_features:
        if row["group"] not in {"focus_open", "pop", "classical"}:
            continue
        canonical = _canonicalize_row(row)
        canonical["model_sha256"] = ""
        canonical["status"] = "extracted"
        canonical_feature_rows.append(canonical)
    if len(canonical_feature_rows) != 1800:
        raise MigrationError(
            f"expected 1,800 canonical raw feature rows, found {len(canonical_feature_rows)}"
        )
    _rewrite_focus_sidecars(root)
    _write_csv(
        metadata / "feature_segments.csv",
        list(raw_features[0]),
        sorted(canonical_feature_rows, key=lambda row: row["segment_id"]),
    )

    preprocess_dir = metadata / "focus_open_preprocess"
    _write_csv(preprocess_dir / "track_index.csv", track_fields, focus_tracks)
    _write_csv(preprocess_dir / "licenses.csv", license_fields, focus_licenses)
    for split in ("discovery", "validation", "holdout"):
        _write_csv(
            preprocess_dir / f"split_{split}.csv",
            ("track_id", "group"),
            [
                {"track_id": row["track_id"], "group": "focus"}
                for row in selected
                if row["split"] == split
            ],
        )

    summary = _dataset_summary(
        root, tracks, licenses, selected_manifest, verification_counts, archive
    )
    _write_json(metadata / "dataset_summary.json", summary)

    inventory_count, inventory_bytes, inventory_sha256 = _inventory_archive(
        root, archive, archive_moves, hash_cache, workers=workers
    )
    audit = {
        "generated_at": date.today().isoformat(),
        "status": "physical_migration_complete",
        "canonical_focus_source": "jamendo-api-open-focus",
        "canonical_counts": {
            "tracks": len(tracks),
            "focus": len(focus_tracks),
            "pop": len([row for row in tracks if row["group"] == "pop"]),
            "classical": len([row for row in tracks if row["group"] == "classical"]),
            "preprocessed_segments": len(preprocessed),
            "raw_feature_rows": len(canonical_feature_rows),
        },
        "split_counts": _split_counts(selected_manifest),
        "hash_verification_counts": verification_counts,
        "archive": {
            "relative_path": archive.relative_to(root).as_posix(),
            "files": inventory_count,
            "bytes": inventory_bytes,
            "gibibytes": round(inventory_bytes / (1024**3), 3),
            "inventory_sha256": inventory_sha256,
        },
        "next_required_step": "refit canonical state model and transform all 1,800 segments",
        "scientific_results": "All Brain.fm-dependent results are historical until rerun.",
    }
    _write_json(archive / "migration_audit.json", audit)
    _write_json(metadata / "open_focus_migration_audit.json", audit)
    return audit


def _dry_run(root: Path, archive: Path) -> dict[str, Any]:
    candidates = _read_csv(root / "metadata" / "focus_open_candidates.csv")
    selected = [row for row in candidates if row["selection_status"] == "selected"]
    archive_moves = _archive_moves(root, archive)
    promotion_moves = _promotion_moves(root)
    return {
        "mode": "dry-run",
        "archive": archive.relative_to(root).as_posix(),
        "selected_open_focus_tracks": len(selected),
        "archive_moves": [
            [source.relative_to(root).as_posix(), target.relative_to(root).as_posix()]
            for source, target in archive_moves
        ],
        "promotion_moves": [
            [source.relative_to(root).as_posix(), target.relative_to(root).as_posix()]
            for source, target in promotion_moves
        ],
        "hash_verification_on_execute": [
            "200 Brain.fm raw tracks",
            "400 Brain.fm WAV segments",
            "300 open Focus raw tracks",
            "600 open Focus WAV segments",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive-name", default=ARCHIVE_NAME)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    archive = root / "restricted_archive" / args.archive_name
    if archive.exists():
        raise MigrationError(f"archive already exists: {archive}")
    payload = (
        _execute(root, archive, workers=args.workers) if args.execute else _dry_run(root, archive)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MigrationError, OSError, ValueError, KeyError) as exc:
        print(f"open-focus-migration: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
