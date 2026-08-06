from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ARCHIVE_NAME = "pop_music_legacy_2026-08-02"
FILTERED_METADATA = (
    "track_index.csv",
    "licenses.csv",
    "split_discovery.csv",
    "split_validation.csv",
    "split_holdout.csv",
    "duration_corrections.csv",
    "preprocessed_segments.csv",
    "feature_segments.csv",
)
DIRECT_METADATA = ("control_pop.csv", "control_pop_exclusions.csv")
CANONICAL_FEATURE_VIEWS = (
    "audio",
    "acoustic",
    "chroma",
    "modulation",
    "pitch_v2",
    "rhythm",
    "structure",
    "manifests",
)
FOUR_GROUP_VIEWS = ("pitch_v2", "rhythm", "structure")


class ArchiveError(RuntimeError):
    pass


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ArchiveError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_plan(root: Path, archive: Path) -> list[tuple[Path, Path]]:
    moves = [
        (root / "data_raw" / "pop_music", archive / "data_raw" / "pop_music")
    ]
    for view in CANONICAL_FEATURE_VIEWS:
        for scale in ("180s", "300s"):
            moves.append(
                (
                    root / "features" / view / scale / "pop",
                    archive / "features" / view / scale / "pop",
                )
            )
    for view in FOUR_GROUP_VIEWS:
        for scale in ("180s", "300s"):
            moves.append(
                (
                    root / "features" / "four_group" / view / scale / "pop",
                    archive / "features" / "four_group" / view / scale / "pop",
                )
            )
    for name in ("state_model.json", "state_model.npz"):
        moves.append(
            (root / "features" / "models" / name, archive / "features" / "models" / name)
        )
    for name in DIRECT_METADATA:
        moves.append((root / "metadata" / name, archive / "metadata" / name))
    return moves


def _validate_inputs(root: Path, archive: Path) -> dict[str, Any]:
    if archive.exists():
        raise ArchiveError(f"archive target already exists: {archive}")
    if root not in archive.parents:
        raise ArchiveError("archive target must remain inside the project root")

    missing = [str(source) for source, _ in _move_plan(root, archive) if not source.exists()]
    missing.extend(
        str(root / "metadata" / name)
        for name in FILTERED_METADATA
        if not (root / "metadata" / name).is_file()
    )
    if missing:
        raise ArchiveError("required inputs are missing: " + "; ".join(missing))

    fields, tracks = _read_csv(root / "metadata" / "track_index.csv")
    del fields
    counts = Counter(row.get("group", "") for row in tracks)
    if counts != {"focus": 300, "pop": 300, "classical": 300}:
        raise ArchiveError(f"unexpected canonical track counts: {dict(counts)}")

    _, segments = _read_csv(root / "metadata" / "preprocessed_segments.csv")
    segment_counts = Counter(row.get("group", "") for row in segments)
    if segment_counts != {"focus": 600, "pop": 600, "classical": 600}:
        raise ArchiveError(f"unexpected preprocessed segment counts: {dict(segment_counts)}")
    return {"track_counts": dict(counts), "segment_counts": dict(segment_counts)}


def _filtered_metadata(
    root: Path, archive: Path
) -> tuple[dict[str, int], dict[str, int]]:
    kept_counts: dict[str, int] = {}
    archived_counts: dict[str, int] = {}
    for name in FILTERED_METADATA:
        current = root / "metadata" / name
        fields, rows = _read_csv(current)
        if "group" not in fields:
            raise ArchiveError(f"metadata table has no group column: {current}")
        pop_rows = [row for row in rows if row.get("group") == "pop"]
        kept_rows = [row for row in rows if row.get("group") != "pop"]
        if name != "split_holdout.csv" and not pop_rows:
            raise ArchiveError(f"expected Pop rows in {current}")
        _write_csv_atomic(archive / "metadata" / name, fields, pop_rows)
        _write_csv_atomic(current, fields, kept_rows)
        archived_counts[name] = len(pop_rows)
        kept_counts[name] = len(kept_rows)
    return kept_counts, archived_counts


def _update_dataset_summary(root: Path, archive: Path) -> dict[str, Any]:
    path = root / "metadata" / "dataset_summary.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    groups = {name: old["groups"][name] for name in ("focus", "classical")}
    combined = {
        "tracks": sum(int(group["tracks"]) for group in groups.values()),
        "discovery": sum(int(group["discovery"]) for group in groups.values()),
        "validation": sum(int(group["validation"]) for group in groups.values()),
        "holdout": sum(int(group["holdout"]) for group in groups.values()),
        "duration_hours": round(
            sum(float(group["duration_hours"]) for group in groups.values()), 2
        ),
        "bytes": sum(int(group["bytes"]) for group in groups.values()),
    }
    combined["gibibytes"] = round(combined["bytes"] / (1024**3), 3)
    payload = {
        "generated_at": date.today().isoformat(),
        "schema_version": 3,
        "canonical_groups": ["focus", "classical"],
        "canonical_focus_source": old.get("canonical_focus_source"),
        "groups": groups,
        "combined": combined,
        "audit": {
            "hashes_verified_before_group_migration": bool(
                old.get("audit", {}).get("hashes_verified")
            ),
            "duplicate_audio": bool(old.get("audit", {}).get("duplicate_audio")),
            "split_leakage": bool(old.get("audit", {}).get("split_leakage")),
            "license_records": combined["tracks"],
        },
        "archived_pop": {
            "status": "removed_from_canonical_dataset",
            "archive": archive.relative_to(root).as_posix(),
            "tracks": int(old["groups"]["pop"]["tracks"]),
            "scientific_results": "historical_only; two-group rerun required",
        },
        "legacy_brainfm": old.get("legacy_brainfm", {}),
    }
    _write_json_atomic(path, payload)
    return payload


def _update_preprocessing_summary(root: Path) -> dict[str, Any]:
    metadata = root / "metadata"
    _, rows = _read_csv(metadata / "preprocessed_segments.csv")
    output_paths = [root / row["output_relative_path"] for row in rows]
    missing = [str(path) for path in output_paths if not path.is_file()]
    if missing:
        raise ArchiveError(f"canonical preprocessed files are missing: {missing[:3]}")
    output_bytes = sum(path.stat().st_size for path in output_paths)
    payload = {
        "generated_at": date.today().isoformat(),
        "ok": True,
        "segments": len(rows),
        "tracks": len({row["track_id"] for row in rows}),
        "status_counts": dict(Counter(row.get("status", "") for row in rows)),
        "group_counts": dict(Counter(row["group"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "scale_counts": dict(
            Counter(f"{int(float(row['scale_seconds']))}s" for row in rows)
        ),
        "output_files": len(output_paths),
        "output_bytes": output_bytes,
        "output_gib": round(output_bytes / (1024**3), 3),
        "canonical_focus_source": "jamendo-api-open-focus",
        "canonical_groups": ["focus", "classical"],
    }
    _write_json_atomic(metadata / "preprocessing_summary.json", payload)
    return payload


def _update_control_summary(root: Path, archive: Path) -> None:
    path = root / "metadata" / "control_dataset_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    classical = payload["classical"]
    payload = {
        "generated_at": date.today().isoformat(),
        "canonical_control_groups": ["classical"],
        "classical": classical,
        "combined": {
            "candidate_tracks": classical["candidate_tracks"],
            "verified_tracks": classical["verified_tracks"],
            "discovery": classical["discovery"],
            "validation": classical["validation"],
            "duration_hours": classical["duration_hours"],
            "bytes": classical["bytes"],
            "gibibytes": classical["gibibytes"],
        },
        "archived_pop": {
            "archive": archive.relative_to(root).as_posix(),
            "tracks": 300,
        },
        "audit": {
            "musicnet_archive_bytes": payload["audit"]["musicnet_archive_bytes"],
            "musicnet_archive_md5": payload["audit"]["musicnet_archive_md5"],
            "musicnet_archive_md5_verified": payload["audit"][
                "musicnet_archive_md5_verified"
            ],
            "sha256_verified_before_group_migration": payload["audit"]["sha256_verified"],
            "duplicate_audio": payload["audit"]["duplicate_audio"],
            "split_leakage": payload["audit"]["split_leakage"],
            "errors": [],
            "warnings": [
                "3 Classical fallback MP3 missing durations were filled by contiguous "
                "MPEG frame scan"
            ],
        },
    }
    _write_json_atomic(path, payload)


def _inventory(archive: Path) -> tuple[int, int, str]:
    inventory_path = archive / "file_inventory.csv"
    paths = sorted(
        path
        for path in archive.rglob("*")
        if path.is_file() and path not in {inventory_path, archive / "migration_audit.json"}
    )
    rows = [
        {
            "relative_path": path.relative_to(archive).as_posix(),
            "bytes": str(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    _write_csv_atomic(inventory_path, ["relative_path", "bytes", "sha256"], rows)
    return len(rows), sum(int(row["bytes"]) for row in rows), _sha256(inventory_path)


def _write_readme(archive: Path) -> None:
    (archive / "README.md").write_text(
        "# Archived Pop dataset\n\n"
        "This directory preserves the 300-track Pop comparison corpus removed from the "
        "canonical dataset on 2026-08-02. It contains raw audio, 180 s/300 s preprocessed "
        "audio, extracted Pop features, Pop metadata rows, and the former three-group state "
        "model. The data remain subject to the per-track licenses in `metadata/licenses.csv`.\n\n"
        "The canonical study now compares Focus with Classical only. Files here are retained "
        "for audit and historical reproducibility and must not be mixed into new two-group "
        "confirmatory analyses.\n",
        encoding="utf-8",
    )


def execute(root: Path, archive: Path) -> dict[str, Any]:
    before = _validate_inputs(root, archive)
    moves = _move_plan(root, archive)
    archive.mkdir(parents=True)
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    kept_counts, archived_counts = _filtered_metadata(root, archive)
    dataset = _update_dataset_summary(root, archive)
    preprocessing = _update_preprocessing_summary(root)
    _update_control_summary(root, archive)
    _write_readme(archive)
    files, total_bytes, inventory_sha256 = _inventory(archive)
    payload = {
        "generated_at": date.today().isoformat(),
        "status": "pop_dataset_archived",
        "archive": archive.relative_to(root).as_posix(),
        "before": before,
        "after": {
            "canonical_groups": dataset["canonical_groups"],
            "tracks": dataset["combined"]["tracks"],
            "preprocessed_segments": preprocessing["segments"],
            "metadata_rows": kept_counts,
        },
        "archived_metadata_rows": archived_counts,
        "archive_inventory": {
            "files": files,
            "bytes": total_bytes,
            "gibibytes": round(total_bytes / (1024**3), 3),
            "sha256": inventory_sha256,
        },
        "next_required_step": (
            "refit state_model on Focus/Classical discovery-180s and transform all "
            "canonical segments"
        ),
    }
    _write_json_atomic(archive / "migration_audit.json", payload)
    _write_json_atomic(root / "metadata" / "pop_archive_migration_audit.json", payload)
    return payload


def finalize_existing(root: Path, archive: Path) -> dict[str, Any]:
    if not archive.is_dir():
        raise ArchiveError(f"archive target does not exist: {archive}")
    remaining = [str(source) for source, _ in _move_plan(root, archive) if source.exists()]
    if remaining:
        raise ArchiveError(
            "migration is not ready to finalize; sources remain: " + "; ".join(remaining)
        )
    _, tracks = _read_csv(root / "metadata" / "track_index.csv")
    _, segments = _read_csv(root / "metadata" / "preprocessed_segments.csv")
    track_counts = Counter(row["group"] for row in tracks)
    segment_counts = Counter(row["group"] for row in segments)
    if track_counts != {"focus": 300, "classical": 300}:
        raise ArchiveError(f"unexpected canonical track counts: {dict(track_counts)}")
    if segment_counts != {"focus": 600, "classical": 600}:
        raise ArchiveError(f"unexpected canonical segment counts: {dict(segment_counts)}")

    archived_counts = {}
    kept_counts = {}
    for name in FILTERED_METADATA:
        _, archived_rows = _read_csv(archive / "metadata" / name)
        _, kept_rows = _read_csv(root / "metadata" / name)
        archived_counts[name] = len(archived_rows)
        kept_counts[name] = len(kept_rows)
    files, total_bytes, inventory_sha256 = _inventory(archive)
    payload = {
        "generated_at": date.today().isoformat(),
        "status": "pop_dataset_archived",
        "archive": archive.relative_to(root).as_posix(),
        "before": {
            "track_counts": {"focus": 300, "pop": 300, "classical": 300},
            "segment_counts": {"focus": 600, "pop": 600, "classical": 600},
        },
        "after": {
            "canonical_groups": ["focus", "classical"],
            "tracks": len(tracks),
            "preprocessed_segments": len(segments),
            "metadata_rows": kept_counts,
        },
        "archived_metadata_rows": archived_counts,
        "archive_inventory": {
            "files": files,
            "bytes": total_bytes,
            "gibibytes": round(total_bytes / (1024**3), 3),
            "sha256": inventory_sha256,
        },
        "next_required_step": (
            "refit state_model on Focus/Classical discovery-180s and transform all "
            "canonical segments"
        ),
        "recovery_note": "finalized after the original process ended during inventory hashing",
    }
    _write_json_atomic(archive / "migration_audit.json", payload)
    _write_json_atomic(root / "metadata" / "pop_archive_migration_audit.json", payload)
    return payload


def dry_run(root: Path, archive: Path) -> dict[str, Any]:
    before = _validate_inputs(root, archive)
    return {
        "status": "dry_run",
        "archive": archive.relative_to(root).as_posix(),
        "before": before,
        "moves": [
            {
                "source": source.relative_to(root).as_posix(),
                "target": target.relative_to(root).as_posix(),
            }
            for source, target in _move_plan(root, archive)
        ],
        "filtered_metadata": list(FILTERED_METADATA),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="archive-pop-dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive-name", default=ARCHIVE_NAME)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    archive = (root / "dataset_archive" / args.archive_name).resolve()
    if args.execute and args.finalize_existing:
        raise ArchiveError("choose either --execute or --finalize-existing")
    if args.execute:
        payload = execute(root, archive)
    elif args.finalize_existing:
        payload = finalize_existing(root, archive)
    else:
        payload = dry_run(root, archive)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArchiveError, OSError, ValueError, KeyError) as exc:
        print(f"archive-pop-dataset: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
