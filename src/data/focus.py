from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from focus_topology.data.manifest import validate_metadata

SOURCE_DATASET = "brainfm-authorized-focus"
LICENSE_TYPE = "Brain.fm research authorization"
PRIVATE_MAP_NAME = ".private_filename_map.csv"
SAFE_AUDIO_NAME = re.compile(r"focus_brainfm_[0-9a-f]{12}(?:_\d{2})?\.mp3$")

FOCUS_COLUMNS = [
    "track_id",
    "group",
    "source_dataset",
    "category",
    "split",
    "relative_path",
    "sha256",
    "audio_payload_sha256",
    "duration_seconds",
    "metadata_reported_duration_seconds",
    "size_estimated_duration_seconds",
    "duration_discrepancy",
    "mpeg_frame_count",
    "valid_audio_bytes",
    "file_bytes",
    "sample_rate",
    "channels",
    "average_bitrate_kbps",
    "duration_method",
    "license_type",
    "restricted",
    "redistribution_allowed",
    "classification_evidence",
    "download_status",
    "duplicate_of",
    "cataloged_at",
    "error",
]

PRIVATE_MAP_COLUMNS = [
    "track_id",
    "original_filename",
    "current_filename",
    "category",
    "classification_evidence",
    "sha256",
]


class FocusDatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FrameHeader:
    version: str
    layer: int
    bitrate_kbps: int
    sample_rate: int
    padding: int
    channels: int
    samples: int
    frame_bytes: int


@dataclass(frozen=True, slots=True)
class MP3Scan:
    sha256: str
    audio_payload_sha256: str
    duration_seconds: float
    metadata_reported_duration_seconds: float | None
    size_estimated_duration_seconds: float
    duration_discrepancy: bool
    mpeg_frame_count: int
    valid_audio_bytes: int
    file_bytes: int
    sample_rate: int
    channels: int
    average_bitrate_kbps: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_header(data: mmap.mmap, position: int) -> FrameHeader | None:
    if position < 0 or position + 4 > len(data):
        return None
    value = int.from_bytes(data[position : position + 4], "big")
    if value & 0xFFE00000 != 0xFFE00000:
        return None
    version_bits = (value >> 19) & 0b11
    layer_bits = (value >> 17) & 0b11
    bitrate_index = (value >> 12) & 0b1111
    sample_rate_index = (value >> 10) & 0b11
    padding = (value >> 9) & 1
    channel_mode = (value >> 6) & 0b11
    if version_bits == 1 or layer_bits == 0 or bitrate_index in {0, 15}:
        return None
    if sample_rate_index == 3:
        return None

    version = {3: "1", 2: "2", 0: "2.5"}[version_bits]
    layer = {3: 1, 2: 2, 1: 3}[layer_bits]
    mpeg1_rates = {
        1: [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
        2: [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
        3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    }
    low_rates = {
        1: [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
        2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        3: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    }
    bitrate = (mpeg1_rates if version == "1" else low_rates)[layer][bitrate_index]
    base_sample_rate = [44100, 48000, 32000][sample_rate_index]
    divisor = 1 if version == "1" else 2 if version == "2" else 4
    sample_rate = base_sample_rate // divisor
    if layer == 1:
        samples = 384
        frame_bytes = (12 * bitrate * 1000 // sample_rate + padding) * 4
    elif layer == 2:
        samples = 1152
        frame_bytes = 144 * bitrate * 1000 // sample_rate + padding
    else:
        samples = 1152 if version == "1" else 576
        coefficient = 144 if version == "1" else 72
        frame_bytes = coefficient * bitrate * 1000 // sample_rate + padding
    if frame_bytes < 24:
        return None
    return FrameHeader(
        version=version,
        layer=layer,
        bitrate_kbps=bitrate,
        sample_rate=sample_rate,
        padding=padding,
        channels=1 if channel_mode == 3 else 2,
        samples=samples,
        frame_bytes=frame_bytes,
    )


def _compatible(left: FrameHeader, right: FrameHeader) -> bool:
    return (
        left.version == right.version
        and left.layer == right.layer
        and left.sample_rate == right.sample_rate
    )


def _metadata_reported_duration(path: Path) -> float | None:
    try:
        from mutagen.mp3 import MP3  # type: ignore[import-not-found]

        return float(MP3(path).info.length)
    except (ImportError, OSError, ValueError):
        return None


def scan_mp3(path: Path) -> MP3Scan:
    """Count contiguous MPEG frames instead of trusting a duration header."""

    file_bytes = path.stat().st_size
    if file_bytes < 4:
        raise FocusDatasetError(f"not a usable MP3: {path}")
    file_digest = _sha256(path)
    payload_digest = hashlib.sha256()
    duration = 0.0
    frame_count = 0
    valid_audio_bytes = 0
    sample_rates: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()

    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        view = memoryview(data)
        try:
            position = 0
            while position + 4 <= file_bytes:
                start = data.find(b"\xff", position)
                if start < 0:
                    break
                first = _frame_header(data, start)
                if first is None:
                    position = start + 1
                    continue
                second_position = start + first.frame_bytes
                second = _frame_header(data, second_position)
                if second is None or not _compatible(first, second):
                    position = start + 1
                    continue

                run_start = start
                run_position = start
                run_frames: list[FrameHeader] = []
                reference = first
                while run_position + 4 <= file_bytes:
                    header = _frame_header(data, run_position)
                    if header is None or not _compatible(reference, header):
                        break
                    end = run_position + header.frame_bytes
                    if end > file_bytes:
                        break
                    run_frames.append(header)
                    run_position = end
                if len(run_frames) < 3:
                    position = start + 1
                    continue
                payload_digest.update(view[run_start:run_position])
                for header in run_frames:
                    duration += header.samples / header.sample_rate
                    valid_audio_bytes += header.frame_bytes
                    sample_rates[header.sample_rate] += 1
                    channel_counts[header.channels] += 1
                frame_count += len(run_frames)
                position = run_position
        finally:
            view.release()

    if frame_count == 0 or duration <= 0:
        raise FocusDatasetError(f"no contiguous MPEG audio frames found: {path}")
    average_bitrate = valid_audio_bytes * 8 / duration / 1000
    size_estimate = file_bytes * 8 / average_bitrate / 1000
    reported = _metadata_reported_duration(path)
    comparison = max(size_estimate, reported or 0.0)
    discrepancy = comparison - duration >= 300 and comparison / duration >= 2
    return MP3Scan(
        sha256=file_digest,
        audio_payload_sha256=payload_digest.hexdigest(),
        duration_seconds=duration,
        metadata_reported_duration_seconds=reported,
        size_estimated_duration_seconds=size_estimate,
        duration_discrepancy=discrepancy,
        mpeg_frame_count=frame_count,
        valid_audio_bytes=valid_audio_bytes,
        file_bytes=file_bytes,
        sample_rate=sample_rates.most_common(1)[0][0],
        channels=channel_counts.most_common(1)[0][0],
        average_bitrate_kbps=average_bitrate,
    )


def _direct_category(path: Path) -> tuple[str, str] | None:
    stem = path.stem.lower()
    if re.search(r"deep[ _-]?work", stem):
        return "focus_deepwork", "filename_category_keyword"
    if "learning" in stem:
        return "focus_learning", "filename_category_keyword"
    return None


def classify_paths(
    paths: Sequence[Path],
    seeded: dict[Path, tuple[str, str]] | None = None,
) -> dict[Path, tuple[str, str]]:
    seeded = seeded or {}
    direct = {path: seeded.get(path) or _direct_category(path) for path in paths}
    known = [
        (path, path.stat().st_mtime, result[0])
        for path, result in direct.items()
        if result is not None
    ]
    if not known:
        raise FocusDatasetError("no Deep Work or Learning filename markers were found")
    classifications: dict[Path, tuple[str, str]] = {
        path: result for path, result in direct.items() if result is not None
    }
    for path, result in direct.items():
        if result is not None:
            continue
        modified = path.stat().st_mtime
        nearest = sorted(known, key=lambda item: abs(item[1] - modified))[:4]
        categories = {item[2] for item in nearest}
        if len(nearest) < 4 or len(categories) != 1:
            raise FocusDatasetError(f"cannot infer a unique download batch for {path.name!r}")
        classifications[path] = (nearest[0][2], "nearest_download_batch_4_of_4")
    return classifications


def _allocate_splits(count: int) -> dict[str, int]:
    fractions = {"discovery": 0.65, "validation": 0.20, "holdout": 0.15}
    raw = {name: count * fraction for name, fraction in fractions.items()}
    result = {name: int(value) for name, value in raw.items()}
    remaining = count - sum(result.values())
    order = sorted(fractions, key=lambda name: (raw[name] - result[name], name), reverse=True)
    for name in order[:remaining]:
        result[name] += 1
    return result


def assign_splits(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    split_names = ("discovery", "validation", "holdout")
    total_allocation = _allocate_splits(len(rows))
    cell_counts: dict[tuple[str, str], int] = {}
    fractions: dict[tuple[str, str], float] = {}
    for category, members in by_category.items():
        for split in split_names:
            raw = len(members) * total_allocation[split] / len(rows)
            cell_counts[(category, split)] = int(raw)
            fractions[(category, split)] = raw - int(raw)
    row_deficits = {
        category: len(members)
        - sum(cell_counts[(category, split)] for split in split_names)
        for category, members in by_category.items()
    }
    column_deficits = {
        split: total_allocation[split]
        - sum(cell_counts[(category, split)] for category in by_category)
        for split in split_names
    }
    candidates = sorted(
        cell_counts,
        key=lambda cell: (-fractions[cell], cell[0], split_names.index(cell[1])),
    )
    while any(row_deficits.values()):
        for category, split in candidates:
            if row_deficits[category] and column_deficits[split]:
                cell_counts[(category, split)] += 1
                row_deficits[category] -= 1
                column_deficits[split] -= 1
                break
        else:
            raise FocusDatasetError("cannot construct category-stratified split allocation")

    for category, members in by_category.items():
        members = sorted(members, key=lambda row: (row["sha256"], row["track_id"]))
        index = 0
        for split in split_names:
            count = cell_counts[(category, split)]
            for row in members[index : index + count]:
                assignments[row["track_id"]] = split
            index += count
    return assignments


def _write_csv_atomic(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise FocusDatasetError(f"{path} is missing columns: {sorted(missing)}")
        return [{key: value or "" for key, value in row.items()} for row in reader]


def _private_sources(root: Path) -> tuple[list[Path], dict[str, dict[str, str]]]:
    mapping_path = root / PRIVATE_MAP_NAME
    if mapping_path.is_file():
        mapping = _read_csv(mapping_path, PRIVATE_MAP_COLUMNS)
        mapped_names = {row["current_filename"] for row in mapping}
        existing_rows = [row for row in mapping if (root / row["current_filename"]).is_file()]
        by_current = {row["current_filename"]: row for row in existing_rows}
        unlisted = [
            path
            for path in root.glob("*.mp3")
            if path.name not in mapped_names
        ]
        removed = len(mapping) - len(existing_rows)
        if removed or unlisted:
            print(
                f"Focus private-map reconciliation: removed={removed}, added={len(unlisted)}",
                flush=True,
            )
        paths = [root / row["current_filename"] for row in existing_rows]
        paths.extend(unlisted)
        return sorted(paths, key=lambda path: path.name.casefold()), by_current
    paths = sorted(root.glob("*.mp3"), key=lambda path: path.name.casefold())
    return paths, {}


def catalog_focus(
    root: Path,
    output: Path,
    *,
    workers: int = 4,
    expected_count: int = 200,
) -> list[dict[str, str]]:
    paths, existing_map = _private_sources(root)
    if len(paths) != expected_count:
        raise FocusDatasetError(f"expected {expected_count} MP3 files, found {len(paths)}")
    if existing_map:
        seeded = {
            path: (
                existing_map[path.name]["category"],
                existing_map[path.name]["classification_evidence"],
            )
            for path in paths
            if path.name in existing_map
        }
        classifications = classify_paths(paths, seeded)
    else:
        classifications = classify_paths(paths)

    scans: dict[Path, MP3Scan] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_mp3, path): path for path in paths}
        for index, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            scans[path] = future.result()
            if index % 10 == 0 or index == len(paths):
                print(f"Focus MP3 scans: {index}/{len(paths)}", flush=True)

    id_counts: Counter[str] = Counter()
    staged: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, str]] = []
    for path in paths:
        scan = scans[path]
        base_id = f"focus_brainfm_{scan.sha256[:12]}"
        id_counts[base_id] += 1
        suffix = f"_{id_counts[base_id]:02d}" if id_counts[base_id] > 1 else ""
        track_id = f"{base_id}{suffix}"
        target_name = f"{track_id}.mp3"
        category, evidence = classifications[path]
        staged.append(
            {
                "track_id": track_id,
                "category": category,
                "sha256": scan.sha256,
                "scan": scan,
                "original_path": path,
                "target_name": target_name,
                "classification_evidence": evidence,
            }
        )
        old = existing_map.get(path.name)
        mapping_rows.append(
            {
                "track_id": track_id,
                "original_filename": old["original_filename"] if old else path.name,
                "current_filename": target_name,
                "category": category,
                "classification_evidence": evidence,
                "sha256": scan.sha256,
            }
        )

    mapping_path = root / PRIVATE_MAP_NAME
    _write_csv_atomic(mapping_path, PRIVATE_MAP_COLUMNS, mapping_rows)
    target_names = {row["target_name"] for row in staged}
    if len(target_names) != len(staged):
        raise FocusDatasetError("anonymous target filenames are not unique")
    for row in staged:
        source = row["original_path"]
        target = root / row["target_name"]
        if source == target:
            continue
        if target.exists():
            raise FocusDatasetError(f"anonymous target already exists: {target}")
        os.replace(source, target)

    canonical_by_payload: dict[str, str] = {}
    for row in sorted(staged, key=lambda item: item["track_id"]):
        payload = row["scan"].audio_payload_sha256
        row["duplicate_of"] = canonical_by_payload.get(payload, "")
        canonical_by_payload.setdefault(payload, row["track_id"])
    verified_staged = [row for row in staged if not row["duplicate_of"]]
    assignments = assign_splits(verified_staged)
    today = date.today().isoformat()
    public_rows: list[dict[str, str]] = []
    for row in staged:
        scan: MP3Scan = row["scan"]
        public_rows.append(
            {
                "track_id": row["track_id"],
                "group": "focus",
                "source_dataset": SOURCE_DATASET,
                "category": row["category"],
                "split": assignments.get(row["track_id"], ""),
                "relative_path": f"focus_music/{row['target_name']}",
                "sha256": scan.sha256,
                "audio_payload_sha256": scan.audio_payload_sha256,
                "duration_seconds": f"{scan.duration_seconds:.3f}",
                "metadata_reported_duration_seconds": (
                    f"{scan.metadata_reported_duration_seconds:.3f}"
                    if scan.metadata_reported_duration_seconds is not None
                    else ""
                ),
                "size_estimated_duration_seconds": f"{scan.size_estimated_duration_seconds:.3f}",
                "duration_discrepancy": str(scan.duration_discrepancy).lower(),
                "mpeg_frame_count": str(scan.mpeg_frame_count),
                "valid_audio_bytes": str(scan.valid_audio_bytes),
                "file_bytes": str(scan.file_bytes),
                "sample_rate": str(scan.sample_rate),
                "channels": str(scan.channels),
                "average_bitrate_kbps": f"{scan.average_bitrate_kbps:.3f}",
                "duration_method": "contiguous_mpeg_frame_scan",
                "license_type": LICENSE_TYPE,
                "restricted": "true",
                "redistribution_allowed": "false",
                "classification_evidence": row["classification_evidence"],
                "download_status": (
                    "excluded_duplicate" if row["duplicate_of"] else "verified"
                ),
                "duplicate_of": row["duplicate_of"],
                "cataloged_at": today,
                "error": "",
            }
        )
    public_rows.sort(key=lambda row: row["track_id"])
    _write_csv_atomic(output, FOCUS_COLUMNS, public_rows)
    return public_rows


def _read_existing_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        return columns, [{key: value or "" for key, value in row.items()} for row in reader]


def integrate_focus(manifest: Path, metadata_dir: Path) -> dict[str, int]:
    rows = _read_csv(manifest, FOCUS_COLUMNS)
    verified = [row for row in rows if row["download_status"] == "verified"]
    unexpected = [
        row for row in rows if row["download_status"] not in {"verified", "excluded_duplicate"}
    ]
    if unexpected:
        raise FocusDatasetError("Focus manifest contains an unsupported status")

    track_columns, tracks = _read_existing_csv(metadata_dir / "track_index.csv")
    license_columns, licenses = _read_existing_csv(metadata_dir / "licenses.csv")
    tracks = [row for row in tracks if row.get("group") != "focus"]
    licenses = [row for row in licenses if row.get("group") != "focus"]
    for row in verified:
        tracks.append(
            {
                "track_id": row["track_id"],
                "group": "focus",
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "duration_seconds": row["duration_seconds"],
                "sample_rate": row["sample_rate"],
                "channels": row["channels"],
                "artist_key": "",
                "album_key": "",
                "composer_key": "",
                "instrumental": "true",
                "restricted": "true",
            }
        )
        licenses.append(
            {
                "track_id": row["track_id"],
                "group": "focus",
                "source_url": "",
                "license_type": row["license_type"],
                "downloaded_at": row["cataloged_at"],
                "redistribution_allowed": "false",
                "notes": (
                    f"{SOURCE_DATASET}; {row['category']}; restricted research use; "
                    "original filename retained only in the ignored private map"
                ),
            }
        )
    _write_csv_atomic(
        metadata_dir / "track_index.csv", track_columns, sorted(tracks, key=lambda row: row["track_id"])
    )
    _write_csv_atomic(
        metadata_dir / "licenses.csv",
        license_columns,
        sorted(licenses, key=lambda row: row["track_id"]),
    )

    split_counts: Counter[str] = Counter()
    for split in ("discovery", "validation", "holdout"):
        path = metadata_dir / f"split_{split}.csv"
        columns, members = _read_existing_csv(path)
        members = [row for row in members if row.get("group") != "focus"]
        additions = [
            {"track_id": row["track_id"], "group": "focus"}
            for row in verified
            if row["split"] == split
        ]
        split_counts[split] = len(additions)
        members.extend(additions)
        _write_csv_atomic(path, columns, sorted(members, key=lambda row: row["track_id"]))
    return {"focus": len(verified), **dict(split_counts)}


def audit_focus(
    manifest: Path,
    data_root: Path,
    *,
    verify_hash: bool = False,
    expected_count: int = 200,
) -> dict[str, Any]:
    rows = _read_csv(manifest, FOCUS_COLUMNS)
    errors: list[str] = []
    warnings: list[str] = []
    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} Focus rows, found {len(rows)}")
    if set(row["category"] for row in rows) != {"focus_deepwork", "focus_learning"}:
        errors.append("Focus categories must be exactly deepwork and learning")
    seen_ids: set[str] = set()
    canonical_hashes: dict[str, str] = {}
    canonical_payloads: dict[str, str] = {}
    for row in rows:
        track_id = row["track_id"]
        if track_id in seen_ids:
            errors.append(f"duplicate track_id: {track_id}")
        seen_ids.add(track_id)
        relative = PurePosixPath(row["relative_path"])
        if relative.parent.as_posix() != "focus_music" or not SAFE_AUDIO_NAME.fullmatch(relative.name):
            errors.append(f"non-anonymous Focus path: {track_id}")
        target = data_root / relative
        if not target.is_file():
            errors.append(f"missing Focus audio: {track_id}")
            continue
        if verify_hash and _sha256(target) != row["sha256"]:
            errors.append(f"checksum mismatch: {track_id}")
        duplicate_of = row["duplicate_of"]
        expected_file_duplicate = canonical_hashes.get(row["sha256"], "")
        if expected_file_duplicate and duplicate_of != expected_file_duplicate:
            errors.append(f"unmarked duplicate file audio: {expected_file_duplicate} and {track_id}")
        canonical_hashes.setdefault(row["sha256"], track_id)
        payload = row["audio_payload_sha256"]
        expected_payload_duplicate = canonical_payloads.get(payload, "")
        if expected_payload_duplicate and duplicate_of != expected_payload_duplicate:
            errors.append(
                f"unmarked duplicate MPEG payload: {expected_payload_duplicate} and {track_id}"
            )
        canonical_payloads.setdefault(payload, track_id)
        if duplicate_of and row["download_status"] != "excluded_duplicate":
            errors.append(f"duplicate row is not excluded: {track_id}")
        if not duplicate_of and row["download_status"] != "verified":
            errors.append(f"canonical row is not verified: {track_id}")
        if float(row["duration_seconds"]) <= 0:
            errors.append(f"non-positive duration: {track_id}")
        if row["restricted"] != "true" or row["redistribution_allowed"] != "false":
            errors.append(f"invalid restricted-data policy: {track_id}")
    discrepancy_count = sum(row["duration_discrepancy"] == "true" for row in rows)
    if discrepancy_count:
        warnings.append(
            f"{discrepancy_count} files have inflated metadata/size duration; frame-scan duration is used"
        )
    return {
        "ok": not errors,
        "candidate_tracks": len(rows),
        "verified_tracks": sum(row["download_status"] == "verified" for row in rows),
        "excluded_duplicate_tracks": sum(
            row["download_status"] == "excluded_duplicate" for row in rows
        ),
        "by_category": dict(
            Counter(row["category"] for row in rows if row["download_status"] == "verified")
        ),
        "by_split": dict(Counter(row["split"] for row in rows if row["split"])),
        "duration_discrepancy_tracks": discrepancy_count,
        "sha256_verified": verify_hash and not any("checksum mismatch" in item for item in errors),
        "errors": errors,
        "warnings": warnings,
    }


def write_dataset_summary(
    manifest: Path,
    control_summary: Path,
    output: Path,
    metadata_dir: Path,
    data_root: Path,
    *,
    focus_audit_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _read_csv(manifest, FOCUS_COLUMNS)
    verified = [row for row in rows if row["download_status"] == "verified"]
    controls = json.loads(control_summary.read_text(encoding="utf-8"))
    focus_audit = focus_audit_report or audit_focus(manifest, data_root, verify_hash=True)
    canonical = validate_metadata(metadata_dir, data_root, check_files=True)
    durations = [float(row["duration_seconds"]) for row in verified]
    file_bytes = sum(int(row["file_bytes"]) for row in verified)
    all_file_bytes = sum(int(row["file_bytes"]) for row in rows)
    valid_audio_bytes = sum(int(row["valid_audio_bytes"]) for row in verified)
    categories = Counter(row["category"] for row in verified)
    provided_categories = Counter(row["category"] for row in rows)
    splits = Counter(row["split"] for row in verified)
    sample_rates = Counter(row["sample_rate"] for row in verified)
    channels = Counter(row["channels"] for row in verified)
    discrepancies = [row for row in rows if row["duration_discrepancy"] == "true"]
    verified_discrepancies = [
        row for row in verified if row["duration_discrepancy"] == "true"
    ]
    category_duration_hours = {
        category: round(
            sum(float(row["duration_seconds"]) for row in members) / 3600, 2
        )
        for category, members in (
            (category, [row for row in verified if row["category"] == category])
            for category in sorted(categories)
        )
    }
    duration_bins = {
        "under_15_minutes": sum(value < 15 * 60 for value in durations),
        "minutes_15_to_45": sum(15 * 60 <= value < 45 * 60 for value in durations),
        "minutes_45_to_90": sum(45 * 60 <= value < 90 * 60 for value in durations),
        "minutes_90_plus": sum(value >= 90 * 60 for value in durations),
    }
    control_combined = controls["combined"]
    control_audit = controls["audit"]
    index_rows = _read_existing_csv(metadata_dir / "track_index.csv")[1]
    duplicate_index_hashes = [
        digest for digest, count in Counter(row["sha256"] for row in index_rows).items() if count > 1
    ]
    summary = {
        "generated_at": date.today().isoformat(),
        "focus": {
            "candidate_tracks": len(rows),
            "verified_tracks": len(verified),
            "excluded_duplicate_tracks": len(rows) - len(verified),
            "restricted": True,
            "redistribution_allowed": False,
            "source_dataset": SOURCE_DATASET,
            "license_type": LICENSE_TYPE,
            "category_counts": dict(sorted(categories.items())),
            "category_duration_hours": category_duration_hours,
            "provided_file_category_counts": dict(sorted(provided_categories.items())),
            "split_counts": dict(sorted(splits.items())),
            "duration_seconds": round(sum(durations), 3),
            "duration_hours": round(sum(durations) / 3600, 2),
            "minimum_duration_seconds": round(min(durations), 3),
            "maximum_duration_seconds": round(max(durations), 3),
            "duration_bin_counts": duration_bins,
            "file_bytes": file_bytes,
            "gibibytes": round(file_bytes / 1024**3, 2),
            "provided_file_bytes": all_file_bytes,
            "valid_mpeg_audio_bytes": valid_audio_bytes,
            "duration_discrepancy_tracks": len(verified_discrepancies),
            "provided_duration_discrepancy_files": len(discrepancies),
            "discrepancy_actual_duration_hours": round(
                sum(float(row["duration_seconds"]) for row in verified_discrepancies) / 3600,
                2,
            ),
            "discrepancy_reported_duration_hours": round(
                sum(
                    float(row["metadata_reported_duration_seconds"])
                    for row in verified_discrepancies
                    if row["metadata_reported_duration_seconds"]
                )
                / 3600,
                2,
            ),
            "discrepancy_actual_duration_range_seconds": [
                round(min(float(row["duration_seconds"]) for row in verified_discrepancies), 3),
                round(max(float(row["duration_seconds"]) for row in verified_discrepancies), 3),
            ]
            if verified_discrepancies
            else [],
            "discrepancy_reported_duration_range_seconds": [
                round(
                    min(
                        float(row["metadata_reported_duration_seconds"])
                        for row in verified_discrepancies
                        if row["metadata_reported_duration_seconds"]
                    ),
                    3,
                ),
                round(
                    max(
                        float(row["metadata_reported_duration_seconds"])
                        for row in verified_discrepancies
                        if row["metadata_reported_duration_seconds"]
                    ),
                    3,
                ),
            ]
            if verified_discrepancies
            else [],
            "duration_method": "contiguous_mpeg_frame_scan",
            "sample_rate_counts": dict(sorted(sample_rates.items())),
            "channel_counts": dict(sorted(channels.items())),
        },
        "pop": controls["pop"],
        "classical": controls["classical"],
        "combined": {
            "candidate_tracks": len(rows) + int(control_combined["candidate_tracks"]),
            "verified_tracks": len(verified) + int(control_combined["verified_tracks"]),
            "discovery": splits["discovery"] + int(control_combined["discovery"]),
            "validation": splits["validation"] + int(control_combined["validation"]),
            "holdout": splits["holdout"],
            "duration_hours": round(
                sum(durations) / 3600 + float(control_combined["duration_hours"]), 2
            ),
            "bytes": file_bytes + int(control_combined["bytes"]),
            "gibibytes": round(
                (file_bytes + int(control_combined["bytes"])) / 1024**3, 2
            ),
        },
        "audit": {
            "canonical_manifest_ok": canonical.ok,
            "canonical_counts": canonical.counts,
            "focus_sha256_verified": focus_audit["sha256_verified"],
            "control_sha256_verified": control_audit["sha256_verified"],
            "duplicate_audio": bool(duplicate_index_hashes),
            "split_leakage": any("leakage" in item for item in canonical.errors),
            "errors": [*focus_audit["errors"], *canonical.errors],
            "warnings": [*focus_audit["warnings"], *canonical.warnings],
        },
    }
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-catalog")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path("data_raw/focus_music"))
    build.add_argument("--manifest", type=Path, default=Path("metadata/focus_manifest.csv"))
    build.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    build.add_argument(
        "--control-summary", type=Path, default=Path("metadata/control_dataset_summary.json")
    )
    build.add_argument("--summary", type=Path, default=Path("metadata/dataset_summary.json"))
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--expected-count", type=int, default=200)

    audit = subparsers.add_parser("audit")
    audit.add_argument("manifest", type=Path, nargs="?", default=Path("metadata/focus_manifest.csv"))
    audit.add_argument("--data-root", type=Path, default=Path("data_raw"))
    audit.add_argument("--verify-hash", action="store_true")
    audit.add_argument("--expected-count", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_focus(
                args.manifest,
                args.data_root,
                verify_hash=args.verify_hash,
                expected_count=args.expected_count,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        rows = catalog_focus(
            args.root,
            args.manifest,
            workers=args.workers,
            expected_count=args.expected_count,
        )
        integration = integrate_focus(args.manifest, args.metadata_dir)
        audit_report = audit_focus(
            args.manifest, Path("data_raw"), verify_hash=True, expected_count=args.expected_count
        )
        if not audit_report["ok"]:
            raise FocusDatasetError("Focus audit failed: " + "; ".join(audit_report["errors"]))
        summary = write_dataset_summary(
            args.manifest,
            args.control_summary,
            args.summary,
            args.metadata_dir,
            Path("data_raw"),
            focus_audit_report=audit_report,
        )
        print(
            json.dumps(
                {
                    "cataloged": len(rows),
                    "integration": integration,
                    "focus_audit": audit_report,
                    "combined": summary["combined"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (FocusDatasetError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
