from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .schema import LicenseRecord, SplitName, TrackGroup, TrackRecord

TRACK_COLUMNS = {
    "track_id",
    "group",
    "relative_path",
    "sha256",
    "duration_seconds",
    "sample_rate",
    "channels",
    "artist_key",
    "album_key",
    "composer_key",
    "instrumental",
    "restricted",
}
LICENSE_COLUMNS = {
    "track_id",
    "group",
    "source_url",
    "license_type",
    "downloaded_at",
    "redistribution_allowed",
    "notes",
}


class ManifestError(ValueError):
    """Raised when a metadata file is malformed."""


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise ManifestError(f"missing metadata file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required_columns - columns
        if missing:
            raise ManifestError(f"{path} is missing columns: {sorted(missing)}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _parse_bool(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ManifestError(f"{field_name} must be a boolean, got {value!r}")


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def load_tracks(path: Path) -> list[TrackRecord]:
    records: list[TrackRecord] = []
    seen: set[str] = set()
    for line_number, row in enumerate(_read_rows(path, TRACK_COLUMNS), start=2):
        track_id = row["track_id"]
        if not track_id:
            raise ManifestError(f"{path}:{line_number}: track_id is empty")
        if track_id in seen:
            raise ManifestError(f"{path}:{line_number}: duplicate track_id {track_id!r}")
        seen.add(track_id)
        try:
            group = TrackGroup(row["group"])
            if not row["relative_path"]:
                raise ManifestError("relative_path is empty")
            relative_path = Path(row["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ManifestError("relative_path must stay below data_raw")
            if row["sha256"] and not re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]):
                raise ManifestError("sha256 must contain exactly 64 hexadecimal characters")
            records.append(
                TrackRecord(
                    track_id=track_id,
                    group=group,
                    relative_path=relative_path,
                    sha256=row["sha256"].lower(),
                    duration_seconds=_optional_float(row["duration_seconds"]),
                    sample_rate=_optional_int(row["sample_rate"]),
                    channels=_optional_int(row["channels"]),
                    artist_key=row["artist_key"],
                    album_key=row["album_key"],
                    composer_key=row["composer_key"],
                    instrumental=_parse_bool(row["instrumental"], field_name="instrumental"),
                    restricted=_parse_bool(row["restricted"], field_name="restricted"),
                )
            )
        except (ValueError, ManifestError) as exc:
            raise ManifestError(f"{path}:{line_number}: {exc}") from exc
    return records


def load_licenses(path: Path) -> list[LicenseRecord]:
    records: list[LicenseRecord] = []
    seen: set[str] = set()
    for line_number, row in enumerate(_read_rows(path, LICENSE_COLUMNS), start=2):
        track_id = row["track_id"]
        if not track_id:
            raise ManifestError(f"{path}:{line_number}: track_id is empty")
        if track_id in seen:
            raise ManifestError(f"{path}:{line_number}: duplicate track_id {track_id!r}")
        seen.add(track_id)
        try:
            records.append(
                LicenseRecord(
                    track_id=track_id,
                    group=TrackGroup(row["group"]),
                    source_url=row["source_url"],
                    license_type=row["license_type"],
                    downloaded_at=row["downloaded_at"],
                    redistribution_allowed=_parse_bool(
                        row["redistribution_allowed"], field_name="redistribution_allowed"
                    ),
                    notes=row["notes"],
                )
            )
        except ValueError as exc:
            raise ManifestError(f"{path}:{line_number}: {exc}") from exc
    return records


def load_split(path: Path) -> dict[str, TrackGroup]:
    rows = _read_rows(path, {"track_id", "group"})
    result: dict[str, TrackGroup] = {}
    for line_number, row in enumerate(rows, start=2):
        track_id = row["track_id"]
        if not track_id:
            raise ManifestError(f"{path}:{line_number}: track_id is empty")
        if track_id in result:
            raise ManifestError(f"{path}:{line_number}: duplicate track_id {track_id!r}")
        try:
            result[track_id] = TrackGroup(row["group"])
        except ValueError as exc:
            raise ManifestError(f"{path}:{line_number}: {exc}") from exc
    return result


def _find_leakage(tracks: Iterable[TrackRecord], assignments: Mapping[str, SplitName]) -> list[str]:
    key_splits: dict[tuple[str, str], set[SplitName]] = defaultdict(set)
    for track in tracks:
        split = assignments.get(track.track_id)
        if split is None:
            continue
        for field_name in ("artist_key", "album_key", "composer_key"):
            value = getattr(track, field_name)
            if value:
                key_splits[(field_name, value)].add(split)
    return [
        f"split leakage: {field_name}={value!r} appears in {sorted(s.value for s in splits)}"
        for (field_name, value), splits in key_splits.items()
        if len(splits) > 1
    ]


def validate_metadata(
    metadata_dir: Path,
    data_root: Path,
    *,
    check_files: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    try:
        tracks = load_tracks(metadata_dir / "track_index.csv")
        licenses = load_licenses(metadata_dir / "licenses.csv")
        splits = {
            SplitName.DISCOVERY: load_split(metadata_dir / "split_discovery.csv"),
            SplitName.VALIDATION: load_split(metadata_dir / "split_validation.csv"),
            SplitName.HOLDOUT: load_split(metadata_dir / "split_holdout.csv"),
        }
    except ManifestError as exc:
        report.errors.append(str(exc))
        return report

    track_by_id = {track.track_id: track for track in tracks}
    license_by_id = {license_.track_id: license_ for license_ in licenses}
    assignments: dict[str, SplitName] = {}

    for split_name, members in splits.items():
        for track_id, declared_group in members.items():
            if track_id in assignments:
                report.errors.append(
                    f"track {track_id!r} occurs in both "
                    f"{assignments[track_id].value} and {split_name.value}"
                )
            assignments[track_id] = split_name
            track = track_by_id.get(track_id)
            if track is None:
                report.errors.append(f"{split_name.value}: unknown track_id {track_id!r}")
            elif track.group != declared_group:
                report.errors.append(
                    f"{split_name.value}: group mismatch for {track_id!r}: "
                    f"{declared_group.value} != {track.group.value}"
                )
    for track in tracks:
        license_ = license_by_id.get(track.track_id)
        if track.restricted and license_ and license_.redistribution_allowed:
            report.errors.append(
                f"restricted track {track.track_id!r} cannot be marked redistribution_allowed=true"
            )
        if license_ is None:
            report.errors.append(f"track {track.track_id!r} has no license record")
        if license_ and license_.group != track.group:
            report.errors.append(f"license group mismatch for {track.track_id!r}")
        if (
            license_
            and not track.restricted
            and license_.license_type.strip().lower() in {"", "unknown", "tbd"}
        ):
            report.errors.append(f"open track {track.track_id!r} has no verified license")
        if license_ and not track.restricted and not license_.source_url:
            report.errors.append(f"open track {track.track_id!r} has no source_url")
        if track.track_id not in assignments:
            report.warnings.append(f"track {track.track_id!r} is not assigned to a split")
        if check_files and not (data_root / track.relative_path).is_file():
            report.errors.append(f"missing audio for {track.track_id!r}: {track.relative_path}")

    report.errors.extend(_find_leakage(tracks, assignments))
    group_counts = Counter(track.group.value for track in tracks)
    report.counts = {
        "tracks": len(tracks),
        "licenses": len(licenses),
        **{f"group_{group}": count for group, count in sorted(group_counts.items())},
        **{f"split_{name.value}": len(members) for name, members in splits.items()},
    }
    if not tracks:
        report.warnings.append("track_index.csv has no data rows yet")
    return report
