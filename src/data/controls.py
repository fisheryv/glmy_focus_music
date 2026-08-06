from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import shutil
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

MUSOPEN_ITEM_PAGE = "https://archive.org/details/musopen-lossless-dvd"
MUSOPEN_ARCHIVE_VIEW = (
    "https://archive.org/download/musopen-lossless-dvd/Musopen-Lossless-DVD.zip/"
)
MUSOPEN_STANDARD_ITEM_PAGE = "https://archive.org/details/musopen-dvd"
MUSOPEN_STANDARD_VIEW = "https://archive.org/download/musopen-dvd/Musopen-DVD.zip/"
JAMENDO_FILE_API = (
    "https://api.jamendo.com/v3.0/tracks/file/"
    "?client_id={client_id}&id={track_id}&audioformat=mp32&action=download"
)
JAMENDO_TRACKS_API = "https://api.jamendo.com/v3.0/tracks/"
MUSICNET_RECORD_PAGE = "https://zenodo.org/records/5120004"
MUSICNET_ARCHIVE_URL = (
    "https://zenodo.org/api/records/5120004/files/musicnet.tar.gz/content"
)
MUSICNET_ARCHIVE_MD5 = "844764911fa0d5b97c97da944a057590"
MUSICNET_ARCHIVE_BYTES = 11_097_394_998
MUSICNET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
USER_AGENT = "FocusMusicGLMY/0.1 (non-commercial academic dataset builder)"

CANDIDATE_COLUMNS = [
    "track_id",
    "group",
    "source_dataset",
    "source_track_id",
    "title",
    "creator",
    "artist_key",
    "album_key",
    "composer_key",
    "subpool",
    "duration_seconds",
    "source_url",
    "download_url",
    "license_type",
    "license_url",
    "instrumental_evidence",
    "split",
    "relative_path",
    "sha256",
    "download_status",
    "downloaded_at",
    "error",
]


class ControlDatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JamendoLicense:
    path: str
    source_url: str
    license_text: str
    license_url: str


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _read_candidate_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ControlDatasetError(f"candidate manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CANDIDATE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ControlDatasetError(f"{path} is missing columns: {sorted(missing)}")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def _write_csv_atomic(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _request(url: str, *, headers: dict[str, str] | None = None) -> urllib.response.addinfourl:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    return urllib.request.urlopen(  # noqa: S310 - URLs are fixed or recorded in manifests.
        urllib.request.Request(url, headers=request_headers), timeout=90
    )


def _fetch_text(url: str) -> str:
    with _request(url) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def fetch_file_with_resume(
    url: str,
    target: Path,
    *,
    expected_bytes: int,
    max_attempts: int = 100,
) -> None:
    """Download a large source artifact with HTTP Range retries."""

    if target.is_file() and target.stat().st_size == expected_bytes:
        return
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    last_report = partial.stat().st_size if partial.exists() else 0
    attempt = 0
    while (partial.stat().st_size if partial.exists() else 0) < expected_bytes:
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with _request(url, headers=headers) as response:
                append = existing > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                if existing and not append:
                    existing = 0
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        current = output.tell()
                        if current - last_report >= 256 * 1024 * 1024:
                            print(
                                f"MusicNet archive: {current / (1024**3):.2f}/"
                                f"{expected_bytes / (1024**3):.2f} GiB",
                                flush=True,
                            )
                            last_report = current
            attempt = 0
        except (OSError, urllib.error.URLError) as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise ControlDatasetError(
                    f"large-file download failed after {attempt} retries: {exc}"
                ) from exc
            time.sleep(min(10, 2**min(attempt, 4)))
    actual_bytes = partial.stat().st_size
    if actual_bytes != expected_bytes:
        raise ControlDatasetError(
            f"downloaded size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    os.replace(partial, target)


def _fetch_range_segment(
    url: str,
    path: Path,
    start: int,
    end: int,
    *,
    max_attempts: int = 100,
) -> Path:
    expected = end - start + 1
    attempt = 0
    while (path.stat().st_size if path.exists() else 0) < expected:
        existing = path.stat().st_size if path.exists() else 0
        if existing > expected:
            raise ControlDatasetError(f"segment is larger than expected: {path}")
        before = existing
        request_end = min(end, start + existing + 16 * 1024 * 1024 - 1)
        headers = {"Range": f"bytes={start + existing}-{request_end}"}
        try:
            with _request(url, headers=headers) as response:
                if getattr(response, "status", None) != 206:
                    raise ControlDatasetError(
                        f"server ignored Range for segment {start}-{end}"
                    )
                with path.open("ab") as output:
                    while output.tell() < expected:
                        chunk = response.read(min(1024 * 1024, expected - output.tell()))
                        if not chunk:
                            break
                        output.write(chunk)
            after = path.stat().st_size
            attempt = 0 if after > before else attempt + 1
        except (OSError, urllib.error.URLError) as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise ControlDatasetError(
                    f"segment {start}-{end} failed after {attempt} retries: {exc}"
                ) from exc
            time.sleep(min(10, 2**min(attempt, 4)))
    return path


def fetch_file_segmented(
    url: str,
    target: Path,
    *,
    expected_bytes: int,
    workers: int = 8,
) -> None:
    """Download a static file in independently resumable HTTP Range segments."""

    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    if target.is_file() and target.stat().st_size == expected_bytes:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    segment_size = (expected_bytes + workers - 1) // workers
    specifications: list[tuple[Path, int, int]] = []
    for index in range(workers):
        start = index * segment_size
        if start >= expected_bytes:
            break
        end = min(expected_bytes - 1, start + segment_size - 1)
        path = target.with_name(f"{target.name}.segment-{index:02d}.part")
        specifications.append((path, start, end))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_range_segment, url, path, start, end): index
            for index, (path, start, end) in enumerate(specifications)
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"MusicNet segments: {completed}/{len(futures)} complete", flush=True)
    assembling = target.with_suffix(target.suffix + ".assembling.part")
    assembling.unlink(missing_ok=True)
    with assembling.open("wb") as output:
        for path, _, _ in specifications:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if assembling.stat().st_size != expected_bytes:
        raise ControlDatasetError(
            f"assembled size mismatch: expected {expected_bytes}, got {assembling.stat().st_size}"
        )
    os.replace(assembling, target)
    for path, _, _ in specifications:
        path.unlink(missing_ok=True)


def parse_jamendo_licenses(path: Path) -> dict[str, JamendoLicense]:
    """Parse MTG's official four-line-per-track audio license ledger."""

    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, JamendoLicense] = {}
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        audio_path = lines[index].strip()
        source_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        license_line = lines[index + 2].strip() if index + 2 < len(lines) else ""
        source_match = re.search(r"(https?://\S+)$", source_line)
        license_match = re.search(r"(https?://\S+)$", license_line)
        if not source_match or not license_match:
            raise ControlDatasetError(f"cannot parse license block beginning with {audio_path!r}")
        license_url = license_match.group(1)
        license_text = license_line.removeprefix("Available under a ").split(": http", 1)[0]
        result[audio_path] = JamendoLicense(
            path=audio_path,
            source_url=source_match.group(1),
            license_text=license_text,
            license_url=license_url,
        )
        index += 3
    return result


def _read_mtg_tracks(path: Path) -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:5] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"]:
            raise ControlDatasetError(f"unexpected MTG metadata header in {path}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            tracks[parts[0]] = {
                "track_id": parts[0],
                "artist_id": parts[1],
                "album_id": parts[2],
                "path": parts[3],
                "duration": parts[4],
                "tags": set(parts[5:]),
            }
    return tracks


def _read_mtg_names(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["TRACK_ID"]: {key: (value or "") for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        }


def _read_unanimous_instrumental(path: Path) -> set[str]:
    expected = "voice_instrumental---instrumental,instrumental,instrumental"
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if expected in parts[5:]:
                selected.add(parts[0])
    return selected


def _assign_grouped_split(
    rows: list[dict[str, str]],
    *,
    group_field: str,
    validation_fraction: float,
    seed: int,
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_field]].append(row)
    target = round(len(rows) * validation_fraction)
    validation_count = 0
    ordered = sorted(grouped, key=lambda key: (-len(grouped[key]), _stable_key(seed, key)))
    for key in ordered:
        size = len(grouped[key])
        current_error = abs(target - validation_count)
        proposed_error = abs(target - (validation_count + size))
        split = "validation" if proposed_error <= current_error else "discovery"
        if split == "validation":
            validation_count += size
        for row in grouped[key]:
            row["split"] = split


def select_pop_candidates(
    source_root: Path,
    output: Path,
    *,
    max_per_artist: int = 2,
    validation_fraction: float = 0.25,
    seed: int = 20260716,
    excluded_source_ids: Sequence[str] = (),
    target_count: int = 0,
) -> list[dict[str, str]]:
    """Select high-confidence instrumental Pop tracks from MTG-Jamendo."""

    if max_per_artist < 1:
        raise ValueError("max_per_artist must be positive")
    raw = _read_mtg_tracks(source_root / "data/raw_30s_cleantags_50artists.tsv")
    instrumental = _read_unanimous_instrumental(
        source_root
        / "derived/music-classification-annotations/music-classification-annotations-clean.tsv"
    )
    names = _read_mtg_names(source_root / "data/raw.meta.tsv")
    licenses = parse_jamendo_licenses(source_root / "audio_licenses.txt")

    excluded = {str(int(source_id)) for source_id in excluded_source_ids}
    previous_rows = _read_candidate_rows(output) if output.exists() else []
    previous_by_id = {row["track_id"]: row for row in previous_rows}
    previous_splits: dict[str, str] = {}
    for row in previous_rows:
        previous_splits.setdefault(row["artist_key"], row["split"])

    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track_id in instrumental:
        track = raw.get(track_id)
        source_id = str(int(track_id.split("_")[-1]))
        if (
            source_id not in excluded
            and track
            and "genre---pop" in track["tags"]
            and track["path"] in licenses
        ):
            by_artist[track["artist_id"]].append(track)

    ranked_by_artist: dict[str, list[dict[str, Any]]] = {}
    for artist_id, artist_tracks in by_artist.items():
        ranked_by_artist[artist_id] = sorted(
            artist_tracks,
            key=lambda row: _stable_key(seed, row["track_id"]),
        )[:max_per_artist]

    capacity = sum(len(tracks) for tracks in ranked_by_artist.values())
    if target_count and capacity < target_count:
        raise ControlDatasetError(
            f"Pop selection capacity is {capacity}, below target_count={target_count}"
        )
    selected: list[dict[str, Any]] = []
    for rank in range(max_per_artist):
        layer = [tracks[rank] for tracks in ranked_by_artist.values() if len(tracks) > rank]
        layer.sort(key=lambda row: _stable_key(seed, row["track_id"]))
        if target_count:
            layer = layer[: max(0, target_count - len(selected))]
        selected.extend(layer)
        if target_count and len(selected) >= target_count:
            break

    rows: list[dict[str, str]] = []
    for track in sorted(selected, key=lambda row: row["track_id"]):
        source_id = str(int(track["track_id"].split("_")[-1]))
        record = names.get(track["track_id"], {})
        license_ = licenses[track["path"]]
        local_id = f"pop_mtg_{int(source_id):07d}"
        rows.append(
            {
                "track_id": local_id,
                "group": "pop",
                "source_dataset": "mtg-jamendo-dataset",
                "source_track_id": source_id,
                "title": record.get("TRACK_NAME", ""),
                "creator": record.get("ARTIST_NAME", ""),
                "artist_key": track["artist_id"],
                "album_key": track["album_id"],
                "composer_key": "",
                "subpool": "instrumental_pop",
                "duration_seconds": track["duration"],
                "source_url": license_.source_url,
                "download_url": JAMENDO_FILE_API.format(
                    client_id="{client_id}", track_id=source_id
                ),
                "license_type": license_.license_text,
                "license_url": license_.license_url,
                "instrumental_evidence": "MTG unanimous human voice_instrumental annotation",
                "split": "",
                "relative_path": f"pop_music/{local_id}.mp3",
                "sha256": "",
                "download_status": "pending",
                "downloaded_at": "",
                "error": "",
            }
        )
    _assign_grouped_split(
        rows,
        group_field="artist_key",
        validation_fraction=validation_fraction,
        seed=seed,
    )
    preserve_splits = len(previous_rows) == len(rows)
    for row in rows:
        if preserve_splits and row["artist_key"] in previous_splits:
            row["split"] = previous_splits[row["artist_key"]]
        previous = previous_by_id.get(row["track_id"])
        if previous and previous["download_status"] == "verified":
            for field in ("duration_seconds", "sha256", "download_status", "downloaded_at"):
                row[field] = previous[field]
    _write_csv_atomic(output, CANDIDATE_COLUMNS, rows)
    return rows


def read_pop_exclusions(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["source_track_id"] for row in csv.DictReader(handle)]


def preflight_pop_availability(
    manifest: Path,
    exclusions_path: Path,
    *,
    client_id: str,
) -> dict[str, int]:
    """Record Jamendo candidates that are currently absent or download-disabled."""

    if not client_id:
        raise ControlDatasetError("JAMENDO_CLIENT_ID is required for Pop preflight")
    rows = _read_candidate_rows(manifest)
    availability: dict[str, bool] = {}
    batch_size = 50
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        params: list[tuple[str, str]] = [
            ("client_id", client_id),
            ("format", "json"),
            ("limit", str(batch_size)),
            ("type", "single albumtrack"),
            ("audioformat", "mp32"),
        ]
        params.extend(("id[]", row["source_track_id"]) for row in batch)
        url = f"{JAMENDO_TRACKS_API}?{urllib.parse.urlencode(params)}"
        with _request(url) as response:
            payload = json.load(response)
        headers = payload.get("headers", {})
        if headers.get("status") != "success":
            raise ControlDatasetError(
                f"Jamendo preflight failed: {headers.get('error_message', 'unknown error')}"
            )
        availability.update(
            {
                str(result["id"]): bool(result.get("audiodownload_allowed"))
                for result in payload.get("results", [])
            }
        )

    existing: dict[str, dict[str, str]] = {}
    if exclusions_path.exists():
        with exclusions_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = {row["source_track_id"]: row for row in csv.DictReader(handle)}
    denied = 0
    absent = 0
    for row in rows:
        source_id = row["source_track_id"]
        if source_id not in availability:
            reason = "jamendo_tracks_api_not_returned"
            absent += 1
        elif not availability[source_id]:
            reason = "jamendo_tracks_api_audiodownload_disallowed"
            denied += 1
        else:
            continue
        existing[source_id] = {
            "source_track_id": source_id,
            "reason": reason,
            "recorded_at": date.today().isoformat(),
        }
    _write_csv_atomic(
        exclusions_path,
        ["source_track_id", "reason", "recorded_at"],
        sorted(existing.values(), key=lambda row: int(row["source_track_id"])),
    )
    return {
        "checked": len(rows),
        "available": len(rows) - denied - absent,
        "denied": denied,
        "not_returned": absent,
        "exclusions_total": len(existing),
    }


def _creative_commons_name(url: str) -> str:
    parts = urllib.parse.urlparse(url).path.strip("/").split("/")
    code = (
        parts[1].lower()
        if len(parts) > 1 and parts[0].lower() == "licenses"
        else parts[0].lower()
    )
    names = {
        "by": "Creative Commons Attribution license",
        "by-sa": "Creative Commons Attribution-Share-Alike license",
        "by-nd": "Creative Commons Attribution-No Derivatives license",
        "by-nc": "Creative Commons Attribution-Non-Commercial license",
        "by-nc-sa": "Creative Commons Attribution-Non-Commercial-Share-Alike license",
        "by-nc-nd": "Creative Commons Attribution-Non-Commercial-No Derivatives license",
    }
    return names.get(code, f"Creative Commons {code.upper()} license")


def normalize_pop_license_types(manifest: Path) -> int:
    rows = _read_candidate_rows(manifest)
    updated = 0
    for row in rows:
        if row["group"] != "pop" or not row["license_url"]:
            continue
        normalized = _creative_commons_name(row["license_url"])
        if row["license_type"] != normalized:
            row["license_type"] = normalized
            updated += 1
    _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
    return updated


def supplement_pop_from_jamendo(
    manifest: Path,
    *,
    client_id: str,
    excluded_source_ids: Sequence[str] = (),
    target_total: int = 300,
    max_per_artist: int = 4,
    max_pages: int = 5,
    validation_fraction: float = 0.25,
    seed: int = 20260716,
) -> list[dict[str, str]]:
    """Fill the strict MTG pool with current downloadable Jamendo instrumental Pop."""

    if not client_id:
        raise ControlDatasetError("JAMENDO_CLIENT_ID is required for Jamendo supplementation")
    existing = _read_candidate_rows(manifest)
    existing_by_id = {row["track_id"]: row for row in existing}
    base = [row for row in existing if row["source_dataset"] != "jamendo-api-instrumental-pop"]
    additions_needed = target_total - len(base)
    if additions_needed < 0:
        raise ControlDatasetError(
            f"strict Pop pool has {len(base)} tracks, above target_total={target_total}"
        )
    excluded = {str(int(source_id)) for source_id in excluded_source_ids}
    existing_source_ids = {row["source_track_id"] for row in base}
    candidates: dict[str, dict[str, Any]] = {}
    page_size = 200
    for page in range(max_pages):
        params = {
            "client_id": client_id,
            "format": "json",
            "limit": str(page_size),
            "offset": str(page * page_size),
            "order": "id_asc",
            "type": "single albumtrack",
            "tags": "pop",
            "vocalinstrumental": "instrumental",
            "include": "musicinfo",
            "audioformat": "mp32",
        }
        url = f"{JAMENDO_TRACKS_API}?{urllib.parse.urlencode(params)}"
        payload: dict[str, Any] = {}
        for attempt in range(3):
            with _request(url) as response:
                payload = json.load(response)
            if payload.get("results") or page > 0:
                break
            time.sleep(5 * (attempt + 1))
        headers = payload.get("headers", {})
        if headers.get("status") != "success":
            raise ControlDatasetError(
                f"Jamendo supplement query failed: {headers.get('error_message', 'unknown error')}"
            )
        results = payload.get("results", [])
        if page == 0 and not results:
            raise ControlDatasetError(
                "Jamendo supplement query returned an unexpected empty first page"
            )
        for result in results:
            source_id = str(result["id"])
            license_url = result.get("license_ccurl", "")
            if (
                source_id in excluded
                or source_id in existing_source_ids
                or not result.get("audiodownload_allowed")
                or not license_url
            ):
                continue
            candidates[source_id] = result
        if len(results) < page_size:
            break
        time.sleep(2)

    base_counts = Counter(row["artist_key"] for row in base)
    by_artist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in candidates.values():
        artist_key = f"artist_{int(result['artist_id']):06d}"
        if base_counts[artist_key] < max_per_artist:
            by_artist[artist_key].append(result)
    ranked_by_artist = {
        artist: sorted(
            results,
            key=lambda result: _stable_key(seed, str(result["id"])),
        )[: max(0, max_per_artist - base_counts[artist])]
        for artist, results in by_artist.items()
    }
    capacity = sum(len(results) for results in ranked_by_artist.values())
    if capacity < additions_needed:
        raise ControlDatasetError(
            f"Jamendo supplement capacity is {capacity}, below required {additions_needed}"
        )
    selected: list[dict[str, Any]] = []
    for rank in range(max_per_artist):
        layer = [
            results[rank]
            for results in ranked_by_artist.values()
            if len(results) > rank
        ]
        layer.sort(key=lambda result: _stable_key(seed, str(result["id"])))
        layer = layer[: max(0, additions_needed - len(selected))]
        selected.extend(layer)
        if len(selected) >= additions_needed:
            break

    additions: list[dict[str, str]] = []
    for result in sorted(selected, key=lambda item: int(item["id"])):
        source_id = str(result["id"])
        local_id = f"pop_jamendo_{int(source_id):07d}"
        artist_key = f"artist_{int(result['artist_id']):06d}"
        album_id = str(result.get("album_id") or "")
        album_key = (
            f"album_{int(album_id):06d}" if album_id else f"single_{int(source_id):07d}"
        )
        license_url = str(result["license_ccurl"])
        row = {
            "track_id": local_id,
            "group": "pop",
            "source_dataset": "jamendo-api-instrumental-pop",
            "source_track_id": source_id,
            "title": str(result.get("name", "")),
            "creator": str(result.get("artist_name", "")),
            "artist_key": artist_key,
            "album_key": album_key,
            "composer_key": "",
            "subpool": "instrumental_pop_api_supplement",
            "duration_seconds": str(result.get("duration", "")),
            "source_url": str(result.get("shareurl", "")),
            "download_url": JAMENDO_FILE_API.format(
                client_id="{client_id}",
                track_id=source_id,
            ),
            "license_type": _creative_commons_name(license_url),
            "license_url": license_url,
            "instrumental_evidence": (
                "Jamendo API filters: tags=pop; vocalinstrumental=instrumental; "
                "audiodownload_allowed=true"
            ),
            "split": "",
            "relative_path": f"pop_music/{local_id}.mp3",
            "sha256": "",
            "download_status": "pending",
            "downloaded_at": "",
            "error": "",
        }
        previous = existing_by_id.get(local_id)
        if previous and previous["download_status"] == "verified":
            for field in ("duration_seconds", "sha256", "download_status", "downloaded_at"):
                row[field] = previous[field]
        additions.append(row)
    rows = base + additions
    _assign_grouped_split(
        rows,
        group_field="artist_key",
        validation_fraction=validation_fraction,
        seed=seed,
    )
    _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
    return rows


def _classical_identity(member: str) -> tuple[str, str, str] | None:
    parts = PurePosixPath(member).parts
    if "__MACOSX" in parts or not member.lower().endswith(".m4a"):
        return None
    if "Goldberg Variations" in parts:
        return "bach", "goldberg_variations_bwv_988", "piano_solo"
    if "Schubert - The Piano Sonatas" in parts:
        title = PurePosixPath(member).stem
        work_match = re.search(r"D\.\s*(\d+)", title)
        work = f"schubert_sonata_d_{work_match.group(1)}" if work_match else "schubert_sonatas"
        return "schubert", work, "piano_solo"
    if "String Quartets" not in parts:
        return None
    quartet_index = parts.index("String Quartets")
    if quartet_index + 1 >= len(parts):
        return None
    work_folder = parts[quartet_index + 1]
    composer_aliases = {
        "beethoven": "beethoven",
        "borodin": "borodin",
        "dvorak": "dvorak",
        "haydn": "haydn",
        "mendelssohn": "mendelssohn",
        "mozart": "mozart",
        "suk": "suk",
    }
    lowered = work_folder.lower()
    composer = next((value for key, value in composer_aliases.items() if key in lowered), None)
    if composer is None:
        return None
    work = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return composer, work, "string_quartet"


def _classical_subset_score(
    composers: tuple[str, ...],
    grouped: dict[str, list[dict[str, str]]],
    target: int,
    all_subpools: Counter[str],
    seed: int,
) -> tuple[float, float, int]:
    members = list(itertools.chain.from_iterable(grouped[key] for key in composers))
    count_error = abs(len(members) - target)
    subset_counts = Counter(row["subpool"] for row in members)
    total = max(1, len(members))
    population = max(1, sum(all_subpools.values()))
    distribution_error = sum(
        (subset_counts[key] / total - all_subpools[key] / population) ** 2
        for key in all_subpools
    )
    tie_key = _stable_key(seed, "|".join(sorted(composers)))
    return float(count_error), float(distribution_error), tie_key


def _best_classical_subset(
    grouped: dict[str, list[dict[str, str]]],
    target: int,
    all_subpools: Counter[str],
    seed: int,
    *,
    require_piano_and_chamber: bool,
) -> set[str]:
    if target <= 0:
        return set()
    keys = sorted(grouped)
    candidates: list[
        tuple[tuple[float, float, float, int], tuple[str, ...]]
    ] = []
    for size in range(1, len(keys) + 1):
        for composers in itertools.combinations(keys, size):
            members = list(
                itertools.chain.from_iterable(grouped[key] for key in composers)
            )
            subpools = {row["subpool"] for row in members}
            has_piano = "piano_solo" in subpools
            has_chamber = bool(
                subpools
                & {"string_quartet", "string_chamber", "mixed_chamber"}
            )
            coverage_penalty = int(
                require_piano_and_chamber and not (has_piano and has_chamber)
            )
            base = _classical_subset_score(
                composers, grouped, target, all_subpools, seed
            )
            score = (base[0], float(coverage_penalty), base[1], base[2])
            candidates.append((score, composers))
    if not candidates:
        raise ControlDatasetError("classical split assignment has no composer groups")
    return set(min(candidates, key=lambda item: item[0])[1])


def _assign_classical_split(
    rows: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
    holdout_fraction: float = 0.15,
) -> None:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if validation_fraction + holdout_fraction >= 1.0:
        raise ValueError("validation and holdout fractions must sum to less than 1")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["composer_key"]].append(row)
    all_subpools = Counter(row["subpool"] for row in rows)
    if "piano_solo" not in all_subpools or not (
        set(all_subpools) & {"string_quartet", "string_chamber", "mixed_chamber"}
    ):
        raise ControlDatasetError("classical pool must contain piano and chamber composers")
    validation_target = round(len(rows) * validation_fraction)
    holdout_target = round(len(rows) * holdout_fraction)
    validation = _best_classical_subset(
        grouped,
        validation_target,
        all_subpools,
        seed,
        require_piano_and_chamber=True,
    )
    remaining = {key: value for key, value in grouped.items() if key not in validation}
    holdout = _best_classical_subset(
        remaining,
        holdout_target,
        all_subpools,
        seed + 1,
        require_piano_and_chamber=False,
    )
    for composer, items in grouped.items():
        split = (
            "validation"
            if composer in validation
            else "holdout"
            if composer in holdout
            else "discovery"
        )
        for row in items:
            row["split"] = split


def catalog_classical_candidates(
    output: Path,
    *,
    validation_fraction: float = 0.20,
    holdout_fraction: float = 0.15,
    seed: int = 20260716,
    html: str | None = None,
) -> list[dict[str, str]]:
    """Catalog Musopen's own PDM recordings mirrored by Internet Archive."""

    parser = _LinkParser()
    parser.feed(html if html is not None else _fetch_text(MUSOPEN_ARCHIVE_VIEW))
    direct_links: dict[str, str] = {}
    for link in parser.links:
        absolute = urllib.parse.urljoin(MUSOPEN_ARCHIVE_VIEW, link)
        decoded = urllib.parse.unquote(absolute)
        marker = "Musopen-Lossless-DVD.zip/"
        if marker not in decoded:
            continue
        member = decoded.split(marker, 1)[1]
        if _classical_identity(member):
            direct_links[member] = absolute

    rows: list[dict[str, str]] = []
    for member, url in sorted(direct_links.items()):
        identity = _classical_identity(member)
        if identity is None:
            continue
        composer, work, subpool = identity
        digest = hashlib.sha1(member.encode()).hexdigest()[:12]  # noqa: S324 - stable ID only.
        local_id = f"classical_musopen_{digest}"
        rows.append(
            {
                "track_id": local_id,
                "group": "classical",
                "source_dataset": "musopen-lossless-dvd",
                "source_track_id": member,
                "title": PurePosixPath(member).stem,
                "creator": "Musopen Kickstarter Project performers",
                "artist_key": "",
                "album_key": work,
                "composer_key": composer,
                "subpool": subpool,
                "duration_seconds": "",
                "source_url": MUSOPEN_ITEM_PAGE,
                "download_url": url,
                "license_type": "Public Domain Mark 1.0",
                "license_url": "https://creativecommons.org/publicdomain/mark/1.0/",
                "instrumental_evidence": f"curated Musopen {subpool}",
                "split": "",
                "relative_path": f"classical_music/{local_id}.m4a",
                "sha256": "",
                "download_status": "pending",
                "downloaded_at": "",
                "error": "",
            }
        )
    if not rows:
        raise ControlDatasetError("no eligible Musopen recordings found in archive catalog")
    _assign_classical_split(rows, validation_fraction, seed, holdout_fraction)
    _write_csv_atomic(output, CANDIDATE_COLUMNS, rows)
    return rows


_MUSICNET_OVERLAP_WORKS = {
    ("beethoven", "OP18NO6"),
    ("dvorak", "OP51"),
    ("dvorak", "OP96"),
    ("mozart", "K421"),
    ("mozart", "K465"),
    ("schubert", "D568"),
    ("schubert", "D664"),
    ("schubert", "D784"),
    ("schubert", "D845"),
    ("schubert", "D850"),
    ("schubert", "D958"),
    ("schubert", "D959"),
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _musicnet_subpool(ensemble: str) -> str:
    if ensemble == "Solo Piano":
        return "piano_solo"
    if ensemble in {"String Quartet", "String Sextet", "Viola Quintet"}:
        return "string_chamber"
    if ensemble.startswith("Solo "):
        return "solo_instrument"
    return "mixed_chamber"


def extend_classical_with_musicnet(
    manifest: Path,
    metadata_path: Path,
    *,
    target_total: int = 300,
    max_per_composer: int = 71,
    validation_fraction: float = 0.20,
    holdout_fraction: float = 0.15,
    seed: int = 20260716,
) -> list[dict[str, str]]:
    """Extend the curated Musopen pool with non-overlapping MusicNet recordings."""

    if max_per_composer < 1:
        raise ValueError("max_per_composer must be positive")
    existing = _read_candidate_rows(manifest)
    existing_by_id = {row["track_id"]: row for row in existing}
    base = [row for row in existing if row["source_dataset"] != "musicnet-zenodo-5120004"]
    additions_needed = target_total - len(base)
    if additions_needed < 0:
        raise ControlDatasetError(
            f"base Classical pool has {len(base)} tracks, above target_total={target_total}"
        )

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    by_composer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in metadata_rows:
        composer = _slug(record["composer"])
        work = (composer, record["catalog_name"].upper())
        if record["source"] == "Museopen" or work in _MUSICNET_OVERLAP_WORKS:
            continue
        by_composer[composer].append(record)
    ranked_by_composer = {
        composer: sorted(
            records,
            key=lambda record: _stable_key(seed, record["id"]),
        )[:max_per_composer]
        for composer, records in by_composer.items()
    }
    capacity = sum(len(records) for records in ranked_by_composer.values())
    if capacity < additions_needed:
        raise ControlDatasetError(
            f"MusicNet selection capacity is {capacity}, below required {additions_needed}"
        )
    selected: list[dict[str, str]] = []
    for rank in range(max_per_composer):
        layer = [
            records[rank]
            for records in ranked_by_composer.values()
            if len(records) > rank
        ]
        layer.sort(key=lambda record: _stable_key(seed, record["id"]))
        layer = layer[: max(0, additions_needed - len(selected))]
        selected.extend(layer)
        if len(selected) >= additions_needed:
            break

    additions: list[dict[str, str]] = []
    for record in sorted(selected, key=lambda item: int(item["id"])):
        composer = _slug(record["composer"])
        local_id = f"classical_musicnet_{int(record['id']):04d}"
        title = f"{record['composition']} — {record['movement']}"
        row = {
            "track_id": local_id,
            "group": "classical",
            "source_dataset": "musicnet-zenodo-5120004",
            "source_track_id": record["id"],
            "title": title,
            "creator": record["source"],
            "artist_key": "",
            "album_key": f"musicnet_{composer}_{_slug(record['catalog_name'])}",
            "composer_key": composer,
            "subpool": _musicnet_subpool(record["ensemble"]),
            "duration_seconds": record["seconds"],
            "source_url": MUSICNET_RECORD_PAGE,
            "download_url": MUSICNET_ARCHIVE_URL,
            "license_type": "Creative Commons Attribution 4.0 International",
            "license_url": MUSICNET_LICENSE_URL,
            "instrumental_evidence": f"MusicNet classical metadata: {record['ensemble']}",
            "split": "",
            "relative_path": f"classical_music/{local_id}.wav",
            "sha256": "",
            "download_status": "pending",
            "downloaded_at": "",
            "error": "",
        }
        previous = existing_by_id.get(local_id)
        if previous and previous["download_status"] == "verified":
            for field in ("duration_seconds", "sha256", "download_status", "downloaded_at"):
                row[field] = previous[field]
        additions.append(row)

    rows = base + additions
    _assign_classical_split(rows, validation_fraction, seed, holdout_fraction)
    _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
    return rows


def apply_classical_standard_fallback(manifest: Path, *, html: str | None = None) -> int:
    """Point failed lossless members at the same PDM project's standard MP3 DVD."""

    parser = _LinkParser()
    parser.feed(html if html is not None else _fetch_text(MUSOPEN_STANDARD_VIEW))
    by_title: dict[str, tuple[str, str]] = {}
    for link in parser.links:
        absolute = urllib.parse.urljoin(MUSOPEN_STANDARD_VIEW, link)
        decoded = urllib.parse.unquote(absolute)
        marker = "Musopen-DVD.zip/"
        if marker not in decoded or not decoded.lower().endswith(".mp3"):
            continue
        member = decoded.split(marker, 1)[1]
        by_title[PurePosixPath(member).stem] = (member, absolute)

    rows = _read_candidate_rows(manifest)
    updated = 0
    for row in rows:
        fallback = by_title.get(row["title"])
        if row["group"] != "classical" or row["download_status"] != "failed" or not fallback:
            continue
        member, url = fallback
        row["source_dataset"] = "musopen-dvd-standard-fallback"
        row["source_track_id"] = member
        row["source_url"] = MUSOPEN_STANDARD_ITEM_PAGE
        row["download_url"] = url
        row["relative_path"] = str(PurePosixPath(row["relative_path"]).with_suffix(".mp3"))
        row["download_status"] = "pending"
        row["error"] = ""
        row["instrumental_evidence"] += "; standard-DVD fallback for unavailable lossless member"
        updated += 1
    _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
    return updated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published archive integrity checksum.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_m4a_duration(path: Path) -> float | None:
    data = path.read_bytes()
    marker = data.find(b"mvhd")
    if marker < 0 or marker + 32 > len(data):
        return None
    version = data[marker + 4]
    if version == 0:
        timescale = struct.unpack(">I", data[marker + 16 : marker + 20])[0]
        duration = struct.unpack(">I", data[marker + 20 : marker + 24])[0]
    elif version == 1:
        timescale = struct.unpack(">I", data[marker + 24 : marker + 28])[0]
        duration = struct.unpack(">Q", data[marker + 28 : marker + 36])[0]
    else:
        return None
    return duration / timescale if timescale else None


def _probe_wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (EOFError, wave.Error):
        return None


def _looks_like_audio(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(32)
    suffix = path.suffix.lower()
    if suffix == ".m4a":
        return b"ftyp" in head
    if suffix == ".mp3":
        return head.startswith(b"ID3") or (
            len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0
        )
    if suffix == ".wav":
        return head.startswith(b"RIFF") and head[8:12] == b"WAVE"
    return False


def _download_one(row: dict[str, str], data_root: Path, client_id: str | None) -> dict[str, str]:
    result = dict(row)
    target = data_root / PurePosixPath(row["relative_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _looks_like_audio(target):
        result["sha256"] = _sha256(target)
        result["download_status"] = "verified"
        result["downloaded_at"] = result["downloaded_at"] or date.today().isoformat()
        result["error"] = ""
        if target.suffix.lower() == ".m4a" and not result["duration_seconds"]:
            duration = _probe_m4a_duration(target)
            result["duration_seconds"] = f"{duration:.3f}" if duration else ""
        return result

    url = row["download_url"]
    if "{client_id}" in url:
        if not client_id:
            raise ControlDatasetError("JAMENDO_CLIENT_ID is required for Pop downloads")
        url = url.format(client_id=urllib.parse.quote(client_id, safe=""))
    partial = target.with_suffix(target.suffix + ".part")
    last_error = ""
    for attempt in range(3):
        try:
            headers: dict[str, str] = {}
            existing = partial.stat().st_size if partial.exists() else 0
            if existing:
                headers["Range"] = f"bytes={existing}-"
            with _request(url, headers=headers) as response:
                append = existing > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size == 0:
                raise ControlDatasetError("server returned an empty file")
            os.replace(partial, target)
            if not _looks_like_audio(target):
                preview = ""
                if target.stat().st_size <= 4096:
                    preview = target.read_text(encoding="utf-8", errors="replace").strip()
                target.unlink(missing_ok=True)
                detail = f": {preview[:300]}" if preview else ""
                raise ControlDatasetError(
                    f"downloaded response is not recognized audio{detail}"
                )
            result["sha256"] = _sha256(target)
            result["download_status"] = "verified"
            result["downloaded_at"] = date.today().isoformat()
            result["error"] = ""
            if target.suffix.lower() == ".m4a":
                duration = _probe_m4a_duration(target)
                result["duration_seconds"] = f"{duration:.3f}" if duration else ""
            return result
        except (OSError, urllib.error.URLError, ControlDatasetError) as exc:
            last_error = str(exc)
            time.sleep(2**attempt)
    result["download_status"] = "failed"
    result["error"] = last_error[:500]
    return result


def download_candidates(
    manifest: Path,
    data_root: Path,
    *,
    client_id: str | None = None,
    workers: int = 4,
    limit: int = 0,
    force_track_ids: Sequence[str] = (),
) -> Counter[str]:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    rows = _read_candidate_rows(manifest)
    forced = set(force_track_ids)
    known_ids = {row["track_id"] for row in rows}
    unknown_ids = forced - known_ids
    if unknown_ids:
        raise ControlDatasetError(f"unknown track ids: {', '.join(sorted(unknown_ids))}")
    if forced:
        pending_indices = [index for index, row in enumerate(rows) if row["track_id"] in forced]
        for index in pending_indices:
            target = data_root / PurePosixPath(rows[index]["relative_path"])
            target.unlink(missing_ok=True)
            target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)
    else:
        pending_indices = [
            index for index, row in enumerate(rows) if row["download_status"] != "verified"
        ]
    if limit > 0:
        pending_indices = pending_indices[:limit]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, rows[index], data_root, client_id): index
            for index in pending_indices
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                rows[index] = future.result()
            except Exception as exc:  # Keep the ledger usable after an individual failure.
                rows[index]["download_status"] = "failed"
                rows[index]["error"] = str(exc)[:500]
            completed += 1
            print(
                f"[{completed}/{len(futures)}] {rows[index]['track_id']}: "
                f"{rows[index]['download_status']}",
                flush=True,
            )
            _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
    return Counter(row["download_status"] for row in rows)


def extract_musicnet_archive(
    archive: Path,
    manifest: Path,
    data_root: Path,
    *,
    expected_md5: str = MUSICNET_ARCHIVE_MD5,
) -> Counter[str]:
    """Extract only selected MusicNet WAV members from the official tar archive."""

    if not archive.is_file():
        raise ControlDatasetError(f"MusicNet archive does not exist: {archive}")
    actual_md5 = _md5(archive)
    if expected_md5 and actual_md5.lower() != expected_md5.lower():
        raise ControlDatasetError(
            f"MusicNet archive MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    rows = _read_candidate_rows(manifest)
    wanted: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row["source_dataset"] != "musicnet-zenodo-5120004":
            continue
        target = data_root / PurePosixPath(row["relative_path"])
        if target.is_file() and _looks_like_audio(target):
            row["sha256"] = _sha256(target)
            row["download_status"] = "verified"
            row["downloaded_at"] = row["downloaded_at"] or date.today().isoformat()
            row["error"] = ""
            duration = _probe_wav_duration(target)
            row["duration_seconds"] = f"{duration:.3f}" if duration else row["duration_seconds"]
        else:
            wanted[row["source_track_id"]] = index
    if not wanted:
        _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
        return Counter(row["download_status"] for row in rows)

    completed = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            source_id = PurePosixPath(member.name).stem
            index = wanted.get(source_id)
            if index is None:
                continue
            target = data_root / PurePosixPath(rows[index]["relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            source = handle.extractfile(member)
            if source is None:
                continue
            with source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.replace(partial, target)
            if not _looks_like_audio(target):
                target.unlink(missing_ok=True)
                rows[index]["download_status"] = "failed"
                rows[index]["error"] = "archive member is not recognized PCM WAV audio"
            else:
                rows[index]["sha256"] = _sha256(target)
                rows[index]["download_status"] = "verified"
                rows[index]["downloaded_at"] = date.today().isoformat()
                rows[index]["error"] = ""
                duration = _probe_wav_duration(target)
                rows[index]["duration_seconds"] = (
                    f"{duration:.3f}" if duration else rows[index]["duration_seconds"]
                )
            completed += 1
            wanted.pop(source_id, None)
            print(
                f"[{completed}/{completed + len(wanted)}] {rows[index]['track_id']}: "
                f"{rows[index]['download_status']}",
                flush=True,
            )
            _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
            if not wanted:
                break
    for source_id, index in wanted.items():
        rows[index]["download_status"] = "failed"
        rows[index]["error"] = f"MusicNet archive member not found for id {source_id}"
    _write_csv_atomic(manifest, CANDIDATE_COLUMNS, rows)
    return Counter(row["download_status"] for row in rows)


def _read_existing_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ControlDatasetError(f"canonical metadata file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def finalize_metadata(manifests: Sequence[Path], metadata_dir: Path) -> dict[str, int]:
    candidates = list(
        itertools.chain.from_iterable(_read_candidate_rows(path) for path in manifests)
    )
    verified = [row for row in candidates if row["download_status"] == "verified"]
    control_ids = {row["track_id"] for row in candidates}

    track_columns, existing_tracks = _read_existing_csv(metadata_dir / "track_index.csv")
    license_columns, existing_licenses = _read_existing_csv(metadata_dir / "licenses.csv")
    existing_tracks = [row for row in existing_tracks if row.get("track_id") not in control_ids]
    existing_licenses = [row for row in existing_licenses if row.get("track_id") not in control_ids]

    for row in verified:
        existing_tracks.append(
            {
                "track_id": row["track_id"],
                "group": row["group"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "duration_seconds": row["duration_seconds"],
                "sample_rate": "",
                "channels": "",
                "artist_key": row["artist_key"],
                "album_key": row["album_key"],
                "composer_key": row["composer_key"],
                "instrumental": "true",
                "restricted": "false",
            }
        )
        existing_licenses.append(
            {
                "track_id": row["track_id"],
                "group": row["group"],
                "source_url": row["source_url"],
                "license_type": row["license_type"],
                "downloaded_at": row["downloaded_at"],
                "redistribution_allowed": "true" if row["group"] == "classical" else "false",
                "notes": (
                    f"{row['source_dataset']}; {row['instrumental_evidence']}; "
                    f"license: {row['license_url']}"
                ),
            }
        )

    _write_csv_atomic(
        metadata_dir / "track_index.csv",
        track_columns,
        sorted(existing_tracks, key=lambda row: row["track_id"]),
    )
    _write_csv_atomic(
        metadata_dir / "licenses.csv",
        license_columns,
        sorted(existing_licenses, key=lambda row: row["track_id"]),
    )

    for split in ("discovery", "validation", "holdout"):
        path = metadata_dir / f"split_{split}.csv"
        columns, existing = _read_existing_csv(path)
        existing = [row for row in existing if row.get("track_id") not in control_ids]
        existing.extend(
            {"track_id": row["track_id"], "group": row["group"]}
            for row in verified
            if row["split"] == split
        )
        _write_csv_atomic(path, columns, sorted(existing, key=lambda row: row["track_id"]))
    return dict(Counter(row["group"] for row in verified))


def audit_candidates(
    manifests: Sequence[Path], data_root: Path, *, verify_hash: bool
) -> dict[str, Any]:
    rows = list(itertools.chain.from_iterable(_read_candidate_rows(path) for path in manifests))
    errors: list[str] = []
    warnings: list[str] = []
    seen_hashes: dict[str, str] = {}
    for row in rows:
        target = data_root / PurePosixPath(row["relative_path"])
        if row["download_status"] == "verified":
            if not target.is_file():
                errors.append(f"missing verified file: {row['track_id']}")
                continue
            digest = _sha256(target) if verify_hash else row["sha256"]
            if verify_hash and digest != row["sha256"]:
                errors.append(f"checksum mismatch: {row['track_id']}")
            if digest in seen_hashes:
                errors.append(f"duplicate audio: {seen_hashes[digest]} and {row['track_id']}")
            seen_hashes[digest] = row["track_id"]
        elif row["download_status"] == "failed":
            warnings.append(f"download failed: {row['track_id']}: {row['error']}")

    for field in ("artist_key", "album_key", "composer_key"):
        splits_by_key: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row[field]:
                splits_by_key[row[field]].add(row["split"])
        for key, splits in splits_by_key.items():
            if len(splits) > 1:
                errors.append(f"split leakage: {field}={key!r} appears in {sorted(splits)}")

    return {
        "ok": not errors,
        "candidate_count": len(rows),
        "by_group": dict(Counter(row["group"] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_status": dict(Counter(row["download_status"] for row in rows)),
        "errors": errors,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-controls")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pop = subparsers.add_parser("select-pop")
    pop.add_argument("--source-root", type=Path, default=Path("data_sources/mtg-jamendo-dataset"))
    pop.add_argument("--output", type=Path, default=Path("metadata/control_pop.csv"))
    pop.add_argument("--max-per-artist", type=int, default=2)
    pop.add_argument("--target-count", type=int, default=0)
    pop.add_argument("--validation-fraction", type=float, default=0.25)
    pop.add_argument("--seed", type=int, default=20260716)
    pop.add_argument(
        "--exclude-list",
        type=Path,
        default=Path("metadata/control_pop_exclusions.csv"),
    )

    preflight = subparsers.add_parser("preflight-pop")
    preflight.add_argument(
        "manifest", type=Path, default=Path("metadata/control_pop.csv"), nargs="?"
    )
    preflight.add_argument(
        "--exclude-list",
        type=Path,
        default=Path("metadata/control_pop_exclusions.csv"),
    )
    preflight.add_argument("--jamendo-client-id-env", default="JAMENDO_CLIENT_ID")

    supplement = subparsers.add_parser("supplement-pop-jamendo")
    supplement.add_argument(
        "manifest", type=Path, default=Path("metadata/control_pop.csv"), nargs="?"
    )
    supplement.add_argument(
        "--exclude-list",
        type=Path,
        default=Path("metadata/control_pop_exclusions.csv"),
    )
    supplement.add_argument("--target-total", type=int, default=300)
    supplement.add_argument("--max-per-artist", type=int, default=4)
    supplement.add_argument("--max-pages", type=int, default=5)
    supplement.add_argument("--validation-fraction", type=float, default=0.25)
    supplement.add_argument("--seed", type=int, default=20260716)
    supplement.add_argument("--jamendo-client-id-env", default="JAMENDO_CLIENT_ID")

    normalize_pop = subparsers.add_parser("normalize-pop-licenses")
    normalize_pop.add_argument(
        "manifest", type=Path, default=Path("metadata/control_pop.csv"), nargs="?"
    )

    classical = subparsers.add_parser("catalog-classical")
    classical.add_argument("--output", type=Path, default=Path("metadata/control_classical.csv"))
    classical.add_argument("--validation-fraction", type=float, default=0.20)
    classical.add_argument("--holdout-fraction", type=float, default=0.15)
    classical.add_argument("--seed", type=int, default=20260716)

    musicnet = subparsers.add_parser("extend-classical-musicnet")
    musicnet.add_argument(
        "manifest",
        type=Path,
        default=Path("metadata/control_classical.csv"),
        nargs="?",
    )
    musicnet.add_argument(
        "--metadata",
        type=Path,
        default=Path("data_sources/musicnet/musicnet_metadata.csv"),
    )
    musicnet.add_argument("--target-total", type=int, default=300)
    musicnet.add_argument("--max-per-composer", type=int, default=71)
    musicnet.add_argument("--validation-fraction", type=float, default=0.20)
    musicnet.add_argument("--holdout-fraction", type=float, default=0.15)
    musicnet.add_argument("--seed", type=int, default=20260716)

    fetch_musicnet = subparsers.add_parser("fetch-musicnet")
    fetch_musicnet.add_argument(
        "output",
        type=Path,
        default=Path("data_sources/musicnet/musicnet.tar.gz"),
        nargs="?",
    )
    fetch_musicnet.add_argument("--workers", type=int, default=8)

    extract_musicnet = subparsers.add_parser("extract-musicnet")
    extract_musicnet.add_argument(
        "archive",
        type=Path,
        default=Path("data_sources/musicnet/musicnet.tar.gz"),
        nargs="?",
    )
    extract_musicnet.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/control_classical.csv"),
    )
    extract_musicnet.add_argument("--data-root", type=Path, default=Path("data_raw"))
    extract_musicnet.add_argument("--expected-md5", default=MUSICNET_ARCHIVE_MD5)

    fallback = subparsers.add_parser("fallback-classical")
    fallback.add_argument(
        "manifest",
        type=Path,
        default=Path("metadata/control_classical.csv"),
        nargs="?",
    )

    download = subparsers.add_parser("download")
    download.add_argument("manifest", type=Path)
    download.add_argument("--data-root", type=Path, default=Path("data_raw"))
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--limit", type=int, default=0)
    download.add_argument("--force-track", action="append", default=[])
    download.add_argument("--jamendo-client-id-env", default="JAMENDO_CLIENT_ID")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("manifests", type=Path, nargs="+")
    finalize.add_argument("--metadata-dir", type=Path, default=Path("metadata"))

    audit = subparsers.add_parser("audit")
    audit.add_argument("manifests", type=Path, nargs="+")
    audit.add_argument("--data-root", type=Path, default=Path("data_raw"))
    audit.add_argument("--verify-hash", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "select-pop":
            rows = select_pop_candidates(
                args.source_root,
                args.output,
                max_per_artist=args.max_per_artist,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                excluded_source_ids=read_pop_exclusions(args.exclude_list),
                target_count=args.target_count,
            )
            print(json.dumps({"selected": len(rows), "output": str(args.output)}, indent=2))
        elif args.command == "preflight-pop":
            report = preflight_pop_availability(
                args.manifest,
                args.exclude_list,
                client_id=os.environ.get(args.jamendo_client_id_env, ""),
            )
            print(json.dumps(report, indent=2))
        elif args.command == "supplement-pop-jamendo":
            rows = supplement_pop_from_jamendo(
                args.manifest,
                client_id=os.environ.get(args.jamendo_client_id_env, ""),
                excluded_source_ids=read_pop_exclusions(args.exclude_list),
                target_total=args.target_total,
                max_per_artist=args.max_per_artist,
                max_pages=args.max_pages,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            print(json.dumps({"selected": len(rows), "output": str(args.manifest)}, indent=2))
        elif args.command == "normalize-pop-licenses":
            updated = normalize_pop_license_types(args.manifest)
            print(json.dumps({"updated": updated}, indent=2))
        elif args.command == "catalog-classical":
            rows = catalog_classical_candidates(
                args.output,
                validation_fraction=args.validation_fraction,
                holdout_fraction=args.holdout_fraction,
                seed=args.seed,
            )
            print(json.dumps({"selected": len(rows), "output": str(args.output)}, indent=2))
        elif args.command == "extend-classical-musicnet":
            rows = extend_classical_with_musicnet(
                args.manifest,
                args.metadata,
                target_total=args.target_total,
                max_per_composer=args.max_per_composer,
                validation_fraction=args.validation_fraction,
                holdout_fraction=args.holdout_fraction,
                seed=args.seed,
            )
            print(json.dumps({"selected": len(rows), "output": str(args.manifest)}, indent=2))
        elif args.command == "fetch-musicnet":
            fetch_file_segmented(
                MUSICNET_ARCHIVE_URL,
                args.output,
                expected_bytes=MUSICNET_ARCHIVE_BYTES,
                workers=args.workers,
            )
            print(json.dumps({"downloaded": str(args.output)}, indent=2))
        elif args.command == "extract-musicnet":
            counts = extract_musicnet_archive(
                args.archive,
                args.manifest,
                args.data_root,
                expected_md5=args.expected_md5,
            )
            print(json.dumps(dict(counts), indent=2))
        elif args.command == "fallback-classical":
            updated = apply_classical_standard_fallback(args.manifest)
            print(json.dumps({"fallbacks_applied": updated}, indent=2))
        elif args.command == "download":
            client_id = os.environ.get(args.jamendo_client_id_env)
            counts = download_candidates(
                args.manifest,
                args.data_root,
                client_id=client_id,
                workers=args.workers,
                limit=args.limit,
                force_track_ids=args.force_track,
            )
            print(json.dumps(dict(counts), indent=2))
        elif args.command == "finalize":
            counts = finalize_metadata(args.manifests, args.metadata_dir)
            print(json.dumps(counts, indent=2))
        elif args.command == "audit":
            report = audit_candidates(args.manifests, args.data_root, verify_hash=args.verify_hash)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
    except (ControlDatasetError, OSError, tarfile.TarError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
