from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from data.focus import FocusDatasetError, scan_mp3

JAMENDO_TRACKS_API = "https://api.jamendo.com/v3.0/tracks/"
JAMENDO_FILE_API = (
    "https://api.jamendo.com/v3.0/tracks/file/"
    "?client_id={client_id}&id={track_id}&audioformat=mp32&action=download"
)
USER_AGENT = "FocusMusicGLMY/0.3 (non-commercial academic dataset builder)"

MOOD_TAGS = ("study", "focus", "work", "meditation", "relaxing", "deepwork", "background")
GENRE_TAGS = ("ambient", "lofi", "chillout", "downtempo", "drone", "newage", "minimalism")
INSTRUMENT_TAGS = ("piano", "synthesizer", "acousticguitar", "soundscape")
SPEEDS = ("verylow", "low")
CANONICAL_SELECTED_DIRECTORY = "focus_music"
EXCLUDED_DIRECTORY = "focus_open_music_excluded"

OPEN_FOCUS_COLUMNS = [
    "track_id",
    "group",
    "source_dataset",
    "source_track_id",
    "title",
    "creator",
    "artist_key",
    "album_key",
    "album_name",
    "duration_seconds",
    "source_url",
    "download_url",
    "license_type",
    "license_url",
    "matched_mood_tags",
    "matched_genre_tags",
    "matched_instrument_tags",
    "query_tags",
    "instrumental_evidence",
    "speed_evidence",
    "requested_audioformat",
    "selection_score",
    "selection_rank",
    "selection_status",
    "split",
    "relative_path",
    "sha256",
    "audio_payload_sha256",
    "average_bitrate_kbps",
    "sample_rate",
    "channels",
    "download_status",
    "downloaded_at",
    "duplicate_of",
    "error",
]

TRACK_INDEX_COLUMNS = [
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
]
LICENSE_COLUMNS = [
    "track_id",
    "group",
    "source_url",
    "license_type",
    "downloaded_at",
    "redistribution_allowed",
    "notes",
]


class OpenFocusDatasetError(RuntimeError):
    pass


@dataclass(slots=True)
class JamendoCandidate:
    result: dict[str, Any]
    query_tags: set[str] = field(default_factory=set)
    speeds: set[str] = field(default_factory=set)


def _normalize_tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _creative_commons_name(url: str) -> str:
    parts = urllib.parse.urlparse(url).path.strip("/").split("/")
    code = parts[1].lower() if len(parts) > 1 and parts[0] == "licenses" else parts[0].lower()
    names = {
        "by": "Creative Commons Attribution license",
        "by-sa": "Creative Commons Attribution-Share-Alike license",
        "by-nd": "Creative Commons Attribution-No Derivatives license",
        "by-nc": "Creative Commons Attribution-Non-Commercial license",
        "by-nc-sa": "Creative Commons Attribution-Non-Commercial-Share-Alike license",
        "by-nc-nd": "Creative Commons Attribution-Non-Commercial-No Derivatives license",
        "cc0": "Creative Commons Zero",
    }
    return names.get(code, f"Creative Commons {code.upper()} license")


def _write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPEN_FOCUS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_table_atomic(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        optional_columns = {"album_name"}
        missing = set(OPEN_FOCUS_COLUMNS) - optional_columns - set(reader.fieldnames or [])
        if missing:
            raise OpenFocusDatasetError(f"{path} is missing columns: {sorted(missing)}")
        return [
            {key: (row.get(key) or "") for key in OPEN_FOCUS_COLUMNS}
            for row in reader
        ]


def _request_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
                return json.load(response)
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise OpenFocusDatasetError(f"Jamendo request failed after three attempts: {last_error}")


def _musicinfo_tags(result: dict[str, Any]) -> tuple[set[str], set[str]]:
    musicinfo = result.get("musicinfo") or {}
    tags = musicinfo.get("tags") or {}
    genres = {_normalize_tag(str(tag)) for tag in tags.get("genres", [])}
    instruments = {_normalize_tag(str(tag)) for tag in tags.get("instruments", [])}
    vartags = {_normalize_tag(str(tag)) for tag in tags.get("vartags", [])}
    return genres | vartags, instruments | vartags


def query_jamendo_candidates(
    *,
    client_id: str,
    max_pages: int = 3,
    page_size: int = 200,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
) -> dict[str, JamendoCandidate]:
    """Collect the union of mood queries under mandatory API-side filters."""

    if not client_id:
        raise OpenFocusDatasetError("JAMENDO_CLIENT_ID is required")
    if max_pages < 1 or page_size < 1 or page_size > 200:
        raise ValueError("max_pages must be positive and page_size must be between 1 and 200")
    candidates: dict[str, JamendoCandidate] = {}
    for mood in MOOD_TAGS:
        for speed in SPEEDS:
            for page in range(max_pages):
                params = {
                    "client_id": client_id,
                    "format": "json",
                    "limit": str(page_size),
                    "offset": str(page * page_size),
                    "order": "id_asc",
                    "type": "single albumtrack",
                    "tags": mood,
                    "vocalinstrumental": "instrumental",
                    "speed": speed,
                    "include": "musicinfo",
                    "audioformat": "mp32",
                }
                url = f"{JAMENDO_TRACKS_API}?{urllib.parse.urlencode(params)}"
                payload = request_json(url)
                headers = payload.get("headers", {})
                if headers.get("status") != "success":
                    raise OpenFocusDatasetError(
                        "Jamendo query failed: " + headers.get("error_message", "unknown error")
                    )
                results = payload.get("results", [])
                for result in results:
                    source_id = str(result["id"])
                    if not result.get("audiodownload_allowed") or not result.get("license_ccurl"):
                        continue
                    candidate = candidates.setdefault(source_id, JamendoCandidate(result=result))
                    candidate.query_tags.add(mood)
                    candidate.speeds.add(speed)
                if len(results) < page_size:
                    break
                time.sleep(0.25)
    return candidates


def _jamendo_row(candidate: JamendoCandidate) -> dict[str, str] | None:
    result = candidate.result
    source_id = str(result["id"])
    observed_tags, observed_instruments = _musicinfo_tags(result)
    moods = sorted(candidate.query_tags | (observed_tags & set(MOOD_TAGS)))
    genres = sorted(observed_tags & set(GENRE_TAGS))
    instruments = sorted(observed_instruments & set(INSTRUMENT_TAGS))
    if not moods or not (genres or instruments):
        return None
    duration = float(result.get("duration") or 0)
    if duration < 120 or duration > 1_200:
        return None
    speed = sorted(candidate.speeds, key=SPEEDS.index)
    score = 5 * len(moods) + 3 * len(genres) + 2 * len(instruments)
    score += 2 if "verylow" in speed else 1
    album_id = str(result.get("album_id") or "")
    license_url = str(result["license_ccurl"])
    local_id = f"focus_jamendo_{int(source_id):07d}"
    return {
        "track_id": local_id,
        "group": "focus",
        "source_dataset": "jamendo-api-open-focus",
        "source_track_id": source_id,
        "title": str(result.get("name", "")),
        "creator": str(result.get("artist_name", "")),
        "artist_key": f"jamendo_artist_{int(result['artist_id']):06d}",
        "album_key": (
            f"jamendo_album_{int(album_id):06d}"
            if album_id
            else f"jamendo_single_{int(source_id):07d}"
        ),
        "album_name": str(result.get("album_name", "")),
        "duration_seconds": f"{duration:.3f}",
        "source_url": str(result.get("shareurl", "")),
        "download_url": JAMENDO_FILE_API.format(client_id="{client_id}", track_id=source_id),
        "license_type": _creative_commons_name(license_url),
        "license_url": license_url,
        "matched_mood_tags": " ".join(moods),
        "matched_genre_tags": " ".join(genres),
        "matched_instrument_tags": " ".join(instruments),
        "query_tags": " ".join(sorted(candidate.query_tags)),
        "instrumental_evidence": "Jamendo API vocalinstrumental=instrumental",
        "speed_evidence": "Jamendo API speed=" + " ".join(speed),
        "requested_audioformat": "mp32",
        "selection_score": str(score),
        "selection_rank": "",
        "selection_status": "",
        "split": "",
        "relative_path": f"{CANONICAL_SELECTED_DIRECTORY}/{local_id}.mp3",
        "sha256": "",
        "audio_payload_sha256": "",
        "average_bitrate_kbps": "",
        "sample_rate": "",
        "channels": "",
        "download_status": "pending",
        "downloaded_at": "",
        "duplicate_of": "",
        "error": "",
    }


def _rank_with_caps(
    rows: Sequence[dict[str, str]],
    *,
    count: int,
    max_per_artist: int,
    max_per_album: int,
    seed: int,
) -> list[dict[str, str]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row["selection_score"]),
            _stable_key(seed, row["track_id"]),
        ),
    )
    artist_counts: Counter[str] = Counter()
    album_counts: Counter[str] = Counter()
    selected: list[dict[str, str]] = []
    for row in ranked:
        if artist_counts[row["artist_key"]] >= max_per_artist:
            continue
        if album_counts[row["album_key"]] >= max_per_album:
            continue
        selected.append(row)
        artist_counts[row["artist_key"]] += 1
        album_counts[row["album_key"]] += 1
        if len(selected) == count:
            break
    return selected


def _read_fma_table(path: Path) -> tuple[list[tuple[str, str]], list[list[str]]]:
    """Read an FMA pandas-MultiIndex CSV without requiring pandas."""

    if not path.is_file():
        raise OpenFocusDatasetError(f"FMA metadata file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        top = next(reader)
        second = next(reader)
        width = max(len(top), len(second))
        top += [""] * (width - len(top))
        second += [""] * (width - len(second))
        headers = [(top[index].strip(), second[index].strip()) for index in range(width)]
        return headers, [row for row in reader if row and row[0].strip().isdigit()]


def _fma_records(path: Path) -> dict[str, dict[str, str]]:
    headers, rows = _read_fma_table(path)
    records: dict[str, dict[str, str]] = {}
    for values in rows:
        values += [""] * (len(headers) - len(values))
        records[str(int(values[0]))] = {
            f"{group}.{name}".strip("."): value
            for (group, name), value in zip(headers[1:], values[1:], strict=False)
        }
    return records


def _fma_feature_records(path: Path) -> dict[str, dict[str, float]]:
    """Read echonest.csv, whose column header can occupy two or three rows."""

    if not path.is_file():
        raise OpenFocusDatasetError(f"FMA feature file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    feature_row = next(
        (index for index, row in enumerate(rows[:6]) if "instrumentalness" in row),
        None,
    )
    if feature_row is None:
        raise OpenFocusDatasetError(f"cannot locate FMA audio feature header in {path}")
    names = rows[feature_row]
    wanted = ("instrumentalness", "tempo", "speechiness", "energy")
    positions = {name: names.index(name) for name in wanted if name in names}
    missing = set(wanted) - set(positions)
    if missing:
        raise OpenFocusDatasetError(f"FMA features are missing: {sorted(missing)}")
    result: dict[str, dict[str, float]] = {}
    for row in rows[feature_row + 1 :]:
        if not row or not row[0].strip().isdigit():
            continue
        try:
            result[str(int(row[0]))] = {
                name: float(row[index]) for name, index in positions.items()
            }
        except (IndexError, ValueError):
            continue
    return result


def _read_fma_genres(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("genre_id", "")): _normalize_tag(row.get("title", ""))
            for row in reader
        }


def select_fma_fallback(
    metadata_dir: Path,
    audio_root: Path,
    *,
    count: int,
    seed: int = 20260801,
) -> list[dict[str, str]]:
    """Select locally available FMA audio only when Jamendo cannot fill the target."""

    if count <= 0:
        return []
    tracks = _fma_records(metadata_dir / "tracks.csv")
    features = _fma_feature_records(metadata_dir / "echonest.csv")
    genre_names = _read_fma_genres(metadata_dir / "genres.csv")
    rows: list[dict[str, str]] = []
    for source_id, feature in features.items():
        track = tracks.get(source_id)
        if not track:
            continue
        if (
            feature["instrumentalness"] < 0.90
            or feature["tempo"] > 100
            or feature["speechiness"] > 0.10
            or feature["energy"] > 0.65
        ):
            continue
        padded = f"{int(source_id):06d}"
        audio = audio_root / padded[:3] / f"{padded}.mp3"
        if not audio.is_file():
            continue
        raw_genres = " ".join(
            (
                track.get("track.genre_top", ""),
                track.get("track.tags", ""),
                track.get("track.genres_all", ""),
            )
        )
        normalized = _normalize_tag(raw_genres)
        numeric_genres = set(re.findall(r"\d+", track.get("track.genres_all", "")))
        observed = {
            tag
            for tag in GENRE_TAGS
            if _normalize_tag(tag) in normalized
        }
        observed.update(
            name for genre_id, name in genre_names.items() if genre_id in numeric_genres
        )
        matched_genres = sorted(set(GENRE_TAGS) & observed)
        if not matched_genres:
            continue
        artist_id = track.get("artist.id", "") or source_id
        album_id = track.get("album.id", "") or source_id
        duration = track.get("track.duration", "")
        try:
            duration_value = float(duration)
        except ValueError:
            duration_value = 0
        if duration_value and (duration_value < 120 or duration_value > 1_200):
            continue
        local_id = f"focus_fma_{int(source_id):06d}"
        score = 6 + 3 * len(matched_genres)
        score += 2 if feature["tempo"] <= 80 else 1
        license_type = track.get("track.license", "") or "FMA per-track license"
        rows.append(
            {
                "track_id": local_id,
                "group": "focus",
                "source_dataset": "free-music-archive-focus-fallback",
                "source_track_id": source_id,
                "title": track.get("track.title", ""),
                "creator": track.get("artist.name", ""),
                "artist_key": f"fma_artist_{artist_id}",
                "album_key": f"fma_album_{album_id}",
                "album_name": track.get("album.title", ""),
                "duration_seconds": f"{duration_value:.3f}" if duration_value else "",
                "source_url": "https://github.com/mdeff/fma",
                "download_url": audio.resolve().as_uri(),
                "license_type": license_type,
                "license_url": track.get("track.license_url", ""),
                "matched_mood_tags": "focus_proxy",
                "matched_genre_tags": " ".join(matched_genres),
                "matched_instrument_tags": "",
                "query_tags": "FMA measured-feature fallback",
                "instrumental_evidence": (
                    f"FMA Echonest instrumentalness={feature['instrumentalness']:.3f}; "
                    f"speechiness={feature['speechiness']:.3f}"
                ),
                "speed_evidence": (
                    f"FMA Echonest tempo={feature['tempo']:.3f}; energy={feature['energy']:.3f}"
                ),
                "requested_audioformat": "source MP3 (not Jamendo mp32)",
                "selection_score": str(score),
                "selection_rank": "",
                "selection_status": "",
                "split": "",
                "relative_path": f"{CANONICAL_SELECTED_DIRECTORY}/{local_id}.mp3",
                "sha256": "",
                "audio_payload_sha256": "",
                "average_bitrate_kbps": "",
                "sample_rate": "",
                "channels": "",
                "download_status": "pending",
                "downloaded_at": "",
                "duplicate_of": "",
                "error": "",
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (-int(row["selection_score"]), _stable_key(seed, row["track_id"])),
    )
    if len(ranked) < count:
        raise OpenFocusDatasetError(
            f"FMA fallback has {len(ranked)} eligible local tracks, below required {count}"
        )
    return ranked[:count]


def _assign_splits(rows: list[dict[str, str]], seed: int) -> None:
    active = [row for row in rows if row["selection_status"] == "selected"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in active:
        grouped[row["artist_key"]].append(row)
    targets = {
        "discovery": round(len(active) * 0.65),
        "validation": round(len(active) * 0.20),
    }
    targets["holdout"] = len(active) - targets["discovery"] - targets["validation"]
    counts = Counter()
    for artist in sorted(grouped, key=lambda key: (-len(grouped[key]), _stable_key(seed, key))):
        split = min(
            targets,
            key=lambda name: (
                abs(targets[name] - (counts[name] + len(grouped[artist])))
                - abs(targets[name] - counts[name]),
                name,
            ),
        )
        for row in grouped[artist]:
            row["split"] = split
        counts[split] += len(grouped[artist])


def _genre_tokens(row: dict[str, str]) -> set[str]:
    return {_normalize_tag(value) for value in row["matched_genre_tags"].split() if value}


def _mood_tokens(row: dict[str, str]) -> set[str]:
    return {_normalize_tag(value) for value in row["matched_mood_tags"].split() if value}


def _tag_score(
    row: dict[str, str],
    allowed_genres: set[str],
    allowed_moods: set[str],
    preferred_text_stem: str = "",
) -> int:
    genres = _genre_tokens(row)
    moods = _mood_tokens(row)
    score = 1_000 * len(moods & allowed_moods)
    score += 800 * len(genres & allowed_genres)
    score += 200 * len(moods & {"study", "focus", "deepwork"})
    score += 60 if "meditation" in moods else 0
    score += 20 if "relaxing" in moods else 0
    score += 50 if "lofi" in genres else 0
    score += 25 if "ambient" in genres else 0
    title_album = _normalize_text(f"{row['title']} {row['album_name']}")
    if preferred_text_stem and preferred_text_stem in title_album:
        score += 10_000
    score += 3 if "verylow" in row["speed_evidence"] else 1
    return score


def _read_local_id3(path: Path) -> dict[str, str]:
    try:
        from mutagen.easyid3 import EasyID3

        tags = EasyID3(path)
    except (ImportError, OSError, ValueError):
        return {}
    return {
        key: " ".join(tags.get(key, [])).strip()
        for key in ("title", "album", "artist", "genre", "copyright", "website")
    }


def _id3_orphan_row(path: Path, tags: dict[str, str]) -> dict[str, str]:
    track_id = path.stem
    match = re.fullmatch(r"focus_jamendo_(\d+)", track_id)
    if match is None:
        raise OpenFocusDatasetError(f"unexpected local Jamendo filename: {path.name}")
    source_id = str(int(match.group(1)))
    scan = scan_mp3(path)
    website = tags.get("website", "")
    artist_match = re.search(r"/artist/(\d+)", website)
    artist_token = artist_match.group(1) if artist_match else hashlib.sha256(
        tags.get("artist", "").encode()
    ).hexdigest()[:12]
    album_name = tags.get("album", "")
    album_token = hashlib.sha256(
        f"{tags.get('artist', '')}:{album_name}".encode()
    ).hexdigest()[:12]
    license_url = tags.get("copyright", "")
    license_type = (
        _creative_commons_name(license_url)
        if "creativecommons.org" in license_url
        else "Jamendo per-track license"
    )
    genre = _normalize_tag(tags.get("genre", ""))
    return {
        "track_id": track_id,
        "group": "focus",
        "source_dataset": "jamendo-api-open-focus-early-local-candidate",
        "source_track_id": source_id,
        "title": tags.get("title", ""),
        "creator": tags.get("artist", ""),
        "artist_key": f"jamendo_artist_{artist_token}",
        "album_key": f"jamendo_id3_album_{album_token}",
        "album_name": album_name,
        "duration_seconds": f"{scan.duration_seconds:.3f}",
        "source_url": f"https://www.jamendo.com/track/{source_id}",
        "download_url": JAMENDO_FILE_API.format(
            client_id="{client_id}", track_id=source_id
        ),
        "license_type": license_type,
        "license_url": license_url,
        "matched_mood_tags": "meditation",
        "matched_genre_tags": genre,
        "matched_instrument_tags": "",
        "query_tags": "local ID3 title/album meditation evidence",
        "instrumental_evidence": (
            "inherited early Jamendo candidate query: vocalinstrumental=instrumental"
        ),
        "speed_evidence": "inherited early Jamendo candidate query: speed=low/verylow",
        "requested_audioformat": "mp32",
        "selection_score": "",
        "selection_rank": "",
        "selection_status": "reserve",
        "split": "",
        "relative_path": f"{path.parent.name}/{path.name}",
        "sha256": scan.sha256,
        "audio_payload_sha256": scan.audio_payload_sha256,
        "average_bitrate_kbps": f"{scan.average_bitrate_kbps:.3f}",
        "sample_rate": str(scan.sample_rate),
        "channels": str(scan.channels),
        "download_status": "excluded_selection",
        "downloaded_at": date.fromtimestamp(path.stat().st_mtime).isoformat(),
        "duplicate_of": "",
        "error": "",
    }


def rebuild_local_style_selection(
    manifest: Path,
    selected_dir: Path,
    excluded_dir: Path,
    *,
    allowed_genres: Sequence[str] = ("ambient", "lofi"),
    allowed_moods: Sequence[str] = (),
    prefer_title_album_stem: str = "",
    import_id3_orphans_stem: str = "",
    force_include: Sequence[str] = (),
    force_exclude: Sequence[str] = (),
    target_count: int = 300,
    max_per_artist: int = 4,
    max_per_album: int = 2,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Rebuild the local corpus from downloaded files using tag and metadata evidence."""

    selected_dir = selected_dir.resolve()
    excluded_dir = excluded_dir.resolve()
    if selected_dir == excluded_dir:
        raise OpenFocusDatasetError("selected and excluded directories must differ")
    if selected_dir.parent != excluded_dir.parent:
        raise OpenFocusDatasetError("selected and excluded directories must share a parent")
    rows = _read_rows(manifest)
    local_files: dict[str, Path] = {}
    for directory in (selected_dir, excluded_dir):
        if not directory.exists():
            continue
        for path in directory.glob("*.mp3"):
            if path.name in local_files:
                raise OpenFocusDatasetError(f"duplicate local filename: {path.name}")
            local_files[path.name] = path
    row_by_name = {f"{row['track_id']}.mp3": row for row in rows}
    id3_by_name: dict[str, dict[str, str]] = {}
    for name, path in local_files.items():
        tags = _read_local_id3(path)
        id3_by_name[name] = tags
        existing = row_by_name.get(name)
        if existing is not None:
            existing["album_name"] = tags.get("album", existing["album_name"])
            existing["title"] = existing["title"] or tags.get("title", "")
            existing["creator"] = existing["creator"] or tags.get("artist", "")
    import_stem = _normalize_text(import_id3_orphans_stem).strip()
    imported = 0
    if import_stem:
        for name, path in local_files.items():
            if name in row_by_name:
                continue
            tags = id3_by_name[name]
            title_album = _normalize_text(f"{tags.get('title', '')} {tags.get('album', '')}")
            if import_stem not in title_album:
                continue
            row = _id3_orphan_row(path, tags)
            rows.append(row)
            row_by_name[name] = row
            imported += 1
    allowed = {_normalize_tag(value) for value in allowed_genres}
    allowed_mood_set = {_normalize_tag(value) for value in allowed_moods}
    if not allowed and not allowed_mood_set:
        raise ValueError("at least one allowed genre or mood must be provided")
    preferred_stem = _normalize_text(prefer_title_album_stem).strip()
    forced_in = set(force_include)
    forced_out = set(force_exclude)
    if forced_in & forced_out:
        raise OpenFocusDatasetError("the same track cannot be forced in and forced out")

    def matches(row: dict[str, str]) -> bool:
        genre_match = not allowed or bool(_genre_tokens(row) & allowed)
        mood_match = not allowed_mood_set or bool(_mood_tokens(row) & allowed_mood_set)
        return genre_match and mood_match

    eligible = [
        row
        for name, row in row_by_name.items()
        if name in local_files and matches(row)
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -_tag_score(row, allowed, allowed_mood_set, preferred_stem),
            _stable_key(seed, row["track_id"]),
        ),
    )
    artist_counts: Counter[str] = Counter()
    album_counts: Counter[str] = Counter()
    eligible_by_id = {row["track_id"]: row for row in eligible}
    missing_forced = forced_in - set(eligible_by_id)
    if missing_forced:
        raise OpenFocusDatasetError(
            f"forced tracks are not local eligible candidates: {sorted(missing_forced)}"
        )
    selected: list[dict[str, str]] = [
        eligible_by_id[track_id] for track_id in sorted(forced_in)
    ]
    for row in selected:
        artist_counts[row["artist_key"]] += 1
        album_counts[row["album_key"]] += 1
    for row in ranked:
        if row["track_id"] in forced_in or row["track_id"] in forced_out:
            continue
        if artist_counts[row["artist_key"]] >= max_per_artist:
            continue
        if album_counts[row["album_key"]] >= max_per_album:
            continue
        selected.append(row)
        artist_counts[row["artist_key"]] += 1
        album_counts[row["album_key"]] += 1
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise OpenFocusDatasetError(
            f"only {len(selected)} local tracks satisfy the style and diversity constraints; "
            f"target is {target_count}"
        )
    selected_ids = {row["track_id"] for row in selected}
    selected_rank = {row["track_id"]: index for index, row in enumerate(selected, start=1)}
    for row in rows:
        name = f"{row['track_id']}.mp3"
        is_local = name in local_files
        tag_match = matches(row)
        row["split"] = ""
        row["selection_rank"] = str(selected_rank.get(row["track_id"], ""))
        if tag_match:
            row["selection_score"] = str(
                _tag_score(row, allowed, allowed_mood_set, preferred_stem)
            )
        if row["track_id"] in selected_ids:
            row["selection_status"] = "selected"
            row["download_status"] = "verified"
            row["relative_path"] = f"{CANONICAL_SELECTED_DIRECTORY}/{name}"
            row["duplicate_of"] = ""
            row["error"] = ""
        elif is_local and tag_match:
            row["selection_status"] = "reserve"
            row["download_status"] = "excluded_selection"
            row["relative_path"] = f"{EXCLUDED_DIRECTORY}/{name}"
            row["error"] = "eligible tag candidate not selected under diversity caps"
        elif is_local:
            row["selection_status"] = "rejected"
            row["download_status"] = "excluded_tags"
            row["relative_path"] = f"{EXCLUDED_DIRECTORY}/{name}"
            row["error"] = "tag evidence does not match the configured filters"
        else:
            row["selection_status"] = "reserve"
            row["download_status"] = "pending"
    _assign_splits(rows, seed)

    selected_dir.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)
    for name, source in local_files.items():
        destination_root = (
            selected_dir if name.removesuffix(".mp3") in selected_ids else excluded_dir
        )
        destination = destination_root / name
        if source == destination:
            continue
        if destination.exists():
            raise OpenFocusDatasetError(f"destination already exists: {destination}")
    for name, source in local_files.items():
        destination_root = (
            selected_dir if name.removesuffix(".mp3") in selected_ids else excluded_dir
        )
        destination = destination_root / name
        if source != destination:
            source.replace(destination)
    _write_csv_atomic(manifest, rows)
    return {
        "selected": len(selected),
        "eligible_local": len(eligible),
        "selected_artists": len(artist_counts),
        "selected_albums": len(album_counts),
        "allowed_genres": sorted(allowed),
        "allowed_moods": sorted(allowed_mood_set),
        "preferred_title_album_stem": preferred_stem,
        "preferred_title_album_selected": sum(
            bool(preferred_stem)
            and preferred_stem in _normalize_text(f"{row['title']} {row['album_name']}")
            for row in selected
        ),
        "imported_id3_orphans": imported,
        "forced_included": sorted(forced_in),
        "forced_excluded": sorted(forced_out),
        "selected_directory": str(selected_dir),
        "excluded_directory": str(excluded_dir),
        "excluded_local_files": len(local_files) - len(selected),
    }


def select_jamendo_focus(
    output: Path,
    *,
    client_id: str,
    target_count: int = 300,
    reserve_count: int = 60,
    max_per_artist: int = 4,
    max_per_album: int = 2,
    max_pages: int = 3,
    seed: int = 20260801,
    fma_metadata_dir: Path | None = None,
    fma_audio_root: Path | None = None,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
) -> list[dict[str, str]]:
    pool = query_jamendo_candidates(
        client_id=client_id,
        max_pages=max_pages,
        request_json=request_json,
    )
    eligible = [row for item in pool.values() if (row := _jamendo_row(item)) is not None]
    wanted = target_count + reserve_count
    rows = _rank_with_caps(
        eligible,
        count=wanted,
        max_per_artist=max_per_artist,
        max_per_album=max_per_album,
        seed=seed,
    )
    if len(rows) < target_count:
        if fma_metadata_dir is None or fma_audio_root is None:
            raise OpenFocusDatasetError(
                f"Jamendo produced {len(rows)} capped eligible tracks; "
                f"{target_count - len(rows)} FMA tracks are required. Provide both FMA paths."
            )
        rows.extend(
            select_fma_fallback(
                fma_metadata_dir,
                fma_audio_root,
                count=target_count - len(rows),
                seed=seed,
            )
        )
    previous = {row["track_id"]: row for row in _read_rows(output)}
    for rank, row in enumerate(rows, start=1):
        row["selection_rank"] = str(rank)
        row["selection_status"] = "selected" if rank <= target_count else "reserve"
        old = previous.get(row["track_id"])
        if old and old["download_status"] in {"verified", "rejected_duplicate", "rejected_bitrate"}:
            for field in (
                "sha256",
                "audio_payload_sha256",
                "average_bitrate_kbps",
                "sample_rate",
                "channels",
                "download_status",
                "downloaded_at",
                "duplicate_of",
                "error",
            ):
                row[field] = old[field]
    _assign_splits(rows, seed)
    _write_csv_atomic(output, rows)
    return rows


def _looks_like_mp3(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(3)
    return head == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def _download_one(
    row: dict[str, str],
    data_root: Path,
    client_id: str,
    minimum_bitrate_kbps: float,
) -> dict[str, str]:
    result = dict(row)
    target = data_root / PurePosixPath(row["relative_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    url = row["download_url"].format(client_id=urllib.parse.quote(client_id, safe=""))
    partial = target.with_suffix(target.suffix + ".part")
    last_error = ""
    for attempt in range(3):
        try:
            if not target.is_file():
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                    with partial.open("wb") as handle:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                os.replace(partial, target)
            if not _looks_like_mp3(target):
                raise OpenFocusDatasetError("downloaded response is not an MP3")
            scan = scan_mp3(target)
            result.update(
                {
                    "duration_seconds": f"{scan.duration_seconds:.3f}",
                    "sha256": scan.sha256,
                    "audio_payload_sha256": scan.audio_payload_sha256,
                    "average_bitrate_kbps": f"{scan.average_bitrate_kbps:.3f}",
                    "sample_rate": str(scan.sample_rate),
                    "channels": str(scan.channels),
                    "downloaded_at": date.today().isoformat(),
                    "error": "",
                }
            )
            if scan.average_bitrate_kbps < minimum_bitrate_kbps:
                result["download_status"] = "rejected_bitrate"
                result["error"] = (
                    f"actual average bitrate {scan.average_bitrate_kbps:.1f} kbps is below "
                    f"required {minimum_bitrate_kbps:.1f} kbps"
                )
            else:
                result["download_status"] = "verified"
            return result
        except (OSError, urllib.error.URLError, FocusDatasetError, OpenFocusDatasetError) as exc:
            last_error = str(exc)
            partial.unlink(missing_ok=True)
            time.sleep(2**attempt)
    result["download_status"] = "failed"
    result["error"] = last_error[:500]
    return result


def _mark_duplicates(rows: list[dict[str, str]]) -> None:
    canonical: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: int(item["selection_rank"])):
        if row["download_status"] != "verified":
            continue
        payload = row["audio_payload_sha256"]
        if payload in canonical:
            row["download_status"] = "rejected_duplicate"
            row["duplicate_of"] = canonical[payload]
            row["error"] = "duplicate MPEG audio payload"
        else:
            canonical[payload] = row["track_id"]


def download_open_focus(
    manifest: Path,
    data_root: Path,
    *,
    client_id: str,
    target_count: int = 300,
    workers: int = 4,
    minimum_bitrate_kbps: float = 300.0,
    seed: int = 20260801,
) -> Counter[str]:
    if not client_id:
        raise OpenFocusDatasetError("JAMENDO_CLIENT_ID is required")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    rows = _read_rows(manifest)
    for row in rows:
        if not row["average_bitrate_kbps"]:
            continue
        bitrate = float(row["average_bitrate_kbps"])
        if row["download_status"] == "verified" and bitrate < minimum_bitrate_kbps:
            row["download_status"] = "rejected_bitrate"
            row["error"] = (
                f"actual average bitrate {bitrate:.1f} kbps is below "
                f"required {minimum_bitrate_kbps:.1f} kbps"
            )
        elif row["download_status"] == "rejected_bitrate" and bitrate >= minimum_bitrate_kbps:
            row["download_status"] = "verified"
            row["error"] = ""
    while sum(row["download_status"] == "verified" for row in rows) < target_count:
        verified = sum(row["download_status"] == "verified" for row in rows)
        needed = target_count - verified
        pending = [
            index
            for index, row in enumerate(rows)
            if row["download_status"] == "pending"
            and row["selection_status"] in {"selected", "reserve"}
        ][:needed]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_one,
                    rows[index],
                    data_root,
                    client_id,
                    minimum_bitrate_kbps,
                ): index
                for index in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                try:
                    rows[index] = future.result()
                except Exception as exc:  # Preserve the ledger after an individual failure.
                    rows[index]["download_status"] = "failed"
                    rows[index]["error"] = str(exc)[:500]
                print(
                    f"[{completed}/{len(futures)}] {rows[index]['track_id']}: "
                    f"{rows[index]['download_status']}",
                    flush=True,
                )
                _write_csv_atomic(manifest, rows)
        _mark_duplicates(rows)
        _write_csv_atomic(manifest, rows)
    verified_rows = sorted(
        (row for row in rows if row["download_status"] == "verified"),
        key=lambda row: int(row["selection_rank"]),
    )[:target_count]
    verified_ids = {row["track_id"] for row in verified_rows}
    for row in rows:
        row["split"] = ""
        if row["track_id"] in verified_ids:
            row["selection_status"] = "selected"
        elif row["download_status"] in {"pending", ""}:
            row["selection_status"] = "reserve"
        else:
            row["selection_status"] = "rejected"
    _assign_splits(rows, seed)
    _write_csv_atomic(manifest, rows)
    return Counter(row["download_status"] for row in rows)


def audit_open_focus(
    manifest: Path,
    data_root: Path,
    *,
    target_count: int = 300,
    minimum_bitrate_kbps: float = 300.0,
    verify_hash: bool = False,
    allowed_genres: Sequence[str] = (),
    allowed_moods: Sequence[str] = (),
) -> dict[str, Any]:
    rows = _read_rows(manifest)
    verified = [row for row in rows if row["download_status"] == "verified"]
    errors: list[str] = []
    warnings: list[str] = []
    allowed = {_normalize_tag(value) for value in allowed_genres}
    allowed_mood_set = {_normalize_tag(value) for value in allowed_moods}
    if len(verified) != target_count:
        errors.append(f"expected {target_count} verified tracks, found {len(verified)}")
    payloads: set[str] = set()
    for row in verified:
        target = data_root / PurePosixPath(row["relative_path"])
        if not target.is_file():
            errors.append(f"missing audio: {row['track_id']}")
            continue
        if verify_hash:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                errors.append(f"checksum mismatch: {row['track_id']}")
        if row["audio_payload_sha256"] in payloads:
            errors.append(f"duplicate MPEG payload: {row['track_id']}")
        payloads.add(row["audio_payload_sha256"])
        if float(row["average_bitrate_kbps"]) < minimum_bitrate_kbps:
            errors.append(f"bitrate below threshold: {row['track_id']}")
        if allowed and not (_genre_tokens(row) & allowed):
            errors.append(f"genre outside allowed set: {row['track_id']}")
        if allowed_mood_set and not (_mood_tokens(row) & allowed_mood_set):
            errors.append(f"mood outside allowed set: {row['track_id']}")
        if not row["matched_mood_tags"] or not (
            row["matched_genre_tags"] or row["matched_instrument_tags"]
        ):
            errors.append(f"insufficient tag evidence: {row['track_id']}")
    for identity_field in ("artist_key", "album_key"):
        splits: dict[str, set[str]] = defaultdict(set)
        for row in verified:
            splits[row[identity_field]].add(row["split"])
        for key, names in splits.items():
            if len(names) > 1:
                errors.append(f"split leakage: {identity_field}={key!r} in {sorted(names)}")
    rejected = Counter(row["download_status"] for row in rows)
    if rejected["rejected_bitrate"]:
        warnings.append(f"{rejected['rejected_bitrate']} downloads failed the bitrate threshold")
    return {
        "ok": not errors,
        "candidate_rows": len(rows),
        "verified_tracks": len(verified),
        "by_source": dict(Counter(row["source_dataset"] for row in verified)),
        "by_split": dict(Counter(row["split"] for row in verified)),
        "by_status": dict(rejected),
        "minimum_bitrate_kbps": minimum_bitrate_kbps,
        "allowed_genres": sorted(allowed),
        "allowed_moods": sorted(allowed_mood_set),
        "hashes_verified": verify_hash and not any("checksum mismatch" in item for item in errors),
        "errors": errors,
        "warnings": warnings,
    }


def export_preprocess_metadata(
    manifest: Path,
    output_dir: Path,
    data_root: Path,
    *,
    expected_count: int = 300,
    duration_corrections: Path | None = None,
) -> dict[str, int]:
    """Export verified open-Focus tracks as an isolated canonical metadata view."""

    rows = sorted(
        (row for row in _read_rows(manifest) if row["download_status"] == "verified"),
        key=lambda row: row["track_id"],
    )
    if len(rows) != expected_count:
        raise OpenFocusDatasetError(
            f"expected {expected_count} verified open-Focus tracks, found {len(rows)}"
        )
    corrected_durations: dict[str, str] = {}
    if duration_corrections is not None and duration_corrections.is_file():
        with duration_corrections.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"track_id", "corrected_duration_seconds"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise OpenFocusDatasetError(
                    f"{duration_corrections} is missing columns: {sorted(missing)}"
                )
            corrected_durations = {
                row["track_id"]: row["corrected_duration_seconds"] for row in reader
            }
    track_rows = [
        {
            "track_id": row["track_id"],
            "group": "focus",
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "duration_seconds": corrected_durations.get(
                row["track_id"], row["duration_seconds"]
            ),
            "sample_rate": row["sample_rate"],
            "channels": row["channels"],
            "artist_key": row["artist_key"],
            "album_key": row["album_key"],
            "composer_key": "",
            "instrumental": "true",
            "restricted": "false",
        }
        for row in rows
    ]
    license_rows = [
        {
            "track_id": row["track_id"],
            "group": "focus",
            "source_url": row["source_url"],
            "license_type": row["license_type"],
            "downloaded_at": row["downloaded_at"],
            "redistribution_allowed": "true",
            "notes": (
                "jamendo-api-open-focus; redistribution is subject to attribution and "
                "the per-track CC terms (including NC/ND/SA where applicable); "
                f"per-track license URL: {row['license_url']}"
            ),
        }
        for row in rows
    ]
    _write_table_atomic(output_dir / "track_index.csv", TRACK_INDEX_COLUMNS, track_rows)
    _write_table_atomic(output_dir / "licenses.csv", LICENSE_COLUMNS, license_rows)
    split_counts: Counter[str] = Counter()
    for split in ("discovery", "validation", "holdout"):
        members = [
            {"track_id": row["track_id"], "group": "focus"}
            for row in rows
            if row["split"] == split
        ]
        split_counts[split] = len(members)
        _write_table_atomic(output_dir / f"split_{split}.csv", ["track_id", "group"], members)

    from data.manifest import validate_metadata

    report = validate_metadata(output_dir, data_root, check_files=True)
    if not report.ok:
        raise OpenFocusDatasetError(
            "exported preprocessing metadata failed validation: " + "; ".join(report.errors)
        )
    return {"tracks": len(rows), **dict(split_counts)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-open")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-jamendo")
    select.add_argument("--output", type=Path, default=Path("metadata/focus_open_candidates.csv"))
    select.add_argument("--target-count", type=int, default=300)
    select.add_argument("--reserve-count", type=int, default=60)
    select.add_argument("--max-per-artist", type=int, default=4)
    select.add_argument("--max-per-album", type=int, default=2)
    select.add_argument("--max-pages", type=int, default=3)
    select.add_argument("--seed", type=int, default=20260801)
    select.add_argument("--fma-metadata-dir", type=Path)
    select.add_argument("--fma-audio-root", type=Path)
    select.add_argument("--jamendo-client-id-env", default="JAMENDO_CLIENT_ID")
    download = subparsers.add_parser("download")
    download.add_argument(
        "manifest", type=Path, nargs="?", default=Path("metadata/focus_open_candidates.csv")
    )
    download.add_argument("--data-root", type=Path, default=Path("data_raw"))
    download.add_argument("--target-count", type=int, default=300)
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--minimum-bitrate-kbps", type=float, default=300.0)
    download.add_argument("--jamendo-client-id-env", default="JAMENDO_CLIENT_ID")
    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "manifest", type=Path, nargs="?", default=Path("metadata/focus_open_candidates.csv")
    )
    audit.add_argument("--data-root", type=Path, default=Path("data_raw"))
    audit.add_argument("--target-count", type=int, default=300)
    audit.add_argument("--minimum-bitrate-kbps", type=float, default=300.0)
    audit.add_argument("--verify-hash", action="store_true")
    audit.add_argument("--allowed-genres", default="")
    audit.add_argument("--allowed-moods", default="")
    prepare = subparsers.add_parser("prepare-preprocess")
    prepare.add_argument(
        "manifest", type=Path, nargs="?", default=Path("metadata/focus_open_candidates.csv")
    )
    prepare.add_argument(
        "--output-dir", type=Path, default=Path("metadata/focus_open_preprocess")
    )
    prepare.add_argument("--data-root", type=Path, default=Path("data_raw"))
    prepare.add_argument("--expected-count", type=int, default=300)
    prepare.add_argument("--duration-corrections", type=Path)
    rebuild = subparsers.add_parser(
        "rebuild-local-style", aliases=["rebuild-local-tags"]
    )
    rebuild.add_argument(
        "manifest", type=Path, nargs="?", default=Path("metadata/focus_open_candidates.csv")
    )
    rebuild.add_argument(
        "--selected-dir", type=Path, default=Path("data_raw/focus_music")
    )
    rebuild.add_argument(
        "--excluded-dir", type=Path, default=Path("data_raw/focus_open_music_excluded")
    )
    rebuild.add_argument("--allowed-genres", default="")
    rebuild.add_argument("--allowed-moods", default="")
    rebuild.add_argument("--prefer-title-album-stem", default="")
    rebuild.add_argument("--import-id3-orphans-stem", default="")
    rebuild.add_argument("--force-include", action="append", default=[])
    rebuild.add_argument("--force-exclude", action="append", default=[])
    rebuild.add_argument("--target-count", type=int, default=300)
    rebuild.add_argument("--max-per-artist", type=int, default=4)
    rebuild.add_argument("--max-per-album", type=int, default=2)
    rebuild.add_argument("--seed", type=int, default=20260801)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "select-jamendo":
            rows = select_jamendo_focus(
                args.output,
                client_id=os.environ.get(args.jamendo_client_id_env, ""),
                target_count=args.target_count,
                reserve_count=args.reserve_count,
                max_per_artist=args.max_per_artist,
                max_per_album=args.max_per_album,
                max_pages=args.max_pages,
                seed=args.seed,
                fma_metadata_dir=args.fma_metadata_dir,
                fma_audio_root=args.fma_audio_root,
            )
            print(json.dumps({"candidates": len(rows), "output": str(args.output)}, indent=2))
        elif args.command == "download":
            counts = download_open_focus(
                args.manifest,
                args.data_root,
                client_id=os.environ.get(args.jamendo_client_id_env, ""),
                target_count=args.target_count,
                workers=args.workers,
                minimum_bitrate_kbps=args.minimum_bitrate_kbps,
            )
            print(json.dumps(dict(counts), indent=2))
        elif args.command == "audit":
            allowed_genres = [
                value.strip() for value in args.allowed_genres.split(",") if value.strip()
            ]
            allowed_moods = [
                value.strip() for value in args.allowed_moods.split(",") if value.strip()
            ]
            report = audit_open_focus(
                args.manifest,
                args.data_root,
                target_count=args.target_count,
                minimum_bitrate_kbps=args.minimum_bitrate_kbps,
                verify_hash=args.verify_hash,
                allowed_genres=allowed_genres,
                allowed_moods=allowed_moods,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        elif args.command == "prepare-preprocess":
            counts = export_preprocess_metadata(
                args.manifest,
                args.output_dir,
                args.data_root,
                expected_count=args.expected_count,
                duration_corrections=args.duration_corrections,
            )
            print(json.dumps(counts, indent=2))
        else:
            allowed_genres = [
                value.strip() for value in args.allowed_genres.split(",") if value.strip()
            ]
            allowed_moods = [
                value.strip() for value in args.allowed_moods.split(",") if value.strip()
            ]
            report = rebuild_local_style_selection(
                args.manifest,
                args.selected_dir,
                args.excluded_dir,
                allowed_genres=allowed_genres,
                allowed_moods=allowed_moods,
                prefer_title_album_stem=args.prefer_title_album_stem,
                import_id3_orphans_stem=args.import_id3_orphans_stem,
                force_include=args.force_include,
                force_exclude=args.force_exclude,
                target_count=args.target_count,
                max_per_artist=args.max_per_artist,
                max_per_album=args.max_per_album,
                seed=args.seed,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, OpenFocusDatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
