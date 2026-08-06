from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TrackGroup(str, Enum):
    FOCUS = "focus"
    POP = "pop"
    CLASSICAL = "classical"


class SplitName(str, Enum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


@dataclass(frozen=True, slots=True)
class TrackRecord:
    track_id: str
    group: TrackGroup
    relative_path: Path
    sha256: str = ""
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    artist_key: str = ""
    album_key: str = ""
    composer_key: str = ""
    instrumental: bool = False
    restricted: bool = False


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    track_id: str
    group: TrackGroup
    source_url: str
    license_type: str
    downloaded_at: str
    redistribution_allowed: bool
    notes: str = ""

