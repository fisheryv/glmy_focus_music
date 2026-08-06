"""Metadata, licensing, and split validation."""

from .manifest import ValidationReport, validate_metadata
from .schema import LicenseRecord, SplitName, TrackGroup, TrackRecord

__all__ = [
    "LicenseRecord",
    "SplitName",
    "TrackGroup",
    "TrackRecord",
    "ValidationReport",
    "validate_metadata",
]

