from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST = ROOT / "metadata" / "preprocessed_segments.csv"
ASSIGNMENTS = ROOT / "metadata" / "split_assignment_v2.csv"
NEW_MANIFEST = ROOT / "metadata" / "preprocessed_segments_v2_seed.csv"
PROVENANCE = ROOT / "metadata" / "preprocessed_segments_v2_materialization.json"
OUTPUT_ROOT = Path("features/audio_symmetric_holdout_v2")
EXPECTED_PER_SCALE = {
    ("classical", "discovery"): 195,
    ("classical", "validation"): 60,
    ("classical", "holdout"): 45,
    ("focus", "discovery"): 195,
    ("focus", "validation"): 60,
    ("focus", "holdout"): 45,
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scale_token(raw: str) -> str:
    value = float(raw)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def _materialize(source: Path, destination: Path, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing destination size mismatch: {destination}")
        if _sha256(destination) != expected_sha256:
            raise RuntimeError(f"existing destination hash mismatch: {destination}")
        return "verified_existing"
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy"
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"materialized size mismatch: {destination}")
    if _sha256(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"materialized hash mismatch: {destination}")
    return method


def main() -> int:
    columns, rows = _read_csv(OLD_MANIFEST)
    _, assignment_rows = _read_csv(ASSIGNMENTS)
    assignments = {row["track_id"]: (row["group"], row["split"]) for row in assignment_rows}
    if len(assignments) != 600:
        raise RuntimeError(f"expected 600 assignments, got {len(assignments)}")
    if len(rows) != 1200:
        raise RuntimeError(f"expected 1200 prior segments, got {len(rows)}")
    if len({row["segment_id"] for row in rows}) != len(rows):
        raise RuntimeError("prior manifest has duplicate segment IDs")

    methods: Counter[str] = Counter()
    counts: Counter[tuple[str, str, str]] = Counter()
    migrated: list[dict[str, str]] = []
    for completed, row in enumerate(rows, start=1):
        track_id = row["track_id"]
        if track_id not in assignments:
            raise RuntimeError(f"missing v2 assignment: {track_id}")
        group, split = assignments[track_id]
        if group != row["group"]:
            raise RuntimeError(f"group mismatch for {track_id}: {row['group']} != {group}")
        token = _scale_token(row["scale_seconds"])
        destination_relative = (
            OUTPUT_ROOT / f"{token}s" / group / split / f"{row['segment_id']}.wav"
        )
        source = ROOT / Path(row["output_relative_path"])
        destination = ROOT / destination_relative
        if not source.is_file():
            raise RuntimeError(f"missing prior WAV: {source}")
        method = _materialize(source, destination, row["sha256"])
        methods[method] += 1
        updated = dict(row)
        updated["split"] = split
        updated["output_relative_path"] = destination_relative.as_posix()
        updated["status"] = "verified_materialized"
        updated["error"] = ""
        migrated.append(updated)
        counts[(token, group, split)] += 1
        if completed == 1 or completed % 100 == 0 or completed == len(rows):
            print(f"materialize: {completed}/{len(rows)}", flush=True)

    for token in ("180", "300"):
        actual = {
            (group, split): counts[(token, group, split)]
            for group, split in EXPECTED_PER_SCALE
        }
        if actual != EXPECTED_PER_SCALE:
            raise RuntimeError(f"unexpected {token}s split counts: {actual}")

    migrated.sort(key=lambda row: (row["group"], row["track_id"], float(row["scale_seconds"])))
    _write_csv(NEW_MANIFEST, columns, migrated)
    payload = {
        "generated_at": date.today().isoformat(),
        "split_version": "symmetric_holdout_v2",
        "source_manifest": OLD_MANIFEST.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": _sha256(OLD_MANIFEST),
        "assignment_manifest": ASSIGNMENTS.relative_to(ROOT).as_posix(),
        "assignment_manifest_sha256": _sha256(ASSIGNMENTS),
        "seed_manifest": NEW_MANIFEST.relative_to(ROOT).as_posix(),
        "seed_manifest_sha256": _sha256(NEW_MANIFEST),
        "output_root": OUTPUT_ROOT.as_posix(),
        "segments": len(migrated),
        "methods": dict(sorted(methods.items())),
        "counts_by_scale_group_split": {
            f"{token}s/{group}/{split}": counts[(token, group, split)]
            for token in ("180", "300")
            for group, split in EXPECTED_PER_SCALE
        },
        "scientific_note": (
            "Audio samples are byte-identical to the previously verified preprocessing. "
            "Only the versioned path and symmetric_holdout_v2 split assignment changed."
        ),
        "ok": True,
    }
    _write_json(PROVENANCE, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
