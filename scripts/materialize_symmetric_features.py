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
OLD_MANIFEST = ROOT / "metadata" / "feature_segments.csv"
PREPROCESSED = ROOT / "metadata" / "preprocessed_segments_v2.csv"
NEW_MANIFEST = ROOT / "metadata" / "feature_segments_v2_seed.csv"
PROVENANCE = ROOT / "metadata" / "feature_segments_v2_materialization.json"
VIEWS = ("acoustic", "chroma", "rhythm", "modulation", "structure")
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if source == destination:
        if not source.is_file():
            raise RuntimeError(f"missing feature file: {source}")
        return "same_path"
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
    _, preprocessed_rows = _read_csv(PREPROCESSED)
    preprocessed = {row["segment_id"]: row for row in preprocessed_rows}
    if len(rows) != 1200 or len(preprocessed) != 1200:
        raise RuntimeError(
            f"expected 1200 feature and preprocessing rows, got {len(rows)} and "
            f"{len(preprocessed)}"
        )
    if len({row["segment_id"] for row in rows}) != len(rows):
        raise RuntimeError("prior feature manifest has duplicate segment IDs")

    methods: Counter[str] = Counter()
    counts: Counter[tuple[str, str, str]] = Counter()
    migrated: list[dict[str, str]] = []
    for completed, row in enumerate(rows, start=1):
        segment_id = row["segment_id"]
        prep = preprocessed.get(segment_id)
        if prep is None:
            raise RuntimeError(f"missing v2 preprocessing row: {segment_id}")
        if prep["group"] != row["group"]:
            raise RuntimeError(f"group mismatch for {segment_id}")
        token = _scale_token(row["scale_seconds"])
        group = prep["group"]
        split = prep["split"]
        updated = dict(row)
        updated["split"] = split
        updated["input_relative_path"] = prep["output_relative_path"]
        updated["input_sha256"] = prep["sha256"]

        output_payload: dict[str, dict[str, Any]] = {}
        for view in VIEWS:
            source = ROOT / Path(row[f"{view}_relative_path"])
            destination_relative = (
                Path("features")
                / view
                / f"{token}s"
                / group
                / split
                / f"{segment_id}.npz"
            )
            destination = ROOT / destination_relative
            method = _materialize(source, destination, row[f"{view}_sha256"])
            methods[method] += 1
            updated[f"{view}_relative_path"] = destination_relative.as_posix()
            output_payload[view] = {
                "relative_path": destination_relative.as_posix(),
                "sha256": row[f"{view}_sha256"],
            }

        sidecar_source = ROOT / Path(row["sidecar_relative_path"])
        sidecar = json.loads(sidecar_source.read_text(encoding="utf-8"))
        sidecar_relative = (
            Path("features")
            / "manifests"
            / f"{token}s"
            / group
            / split
            / f"{segment_id}.json"
        )
        sidecar["group"] = group
        sidecar["split"] = split
        sidecar["input_relative_path"] = prep["output_relative_path"]
        sidecar["input_sha256"] = prep["sha256"]
        sidecar["sidecar_relative_path"] = sidecar_relative.as_posix()
        for view in VIEWS:
            sidecar["outputs"][view]["relative_path"] = output_payload[view]["relative_path"]
            sidecar["outputs"][view]["sha256"] = output_payload[view]["sha256"]
        _write_json(ROOT / sidecar_relative, sidecar)
        updated["sidecar_relative_path"] = sidecar_relative.as_posix()
        updated["status"] = "verified_materialized"
        updated["error"] = ""
        migrated.append(updated)
        counts[(token, group, split)] += 1
        if completed == 1 or completed % 100 == 0 or completed == len(rows):
            print(f"materialize features: {completed}/{len(rows)}", flush=True)

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
        "preprocessed_manifest": PREPROCESSED.relative_to(ROOT).as_posix(),
        "preprocessed_manifest_sha256": _sha256(PREPROCESSED),
        "seed_manifest": NEW_MANIFEST.relative_to(ROOT).as_posix(),
        "seed_manifest_sha256": _sha256(NEW_MANIFEST),
        "segments": len(migrated),
        "files": len(migrated) * len(VIEWS),
        "methods": dict(sorted(methods.items())),
        "counts_by_scale_group_split": {
            f"{token}s/{group}/{split}": counts[(token, group, split)]
            for token in ("180", "300")
            for group, split in EXPECTED_PER_SCALE
        },
        "scientific_note": (
            "Continuous feature arrays are byte-identical to the prior extraction. "
            "State arrays remain historical until the v2 discovery-only model is refit "
            "and all segments are transformed."
        ),
        "ok": True,
    }
    _write_json(PROVENANCE, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
