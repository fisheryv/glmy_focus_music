"""Build one group-first Hugging Face release for Focus and Classical."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_raw"
OUTPUT = ROOT / "hf_release" / "open-focus-classical-600"
FOCUS_MANIFEST = ROOT / "metadata" / "focus_open_candidates.csv"
CLASSICAL_MANIFEST = ROOT / "metadata" / "control_classical.csv"
LICENSES = ROOT / "metadata" / "licenses.csv"
SUMMARY = ROOT / "metadata" / "dataset_summary.json"
SPLITS = ("discovery", "validation", "holdout")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ordered_union(*field_sets: list[str]) -> list[str]:
    result: list[str] = []
    for fields in field_sets:
        for field in fields:
            if field not in result:
                result.append(field)
    return result


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty release directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    focus = [
        row
        for row in read_csv(FOCUS_MANIFEST)
        if row["selection_status"] == "selected" and row["download_status"] == "verified"
    ]
    classical = [
        row
        for row in read_csv(CLASSICAL_MANIFEST)
        if row["download_status"] == "verified"
    ]
    if len(focus) != 300 or len(classical) != 300:
        raise RuntimeError(f"Expected 300+300 tracks, found {len(focus)}+{len(classical)}")

    fields = ["file_name", *ordered_union(list(focus[0]), list(classical[0]))]
    exported: list[dict[str, str]] = []
    per_group_split: dict[tuple[str, str], list[dict[str, str]]] = {
        (group, split): [] for group in ("focus", "classical") for split in SPLITS
    }
    checksum_lines: list[str] = []

    for group, rows in (("focus", focus), ("classical", classical)):
        for row in rows:
            source = DATA_ROOT / row["relative_path"]
            if not source.is_file():
                raise FileNotFoundError(source)
            if group == "focus" and source.suffix.lower() != ".mp3":
                raise RuntimeError(f"Focus source is not MP3: {source}")
            relative = Path("data") / group / row["split"] / f"{row['track_id']}{source.suffix.lower()}"
            target = OUTPUT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)

            global_row = dict(row)
            global_row["file_name"] = relative.as_posix()
            exported.append(global_row)
            local_row = dict(global_row)
            local_row["file_name"] = target.name
            per_group_split[(group, row["split"])].append(local_row)
            checksum_lines.append(f"{row['sha256']}  {relative.as_posix()}")

    write_csv(OUTPUT / "metadata" / "tracks.csv", exported, fields)
    write_csv(OUTPUT / "metadata" / "focus" / "source_manifest.csv", focus, list(focus[0]))
    write_csv(
        OUTPUT / "metadata" / "classical" / "source_manifest.csv",
        classical,
        list(classical[0]),
    )
    for (group, split), rows in per_group_split.items():
        write_csv(OUTPUT / "data" / group / split / "metadata.csv", rows, fields)

    track_ids = {row["track_id"] for row in exported}
    licenses = [row for row in read_csv(LICENSES) if row["track_id"] in track_ids]
    if len(licenses) != 600 or any(row["redistribution_allowed"] != "true" for row in licenses):
        raise RuntimeError("Combined license ledger is incomplete or contains a blocked track")
    write_csv(OUTPUT / "metadata" / "licenses.csv", licenses, list(licenses[0]))

    source_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    release_summary = {
        "release_type": "group-first raw-audio",
        "audio_included": True,
        "audio_bytes_are_identical_to_audited_sources": True,
        "audio_processing": "none",
        "configs": ["paired", "focus", "classical"],
        "groups": source_summary["groups"],
        "combined": source_summary["combined"],
    }
    (OUTPUT / "metadata" / "dataset_summary.json").write_text(
        json.dumps(release_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    (OUTPUT / ".gitattributes").write_text(
        "*.mp3 filter=lfs diff=lfs merge=lfs -text\n"
        "*.m4a filter=lfs diff=lfs merge=lfs -text\n"
        "*.wav filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text(DATASET_CARD, encoding="utf-8")
    print(OUTPUT)


DATASET_CARD = """---
pretty_name: Open Focus and Classical 600
license: other
task_categories:
- audio-classification
tags:
- music
- focus-music
- classical-music
configs:
- config_name: paired
  default: true
  data_files:
  - split: discovery
    path:
    - data/focus/discovery/**
    - data/classical/discovery/**
  - split: validation
    path:
    - data/focus/validation/**
    - data/classical/validation/**
  - split: holdout
    path:
    - data/focus/holdout/**
    - data/classical/holdout/**
- config_name: focus
  data_files:
  - split: discovery
    path: data/focus/discovery/**
  - split: validation
    path: data/focus/validation/**
  - split: holdout
    path: data/focus/holdout/**
- config_name: classical
  data_files:
  - split: discovery
    path: data/classical/discovery/**
  - split: validation
    path: data/classical/validation/**
  - split: holdout
    path: data/classical/holdout/**
---

# Open Focus and Classical 600

This repository contains two independently usable but analysis-aligned music collections. The default `paired` configuration loads both groups; the `focus` and `classical` configurations load either group independently. Each configuration preserves discovery, validation, and holdout splits.

```python
from datasets import load_dataset

paired = load_dataset("OWNER/open-focus-classical-600", "paired")
focus = load_dataset("OWNER/open-focus-classical-600", "focus")
classical = load_dataset("OWNER/open-focus-classical-600", "classical")
```

## Contents

- Open Focus: 300 original MP3 files, 29.50 hours, split 195/60/45.
- Classical: 300 original WAV/M4A/MP3 files, 28.31 hours, split 195/60/45.
- Paired: 600 tracks, 57.81 hours, split 390/120/90.

All audio files are byte-identical hard-linked copies of the audited source downloads. No file has been trimmed, transcoded, resampled, normalized, retagged, or otherwise modified. `SHA256SUMS` covers all 600 published audio files.

## License boundary

There is no single blanket license for this repository. Every track retains its own status in `metadata/licenses.csv`.

Open Focus contains 227 CC BY-NC-ND, 41 CC BY-NC-SA, 18 CC BY-SA, 8 CC BY-NC, and 6 CC BY tracks, with several license versions represented. Classical contains 205 MusicNet recordings under CC BY 4.0 and 95 Musopen recordings marked Public Domain Mark 1.0. Public Domain Mark is not a license and status can differ by jurisdiction.

Users must follow the per-track attribution, non-commercial, no-derivatives, and share-alike terms where applicable. Modified versions of CC BY-NC-ND recordings must not be redistributed without separate permission.

## Metadata and provenance

`metadata/tracks.csv` is the unified index. Group-specific source manifests remain available under `metadata/focus/` and `metadata/classical/`. Per-split `metadata.csv` files support independent Hugging Face loading without erasing group-specific provenance.

Open Focus selection used Jamendo instrumental and very-low/low-speed evidence plus focus-related metadata. These functional tags are not evidence of clinical, cognitive, therapeutic, or causal effects. The Classical corpus is a constrained research control, not a representative survey of all classical music.
"""


if __name__ == "__main__":
    main()
