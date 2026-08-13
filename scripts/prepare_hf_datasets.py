"""Prepare auditable Hugging Face releases for Open Focus and Classical.

Open Focus is metadata-only because most tracks are CC BY-NC-ND. Classical
contains byte-identical hard links to the audited source audio.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCUS_MANIFEST = ROOT / "metadata" / "focus_open_candidates.csv"
CLASSICAL_MANIFEST = ROOT / "metadata" / "control_classical.csv"
LICENSES = ROOT / "metadata" / "licenses.csv"
SUMMARY = ROOT / "metadata" / "dataset_summary.json"
DATA_ROOT = ROOT / "data_raw"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty release directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_common_files(repo: Path) -> None:
    (repo / ".gitattributes").write_text(
        "*.mp3 filter=lfs diff=lfs merge=lfs -text\n"
        "*.m4a filter=lfs diff=lfs merge=lfs -text\n"
        "*.flac filter=lfs diff=lfs merge=lfs -text\n"
        "*.wav filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def filtered_licenses(track_ids: set[str]) -> list[dict[str, str]]:
    return [row for row in read_csv(LICENSES) if row["track_id"] in track_ids]


def prepare_focus(repo: Path, summary: dict[str, object]) -> None:
    ensure_empty(repo)
    rows = [
        row
        for row in read_csv(FOCUS_MANIFEST)
        if row["selection_status"] == "selected" and row["download_status"] == "verified"
    ]
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 verified selected Focus tracks, found {len(rows)}")

    track_ids = {row["track_id"] for row in rows}
    licenses = filtered_licenses(track_ids)
    if len(licenses) != 300 or any(row["redistribution_allowed"] != "true" for row in licenses):
        raise RuntimeError("Focus license ledger is incomplete or contains a blocked track")

    exported: list[dict[str, str]] = []
    checksum_lines: list[str] = []
    by_split: dict[str, list[dict[str, str]]] = {
        name: [] for name in ("discovery", "validation", "holdout")
    }
    for row in rows:
        source = DATA_ROOT / row["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".mp3":
            raise RuntimeError(f"Open Focus source is not MP3: {source}")
        relative = Path("data") / row["split"] / f"{row['track_id']}.mp3"
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        item = dict(row)
        item["file_name"] = relative.as_posix()
        exported.append(item)
        split_item = dict(item)
        split_item["file_name"] = target.name
        by_split[row["split"]].append(split_item)
        checksum_lines.append(f"{row['sha256']}  {relative.as_posix()}")

    fields = ["file_name", *list(rows[0])]
    write_csv(repo / "metadata" / "tracks.csv", exported, fields)
    write_csv(repo / "metadata" / "licenses.csv", licenses, list(licenses[0]))
    for split, split_rows in by_split.items():
        write_csv(repo / "data" / split / "metadata.csv", split_rows, fields)
    (repo / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    write_csv(
        repo / "metadata" / "source_audio_sha256.csv",
        [
            {
                "track_id": row["track_id"],
                "split": row["split"],
                "sha256": row["sha256"],
                "audio_payload_sha256": row["audio_payload_sha256"],
                "source_url": row["source_url"],
            }
            for row in rows
        ],
        ["track_id", "split", "sha256", "audio_payload_sha256", "source_url"],
    )
    release_summary = {
        "release_type": "raw-audio",
        "audio_included": True,
        "audio_bytes_are_identical_to_audited_sources": True,
        "audio_processing": "none",
        "group": summary["groups"]["focus"],
    }
    (repo / "metadata" / "dataset_summary.json").write_text(
        json.dumps(release_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (repo / "README.md").write_text(FOCUS_CARD, encoding="utf-8")
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "download_from_jamendo.py").write_text(
        FOCUS_DOWNLOADER, encoding="utf-8"
    )
    write_common_files(repo)


def prepare_classical(repo: Path, summary: dict[str, object]) -> None:
    ensure_empty(repo)
    rows = [row for row in read_csv(CLASSICAL_MANIFEST) if row["download_status"] == "verified"]
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 verified Classical tracks, found {len(rows)}")

    track_ids = {row["track_id"] for row in rows}
    licenses = filtered_licenses(track_ids)
    if len(licenses) != 300 or any(row["redistribution_allowed"] != "true" for row in licenses):
        raise RuntimeError("Classical license ledger is incomplete or contains a blocked track")

    exported: list[dict[str, str]] = []
    checksum_lines: list[str] = []
    by_split: dict[str, list[dict[str, str]]] = {name: [] for name in ("discovery", "validation", "holdout")}
    for row in rows:
        source = DATA_ROOT / row["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        relative = Path("data") / row["split"] / f"{row['track_id']}{suffix}"
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        item = dict(row)
        item["file_name"] = relative.as_posix()
        exported.append(item)
        split_item = dict(item)
        split_item["file_name"] = target.name
        by_split[row["split"]].append(split_item)
        checksum_lines.append(f"{row['sha256']}  {relative.as_posix()}")

    fields = ["file_name", *list(rows[0])]
    write_csv(repo / "metadata" / "tracks.csv", exported, fields)
    write_csv(repo / "metadata" / "licenses.csv", licenses, list(licenses[0]))
    for split, split_rows in by_split.items():
        write_csv(repo / "data" / split / "metadata.csv", split_rows, fields)
    (repo / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    release_summary = {
        "release_type": "raw-audio",
        "audio_included": True,
        "audio_bytes_are_identical_to_audited_sources": True,
        "group": summary["groups"]["classical"],
    }
    (repo / "metadata" / "dataset_summary.json").write_text(
        json.dumps(release_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (repo / "README.md").write_text(CLASSICAL_CARD, encoding="utf-8")
    write_common_files(repo)


FOCUS_CARD = """---
pretty_name: Open Focus 300
license: other
task_categories:
- audio-classification
tags:
- music
- focus-music
configs:
- config_name: default
  data_files:
  - split: discovery
    path: data/discovery/**
  - split: validation
    path: data/validation/**
  - split: holdout
    path: data/holdout/**
---

# Open Focus 300

This release contains 300 instrumental Open Focus tracks selected from Jamendo: 195 discovery, 60 validation, and 45 holdout tracks (29.50 hours). Every MP3 is byte-identical to the audited source download. Audio has not been trimmed, transcoded, resampled, normalized, retagged, or otherwise modified.

## License boundary

There is no single blanket license for all tracks. Every track retains the license in `metadata/licenses.csv`. The collection includes 227 CC BY-NC-ND tracks, 41 CC BY-NC-SA tracks, 18 CC BY-SA tracks, 8 CC BY-NC tracks, and 6 CC BY tracks, with several license versions represented. Users must follow the per-track attribution, non-commercial, no-derivatives, and share-alike requirements as applicable. The raw audio is redistributed unchanged; processed audio is intentionally excluded.

## Integrity

`SHA256SUMS` records every published MP3 hash. `metadata/source_audio_sha256.csv` also records MPEG-payload hashes. Any downstream modification must be stored separately and must comply with the relevant per-track license; CC BY-NC-ND tracks must not be redistributed in modified form without separate permission.

## Reproduction

`scripts/download_from_jamendo.py` requires the user's own read-only `JAMENDO_CLIENT_ID`. It accepts a download only when the resulting file matches the frozen SHA-256 in `metadata/tracks.csv`; changed or unavailable upstream files are reported rather than silently substituted.

## Provenance and limitations

Selection used Jamendo instrumental and very-low/low-speed evidence plus focus, meditation, relaxing, study, and deep-work-related metadata. Eight rows retain the provenance label `jamendo-api-open-focus-early-local-candidate`; this label must not be interpreted as fresh API verification. Functional tags are not evidence of clinical, cognitive, therapeutic, or causal effects.
"""


CLASSICAL_CARD = """---
pretty_name: Classical Control 300
license: other
task_categories:
- audio-classification
tags:
- music
- classical-music
configs:
- config_name: default
  data_files:
  - split: discovery
    path: data/discovery/**
  - split: validation
    path: data/validation/**
  - split: holdout
    path: data/holdout/**
---

# Classical Control 300

This release contains 300 byte-identical source recordings (28.31 hours): 195 discovery, 60 validation, and 45 holdout tracks. The split is grouped by composer/work to reduce leakage.

## Sources and licenses

- 205 MusicNet recordings from Zenodo record 5120004: CC BY 4.0.
- 95 Musopen Kickstarter recordings from the Musopen Lossless DVD or its standard-DVD fallback: Public Domain Mark 1.0.

There is no new blanket license over the audio. Each recording retains the status in `metadata/licenses.csv`; attribution, source URL, and license URL are recorded in `metadata/tracks.csv`. Public Domain Mark is not a license and public-domain status can differ by jurisdiction.

## Integrity

Audio bytes were not transcoded, trimmed, normalized, or otherwise modified. `SHA256SUMS` records every published file hash. The repository excludes all analysis-time preprocessed WAV files.

## Intended use and limitations

The corpus is an observational research control dataset, not a representative survey of all classical music. Its composer, work, ensemble, recording-era, and source distributions are intentionally constrained and must be considered in downstream interpretation.
"""


FOCUS_DOWNLOADER = r'''"""Reconstruct Open Focus from Jamendo and verify frozen hashes."""

import csv
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "tracks.csv"
OUTPUT = ROOT / "downloaded_audio"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    client_id = os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        raise SystemExit("Set your own JAMENDO_CLIENT_ID environment variable.")
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        target = OUTPUT / row["split"] / f"{row['track_id']}.mp3"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == row["sha256"]:
            print("verified", row["track_id"])
            continue
        part = target.with_suffix(".part")
        url = row["download_url"].replace("{client_id}", client_id)
        request = Request(url, headers={"User-Agent": "open-focus-repro/1.0"})
        try:
            with urlopen(request, timeout=120) as response, part.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            actual = sha256(part)
            if actual != row["sha256"]:
                part.unlink(missing_ok=True)
                print("HASH_MISMATCH", row["track_id"], actual)
                continue
            part.replace(target)
            print("downloaded", row["track_id"])
        except Exception as exc:
            part.unlink(missing_ok=True)
            print("ERROR", row["track_id"], type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=ROOT / "hf_release")
    parser.add_argument(
        "--dataset", choices=("all", "focus", "classical"), default="all"
    )
    args = parser.parse_args()
    release_root = args.release_root.resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if args.dataset in ("all", "focus"):
        prepare_focus(release_root / "open-focus-300", summary)
        print(release_root / "open-focus-300")
    if args.dataset in ("all", "classical"):
        prepare_classical(release_root / "classical-control-300", summary)
        print(release_root / "classical-control-300")


if __name__ == "__main__":
    main()
