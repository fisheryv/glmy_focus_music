from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from data.controls import CANDIDATE_COLUMNS, _assign_classical_split

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
CLASSICAL = METADATA / "control_classical.csv"
SPLIT_NAMES = ("discovery", "validation", "holdout")
TARGETS = {"discovery": 195, "validation": 60, "holdout": 45}
SEED = 20260716


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path, columns: list[str] | tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_assignments() -> dict[str, tuple[str, str]]:
    assignments: dict[str, tuple[str, str]] = {}
    for split in SPLIT_NAMES:
        _, rows = _read_csv(METADATA / f"split_{split}.csv")
        for row in rows:
            track_id = row["track_id"]
            if track_id in assignments:
                raise RuntimeError(f"duplicate canonical split assignment: {track_id}")
            assignments[track_id] = (row["group"], split)
    return assignments


def _stale_manifests(classical_ids: set[str]) -> list[str]:
    stale: list[str] = []
    excluded = {
        CLASSICAL.resolve(),
        (METADATA / "classical_split_change_log.csv").resolve(),
        (METADATA / "split_assignment_v2.csv").resolve(),
    }
    for path in sorted(METADATA.glob("*.csv")):
        if path.resolve() in excluded or path.name.startswith("split_"):
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not {"track_id", "split"}.issubset(reader.fieldnames):
                    continue
                if any(row.get("track_id") in classical_ids for row in reader):
                    stale.append(path.relative_to(ROOT).as_posix())
        except UnicodeDecodeError, csv.Error:
            continue
    return stale


def _update_summaries() -> None:
    dataset_path = METADATA / "dataset_summary.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["groups"]["classical"].update(TARGETS)
    dataset["combined"].update(
        {
            "discovery": 390,
            "validation": 120,
            "holdout": 90,
        }
    )
    dataset["audit"].update(
        {
            "split_version": "symmetric_holdout_v2",
            "symmetric_group_counts": True,
            "downstream_artifacts_require_rerun": True,
        }
    )
    _write_json(dataset_path, dataset)

    control_path = METADATA / "control_dataset_summary.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["classical"].update(TARGETS)
    control["combined"].update(TARGETS)
    control["audit"].update(
        {
            "split_version": "symmetric_holdout_v2",
            "symmetric_with_open_focus": True,
            "downstream_artifacts_require_rerun": True,
        }
    )
    _write_json(control_path, control)


def main() -> int:
    columns, rows = _read_csv(CLASSICAL)
    if columns != list(CANDIDATE_COLUMNS):
        raise RuntimeError("control_classical.csv does not match CANDIDATE_COLUMNS")
    verified = [row for row in rows if row["download_status"] == "verified"]
    if len(rows) != 300 or len(verified) != 300:
        raise RuntimeError(
            f"expected 300 verified Classical rows, got rows={len(rows)}, verified={len(verified)}"
        )

    audit_path = METADATA / "symmetric_holdout_audit.json"
    current_counts = Counter(row["split"] for row in verified)
    if dict(current_counts) == TARGETS and audit_path.is_file():
        canonical = _canonical_assignments()
        canonical_counts: dict[str, Counter[str]] = {
            group: Counter() for group in ("focus", "classical")
        }
        for group, split in canonical.values():
            canonical_counts[group][split] += 1
        expected = {group: TARGETS for group in ("focus", "classical")}
        actual = {group: dict(counts) for group, counts in canonical_counts.items()}
        if actual == expected:
            print(audit_path.read_text(encoding="utf-8"), end="")
            return 0

    old_assignments = {row["track_id"]: row["split"] for row in rows}
    old_canonical = _canonical_assignments()
    old_focus = {
        track_id: split for track_id, (group, split) in old_canonical.items() if group == "focus"
    }

    _assign_classical_split(
        rows,
        validation_fraction=0.20,
        holdout_fraction=0.15,
        seed=SEED,
    )
    counts = Counter(row["split"] for row in verified)
    if dict(counts) != TARGETS:
        raise RuntimeError(f"Classical split targets were not met exactly: {dict(counts)}")

    composer_splits: dict[str, set[str]] = defaultdict(set)
    album_splits: dict[str, set[str]] = defaultdict(set)
    for row in verified:
        composer_splits[row["composer_key"]].add(row["split"])
        if row["album_key"]:
            album_splits[row["album_key"]].add(row["split"])
    leakage = {
        "composer": {
            key: sorted(values) for key, values in composer_splits.items() if len(values) > 1
        },
        "album": {key: sorted(values) for key, values in album_splits.items() if len(values) > 1},
    }
    if leakage["composer"] or leakage["album"]:
        raise RuntimeError(f"new Classical split leaks grouped identities: {leakage}")

    _write_csv(CLASSICAL, columns, rows)
    by_track = {row["track_id"]: row for row in rows}
    for split in SPLIT_NAMES:
        path = METADATA / f"split_{split}.csv"
        split_columns, existing = _read_csv(path)
        kept = [row for row in existing if row["group"] != "classical"]
        kept.extend(
            {"track_id": row["track_id"], "group": "classical"}
            for row in rows
            if row["split"] == split
        )
        kept.sort(key=lambda row: (row["group"], row["track_id"]))
        _write_csv(path, split_columns, kept)

    new_canonical = _canonical_assignments()
    new_focus = {
        track_id: split for track_id, (group, split) in new_canonical.items() if group == "focus"
    }
    if old_focus != new_focus:
        raise RuntimeError("Focus assignments changed while rebuilding Classical holdout")

    combined_counts: dict[str, Counter[str]] = {
        group: Counter() for group in ("focus", "classical")
    }
    assignment_rows: list[dict[str, str]] = []
    for track_id, (group, split) in sorted(new_canonical.items()):
        combined_counts[group][split] += 1
        assignment_rows.append({"track_id": track_id, "group": group, "split": split})
    expected = {group: TARGETS for group in ("focus", "classical")}
    actual = {group: dict(counts) for group, counts in combined_counts.items()}
    if actual != expected:
        raise RuntimeError(f"canonical symmetric split mismatch: {actual}")
    _write_csv(
        METADATA / "split_assignment_v2.csv",
        ["track_id", "group", "split"],
        assignment_rows,
    )

    changes: list[dict[str, str]] = []
    transition_counts: Counter[str] = Counter()
    for track_id, row in sorted(by_track.items()):
        old_split = old_assignments[track_id]
        new_split = row["split"]
        if old_split == new_split:
            continue
        transition_counts[f"{old_split}->{new_split}"] += 1
        changes.append(
            {
                "track_id": track_id,
                "composer_key": row["composer_key"],
                "album_key": row["album_key"],
                "subpool": row["subpool"],
                "old_split": old_split,
                "new_split": new_split,
            }
        )
    _write_csv(
        METADATA / "classical_split_change_log.csv",
        [
            "track_id",
            "composer_key",
            "album_key",
            "subpool",
            "old_split",
            "new_split",
        ],
        changes,
    )

    _update_summaries()
    stale = _stale_manifests(set(by_track))
    split_composers = {
        split: sorted(
            composer for composer, assigned in composer_splits.items() if assigned == {split}
        )
        for split in SPLIT_NAMES
    }
    split_subpools = {
        split: dict(
            sorted(Counter(row["subpool"] for row in verified if row["split"] == split).items())
        )
        for split in SPLIT_NAMES
    }
    audit_paths = [
        CLASSICAL,
        *(METADATA / f"split_{split}.csv" for split in SPLIT_NAMES),
        METADATA / "split_assignment_v2.csv",
        METADATA / "classical_split_change_log.csv",
        METADATA / "dataset_summary.json",
        METADATA / "control_dataset_summary.json",
    ]
    audit = {
        "generated_at": date.today().isoformat(),
        "split_version": "symmetric_holdout_v2",
        "seed": SEED,
        "targets_per_group": TARGETS,
        "actual_per_group": actual,
        "focus_assignments_unchanged": True,
        "classical_grouping_key": "composer_key",
        "classical_composers_by_split": split_composers,
        "classical_subpools_by_split": split_subpools,
        "classical_leakage": leakage,
        "changed_classical_tracks": len(changes),
        "transition_counts": dict(sorted(transition_counts.items())),
        "holdout_composition_warning": (
            "Classical holdout has no piano_solo tracks because exact 195/60/45 "
            "counts and composer-level isolation cannot place Bach or Beethoven "
            "outside discovery while retaining a representative validation split."
        ),
        "downstream_status": (
            "All preprocessed, feature, topology, statistics, and report artifacts "
            "built with the previous split are historical and require a full refit/rerun."
        ),
        "stale_metadata_manifests": stale,
        "sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in audit_paths},
        "ok": True,
    }
    _write_json(METADATA / "symmetric_holdout_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
