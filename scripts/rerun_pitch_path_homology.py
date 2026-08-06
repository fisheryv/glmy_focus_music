from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from topology.batch import (
    _config_hash,
    _load_feature_jobs,
    load_topology_config,
    run_topology_batch,
)

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "metadata" / "pitch_topology_segments.csv"
FILTRATION = ROOT / "metadata" / "pitch_topology_filtration.csv"
SENSITIVITY = ROOT / "metadata" / "pitch_topology_filtration_sensitivity.csv"
SUMMARY = ROOT / "metadata" / "pitch_topology_summary.json"


def main() -> int:
    base_config = load_topology_config(ROOT)
    config = replace(base_config, views=("pitch",))
    jobs = _load_feature_jobs(ROOT / "metadata" / "feature_segments.csv", config)
    rows = run_topology_batch(
        jobs,
        root=ROOT,
        config=config,
        workers=6,
        overwrite=True,
        segment_manifest=SEGMENTS,
        filtration_manifest=FILTRATION,
        sensitivity_filtration_manifest=SENSITIVITY,
    )
    successful = [row for row in rows if row.get("status") != "failed"]
    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "pitch-only independent rerun",
        "ok": len(successful) == len(rows),
        "segment_views": len(rows),
        "segments": len({row["segment_id"] for row in rows}),
        "tracks": len({row["track_id"] for row in rows}),
        "status_counts": dict(Counter(row.get("status", "") for row in rows)),
        "group_counts": dict(Counter(row["group"] for row in successful)),
        "split_counts": dict(Counter(row["split"] for row in successful)),
        "scale_counts": dict(Counter(f"{int(float(row['scale_seconds']))}s" for row in successful)),
        "h1_nonzero_segment_views": sum(
            float(row["h1_betti_max"]) > 0 for row in successful
        ),
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "artifacts": {
            "segments": SEGMENTS.relative_to(ROOT).as_posix(),
            "filtration": FILTRATION.relative_to(ROOT).as_posix(),
            "sensitivity_filtration": SENSITIVITY.relative_to(ROOT).as_posix(),
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
