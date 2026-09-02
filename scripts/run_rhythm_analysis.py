from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from data.analysis_inputs import audit_analysis_inputs
from features.batch import _sha256
from topology.batch import (
    _config_hash,
    _load_feature_jobs,
    load_topology_config,
    run_topology_batch,
)
from topology.rhythm_statistics import run_statistics

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "metadata" / "rhythm_topology_segments.csv"
FILTRATION = ROOT / "metadata" / "rhythm_topology_filtration.csv"
SENSITIVITY = ROOT / "metadata" / "rhythm_topology_filtration_sensitivity.csv"
SUMMARY = ROOT / "metadata" / "rhythm_topology_summary.json"


def run_topology() -> int:
    input_audit = audit_analysis_inputs(root=ROOT)
    base_config = load_topology_config(ROOT)
    config = replace(base_config, views=("rhythm",))
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
    group_counts = dict(Counter(row["group"] for row in successful))
    expected_groups = {"classical": 600, "focus": 600}
    complete = (
        len(rows) == 1200
        and len({row["track_id"] for row in rows}) == 600
        and group_counts == expected_groups
        and len(successful) == len(rows)
    )
    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "rhythm-only Path Homology topology stage on Focus/Classical dataset",
        "canonical_groups": ["classical", "focus"],
        "input_provenance": input_audit,
        "ok": complete,
        "segment_views": len(rows),
        "segments": len({row["segment_id"] for row in rows}),
        "tracks": len({row["track_id"] for row in rows}),
        "status_counts": dict(Counter(row.get("status", "") for row in rows)),
        "group_counts": group_counts,
        "split_counts": dict(Counter(row["split"] for row in successful)),
        "scale_counts": dict(
            Counter(f"{int(float(row['scale_seconds']))}s" for row in successful)
        ),
        "h1_nonzero_segment_views": sum(
            float(row["h1_betti_max"]) > 0 for row in successful
        ),
        "state_model_sha256": json.loads(
            (ROOT / "features" / "models" / "state_model.json").read_text(
                encoding="utf-8"
            )
        )["model_sha256"],
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "artifacts": {
            "segments": SEGMENTS.relative_to(ROOT).as_posix(),
            "filtration": FILTRATION.relative_to(ROOT).as_posix(),
            "sensitivity_filtration": SENSITIVITY.relative_to(ROOT).as_posix(),
        },
    }
    summary["artifact_sha256"] = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (SEGMENTS, FILTRATION, SENSITIVITY)
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


def main() -> int:
    topology_status = run_topology()
    if topology_status != 0:
        return topology_status
    return run_statistics()


if __name__ == "__main__":
    raise SystemExit(main())
