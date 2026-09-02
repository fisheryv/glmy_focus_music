from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "metadata" / "holdout_gate.json"
VIEWS = {
    "pitch": {
        "segments": ROOT / "metadata" / "pitch_v2_topology_segments.csv",
        "tests": ROOT / "metadata" / "pitch_v2_statistical_tests.csv",
        "summary": ROOT / "metadata" / "pitch_v2_summary.json",
    },
    "rhythm": {
        "segments": ROOT / "metadata" / "rhythm_topology_segments.csv",
        "tests": ROOT / "metadata" / "rhythm_statistical_tests.csv",
        "summary": ROOT / "metadata" / "rhythm_analysis_summary.json",
    },
    "modulation": {
        "segments": ROOT / "metadata" / "modulation_smp_prototype_topology_segments.csv",
        "tests": ROOT / "metadata" / "modulation_smp_prototype_statistical_tests.csv",
        "summary": ROOT / "metadata" / "modulation_smp_prototype_summary.json",
    },
    "structure": {
        "segments": ROOT / "metadata" / "structure_topology_segments.csv",
        "tests": ROOT / "metadata" / "structure_statistical_tests.csv",
        "summary": ROOT / "metadata" / "structure_analysis_summary.json",
    },
}
MODEL_PATHS = (
    ROOT / "features" / "models" / "state_model.npz",
    ROOT / "features" / "models" / "state_model.json",
    ROOT / "features" / "models" / "pitch_v2_codebook.npz",
    ROOT / "features" / "models" / "pitch_v2_codebook.json",
    ROOT / "features" / "models" / "modulation_smp_shared_transform.npz",
    ROOT / "features" / "models" / "modulation_smp_shared_transform.json",
    ROOT / "features" / "models" / "modulation_smp_proto_k10.npz",
    ROOT / "features" / "models" / "modulation_smp_proto_k10.json",
)
INPUT_PATHS = (
    ROOT / "metadata" / "track_index.csv",
    ROOT / "metadata" / "split_discovery.csv",
    ROOT / "metadata" / "split_validation.csv",
    ROOT / "metadata" / "split_holdout.csv",
    ROOT / "metadata" / "preprocessed_segments.csv",
    ROOT / "metadata" / "feature_segments.csv",
    *(payload["segments"] for payload in VIEWS.values()),
)
VALIDATION_PATHS = (
    *(payload["tests"] for payload in VIEWS.values()),
    *(payload["summary"] for payload in VIEWS.values()),
    ROOT / "metadata" / "multiview_fusion_permanova.csv",
    ROOT / "metadata" / "multiview_fusion_incremental.csv",
    ROOT / "metadata" / "multiview_fusion_classification.csv",
    ROOT / "metadata" / "multiview_fusion_summary.json",
)
CONFIG_PATHS = (
    ROOT / "configs" / "pipeline.toml",
    ROOT / "scripts" / "run_pitch_v2_analysis.py",
    ROOT / "scripts" / "rerun_rhythm_path_homology.py",
    ROOT / "scripts" / "analyze_rhythm_results.py",
    ROOT / "scripts" / "run_modulation_smp_prototype_analysis.py",
    ROOT / "scripts" / "rerun_structure_path_homology.py",
    ROOT / "scripts" / "analyze_structure_results.py",
    ROOT / "scripts" / "run_multiview_fusion_analysis.py",
    ROOT / "scripts" / "run_holdout_confirmation.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze missing artifacts: {missing}")
    return {_relative(path): _sha256(path) for path in paths}


def _selected_directional_metrics() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for view, payload in VIEWS.items():
        with payload["tests"].open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["analysis_set"] != "primary_validation_180":
                continue
            if view == "modulation" and int(row["state_count"]) != 10:
                continue
            if float(row["p_fdr_bh"]) > 0.05:
                continue
            classical = float(row["classical_median"])
            focus = float(row["focus_median"])
            if focus == classical:
                continue
            selected.append(
                {
                    "view": view,
                    "metric": row["metric"],
                    "expected_focus_direction": "greater" if focus > classical else "less",
                    "validation_classical_median": classical,
                    "validation_focus_median": focus,
                    "validation_p_fdr_bh": float(row["p_fdr_bh"]),
                }
            )
    return sorted(selected, key=lambda row: (row["view"], row["metric"]))


def main() -> int:
    if GATE.exists():
        raise RuntimeError(f"holdout gate already exists: {GATE}")
    selected = _selected_directional_metrics()
    if not selected:
        raise RuntimeError("validation selected no directional metrics for holdout confirmation")
    payload = {
        "schema_version": 1,
        "frozen_at": date.today().isoformat(),
        "split_version": "open-focus-classical-600-frozen-release",
        "status": "frozen_before_holdout_statistical_testing",
        "scientific_scope": (
            "Operational/descriptive rerun only. These holdout tracks were accessed in prior "
            "work, so this is not pristine independent confirmation and cannot upgrade the "
            "validation evidence tier."
        ),
        "analysis_specification": {
            "primary_scale_seconds": 180.0,
            "duration_sensitivity_scale_seconds": 300.0,
            "groups": ["classical", "focus"],
            "expected_holdout_per_group_per_scale": 45,
            "topology_metrics_per_view": 20,
            "block_transform": "discovery-fitted rank-normalized Mahalanobis",
            "primary_feature_set": "local",
            "primary_test": "two-group permutation pseudo-F",
            "primary_alpha": 0.05,
            "local_views": ["pitch", "rhythm", "modulation"],
            "local_weights": {"pitch": 1 / 3, "rhythm": 1 / 3, "modulation": 1 / 3},
            "secondary_feature_sets": [
                "pitch",
                "rhythm",
                "modulation",
                "structure",
                "hierarchical",
            ],
            "hierarchical_weights": {"local": 0.5, "structure": 0.5},
            "incremental_tests": ["local_vs_pitch", "add_structure"],
            "permutations": 999,
            "seed": 20260716,
            "secondary_fdr": "Benjamini-Hochberg within each locked family and scale",
            "secondary_q": 0.05,
            "directional_metric_selection": (
                "all validation/180s single-view metrics with validation BH q <= 0.05; "
                "expected direction fixed from validation medians"
            ),
            "directional_metrics": selected,
            "holdout_prohibitions": [
                "no parameter refitting",
                "no metric reselection",
                "no direction changes",
                "no fusion-weight changes",
                "no threshold changes",
                "no FDR-family changes",
            ],
        },
        "config_sha256": _hashes(CONFIG_PATHS),
        "input_sha256": _hashes(INPUT_PATHS),
        "model_sha256": _hashes(MODEL_PATHS),
        "validation_artifact_sha256": _hashes(VALIDATION_PATHS),
    }
    temporary = GATE.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, GATE)
    print(json.dumps({**payload, "gate_sha256": _sha256(GATE)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
