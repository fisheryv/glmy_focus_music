from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.batch import _sha256
from topology.statistics import _omnibus_and_pairwise

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_FDR_Q = 0.05
SEGMENTS = ROOT / "metadata" / "structure_topology_segments.csv"
SENSITIVITY = ROOT / "metadata" / "structure_topology_filtration_sensitivity.csv"
TESTS = ROOT / "metadata" / "structure_statistical_tests.csv"
PAIRWISE = ROOT / "metadata" / "structure_pairwise_tests.csv"
SUMMARY = ROOT / "metadata" / "structure_analysis_summary.json"

METRICS = (
    "vertex_count",
    "edge_count",
    "edge_density",
    "self_transition_ratio",
    "path_entropy",
    "directed_recurrence",
    "reciprocity",
    "h0_betti_mean",
    "h1_betti_max",
    "h1_observed_persistence",
)


def _incidence(frame: pd.DataFrame, value: str) -> dict[str, dict[str, int]]:
    subset = frame[
        (frame["split"] == "validation")
        & np.isclose(frame["scale_seconds"].astype(float), 180.0)
    ]
    if value == "h1_betti":
        subset = subset.groupby(["segment_id", "group"], as_index=False)[value].max()
    return {
        str(group): {
            "nonzero": int((rows[value].astype(float) > 0).sum()),
            "total": int(len(rows)),
        }
        for group, rows in subset.groupby("group")
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _mechanism_example(segments: pd.DataFrame) -> dict[str, Any]:
    subset = segments[
        (segments["split"] == "validation")
        & np.isclose(segments["scale_seconds"].astype(float), 180.0)
    ]
    candidates: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        arrays = _load_npz(ROOT / str(row.sensitivity_persistence_relative_path))
        dimensions = arrays["interval_dimension"].astype(int)
        censored = arrays["interval_censored"].astype(bool)
        finite_h1 = np.flatnonzero((dimensions == 1) & ~censored)
        if finite_h1.size == 0:
            continue
        best = int(finite_h1[np.argmax(arrays["interval_lifetime"][finite_h1])])
        candidates.append(
            {
                "segment_id": str(row.segment_id),
                "track_id": str(row.track_id),
                "group": str(row.group),
                "split": str(row.split),
                "scale_seconds": int(float(row.scale_seconds)),
                "finite_h1_intervals": int(finite_h1.size),
                "birth_threshold": float(arrays["interval_birth_threshold"][best]),
                "death_threshold": float(arrays["interval_death_threshold"][best]),
                "lifetime": float(arrays["interval_lifetime"][best]),
            }
        )
    if not candidates:
        return {
            "available": False,
            "finite_h1_intervals": 0,
            "analysis_set": "validation/180s sensitivity filtration",
            "interpretation": (
                "No finite H1 interval was observed in either group; no mechanism "
                "example was selected."
            ),
        }
    candidates.sort(
        key=lambda item: (
            item["group"] != "focus",
            item["finite_h1_intervals"] != 1,
            -item["lifetime"],
            item["segment_id"],
        )
    )
    selected = candidates[0]
    selected["available"] = True
    selected["selection_rule"] = (
        "prefer Focus validation/180s finite sensitivity-filtration H1 intervals; "
        "if none exist, use either group; then prefer exactly one interval, largest "
        "lifetime, and deterministic segment-id tie break"
    )
    return selected


def main() -> int:
    segments = pd.read_csv(SEGMENTS)
    sensitivity = pd.read_csv(SENSITIVITY)
    group_counts = segments["group"].value_counts().to_dict()
    if (
        len(segments) != 1_200
        or segments["track_id"].nunique() != 600
        or group_counts != {"classical": 600, "focus": 600}
        or (segments["status"] == "failed").any()
    ):
        raise RuntimeError(
            "two-group structure rerun is incomplete: "
            f"rows={len(segments)}, tracks={segments['track_id'].nunique()}, "
            f"groups={group_counts}"
        )

    tests, pairwise = _omnibus_and_pairwise(segments)
    tests.to_csv(TESTS, index=False, encoding="utf-8", lineterminator="\n")
    pairwise.to_csv(PAIRWISE, index=False, encoding="utf-8", lineterminator="\n")
    primary = tests[tests["analysis_set"] == "primary_validation_180"].copy()
    primary_pairwise = pairwise[
        pairwise["analysis_set"] == "primary_validation_180"
    ].copy()
    sensitivity_tests = tests[tests["analysis_set"] == "sensitivity_validation_300"]
    significant_primary = primary[
        primary["p_fdr_bh"] <= CONFIRMATORY_FDR_Q
    ]
    sensitivity_by_metric = sensitivity_tests.set_index("metric")
    replicated_same_direction = 0
    for row in significant_primary.itertuples(index=False):
        other = sensitivity_by_metric.loc[row.metric]
        direction_180 = np.sign(float(row.focus_median) - float(row.classical_median))
        direction_300 = np.sign(
            float(other["focus_median"]) - float(other["classical_median"])
        )
        if (
            float(other["p_fdr_bh"]) <= CONFIRMATORY_FDR_Q
            and direction_180 == direction_300
        ):
            replicated_same_direction += 1
    medians: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        row = primary[primary["metric"] == metric].iloc[0]
        medians[metric] = {
            group: float(row[f"{group}_median"])
            for group in ("classical", "focus")
        }

    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "structure-only Path Homology analysis on Focus/Classical dataset",
        "canonical_groups": ["classical", "focus"],
        "ok": True,
        "segments": int(len(segments)),
        "tracks": int(segments["track_id"].nunique()),
        "primary_validation_n": int(
            len(
                segments[
                    (segments["split"] == "validation")
                    & np.isclose(segments["scale_seconds"].astype(float), 180.0)
                ]
            )
        ),
        "primary_tests": int(len(primary)),
        "confirmatory_fdr_q": CONFIRMATORY_FDR_Q,
        "primary_fdr_discoveries_q_0_05": int(
            (primary["p_fdr_bh"] <= CONFIRMATORY_FDR_Q).sum()
        ),
        "sensitivity_fdr_discoveries_q_0_05": int(
            (sensitivity_tests["p_fdr_bh"] <= CONFIRMATORY_FDR_Q).sum()
        ),
        "primary_pairwise_fdr_discoveries_q_0_05": int(
            (primary_pairwise["p_fdr_bh"] <= CONFIRMATORY_FDR_Q).sum()
        ),
        "primary_fdr_discoveries_q_0_10": int((primary["p_fdr_bh"] <= 0.10).sum()),
        "sensitivity_fdr_discoveries_q_0_10": int(
            (sensitivity_tests["p_fdr_bh"] <= 0.10).sum()
        ),
        "replicated_same_direction": int(replicated_same_direction),
        "primary_pairwise_fdr_discoveries_q_0_10": int(
            (primary_pairwise["p_fdr_bh"] <= 0.10).sum()
        ),
        "validation_180_group_medians": medians,
        "validation_180_h1_counts": _incidence(segments, "h1_betti_max"),
        "validation_180_sensitivity_h1_counts": _incidence(
            sensitivity, "h1_betti"
        ),
        "mechanism_example": _mechanism_example(segments),
        "artifacts": {
            "tests": TESTS.relative_to(ROOT).as_posix(),
            "pairwise": PAIRWISE.relative_to(ROOT).as_posix(),
        },
    }
    summary["artifact_sha256"] = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (TESTS, PAIRWISE)
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
