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
SEGMENTS = ROOT / "metadata" / "rhythm_topology_segments.csv"
SENSITIVITY = ROOT / "metadata" / "rhythm_topology_filtration_sensitivity.csv"
TESTS = ROOT / "metadata" / "rhythm_statistical_tests.csv"
PAIRWISE = ROOT / "metadata" / "rhythm_pairwise_tests.csv"
SUMMARY = ROOT / "metadata" / "rhythm_analysis_summary.json"


def _h1_incidence(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    subset = frame[
        (frame["split"] == "validation")
        & np.isclose(frame["scale_seconds"].astype(float), 180.0)
    ]
    output: dict[str, dict[str, int]] = {}
    for group, rows in subset.groupby("group"):
        output[str(group)] = {
            "nonzero": int((rows["h1_betti_max"].astype(float) > 0).sum()),
            "total": int(len(rows)),
        }
    return output


def _sensitivity_h1_incidence(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    subset = frame[
        (frame["split"] == "validation")
        & np.isclose(frame["scale_seconds"].astype(float), 180.0)
    ]
    maxima = subset.groupby(["segment_id", "group"], as_index=False)["h1_betti"].max()
    output: dict[str, dict[str, int]] = {}
    for group, rows in maxima.groupby("group"):
        output[str(group)] = {
            "nonzero": int((rows["h1_betti"].astype(float) > 0).sum()),
            "total": int(len(rows)),
        }
    return output


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _mechanism_example(segments: pd.DataFrame) -> dict[str, Any]:
    subset = segments[
        (segments["split"] == "validation")
        & np.isclose(segments["scale_seconds"].astype(float), 180.0)
    ]
    candidates: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        arrays = _read_npz(ROOT / str(row.sensitivity_persistence_relative_path))
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
        raise RuntimeError("no finite H1 interval found for a validation/180s mechanism example")
    candidates.sort(
        key=lambda item: (
            item["group"] != "focus",
            item["finite_h1_intervals"] != 1,
            -item["lifetime"],
            item["segment_id"],
        )
    )
    selected = candidates[0]
    selected["selection_rule"] = (
        "prefer Focus validation/180s with exactly one finite sensitivity-filtration "
        "H1 interval; then largest lifetime; deterministic segment-id tie break"
    )
    return selected


def main() -> int:
    segments = pd.read_csv(SEGMENTS)
    sensitivity = pd.read_csv(SENSITIVITY)
    expected_groups = {"classical": 600, "focus": 600}
    group_counts = segments["group"].value_counts().to_dict()
    if (
        len(segments) != 1200
        or segments["track_id"].nunique() != 600
        or group_counts != expected_groups
        or (segments["status"] == "failed").any()
    ):
        raise RuntimeError(
            "canonical open-dataset rhythm rerun is incomplete: "
            f"rows={len(segments)}, tracks={segments['track_id'].nunique()}, "
            f"groups={group_counts}"
        )

    tests, pairwise = _omnibus_and_pairwise(
        segments,
        bootstrap_resamples=3000,
        bootstrap_seed=20260716,
    )
    tests.to_csv(TESTS, index=False, encoding="utf-8", lineterminator="\n")
    pairwise.to_csv(PAIRWISE, index=False, encoding="utf-8", lineterminator="\n")

    primary = tests[tests["analysis_set"] == "primary_validation_180"]
    focus_classical = pairwise[
        (pairwise["analysis_set"] == "primary_validation_180")
        & (pairwise["group_a"].isin(["classical", "focus"]))
        & (pairwise["group_b"].isin(["classical", "focus"]))
    ]
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
    for metric in (
        "vertex_count",
        "edge_count",
        "edge_density",
        "self_transition_ratio",
        "path_entropy",
        "directed_recurrence",
        "reciprocity",
        "h0_betti_mean",
        "h1_betti_max",
    ):
        row = primary[primary["metric"] == metric].iloc[0]
        medians[metric] = {
            group: float(row[f"{group}_median"])
            for group in ("classical", "focus")
        }

    summary = {
        "generated_at": date.today().isoformat(),
        "scope": "rhythm-only Path Homology analysis on Focus/Classical dataset",
        "canonical_focus_source": "Jamendo Open Focus",
        "canonical_groups": ["classical", "focus"],
        "ok": True,
        "segment_views": int(len(segments)),
        "tracks": int(segments["track_id"].nunique()),
        "graph_input": "adjacent frozen state transitions",
        "ssm_used": False,
        "primary_validation_n": int(
            len(
                segments[
                    (segments["split"] == "validation")
                    & np.isclose(segments["scale_seconds"].astype(float), 180.0)
                ]
            )
        ),
        "confirmatory_fdr_q": CONFIRMATORY_FDR_Q,
        "primary_fdr_discoveries": int(
            (primary["p_fdr_bh"] <= CONFIRMATORY_FDR_Q).sum()
        ),
        "primary_tests": int(len(primary)),
        "sensitivity_fdr_discoveries": int(
            (sensitivity_tests["p_fdr_bh"] <= CONFIRMATORY_FDR_Q).sum()
        ),
        "replicated_same_direction": int(replicated_same_direction),
        "focus_classical_fdr_discoveries": int(
            (focus_classical["p_fdr_bh"] <= CONFIRMATORY_FDR_Q).sum()
        ),
        "validation_180_group_medians": medians,
        "validation_180_h1_counts": _h1_incidence(segments),
        "validation_180_sensitivity_h1_counts": _sensitivity_h1_incidence(sensitivity),
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
