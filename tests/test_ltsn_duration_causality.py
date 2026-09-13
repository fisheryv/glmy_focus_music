from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from generation.ltsn_duration_causality import (
    RADII,
    analyze_duration_response_points,
    build_duration_causality_report,
    prepare_duration_plan,
)
from generation.ltsn_exact_labeling import _batch_input_sha256


def _points(duration: int, *, unstable: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor in range(16):
        common = {
            "duration_seconds": duration,
            "anchor_sample_id": f"anchor_{duration}_{anchor}",
            "matched_anchor_index": anchor,
            "prompt_id": f"prompt_{anchor}",
            "seed": 1000 + anchor,
            "step_number": 5,
        }
        rows.append(
            {
                **common,
                "sample_id": f"center_{duration}_{anchor}",
                "direction_index": -1,
                "rms_ratio": 0.0,
                "sign": 0.0,
                "target_distance": 2.0,
                "pitch_distance": 2.0,
                "path_acoustic_phase_distance": 2.0,
                "path_chroma_phase_distance": 2.0,
                "ood_label": "false",
            }
        )
        for direction in range(4):
            for radius_index, radius in enumerate(RADII):
                response = 2.0 + direction / 10.0
                if unstable and radius_index == 0 and (anchor + direction) % 2 == 0:
                    response *= -1.0
                for sign in (-1.0, 1.0):
                    distance = 2.0 - sign * radius * response + radius * radius
                    rows.append(
                        {
                            **common,
                            "sample_id": f"p_{duration}_{anchor}_{direction}_{radius}_{sign}",
                            "direction_index": direction,
                            "rms_ratio": radius,
                            "sign": sign,
                            "target_distance": distance,
                            "pitch_distance": distance * 2.0,
                            "path_acoustic_phase_distance": distance * 2.0,
                            "path_chroma_phase_distance": distance * 2.0,
                            "ood_label": "false",
                        }
                    )
    return rows


def test_native_duration_response_uses_64_independent_directions() -> None:
    report, outcomes = analyze_duration_response_points(_points(30))

    assert report["anchors"] == 16
    assert report["directions"] == 64
    assert report["pairs"] == 192
    assert report["cross_radius_sign"]["comparisons"] == 128
    assert report["cross_radius_sign"]["agreement"] == 1.0
    assert report["diagnostic_only"] is True
    assert report["qualification_eligible"] is False
    assert len(outcomes) == 192


def test_matched_cluster_report_detects_short_duration_improvement(tmp_path: Path) -> None:
    reports = {}
    for duration, unstable in ((180, True), (60, False), (30, False)):
        report, _ = analyze_duration_response_points(_points(duration, unstable=unstable))
        path = tmp_path / f"{duration}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        reports[duration] = path

    payload = build_duration_causality_report(
        duration_report_paths=reports,
        output_path=tmp_path / "report.json",
        bootstrap_resamples=200,
    )

    assert payload["status"] == "duration_causality_supported"
    assert payload["matched_cluster_comparisons"]["60s_vs_180s"]["gates"] == {
        "agreement_at_least_0_80": True,
        "wilson_lower_above_0_50": True,
        "improvement_over_180s_at_least_0_10": True,
        "cluster_bootstrap_lower_above_0": True,
    }
    assert payload["qualification_eligible"] is False


def test_duration_prompt_plan_is_family_balanced_and_seed_matched(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.csv"
    fields = ["prompt_id", "caption", "split", "seed", "bpm", "keyscale", "timesignature"]
    with prompt_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in range(4):
            for variant in range(6):
                writer.writerow(
                    {
                        "prompt_id": f"p{family:02d}__v{variant:02d}",
                        "caption": "matched prompt",
                        "split": "development",
                        "seed": "",
                        "bpm": "80",
                        "keyscale": "",
                        "timesignature": "4",
                    }
                )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "experiment": "ltsn_duration_causality_v1",
                "durations_seconds": [180, 60, 30],
                "anchor_step": 5,
                "anchors": 16,
                "directions_per_anchor": 4,
                "rms_ratios": [0.0025, 0.005, 0.0075],
                "reference_rms_ratio": 0.005,
                "phase_block_frames": 2,
                "phase_min_raw_steps": 36,
            }
        ),
        encoding="utf-8",
    )

    payload = prepare_duration_plan(
        prompt_manifest_path=prompt_path,
        output_dir=tmp_path / "out",
        protocol_path=protocol,
        seed_start=9000,
    )

    rows = list(csv.DictReader((tmp_path / "out" / "duration_prompts.csv").open()))
    assert payload["anchors"] == 16
    assert {row["base_family"] for row in rows} == {"p00", "p01", "p02", "p03"}
    assert [int(row["seed"]) for row in rows] == list(range(9000, 9016))


def test_short_duration_changes_exact_batch_resume_identity() -> None:
    rows = [
        {
            "sample_id": "a",
            "audio_sha256": "a" * 64,
            "latent_sha256": "b" * 64,
        }
    ]
    legacy = _batch_input_sha256(rows)
    assert legacy == _batch_input_sha256(rows, duration_seconds=180.0)
    assert legacy != _batch_input_sha256(rows, duration_seconds=60.0)
