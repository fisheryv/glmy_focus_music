from __future__ import annotations

from typing import Any

from generation.tac_v52c import V52C_RADII, analyze_response_points


def _synthetic_points(*, flip_small_radius: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor_index in range(4):
        anchor_id = f"anchor_{anchor_index}"
        rows.append(
            {
                "sample_id": anchor_id,
                "anchor_sample_id": anchor_id,
                "direction_index": -1,
                "step_number": 4,
                "rms_ratio": 0.0,
                "sign": 0.0,
                "target_distance": 2.0,
                "pitch_distance": 2.0,
                "path_acoustic_phase_distance": 2.0,
                "path_chroma_phase_distance": 2.0,
                "ood_label": "false",
            }
        )
        for direction in range(8):
            for radius in V52C_RADII:
                response = 3.0 + direction / 10.0
                if flip_small_radius and radius == V52C_RADII[0]:
                    response *= -1.0
                for sign in (-1.0, 1.0):
                    distance = 2.0 - sign * radius * response + radius * radius
                    rows.append(
                        {
                            "sample_id": f"{anchor_id}_{direction}_{radius}_{sign}",
                            "anchor_sample_id": anchor_id,
                            "direction_index": direction,
                            "step_number": 4,
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


def test_stable_response_curve_passes_frozen_stage1_gates() -> None:
    report, outcomes = analyze_response_points(_synthetic_points())

    assert len(outcomes) == 4 * 8 * 5
    assert report["cross_radius_sign"]["agreement"] == 1.0
    assert report["gates"] == {
        "cross_radius_sign_agreement_at_least_0_80": True,
        "cross_radius_wilson_lower_above_0_50": True,
    }
    assert report["stage1_status"] == "supported_for_critic_collection"
    assert report["next_stage_authorized"] is True


def test_radius_sign_flip_fails_without_lowering_gate() -> None:
    report, _ = analyze_response_points(_synthetic_points(flip_small_radius=True))

    assert report["cross_radius_sign"]["agreement"] == 0.75
    assert report["stage1_status"] == "not_supported"
    assert report["next_stage_authorized"] is False
