from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from generation.tac_v52d import (
    V52D_ALL_BASES,
    V52D_CFG_STATUS,
    V52D_ON_MANIFOLD_BASES,
    V52D_RANDOM_BASES,
    V52D_SCALES,
    analyze_actions,
    analyze_repeatability,
    build_action_bases,
)


def test_action_bases_are_deterministic_unit_rms_and_matched() -> None:
    rng = np.random.default_rng(20260912)
    history = rng.normal(size=(5, 96, 64))

    first, audit = build_action_bases(history, seed=17)
    second, second_audit = build_action_bases(history, seed=17)

    assert tuple(first) == V52D_ALL_BASES
    assert audit == second_audit
    assert audit["cfg_residual"] == V52D_CFG_STATUS
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert all(np.sqrt(np.mean(value * value)) == pytest.approx(1.0) for value in first.values())
    matrix = np.stack([first[name].reshape(-1) for name in V52D_ALL_BASES])
    cosine = matrix @ matrix.T
    norms = np.sqrt(np.sum(matrix * matrix, axis=1))
    cosine /= norms[:, None] * norms[None, :]
    np.testing.assert_allclose(cosine, np.eye(6), atol=1e-12)


def _repeatability_rows(*, changed: bool = False) -> list[dict[str, Any]]:
    rows = []
    for anchor_index in range(8):
        for repeat in range(3):
            for extraction in range(2):
                coordinate = float(changed and anchor_index == 0 and repeat == 2)
                rows.append(
                    {
                        "anchor_sample_id": f"anchor_{anchor_index}",
                        "rollout_repeat": repeat,
                        "extraction_repeat": extraction,
                        "coordinates_json": json.dumps([coordinate] + [0.0] * 17),
                        "target_distance": coordinate,
                        "final_latent_sha256": (
                            f"latent_{anchor_index}_{int(changed and repeat == 2)}"
                        ),
                        "audio_sha256": f"audio_{anchor_index}_{int(changed and repeat == 2)}",
                        "descriptor_signature_sha256": f"descriptor_{coordinate}",
                        "ood_label": 0,
                    }
                )
    return rows


def test_repeatability_gate_requires_bitwise_full_pipeline_reproduction() -> None:
    passed = analyze_repeatability(_repeatability_rows())
    failed = analyze_repeatability(_repeatability_rows(changed=True))

    assert passed["repeatability_status"] == "passed"
    assert passed["action_collection_authorized"] is True
    assert passed["target_distance_noise_q95"] == 0.0
    assert failed["repeatability_status"] == "failed"
    assert failed["action_collection_authorized"] is False


def _action_rows() -> list[dict[str, Any]]:
    rows = []
    for anchor_index in range(8):
        for basis in V52D_ALL_BASES:
            for scale in V52D_SCALES:
                stable = basis in V52D_ON_MANIFOLD_BASES or scale == 0.5
                response_sign = 1.0 if stable else -1.0
                for sign in (-1.0, 1.0):
                    rows.append(
                        {
                            "anchor_sample_id": f"anchor_{anchor_index}",
                            "basis_name": basis,
                            "action_scale": scale,
                            "sign": sign,
                            "target_distance": 2.0 - sign * scale * response_sign,
                            "ood_label": 0,
                        }
                    )
    return rows


def test_on_manifold_actions_must_beat_one_to_one_random_controls() -> None:
    repeatability = analyze_repeatability(_repeatability_rows())
    report, outcomes = analyze_actions(_action_rows(), repeatability_report=repeatability)

    assert len(outcomes) == 8 * 6 * 3
    assert report["on_manifold_cross_scale"]["agreement"] == 1.0
    assert report["random_cross_scale_agreement"] == 0.0
    assert report["matched_control_mcnemar"]["on_manifold_only_stable"] == 48
    assert report["matched_control_mcnemar"]["random_only_stable"] == 0
    assert report["stage1d_status"] == "supported_for_critic_collection"
    assert report["next_stage_authorized"] is True
    assert set(report["action_bases"]["matched_random_control"]) == set(V52D_RANDOM_BASES)
