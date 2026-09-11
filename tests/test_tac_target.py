from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generation.tac_target import TACTopologyTarget, sha256_file

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "metadata" / "tac_topology_target_v1.json"
SCORES = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_scores.csv"
FEATURE_ORDER = [
    *[f"pitch_whitened_{index:02d}" for index in range(16)],
    "path_acoustic_phase__loop_score",
    "path_chroma_phase__loop_score",
]


def test_target_is_hash_bound_and_diagnostic_only() -> None:
    payload = json.loads(TARGET.read_text(encoding="utf-8"))
    target = TACTopologyTarget.from_json(TARGET, root=ROOT)

    assert target.payload == payload
    assert payload["target_id"] == "tac_topology_target_v1"
    assert payload["dimensions"] == 18
    assert payload["reference_count"] == 195
    assert payload["distance_definition"]["weights"] == [0.5, 0.25, 0.25]
    assert payload["blocks"]["pitch"]["indices"] == list(range(3, 16))
    assert payload["blocks"]["pitch"]["inactive_indices"] == [0, 1, 2]
    assert payload["fingerprint_sha256"] == sha256_file(
        ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
    )
    for relative, expected in payload["source_sha256"].items():
        assert sha256_file(ROOT / relative) == expected
    assert payload["evidence"] == {
        "role": "development_diagnostic",
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }


def test_reference_distance_statistics_recompute_exactly() -> None:
    target = TACTopologyTarget.from_json(TARGET, root=ROOT)
    scores = pd.read_csv(SCORES)
    reference = scores[
        (scores["split"] == "discovery")
        & (scores["group"] == "focus")
        & np.isclose(scores["scale_seconds"], 180.0)
    ]
    coordinates = reference.loc[:, FEATURE_ORDER].to_numpy(float)
    distances = target.distance(coordinates)
    expected = target.payload["reference_distance"]

    assert len(distances) == 195
    assert np.isfinite(distances).all()
    assert (distances >= 0.0).all()
    assert float(np.min(distances)) == pytest.approx(expected["minimum"], abs=1e-14)
    assert float(np.max(distances)) == pytest.approx(expected["maximum"], abs=1e-14)
    assert float(np.mean(distances)) == pytest.approx(expected["mean"], abs=1e-14)
    for text, value in expected["quantiles"].items():
        assert float(np.quantile(distances, float(text))) == pytest.approx(value, abs=1e-14)


def test_reward_is_positive_only_when_target_distance_decreases() -> None:
    target = TACTopologyTarget.from_json(TARGET, root=ROOT)
    payload = target.payload
    center = np.zeros(18, dtype=float)
    pitch = payload["blocks"]["pitch"]
    center[np.asarray(pitch["indices"], dtype=int)] = pitch["center"]
    center[16] = payload["blocks"]["path_acoustic_phase"]["center"][0]
    center[17] = payload["blocks"]["path_chroma_phase"]["center"][0]
    farther = center.copy()
    farther[3] += pitch["scale"][0]
    farther[16] += payload["blocks"]["path_acoustic_phase"]["scale"][0]

    np.testing.assert_allclose(target.distance(center), [0.0], atol=1e-15)
    assert target.reward(farther, center)[0] > 0.0
    assert target.reward(center, farther)[0] < 0.0
