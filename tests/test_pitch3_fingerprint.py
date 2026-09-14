from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generation.pitch3_contract import (
    PITCH3_FEATURE_ORDER,
    PITCH3_INPUT_FEATURES,
    load_pitch3_contract,
)
from generation.pitch3_exact_scorer import ExactPitch3Scorer
from topology.metrics import TOPOLOGY_METRICS

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"
SCORES = ROOT / "metadata" / "focus_pitch3_fingerprint_v1_scores.csv"
SUMMARY = ROOT / "metadata" / "focus_pitch3_fingerprint_v1_summary.json"
RELEASE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1_release.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pitch3_profile_is_a_signed_development_only_contract() -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    contract = load_pitch3_contract(PROFILE)

    assert contract.input_features == PITCH3_INPUT_FEATURES
    assert contract.feature_order == PITCH3_FEATURE_ORDER
    assert payload["dimensions"] == 3
    assert payload["expected_focus_direction"] == ["less", "greater", "greater"]
    assert payload["reference_split"] == "discovery"
    assert payload["reference_scale_seconds"] == 180.0
    assert payload["reference_sample_count"] == 390
    assert payload["evidence"]["selection_status"] == "post-validation downstream refreeze"
    assert payload["runtime_status"]["sampling_guidance"] == (
        "disabled_until_new_qualification_passes"
    )


def test_exact_pitch3_scorer_reconstructs_all_published_coordinates() -> None:
    source = pd.read_csv(ROOT / "metadata" / "pitch_v2_topology_segments.csv")
    published = pd.read_csv(SCORES)
    scorer = ExactPitch3Scorer.from_json(PROFILE)

    score = scorer.score(source.loc[:, TOPOLOGY_METRICS].to_numpy(float))

    np.testing.assert_allclose(
        score.coordinates,
        published.loc[:, PITCH3_FEATURE_ORDER].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(score.focus_logit, published["focus_logit"], atol=1e-12)
    np.testing.assert_allclose(
        score.focus_probability, published["focus_probability"], atol=1e-12
    )
    np.testing.assert_allclose(
        score.target_band_loss, published["focus_target_band_loss"], atol=1e-12
    )


def test_pitch3_release_hashes_and_evidence_boundaries() -> None:
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert release["profile_sha256"] == _sha256(PROFILE)
    assert release["release_status"] == "issued_for_ltsn_development_only"
    assert release["signing_gates"]["guidance_qualification"] == "not_run"
    for relative, expected in release["artifact_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    assert summary["authorization"] == {
        "scientific_claim": "post_validation_downstream_target_only",
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    assert summary["validation"]["primary_validation_180"]["balanced_accuracy"] > 0.90
    assert min(summary["duration_stability_spearman"].values()) > 0.95
