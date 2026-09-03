from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generation.ltsn_contract import load_fingerprint_contract
from generation.path_homology_exact_scorer import ExactPathHomologyScorer
from topology.statistics import TOPOLOGY_METRICS

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
SCORES = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_scores.csv"
DIRECTIONS = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_directions.csv"
SUMMARY = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_summary.json"
RELEASE = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_release.json"
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
FEATURE_ORDER = tuple(
    [f"pitch_whitened_{index:02d}" for index in range(16)]
    + ["path_acoustic_phase__loop_score", "path_chroma_phase__loop_score"]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_profile_is_signed_frozen_18d_path_homology() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    contract = load_fingerprint_contract(PROFILE)

    assert profile["fingerprint_id"] == "focus_path_homology_fingerprint_v2"
    assert profile["spec_revision"] == "2026-08-17-frozen-18d"
    assert profile["dimensions"] == 18
    assert tuple(profile["feature_order"]) == FEATURE_ORDER
    assert profile["distance_weights"] == [0.5, 0.25, 0.25]
    assert set(profile["block_transforms"]) == {
        "pitch",
        "path_acoustic_phase",
        "path_chroma_phase",
    }
    assert profile["block_transforms"]["pitch"]["output_dimensions"] == 16
    assert profile["block_transforms"]["pitch"]["effective_rank"] == 13
    assert len(profile["classifier_coef"]) == 18
    assert profile["classifier_sha256"] == contract.classifier_sha256
    assert profile["contains_tda_features"] is False
    assert profile["runtime_status"]["exact_scoring"] == "enabled"
    assert profile["runtime_status"]["legacy_51d_scorer"] == "reject"
    assert profile["runtime_status"]["sampling_guidance"] == "disabled_until_all_gates_pass"


def test_v2_scores_and_directional_signature_are_complete() -> None:
    scores = pd.read_csv(SCORES)
    directions = pd.read_csv(DIRECTIONS)

    assert len(scores) == 1200
    assert scores["segment_id"].nunique() == 1200
    assert set(scores["split"]) == {"discovery", "validation", "holdout"}
    assert set(FEATURE_ORDER).issubset(scores.columns)
    assert np.isfinite(scores.loc[:, FEATURE_ORDER].to_numpy(float)).all()
    assert scores["focus_probability"].between(0.0, 1.0).all()
    assert (scores["focus_band_loss"] >= 0.0).all()
    assert len(directions) == 15
    assert directions.groupby("layer").size().to_dict() == {"phase": 2, "pitch": 13}
    forbidden = r"rhythm|modulation|structure|tda"
    assert not directions["view"].str.contains(forbidden, case=False, regex=True).any()


def test_runtime_exact_scorer_reconstructs_published_scores() -> None:
    pitch = pd.read_csv(ROOT / "metadata" / "pitch_v2_topology_segments.csv")
    pitch = pitch.set_index(IDENTITY).sort_index()
    phase = pd.read_csv(ROOT / "metadata" / "phase_lifted_path_homology_features.csv")
    phase = phase.pivot(
        index=IDENTITY, columns="representation", values="loop_score"
    ).reindex(pitch.index)
    published = pd.read_csv(SCORES).set_index(IDENTITY).sort_index()
    scorer = ExactPathHomologyScorer.from_json(PROFILE)

    result = scorer.score(
        pitch.loc[:, TOPOLOGY_METRICS].to_numpy(float),
        phase.loc[:, ["path_acoustic_phase"]].to_numpy(float),
        phase.loc[:, ["path_chroma_phase"]].to_numpy(float),
    )

    np.testing.assert_allclose(
        result.coordinates,
        published.loc[:, FEATURE_ORDER].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.focus_logit, published["focus_logit"], atol=1e-12)
    np.testing.assert_allclose(
        result.focus_probability, published["focus_probability"], atol=1e-12
    )
    np.testing.assert_allclose(
        result.focus_band_loss, published["focus_band_loss"], atol=1e-12
    )


def test_release_manifest_hashes_and_validation_gate() -> None:
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert release["release_status"] == "issued"
    assert release["profile_sha256"] == _sha256(PROFILE)
    assert release["signing_gates"] == {
        "frozen_dimensions": "passed",
        "feature_order": "passed",
        "distance_weights": "passed",
        "validation_180_reproduction": "passed",
        "legacy_51d_archived": "passed",
    }
    for relative, expected in release["artifact_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    reproduction = summary["validation_180_reproduction"]
    assert reproduction["status"] == "passed"
    assert max(reproduction["absolute_error"].values()) <= reproduction["tolerance"]
