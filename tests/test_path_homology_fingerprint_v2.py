from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
SCORES = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_scores.csv"
DIRECTIONS = ROOT / "metadata" / "focus_path_homology_fingerprint_v2_directions.csv"


def test_v2_profile_is_pure_path_homology() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))

    assert profile["fingerprint_id"] == "focus_path_homology_fingerprint_v2"
    assert profile["contains_tda_features"] is False
    assert profile["primary_layers"]["L"] == ["pitch", "rhythm", "modulation"]
    assert profile["primary_layers"]["P"] == [
        "path_acoustic_phase",
        "path_chroma_phase",
    ]
    assert profile["primary_layers"]["LP_dimensions"] == 51
    assert len(profile["classifier"]["coefficient"]) == 51
    assert "structure" in profile["auxiliary_layers"]
    assert "all Vietoris-Rips TDA endpoints" in profile["excluded"]


def test_v2_scores_and_directional_signature_are_complete() -> None:
    scores = pd.read_csv(SCORES)
    directions = pd.read_csv(DIRECTIONS)

    assert len(scores) == 1200
    assert scores["segment_id"].nunique() == 1200
    assert set(scores["split"]) == {"discovery", "validation", "holdout"}
    assert scores["focus_probability"].between(0.0, 1.0).all()
    assert (scores["focus_band_loss"] >= 0.0).all()
    assert len(directions) == 46
    assert directions.groupby("layer").size().to_dict() == {
        "local": 38,
        "macro_auxiliary": 6,
        "phase": 2,
    }
