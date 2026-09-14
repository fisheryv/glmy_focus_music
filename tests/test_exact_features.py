from __future__ import annotations

from pathlib import Path

import pytest

from generation.exact_features import _raise_stage_failures


def test_exact_feature_failure_reports_manifest_samples_and_distinct_errors() -> None:
    failures = [
        {"segment_id": "zero_1", "error": "audio is effectively silent"},
        {"segment_id": "zero_2", "error": "audio is effectively silent"},
        {"segment_id": "other", "error": "missing optional dependency"},
    ]

    with pytest.raises(RuntimeError) as captured:
        _raise_stage_failures(
            "preprocessing",
            failures,
            Path("batch/manifests/preprocessed_candidates.csv"),
        )

    message = str(captured.value)
    assert "preprocessing failed for 3 candidate(s)" in message
    assert "batch/manifests/preprocessed_candidates.csv" in message
    assert "zero_1, zero_2, other" in message
    assert "2x audio is effectively silent" in message
    assert "1x missing optional dependency" in message
