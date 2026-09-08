from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from generation.ltsn_contract import LTSNContractError, load_fingerprint_contract

pytest.importorskip("torch")

from generation.ltsn_evaluation import (
    GUIDANCE_PROMOTION_GATE_NAME,
    evaluate_guidance_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"


def _write_pairs(
    path: Path,
    fingerprint_sha256: str,
    *,
    diversity: bool,
    quality: str = "true",
    scope: str = "qualified_confirmation",
) -> None:
    rows = [
        {
            "prompt_id": f"p{index}",
            "fingerprint_json_sha256": fingerprint_sha256,
            "exact_focus_band_loss_before": 1.0,
            "exact_focus_band_loss_after": 0.5,
            "proxy_focus_band_loss_before": 1.0,
            "proxy_focus_band_loss_after": 0.6,
            "quality_noninferior": quality,
            "prompt_noninferior": "true",
            "diversity_preserved": str(diversity).lower(),
            "guided_technical_quality_eligible": "true",
            "authorization_scope": scope,
        }
        for index in range(2)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_confirmation_is_the_guidance_promotion_gate(tmp_path: Path) -> None:
    fingerprint_sha256 = load_fingerprint_contract(FINGERPRINT).artifact_sha256
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "qualification_passed": True,
                "fingerprint_json_sha256": fingerprint_sha256,
            }
        ),
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs, fingerprint_sha256, diversity=True)

    report = evaluate_guidance_pairs(
        pair_table=pairs,
        output_path=tmp_path / "confirmation.json",
        fingerprint_sha256=fingerprint_sha256,
        mode="confirmation",
        qualification_report=qualification,
        bootstrap_resamples=100,
    )

    assert report["gate"] == GUIDANCE_PROMOTION_GATE_NAME
    assert report["status"] == "passed"
    assert report["guidance_promotion_eligible"] is True
    assert len(report["qualification_report_sha256"]) == 64
    assert report["blind_quality_is_gate"] is False
    assert report["optimized_exact_improved_pairs"] == 2
    assert report["optimized_exact_tied_pairs"] == 0
    assert report["optimized_exact_worsened_pairs"] == 0

    _write_pairs(pairs, fingerprint_sha256, diversity=False)
    report = evaluate_guidance_pairs(
        pair_table=pairs,
        output_path=tmp_path / "blocked_confirmation.json",
        fingerprint_sha256=fingerprint_sha256,
        mode="confirmation",
        qualification_report=qualification,
        bootstrap_resamples=100,
    )
    assert report["status"] == "failed"
    assert report["guidance_promotion_eligible"] is False


@pytest.mark.parametrize("quality", ["false", "not_evaluated"])
def test_blind_quality_is_diagnostic_not_a_gate(tmp_path: Path, quality: str) -> None:
    fingerprint_sha256 = load_fingerprint_contract(FINGERPRINT).artifact_sha256
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "qualification_passed": True,
                "fingerprint_json_sha256": fingerprint_sha256,
            }
        ),
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs, fingerprint_sha256, diversity=True, quality=quality)

    report = evaluate_guidance_pairs(
        pair_table=pairs,
        output_path=tmp_path / "confirmation.json",
        fingerprint_sha256=fingerprint_sha256,
        mode="confirmation",
        qualification_report=qualification,
        bootstrap_resamples=100,
    )

    assert report["status"] == "passed"
    assert report["guidance_promotion_eligible"] is True
    assert report["quality_noninferior"] is (False if quality == "false" else None)
    assert report["blind_quality_evidence_available"] is (quality != "not_evaluated")


def test_confirmation_rejects_missing_qualification(tmp_path: Path) -> None:
    fingerprint_sha256 = load_fingerprint_contract(FINGERPRINT).artifact_sha256
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs, fingerprint_sha256, diversity=True)

    with pytest.raises(LTSNContractError, match="qualification"):
        evaluate_guidance_pairs(
            pair_table=pairs,
            output_path=tmp_path / "confirmation.json",
            fingerprint_sha256=fingerprint_sha256,
            mode="confirmation",
            bootstrap_resamples=100,
        )


def test_development_pairs_are_scope_limited_and_never_promotable(tmp_path: Path) -> None:
    fingerprint_sha256 = load_fingerprint_contract(FINGERPRINT).artifact_sha256
    pairs = tmp_path / "pairs.csv"
    _write_pairs(
        pairs,
        fingerprint_sha256,
        diversity=True,
        scope="development_only",
    )

    report = evaluate_guidance_pairs(
        pair_table=pairs,
        output_path=tmp_path / "development.json",
        fingerprint_sha256=fingerprint_sha256,
        mode="development",
        bootstrap_resamples=100,
    )

    assert report["authorization_scope"] == "development_only"
    assert report["guidance_promotion_eligible"] is False


def test_guidance_gate_requires_minimum_optimization_coverage(tmp_path: Path) -> None:
    fingerprint_sha256 = load_fingerprint_contract(FINGERPRINT).artifact_sha256
    pairs = tmp_path / "pairs.csv"
    _write_pairs(
        pairs,
        fingerprint_sha256,
        diversity=True,
        scope="development_only",
    )

    report = evaluate_guidance_pairs(
        pair_table=pairs,
        output_path=tmp_path / "development.json",
        fingerprint_sha256=fingerprint_sha256,
        mode="development",
        bootstrap_resamples=100,
        minimum_optimized_pairs=3,
        minimum_optimized_prompts=3,
    )

    assert report["proxy_optimized_pairs"] == 2
    assert report["optimization_coverage_gate_passed"] is False
    assert report["proxy_exact_direction_gate_passed"] is False


@pytest.mark.parametrize(
    ("mode", "scope"),
    [
        ("development", "qualified_confirmation"),
        ("confirmation", "development_only"),
    ],
)
def test_guidance_evaluator_rejects_cross_scope_pairs(
    tmp_path: Path, mode: str, scope: str
) -> None:
    fingerprint_sha256 = load_fingerprint_contract(FINGERPRINT).artifact_sha256
    pairs = tmp_path / "pairs.csv"
    _write_pairs(pairs, fingerprint_sha256, diversity=True, scope=scope)
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "qualification_passed": True,
                "fingerprint_json_sha256": fingerprint_sha256,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LTSNContractError, match="authorization_scope"):
        evaluate_guidance_pairs(
            pair_table=pairs,
            output_path=tmp_path / "report.json",
            fingerprint_sha256=fingerprint_sha256,
            mode=mode,
            qualification_report=qualification if mode == "confirmation" else None,
            bootstrap_resamples=100,
        )
