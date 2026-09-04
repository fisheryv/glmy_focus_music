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


def _write_pairs(path: Path, fingerprint_sha256: str, *, diversity: bool) -> None:
    rows = [
        {
            "prompt_id": f"p{index}",
            "fingerprint_json_sha256": fingerprint_sha256,
            "exact_focus_band_loss_before": 1.0,
            "exact_focus_band_loss_after": 0.5,
            "proxy_focus_band_loss_before": 1.0,
            "proxy_focus_band_loss_after": 0.6,
            "quality_noninferior": "true",
            "prompt_noninferior": "true",
            "diversity_preserved": str(diversity).lower(),
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
