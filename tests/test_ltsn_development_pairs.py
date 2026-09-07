from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from generation.ltsn_contract import LTSNContractError
from generation.ltsn_development_pairs import (
    AUTHORIZATION_SCOPE,
    _validate_pair_decode_identity,
    build_development_plan,
    finalize_development_pairs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_development_plan_uses_global_prompt_indices_and_excludes_other_splits(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "prompts.csv"
    rows = [
        {"prompt_id": "train", "caption": "train", "split": "train", "seed": ""},
        {"prompt_id": "dev_a", "caption": "a", "split": "development", "seed": ""},
        {
            "prompt_id": "qualification",
            "caption": "qualification",
            "split": "qualification",
            "seed": "",
        },
        {"prompt_id": "dev_b", "caption": "b", "split": "development", "seed": ""},
    ]
    _write_csv(manifest, rows)

    plan = build_development_plan(
        manifest,
        seed_start=100,
        seeds_per_prompt=2,
        expected_development_prompts=2,
    )

    assert [row["prompt_id"] for row in plan] == ["dev_a", "dev_a", "dev_b", "dev_b"]
    assert [row["seed"] for row in plan] == [102, 103, 106, 107]
    assert len({row["pair_id"] for row in plan}) == 4
    assert "qualification" not in {row["prompt_id"] for row in plan}


def test_development_plan_rejects_count_or_manual_seed_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.csv"
    _write_csv(
        manifest,
        [{"prompt_id": "dev", "caption": "a", "split": "development", "seed": "7"}],
    )
    with pytest.raises(LTSNContractError, match="globally indexed"):
        build_development_plan(
            manifest,
            seed_start=100,
            seeds_per_prompt=2,
            expected_development_prompts=1,
        )

    _write_csv(
        manifest,
        [{"prompt_id": "dev", "caption": "a", "split": "development", "seed": ""}],
    )
    with pytest.raises(LTSNContractError, match="prompt count changed"):
        build_development_plan(
            manifest,
            seed_start=100,
            seeds_per_prompt=2,
            expected_development_prompts=64,
        )


def _raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index in range(64):
        for seed_index in range(4):
            seed = 1000 + prompt_index * 4 + seed_index
            pair_id = f"p{prompt_index}__seed{seed}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "prompt_id": f"p{prompt_index}",
                    "seed": seed,
                    "fingerprint_json_sha256": "a" * 64,
                    "generation_manifest_sha256": "d" * 64,
                    "generation_plan_sha256": "e" * 64,
                    "exact_descriptor_table_sha256": "f" * 64,
                    "baseline_candidate_id": f"{pair_id}__baseline",
                    "guided_candidate_id": f"{pair_id}__guided",
                    "baseline_audio_sha256": "b" * 64,
                    "guided_audio_sha256": "c" * 64,
                    "exact_focus_band_loss_before": 1.0,
                    "exact_focus_band_loss_after": 0.5,
                    "proxy_focus_band_loss_before": 1.0,
                    "proxy_focus_band_loss_after": 0.6,
                    "baseline_technical_quality_eligible": "true",
                    "guided_technical_quality_eligible": "true",
                    "authorization_scope": AUTHORIZATION_SCOPE,
                }
            )
    return rows


def _evidence_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "pair_id": row["pair_id"],
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "quality_baseline": 0.5,
            "quality_guided": 0.6,
            "prompt_baseline": 0.5,
            "prompt_guided": 0.6,
            "diversity_baseline": 0.5,
            "diversity_guided": 0.6,
        }
        for row in raw_rows
    ]


def _protocol(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen_before_generation",
                "criteria": {
                    name: {
                        "metric": name,
                        "direction": "higher_is_better",
                        "margin": 0.0,
                    }
                    for name in ("quality", "prompt", "diversity")
                },
            }
        ),
        encoding="utf-8",
    )


def _v2_protocol(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "frozen_before_generation",
                "gate_contract": "latent_guidance_promotion_v2",
                "criteria": {
                    "quality": {
                        "metric": "blind_quality_score",
                        "direction": "higher_is_better",
                        "margin": 0.0,
                        "evidence_required": False,
                        "gate_modes": [],
                    },
                    "prompt": {
                        "metric": "prompt",
                        "direction": "higher_is_better",
                        "margin": 0.0,
                        "evidence_required": True,
                        "gate_modes": ["development", "confirmation"],
                    },
                    "diversity": {
                        "metric": "diversity",
                        "direction": "higher_is_better",
                        "margin": 0.0,
                        "evidence_required": True,
                        "gate_modes": ["confirmation"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_finalize_binds_numeric_evidence_and_remains_non_promotable(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    evidence = tmp_path / "evidence.csv"
    protocol = tmp_path / "protocol.json"
    raw_rows = _raw_rows()
    _write_csv(raw, raw_rows)
    _write_csv(evidence, _evidence_rows(raw_rows))
    _protocol(protocol)

    report = finalize_development_pairs(
        raw_pair_table=raw,
        evidence_table=evidence,
        protocol_path=protocol,
        output_dir=tmp_path,
        bootstrap_resamples=100,
    )

    assert report["authorization_scope"] == AUTHORIZATION_SCOPE
    assert report["guidance_promotion_eligible"] is False
    assert report["prompts"] == 64
    assert all(item["passed"] for item in report["criteria"].values())
    final_rows = list(csv.DictReader((tmp_path / "development_pairs.csv").open()))
    assert {row["authorization_scope"] for row in final_rows} == {AUTHORIZATION_SCOPE}
    assert {row["quality_noninferior"] for row in final_rows} == {"true"}
    assert not (tmp_path / "development_pairs.csv.part").exists()


def test_finalize_rejects_incomplete_or_non_numeric_evidence(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    evidence = tmp_path / "evidence.csv"
    protocol = tmp_path / "protocol.json"
    raw_rows = _raw_rows()
    _write_csv(raw, raw_rows)
    incomplete = _evidence_rows(raw_rows)[:-1]
    _write_csv(evidence, incomplete)
    _protocol(protocol)
    with pytest.raises(LTSNContractError, match="exactly"):
        finalize_development_pairs(
            raw_pair_table=raw,
            evidence_table=evidence,
            protocol_path=protocol,
            output_dir=tmp_path,
            bootstrap_resamples=10,
        )

    malformed = _evidence_rows(raw_rows)
    malformed[0]["quality_guided"] = ""
    _write_csv(evidence, malformed)
    with pytest.raises(LTSNContractError, match="numeric quality"):
        finalize_development_pairs(
            raw_pair_table=raw,
            evidence_table=evidence,
            protocol_path=protocol,
            output_dir=tmp_path,
            bootstrap_resamples=10,
        )


def test_v2_finalize_accepts_missing_blind_quality_as_diagnostic(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.csv"
    evidence = tmp_path / "evidence.csv"
    protocol = tmp_path / "protocol.json"
    raw_rows = _raw_rows()
    evidence_rows = _evidence_rows(raw_rows)
    for row in evidence_rows:
        row["quality_baseline"] = ""
        row["quality_guided"] = ""
    _write_csv(raw, raw_rows)
    _write_csv(evidence, evidence_rows)
    _v2_protocol(protocol)

    report = finalize_development_pairs(
        raw_pair_table=raw,
        evidence_table=evidence,
        protocol_path=protocol,
        output_dir=tmp_path,
        bootstrap_resamples=100,
    )

    assert report["schema_version"] == 2
    assert report["blind_quality_is_gate"] is False
    assert report["blind_quality_evidence_available"] is False
    assert report["quality_noninferior"] is None
    assert report["prompt_noninferior"] is True
    rows = list(csv.DictReader((tmp_path / "development_pairs.csv").open()))
    assert {row["quality_noninferior"] for row in rows} == {"not_evaluated"}


def test_identical_latents_require_identical_shared_decode() -> None:
    baseline = {"latent_sha256": "a" * 64, "audio_sha256": "b" * 64}
    assert _validate_pair_decode_identity(baseline, dict(baseline)) is False
    with pytest.raises(LTSNContractError, match="identical.*latents"):
        _validate_pair_decode_identity(
            baseline,
            {"latent_sha256": "a" * 64, "audio_sha256": "c" * 64},
        )
    assert _validate_pair_decode_identity(
        baseline,
        {"latent_sha256": "d" * 64, "audio_sha256": "c" * 64},
    ) is True
