from __future__ import annotations

import json
from pathlib import Path

from generation.ltsn_contract import sha256_file
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.ltsn_v6_tac_correction import correct_v6_tac_direction_evidence
from generation.tac_target import TACTopologyTarget

ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "metadata" / "tac_topology_target_v1.json"


def _center_coordinates(target: TACTopologyTarget) -> list[float]:
    coordinates = [0.0] * 18
    for block in target.payload["blocks"].values():
        for index, center in zip(block["indices"], block["center"], strict=True):
            coordinates[int(index)] = float(center)
    return coordinates


def test_v6_correction_replaces_focus_band_with_exact_tac_derivative(tmp_path: Path) -> None:
    target = TACTopologyTarget.from_json(TARGET_PATH, verify_sources=False)
    center = _center_coordinates(target)
    displaced = list(center)
    pitch = target.payload["blocks"]["pitch"]
    displaced[int(pitch["indices"][0])] += float(pitch["scale"][0])
    fingerprint = target.payload["source_sha256"][
        "metadata/focus_path_homology_fingerprint_v2.json"
    ]

    coordinate_rows = []
    pair_rows = []
    legacy_rows = []
    specifications = (
        ("pair_seen", "seen_anchor_heldout_direction", displaced, center, 1.0),
        ("pair_unseen", "unseen_anchor", center, displaced, -1.0),
        ("pair_unseen_step6", "unseen_anchor", displaced, center, 1.0),
    )
    for index, (pair_id, split, minus_coordinates, plus_coordinates, predicted) in enumerate(
        specifications
    ):
        minus_id = f"minus_{index}"
        plus_id = f"plus_{index}"
        minus_hash = f"{index + 1:064x}"
        plus_hash = f"{index + 11:064x}"
        for sample_id, latent_hash, coordinates in (
            (minus_id, minus_hash, minus_coordinates),
            (plus_id, plus_hash, plus_coordinates),
        ):
            coordinate_rows.append(
                {
                    "sample_id": sample_id,
                    "latent_sha256": latent_hash,
                    "fingerprint_json_sha256": fingerprint,
                    "coordinates_json": json.dumps(coordinates),
                }
            )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "evaluation_split": split,
                "step_number": 4 + index,
                "rms_ratio": 0.005,
                "minus_sample_id": minus_id,
                "plus_sample_id": plus_id,
                "minus_latent_sha256": minus_hash,
                "plus_latent_sha256": plus_hash,
            }
        )
        legacy_rows.append(
            {
                "pair_id": pair_id,
                "evaluation_split": split,
                "step_number": 4 + index,
                "rms_ratio": 0.005,
                "exact_derivative": -predicted,
                "predicted_derivative": predicted,
                "direction_correct": 0,
            }
        )

    coordinates_path = tmp_path / "coordinates.csv"
    pairs_path = tmp_path / "pairs.csv"
    legacy_outcomes_path = tmp_path / "legacy_outcomes.csv"
    legacy_report_path = tmp_path / "legacy_report.json"
    output_report_path = tmp_path / "report_tac.json"
    output_outcomes_path = tmp_path / "outcomes_tac.csv"
    write_csv_atomic(coordinates_path, coordinate_rows)
    write_csv_atomic(pairs_path, pair_rows)
    write_csv_atomic(legacy_outcomes_path, legacy_rows)
    write_json_atomic(
        legacy_report_path,
        {
            "schema_version": 1,
            "experiment": "ltsn_v6_final_target_screen",
            "diagnostic_only": True,
            "scientific_evidence": False,
            "qualification_eligible": False,
            "guidance_promotion_eligible": False,
            "production_authorization": False,
            "new_audio_files": 0,
            "fingerprint_sha256": fingerprint,
            "tac_target_sha256": sha256_file(TARGET_PATH),
            "pair_manifest_sha256": sha256_file(pairs_path),
            "pair_outcomes_sha256": sha256_file(legacy_outcomes_path),
            "thresholds": {
                "minimum_train_final_distance_spearman": 0.7,
                "minimum_development_final_distance_spearman": 0.5,
                "minimum_heldout_direction_pairs": 2,
                "minimum_heldout_direction_agreement": 0.6,
                "minimum_heldout_derivative_spearman": 0.15,
            },
            "criteria": {
                "no_new_audio": True,
                "minimum_train_final_distance_spearman": True,
                "minimum_development_final_distance_spearman": True,
                "minimum_heldout_direction_pairs": True,
                "minimum_heldout_direction_agreement": False,
                "minimum_heldout_derivative_spearman": False,
            },
        },
    )

    payload = correct_v6_tac_direction_evidence(
        legacy_report_path=legacy_report_path,
        legacy_outcomes_path=legacy_outcomes_path,
        pair_manifest_path=pairs_path,
        coordinate_manifest_path=coordinates_path,
        tac_target_path=TARGET_PATH,
        output_report_path=output_report_path,
        output_outcomes_path=output_outcomes_path,
    )

    assert payload["schema_version"] == 2
    assert payload["legacy_direction_evidence_valid"] is False
    assert payload["heldout_direction_agreement"] == 1.0
    assert payload["heldout_derivative_spearman"] == 1.0
    assert payload["final_target_signal_supported"] is True
    outcomes = output_outcomes_path.read_text(encoding="utf-8")
    assert "exact_tac_derivative" in outcomes
    assert "legacy_focus_band_derivative" in outcomes
