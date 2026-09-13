from __future__ import annotations

import json
from pathlib import Path

from generation.ltsn_contract import sha256_file
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.ltsn_v61_data import prepare_v61_pair_view
from generation.tac_target import TACTopologyTarget

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
TARGET = ROOT / "metadata" / "tac_topology_target_v1.json"


def _center(target: TACTopologyTarget) -> list[float]:
    coordinates = [0.0] * 18
    for block in target.payload["blocks"].values():
        for index, value in zip(block["indices"], block["center"], strict=True):
            coordinates[int(index)] = float(value)
    return coordinates


def test_prepare_v61_uses_disjoint_fit_validation_and_heldout_pairs(tmp_path: Path) -> None:
    target = TACTopologyTarget.from_json(TARGET, verify_sources=False)
    base = _center(target)
    pitch = target.payload["blocks"]["pitch"]
    pitch_index = int(pitch["indices"][0])
    pitch_scale = float(pitch["scale"][0])
    fingerprint_hash = sha256_file(FINGERPRINT)
    coordinate_rows = []
    pair_rows = []

    def add_pair(pair_id: str, split: str, anchor: str, step: int, amount: float) -> None:
        minus_id = f"{pair_id}_minus"
        plus_id = f"{pair_id}_plus"
        minus_hash = f"{len(coordinate_rows) + 1:064x}"
        plus_hash = f"{len(coordinate_rows) + 2:064x}"
        minus_coordinates = list(base)
        plus_coordinates = list(base)
        minus_coordinates[pitch_index] += amount * pitch_scale
        for sample_id, latent_hash, coordinates in (
            (minus_id, minus_hash, minus_coordinates),
            (plus_id, plus_hash, plus_coordinates),
        ):
            coordinate_rows.append(
                {
                    "sample_id": sample_id,
                    "latent_sha256": latent_hash,
                    "fingerprint_json_sha256": fingerprint_hash,
                    "coordinates_json": json.dumps(coordinates),
                }
            )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "anchor_sample_id": anchor,
                "prompt_id": anchor,
                "evaluation_split": split,
                "step_number": step,
                "timestep": 0.75,
                "rms_ratio": 0.005,
                "minus_sample_id": minus_id,
                "plus_sample_id": plus_id,
                "minus_latent_path": f"latents/{minus_id}.npy",
                "plus_latent_path": f"latents/{plus_id}.npy",
                "minus_latent_sha256": minus_hash,
                "plus_latent_sha256": plus_hash,
            }
        )

    for step in (4, 5, 6):
        for anchor_index in range(3):
            anchor = f"train_s{step}_a{anchor_index}"
            for direction in range(2):
                add_pair(
                    f"{anchor}_d{direction}",
                    "train",
                    anchor,
                    step,
                    1.0 + anchor_index + direction * 0.25,
                )
        add_pair(f"seen_s{step}", "seen_anchor_heldout_direction", f"seen_{step}", step, 1.5)
        add_pair(f"unseen_s{step}", "unseen_anchor", f"unseen_{step}", step, 2.0)

    pair_manifest = tmp_path / "pairs.csv"
    coordinate_manifest = tmp_path / "coordinates.csv"
    v6_view = tmp_path / "v6_view.csv"
    v6_checkpoint = tmp_path / "v6.pt"
    v6_preparation = tmp_path / "v6_preparation.json"
    config = tmp_path / "v61.toml"
    output_dir = tmp_path / "v61"
    write_csv_atomic(pair_manifest, pair_rows)
    write_csv_atomic(coordinate_manifest, coordinate_rows)
    v6_view.write_text("placeholder\n", encoding="utf-8")
    v6_checkpoint.write_bytes(b"checkpoint")
    config.write_text("[training]\n", encoding="utf-8")
    write_json_atomic(
        v6_preparation,
        {
            "experiment": "ltsn_v6_final_target_screen",
            "diagnostic_only": True,
            "new_audio_files": 0,
            "fingerprint_sha256": fingerprint_hash,
            "tac_target_sha256": sha256_file(TARGET),
            "view_sha256": sha256_file(v6_view),
        },
    )

    payload = prepare_v61_pair_view(
        fingerprint_path=FINGERPRINT,
        tac_target_path=TARGET,
        v6_preparation_path=v6_preparation,
        v6_view_path=v6_view,
        v6_checkpoint_path=v6_checkpoint,
        pair_manifest_path=pair_manifest,
        coordinate_manifest_path=coordinate_manifest,
        config_path=config,
        output_dir=output_dir,
        validation_anchors_per_step=2,
        split_seed=123,
    )

    assert payload["new_audio_files"] == 0
    assert payload["copied_latent_files"] == 0
    assert payload["pair_counts"] == {
        "direction_fit": 6,
        "direction_validation": 12,
        "heldout_seen_anchor": 3,
        "heldout_unseen_anchor": 3,
    }
    assert len(payload["validation_anchors"]) == 6
    assert set(payload["direction_transforms_by_step"]) == {"4", "5", "6"}
