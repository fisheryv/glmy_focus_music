from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from generation.ltsn_contract import load_fingerprint_contract, sha256_file
from generation.ltsn_pipeline import write_csv_atomic
from generation.ltsn_v6_final_target import (
    V6FinalTargetModel,
    V6HeadConfig,
    _flat_direction_metrics,
    prepare_v6_final_target_view,
    read_v6_view,
)
from generation.path_homology_surrogate import LTSNConfig

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
TAC_TARGET = ROOT / "metadata" / "tac_topology_target_v1.json"


def _latent(path: Path, value: float) -> str:
    array = np.full((80, 64), value, dtype=np.float32)
    np.save(path, array, allow_pickle=False)
    return sha256_file(path)


def _config(path: Path) -> None:
    path.write_text(
        """
[screen]
minimum_train_final_distance_spearman = 0.0
minimum_development_final_distance_spearman = 0.0
minimum_heldout_direction_pairs = 2
minimum_heldout_direction_agreement = 0.0
minimum_heldout_derivative_spearman = 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _source_manifest(tmp_path: Path) -> Path:
    contract = load_fingerprint_contract(FINGERPRINT)
    exact = tmp_path / "exact_labels.csv"
    exact.write_text("sample_id\nplaceholder\n", encoding="utf-8")
    exact_hash = sha256_file(exact)
    rows = []
    for trajectory_index, split in enumerate(("train", "train", "train", "development"), 1):
        trajectory_id = f"trajectory_{trajectory_index}"
        prompt_id = f"prompt_{trajectory_index}"
        coordinates = [0.0] * 18
        for index in range(3, 16):
            coordinates[index] = trajectory_index * (index - 1) * 0.01
        coordinates[16] = trajectory_index * 0.13
        coordinates[17] = trajectory_index * -0.17
        for step in (4, 5, 6, 8):
            sample_id = f"{trajectory_id}_step{step:02d}"
            latent = tmp_path / f"{sample_id}.npy"
            latent_hash = _latent(latent, trajectory_index + step / 100.0)
            rows.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "trajectory_id": trajectory_id,
                    "split": split,
                    "step_number": step,
                    "timestep": 0.0 if step == 8 else (1.0 - step / 24.0),
                    "latent_path": latent.name,
                    "latent_sha256": latent_hash,
                    "coordinates_json": json.dumps(coordinates),
                    "focus_logit": 0.0,
                    "ood_label": 0.0,
                    "fingerprint_json_sha256": contract.artifact_sha256,
                    "feature_order_json": json.dumps(list(contract.feature_order)),
                    "label_scope": "per_snapshot_exact",
                    "exact_label_table_sha256": exact_hash,
                    "exact_label_table_path": exact.name,
                    "is_final": step == 8,
                }
            )
    path = tmp_path / "ltsn_manifest.csv"
    write_csv_atomic(path, rows)
    return path


def _pair_manifest(tmp_path: Path) -> Path:
    splits = (
        "train",
        "seen_anchor_heldout_direction",
        "unseen_anchor",
        "train_rms_sensitivity",
        "seen_anchor_heldout_direction_rms_sensitivity",
        "unseen_anchor_rms_sensitivity",
    )
    rows = []
    for index, split in enumerate(splits):
        minus = tmp_path / f"pair_{index}_minus.npy"
        plus = tmp_path / f"pair_{index}_plus.npy"
        minus_hash = _latent(minus, index + 0.1)
        plus_hash = _latent(plus, index + 0.2)
        rows.append(
            {
                "pair_id": f"pair_{index}",
                "evaluation_split": split,
                "step_number": 4 + index % 3,
                "timestep": 0.75,
                "rms_ratio": 0.005,
                "minus_latent_path": minus.name,
                "plus_latent_path": plus.name,
                "minus_latent_sha256": minus_hash,
                "plus_latent_sha256": plus_hash,
                "true_target": int(index % 2 == 0),
                "matched_control_target": int(index % 2 == 1),
                "exact_derivative": -1.0 if index % 2 else 1.0,
            }
        )
    path = tmp_path / "v52b_pair_manifest.csv"
    write_csv_atomic(path, rows)
    return path


def test_prepare_v6_reuses_latents_and_final_targets_without_audio(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path)
    pairs = _pair_manifest(tmp_path)
    config = tmp_path / "v6.toml"
    output = tmp_path / "v6"
    _config(config)

    payload = prepare_v6_final_target_view(
        root=ROOT,
        fingerprint_path=FINGERPRINT,
        tac_target_path=TAC_TARGET,
        source_manifest_path=source,
        pair_manifest_path=pairs,
        config_path=config,
        output_dir=output,
    )

    assert payload["new_audio_files"] == 0
    assert payload["copied_latent_files"] == 0
    assert payload["reused_latents"] == 12
    assert payload["trajectories"] == 4
    assert not list(output.rglob("*.wav"))
    view = read_v6_view(output / "v6_final_target_view.csv")
    assert {row.step_number for row in view} == {4, 5, 6}
    by_trajectory: dict[str, set[tuple[float, ...]]] = {}
    for row in view:
        by_trajectory.setdefault(row.trajectory_id, set()).add(row.targets)
        assert row.latent_path.parent == tmp_path
    assert all(len(targets) == 1 for targets in by_trajectory.values())


def test_v6_model_uses_three_scalar_head_and_freezes_legacy_heads() -> None:
    contract = load_fingerprint_contract(FINGERPRINT)
    config = LTSNConfig(
        condition_dim=8,
        stem_channels=8,
        local_channels=8,
        global_channels=8,
        transformer_heads=2,
        transformer_layers=1,
        dropout=0.0,
        stem_dropout=0.0,
    )
    model = V6FinalTargetModel(contract, config, V6HeadConfig(hidden_dim=32))
    output = model(
        torch.zeros(2, 80, 64),
        torch.tensor([0.8, 0.7]),
        torch.tensor([4, 5]),
        torch.ones(2, 80, dtype=torch.bool),
    )
    assert output.shape == (2, 3)
    assert not any(
        parameter.requires_grad for parameter in model.encoder.coordinate_mean_head.parameters()
    )
    assert not next(model.encoder.coordinate_logvar_head.parameters()).requires_grad
    assert not next(model.encoder.ood_head.parameters()).requires_grad
    assert next(model.final_target_head.parameters()).requires_grad


def test_v6_direction_metrics_use_exact_derivative_sign_and_rank() -> None:
    rows = [
        {"direction_correct": 1, "exact_derivative": -2.0, "predicted_derivative": -3.0},
        {"direction_correct": 1, "exact_derivative": 1.0, "predicted_derivative": 2.0},
        {"direction_correct": 0, "exact_derivative": 3.0, "predicted_derivative": -1.0},
    ]
    metrics = _flat_direction_metrics(rows)
    assert metrics["pairs"] == 3
    assert metrics["direction_agreement"] == 2 / 3
    assert metrics["derivative_spearman"] == 0.5
