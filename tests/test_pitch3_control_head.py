from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generation.latent_topology_control_head import (  # noqa: E402
    PITCH3_OOD_STAT_FEATURES,
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
    Pitch3ControlOutput,
)
from generation.ltsn_contract import sha256_file  # noqa: E402
from generation.pitch3_contract import load_pitch3_contract  # noqa: E402
from generation.pitch3_training import (  # noqa: E402
    Pitch3BalancedBatchSampler,
    Pitch3LossWeights,
    Pitch3Snapshot,
    load_pitch3_training_config,
    pitch3_loss,
    read_pitch3_manifest,
    train_pitch3_control_head,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "metadata" / "focus_pitch3_fingerprint_v1.json"


def test_pitch3_control_head_shapes_gradients_and_parameter_budget() -> None:
    contract = load_pitch3_contract(PROFILE)
    config = Pitch3ControlHeadConfig(
        model_dim=32,
        condition_dim=32,
        transformer_heads=4,
        transformer_layers=1,
        feedforward_dim=64,
        temporal_stride=2,
        dropout=0.0,
    )
    model = LatentTopologyControlHead(contract, config)
    latent = torch.randn(2, 31, 64, requires_grad=True)
    mask = torch.ones(2, 31, dtype=torch.bool)
    mask[1, 25:] = False

    output = model(latent, torch.tensor([0.5, 0.25]), torch.tensor([4, 6]), mask)
    loss, parts = pitch3_loss(
        output,
        torch.randn(2, 3),
        torch.randn(2),
        torch.tensor([0.0, 1.0]),
        Pitch3LossWeights(
            band=0.5,
            band_rank=0.1,
            ood_margin=0.1,
            ood_margin_value=1.0,
        ),
        target_lower=torch.tensor(contract.target_lower),
        target_upper=torch.tensor(contract.target_upper),
        distance_weights=torch.tensor(contract.distance_weights),
    )
    loss.backward()

    assert output.coordinate_mean.shape == (2, 3)
    assert output.coordinate_logvar.shape == (2, 3)
    assert output.ood_logit.shape == (2,)
    assert output.focus_logit.shape == (2,)
    assert set(parts) == {
        "coordinate",
        "nll",
        "focus",
        "ood",
        "band",
        "band_rank",
        "band_region",
        "ood_margin",
    }
    assert torch.isfinite(loss)
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert model.trainable_parameters < 7_000_000


def _loss_output(coordinate_mean: torch.Tensor) -> Pitch3ControlOutput:
    batch = coordinate_mean.shape[0]
    return Pitch3ControlOutput(
        coordinate_mean=coordinate_mean,
        coordinate_logvar=torch.zeros_like(coordinate_mean),
        ood_logit=torch.zeros(batch),
        focus_logit=torch.zeros(batch),
    )


def test_pitch3_v24_region_loss_corrects_false_inside_prediction() -> None:
    contract = load_pitch3_contract(PROFILE)
    lower = torch.tensor(contract.target_lower)
    upper = torch.tensor(contract.target_upper)
    distance_weights = torch.tensor(contract.distance_weights)
    predicted = ((lower + upper) / 2.0).unsqueeze(0).requires_grad_(True)
    exact = predicted.detach().clone()
    exact[0, 2] = upper[2] + 1.0

    _, parts = pitch3_loss(
        _loss_output(predicted),
        exact,
        torch.zeros(1),
        torch.zeros(1),
        Pitch3LossWeights(band_region=1.0),
        target_lower=lower,
        target_upper=upper,
        distance_weights=distance_weights,
    )
    parts["band_region"].backward()

    assert parts["band_region"] > 0.0
    assert predicted.grad is not None
    assert predicted.grad[0, 2] < 0.0
    assert torch.equal(predicted.grad[0, :2], torch.zeros(2))


def test_pitch3_v24_smooth_band_restores_near_boundary_gradient() -> None:
    contract = load_pitch3_contract(PROFILE)
    lower = torch.tensor(contract.target_lower)
    upper = torch.tensor(contract.target_upper)
    distance_weights = torch.tensor(contract.distance_weights)
    exact = ((lower + upper) / 2.0).unsqueeze(0)
    exact[0, 2] = upper[2] + 0.5

    hard_predicted = ((lower + upper) / 2.0).unsqueeze(0)
    hard_predicted[0, 2] = upper[2] - 0.01
    hard_predicted.requires_grad_(True)
    _, hard_parts = pitch3_loss(
        _loss_output(hard_predicted),
        exact,
        torch.zeros(1),
        torch.zeros(1),
        Pitch3LossWeights(band=1.0),
        target_lower=lower,
        target_upper=upper,
        distance_weights=distance_weights,
    )
    hard_parts["band"].backward()

    smooth_predicted = hard_predicted.detach().clone().requires_grad_(True)
    _, smooth_parts = pitch3_loss(
        _loss_output(smooth_predicted),
        exact,
        torch.zeros(1),
        torch.zeros(1),
        Pitch3LossWeights(band=1.0, band_smooth_temperature_fraction=0.05),
        target_lower=lower,
        target_upper=upper,
        distance_weights=distance_weights,
    )
    smooth_parts["band"].backward()

    assert hard_predicted.grad is not None
    assert hard_predicted.grad[0, 2] == 0.0
    assert smooth_predicted.grad is not None
    assert smooth_predicted.grad[0, 2] < 0.0


def test_pitch3_v24_config_is_a_controlled_v23_objective_ablation() -> None:
    model, training, weights = load_pitch3_training_config(
        ROOT / "configs" / "pitch3_control_head_training_v24.toml"
    )

    assert model.ood_stats_dim == 32
    assert training.seed == 20260917
    assert training.id_band_stratified is True
    assert training.scale_high_training_target_scales == (2.5, 3.5, 4.0)
    assert weights.band_region == 0.25
    assert weights.band_smooth_temperature_fraction == 0.05


def test_pitch3_v23_raw_ood_statistics_preserve_scale_signal() -> None:
    latent = torch.randn(2, 17, 64)
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[1, 13:] = False
    scale = 2.5

    original = LatentTopologyControlHead.raw_ood_statistics(latent, mask)
    scaled = LatentTopologyControlHead.raw_ood_statistics(latent * scale, mask)
    delta = scaled - original
    expected_shift = torch.full((2,), float(np.log(scale)))

    for index in (0, 1, 3, 4, 5, 6):
        assert torch.allclose(delta[:, index], expected_shift, atol=1e-5, rtol=1e-5)
    for index in (2, 7):
        assert torch.allclose(delta[:, index], torch.zeros(2), atol=1e-5, rtol=1e-5)


def test_pitch3_v23_ood_branch_and_legacy_shape_compatibility() -> None:
    contract = load_pitch3_contract(PROFILE)
    shared = {
        "model_dim": 32,
        "condition_dim": 32,
        "transformer_heads": 4,
        "transformer_layers": 1,
        "feedforward_dim": 64,
        "temporal_stride": 2,
        "dropout": 0.0,
    }
    legacy = LatentTopologyControlHead(contract, Pitch3ControlHeadConfig(**shared))
    legacy_reloaded = LatentTopologyControlHead(contract, Pitch3ControlHeadConfig(**shared))
    legacy_reloaded.load_state_dict(legacy.state_dict(), strict=True)
    v23 = LatentTopologyControlHead(contract, Pitch3ControlHeadConfig(**shared, ood_stats_dim=32))
    latent = torch.randn(2, 19, 64)

    output = v23(latent, torch.tensor([0.5, 0.25]), torch.tensor([4, 6]))

    assert output.ood_logit.shape == (2,)
    assert legacy.ood_stats_projection is None
    assert legacy.ood_head.in_features == 32
    assert v23.ood_stats_projection is not None
    assert v23.ood_head.in_features == 64
    assert legacy.state_dict()["ood_head.weight"].shape == (1, 32)
    assert len(PITCH3_OOD_STAT_FEATURES) == 8


def _write_smoke_manifest(tmp_path: Path) -> Path:
    contract = load_pitch3_contract(PROFILE)
    rows = []
    for index, (split, ood) in enumerate(
        (("train", 0.0), ("train", 1.0), ("development", 0.0), ("development", 1.0))
    ):
        latent_path = tmp_path / f"latent_{index}.npy"
        np.save(latent_path, np.full((8 + index, 64), index / 10.0, np.float32))
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "prompt_id": f"prompt_{index}",
                "trajectory_id": f"trajectory_{index}",
                "split": split,
                "step_number": 4,
                "timestep": 0.5,
                "latent_path": latent_path.name,
                "latent_sha256": sha256_file(latent_path),
                "coordinates_json": json.dumps([0.1 * index, 0.2 * index, 0.3 * index]),
                "focus_logit": 0.25 * index,
                "ood_label": ood,
                "ood_kind": "" if ood == 0.0 else "ood_scale_high",
                "ood_transform_version": ("" if ood == 0.0 else "pitch3_latent_ood_v2"),
                "ood_label_source": ("" if ood == 0.0 else "deterministic_latent_transform_v2"),
                "fingerprint_json_sha256": contract.artifact_sha256,
                "feature_order_json": json.dumps(list(contract.feature_order)),
                "label_scope": "per_snapshot_exact",
                "is_final": "false",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_pitch3_training_writes_hash_bound_non_authorizing_checkpoint(tmp_path: Path) -> None:
    manifest = _write_smoke_manifest(tmp_path)
    config = tmp_path / "training.toml"
    config.write_text(
        """
[model]
latent_dim = 64
model_dim = 32
condition_dim = 32
transformer_heads = 4
transformer_layers = 1
feedforward_dim = 64
temporal_stride = 2
dropout = 0.0
logvar_min = -8.0
logvar_max = 4.0

[training]
learning_rate = 0.001
weight_decay = 0.0
micro_batch_size = 2
max_epochs = 2
minimum_epochs = 1
early_stopping_patience = 1
gradient_clip_norm = 1.0
num_workers = 0
use_bf16 = false
seed = 17
require_ood_both_classes = true
scale_high_training_target_scales = [2.5, 3.5, 4.0]

[loss]
coordinate = 1.0
nll = 0.1
focus = 0.1
ood = 0.1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = train_pitch3_control_head(
        fingerprint_path=PROFILE,
        training_manifest=manifest,
        config_path=config,
        output_dir=tmp_path / "output",
        device_name="cpu",
    )

    checkpoint = Path(result["checkpoint"])
    assert checkpoint.is_file()
    assert result["checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["guidance_promotion_eligible"] is False
    assert result["ood_class_counts"] == {
        "train": {"id": 1, "ood": 1},
        "development": {"id": 1, "ood": 1},
    }
    assert result["production_authorization"] is False
    assert result["band_training_objective"] == {
        "kind": "hard_excursion_v23_compatible",
        "region_consistency_weight": 0.0,
        "smooth_temperature_fraction_of_band_width": 0.0,
        "evaluation_band_formula_changed": False,
    }
    augmentation = result["training_virtual_ood_augmentation"]
    assert augmentation["assignment_count"] == 1
    assert augmentation["target_scales"] == [2.5, 3.5, 4.0]
    assert augmentation["coordinate_targets_used"] is False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert (
        payload["metadata"]["fingerprint_json_sha256"]
        == load_pitch3_contract(PROFILE).artifact_sha256
    )


def test_pitch3_balanced_batch_sampler_has_fixed_class_counts(tmp_path: Path) -> None:
    manifest = _write_smoke_manifest(tmp_path)
    records = [
        record
        for record in read_pitch3_manifest(manifest, load_pitch3_contract(PROFILE))
        if record.split == "train"
    ]
    sampler = Pitch3BalancedBatchSampler(
        records,
        id_per_batch=1,
        ood_per_batch=1,
        seed=17,
    )
    sampler.set_epoch(3)
    batches = list(sampler)
    assert batches
    assert all(
        sorted(records[index].ood_label for index in batch) == [0.0, 1.0] for batch in batches
    )


def test_pitch3_v23_sampler_stratifies_each_id_batch() -> None:
    contract = load_pitch3_contract(PROFILE)
    records = []
    for index in range(9):
        records.append(
            Pitch3Snapshot(
                sample_id=f"id_{index}",
                prompt_id=f"prompt_id_{index}",
                trajectory_id=f"trajectory_id_{index}",
                split="train",
                step_number=4,
                timestep=0.5,
                latent_path=Path(f"id_{index}.npy"),
                latent_sha256="0" * 64,
                coordinates=(float(index), float(index) / 2.0, float(index) / 3.0),
                focus_logit=0.0,
                ood_label=0.0,
                is_final=False,
                ood_kind="",
                ood_transform_version="",
                ood_label_source="",
            )
        )
    for index, kind in enumerate(("ood_scale_high", "ood_block_shuffle")):
        records.append(
            Pitch3Snapshot(
                sample_id=f"ood_{index}",
                prompt_id=f"prompt_ood_{index}",
                trajectory_id=f"trajectory_ood_{index}",
                split="train",
                step_number=4,
                timestep=0.5,
                latent_path=Path(f"ood_{index}.npy"),
                latent_sha256="1" * 64,
                coordinates=(0.0, 0.0, 0.0),
                focus_logit=0.0,
                ood_label=1.0,
                is_final=False,
                ood_kind=kind,
                ood_transform_version="pitch3_latent_ood_v2",
                ood_label_source="deterministic_latent_transform_v2",
            )
        )
    sampler = Pitch3BalancedBatchSampler(
        records,
        id_per_batch=6,
        ood_per_batch=2,
        seed=20260917,
        contract=contract,
        id_band_stratified=True,
    )
    strata = {name: set(indices) for name, indices in sampler.id_by_stratum.items()}

    assert set(strata) == {"low", "middle", "high"}
    assert all(summary["samples"] == 3 for summary in sampler.id_strata_summary.values())
    for batch in sampler:
        assert sum(records[index].ood_label < 0.5 for index in batch) == 6
        assert sum(records[index].ood_label >= 0.5 for index in batch) == 2
        for indices in strata.values():
            assert sum(index in indices for index in batch) == 2
