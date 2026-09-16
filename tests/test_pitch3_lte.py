from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import write_csv_atomic
from generation.pitch3_lte_data import (
    _anchor_index,
    _direction,
    merge_pitch3_lte_dataset_shards,
    pitch3_lte_prompt_shard,
    validate_pitch3_lte_dataset_preflight,
)


def test_lte_seed_difference_direction_has_unit_rms() -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 8) - 7.0
    direction = _direction(values)
    assert direction.dtype == np.float32
    assert np.sqrt(np.mean(np.square(direction), dtype=np.float64)) == pytest.approx(1.0)
    assert _anchor_index("p01__v01") == _anchor_index("p01__v01")
    assert 0 <= _anchor_index("p01__v01") < 4


def test_lte_multi_gpu_merge_keeps_prompt_groups_disjoint(tmp_path) -> None:
    shard_count = 2
    shard_dirs = [tmp_path / "shards" / f"shard_{index:02d}" for index in range(shard_count)]
    common = {
        "source_manifest_sha256": "1" * 64,
        "prompt_embedding_manifest_sha256": "2" * 64,
        "ace_config_sha256": "3" * 64,
        "fingerprint_json_sha256": "4" * 64,
        "ace_model_sha256": "5" * 64,
        "vae_sha256": "6" * 64,
        "include_splits": ["development", "train"],
        "local_splits": ["development", "train"],
        "radius_ratio": 0.05,
        "anchor_rule": "sha256(v3-lte-anchor|prompt_id)-mod-4",
        "direction_rule": ["normalize_rms(z1-z0)", "normalize_rms(z3-z2)"],
        "directions_per_prompt": 2,
    }
    prompts = [
        (f"p{index:03d}__v01", "train" if index < 320 else "development") for index in range(384)
    ]
    for shard_index, shard_dir in enumerate(shard_dirs):
        shard_dir.mkdir(parents=True)
        np.save(shard_dir / "latent.npy", np.zeros((2, 64), dtype=np.float32))
        np.savez(
            shard_dir / "prompt.npz",
            hidden=np.zeros((2, 1024), dtype=np.float32),
            mask=np.ones(2, dtype=np.bool_),
        )
        assigned = [
            (prompt_id, split)
            for prompt_id, split in prompts
            if pitch3_lte_prompt_shard(prompt_id, shard_count) == shard_index
        ]
        rows = []
        for prompt_id, split in assigned:
            for seed in range(4):
                rows.append(
                    {
                        "sample_id": f"{prompt_id}__base{seed}",
                        "prompt_id": prompt_id,
                        "prompt_family": prompt_id.rsplit("__v", 1)[0],
                        "trajectory_id": f"{prompt_id}__seed{seed}",
                        "split": split,
                        "source_kind": "base_step4_seed",
                        "latent_path": "latent.npy",
                        "latent_sha256": "7" * 64,
                        "prompt_embedding_path": "prompt.npz",
                        "prompt_embedding_sha256": "8" * 64,
                        "exact_band": seed,
                        "energy_target": seed,
                        "coordinates_json": "[0,0,0]",
                        "direction_id": "",
                        "direction_sign": 0,
                        "epsilon": 0,
                        "radius_ratio": 0,
                        "step_number": 4,
                        "timestep": 0.8333333,
                        "fingerprint_json_sha256": "4" * 64,
                        "ace_model_sha256": "5" * 64,
                        "vae_sha256": "6" * 64,
                        "dataset_plan_sha256": "9" * 64,
                    }
                )
            for direction in range(2):
                for sign, target in ((-1, 0.1), (1, 0.2)):
                    rows.append(
                        {
                            **rows[-1],
                            "sample_id": f"{prompt_id}__d{direction}__{sign}",
                            "source_kind": "local_finite_difference",
                            "direction_id": f"{prompt_id}__d{direction}",
                            "direction_sign": sign,
                            "epsilon": 0.05,
                            "radius_ratio": 0.05,
                            "energy_target": target + direction,
                        }
                    )
        manifest_path = shard_dir / "pitch3_lte_examples.csv"
        write_csv_atomic(manifest_path, rows)
        plan = {
            "schema_version": 1,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "prompt_ids": [prompt_id for prompt_id, _ in assigned],
            "planned_local_samples": len(assigned) * 4,
            **common,
        }
        (shard_dir / "pitch3_lte_dataset_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        summary = {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "local_preflight_passed": True,
        }
        (shard_dir / "pitch3_lte_dataset_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    legacy_plan = {
        "schema_version": 1,
        "stage": "pitch3_lte_exact_local_dataset",
        "model_family": "pitch3_prompt_conditioned_local_energy_v3",
        **common,
        "source_manifest_sha256": "f" * 64,
        "sharded": True,
        "shard_count": shard_count,
        "prompt_assignment": "sha256(v3-lte-data-shard|prompt_id)-mod-shard_count",
        "shard_plan_sha256": ["a" * 64, "b" * 64],
        "planned_local_samples": 1536,
        "items_sha256": "c" * 64,
        "prompt_ids": sorted(prompt_id for prompt_id, _ in prompts),
        "wav_policy": "ephemeral_delete_after_each_exact_batch",
    }
    legacy_path = tmp_path / "pitch3_lte_dataset_plan.json"
    legacy_path.write_text(json.dumps(legacy_plan), encoding="utf-8")
    legacy_sha256 = sha256_file(legacy_path)
    result = merge_pitch3_lte_dataset_shards(
        output_dir=tmp_path,
        shard_dirs=shard_dirs,
        devices=("cuda:0", "cuda:1"),
    )
    assert result["multi_gpu"] is True
    assert result["base_samples"] == 1536
    assert result["local_samples"] == 1536
    assert result["replaced_legacy_plan_sha256"] == legacy_sha256
    assert (tmp_path / f"pitch3_lte_dataset_plan_superseded_{legacy_sha256[:12]}.json").is_file()
    published = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert published["publication_kind"] == "canonical_merged_dataset"
    assert "shard_count" not in published
    rerun = merge_pitch3_lte_dataset_shards(
        output_dir=tmp_path,
        shard_dirs=shard_dirs,
        devices=("cuda:1", "cuda:0"),
    )
    assert rerun["dataset_plan_sha256"] == result["dataset_plan_sha256"]
    assert rerun["replaced_legacy_plan_sha256"] is None
    (tmp_path / "pitch3_lte_dataset_summary.json").unlink()
    recovered = validate_pitch3_lte_dataset_preflight(
        tmp_path / "pitch3_lte_examples.csv"
    )
    assert recovered["local_preflight_passed"] is True
    assert recovered["dataset_manifest_sha256"] == result["dataset_manifest_sha256"]
    assert recovered["preflight_source"] == "recomputed_from_canonical_plan_and_manifest"
    published["items_sha256"] = "d" * 64
    legacy_path.write_text(json.dumps(published), encoding="utf-8")
    with pytest.raises(LTSNContractError, match="changed in fields items_sha256"):
        merge_pitch3_lte_dataset_shards(
            output_dir=tmp_path,
            shard_dirs=shard_dirs,
            devices=("cuda:0", "cuda:1"),
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is server-only")
def test_lte_model_losses_and_one_shot_guidance() -> None:
    import torch

    from generation.pitch3_lte import (
        Pitch3LTEConfig,
        Pitch3LTECorrector,
        Pitch3LTEEnergyEnsemble,
        Pitch3LTEGuidanceConfig,
        PromptConditionedTopologyEnergy,
    )
    from generation.pitch3_lte_training import pitch3_lte_raw_losses

    torch.manual_seed(7)
    model = PromptConditionedTopologyEnergy(
        Pitch3LTEConfig(
            model_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            feedforward_dim=32,
            temporal_stride=2,
            dropout=0.0,
        )
    )
    latent = torch.randn(8, 12, 64)
    latent_mask = torch.ones(8, 12, dtype=torch.bool)
    text = torch.randn(8, 5, 1024)
    text_mask = torch.ones(8, 5, dtype=torch.bool)
    predicted = model(latent, latent_mask, text, text_mask).energy
    assert predicted.shape == (8,)
    batch = {
        "source_kind": ["base_step4_seed"] * 4 + ["local_finite_difference"] * 4,
        "prompt_id": ["p1"] * 8,
        "energy_target": torch.linspace(0.1, 0.8, 8),
        "direction_id": ["", "", "", "", "d1", "d1", "d2", "d2"],
        "direction_sign": torch.tensor([0, 0, 0, 0, -1, 1, -1, 1]),
        "epsilon": torch.tensor([0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1]),
    }
    losses = pitch3_lte_raw_losses(predicted, batch, huber_delta=1.0, rank_min_delta=1e-6)
    assert set(losses) == {"value", "prompt_rank", "local_fd"}
    assert all(torch.isfinite(value) and value >= 0 for value in losses.values())
    robust = pitch3_lte_raw_losses(
        predicted,
        batch,
        huber_delta=1.0,
        rank_min_delta=1e-6,
        local_objective="robust_direction_v31",
        local_derivative_scale=0.5,
        local_delta_scale=0.1,
    )
    assert set(robust) == {"value", "prompt_rank", "local_robust"}
    assert all(torch.isfinite(value) and value >= 0 for value in robust.values())
    v32_batch = {**batch, "prompt_id": ["p1", "p1", "p2", "p2", "p1", "p1", "p2", "p2"]}
    decomposed = pitch3_lte_raw_losses(
        predicted,
        v32_batch,
        huber_delta=1.0,
        rank_min_delta=1e-6,
        local_objective="decomposed_direction_v32",
        local_derivative_scale=0.5,
        local_delta_scale=0.1,
        include_cross_prompt_rank=True,
        energy_strata_thresholds=(0.0, 0.4),
        energy_strata_weights=(1.0, 1.5, 2.0),
    )
    assert set(decomposed) == {
        "value",
        "prompt_rank",
        "cross_prompt_rank",
        "local_direction",
        "local_shape",
        "local_flat",
    }
    assert all(torch.isfinite(value) and value >= 0 for value in decomposed.values())
    anchored_direction_flat = pitch3_lte_raw_losses(
        predicted,
        v32_batch,
        huber_delta=1.0,
        rank_min_delta=1e-6,
        local_objective="anchored_direction_flat_v35r",
        local_derivative_scale=0.5,
        local_delta_scale=0.1,
    )
    assert set(anchored_direction_flat) == {
        "value",
        "prompt_rank",
        "local_direction",
        "local_flat",
    }
    assert "local_shape" not in anchored_direction_flat
    assert all(
        torch.isfinite(value) and value >= 0
        for value in anchored_direction_flat.values()
    )

    residual_model = PromptConditionedTopologyEnergy(
        Pitch3LTEConfig(
            model_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            feedforward_dim=32,
            temporal_stride=2,
            dropout=0.0,
            fusion_mode="latent_primary_residual_v31",
        )
    ).eval()
    residual_first = residual_model(latent, latent_mask, text, text_mask).energy
    residual_second = residual_model(latent, latent_mask, text.flip(1), text_mask).energy
    assert torch.allclose(residual_first, residual_second)

    dual_model = PromptConditionedTopologyEnergy(
        Pitch3LTEConfig(
            model_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            feedforward_dim=32,
            temporal_stride=2,
            dropout=0.0,
            fusion_mode="latent_primary_residual_v31",
            latent_stem_mode="dual_rms_v32",
        )
    )
    assert dual_model(latent, latent_mask, text, text_mask).energy.shape == (8,)

    v33_model = PromptConditionedTopologyEnergy(
        Pitch3LTEConfig(
            model_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            feedforward_dim=32,
            temporal_stride=2,
            dropout=0.0,
            fusion_mode="latent_primary_residual_v31",
            latent_stem_mode="dual_rms_v32",
            potential_mode="global_local_v33",
        )
    ).eval()
    latent_state = v33_model.encode_latent(latent, latent_mask)
    prompt_state = v33_model.encode_prompt(text, text_mask)
    components = v33_model.potential_components_from_states(latent_state, prompt_state)
    assert torch.allclose(components.energy, components.global_energy + components.local_energy)
    assert torch.count_nonzero(components.local_energy) == 0

    v34_model = PromptConditionedTopologyEnergy(
        Pitch3LTEConfig(
            model_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            feedforward_dim=32,
            temporal_stride=2,
            dropout=0.0,
            fusion_mode="latent_primary_residual_v31",
            latent_stem_mode="dual_rms_v32",
            potential_mode="structured_coordinate_v34",
        )
    ).eval()
    v34_latent_state = v34_model.encode_latent(latent, latent_mask)
    v34_prompt_state = v34_model.encode_prompt(text, text_mask)
    v34_components = v34_model.potential_components_from_states(
        v34_latent_state, v34_prompt_state
    )
    assert v34_components.coordinates is not None
    below = torch.relu(v34_model.coordinate_lower - v34_components.coordinates)
    above = torch.relu(v34_components.coordinates - v34_model.coordinate_upper)
    analytic = torch.log1p(
        (
            (below.square() + above.square())
            * v34_model.coordinate_distance_weights
        ).sum(dim=1)
    )
    assert v34_components.coordinates.shape == (8, 3)
    assert torch.allclose(v34_components.global_energy, analytic)
    assert torch.allclose(
        v34_components.energy,
        v34_components.global_energy + v34_components.local_energy,
    )
    assert torch.count_nonzero(v34_components.local_energy) == 0
    v34_batch = {
        **batch,
        "source_kind": ["base_step4_seed"] * 8,
        "prompt_id": ["p1"] * 8,
        "prompt_family": ["family_a"] * 8,
        "coordinates": torch.randn(8, 3),
        "direction_id": [""] * 8,
        "direction_sign": torch.zeros(8, dtype=torch.long),
        "epsilon": torch.zeros(8),
    }
    v34_losses = pitch3_lte_raw_losses(
        v34_components.global_energy,
        v34_batch,
        huber_delta=1.0,
        rank_min_delta=1e-6,
        predicted_coordinates=v34_components.coordinates,
        include_coordinate_loss=True,
        include_family_listwise=True,
        family_rank_temperature=0.1,
    )
    assert {"coordinate", "family_listwise"} <= set(v34_losses)
    assert all(torch.isfinite(value) and value >= 0 for value in v34_losses.values())

    v35_model = PromptConditionedTopologyEnergy(
        Pitch3LTEConfig(
            model_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            feedforward_dim=32,
            temporal_stride=2,
            dropout=0.0,
            fusion_mode="latent_primary_residual_v31",
            latent_stem_mode="dual_rms_v32",
            potential_mode="structured_anchored_v35",
        )
    ).eval()
    torch.nn.init.normal_(v35_model.local_energy_head[-1].weight)
    current = latent.detach().clone().requires_grad_(True)
    anchored = v35_model.potential_components(
        current,
        latent_mask,
        text,
        text_mask,
        anchor_latent=latent.detach(),
        anchor_attention_mask=latent_mask,
    )
    assert torch.count_nonzero(anchored.local_energy) == 0
    anchored.energy.sum().backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    moved = v35_model.potential_components(
        latent + 0.01,
        latent_mask,
        text,
        text_mask,
        anchor_latent=latent,
        anchor_attention_mask=latent_mask,
    )
    assert torch.count_nonzero(moved.local_energy) > 0

    structured_batch = {
        **batch,
        "prompt_family": ["family_a"] * 8,
        "coordinates": torch.randn(8, 3),
    }
    structured_losses = pitch3_lte_raw_losses(
        predicted,
        structured_batch,
        huber_delta=1.0,
        rank_min_delta=1e-6,
        predicted_coordinates=torch.randn(8, 3),
        predicted_coordinates_shuffled=torch.randn(8, 3),
        include_coordinate_loss=True,
        include_boundary_region=True,
        include_band_component=True,
        include_coordinate_fd=True,
        include_coordinate_prompt_consistency=True,
        coordinate_lower=(0.0, 0.0, 0.0),
        coordinate_upper=(1.0, 1.0, 1.0),
        coordinate_distance_weights=(1 / 3, 1 / 3, 1 / 3),
        coordinate_region_weights=((1.0, 1.0, 1.0),) * 3,
        coordinate_fd_scales=(1.0, 1.0, 1.0),
    )
    assert {
        "boundary_region",
        "band_component",
        "coordinate_fd",
        "coordinate_prompt_consistency",
    } <= set(structured_losses)
    assert all(torch.isfinite(value) and value >= 0 for value in structured_losses.values())
    ensemble = Pitch3LTEEnergyEnsemble([dual_model.eval(), v33_model])
    ensemble_energy = ensemble(latent, latent_mask, text, text_mask).energy
    expected_energy = 0.5 * (
        dual_model(latent, latent_mask, text, text_mask).energy
        + v33_model(latent, latent_mask, text, text_mask).energy
    )
    assert torch.allclose(ensemble_energy, expected_energy)

    corrector = Pitch3LTECorrector(
        model,
        text[:1],
        text_mask[:1],
        {
            "model_family": "pitch3_prompt_conditioned_local_energy_v3",
            "guidance_steps": [4],
        },
        Pitch3LTEGuidanceConfig(enabled=True),
    )
    corrected, diagnostics = corrector.propose_clean_update(latent[:1], latent_mask[:1])
    clean_rms = latent[:1].square().mean().sqrt()
    update_rms = (corrected - latent[:1]).square().mean().sqrt()
    assert update_rms <= 0.025 * clean_rms + 1e-6
    if diagnostics.applied.item():
        assert diagnostics.energy_after.item() <= diagnostics.energy_before.item()

    untouched, skipped = corrector.apply_with_diagnostics(
        xt_next=latent[:1],
        xt_before_step=latent[:1],
        velocity=torch.zeros_like(latent[:1]),
        timestep=0.75,
        next_timestep=0.625,
        step_index=4,
        attention_mask=latent_mask[:1],
    )
    assert skipped is None
    assert torch.equal(untouched, latent[:1])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is server-only")
def test_lte_v33_family_round_robin_covers_within_family_pairs() -> None:
    from generation.pitch3_lte_training import PromptBatchSampler

    records = []
    for family in ("family_a", "family_b"):
        for variant in range(4):
            prompt_id = f"{family}__v{variant:02d}"
            records.extend(
                SimpleNamespace(
                    prompt_id=prompt_id,
                    prompt_family=family,
                    source_kind="base_step4_seed",
                    direction_sign=0,
                    direction_id="",
                )
                for _ in range(4)
            )
            for direction in range(2):
                for sign in (-1, 1):
                    records.append(
                        SimpleNamespace(
                            prompt_id=prompt_id,
                            prompt_family=family,
                            source_kind="local_finite_difference",
                            direction_sign=sign,
                            direction_id=f"{prompt_id}__d{direction}",
                        )
                    )
    sampler = PromptBatchSampler(
        records,
        seed=17,
        shuffle=False,
        groups_per_batch=2,
        pairing_mode="family_round_robin_v33",
    )
    observed: dict[str, set[tuple[str, str]]] = {"family_a": set(), "family_b": set()}
    for epoch in range(1, 4):
        sampler.set_epoch(epoch)
        for batch in sampler:
            prompt_ids = sorted({records[index].prompt_id for index in batch})
            families = {records[index].prompt_family for index in batch}
            assert len(prompt_ids) == 2
            assert len(families) == 1
            observed[families.pop()].add(tuple(prompt_ids))
    assert all(len(pairs) == 6 for pairs in observed.values())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is server-only")
def test_lte_v34_full_family_sampler_uses_all_base_rows() -> None:
    from generation.pitch3_lte_training import PromptBatchSampler

    records = []
    for family in ("family_a", "family_b"):
        for variant in range(16):
            prompt_id = f"{family}__v{variant:02d}"
            records.extend(
                SimpleNamespace(
                    prompt_id=prompt_id,
                    prompt_family=family,
                    source_kind="base_step4_seed",
                    direction_sign=0,
                    direction_id="",
                )
                for _ in range(4)
            )
            for direction in range(2):
                for sign in (-1, 1):
                    records.append(
                        SimpleNamespace(
                            prompt_id=prompt_id,
                            prompt_family=family,
                            source_kind="local_finite_difference",
                            direction_sign=sign,
                            direction_id=f"{prompt_id}__d{direction}",
                        )
                    )
    sampler = PromptBatchSampler(
        records,
        seed=23,
        shuffle=False,
        groups_per_batch=16,
        pairing_mode="family_full_base_v34",
    )
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 64
        assert len({records[index].prompt_id for index in batch}) == 16
        assert len({records[index].prompt_family for index in batch}) == 1
        assert {records[index].source_kind for index in batch} == {"base_step4_seed"}

    v35_sampler = PromptBatchSampler(
        records,
        seed=23,
        shuffle=False,
        groups_per_batch=16,
        pairing_mode="family_full_all_v35",
    )
    v35_batches = list(v35_sampler)
    assert len(v35_batches) == 4
    for batch in v35_batches:
        assert len(batch) == 64
        assert len({records[index].prompt_id for index in batch}) == 16
        assert len({records[index].prompt_family for index in batch}) == 1
        kinds = [records[index].source_kind for index in batch]
        assert set(kinds) in ({"base_step4_seed"}, {"local_finite_difference"})

    v35r_sampler = PromptBatchSampler(
        records,
        seed=23,
        shuffle=False,
        groups_per_batch=16,
        pairing_mode="family_full_base_v34",
    )
    v35r_batches = list(v35r_sampler)
    assert len(v35r_batches) == 2
    assert all(len(batch) == 64 for batch in v35r_batches)
    assert all(
        {records[index].source_kind for index in batch} == {"base_step4_seed"}
        for batch in v35r_batches
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is server-only")
def test_lte_v35r_frozen_training_contract() -> None:
    from generation.pitch3_lte_ensemble import _ENSEMBLE_KINDS
    from generation.pitch3_lte_training import (
        _loss_component_weights,
        _stabilize_loss_normalizer_medians,
        load_pitch3_lte_config,
    )

    config_path = Path(__file__).resolve().parents[1] / "configs" / "pitch3_lte_v35r.toml"
    model, training = load_pitch3_lte_config(config_path)
    assert model.potential_mode == "structured_anchored_v35"
    assert training.training_schedule == "structured_anchored_global_then_local_v35r"
    assert training.local_objective == "anchored_direction_flat_v35r"
    assert training.local_learning_rate == pytest.approx(2e-5)
    assert training.global_value_base_only is True
    assert training.local_shape_weight == 0
    assert (
        training.boundary_region_weight,
        training.band_component_weight,
        training.coordinate_fd_weight,
        training.coordinate_prompt_consistency_weight,
    ) == (0, 0, 0, 0)
    assert (
        _ENSEMBLE_KINDS["v3.5r_minimal_global_anchored_direction"]
        == "equal_weight_anchored_scalar_energy_v35r"
    )
    enabled = {
        "value": 1.0,
        "prompt_rank": 1.0,
        "coordinate": 1.0,
        "family_listwise": 1.0,
        "local_direction": 1.0,
        "local_flat": 1.0,
    }
    stabilized = _stabilize_loss_normalizer_medians(enabled)
    assert stabilized == enabled
    assert "coordinate_prompt_consistency" not in stabilized
    assert set(_loss_component_weights(training, tuple(stabilized))) == set(enabled)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is server-only")
def test_lte_quality_finalizer_uses_complete_clustered_cohort(tmp_path) -> None:
    from generation.pitch3_lte_evaluation import finalize_pitch3_lte_quality

    metrics = tmp_path / "metrics.csv"
    rows = []
    for prompt in range(64):
        for seed in range(4):
            rows.append(
                {
                    "pair_id": f"p{prompt}_s{seed}",
                    "prompt_id": f"p{prompt}",
                    "quality_baseline": "0.2",
                    "quality_guided": "0.3",
                    "prompt_baseline": "0.4",
                    "prompt_guided": "0.5",
                    "diversity_baseline": "0.6",
                    "diversity_guided": "0.7",
                }
            )
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = finalize_pitch3_lte_quality(
        metric_table_path=metrics,
        output_path=tmp_path / "quality.json",
        bootstrap_resamples=1000,
    )
    assert report["status"] == "passed"
    assert report["quality_noninferior"] is True
