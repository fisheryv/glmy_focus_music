from __future__ import annotations

import csv
import importlib.util

import numpy as np
import pytest

from generation.pitch3_lte_data import _anchor_index, _direction


def test_lte_seed_difference_direction_has_unit_rms() -> None:
    values = np.arange(24, dtype=np.float32).reshape(3, 8) - 7.0
    direction = _direction(values)
    assert direction.dtype == np.float32
    assert np.sqrt(np.mean(np.square(direction), dtype=np.float64)) == pytest.approx(1.0)
    assert _anchor_index("p01__v01") == _anchor_index("p01__v01")
    assert 0 <= _anchor_index("p01__v01") < 4


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is server-only")
def test_lte_model_losses_and_one_shot_guidance() -> None:
    import torch

    from generation.pitch3_lte import (
        Pitch3LTEConfig,
        Pitch3LTECorrector,
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
        "energy_target": torch.linspace(0.1, 0.8, 8),
        "direction_id": ["", "", "", "", "d1", "d1", "d2", "d2"],
        "direction_sign": torch.tensor([0, 0, 0, 0, -1, 1, -1, 1]),
        "epsilon": torch.tensor([0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1]),
    }
    losses = pitch3_lte_raw_losses(predicted, batch, huber_delta=1.0, rank_min_delta=1e-6)
    assert set(losses) == {"value", "prompt_rank", "local_fd"}
    assert all(torch.isfinite(value) and value >= 0 for value in losses.values())

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
