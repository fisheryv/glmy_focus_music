"""Synthetic tensor checks; no real-data accuracy or qualification claims."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .ltsn_contract import sha256_file
from .pitch3_lte_training import (
    _seed_everything,
    load_pitch3_lte_checkpoint,
    load_pitch3_lte_config,
)
from .pitch3_lte_v39a import build_matched_models, check_initial_identity


def run_checks(config_path: Path, device_name="cpu") -> dict:
    config, training = load_pitch3_lte_config(config_path)
    device = torch.device(device_name)
    base, model, initial = build_matched_models(config, 20260941)
    base.to(device).eval()
    model.to(device).eval()
    _seed_everything(7)
    z = torch.randn(8, 33, config.latent_dim, device=device)
    mask = torch.ones(8, 33, device=device, dtype=torch.bool)
    mask[0, 25:] = False
    text = torch.randn(8, 3, config.text_dim, device=device)
    text_mask = torch.ones(8, 3, device=device, dtype=torch.bool)
    bands = z.new_tensor([0, 0.02, 0.1, 0.3, 0, 0.04, 0.8, 1.2])
    sample = {
        "latent": z,
        "attention_mask": mask,
        "text_hidden": text,
        "text_mask": text_mask,
        "sample_id": [f"synthetic_{i}" for i in range(8)],
        "source_kind": ["base_step4_seed"] * 8,
        "prompt_id": ["a"] * 4 + ["b"] * 4,
        "prompt_family": ["synthetic"] * 8,
        "energy_target": bands.log1p(),
        "exact_band": bands,
        "coordinates": torch.randn(8, 3, device=device),
        "direction_id": [""] * 8,
        "direction_sign": torch.zeros(8, dtype=torch.int64, device=device),
        "epsilon": z.new_zeros(8),
    }
    stats = {
        "local_training_scales": {"local_derivative_scale": 1.0, "local_delta_scale": 1.0},
        "energy_stratification": {"thresholds": [0, 0.03, 0.2], "weights": [1.0, 1.0, 1.0, 1.0]},
        "coordinate_auxiliary": {
            "coordinate_lower": config.coordinate_lower,
            "coordinate_upper": config.coordinate_upper,
        },
    }
    weights = {"value": 1.0, "prompt_rank": 1.0, "coordinate": 0.25, "family_stratified_rank": 1.0}
    initial.update(check_initial_identity(base, model, sample, training, stats, weights))
    # Random dropout draws in common modules must also match at initialization.
    base.train()
    model.train()
    _seed_everything(8)
    a = base.encode_latent(z, mask)
    _seed_everything(8)
    b = model.encode_latent(z, mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.eval()
    branch = model.transition_branch
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    (branch(z, mask) - 1).square().mean().backward()
    if not branch.output.weight.grad.abs().sum() > 0:
        raise AssertionError("Zero projection does not receive its first-step gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (branch(z, mask) - 1).square().mean().backward()
    if not branch.raw_projection.weight.grad.abs().sum() > 0:
        raise AssertionError("Feature path remains blocked after the first optimizer step")
    # Check actual branch behavior after it has acquired nonzero weights.
    clean = z[:2, :19].detach().clone().requires_grad_(True)
    valid = torch.ones(2, 19, dtype=torch.bool, device=device)
    valid[0, 7] = False
    changed = clean.detach().clone()
    changed[~valid] = float("nan")
    y = branch(clean, valid)
    torch.testing.assert_close(y, branch(changed, valid), rtol=1e-5, atol=1e-6)
    padded = torch.cat(
        (changed, torch.full((2, 9, config.latent_dim), float("nan"), device=device)), 1
    )
    padded_mask = torch.cat((valid, torch.zeros(2, 9, dtype=torch.bool, device=device)), 1)
    torch.testing.assert_close(y, branch(padded, padded_mask), rtol=1e-5, atol=1e-6)
    grad = torch.autograd.grad(y.square().sum(), clean)[0]
    if not torch.isfinite(grad).all() or grad[~valid].count_nonzero():
        raise AssertionError("Padding affected transition input gradients")
    tiny, flags = branch.features(z[:2, :1], torch.ones(2, 1, dtype=torch.bool, device=device))
    if flags.count_nonzero() or not torch.isfinite(tiny).all():
        raise AssertionError("Missing lags must have explicit zero validity")
    # Guidance differentiates a scalar energy. Compare its autograd directional
    # derivative with a central difference on synthetic inputs, not exact audio.
    x = z[:1].detach().clone().requires_grad_(True)
    u = torch.randn_like(x)
    u /= u.norm()

    def energy(values):
        return model(values, mask[:1], text[:1], text_mask[:1]).energy

    e = energy(x)
    derivative = (torch.autograd.grad(e.sum(), x)[0] * u).sum()
    with torch.no_grad():
        fd = ((energy(x + 0.01 * u) - energy(x - 0.01 * u)) / 0.02).sum()
    torch.testing.assert_close(derivative, fd, rtol=0.08, atol=3e-4)
    with TemporaryDirectory(prefix="pitch3_v39a_check_") as folder:
        path = Path(folder) / "model.pt"
        from dataclasses import asdict

        torch.save(
            {
                "model_config": asdict(model.config),
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "model_family": "pitch3_prompt_conditioned_local_energy_v3",
                    "guidance_steps": [4],
                },
            },
            path,
        )
        restored, _ = load_pitch3_lte_checkpoint(
            path, device=device, expected_sha256=sha256_file(path)
        )
        with torch.no_grad():
            torch.testing.assert_close(
                energy(z[:1]),
                restored(z[:1], mask[:1], text[:1], text_mask[:1]).energy,
                rtol=0,
                atol=0,
            )
            components = restored.potential_components(
                z, mask, text, text_mask, anchor_latent=z, anchor_attention_mask=mask
            )
            if components.local_energy.count_nonzero():
                raise AssertionError("Global-only checkpoint has nonzero local residual")
    return {
        "stage": "pitch3_lte_v39a_network_checks",
        "all_checks_passed": True,
        "synthetic_only": True,
        "accuracy_evaluated": False,
        "production_authorization": False,
        "device": str(device),
        "torch": str(torch.__version__),
        "initialization": initial,
        "checks": [
            "shared_parameters_outputs_losses",
            "matched_training_dropout",
            "branch_gradient_activation",
            "masked_padding_invariance",
            "masked_input_gradient",
            "missing_lag_flags",
            "scalar_input_derivative",
            "checkpoint_roundtrip",
            "zero_local_residual",
        ],
    }
