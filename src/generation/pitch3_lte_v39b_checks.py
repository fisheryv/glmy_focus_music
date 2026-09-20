"""Synthetic gradient/optimizer checks for the transition auxiliary task."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from .pitch3_lte_training import _set_training_stage_trainable, load_pitch3_lte_config
from .pitch3_lte_v39a import build_matched_models
from .pitch3_lte_v39a_checks import run_checks as run_a_checks
from .pitch3_lte_v39b import forward_bundle


def synthetic_case(config, device):
    torch.manual_seed(391)
    latent = torch.randn(8, 17, config.latent_dim, device=device)
    mask = torch.ones(8, 17, dtype=torch.bool, device=device)
    mask[0, 12:] = False
    band = latent.new_tensor([0, 0.001, 0.01, 0.2, 0, 0.002, 0.04, 0.8])
    sample = {
        "latent": latent,
        "attention_mask": mask,
        "text_hidden": torch.randn(8, 2, config.text_dim, device=device),
        "text_mask": torch.ones(8, 2, dtype=torch.bool, device=device),
        "sample_id": [f"synthetic_{i}" for i in range(8)],
        "source_kind": ["base_step4_seed"] * 8,
        "prompt_family": ["synthetic"] * 8,
        "prompt_id": ["p0"] * 4 + ["p1"] * 4,
        "energy_target": band.log1p(),
        "exact_band": band,
        "coordinates": torch.randn(8, 3, device=device),
        "direction_id": [""] * 8,
        "direction_sign": torch.zeros(8, dtype=torch.long, device=device),
        "epsilon": band * 0,
    }
    targets = {}
    for i, sid in enumerate(sample["sample_id"]):
        p = np.zeros((16, 16), dtype=np.float64)
        p[i, i] = 0.7
        p[i, (i + 1) % 16] = 0.3
        targets[sid] = p
    stats = {
        "energy_stratification": {"thresholds": [0, 0.01, 0.1], "weights": [1.0] * 4},
        "coordinate_auxiliary": {
            "coordinate_lower": config.coordinate_lower,
            "coordinate_upper": config.coordinate_upper,
        },
        "local_training_scales": {"local_derivative_scale": 1.0, "local_delta_scale": 1.0},
    }
    contract = SimpleNamespace(
        transform_center=(0.0, 0.0, 0.0),
        transform_scale=(1.0, 1.0, 1.0),
        target_lower=config.coordinate_lower,
        target_upper=config.coordinate_upper,
    )
    return sample, stats, targets, contract


def run_checks(config_path, device_name="cpu"):
    inherited = run_a_checks(config_path, device_name)
    config, training = load_pitch3_lte_config(config_path)
    device = torch.device(device_name)
    _, model, _ = build_matched_models(config, 20260941)
    model.to(device)
    _set_training_stage_trainable(model, training, "global")
    model.eval()
    head = torch.nn.Linear(config.model_dim, 256).to(device)
    sample, stats, targets, contract = synthetic_case(config, device)
    base_losses, base_parts = forward_bundle(model, None, sample, training, stats, {}, contract)
    losses, parts = forward_bundle(model, head, sample, training, stats, targets, contract)
    torch.testing.assert_close(base_parts.global_energy, parts.global_energy, rtol=0, atol=0)
    for key in ("value", "prompt_rank", "coordinate", "family_stratified_rank"):
        torch.testing.assert_close(base_losses[key], losses[key], rtol=0, atol=0)
    optimizer = torch.optim.AdamW(
        [*model.transition_branch.parameters(), *head.parameters()], lr=2e-4
    )
    before = model.transition_branch.pair_mlp[0].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    (losses["transition_kl"] + losses["transition_coordinate"]).backward()
    gradient = model.transition_branch.pair_mlp[0].weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
        raise AssertionError("Teacher task does not reach pair features")
    optimizer.step()
    if torch.equal(before, model.transition_branch.pair_mlp[0].weight.detach()):
        raise AssertionError("Teacher task did not update pair features")
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "aux.pt"
        torch.save(head.state_dict(), checkpoint)
        restored = torch.nn.Linear(config.model_dim, 256).to(device)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        for left, right in zip(head.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    return {
        "all_checks_passed": True,
        "synthetic_only": True,
        "inherited_a_checks": inherited,
        "auxiliary_does_not_change_initial_global_outputs": True,
        "teacher_gradient_reaches_pair_features": True,
        "optimizer_changes_pair_features": True,
        "auxiliary_checkpoint_roundtrip": True,
        "production_authorization": False,
    }
