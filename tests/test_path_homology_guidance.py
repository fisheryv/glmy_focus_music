from __future__ import annotations

# ruff: noqa: E402, I001

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from generation.ltsn_contract import (
    CANONICAL_FEATURE_ORDER,
    DISTANCE_WEIGHTS,
    FingerprintContract,
)
from generation.ltsn_losses import block_balanced_huber
from generation.path_homology_surrogate import LTSNConfig, LTSNOutput, PathHomologySurrogate
from generation.topology_corrector import TopologyCorrector, TopologyCorrectorConfig


def _contract(threshold: float = 10.0) -> FingerprintContract:
    coef = (1.0,) * 18
    classifier_payload = json.dumps(
        {"coef": list(coef), "intercept": 0.0},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return FingerprintContract(
        fingerprint_id="focus_path_homology_fingerprint_v2",
        spec_revision="test",
        artifact_sha256="a" * 64,
        feature_order=CANONICAL_FEATURE_ORDER,
        distance_weights=DISTANCE_WEIGHTS,
        classifier_coef=coef,
        classifier_intercept=0.0,
        focus_band_threshold=threshold,
        classifier_sha256=hashlib.sha256(classifier_payload).hexdigest(),
        input_sha256="b" * 64,
        config_sha256="d" * 64,
        code_sha256="c" * 64,
    )


def _checkpoint(contract: FingerprintContract) -> dict[str, object]:
    return {
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": 18,
        "feature_order": list(contract.feature_order),
        "distance_weights": list(contract.distance_weights),
        "classifier_sha256": contract.classifier_sha256,
        "ltsn_config_sha256": "1" * 64,
        "training_manifest_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "exact_label_table_sha256": "4" * 64,
        "ace_model_sha256": "5" * 64,
        "vae_sha256": "6" * 64,
    }


class _LinearSurrogate(nn.Module):
    def __init__(self, ood_logit: float = -20.0) -> None:
        super().__init__()
        self.ood_value = ood_logit

    def forward(self, latent, timestep, step_number, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(latent.dtype)
        pooled = (latent * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1.0)
        mean = pooled.unsqueeze(1).expand(-1, 18)
        logvar = torch.full_like(mean, -8.0)
        ood = torch.full_like(pooled, self.ood_value)
        return LTSNOutput(mean, logvar, ood, mean.sum(dim=1))


def _corrector(model: nn.Module, contract: FingerprintContract) -> TopologyCorrector:
    config = TopologyCorrectorConfig(
        enabled=True,
        qualification_passed=True,
        rms_clip_ratio=0.005,
        ood_probability_threshold=0.5,
        max_aleatoric_variance=1.0,
        max_epistemic_variance=1.0,
        max_interval_width=10.0,
    )
    return TopologyCorrector([model], contract, [_checkpoint(contract)], config)


def test_surrogate_outputs_frozen_dimensions_and_time_conditioning() -> None:
    torch.manual_seed(7)
    model = PathHomologySurrogate(_contract(), LTSNConfig(dropout=0.0)).eval()
    latent = torch.randn(2, 80, 64)
    mask = torch.ones(2, 80, dtype=torch.bool)

    first = model(latent, 0.75, 4, mask)
    second = model(latent, 0.50, 6, mask)

    assert first.coordinate_mean.shape == (2, 18)
    assert first.coordinate_logvar.shape == (2, 18)
    assert first.ood_logit.shape == (2,)
    assert not torch.allclose(first.coordinate_mean, second.coordinate_mean)


def test_block_balanced_loss_does_not_overweight_pitch_dimension_count() -> None:
    target = torch.zeros(1, 18)
    prediction = torch.full((1, 18), 0.5)

    loss = block_balanced_huber(prediction, target)

    assert loss.item() == pytest.approx(0.125)


def test_corrector_only_changes_valid_unprotected_frames_and_respects_rms_clip() -> None:
    contract = _contract()
    corrector = _corrector(_LinearSurrogate(), contract)
    before = torch.ones(1, 20, 64)
    velocity = torch.zeros_like(before)
    next_latent = before.clone()
    attention = torch.ones(1, 20, dtype=torch.bool)
    attention[:, 15:] = False
    repaint = torch.ones_like(attention)
    repaint[:, :5] = False

    corrected, diagnostics = corrector.apply_with_diagnostics(
        xt_next=next_latent,
        xt_before_step=before,
        velocity=velocity,
        timestep=0.75,
        next_timestep=0.64,
        step_index=3,
        attention_mask=attention,
        repaint_mask=repaint,
    )

    update = corrected - next_latent
    assert diagnostics is not None and diagnostics.applied.item()
    assert torch.equal(update[:, :5], torch.zeros_like(update[:, :5]))
    assert torch.equal(update[:, 15:], torch.zeros_like(update[:, 15:]))
    valid_rms = update[:, 5:15].square().mean().sqrt().item()
    assert valid_rms <= (1.0 - 0.64) * 0.5 * 0.005 + 1e-7


def test_corrector_is_noop_outside_window_or_on_ood() -> None:
    contract = _contract()
    latent = torch.ones(1, 12, 64)
    mask = torch.ones(1, 12, dtype=torch.bool)
    kwargs = dict(
        xt_next=latent,
        xt_before_step=latent,
        velocity=torch.zeros_like(latent),
        timestep=0.9,
        next_timestep=0.83,
        attention_mask=mask,
    )

    outside = _corrector(_LinearSurrogate(), contract)(step_index=1, **kwargs)
    ood = _corrector(_LinearSurrogate(ood_logit=20.0), contract)(step_index=3, **kwargs)

    assert torch.equal(outside, latent)
    assert torch.equal(ood, latent)
