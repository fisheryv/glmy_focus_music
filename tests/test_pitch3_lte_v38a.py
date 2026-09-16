from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_v38a_retains_v37_model_and_nonranking_hyperparameters() -> None:
    old = tomllib.loads((ROOT / "configs/pitch3_lte_v37_dte.toml").read_text())
    new = tomllib.loads((ROOT / "configs/pitch3_lte_v38a.toml").read_text())
    assert old["model"] == new["model"]
    allowed = {"seed", "training_schedule", "family_listwise_weight", "family_rank_temperature"}
    for key, value in old["training"].items():
        if key not in allowed:
            assert new["training"][key] == value
    assert new["training"]["ranking_objective"] == "pre_softplus_stratified_v38a"
    assert new["training"]["family_listwise_weight"] == 0
    assert new["training"]["family_stratified_rank_weight"] == 1
    assert new["training"]["logit_rank_temperature"] == 1


def _batch(torch, bands, families=None, prompts=None):
    n = len(bands)
    exact = torch.tensor(bands, dtype=torch.float32)
    return {
        "source_kind": ["base_step4_seed"] * n,
        "prompt_family": families or ["f"] * n,
        "prompt_id": prompts or ["p"] * n,
        "exact_band": exact,
        "energy_target": torch.log1p(exact),
        "direction_id": [""] * n,
        "direction_sign": torch.zeros(n, dtype=torch.long),
        "epsilon": torch.ones(n) * 0.05,
    }


def test_logit_rank_has_gradient_at_low_energy_and_uses_exact_band_ties() -> None:
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte_training import pitch3_lte_raw_losses

    batch = _batch(torch, [0.0, 1e-5])
    logits = torch.tensor([-8.0, -9.0], requires_grad=True)
    energy = torch.nn.functional.softplus(logits)
    legacy = pitch3_lte_raw_losses(energy, batch, huber_delta=1, rank_min_delta=1e-6)
    losses = pitch3_lte_raw_losses(
        energy,
        batch,
        huber_delta=1,
        rank_min_delta=1e-6,
        ranking_logits=logits,
        ranking_objective="pre_softplus_stratified_v38a",
    )
    new_grad = torch.autograd.grad(losses["prompt_rank"], logits, retain_graph=True)[0]
    old_grad = torch.autograd.grad(legacy["prompt_rank"], logits)[0]
    assert torch.all(new_grad.abs() > 100 * old_grad.abs())
    assert new_grad.tolist() == pytest.approx([1 / (1 + math.exp(-1)), -1 / (1 + math.exp(-1))])
    assert losses["value"].item() == legacy["value"].item()
    for bands, sortable in [([0.0, 0.0], False), ([0.0, 1e-6], False), ([100.0, 100.00005], True)]:
        batch = _batch(torch, bands)
        losses = pitch3_lte_raw_losses(
            energy,
            batch,
            huber_delta=1,
            rank_min_delta=1e-6,
            ranking_logits=logits,
            ranking_objective="pre_softplus_stratified_v38a",
        )
        assert (losses["prompt_rank"].item() > 0) == sortable
    with pytest.raises(ValueError, match="logit ranking"):
        from generation.pitch3_lte_training import load_pitch3_lte_config

        _, training = load_pitch3_lte_config(ROOT / "configs/pitch3_lte_v38a.toml")
        replace(training, logit_rank_temperature=0.1).validate()


def test_family_rank_balances_stratum_pairs_and_families_excluding_local_and_ties() -> None:
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte_training import _stratified_family_logit_rank

    bands = np.array([0, 0, 0.01, 0.02, 0.03, 0.5, 0, 0.02, 0.8, 100.0])
    scores = np.array([-4, 2, -3, -5, -2, 1, 0, -1, 2, -100.0], dtype=np.float32)
    families = ["a"] * 6 + ["b"] * 3 + ["a"]
    batch = _batch(torch, bands, families)
    batch["source_kind"][-1] = "local_finite_difference"
    thresholds = [0.0, 0.05, 0.2]
    groups = defaultdict(lambda: defaultdict(list))
    buckets = np.searchsorted(thresholds, np.log1p(bands), side="left")
    for i in range(9):
        for j in range(i + 1, 9):
            if families[i] != families[j] or abs(bands[i] - bands[j]) <= 1e-6:
                continue
            key = tuple(sorted((buckets[i], buckets[j])))
            groups[families[i]][key].append(
                np.logaddexp(0, -np.sign(bands[i] - bands[j]) * (scores[i] - scores[j]))
            )
    expected = np.mean([np.mean([np.mean(v) for v in g.values()]) for g in groups.values()])
    logits = torch.tensor(scores, requires_grad=True)
    loss = _stratified_family_logit_rank(
        logits, batch["exact_band"], batch["energy_target"], batch, thresholds, 1.0, 1e-6
    )
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert logits.grad[-1].item() == 0
    tied = _batch(torch, [0, 0])
    empty = _stratified_family_logit_rank(
        logits[:2], tied["exact_band"], tied["energy_target"], tied, thresholds, 1, 1e-6
    )
    assert empty.item() == 0 and torch.isfinite(empty)


def test_direct_logit_is_exposed_but_ensemble_still_averages_energy() -> None:
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte import (
        Pitch3LTEConfig,
        Pitch3LTEEnergyEnsemble,
        PromptConditionedTopologyEnergy,
    )

    config = Pitch3LTEConfig(
        latent_dim=4,
        text_dim=3,
        model_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        feedforward_dim=16,
        dropout=0,
        fusion_mode="latent_primary_residual_v31",
        potential_mode="direct_anchored_v37",
    )
    models = [PromptConditionedTopologyEnergy(config).eval() for _ in range(2)]
    latent = torch.randn(2, 8, 4, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    text = torch.randn(2, 3, 3)
    text_mask = torch.ones(2, 3, dtype=torch.bool)
    parts = models[0].potential_components(latent, mask, text, text_mask)
    assert torch.equal(parts.global_energy, torch.nn.functional.softplus(parts.global_logit))
    assert parts.local_energy.count_nonzero() == 0
    parts.global_logit.sum().backward()
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    with torch.no_grad():
        for model, value in zip(models, [-5.0, 2.0], strict=True):
            model.latent_energy_head[-1].weight.zero_()
            model.latent_energy_head[-1].bias.fill_(value)
    ensemble = Pitch3LTEEnergyEnsemble(models).eval()
    result = ensemble.potential_components(latent, mask, text, text_mask)
    expected = torch.nn.functional.softplus(torch.tensor([-5.0, 2.0])).mean()
    assert torch.allclose(result.energy, expected.expand(2))
    assert result.global_logit is None
    assert not torch.allclose(result.energy, torch.nn.functional.softplus(torch.tensor(-1.5)))


def test_cv_never_uses_development_and_training_thresholds_remain_isolated() -> None:
    pytest.importorskip("torch")
    from generation.pitch3_lte_training import _energy_strata, _train_family_cv_split

    records = [
        SimpleNamespace(
            sample_id=f"f{family:02}_{i}",
            prompt_family=f"f{family:02}",
            split="train",
            energy_target=value,
        )
        for family in range(20)
        for i, value in enumerate([0, 0.01, 0.03, 0.2, 1.0, 2.0])
    ]
    seen = set()
    for fold in range(5):
        train, validation = _train_family_cv_split(records, fold)
        families = {r.prompt_family for r in validation}
        assert len(families) == 4 and not families & seen
        assert not families & {r.prompt_family for r in train}
        seen |= families
        expected = _energy_strata(train, 3, 4)
        originals = [r.energy_target for r in validation]
        for r in validation:
            r.energy_target = 1e10
        assert _energy_strata(train, 3, 4) == expected
        for r, value in zip(validation, originals, strict=True):
            r.energy_target = value
    assert len(seen) == 20
    with pytest.raises(Exception, match="original train"):
        _train_family_cv_split([SimpleNamespace(split="development")], 0)


def test_v38a_loss_passes_use_complete_families_for_global_metrics() -> None:
    pytest.importorskip("torch")
    from generation.pitch3_lte_training import _loss_passes, load_pitch3_lte_config

    _, training = load_pitch3_lte_config(ROOT / "configs/pitch3_lte_v38a.toml")
    full_family, prompt_pairs = object(), object()
    result = _loss_passes(
        training, prompt_pairs, full_family, ("value", "family_stratified_rank", "local_flat")
    )
    assert result == [
        (full_family, {"value", "family_stratified_rank"}),
        (prompt_pairs, {"local_flat"}),
    ]


def test_cv_summary_rejects_missing_folds_and_changed_split_hash(tmp_path) -> None:
    from scripts.summarize_pitch3_lte_cv import summarize

    with pytest.raises(FileNotFoundError):
        summarize(tmp_path)
    folder = tmp_path / "fold_0/models"
    folder.mkdir(parents=True)
    protocol = folder / "pitch3_lte_run_protocol.json"
    protocol.write_text("{}")
    manifest = {
        "validation_scope": "train_family_cv",
        "cv_fold": 0,
        "run_protocol_sha256": hashlib.sha256(b"original").hexdigest(),
    }
    (folder / "pitch3_lte_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol hash"):
        summarize(tmp_path)


def test_cv_summary_covers_each_training_family_once(tmp_path) -> None:
    from scripts.summarize_pitch3_lte_cv import summarize

    def write(path, payload):
        content = json.dumps(payload).encode()
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    families = [f"f{i:02}" for i in range(20)]
    for fold in range(5):
        folder = tmp_path / f"fold_{fold}/models"
        folder.mkdir(parents=True)
        validation = families[fold::5]
        protocol_hash = write(
            folder / "pitch3_lte_run_protocol.json",
            {
                "selection_families": validation,
                "train_families": sorted(set(families) - set(validation)),
                "train_sample_ids": ["train"],
                "selection_sample_ids": ["validation"],
            },
        )
        checkpoint_hash = write(folder / "model.pt", {"fixture": True})
        prediction_hash = write(folder / "pitch3_lte_selection_predictions.csv", {})
        write(
            folder / "pitch3_lte_manifest.json",
            {
                "validation_scope": "train_family_cv",
                "cv_fold": fold,
                "run_protocol_sha256": protocol_hash,
                "training_manifest_sha256": "1" * 64,
                "fingerprint_json_sha256": "2" * 64,
                "source_training_config_sha256": "3" * 64,
                "architecture_revision": "v3.8a_logit_stratified_rank_energy",
                "seed": 41,
                "local_residual_mode": "zero_global_only",
                "checkpoint": "model.pt",
                "checkpoint_sha256": checkpoint_hash,
                "selection_predictions_sha256": prediction_hash,
                "best_epoch": 2,
                "best_development_metrics": {
                    "prompt_family_spearman": dict.fromkeys(validation, 0.4 + 0.03 * fold),
                    "all_gates_passed": False,
                },
            },
        )
    report = summarize(tmp_path)
    assert len(report["family_spearman"]) == 20
    assert report["families_passing_original_rho_gate"] == 4
    assert report["minimum_family_spearman"] == pytest.approx(0.4)
    assert report["development_used"] is False
