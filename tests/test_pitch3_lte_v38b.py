from __future__ import annotations

import copy
import csv
import json
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generation.pitch3_lte_metrics import pitch3_lte_metrics
from generation.pitch3_lte_v38b_protocol import (
    REVISION,
    SELECTION,
    describe_predictions,
    file_hash,
    gradient_vector_statistics,
    validate_protocol,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _dataset():
    rows = []
    for family in range(24):
        for i in range(4):
            sid = f"f{family:02}_base{i}"
            rows.append(
                {
                    "sample_id": sid,
                    "trajectory_id": sid,
                    "prompt_family": f"f{family:02}",
                    "prompt_id": f"p{family}",
                    "split": "train" if family < 20 else "development",
                    "source_kind": "base_step4_seed",
                    "direction_id": "",
                    "direction_sign": "0",
                    "epsilon": "0.0",
                    "exact_band": str(float(i)),
                    "energy_target": str(math.log1p(i)),
                }
            )
    return rows


def _local_pair_rows(base_rows):
    rows = []
    for base in base_rows:
        for sign in [-1, 1]:
            band = max(0.0, float(base["exact_band"]) + sign * 0.1)
            rows.append(
                {
                    **base,
                    "sample_id": f"{base['sample_id']}_local_{sign}",
                    "source_kind": "local_finite_difference",
                    "direction_id": f"{base['sample_id']}_direction",
                    "direction_sign": str(float(sign)),
                    "epsilon": "0.5",
                    "exact_band": str(band),
                    "energy_target": str(math.log1p(band)),
                }
            )
    return rows


def _protocol(rows, fold=0):
    held = {f"f{i:02}" for i in range(20)[fold::5]} if fold is not None else set()
    fit = [r for r in rows if r["split"] == "train" and r["prompt_family"] not in held]
    outer = [r for r in rows if r["split"] == "train" and r["prompt_family"] in held]
    return {
        "cv_fold": fold,
        "validation_scope": "train_family_cv" if fold is not None else "development",
        "train_sample_ids": [r["sample_id"] for r in fit],
        "selection_sample_ids": [r["sample_id"] for r in outer],
        "train_families": sorted({r["prompt_family"] for r in fit}),
        "selection_families": sorted(held),
        "checkpoint_selection": SELECTION,
        "development_used": False,
        "outer_evaluation_count": 1 if fold is not None else 0,
        "experiment_contract": {"global_variant": "g1", "local_variant": "l0"},
    }


def _predictions(rows):
    return [
        {
            **{
                k: r[k]
                for k in [
                    "sample_id",
                    "prompt_id",
                    "prompt_family",
                    "source_kind",
                    "direction_id",
                    "direction_sign",
                    "epsilon",
                    "exact_band",
                ]
            },
            "exact_energy": r["energy_target"],
            "predicted_energy": r["energy_target"],
            "predicted_global_energy": r["energy_target"],
            "predicted_local_energy": "0.0",
            "predicted_coordinates_json": "[0,0,0]",
            "anchor_sample_id": r["trajectory_id"],
        }
        for r in rows
    ]


def test_protocol_keeps_outer_and_development_out_of_fit():
    rows = _dataset()
    for fold in [None, 0, 1, 2, 3, 4]:
        p = _protocol(rows, fold)
        validate_protocol(p, rows)
        bad = copy.deepcopy(p)
        bad["train_sample_ids"].append(rows[-1]["sample_id"])
        with pytest.raises(ValueError, match="original grouped split"):
            validate_protocol(bad, rows)
    bad = _protocol(rows)
    bad["checkpoint_selection"] = "best_outer_rho"
    with pytest.raises(ValueError, match="checkpoint selection"):
        validate_protocol(bad, rows)
    # A local point cannot move without its anchor.
    rows[0]["source_kind"] = "local_finite_difference"
    rows[0]["trajectory_id"] = rows[-1]["sample_id"]
    with pytest.raises(ValueError, match="anchor"):
        validate_protocol(_protocol(rows), rows)


def test_frozen_metrics_keep_family_gate_and_exact_ties():
    rows = _predictions(_dataset()[:8])
    metrics = pitch3_lte_metrics(rows)
    assert metrics["direct_energy_spearman"] == pytest.approx(1)
    assert metrics["same_prompt_rank_pairs"] == 12
    assert not metrics["gates"]["local_direction_sign_accuracy"]
    for row in rows[:4]:
        row["predicted_energy"] = "0.0"
    metrics = pitch3_lte_metrics(rows)
    assert not metrics["gates"]["every_prompt_family_spearman"]
    assert metrics["collapsed_prompt_count"] == 1
    assert metrics["prompt_collapse_definition"] == {
        "maximum_predicted_range": 0.001,
        "minimum_exact_range": 0.1,
        "maximum_fraction": 0.05,
    }


def test_diagnostics_use_supplied_training_strata_and_handle_constant_targets():
    rows = _predictions(_dataset()[:4])
    truth = {r["sample_id"]: SimpleNamespace(coordinates=(0, 0, 0)) for r in rows}
    result = describe_predictions(rows, truth, [0, 0.5, 1.0], 1.0, [1, 1, 1])
    family = result["families"]["f00"]
    assert family["zero_positive_auc"] == 1
    assert [s["n"] for s in family["train_defined_strata"]] == [1, 0, 1, 2]
    assert family["coordinate_rho"] == [None, None, None]
    assert result["local"]["nontrivial_pairs"] == 0
    json.dumps(result, allow_nan=False)


def test_gradient_statistics_norms_cosines_and_zero_vectors():
    vectors = {
        "a": np.array([3, 4], dtype=np.float32),
        "same": np.array([6, 8], dtype=np.float32),
        "opposite": np.array([-3, -4], dtype=np.float32),
        "orthogonal": np.array([-4, 3], dtype=np.float32),
        "zero": np.zeros(2, dtype=np.float32),
    }
    norms, cosine = gradient_vector_statistics(vectors)
    assert norms == pytest.approx({"a": 5, "same": 10, "opposite": 5, "orthogonal": 5, "zero": 0})
    assert cosine["a"]["same"] == pytest.approx(1)
    assert cosine["a"]["opposite"] == pytest.approx(-1)
    assert cosine["a"]["orthogonal"] == pytest.approx(0, abs=1e-15)
    for a in vectors:
        for b in vectors:
            if "zero" in (a, b):
                assert cosine[a][b] is None
            else:
                assert cosine[a][b] == pytest.approx(cosine[b][a])
                assert -1 <= cosine[a][b] <= 1
        if a != "zero":
            assert cosine[a][a] == pytest.approx(1)
    json.dumps({"norms": norms, "cosine": cosine}, allow_nan=False)


def test_gradient_statistics_accumulate_float32_extremes_in_float64():
    vectors = {
        "large": np.array([3e30, 4e30], dtype=np.float32),
        "small": np.array([3e-30, 4e-30], dtype=np.float32),
    }
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        norms, cosine = gradient_vector_statistics(vectors)
    assert norms["large"] == pytest.approx(5e30, rel=1e-6)
    assert norms["small"] == pytest.approx(5e-30, rel=1e-6, abs=0)
    assert cosine["large"]["small"] == pytest.approx(1)
    json.dumps({"norms": norms, "cosine": cosine}, allow_nan=False)


@pytest.mark.parametrize(
    "vectors, message",
    [
        ({"a": np.ones((2, 2))}, "one-dimensional"),
        ({"a": np.ones(2), "b": np.ones(3)}, "matching lengths"),
        ({"a": np.array([np.nan])}, "finite"),
        ({"a": np.array([np.inf])}, "finite"),
    ],
)
def test_gradient_statistics_reject_invalid_vectors(vectors, message):
    with pytest.raises(ValueError, match=message):
        gradient_vector_statistics(vectors)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_snapshot_avoids_cublas_dot_and_preserves_parameters(monkeypatch, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from generation import pitch3_lte_v38b as experiment

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([2.0, -1.0], device=device)))
    model.register_parameter("unused", torch.nn.Parameter(torch.tensor([7.0], device=device)))
    model.weight.grad = torch.ones_like(model.weight)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    def losses(model, *_args):
        a = 3 * model.weight[0] + 4 * model.weight[1]
        return {"a": a, "opposite": -a, "zero": a * 0}

    def unsupported_dot(*_args, **_kwargs):
        raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED when calling cublasSdot")

    monkeypatch.setattr(torch, "dot", unsupported_dot)
    monkeypatch.setattr(experiment, "forward_losses", losses)
    monkeypatch.setattr(experiment, "_set_training_stage_trainable", lambda *_args: None)
    result = experiment.gradient_snapshot(
        model,
        {"sample_id": ["diagnostic-fixture"]},
        "global",
        None,
        {"loss_normalizers": {"a": 4.0, "opposite": 2.0, "zero": 1.0}},
        {"a": 2.0, "opposite": 0.5, "zero": 1.0},
    )
    assert result["gradient_statistics_backend"] == "cpu_numpy_float64"
    assert result["weighted_normalized_gradient_norms"] == pytest.approx(
        {"a": 2.5, "opposite": 1.25, "zero": 0}
    )
    assert result["gradient_cosine"]["a"]["opposite"] == pytest.approx(-1)
    assert result["gradient_cosine"]["a"]["zero"] is None
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
        assert parameter.device == before[name].device
    assert torch.equal(model.weight.grad, torch.ones_like(model.weight))
    assert model.unused.grad is None
    json.dumps(result, allow_nan=False)


def test_energy_average_is_id_aligned_and_rejects_label_mismatch():
    from scripts.summarize_pitch3_lte_v38b import average_rows

    a = _predictions(_dataset()[:4])
    b = copy.deepcopy(a[::-1])
    for row in b:
        row["predicted_energy"] = float(row["predicted_energy"]) + 2
    result = average_rows([a, b])
    assert [float(r["predicted_energy"]) for r in result] == pytest.approx(
        [math.log1p(i) + 1 for i in range(4)]
    )
    b[0]["exact_energy"] = "999"
    with pytest.raises(ValueError, match="labels differ"):
        average_rows([a, b])


def test_direction_summary_detects_wrong_large_minority():
    from scripts.summarize_pitch3_lte_v38b import direction_disagreement

    members = []
    for derivative in [1.0, 2.0, -5.0]:
        members.append(
            [
                {
                    "direction_id": "d",
                    "direction_sign": sign,
                    "epsilon": 0.5,
                    "exact_energy": float(sign > 0),
                    "predicted_energy": derivative if sign > 0 else 0,
                }
                for sign in [-1, 1]
            ]
        )
    d = direction_disagreement(members)
    assert d["member_correct"] == [1, 1, 0]
    assert d["majority_correct_mean_wrong"] == 1 and d["mean_correct"] == 0


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return file_hash(path)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("with_local_pairs", [False, True])
def test_cv_summary_verifies_hashes_splits_and_recomputes_gates(tmp_path, with_local_pairs):
    from scripts.summarize_pitch3_lte_v38b import summarize

    rows = _dataset()
    if with_local_pairs:
        rows += _local_pair_rows(rows)
    dataset = tmp_path / "dataset.csv"
    _write_csv(dataset, rows)
    for fold in range(5):
        folder = tmp_path / f"fold_{fold}/seed_41/models"
        folder.mkdir(parents=True)
        protocol = {**_protocol(rows, fold), "dataset_manifest_sha256": file_hash(dataset)}
        p_hash = _write_json(folder / "pitch3_lte_run_protocol.json", protocol)
        stats_hash = _write_json(
            folder / "pitch3_lte_training_statistics.json", {"run_protocol_sha256": p_hash}
        )
        config_hash = _write_json(folder / "pitch3_lte_effective_config.json", {})
        checkpoint_hash = _write_json(folder / "fixture.pt", {"synthetic_protocol_fixture": True})
        fields = {}
        for split in ["train", "selection"]:
            pred = _predictions(
                [r for r in rows if r["sample_id"] in protocol[f"{split}_sample_ids"]]
            )
            # Match the real exporter: teacher CSV -1.0/1.0 becomes prediction -1/1.
            for row in pred:
                row["direction_sign"] = str(int(float(row["direction_sign"])))
            path = folder / f"pitch3_lte_{split}_predictions.csv"
            _write_csv(path, pred)
            fields[f"{split}_predictions_sha256"] = file_hash(path)
            if split == "selection":
                fields["best_development_metrics"] = pitch3_lte_metrics(pred)
        _write_json(
            folder / "pitch3_lte_manifest.json",
            {
                **protocol,
                **fields,
                "architecture_revision": REVISION,
                "seed": 41,
                "run_protocol_sha256": p_hash,
                "training_statistics_sha256": stats_hash,
                "training_config_sha256": config_hash,
                "checkpoint": "fixture.pt",
                "checkpoint_sha256": checkpoint_hash,
                "training_manifest_sha256": file_hash(dataset),
                "source_training_config_sha256": "a" * 64,
                "fingerprint_json_sha256": "b" * 64,
                "implementation_sha256": {},
                "local_residual_mode": "zero_global_only",
            },
        )
    artifacts = {p: file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = summarize(tmp_path, dataset, [41])
    assert report["families_passing_original_rho_gate"] == 20
    assert report["folds_passing_all_original_model_gates"] == (5 if with_local_pairs else 0)
    assert all(file_hash(p) == digest for p, digest in artifacts.items())
    path = tmp_path / "fold_0/seed_41/models/pitch3_lte_selection_predictions.csv"
    if with_local_pairs:
        # A true sign mismatch must fail even if the prediction file hash matches.
        with path.open(newline="", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        next(row for row in predictions if row["direction_sign"] == "-1")["direction_sign"] = "1"
        _write_csv(path, predictions)
        manifest_path = path.parent / "pitch3_lte_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["selection_predictions_sha256"] = file_hash(path)
        _write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="Prediction detached from true direction_sign"):
            summarize(tmp_path, dataset, [41])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash-mismatched"):
        summarize(tmp_path, dataset, [41])


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_summary_direction_sign_compares_numeric_labels_without_rewriting_truth(sign):
    from scripts.summarize_pitch3_lte_v38b import _validate_prediction_labels

    truth = _dataset()[0]
    truth["direction_sign"] = str(float(sign))
    prediction = _predictions([truth])[0]
    prediction["direction_sign"] = str(sign)
    original = dict(truth)
    _validate_prediction_labels(prediction, truth)
    assert truth == original
    assert prediction["direction_sign"] == str(sign)
    # Decimal-form predictions also become readable by the unchanged gate code.
    prediction["direction_sign"] = str(float(sign))
    _validate_prediction_labels(prediction, truth)
    assert prediction["direction_sign"] == str(sign)


@pytest.mark.parametrize("field_source", ["prediction", "dataset"])
@pytest.mark.parametrize("invalid", ["1.5", "1.00000000000000000001", "NaN", "Infinity", "2", ""])
def test_summary_rejects_invalid_direction_signs_without_truncation(field_source, invalid):
    from scripts.summarize_pitch3_lte_v38b import _validate_prediction_labels

    truth = _dataset()[0]
    prediction = _predictions([truth])[0]
    (prediction if field_source == "prediction" else truth)["direction_sign"] = invalid
    with pytest.raises(ValueError, match="Invalid direction_sign"):
        _validate_prediction_labels(prediction, truth)


def test_runner_local_branches_share_global_parent_and_default_cv_does_not_screen():
    from scripts.run_pitch3_lte_v38b import build_jobs

    args = SimpleNamespace(
        root=ROOT,
        run_root=ROOT / "runs/test",
        dataset_manifest=ROOT / "unused.csv",
        fingerprint=ROOT / "fingerprint.json",
        config=ROOT / "configs/pitch3_lte_v38b.toml",
        python="python",
        stage="cv-global",
        global_variants=None,
        local_variants=None,
        seeds=None,
        folds=list(range(5)),
        devices=["cuda:1", "cuda:2"],
    )
    jobs = build_jobs(args)
    assert len(jobs) == 10 and all("screen-development" not in j["command"] for j in jobs)
    args.stage, args.global_variants, args.folds = "cv-local", ["g1"], [0]
    jobs = build_jobs(args)
    assert len(jobs) == 2
    parents = [j["command"][j["command"].index("--global-manifest") + 1] for j in jobs]
    assert parents[0] == parents[1] and "l0" in parents[0]


def test_v38b_config_preserves_frozen_global_contract():
    new = tomllib.loads((ROOT / "configs/pitch3_lte_v38b.toml").read_text())
    old = tomllib.loads((ROOT / "configs/pitch3_lte_v38a.toml").read_text())
    assert {k: v for k, v in new["model"].items() if k != "ordinal_auxiliary"} == old["model"]
    assert new["training"] == old["training"]
    assert new["v38b"]["parameter_source"] == "online"
    assert new["v38b"]["global_epochs"] == new["v38b"]["local_epochs"] == 12


def test_ordinal_logits_are_monotone_and_do_not_change_energy_or_anchor():
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte import Pitch3LTEConfig, PromptConditionedTopologyEnergy

    cfg = Pitch3LTEConfig(
        latent_dim=4,
        text_dim=3,
        model_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        feedforward_dim=16,
        dropout=0,
        fusion_mode="latent_primary_residual_v31",
        potential_mode="direct_anchored_v37",
        ordinal_auxiliary=True,
    )
    model = PromptConditionedTopologyEnergy(cfg).eval()
    latent, prompt = torch.randn(5, 8), torch.randn(5, 8)
    before = model.potential_components_from_states(latent, prompt, anchor_latent_state=latent)
    assert torch.all(before.ordinal_logits[:, 1:] <= before.ordinal_logits[:, :-1])
    assert torch.equal(before.global_energy, torch.nn.functional.softplus(before.global_logit))
    with torch.no_grad():
        for p in model.ordinal_head.parameters():
            p.add_(100)
    after = model.potential_components_from_states(latent, prompt, anchor_latent_state=latent)
    assert torch.equal(before.energy, after.energy)
    assert after.local_energy.count_nonzero() == 0


def test_ordinal_supervision_uses_strict_thresholds_and_ignores_local_points():
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte_v38b import ordinal_loss

    logits = torch.zeros(4, 3, requires_grad=True)
    batch = {
        "source_kind": ["base_step4_seed"] * 3 + ["local_finite_difference"],
        "energy_target": torch.tensor([0.0, 0.2, 1.0, 999.0]),
    }
    loss = ordinal_loss(logits, batch, [0, 0.2, 1])
    loss.backward()
    assert torch.all(logits.grad[0] > 0)
    assert logits.grad[1, 0] < 0 and logits.grad[1, 1] > 0
    assert logits.grad[2, 2] > 0 and logits.grad[3].count_nonzero() == 0


def test_shape_loss_targets_total_derivative_magnitude_and_skips_flat_pairs():
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte_v38b import derivative_shape_loss

    batch = {
        "source_kind": ["local_finite_difference"] * 4,
        "prompt_id": ["p"] * 4,
        "direction_id": ["a", "a", "b", "b"],
        "direction_sign": torch.tensor([-1, 1, -1, 1]),
        "epsilon": torch.tensor([0.5] * 4),
        "energy_target": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    energy = torch.tensor([0.0, 3.0, 0.0, 10.0], requires_grad=True)
    derivative_shape_loss(energy, batch, 1.0).backward()
    assert energy.grad[1] > 0 and energy.grad[0] < 0
    assert energy.grad[2:].count_nonzero() == 0
    assert derivative_shape_loss(batch["energy_target"], batch, 1.0) == 0
    batch["epsilon"][1] = 0.1
    with pytest.raises(Exception, match="epsilon"):
        derivative_shape_loss(energy, batch, 1.0)


def test_local_training_freezes_global_and_ordinal_parameters():
    pytest.importorskip("torch")
    from generation.pitch3_lte import Pitch3LTEConfig, PromptConditionedTopologyEnergy
    from generation.pitch3_lte_training import _set_training_stage_trainable, load_pitch3_lte_config

    _, training = load_pitch3_lte_config(ROOT / "configs/pitch3_lte_v38b.toml")
    cfg = Pitch3LTEConfig(
        model_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        feedforward_dim=16,
        fusion_mode="latent_primary_residual_v31",
        potential_mode="direct_anchored_v37",
        ordinal_auxiliary=True,
    )
    model = PromptConditionedTopologyEnergy(cfg)
    _set_training_stage_trainable(model, training, "local")
    assert all(
        p.requires_grad == name.startswith("local_energy_head.")
        for name, p in model.named_parameters()
    )
    assert not model.training and not model.ordinal_head.training


def test_fixed_epoch_training_and_shared_parent_local_branches(tmp_path, monkeypatch):
    """CPU integration: real tensors/optimizers, synthetic labels; no model efficacy claim."""
    torch = pytest.importorskip("torch")
    from generation import pitch3_lte_v38b as module
    from generation.pitch3_lte_training import Pitch3LTEExample, load_pitch3_lte_checkpoint

    torch.set_num_threads(1)
    text = (ROOT / "configs/pitch3_lte_v38b.toml").read_text()
    for old, new in [
        ("model_dim = 128", "model_dim = 8"),
        ("transformer_heads = 4", "transformer_heads = 2"),
        ("transformer_layers = 3", "transformer_layers = 1"),
        ("feedforward_dim = 512", "feedforward_dim = 16"),
        ("dropout = 0.1", "dropout = 0.0"),
        ("global_epochs = 12", "global_epochs = 1"),
        ("local_epochs = 12", "local_epochs = 1"),
    ]:
        text = text.replace(old, new)
    config = tmp_path / "config.toml"
    config.write_text(text)
    cfg = tomllib.loads(text)["model"]
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text("{}")
    contract = SimpleNamespace(
        artifact_sha256=file_hash(fingerprint),
        fingerprint_id="synthetic",
        spec_revision="test",
        target_lower=cfg["coordinate_lower"],
        target_upper=cfg["coordinate_upper"],
        target_center=cfg["coordinate_center"],
        distance_weights=cfg["coordinate_distance_weights"],
    )
    monkeypatch.setattr(module, "load_pitch3_contract", lambda p: contract)
    rng = np.random.default_rng(71)
    embedding = tmp_path / "prompt.npz"
    np.savez(
        embedding, hidden=rng.normal(size=(2, 1024)).astype("float32"), mask=np.ones(2, dtype=bool)
    )
    paths = []
    for i in range(8):
        p = tmp_path / f"z{i}.npy"
        np.save(p, rng.normal(size=(8, 64)).astype("float32"))
        paths.append(p)
    records, rows = [], []
    for family in range(24):
        for prompt_index in range(16):
            pid = f"f{family:02}_p{prompt_index:02}"
            for i in range(8):
                base = i < 4
                sid = f"{pid}_b{i}" if base else f"{pid}_d{(i - 4) // 2}_{i % 2}"
                anchor = sid if base else f"{pid}_b1"
                q = list(contract.target_center)
                amount = [0, 1, 2, 3, 0.9, 1.1, 1.2, 0.8][i]
                if amount:
                    q[2] = contract.target_upper[2] + amount
                band = amount**2 / 3
                record = Pitch3LTEExample(
                    sample_id=sid,
                    prompt_id=pid,
                    prompt_family=f"f{family:02}",
                    trajectory_id=anchor,
                    split="train" if family < 20 else "development",
                    source_kind="base_step4_seed" if base else "local_finite_difference",
                    latent_path=paths[i],
                    latent_sha256=file_hash(paths[i]),
                    prompt_embedding_path=embedding,
                    prompt_embedding_sha256=file_hash(embedding),
                    exact_band=band,
                    energy_target=math.log1p(band),
                    coordinates=tuple(q),
                    direction_id="" if base else f"{pid}_d{(i - 4) // 2}",
                    direction_sign=0 if base else (-1 if i % 2 == 0 else 1),
                    epsilon=0 if base else 0.05,
                    timestep=0.0,
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                )
                records.append(record)
                rows.append(
                    {
                        k: str(getattr(record, k))
                        for k in [
                            "sample_id",
                            "trajectory_id",
                            "prompt_id",
                            "prompt_family",
                            "split",
                            "source_kind",
                            "direction_id",
                            "direction_sign",
                            "epsilon",
                            "exact_band",
                            "energy_target",
                        ]
                    }
                )
    dataset = tmp_path / "dataset.csv"
    _write_csv(dataset, rows)
    plan_hash = _write_json(tmp_path / "pitch3_lte_dataset_plan.json", {"synthetic": True})
    _write_json(
        tmp_path / "pitch3_lte_dataset_summary.json",
        {
            "local_preflight_passed": True,
            "dataset_manifest_sha256": file_hash(dataset),
            "dataset_plan_sha256": plan_hash,
        },
    )
    monkeypatch.setattr(module, "_read_examples", lambda *args: records)
    exported = []
    original_export = module.export_predictions

    def checked_export(model, selected, *args):
        assert all(r.split == "train" for r in selected)
        exported.append({r.sample_id for r in selected})
        return original_export(model, selected, *args)

    monkeypatch.setattr(module, "export_predictions", checked_export)
    common = dict(
        fingerprint_path=fingerprint,
        dataset_manifest=dataset,
        config_path=config,
        seed=41,
        cv_fold=0,
        device_name="cpu",
        global_variant="g1",
    )
    global_dir = tmp_path / "g1"
    global_result = module.train_v38b(**common, output_dir=global_dir)
    parent = global_dir / "pitch3_lte_manifest.json"
    verify_manifest(parent, rows)
    assert len(exported) == 2 and not exported[0] & exported[1]
    assert global_result["epochs_completed"] == 1
    assert all(not h["selection_evaluated"] for h in global_result["training_history"])
    base_model, _ = load_pitch3_lte_checkpoint(
        Path(global_result["checkpoint"]), device=torch.device("cpu")
    )
    for variant in ["l1", "l2"]:
        output = tmp_path / variant
        result = module.train_v38b(
            **common, output_dir=output, local_variant=variant, global_manifest=parent
        )
        verify_manifest(output / "pitch3_lte_manifest.json", rows)
        assert result["source_global_checkpoint_sha256"] == global_result["checkpoint_sha256"]
        model, _ = load_pitch3_lte_checkpoint(
            Path(result["checkpoint"]), device=torch.device("cpu")
        )
        for name, value in base_model.state_dict().items():
            if not name.startswith("local_energy_head."):
                assert torch.equal(model.state_dict()[name], value)
    assert len(exported) == 6
    with pytest.raises(FileExistsError):
        module.train_v38b(**common, output_dir=global_dir)
