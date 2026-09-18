from __future__ import annotations

import copy
import csv
import json
import math
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from generation.pitch3_lte_metrics import pitch3_lte_metrics
from generation.pitch3_lte_v39a_protocol import (
    REVISION,
    SELECTION,
    content_hash,
    file_hash,
    grouped_split,
    validate_config,
    validate_protocol,
    verify_manifest,
)
from scripts.run_pitch3_lte_v39a import build_jobs
from scripts.summarize_pitch3_lte_v39a import summarize

ROOT = Path(__file__).resolve().parents[1]


def dataset_rows():
    rows = []
    for family in range(24):
        for sample in range(4):
            sid = f"f{family:02}_b{sample}"
            rows.append(
                {
                    "sample_id": sid,
                    "trajectory_id": sid,
                    "prompt_id": f"p{family}",
                    "prompt_family": f"f{family:02}",
                    "split": "train" if family < 20 else "development",
                    "source_kind": "base_step4_seed",
                    "direction_id": "",
                    "direction_sign": "0.0",
                    "epsilon": "0",
                    "exact_band": str(sample),
                    "energy_target": str(math.log1p(sample)),
                }
            )
    return rows


def protocol(rows, fold=0, diagnostic=False, variant="baseline", seed=41):
    fit, outer = grouped_split(rows, fold, diagnostic)
    mode = "diagnostic" if diagnostic else "full" if fold is None else "cv"
    return {
        "run_mode": mode,
        "cv_fold": fold,
        "seed": seed,
        "validation_scope": {
            "diagnostic": "train_fit_diagnostic",
            "cv": "train_family_cv",
            "full": "development",
        }[mode],
        "train_sample_ids": sorted(r["sample_id"] for r in fit),
        "selection_sample_ids": sorted(r["sample_id"] for r in outer),
        "train_families": sorted({r["prompt_family"] for r in fit}),
        "selection_families": sorted({r["prompt_family"] for r in outer}),
        "checkpoint_selection": SELECTION,
        "development_used": False,
        "outer_evaluation_count": int(mode == "cv"),
        "experiment_contract": {"variant": variant, "global_epochs": 12, "local_variant": "l0"},
    }


def test_config_is_the_matched_g0_recipe_without_ordinal_or_local_changes():
    new = tomllib.loads((ROOT / "configs/pitch3_lte_v39a.toml").read_text())
    old = tomllib.loads((ROOT / "configs/pitch3_lte_v38b.toml").read_text())
    validate_config(new)
    assert new["training"] == old["training"]
    assert {
        k: v for k, v in new["model"].items() if k not in {"ordinal_auxiliary", "transition_branch"}
    } == {k: v for k, v in old["model"].items() if k != "ordinal_auxiliary"}
    for section, key, value in (
        ("model", "ordinal_auxiliary", True),
        ("training", "global_value_base_only", False),
        ("v39a", "parameter_source", "best_outer"),
    ):
        changed = copy.deepcopy(new)
        changed[section][key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_protocol_keeps_fit_probes_and_all_outer_labels_separate():
    rows = dataset_rows()
    for fold in range(5):
        p = protocol(rows, fold, diagnostic=True)
        validate_protocol(p, rows)
        assert len(p["train_families"]) == 2 and p["selection_sample_ids"] == []
        held = protocol(rows, fold)["selection_families"]
        assert not set(held) & set(p["train_families"])
        p["train_sample_ids"].append(rows[-1]["sample_id"])
        with pytest.raises(ValueError, match="grouped"):
            validate_protocol(p, rows)
    full = protocol(rows, fold=None)
    validate_protocol(full, rows)
    assert len(full["train_families"]) == 20 and not full["selection_sample_ids"]
    full["development_used"] = True
    with pytest.raises(ValueError, match="forbidden"):
        validate_protocol(full, rows)
    with pytest.raises(ValueError, match="require a fold"):
        grouped_split(rows, None, True)
    rows[0].update(source_kind="local_finite_difference", trajectory_id=rows[-1]["sample_id"])
    with pytest.raises(ValueError, match="anchor"):
        grouped_split(rows, 1)


def args(tmp_path, **overrides):
    fields = dict(
        root=ROOT,
        run_root=tmp_path,
        config=ROOT / "configs/pitch3_lte_v39a.toml",
        fingerprint=tmp_path / "f.json",
        dataset_manifest=tmp_path / "data.csv",
        python="python",
        devices=["cuda:1", "cuda:2", "cuda:3"],
        folds=list(range(5)),
        seeds=None,
        variants=None,
        stage="cv-global",
    )
    return SimpleNamespace(**(fields | overrides))


def test_runner_pairs_share_a_device_and_never_auto_screen(tmp_path):
    jobs = build_jobs(args(tmp_path))
    assert len(jobs) == 10
    for i in range(0, 10, 2):
        left, right = jobs[i : i + 2]
        assert left["device"] == right["device"]
        for flag in ("--seed", "--cv-fold", "--config"):
            assert (
                left["command"][left["command"].index(flag) + 1]
                == right["command"][right["command"].index(flag) + 1]
            )
        assert "screen-development" not in left["command"]
    jobs = build_jobs(args(tmp_path, stage="diagnose", folds=[0]))
    assert len(jobs) == 2 and all("--diagnostic" in j["command"] for j in jobs)
    with pytest.raises(ValueError, match="Explicitly"):
        build_jobs(args(tmp_path, stage="train-global"))
    with pytest.raises(ValueError, match="three"):
        build_jobs(args(tmp_path, stage="train-global", variants=["transition"], seeds=[41]))
    assert len(build_jobs(args(tmp_path, stage="train-global", variants=["transition"]))) == 3


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_hash(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return file_hash(path)


def make_cv(tmp_path):
    rows = dataset_rows()
    dataset = tmp_path / "dataset.csv"
    dataset_hash = write_csv(dataset, rows)
    for variant in ("baseline", "transition"):
        for fold in range(5):
            folder = tmp_path / f"cv/{variant}/fold_{fold}/seed_41/models"
            folder.mkdir(parents=True)
            p = protocol(rows, fold, variant=variant)
            p["dataset_manifest_sha256"] = dataset_hash
            phash = write_json(folder / "pitch3_lte_run_protocol.json", p)
            normalizers = {"value": 1.0, "prompt_rank": 1.0}
            shash = write_json(
                folder / "pitch3_lte_training_statistics.json",
                {"run_protocol_sha256": phash, "loss_normalizers": normalizers},
            )
            chash = write_json(
                folder / "pitch3_lte_effective_config.json",
                {
                    "model": {
                        "transition_branch": variant == "transition",
                        "ordinal_auxiliary": False,
                    },
                    "v39a": p["experiment_contract"],
                },
            )
            ihash = write_json(
                folder / "pitch3_lte_initialization.json",
                {"shared_state_sha256": "shared", "outputs_and_losses_equal": True},
            )
            checkpoint = folder / "synthetic.pt"
            with zipfile.ZipFile(checkpoint, "w") as z:
                z.writestr("fixture", "protocol-only, not a real model")
            fields = {}
            for split in ("train", "selection"):
                predictions = [
                    {
                        **{
                            k: row[k]
                            for k in (
                                "sample_id",
                                "prompt_id",
                                "prompt_family",
                                "source_kind",
                                "direction_id",
                                "direction_sign",
                                "epsilon",
                                "exact_band",
                            )
                        },
                        "anchor_sample_id": row["trajectory_id"],
                        "exact_energy": row["energy_target"],
                        "predicted_energy": row["energy_target"],
                        "predicted_global_energy": row["energy_target"],
                        "predicted_local_energy": "0",
                        "predicted_coordinates_json": "[0,0,0]",
                    }
                    for row in rows
                    if row["sample_id"] in p[f"{split}_sample_ids"]
                ]
                for row in predictions:
                    row["direction_sign"] = "0"
                fields[f"{split}_predictions_sha256"] = write_csv(
                    folder / f"pitch3_lte_{split}_predictions.csv", predictions
                )
                if split == "selection":
                    fields["best_development_metrics"] = pitch3_lte_metrics(predictions)
            write_json(
                folder / "pitch3_lte_manifest.json",
                {
                    **p,
                    **fields,
                    "architecture_revision": REVISION,
                    "checkpoint": "synthetic.pt",
                    "checkpoint_sha256": file_hash(checkpoint),
                    "run_protocol_sha256": phash,
                    "training_statistics_sha256": shash,
                    "training_config_sha256": chash,
                    "diagnostic_artifacts_sha256": {"pitch3_lte_initialization.json": ihash},
                    "training_manifest_sha256": dataset_hash,
                    "source_training_config_sha256": "config",
                    "fingerprint_json_sha256": "fingerprint",
                    "implementation_sha256": {},
                    "local_residual_mode": "zero_global_only",
                    "shared_initial_state_sha256": "shared",
                    "initial_normalizers_sha256": content_hash(normalizers),
                    "epochs_completed": 12,
                    "production_authorization": False,
                    "qualification_eligible": False,
                },
            )
    return dataset


def test_summary_reproduces_gates_and_rejects_unmatched_initialization(tmp_path):
    dataset = make_cv(tmp_path)
    report = summarize(tmp_path, dataset, [41])
    assert report["paired_comparison"]["matched_initialization_and_normalizers"]
    assert report["variants"]["transition"]["families_passing_original_rho_gate"] == 20
    # No local pairs in this fixture: passing rho is not a full gate pass.
    assert report["variants"]["transition"]["folds_passing_all_original_model_gates"] == 0
    folder = tmp_path / "cv/transition/fold_0/seed_41/models"
    path = folder / "pitch3_lte_manifest.json"
    m = json.loads(path.read_text())
    m["shared_initial_state_sha256"] = "different_initialization"
    m["diagnostic_artifacts_sha256"]["pitch3_lte_initialization.json"] = write_json(
        folder / "pitch3_lte_initialization.json",
        {"shared_state_sha256": "different_initialization", "outputs_and_losses_equal": True},
    )
    write_json(path, m)
    with pytest.raises(ValueError, match="Unmatched"):
        summarize(tmp_path, dataset, [41])


def test_verifier_rejects_corrupt_checkpoint_even_after_recorded_hash_is_changed(tmp_path):
    make_cv(tmp_path)
    folder = tmp_path / "cv/baseline/fold_0/seed_41/models"
    path = folder / "pitch3_lte_manifest.json"
    checkpoint = folder / "synthetic.pt"
    checkpoint.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(ValueError, match="hash-mismatched"):
        verify_manifest(path)
    m = json.loads(path.read_text())
    m["checkpoint_sha256"] = file_hash(checkpoint)
    write_json(path, m)
    with pytest.raises(ValueError, match="Incomplete checkpoint ZIP"):
        verify_manifest(path)


def test_tensor_contracts_on_cpu():
    pytest.importorskip("torch")
    from generation.pitch3_lte_v39a_checks import run_checks

    result = run_checks(ROOT / "configs/pitch3_lte_v39a.toml", "cpu")
    assert result["all_checks_passed"] and result["synthetic_only"]


def test_fixed_epoch_trainer_roundtrip_and_fit_only_probe(tmp_path, monkeypatch):
    """Real CPU optimizer on synthetic tensors; validates workflow, not accuracy."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from generation import pitch3_lte_v39a as module
    from generation.pitch3_lte_training import Pitch3LTEExample, load_pitch3_lte_checkpoint
    from generation.pitch3_lte_v39a_protocol import verify_diagnostic

    torch.set_num_threads(1)
    text = (ROOT / "configs/pitch3_lte_v39a.toml").read_text()
    text = text.replace("global_epochs = 12", "global_epochs = 1")
    text = text.replace("diagnostic_epochs = 24", "diagnostic_epochs = 1")
    config = tmp_path / "config.toml"
    config.write_text(text, encoding="utf-8")
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
    monkeypatch.setattr(module, "load_pitch3_contract", lambda _: contract)
    rng = np.random.default_rng(71)
    embedding = tmp_path / "prompt.npz"
    np.savez(embedding, hidden=rng.normal(size=(2, 1024)).astype("float32"), mask=np.ones(2, bool))
    paths = []
    for i in range(8):
        path = tmp_path / f"latent_{i}.npy"
        np.save(path, rng.normal(size=(9, 64)).astype("float32"))
        paths.append(path)
    records, rows = [], []
    for family in range(24):
        for prompt in range(16):
            pid = f"f{family:02}_p{prompt:02}"
            for i in range(8):
                base = i < 4
                sid = f"{pid}_{i}"
                amount = [0, 1, 2, 3, 0.9, 1.1, 1.2, 0.8][i]
                coordinates = list(contract.target_center)
                if amount:
                    coordinates[2] = contract.target_upper[2] + amount
                band = amount**2 / 3
                record = Pitch3LTEExample(
                    sample_id=sid,
                    prompt_id=pid,
                    prompt_family=f"f{family:02}",
                    trajectory_id=sid if base else f"{pid}_1",
                    split="train" if family < 20 else "development",
                    source_kind="base_step4_seed" if base else "local_finite_difference",
                    latent_path=paths[i],
                    latent_sha256=file_hash(paths[i]),
                    prompt_embedding_path=embedding,
                    prompt_embedding_sha256=file_hash(embedding),
                    exact_band=band,
                    energy_target=math.log1p(band),
                    coordinates=tuple(coordinates),
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
                        for k in (
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
                        )
                    }
                )
    dataset = tmp_path / "dataset.csv"
    write_csv(dataset, rows)
    plan_hash = write_json(tmp_path / "pitch3_lte_dataset_plan.json", {"synthetic": True})
    write_json(
        tmp_path / "pitch3_lte_dataset_summary.json",
        {
            "local_preflight_passed": True,
            "dataset_manifest_sha256": file_hash(dataset),
            "dataset_plan_sha256": plan_hash,
        },
    )
    monkeypatch.setattr(module, "_read_examples", lambda *_args: records)
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
    )
    results = []
    for variant in ("baseline", "transition"):
        output = tmp_path / variant
        result = module.train_v39a(**common, variant=variant, output_dir=output)
        results.append(result)
        verify_manifest(output / "pitch3_lte_manifest.json", rows)
        restored, metadata = load_pitch3_lte_checkpoint(
            Path(result["checkpoint"]),
            device=torch.device("cpu"),
            expected_sha256=result["checkpoint_sha256"],
        )
        assert restored.config.transition_branch == (variant == "transition")
        assert metadata["epochs_completed"] == 1 and metadata["outer_evaluation_count"] == 1
        history = json.loads((output / "pitch3_lte_training_history.json").read_text())["history"]
        assert len(history) == 1 and not history[0]["selection_evaluated"]
        if variant == "transition":
            assert (
                history[0]["block_parameter_updates"]["transition_features"]["epoch_net_update_l2"]
                > 0
            )
        with pytest.raises(FileExistsError):
            module.train_v39a(**common, variant=variant, output_dir=output)
    assert results[0]["shared_initial_state_sha256"] == results[1]["shared_initial_state_sha256"]
    assert results[0]["initial_normalizers_sha256"] == results[1]["initial_normalizers_sha256"]
    assert len(exported) == 4 and not exported[0] & exported[1]
    probe = tmp_path / "probe"
    module.train_v39a(**common, variant="transition", output_dir=probe, diagnostic=True)
    report = verify_diagnostic(probe / "pitch3_lte_diagnostic_complete.json")
    assert len(report["fit_families"]) == 2 and not report["outer_evaluation_count"]
    assert not list(probe.glob("*.pt")) and len(exported) == 5
