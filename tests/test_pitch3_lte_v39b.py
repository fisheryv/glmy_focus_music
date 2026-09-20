from __future__ import annotations

import copy
import csv
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generation.pitch3_contract import load_pitch3_contract
from generation.pitch3_lte_v39a_protocol import content_hash, file_hash
from generation.pitch3_lte_v39b_protocol import (
    coordinates_from_counts,
    joint_counts,
    load_teacher,
    split_rows,
    validate_config,
    validate_protocol,
    variants_for,
)
from scripts.run_pitch3_lte_v39b import build_jobs

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "metadata/focus_pitch3_fingerprint_v1.json"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_fixture():
    rows = []
    for family in range(24):
        for i in range(4):
            sid = f"f{family:02}_b{i}"
            rows.append(
                {
                    "sample_id": sid,
                    "trajectory_id": sid,
                    "prompt_id": f"p{family}",
                    "prompt_family": f"f{family:02}",
                    "split": "train" if family < 20 else "development",
                    "source_kind": "base_step4_seed",
                    "latent_sha256": "a" * 64,
                }
            )
    return rows


def test_budget_pairs_are_distinct_and_never_include_outer_labels():
    rows = rows_fixture()
    combinations = []
    families = set()
    for fold in range(5):
        fit, outer = split_rows(rows, fold, "budget")
        _, held = split_rows(rows, fold, "cv")
        group = {r["prompt_family"] for r in fit}
        assert not group & {r["prompt_family"] for r in held}
        assert len(group) == 2 and not outer
        assert all(r["split"] == "train" for r in fit)
        combinations.append(tuple(sorted(group)))
        families.update(group)
    assert len(set(combinations)) == 5 and len(families) == 10
    assert variants_for("budget") == ("baseline", "transition")
    assert variants_for("cv") == ("transition", "distill")


def test_budget_protocol_rejects_teacher_and_outer_leakage():
    rows = rows_fixture()
    fit, _ = split_rows(rows, 0, "budget")
    p = {
        "mode": "budget",
        "variant": "baseline",
        "cv_fold": 0,
        "train_sample_ids": sorted(r["sample_id"] for r in fit),
        "selection_sample_ids": [],
        "train_families": sorted({r["prompt_family"] for r in fit}),
        "selection_families": [],
        "teacher_fit_sample_ids": [],
        "teacher_sha256": None,
        "epochs": 48,
        "experiment": {"budget_epochs": 48},
        "development_used": False,
        "checkpoint_selection": "fixed_epoch_online_no_selection_v39b",
        "outer_evaluation_count": 0,
    }
    validate_protocol(p, rows)
    for key, value in (
        ("selection_sample_ids", [rows[0]["sample_id"]]),
        ("teacher_fit_sample_ids", [rows[0]["sample_id"]]),
        ("development_used", True),
        ("epochs", 24),
    ):
        altered = copy.deepcopy(p)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_protocol(altered, rows)


def test_config_preserves_g0_and_freezes_new_budgets():
    cfg = tomllib.loads((ROOT / "configs/pitch3_lte_v39b.toml").read_text())
    old = tomllib.loads((ROOT / "configs/pitch3_lte_v39a.toml").read_text())
    assert {k: cfg[k] for k in old} == old
    assert validate_config(cfg)["budget_epochs"] == 48
    for section, key, value in (
        ("training", "gradient_clip_norm", 10),
        ("v39b", "cv_epochs", 48),
        ("v39b", "transition_kl_weight", 1),
    ):
        changed = copy.deepcopy(cfg)
        changed[section][key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_joint_counts_include_self_and_do_not_bridge_invalid_gaps():
    c = joint_counts(np.array([0, 0, 1, -1, 2, 2, 0]))
    assert c.sum() == 4
    assert c[0, 0] == c[0, 1] == c[2, 2] == c[2, 0] == 1
    assert c[1, 2] == 0
    p, q = coordinates_from_counts(c, load_pitch3_contract(FINGERPRINT))
    assert p.sum() == 1 and np.trace(p) == 0.5 and np.square(p).sum() == 0.25
    assert q.shape == (2,)


@pytest.mark.parametrize("states", [[0.0, 1.0], [0, 16], [0, -2], [0, -1, 1], []])
def test_invalid_state_sequences_fail_closed(states):
    with pytest.raises(ValueError):
        joint_counts(np.asarray(states))


def teacher_fixture(tmp_path):
    rows = rows_fixture()
    contract = load_pitch3_contract(FINGERPRINT)
    c = joint_counts(np.array([0, 0, 1, 1, 0]))
    _, q = coordinates_from_counts(c, contract)
    for row in rows:
        row["coordinates_json"] = json.dumps([0, *q])
    dataset = tmp_path / "dataset.csv"
    write_csv(dataset, rows)
    dataset_plan = tmp_path / "pitch3_lte_dataset_plan.json"
    write_json(dataset_plan, {"synthetic": True})
    train = [r for r in rows if r["split"] == "train"]
    target = tmp_path / "targets.npz"
    np.savez(
        target,
        sample_ids=np.array([r["sample_id"] for r in train]),
        counts=np.stack([c] * len(train)),
    )
    manifest = tmp_path / "teacher_manifest.json"
    plan = {
        "stage": "pitch3_lte_transition_teacher_v39b",
        "dataset_manifest_sha256": file_hash(dataset),
        "dataset_plan_sha256": file_hash(dataset_plan),
        "fingerprint_json_sha256": file_hash(FINGERPRINT),
        "target_semantics": "single_frozen_window_global_joint_counts_including_self",
        "sample_ids": [r["sample_id"] for r in train],
        "development_used": False,
        "duration_seconds": 180.0,
        "reconstruction_atol": 1e-7,
    }
    write_json(tmp_path / "teacher_plan.json", plan)
    value = {
        **plan,
        "completed": True,
        "production_authorization": False,
        "targets_file": target.name,
        "targets_sha256": file_hash(target),
        "samples": [
            {
                **{k: r[k] for k in ("sample_id", "latent_sha256")},
                "plan_sha256": content_hash(plan),
            }
            for r in train
        ],
    }
    write_json(manifest, value)
    return dataset, manifest, target, rows, value


def test_teacher_returns_only_fit_targets_and_checks_labels(tmp_path):
    dataset, manifest, _, rows, value = teacher_fixture(tmp_path)
    fit, _ = split_rows(rows, 0, "budget")
    ids = [r["sample_id"] for r in fit]
    targets, _ = load_teacher(manifest, dataset, FINGERPRINT, ids)
    assert set(targets) == set(ids)
    with pytest.raises(ValueError, match="non-fit"):
        load_teacher(manifest, dataset, FINGERPRINT, [rows[-1]["sample_id"]])
    rows[0]["coordinates_json"] = "[0,0,0]"
    write_csv(dataset, rows)
    value["dataset_manifest_sha256"] = file_hash(dataset)
    plan_path = tmp_path / "teacher_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["dataset_manifest_sha256"] = file_hash(dataset)
    write_json(plan_path, plan)
    for receipt in value["samples"]:
        receipt["plan_sha256"] = content_hash(plan)
    write_json(manifest, value)
    with pytest.raises(ValueError, match="reconstruction"):
        load_teacher(manifest, dataset, FINGERPRINT)


def test_teacher_rejects_partial_pack_and_row_normalized_substitution(tmp_path):
    dataset, manifest, target, _, value = teacher_fixture(tmp_path)
    with np.load(target) as z:
        ids, counts = z["sample_ids"], z["counts"]
    for bad_ids, bad_counts in ((ids[:-1], counts[:-1]), (ids, counts.astype(float))):
        np.savez(target, sample_ids=bad_ids, counts=bad_counts)
        value["targets_sha256"] = file_hash(target)
        write_json(manifest, value)
        with pytest.raises(ValueError):
            load_teacher(manifest, dataset, FINGERPRINT)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("development_used", True),
        ("production_authorization", True),
        ("duration_seconds", 300.0),
        ("reconstruction_atol", 0.1),
    ],
)
def test_teacher_rejects_changed_scope_and_tolerance(tmp_path, key, bad):
    dataset, manifest, _, _, value = teacher_fixture(tmp_path)
    value[key] = bad
    write_json(manifest, value)
    with pytest.raises(ValueError, match="scope"):
        load_teacher(manifest, dataset, FINGERPRINT)


def test_teacher_rejects_detached_receipt_and_resume_plan(tmp_path):
    dataset, manifest, _, _, value = teacher_fixture(tmp_path)
    original = copy.deepcopy(value)
    value["samples"][0]["plan_sha256"] = "b" * 64
    write_json(manifest, value)
    with pytest.raises(ValueError, match="receipt"):
        load_teacher(manifest, dataset, FINGERPRINT)
    original["sample_ids"] = list(reversed(original["sample_ids"]))
    write_json(manifest, original)
    with pytest.raises(ValueError, match="resume plan"):
        load_teacher(manifest, dataset, FINGERPRINT)


def args(tmp_path, stage="budget", **kw):
    values = dict(
        root=ROOT,
        run_root=tmp_path,
        config=ROOT / "configs/pitch3_lte_v39b.toml",
        dataset_manifest=tmp_path / "data.csv",
        fingerprint=FINGERPRINT,
        teacher_manifest=tmp_path / "teacher/teacher_manifest.json",
        trajectory_manifest=None,
        ace_config=None,
        workers=4,
        python="python",
        devices=["cuda:1", "cuda:2", "cuda:3"],
        folds=list(range(5)),
        seeds=[20260941],
        variants=None,
        stage=stage,
    )
    return SimpleNamespace(**(values | kw))


@pytest.mark.parametrize("stage", ["budget", "distill-diagnose", "cv"])
def test_runner_keeps_matched_pairs_together_without_auto_promotion(tmp_path, stage):
    jobs = build_jobs(args(tmp_path, stage))
    assert len(jobs) == 10
    for i in range(0, 10, 2):
        assert jobs[i]["device"] == jobs[i + 1]["device"]
        assert jobs[i]["require_empty"]
    commands = [j["command"] for j in jobs]
    assert all(("--teacher-manifest" in c) == (stage != "budget") for c in commands)
    assert all("screen-development" not in c for c in commands)
    with pytest.raises(ValueError):
        build_jobs(args(tmp_path, stage, variants=["distill" if stage == "budget" else "baseline"]))


def test_teacher_stage_is_resumable_and_single_gpu(tmp_path):
    jobs = build_jobs(args(tmp_path, "teacher"))
    assert len(jobs) == 1 and not jobs[0]["require_empty"]
    assert "--ace-config" not in jobs[0]["command"]


def test_new_auxiliary_gradient_and_optimizer_on_real_tensors():
    torch = pytest.importorskip("torch")
    from generation.pitch3_lte_v39b_checks import run_checks

    torch.set_num_threads(1)
    result = run_checks(ROOT / "configs/pitch3_lte_v39b.toml", "cpu")
    assert result["all_checks_passed"] and result["synthetic_only"]


def test_real_cpu_trainer_matched_budget_distillation_and_outer_once(tmp_path, monkeypatch):
    """Synthetic workflow regression; never evidence for real-data accuracy."""
    torch = pytest.importorskip("torch")
    from generation import pitch3_lte_v39b as module
    from generation.pitch3_lte_training import Pitch3LTEExample
    from generation.pitch3_lte_v39b_protocol import verify_run
    from scripts.summarize_pitch3_lte_v39b import summarize

    torch.set_num_threads(1)
    config = ROOT / "configs/pitch3_lte_v39b.toml"
    contract = load_pitch3_contract(FINGERPRINT)
    experiment = validate_config(tomllib.loads(config.read_text()))
    experiment.update(budget_epochs=1, diagnostic_epochs=1, cv_epochs=1)
    monkeypatch.setattr(module, "validate_config", lambda _payload: experiment)
    rng = np.random.default_rng(93)
    embedding = tmp_path / "text.npz"
    np.savez(embedding, hidden=rng.normal(size=(2, 1024)).astype("float32"), mask=np.ones(2, bool))
    latents = []
    for i in range(4):
        p = tmp_path / f"z{i}.npy"
        np.save(p, rng.normal(size=(9, 64)).astype("float32"))
        latents.append(p)
    targets, records, rows = {}, [], []
    for fi in range(24):
        for pi in range(16):
            pid = f"f{fi:02}_p{pi}"
            for i in range(4):
                sid = f"{pid}_b{i}"
                counts = np.zeros((16, 16), dtype=np.int64)
                counts[i, i] = i + 1
                counts[i, (i + 1) % 16] = 4 - i
                p, q = coordinates_from_counts(counts, contract)
                coords = (contract.target_center[0], *q)
                band = sum(
                    w * (max(lo - x, 0) ** 2 + max(x - hi, 0) ** 2)
                    for x, lo, hi, w in zip(
                        coords,
                        contract.target_lower,
                        contract.target_upper,
                        contract.distance_weights,
                        strict=True,
                    )
                )
                record = Pitch3LTEExample(
                    sample_id=sid,
                    prompt_id=pid,
                    prompt_family=f"f{fi:02}",
                    trajectory_id=sid,
                    split="train" if fi < 20 else "development",
                    source_kind="base_step4_seed",
                    latent_path=latents[i],
                    latent_sha256=file_hash(latents[i]),
                    prompt_embedding_path=embedding,
                    prompt_embedding_sha256=file_hash(embedding),
                    exact_band=band,
                    energy_target=float(np.log1p(band)),
                    coordinates=coords,
                    direction_id="",
                    direction_sign=0,
                    epsilon=0.0,
                    timestep=0.8333333333,
                    ace_model_sha256="a" * 64,
                    vae_sha256="b" * 64,
                )
                records.append(record)
                row = {
                    name: getattr(record, name)
                    for name in (
                        "sample_id",
                        "prompt_id",
                        "prompt_family",
                        "trajectory_id",
                        "split",
                        "source_kind",
                        "exact_band",
                        "energy_target",
                        "direction_id",
                        "direction_sign",
                        "epsilon",
                    )
                }
                row["coordinates_json"] = json.dumps(coords)
                rows.append(row)
                targets[sid] = p
    dataset = tmp_path / "dataset.csv"
    write_csv(dataset, rows)
    monkeypatch.setattr(module, "prepare_data", lambda *_a: (contract, {}, records, rows, [], []))
    teacher = tmp_path / "teacher.json"
    teacher.write_text("{}")
    requested_teacher_ids = []

    def fit_teacher(_path, _data, _fp, ids):
        requested_teacher_ids.append(set(ids))
        return {sid: targets[sid] for sid in ids}, {"completed": True}

    monkeypatch.setattr(module, "load_teacher", fit_teacher)
    real_export = module.export_predictions
    exports = []

    def export(model, selected, *args):
        assert all(r.split == "train" for r in selected)
        exports.append((args[-1], {r.sample_id for r in selected}))
        return real_export(model, selected, *args)

    monkeypatch.setattr(module, "export_predictions", export)
    common = dict(
        fingerprint_path=FINGERPRINT,
        dataset_manifest=dataset,
        config_path=config,
        cv_fold=0,
        seed=41,
        device_name="cpu",
    )
    for mode in ("budget", "distill-diagnose", "cv"):
        completions = []
        for variant in variants_for(mode):
            folder = tmp_path / "runs" / mode / variant / "fold_0/seed_41"
            result = module.train_v39b(
                **common,
                output_dir=folder,
                mode=mode,
                variant=variant,
                teacher_manifest=teacher if mode != "budget" else None,
            )
            verify_run(folder / "pitch3_lte_v39b_complete.json", rows)
            assert bool(list(folder.glob("*.pt"))) == (mode == "cv")
            assert result["outer_evaluation_count"] == int(mode == "cv")
            completions.append(result)
            with pytest.raises(FileExistsError):
                module.train_v39b(**common, output_dir=folder, mode=mode, variant=variant)
        assert (
            completions[0]["shared_initial_state_sha256"]
            == completions[1]["shared_initial_state_sha256"]
        )
        assert (
            completions[0]["initial_normalizers_sha256"]
            == completions[1]["initial_normalizers_sha256"]
        )
        report = summarize(tmp_path / "runs", dataset, mode, [0], [41])
        assert report["matched_initialization_and_normalizers"]
        assert not report["checkpoint_selection_performed"]
    assert sum(name == "selection" for name, _ in exports) == 2
    allowed = {r["sample_id"] for r in split_rows(rows, 0, "cv")[0]}
    assert all(ids <= allowed for ids in requested_teacher_ids)
