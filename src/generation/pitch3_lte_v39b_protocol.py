"""Torch-free contracts for budget probes and transition-teacher experiments."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from .pitch3_contract import load_pitch3_contract
from .pitch3_lte_v38b_protocol import local_file, read_csv, read_json, verify_file
from .pitch3_lte_v39a_protocol import content_hash, grouped_split
from .pitch3_lte_v39a_protocol import validate_config as validate_a

REVISION = "v3.9b_transition_teacher"
SELECTION = "fixed_epoch_online_no_selection_v39b"
MODES = ("budget", "distill-diagnose", "cv")
# Indices in the original sorted 20 train families. Ten distinct families,
# none held out in the fold in which it is probed. Predeclared after the A audit;
# no run-time selection using metrics.
PROBE_INDICES = ((1, 7), (0, 14), (8, 13), (4, 12), (2, 15))


def validate_config(payload):
    validate_a({k: payload[k] for k in ("model", "training", "v39a")})
    expected = {
        "budget_epochs": 48,
        "diagnostic_epochs": 48,
        "cv_epochs": 12,
        "states": 16,
        "transition_kl_weight": 0.25,
        "transition_coordinate_weight": 0.25,
        "reconstruction_atol": 1e-7,
    }
    if payload.get("v39b") != expected:
        raise ValueError("V3.9-B settings differ from the predeclared experiment")
    return dict(expected)


def split_rows(rows, fold, mode):
    if mode not in MODES or type(fold) is not int or fold not in range(5):
        raise ValueError("A valid mode and outer fold are required")
    fit, outer = grouped_split(rows, fold)
    if mode != "cv":
        families = sorted({r["prompt_family"] for r in rows if r["split"] == "train"})
        chosen = {families[i] for i in PROBE_INDICES[fold]}
        if not chosen <= {r["prompt_family"] for r in fit}:
            raise ValueError("Probe families overlap the outer fold")
        fit = [r for r in fit if r["prompt_family"] in chosen]
        outer = []
    return fit, outer


def variants_for(mode):
    if mode not in MODES:
        raise ValueError("Unknown experiment mode")
    return ("baseline", "transition") if mode == "budget" else ("transition", "distill")


def validate_protocol(p, rows):
    fit, outer = split_rows(rows, p["cv_fold"], p["mode"])
    for name, selected in (("train", fit), ("selection", outer)):
        if p[f"{name}_sample_ids"] != sorted(r["sample_id"] for r in selected):
            raise ValueError(f"Incorrect {name} IDs")
        if p[f"{name}_families"] != sorted({r["prompt_family"] for r in selected}):
            raise ValueError(f"Incorrect {name} families")
    if p["variant"] not in variants_for(p["mode"]):
        raise ValueError("Variant is not part of this matched experiment")
    expected_teacher_ids = (
        sorted(r["sample_id"] for r in fit if r["source_kind"] == "base_step4_seed")
        if p["mode"] != "budget"
        else []
    )
    if p["teacher_fit_sample_ids"] != expected_teacher_ids or bool(p["teacher_sha256"]) != (
        p["mode"] != "budget"
    ):
        raise ValueError("Teacher supervision must be restricted to fit base IDs")
    epoch_key = {
        "budget": "budget_epochs",
        "distill-diagnose": "diagnostic_epochs",
        "cv": "cv_epochs",
    }[p["mode"]]
    if p["epochs"] != p["experiment"][epoch_key]:
        raise ValueError("Budget differs from declared experiment")
    if (
        p["development_used"]
        or p["checkpoint_selection"] != SELECTION
        or p["outer_evaluation_count"] != int(p["mode"] == "cv")
    ):
        raise ValueError("Selection or development leakage")


def joint_counts(states, state_count=16):
    """Count adjacent valid states; invalid gaps never create bridging edges."""
    s = np.asarray(states)
    if s.ndim != 1 or s.dtype.kind not in "iu" or np.any((s < -1) | (s >= state_count)):
        raise ValueError("Expected integer state sequence in [-1,K)")
    valid = (s[:-1] >= 0) & (s[1:] >= 0)
    counts = np.bincount(
        s[:-1][valid] * state_count + s[1:][valid], minlength=state_count**2
    ).reshape(state_count, state_count)
    if not counts.sum():
        raise ValueError("Teacher has no valid adjacent transitions")
    return counts.astype(np.int64)


def coordinates_from_counts(counts, contract):
    c = np.asarray(counts)
    if c.shape != (16, 16) or c.dtype.kind not in "iu" or (c < 0).any() or not c.sum():
        raise ValueError("Expected nonnegative nonempty 16x16 integer joint counts")
    p = c.astype(np.float64) / c.sum()
    raw = np.array([np.trace(p), np.square(p).sum()])
    q = (raw - np.array(contract.transform_center[1:])) / np.array(contract.transform_scale[1:])
    return p, q


def load_teacher(manifest_path, dataset_path, fingerprint_path, fit_ids=None):
    """Verify the train-only archive, then expose only requested fit targets."""
    manifest_path, dataset_path = Path(manifest_path), Path(dataset_path)
    m = read_json(manifest_path)
    if m.get("stage") != "pitch3_lte_transition_teacher_v39b" or not m.get("completed"):
        raise ValueError("Incomplete transition teacher")
    if (
        m.get("development_used") is not False
        or m.get("production_authorization") is not False
        or m.get("duration_seconds") != 180.0
        or m.get("reconstruction_atol") != 1e-7
    ):
        raise ValueError("Teacher scope or frozen reconstruction contract changed")
    plan = read_json(manifest_path.parent / "teacher_plan.json")
    if any(m.get(k) != v for k, v in plan.items()):
        raise ValueError("Teacher manifest differs from its resume plan")
    plan_digest = content_hash(plan)
    verify_file(dataset_path, m["dataset_manifest_sha256"])
    verify_file(dataset_path.parent / "pitch3_lte_dataset_plan.json", m["dataset_plan_sha256"])
    contract = load_pitch3_contract(fingerprint_path, expected_sha256=m["fingerprint_json_sha256"])
    target_file = local_file(manifest_path.parent, m["targets_file"])
    verify_file(target_file, m["targets_sha256"])
    rows = read_csv(dataset_path)
    grouped_split(rows, 0)
    truth = {
        r["sample_id"]: r
        for r in rows
        if r["split"] == "train" and r["source_kind"] == "base_step4_seed"
    }
    if m["target_semantics"] != "single_frozen_window_global_joint_counts_including_self":
        raise ValueError("Unsupported teacher aggregation semantics")
    with np.load(target_file, allow_pickle=False) as z:
        ids, counts = z["sample_ids"].tolist(), z["counts"]
    if (
        len(set(ids)) != len(ids)
        or set(ids) != set(truth)
        or len(counts) != len(ids)
        or ids != m["sample_ids"]
    ):
        raise ValueError("Teacher must cover exactly the original train base samples")
    wanted = set(ids) if fit_ids is None else set(fit_ids)
    if not wanted <= set(ids):
        raise ValueError("Requested non-fit or missing teacher IDs")
    receipts = {r["sample_id"]: r for r in m["samples"]}
    if len(receipts) != len(m["samples"]) or set(receipts) != set(ids):
        raise ValueError("Teacher receipts differ from target IDs")
    result = {}
    for sid, c in zip(ids, counts, strict=True):
        if receipts[sid]["plan_sha256"] != plan_digest:
            raise ValueError("Teacher receipt detached from its resume plan")
        if receipts[sid]["latent_sha256"] != truth[sid]["latent_sha256"]:
            raise ValueError("Teacher detached from input latent")
        p, q = coordinates_from_counts(c, contract)
        if not np.allclose(q, json.loads(truth[sid]["coordinates_json"])[1:], rtol=0, atol=1e-7):
            raise ValueError(f"Transition teacher fails exact q2/q3 reconstruction: {sid}")
        if sid in wanted:
            result[sid] = p
    return result, m


def verify_run(path, dataset_rows=None):
    m = read_json(path)
    if m.get("architecture_revision") != REVISION or not m.get("completed"):
        raise ValueError("Not a completed V3.9-B experiment")
    if m["production_authorization"] or m["guidance_promotion_eligible"]:
        raise ValueError("Internal experiments cannot authorize guidance")
    for name, digest in m["artifacts_sha256"].items():
        verify_file(local_file(path.parent, name), digest)
    p = read_json(path.parent / "pitch3_lte_run_protocol.json")
    if content_hash(p) != m["protocol_content_sha256"]:
        raise ValueError("Protocol detached from completion")
    if dataset_rows is not None:
        validate_protocol(p, dataset_rows)
    for key in ("mode", "variant", "cv_fold", "seed", "epochs", "teacher_sha256"):
        if m[key] != p[key]:
            raise ValueError(f"Completion differs from protocol: {key}")
    history = read_json(path.parent / "pitch3_lte_training_history.json")["history"]
    if len(history) != p["epochs"] or m["epochs_completed"] != p["epochs"]:
        raise ValueError("Incomplete fixed budget")
    if any(h["epoch"] != i or h["selection_evaluated"] for i, h in enumerate(history, 1)):
        raise ValueError("Invalid history or outer checkpoint selection")
    init = read_json(path.parent / "pitch3_lte_initialization.json")
    stats = read_json(path.parent / "pitch3_lte_training_statistics.json")
    if (
        not init["outputs_and_losses_equal"]
        or m["shared_initial_state_sha256"] != init["shared_state_sha256"]
        or m["initial_normalizers_sha256"] != content_hash(stats["loss_normalizers"])
    ):
        raise ValueError("Initialization or normalizers detached")
    for split in ("train", "selection"):
        ids = p[f"{split}_sample_ids"]
        pred = path.parent / f"pitch3_lte_{split}_predictions.csv"
        if not ids:
            if pred.exists():
                raise ValueError("Fit-only experiment exported outer predictions")
            continue
        values = read_csv(pred)
        if sorted(r["sample_id"] for r in values) != ids:
            raise ValueError("Prediction IDs differ from protocol")
        if any(
            float(r["predicted_local_energy"]) != 0
            or float(r["predicted_energy"]) != float(r["predicted_global_energy"])
            for r in values
        ):
            raise ValueError("Local residual must remain exactly zero")
    checkpoints = list(path.parent.glob("*.pt"))
    if p["mode"] != "cv" and checkpoints:
        raise ValueError("Diagnostic cannot publish a checkpoint")
    if p["mode"] == "cv":
        if len(checkpoints) != 1 or checkpoints[0].name not in m["artifacts_sha256"]:
            raise ValueError("Missing or unsigned CV checkpoint")
        if not zipfile.is_zipfile(checkpoints[0]):
            raise ValueError("Incomplete CV checkpoint ZIP")
    return m, p
