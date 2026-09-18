"""Torch-free split, configuration and artifact contracts for V3.9-A."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .pitch3_lte_v38b_protocol import file_hash as file_hash
from .pitch3_lte_v38b_protocol import local_file, read_csv, read_json, verify_file

REVISION = "v3.9a_high_resolution_transition"
SELECTION = "fixed_epoch_online_no_selection_v39a"
VARIANTS = ("baseline", "transition")


def content_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_config(payload: dict) -> dict:
    """Keep the matched G0 recipe; budgets are explicit versioned config values."""
    model, training, experiment = (payload[k] for k in ("model", "training", "v39a"))
    if set(experiment) != {
        "global_epochs",
        "diagnostic_epochs",
        "diagnostic_families",
        "parameter_source",
    }:
        raise ValueError("Unknown or missing V3.9-A experiment setting")
    for key in ("global_epochs", "diagnostic_epochs"):
        if type(experiment[key]) is not int or experiment[key] <= 0:
            raise ValueError("V3.9-A budgets must be positive integers")
    if experiment["diagnostic_families"] != 2 or experiment["parameter_source"] != "online":
        raise ValueError("V3.9-A requires two fit-only diagnostic families and online parameters")
    required_model = {
        "ordinal_auxiliary": False,
        "transition_branch": False,  # The runner selects the sole architecture difference.
        "potential_mode": "direct_anchored_v37",
        "latent_stem_mode": "dual_rms_v32",
        "latent_dim": 64,
        "text_dim": 1024,
        "model_dim": 128,
        "transformer_heads": 4,
        "transformer_layers": 3,
        "feedforward_dim": 512,
        "temporal_stride": 4,
        "prompt_residual_scale": 0.1,
        "dropout": 0.1,
    }
    required_training = {
        "use_bf16": False,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "training_schedule": "direct_energy_logit_rank_global_then_local_v38a",
        "rank_min_delta": 1e-6,
        "ranking_objective": "pre_softplus_stratified_v38a",
        "logit_rank_temperature": 1.0,
        "family_stratified_rank_weight": 1.0,
        "family_listwise_weight": 0.0,
        "coordinate_loss_weight": 0.25,
        "coordinate_width_normalized": True,
        "global_value_base_only": True,
        "full_family_global_batches": True,
        "cross_prompt_rank_weight": 0.0,
        "prompt_dropout_probability": 0.0,
        "prompt_consistency_weight": 0.0,
        "outside_tail_coordinate_weight": 0.0,
        "boundary_region_weight": 0.0,
        "band_component_weight": 0.0,
        "coordinate_fd_weight": 0.0,
        "coordinate_prompt_consistency_weight": 0.0,
    }
    for values, required in ((model, required_model), (training, required_training)):
        for key, expected in required.items():
            if values.get(key) != expected:
                raise ValueError(f"V3.9-A matched contract changed: {key}")
    return dict(experiment)


def grouped_split(rows: list[dict], fold: int | None, diagnostic: bool = False):
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate dataset IDs")
    train = [r for r in rows if r["split"] == "train"]
    families = sorted({r["prompt_family"] for r in train})
    development = {r["prompt_family"] for r in rows if r["split"] == "development"}
    if len(families) != 20 or len(development) != 4 or set(families) & development:
        raise ValueError("Expected the original disjoint 20/4 family split")
    if {r["split"] for r in rows} != {"train", "development"}:
        raise ValueError("Unexpected dataset partition")
    if fold is not None and (type(fold) is not int or fold not in range(5)):
        raise ValueError("Invalid CV fold")
    if diagnostic and fold is None:
        raise ValueError("Diagnostics require a fold to exclude its outer families")
    held = set(families[fold::5]) if fold is not None else set()
    fit_families = set(families) - held
    if diagnostic:
        fit_families = set(sorted(fit_families)[:2])
    fit = [r for r in train if r["prompt_family"] in fit_families]
    outer = [] if diagnostic else [r for r in train if r["prompt_family"] in held]
    for group in (fit, outer):
        ids = {r["sample_id"] for r in group}
        for row in group:
            if row["source_kind"] == "local_finite_difference" and row["trajectory_id"] not in ids:
                raise ValueError("Local point detached from its base anchor")
    return fit, outer


def validate_protocol(protocol: dict, rows: list[dict]) -> None:
    mode = protocol["run_mode"]
    if mode not in {"cv", "diagnostic", "full"}:
        raise ValueError("Invalid V3.9-A run mode")
    fold = protocol["cv_fold"]
    if (mode == "full") != (fold is None):
        raise ValueError("Run mode differs from CV fold")
    fit, outer = grouped_split(rows, fold, mode == "diagnostic")
    for name, group in (("train", fit), ("selection", outer)):
        expected = sorted(r["sample_id"] for r in group)
        if protocol[f"{name}_sample_ids"] != expected:
            raise ValueError(f"Protocol {name} IDs differ from the grouped fit/outer split")
        if protocol[f"{name}_families"] != sorted({r["prompt_family"] for r in group}):
            raise ValueError(f"Protocol {name} families differ")
    scope = {"cv": "train_family_cv", "diagnostic": "train_fit_diagnostic", "full": "development"}[
        mode
    ]
    if protocol["validation_scope"] != scope or protocol["outer_evaluation_count"] != int(
        mode == "cv"
    ):
        raise ValueError("Invalid evaluation scope/count")
    if protocol["development_used"] or protocol["checkpoint_selection"] != SELECTION:
        raise ValueError("Development/outer checkpoint selection is forbidden")
    if protocol["experiment_contract"]["variant"] not in VARIANTS:
        raise ValueError("Unknown V3.9-A variant")


def verify_manifest(path: Path, dataset_rows: list[dict] | None = None) -> tuple[dict, dict]:
    m = read_json(path)
    if m.get("architecture_revision") != REVISION or m.get("checkpoint_selection") != SELECTION:
        raise ValueError("Expected a V3.9-A fixed-epoch manifest")
    if (
        m.get("run_mode") not in {"cv", "full"}
        or m.get("local_residual_mode") != "zero_global_only"
    ):
        raise ValueError("Diagnostic/local models cannot be V3.9-A final checkpoints")
    if m.get("qualification_eligible") or m.get("production_authorization"):
        raise ValueError("Training does not grant qualification or production authorization")
    for filename, key in (
        ("pitch3_lte_run_protocol.json", "run_protocol_sha256"),
        ("pitch3_lte_training_statistics.json", "training_statistics_sha256"),
        ("pitch3_lte_effective_config.json", "training_config_sha256"),
    ):
        verify_file(path.parent / filename, m[key])
    checkpoint = local_file(path.parent, m["checkpoint"])
    verify_file(checkpoint, m["checkpoint_sha256"])
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError(f"Incomplete checkpoint ZIP: {checkpoint}")
    for filename, digest in m["diagnostic_artifacts_sha256"].items():
        verify_file(local_file(path.parent, filename), digest)
    if "pitch3_lte_initialization.json" not in m["diagnostic_artifacts_sha256"]:
        raise ValueError("Missing initialization artifact digest")
    p = read_json(path.parent / "pitch3_lte_run_protocol.json")
    for key in (
        "run_mode",
        "cv_fold",
        "validation_scope",
        "experiment_contract",
        "checkpoint_selection",
        "seed",
    ):
        if m[key] != p[key]:
            raise ValueError(f"Manifest detached from protocol: {key}")
    if p["dataset_manifest_sha256"] != m["training_manifest_sha256"]:
        raise ValueError("Manifest detached from dataset")
    if p["development_used"] or p["outer_evaluation_count"] != int(m["run_mode"] == "cv"):
        raise ValueError("V3.9-A must evaluate outer once and never select on development")
    stats = read_json(path.parent / "pitch3_lte_training_statistics.json")
    if stats["run_protocol_sha256"] != m["run_protocol_sha256"]:
        raise ValueError("Statistics detached from protocol")
    if content_hash(stats["loss_normalizers"]) != m["initial_normalizers_sha256"]:
        raise ValueError("Normalizer binding differs")
    initial = read_json(path.parent / "pitch3_lte_initialization.json")
    if (
        initial["shared_state_sha256"] != m["shared_initial_state_sha256"]
        or not initial["outputs_and_losses_equal"]
    ):
        raise ValueError("Shared initialization identity was not established")
    effective = read_json(path.parent / "pitch3_lte_effective_config.json")
    if effective["model"]["transition_branch"] != (
        m["experiment_contract"]["variant"] == "transition"
    ):
        raise ValueError("Architecture differs from experiment variant")
    if effective["model"]["ordinal_auxiliary"] or effective["v39a"] != m["experiment_contract"]:
        raise ValueError("Effective configuration differs from V3.9-A contract")
    if m["epochs_completed"] != m["experiment_contract"]["global_epochs"]:
        raise ValueError("Incomplete fixed epoch training")
    if dataset_rows is not None:
        validate_protocol(p, dataset_rows)
    for split in ("train", "selection"):
        ids = p[f"{split}_sample_ids"]
        if not ids:
            continue
        pred = path.parent / f"pitch3_lte_{split}_predictions.csv"
        verify_file(pred, m[f"{split}_predictions_sha256"])
        rows = read_csv(pred)
        if sorted(r["sample_id"] for r in rows) != ids:
            raise ValueError("Prediction IDs differ from protocol")
        if any(
            float(r["predicted_local_energy"]) != 0
            or float(r["predicted_energy"]) != float(r["predicted_global_energy"])
            for r in rows
        ):
            raise ValueError("V3.9-A must retain exactly zero local residual")
    return m, p


def verify_diagnostic(path: Path) -> dict:
    report = read_json(path)
    if report.get("stage") != "pitch3_lte_v39a_fit_diagnostic" or not report.get("completed"):
        raise ValueError("Missing completed fit-only diagnostic")
    if (
        report["development_used"]
        or report["outer_evaluation_count"]
        or not report["diagnostic_only"]
    ):
        raise ValueError("Diagnostic used held-out labels")
    for filename, digest in report["artifacts_sha256"].items():
        verify_file(local_file(path.parent, filename), digest)
    return report
