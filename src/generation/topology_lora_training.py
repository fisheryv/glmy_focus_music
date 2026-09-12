"""Audited wrapper around ACE-Step's native LoRA preprocessing and training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifact_hash import sha256_directory
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import canonical_json_sha256, write_json_atomic
from .topology_lora import TOPOLOGY_LORA_EXPERIMENT_V2, TOPOLOGY_LORA_EXPERIMENTS


def load_lora_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the frozen topology-LoRA configuration."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") not in TOPOLOGY_LORA_EXPERIMENTS
        or payload.get("status") != "frozen_before_training"
    ):
        raise LTSNContractError("topology LoRA configuration is not frozen")
    if payload.get("model", {}).get("variant") != "xl_turbo":
        raise LTSNContractError("topology LoRA must use the ACE-Step XL-Turbo base")
    if payload["experiment"] == TOPOLOGY_LORA_EXPERIMENT_V2:
        teacher = payload.get("teacher", {})
        if (
            teacher.get("candidate_count") != 16
            or teacher.get("selection") != "frozen_global_constrained_reranker_v2"
        ):
            raise LTSNContractError("v2 topology LoRA requires constrained best-of-16 teachers")
    return payload


def _teacher_inputs(
    teacher_dir: Path, expected_experiment: str
) -> tuple[Path, dict[str, Any]]:
    report_path = teacher_dir / "teacher_report.json"
    dataset_path = teacher_dir / "ace_lora_dataset.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("experiment") != expected_experiment
        or report.get("dataset_json_sha256") != sha256_file(dataset_path)
        or report.get("winner_samples", 0) < 1
    ):
        raise LTSNContractError("LoRA teacher dataset is missing or its binding changed")
    if expected_experiment == TOPOLOGY_LORA_EXPERIMENT_V2:
        required_hashes = (
            "reranking_gate_sha256",
            "selection_contract_sha256",
            "selector_config_sha256",
            "frozen_selector_sha256",
        )
        if report.get("teacher_source") != "frozen_constrained_exact_18d_bestof16" or any(
            len(str(report.get(name, ""))) != 64 for name in required_hashes
        ):
            raise LTSNContractError("v2 LoRA teacher provenance is incomplete")
    return dataset_path, report


def build_native_command(
    *,
    stage: str,
    project_root: Path,
    config_path: Path,
    teacher_dir: Path,
    tensor_dir: Path,
    output_dir: Path,
    python_bin: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build a hash-bound ACE-Step native preprocess or training command."""

    if stage not in {"preprocess", "train"}:
        raise ValueError("stage must be preprocess or train")
    config = load_lora_config(config_path)
    dataset_path, teacher = _teacher_inputs(teacher_dir, config["experiment"])
    model = config["model"]
    executable = str((python_bin or Path(sys.executable)).resolve())
    checkpoint_dir = (project_root / model["checkpoint_dir"]).resolve()
    command = [
        executable,
        "-m",
        "acestep.training_v2.cli.train_fixed",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--model-variant",
        model["variant"],
        "--device",
        model["device"],
        "--precision",
        model["precision"],
        "--plain",
    ]
    if stage == "preprocess":
        command += [
            "--preprocess",
            "--dataset-json",
            str(dataset_path.resolve()),
            "--tensor-output",
            str(tensor_dir.resolve()),
            "--max-duration",
            str(config["preprocess"]["max_duration_seconds"]),
        ]
    else:
        train = config["training"]
        if not tensor_dir.is_dir() or not any(tensor_dir.rglob("*.pt")):
            raise LTSNContractError("preprocessed LoRA tensor dataset is empty")
        command += [
            "--dataset-dir",
            str(tensor_dir.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--base-model",
            model["base_model"],
            "--adapter-type",
            "lora",
            "--rank",
            str(train["rank"]),
            "--alpha",
            str(train["alpha"]),
            "--dropout",
            str(train["dropout"]),
            "--target-modules",
            *train["target_modules"],
            "--attention-type",
            train["attention_type"],
            "--lr",
            str(train["learning_rate"]),
            "--batch-size",
            str(train["batch_size"]),
            "--gradient-accumulation",
            str(train["gradient_accumulation"]),
            "--epochs",
            str(train["epochs"]),
            "--save-every",
            str(train["save_every_epochs"]),
            "--seed",
            str(train["seed"]),
            "--yes",
        ]
    plan = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "stage": stage,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "teacher_report_sha256": sha256_file(teacher_dir / "teacher_report.json"),
        "dataset_json_sha256": teacher["dataset_json_sha256"],
        "tensor_dir": str(tensor_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "command": command,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return command, plan


def run_native_stage(
    *,
    stage: str,
    project_root: Path,
    config_path: Path,
    teacher_dir: Path,
    tensor_dir: Path,
    output_dir: Path,
    python_bin: Path | None = None,
) -> dict[str, Any]:
    """Freeze a native ACE command, execute it, and record completion."""

    command, plan = build_native_command(
        stage=stage,
        project_root=project_root,
        config_path=config_path,
        teacher_dir=teacher_dir,
        tensor_dir=tensor_dir,
        output_dir=output_dir,
        python_bin=python_bin,
    )
    audit_dir = output_dir.parent / "audit"
    plan_path = audit_dir / f"{stage}_plan.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous.get("plan_sha256") != plan["plan_sha256"]:
            raise LTSNContractError(f"existing {stage} plan differs from the frozen command")
    else:
        write_json_atomic(plan_path, plan)
    checkout = (project_root / load_lora_config(config_path)["model"]["checkout"]).resolve()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(checkout), str(project_root / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(command, cwd=checkout, env=environment, check=True)
    completion = {**plan, "completed": True}
    if stage == "preprocess":
        completion["tensor_bundle_sha256"] = sha256_directory(tensor_dir)
    write_json_atomic(audit_dir / f"{stage}_complete.json", completion)
    return completion


def finalize_lora_artifact(
    output_dir: Path, config_path: Path, teacher_dir: Path
) -> dict[str, Any]:
    """Bind the completed adapter directory to its teacher and frozen config."""

    config = load_lora_config(config_path)
    _teacher_inputs(teacher_dir, config["experiment"])
    adapter_dir = output_dir / "final"
    required = (adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors")
    if not all(path.is_file() for path in required):
        raise LTSNContractError("ACE-Step final LoRA adapter files are missing")
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "trained_unqualified",
        "production_authorization": False,
        "lora_path": str(adapter_dir.resolve()),
        "lora_bundle_sha256": sha256_directory(adapter_dir),
        "config_sha256": sha256_file(config_path),
        "teacher_report_sha256": sha256_file(teacher_dir / "teacher_report.json"),
        "next_stage": "development_scale_selection_then_fresh_qualification",
    }
    write_json_atomic(output_dir.parent / "lora_artifact.json", report)
    return report
