"""Audited wrapper around ACE-Step's native LoRA preprocessing and training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ace_lora_native_entrypoint import RUNTIME_POLICY, VARIANT_DIRECTORY_ALIASES
from .artifact_hash import sha256_directory
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import canonical_json_sha256, write_json_atomic
from .topology_lora import TOPOLOGY_LORA_EXPERIMENT_V2, TOPOLOGY_LORA_EXPERIMENTS

_NATIVE_TRAINING_DISTRIBUTIONS = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "soundfile": "soundfile",
    "safetensors": "safetensors",
    "loguru": "loguru",
    "peft": "peft",
    "lycoris": "lycoris-lora",
    "lightning": "lightning",
    "tensorboard": "tensorboard",
}
_SUPPLEMENTAL_TRAINING_MODULES = {
    "loguru",
    "peft",
    "lycoris",
    "lightning",
    "tensorboard",
}


def _python_executable(python_bin: Path | None) -> Path:
    """Return an absolute interpreter path without dereferencing a venv symlink."""

    requested = (python_bin or Path(sys.executable)).expanduser()
    return Path(os.path.abspath(os.fspath(requested)))


def _native_environment(project_root: Path, checkout: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["FOCUS_LORA_SAFE_ROOT"] = str(project_root.resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(checkout), str(project_root / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return environment


def _freeze_native_stage_plan(
    *,
    audit_dir: Path,
    stage: str,
    plan: dict[str, Any],
    stage_output: Path,
    allow_failed_train_retry: bool = False,
) -> Path | None:
    """Freeze a plan, preserving a failed pre-execution plan when the interpreter changes."""

    plan_path = audit_dir / f"{stage}_plan.json"
    completion_path = audit_dir / f"{stage}_complete.json"
    if not plan_path.is_file():
        write_json_atomic(plan_path, plan)
        return None
    previous = json.loads(plan_path.read_text(encoding="utf-8"))
    if previous.get("plan_sha256") == plan["plan_sha256"]:
        return None
    if completion_path.is_file():
        raise LTSNContractError(f"completed {stage} plan differs from the frozen command")
    output_files = (
        [path for path in stage_output.rglob("*") if path.is_file()]
        if stage_output.is_dir()
        else []
    )
    if output_files:
        previous_command = list(previous.get("command", []))
        current_command = list(plan.get("command", []))
        previous_without_policy = {
            key: value
            for key, value in previous.items()
            if key not in {"command", "native_runtime_policy", "plan_sha256"}
        }
        current_without_policy = {
            key: value
            for key, value in plan.items()
            if key not in {"command", "native_runtime_policy", "plan_sha256"}
        }
        safe_wrapper_transition = bool(
            stage == "preprocess"
            and previous_without_policy == current_without_policy
            and len(previous_command) >= 4
            and len(current_command) >= 4
            and previous_command[0] == current_command[0]
            and previous_command[1] == current_command[1] == "-m"
            and previous_command[2] == "acestep.training_v2.cli.train_fixed"
            and current_command[2] == "generation.ace_lora_native_entrypoint"
            and previous_command[3:] == current_command[3:]
            and all(path.name.endswith((".pt", ".tmp.pt")) for path in output_files)
        )
        safe_train_policy_transition = bool(
            allow_failed_train_retry
            and stage == "train"
            and previous_without_policy == current_without_policy
            and previous_command == current_command
            and not _train_adapter_is_complete(stage_output)
        )
        if not (safe_wrapper_transition or safe_train_policy_transition):
            raise LTSNContractError(
                f"existing {stage} plan differs and its output directory is not empty"
            )
    previous_sha256 = str(previous.get("plan_sha256") or canonical_json_sha256(previous))
    archived = audit_dir / f"{stage}_plan_superseded_{previous_sha256[:12]}.json"
    if not archived.is_file():
        write_json_atomic(archived, previous)
    write_json_atomic(plan_path, plan)
    return archived


def check_native_training_environment(
    *,
    project_root: Path,
    config_path: Path,
    python_bin: Path | None = None,
) -> dict[str, Any]:
    """Check the exact interpreter used by ACE-Step before preprocessing or training."""

    config = load_lora_config(config_path)
    checkout = (project_root / config["model"]["checkout"]).resolve()
    executable_path = _python_executable(python_bin)
    executable = str(executable_path)
    if not executable_path.is_file():
        raise LTSNContractError(f"LoRA Python interpreter is missing: {executable}")
    requirements_path = (
        project_root / "configs" / "topology_lora_training_requirements.txt"
    ).resolve()
    probe = (
        "import importlib.metadata as m, importlib.util as u, json, sys; "
        "items=json.loads(sys.argv[1]); missing=[]; versions={}; "
        "[(missing.append(module) if u.find_spec(module) is None else "
        "versions.__setitem__(dist, m.version(dist))) for module,dist in items.items()]; "
        "print(json.dumps({'missing_modules':missing,'versions':versions},sort_keys=True)); "
        "raise SystemExit(1 if missing else 0)"
    )
    result = subprocess.run(
        [executable, "-c", probe, json.dumps(_NATIVE_TRAINING_DISTRIBUTIONS)],
        cwd=checkout,
        env=_native_environment(project_root, checkout),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise LTSNContractError(
            f"could not inspect ACE-Step training environment: {result.stderr.strip()}"
        ) from exc
    missing = [str(module) for module in payload.get("missing_modules", [])]
    if missing:
        if set(missing) <= _SUPPLEMENTAL_TRAINING_MODULES:
            install = f"{executable} -m pip install -r {requirements_path}"
        else:
            synced_python = checkout / ".venv" / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            install = (
                f"cd {checkout} && uv sync --locked; then use "
                f"{synced_python} as PYTHON_BIN"
            )
        raise LTSNContractError(
            "ACE-Step training environment is incomplete; "
            f"missing modules: {', '.join(missing)}. Install the pinned dependencies with: "
            f"{install}"
        )
    entrypoint = subprocess.run(
        [
            executable,
            "-c",
            "import acestep.training_v2.cli.train_fixed; print('entrypoint_import_ok')",
        ],
        cwd=checkout,
        env=_native_environment(project_root, checkout),
        capture_output=True,
        text=True,
        check=False,
    )
    if entrypoint.returncode != 0:
        detail = (entrypoint.stderr or entrypoint.stdout).strip()
        raise LTSNContractError(
            "ACE-Step native training entrypoint cannot be imported by "
            f"{executable}: {detail}"
        )
    return {
        "ok": True,
        "python_executable": executable,
        "ace_checkout": str(checkout),
        "entrypoint": "acestep.training_v2.cli.train_fixed",
        "entrypoint_importable": True,
        "versions": payload["versions"],
        "requirements_path": str(requirements_path),
        "requirements_sha256": sha256_file(requirements_path),
    }


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
    executable = str(_python_executable(python_bin))
    checkpoint_dir = (project_root / model["checkpoint_dir"]).resolve()
    runtime_model_variant = VARIANT_DIRECTORY_ALIASES.get(
        model["variant"], model["variant"]
    )
    runtime_model_dir = checkpoint_dir / runtime_model_variant
    command = [
        executable,
        "-m",
        "generation.ace_lora_native_entrypoint",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--model-variant",
        runtime_model_variant,
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
        "runtime_model_variant": runtime_model_variant,
        "runtime_model_dir": str(runtime_model_dir),
        "native_runtime_policy": RUNTIME_POLICY,
        "command": command,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return command, plan


def _validate_preprocess_outputs(dataset_path: Path, tensor_dir: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise LTSNContractError("LoRA teacher dataset has no preprocess samples")
    expected_names = [f"{Path(str(sample['audio_path'])).stem}.pt" for sample in samples]
    if len(set(expected_names)) != len(expected_names):
        raise LTSNContractError("LoRA teacher audio stems collide in tensor output")
    final_files = {
        path.name: path
        for path in tensor_dir.glob("*.pt")
        if not path.name.endswith(".tmp.pt")
    }
    temporary_files = sorted(path.name for path in tensor_dir.glob("*.tmp.pt"))
    expected = set(expected_names)
    observed = set(final_files)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    empty = sorted(name for name, path in final_files.items() if path.stat().st_size == 0)
    if missing or unexpected or temporary_files or empty:
        raise LTSNContractError(
            "ACE-Step preprocessing output is incomplete or contaminated: "
            f"expected={len(expected)}, final={len(observed)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, temporary={len(temporary_files)}, empty={len(empty)}"
        )
    return {
        "expected_samples": len(expected),
        "final_tensors": len(observed),
        "temporary_tensors": 0,
        "complete": True,
    }


def _train_adapter_paths(output_dir: Path) -> tuple[Path, Path]:
    adapter_dir = output_dir / "final"
    return adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"


def _train_adapter_is_complete(output_dir: Path) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in _train_adapter_paths(output_dir)
    )


def _validate_train_outputs(output_dir: Path) -> dict[str, Any]:
    required = _train_adapter_paths(output_dir)
    missing = [str(path) for path in required if not path.is_file()]
    empty = [str(path) for path in required if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise LTSNContractError(
            "ACE-Step reported success but the final LoRA adapter is incomplete: "
            f"missing={len(missing)}, empty={len(empty)}"
        )
    return {
        "final_adapter_dir": str((output_dir / "final").resolve()),
        "adapter_config_sha256": sha256_file(required[0]),
        "adapter_model_sha256": sha256_file(required[1]),
        "adapter_config_bytes": required[0].stat().st_size,
        "adapter_model_bytes": required[1].stat().st_size,
        "complete": True,
    }


def _archive_invalid_train_completion(audit_dir: Path, output_dir: Path) -> Path | None:
    """Atomically preserve a false completion marker before a safe retry."""

    completion_path = audit_dir / "train_complete.json"
    if not completion_path.is_file() or _train_adapter_is_complete(output_dir):
        return None
    completion_sha256 = sha256_file(completion_path)
    archived = audit_dir / f"train_complete_invalid_{completion_sha256[:12]}.json"
    suffix = 1
    while archived.exists():
        if sha256_file(archived) == completion_sha256:
            os.replace(completion_path, archived)
            return archived
        archived = audit_dir / f"train_complete_invalid_{completion_sha256[:12]}_{suffix}.json"
        suffix += 1
    os.replace(completion_path, archived)
    return archived


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

    environment_report = check_native_training_environment(
        project_root=project_root,
        config_path=config_path,
        python_bin=python_bin,
    )

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
    invalid_completion = (
        _archive_invalid_train_completion(audit_dir, output_dir)
        if stage == "train"
        else None
    )
    superseded_plan = _freeze_native_stage_plan(
        audit_dir=audit_dir,
        stage=stage,
        plan=plan,
        stage_output=tensor_dir if stage == "preprocess" else output_dir,
        allow_failed_train_retry=invalid_completion is not None,
    )
    checkout = (project_root / load_lora_config(config_path)["model"]["checkout"]).resolve()
    subprocess.run(
        command,
        cwd=checkout,
        env=_native_environment(project_root, checkout),
        check=True,
    )
    preprocess_output = None
    train_output = None
    if stage == "preprocess":
        dataset_path, _ = _teacher_inputs(
            teacher_dir,
            load_lora_config(config_path)["experiment"],
        )
        preprocess_output = _validate_preprocess_outputs(dataset_path, tensor_dir)
    else:
        train_output = _validate_train_outputs(output_dir)
    completion = {
        **plan,
        "completed": True,
        "training_environment": environment_report,
        "superseded_failed_plan": str(superseded_plan) if superseded_plan else None,
        "superseded_invalid_completion": (
            str(invalid_completion) if invalid_completion else None
        ),
        "preprocess_output": preprocess_output,
        "train_output": train_output,
    }
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
    _validate_train_outputs(output_dir)
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
