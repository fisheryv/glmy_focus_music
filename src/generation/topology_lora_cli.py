"""Command-line orchestration for exact reranking and distilled topology LoRA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .experiment import load_experiment_config
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import load_reranking_gate
from .path_homology_exact_scorer import ExactPathHomologyScorer
from .topology_lora import (
    TOPOLOGY_LORA_EXPERIMENT_V2,
    build_reranking_prompt_splits,
    export_lora_teacher_dataset,
)
from .topology_lora_training import (
    finalize_lora_artifact,
    load_lora_config,
    run_native_stage,
)
from .topology_lora_validation import run_paired_validation, select_development_scale


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _resolved(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def command_prepare_prompts(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    config = load_lora_config(_resolved(root, args.config))
    payload = build_reranking_prompt_splits(
        _resolved(root, args.source),
        _resolved(root, args.output_dir),
        experiment=config["experiment"],
    )
    _print(payload)
    return 0


def command_check_gate(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    config = load_lora_config(_resolved(root, args.config))
    gate_path = _resolved(root, args.reranking_gate)
    scorer = ExactPathHomologyScorer.from_json(_resolved(root, args.fingerprint))
    gate = load_reranking_gate(gate_path, scorer.contract)
    if config["experiment"] == TOPOLOGY_LORA_EXPERIMENT_V2:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            payload.get("selection_policy", {}).get("name")
            != "exact_topology_constrained_reranker_v2"
        ):
            raise LTSNContractError("v2 LoRA requires the passed constrained-reranker v2 gate")
    _print(
        {
            "ok": True,
            "gate_sha256": gate.artifact_sha256,
            "median_loss_improvement_fraction": gate.median_loss_improvement_fraction,
            "bootstrap_ci95_low": gate.bootstrap_ci95_low,
        }
    )
    return 0


def command_export_teacher(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    config = load_lora_config(_resolved(root, args.config))
    payload = export_lora_teacher_dataset(
        reranking_run_dir=_resolved(root, args.reranking_run),
        prompt_manifest_path=_resolved(root, args.prompt_manifest),
        fingerprint_path=_resolved(root, args.fingerprint),
        reranking_gate_path=_resolved(root, args.reranking_gate),
        output_dir=_resolved(root, args.output_dir),
        selection_contract_path=(
            _resolved(root, args.selection_contract) if args.selection_contract else None
        ),
        experiment=config["experiment"],
        activation_tag=args.activation_tag,
        include_baseline_replay=not args.no_baseline_replay,
    )
    _print(payload)
    return 0


def command_native(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    payload = run_native_stage(
        stage=args.command,
        project_root=root,
        config_path=_resolved(root, args.config),
        teacher_dir=_resolved(root, args.teacher_dir),
        tensor_dir=_resolved(root, args.tensor_dir),
        output_dir=_resolved(root, args.lora_output),
        python_bin=args.python_bin,
    )
    _print(payload)
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    payload = finalize_lora_artifact(
        _resolved(root, args.lora_output),
        _resolved(root, args.config),
        _resolved(root, args.teacher_dir),
    )
    _print(payload)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if args.prompt_manifest is None or args.validation_run is None or args.scale is None:
        raise ValueError("validate requires --prompt-manifest, --validation-run, and --scale")
    config_path = _resolved(root, args.config)
    artifact_path = _resolved(root, args.lora_artifact)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        artifact.get("status") != "trained_unqualified"
        or artifact.get("config_sha256") != sha256_file(config_path)
    ):
        raise LTSNContractError("LoRA artifact is not bound to this frozen configuration")
    frozen = load_lora_config(config_path)
    if args.split == "development" and args.scale not in [
        float(value) for value in frozen["validation"]["development_scales"]
    ]:
        raise LTSNContractError("development scale was not frozen before training")
    if args.split == "qualification":
        selection_path = _resolved(root, args.scale_selection)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection.get("status") != "selected"
            or selection.get("qualification_authorized") is not True
            or selection.get("lora_bundle_sha256") != artifact["lora_bundle_sha256"]
            or not math.isclose(
                float(selection.get("selected_scale")),
                args.scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise LTSNContractError("qualification scale was not selected on development")
    prompt_path = _resolved(root, args.prompt_manifest)
    rerank_config = load_experiment_config(
        root,
        args.rerank_config,
        run_id=f"topology_lora_{args.split}",
        prompt_manifest=str(prompt_path),
    )
    payload = run_paired_validation(
        project_root=root,
        config=rerank_config,
        lora_config_path=config_path,
        prompt_manifest=prompt_path,
        split=args.split,
        run_dir=_resolved(root, args.validation_run),
        lora_path=Path(artifact["lora_path"]),
        lora_sha256=artifact["lora_bundle_sha256"],
        scale=args.scale,
        seed_start=args.seed_start,
        activation_tag=args.activation_tag,
    )
    _print(payload)
    return 0


def command_select_scale(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    payload = select_development_scale(
        validation_root=_resolved(root, args.validation_root),
        lora_config_path=_resolved(root, args.config),
        output_path=_resolved(root, args.scale_selection),
    )
    _print(payload)
    return 0 if payload["qualification_authorized"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the topology reranking/LoRA orchestration parser."""

    parser = argparse.ArgumentParser(prog="focus-topology-lora")
    parser.add_argument(
        "command",
        choices=(
            "prepare-prompts",
            "check-gate",
            "export-teacher",
            "preprocess",
            "train",
            "finalize",
            "validate",
            "select-scale",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/topology_lora_v1.json"))
    parser.add_argument("--source", type=Path, default=Path("metadata/ltsn_prompts.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/topology_rerank_lora_v1/prompts")
    )
    parser.add_argument("--reranking-run", type=Path)
    parser.add_argument("--selection-contract", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_path_homology_fingerprint_v2.json"),
    )
    parser.add_argument(
        "--reranking-gate",
        type=Path,
        default=Path("metadata/ace_reranking_effect_gate.json"),
    )
    parser.add_argument("--activation-tag", default="topology_focus")
    parser.add_argument("--no-baseline-replay", action="store_true")
    parser.add_argument(
        "--teacher-dir", type=Path, default=Path("runs/topology_rerank_lora_v1/teacher")
    )
    parser.add_argument(
        "--tensor-dir", type=Path, default=Path("runs/topology_rerank_lora_v1/tensors")
    )
    parser.add_argument(
        "--lora-output", type=Path, default=Path("runs/topology_rerank_lora_v1/lora")
    )
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument(
        "--rerank-config", type=Path, default=Path("configs/ace_rerank_180s.toml")
    )
    parser.add_argument(
        "--lora-artifact",
        type=Path,
        default=Path("runs/topology_rerank_lora_v1/lora_artifact.json"),
    )
    parser.add_argument("--split", choices=("development", "qualification"))
    parser.add_argument("--scale", type=float)
    parser.add_argument("--validation-run", type=Path)
    parser.add_argument("--seed-start", type=int, default=2026091100)
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("runs/topology_rerank_lora_v1/validation"),
    )
    parser.add_argument(
        "--scale-selection",
        type=Path,
        default=Path("runs/topology_rerank_lora_v1/scale_selection.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-prompts":
            return command_prepare_prompts(args)
        if args.command == "check-gate":
            return command_check_gate(args)
        if args.command == "export-teacher":
            if args.reranking_run is None or args.prompt_manifest is None:
                parser.error("export-teacher requires --reranking-run and --prompt-manifest")
            return command_export_teacher(args)
        if args.command in {"preprocess", "train"}:
            return command_native(args)
        if args.command == "validate":
            if args.split is None:
                parser.error("validate requires --split")
            return command_validate(args)
        if args.command == "select-scale":
            return command_select_scale(args)
        return command_finalize(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
