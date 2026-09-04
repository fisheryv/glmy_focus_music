from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .ace_adapter import AceStepAdapter
from .experiment import ExperimentConfig, load_experiment_config
from .fake_backend import FakeMusicBackend
from .rerank_experiment import (
    ensure_experiment,
    evaluate_noninferiority_table,
    experiment_root,
    generate_candidates,
    initialize_noninferiority_report,
    issue_reranking_gate,
    issue_surrogate_training_gate,
    score_candidates,
)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load(args: argparse.Namespace) -> tuple[Path, ExperimentConfig]:
    root = args.root.resolve()
    config = load_experiment_config(root, args.config, run_id=args.run_id)
    return root, config


def _preflight(root: Path, config: ExperimentConfig, backend_name: str) -> dict[str, Any]:
    checkout = root / config.ace.checkout
    required_paths = {
        "pipeline_config": root / "configs" / "pipeline.toml",
        "prompt_manifest": root / config.prompt_manifest,
        "state_model": root / "features" / "models" / "state_model.npz",
        "state_model_metadata": root / "features" / "models" / "state_model.json",
        "pitch_codebook": root / "features" / "models" / "pitch_v2_codebook.npz",
        "frozen_18d_scorer": root / config.scoring.fingerprint_path,
        "noninferiority_protocol": root / config.scoring.noninferiority_protocol_path,
        "ace_checkout": checkout / "pyproject.toml",
    }
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "librosa",
            "soundfile",
            "ripser",
            "sklearn",
            "pyloudnorm",
            "imageio_ffmpeg",
        )
    }
    cuda: dict[str, Any] = {"required": backend_name == "ace", "available": None}
    if backend_name == "ace":
        try:
            import torch

            cuda["available"] = bool(torch.cuda.is_available())
            cuda["device_count"] = int(torch.cuda.device_count())
            cuda["devices"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
        except ImportError:
            cuda["available"] = False
            cuda["error"] = "torch is not installed"
    path_status = {name: path.is_file() for name, path in required_paths.items()}
    ok = all(path_status.values()) and all(dependencies.values())
    if backend_name == "ace":
        ok = ok and bool(cuda["available"])
    return {
        "ok": ok,
        "backend": backend_name,
        "paths": path_status,
        "dependencies": dependencies,
        "cuda": cuda,
        "checkpoint_present": (checkout / "checkpoints" / config.ace.model).is_dir(),
        "checkpoint_note": "ACE-Step can download missing checkpoints during initialization",
    }


def _backend(root: Path, config: ExperimentConfig, name: str):
    if name == "fake":
        return FakeMusicBackend()
    return AceStepAdapter(root / config.ace.checkout, config.ace)


def command_preflight(args: argparse.Namespace) -> int:
    root, config = _load(args)
    payload = _preflight(root, config, args.backend)
    _print(payload)
    return 0 if payload["ok"] else 1


def command_plan(args: argparse.Namespace) -> int:
    root, config = _load(args)
    run_root, records = ensure_experiment(root, config)
    _print(
        {
            "ok": True,
            "run_root": str(run_root),
            "prompt_pools": len({record.prompt_id for record in records}),
            "candidates": len(records),
            "duration_seconds": config.duration_seconds,
        }
    )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    root, config = _load(args)
    records = generate_candidates(
        root,
        config,
        _backend(root, config, args.backend),
        retry_failed=args.retry_failed,
    )
    failed = [record for record in records if record.status == "failed"]
    _print(
        {
            "ok": not failed,
            "run_root": str(experiment_root(root, config)),
            "generated": sum(record.status in {"generated", "scored"} for record in records),
            "failed": len(failed),
        }
    )
    return 0 if not failed else 1


def command_score(args: argparse.Namespace) -> int:
    root, config = _load(args)
    _, records = ensure_experiment(root, config)
    failed = [record for record in records if record.status == "failed"]
    pending = [record for record in records if record.status == "planned"]
    if failed or pending:
        raise ValueError(
            "generation is incomplete: "
            f"planned={len(pending)}, failed={len(failed)}; "
            "resume it with `python -m generation.rerank_cli generate "
            f"--root {root} --config {args.config} --backend {args.backend} --retry-failed`"
        )
    summary = score_candidates(root, config, records)
    _print(summary)
    return 0


def command_run(args: argparse.Namespace) -> int:
    generated = command_generate(args)
    if generated:
        return generated
    return command_score(args)


def command_init_evidence(args: argparse.Namespace) -> int:
    root, config = _load(args)
    output = args.noninferiority_report or (
        experiment_root(root, config) / "noninferiority_report.json"
    )
    payload = initialize_noninferiority_report(root, config, output.resolve())
    _print({"ok": True, "output": str(output.resolve()), **payload})
    return 0


def command_evaluate_evidence(args: argparse.Namespace) -> int:
    root, config = _load(args)
    if args.noninferiority_table is None:
        raise ValueError("--noninferiority-table is required")
    output = args.noninferiority_report or (
        experiment_root(root, config) / "noninferiority_report.json"
    )
    payload = evaluate_noninferiority_table(
        root,
        config,
        args.noninferiority_table.resolve(),
        output.resolve(),
    )
    _print({"ok": True, "output": str(output.resolve()), **payload})
    return 0


def command_issue_gate(args: argparse.Namespace) -> int:
    root, config = _load(args)
    report = args.noninferiority_report or (
        experiment_root(root, config) / "noninferiority_report.json"
    )
    output = args.gate_output or (root / "metadata" / "ace_reranking_effect_gate.json")
    payload = issue_reranking_gate(root, config, report.resolve(), output.resolve())
    _print({"ok": True, "output": str(output.resolve()), **payload})
    return 0


def command_issue_training_gate(args: argparse.Namespace) -> int:
    root, config = _load(args)
    report = args.noninferiority_report or (
        experiment_root(root, config) / "noninferiority_report.json"
    )
    output = args.training_gate_output or (
        root / "metadata" / "ltsn_surrogate_training_gate.json"
    )
    payload = issue_surrogate_training_gate(
        root, config, report.resolve(), output.resolve()
    )
    _print({"ok": True, "output": str(output.resolve()), **payload})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-ace-rerank")
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "plan",
            "generate",
            "score",
            "run",
            "init-evidence",
            "evaluate-evidence",
            "issue-training-gate",
            "issue-gate",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/ace_rerank_180s.toml"))
    parser.add_argument("--run-id")
    parser.add_argument("--backend", choices=("ace", "fake"), default="ace")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--noninferiority-report", type=Path)
    parser.add_argument("--noninferiority-table", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--training-gate-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler = globals()[f"command_{args.command.replace('-', '_')}"]
        return int(handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
