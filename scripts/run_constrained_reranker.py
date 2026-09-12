#!/usr/bin/env python3
"""Freeze and apply the calibration-derived constrained reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.constrained_reranker import (
    apply_constrained_selection,
    build_candidate_semantics,
    build_constrained_prompt_manifests,
    freeze_calibrated_selector,
    load_selector_config,
)
from generation.experiment import load_experiment_config
from generation.rerank_experiment import experiment_root


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare-prompts", "build-semantics", "apply", "freeze-calibration"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-config", type=Path)
    parser.add_argument(
        "--selector-config",
        type=Path,
        default=Path("configs/constrained_reranker_v1.json"),
    )
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument(
        "--prompt-source", type=Path, default=Path("metadata/ltsn_prompts.csv")
    )
    parser.add_argument(
        "--confirmation-source",
        type=Path,
        default=Path("generation/prompts/ace_constrained_confirmation_v1.csv"),
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path("runs/topology_rerank_lora_v1/constrained/prompts"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frozen-selector", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    if args.command == "prepare-prompts":
        selector = load_selector_config(_resolve(root, args.selector_config))
        payload = build_constrained_prompt_manifests(
            _resolve(root, args.prompt_source),
            _resolve(root, args.confirmation_source),
            _resolve(root, args.prompt_output),
            experiment=selector["experiment"],
        )
    elif args.command == "freeze-calibration":
        if args.output_dir is None or args.frozen_selector is None:
            raise ValueError("freeze-calibration requires --output-dir and --frozen-selector")
        payload = freeze_calibrated_selector(
            selector_config_path=_resolve(root, args.selector_config),
            calibration_dir=_resolve(root, args.output_dir),
            output_path=_resolve(root, args.frozen_selector),
        )
    else:
        if args.run_config is None:
            raise ValueError(f"{args.command} requires --run-config")
        config = load_experiment_config(root, args.run_config)
        run_root = experiment_root(root, config)
        output_dir = (
            _resolve(root, args.output_dir) if args.output_dir else run_root / "constrained"
        )
        prompt_manifest = (
            _resolve(root, args.prompt_manifest)
            if args.prompt_manifest
            else _resolve(root, Path(config.prompt_manifest))
        )
        if args.command == "build-semantics":
            payload = build_candidate_semantics(
                project_root=root,
                config=config,
                prompt_manifest=prompt_manifest,
                selector_config_path=_resolve(root, args.selector_config),
                output_dir=output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            payload = apply_constrained_selection(
                project_root=root,
                config=config,
                selector_config_path=_resolve(root, args.selector_config),
                semantic_dir=output_dir,
                output_dir=output_dir,
                frozen_selector_path=(
                    _resolve(root, args.frozen_selector) if args.frozen_selector else None
                ),
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
