from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_training_augmentation import build_ltsn_training_augmentation


def _digest(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256 digest")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-labelled train local/OOD augmentation plus held-out "
            "calibration/qualification OOD for LTSN V3/V4."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-model-sha256", type=_digest, required=True)
    parser.add_argument("--vae-sha256", type=_digest, required=True)
    parser.add_argument("--trajectories-per-prompt", type=int, default=1)
    parser.add_argument("--perturbations-per-anchor", type=int, default=2)
    parser.add_argument("--rms-ratio", type=float, default=0.005)
    parser.add_argument("--local-mode", choices=("random", "on_policy"), default="random")
    parser.add_argument("--on-policy-ensemble-manifest", type=Path)
    parser.add_argument(
        "--on-policy-rms-ratios",
        type=float,
        nargs="+",
        default=(0.0025, 0.005, 0.01),
    )
    parser.add_argument("--ood-per-prompt", type=int, default=1)
    parser.add_argument("--evaluation-ood-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026071600)
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exact-batch-size", type=int, default=256)
    parser.add_argument(
        "--materialize-mode",
        choices=("auto", "reflink", "hardlink", "copy"),
        default="auto",
    )
    parser.add_argument("--keep-exact-batches", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    payload = build_ltsn_training_augmentation(
        root=args.root,
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        ace_config_path=args.ace_config,
        fingerprint_path=args.fingerprint,
        output_dir=args.output_dir,
        ace_model_sha256=args.ace_model_sha256,
        vae_sha256=args.vae_sha256,
        trajectories_per_prompt=args.trajectories_per_prompt,
        perturbations_per_anchor=args.perturbations_per_anchor,
        rms_ratio=args.rms_ratio,
        ood_per_prompt=args.ood_per_prompt,
        evaluation_ood_per_prompt=args.evaluation_ood_per_prompt,
        seed=args.seed,
        duration_seconds=args.duration_seconds,
        workers=args.workers,
        exact_batch_size=args.exact_batch_size,
        materialize_mode=args.materialize_mode,
        cleanup_exact_batches=not args.keep_exact_batches,
        device_name=args.device,
        resume=args.resume,
        local_mode=args.local_mode,
        on_policy_ensemble_manifest=args.on_policy_ensemble_manifest,
        on_policy_rms_ratios=tuple(args.on_policy_rms_ratios),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
