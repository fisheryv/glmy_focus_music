from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_development_pairs import (
    finalize_development_pairs,
    generate_development_pairs,
    score_development_pairs,
)


def _digest(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256 digest")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build-ltsn-development-pairs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--root", type=Path, default=Path.cwd())
    generate.add_argument("--ace-config", type=Path, required=True)
    generate.add_argument("--prompt-manifest", type=Path, required=True)
    generate.add_argument("--fingerprint", type=Path, required=True)
    generate.add_argument("--ensemble-manifest", type=Path, required=True)
    generate.add_argument("--calibration", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--ace-model-sha256", type=_digest, required=True)
    generate.add_argument("--vae-sha256", type=_digest, required=True)
    generate.add_argument("--seed-start", type=int, default=2026071600)
    generate.add_argument("--seeds-per-prompt", type=int, default=4)
    generate.add_argument("--expected-development-prompts", type=int, default=64)
    generate.add_argument("--duration-seconds", type=float, default=180.0)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument("--resume", action="store_true")

    score = subparsers.add_parser("score")
    score.add_argument("--root", type=Path, default=Path.cwd())
    score.add_argument("--ace-config", type=Path, required=True)
    score.add_argument("--fingerprint", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--workers", type=int, default=4)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--raw-pair-table", type=Path, required=True)
    finalize.add_argument("--evidence-table", type=Path, required=True)
    finalize.add_argument("--protocol", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--bootstrap-resamples", type=int, default=2000)
    finalize.add_argument("--seed", type=int, default=20260716)

    args = parser.parse_args(argv)
    if args.command == "generate":
        payload = generate_development_pairs(
            root=args.root,
            ace_config=args.ace_config,
            prompt_manifest=args.prompt_manifest,
            fingerprint_path=args.fingerprint,
            ensemble_manifest=args.ensemble_manifest,
            calibration_path=args.calibration,
            output_dir=args.output_dir,
            ace_model_sha256=args.ace_model_sha256,
            vae_sha256=args.vae_sha256,
            seed_start=args.seed_start,
            seeds_per_prompt=args.seeds_per_prompt,
            expected_development_prompts=args.expected_development_prompts,
            duration_seconds=args.duration_seconds,
            device_name=args.device,
            resume=args.resume,
        )
    elif args.command == "score":
        payload = score_development_pairs(
            root=args.root,
            ace_config=args.ace_config,
            fingerprint_path=args.fingerprint,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    else:
        payload = finalize_development_pairs(
            raw_pair_table=args.raw_pair_table,
            evidence_table=args.evidence_table,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
