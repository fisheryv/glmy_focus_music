from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.tac_v52c import collect_v52c_response_curve


def _digest(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256 digest")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect the TAC V5.2c local response curve with ephemeral WAV batches."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--v52a-plan", type=Path, required=True)
    parser.add_argument("--v52a-master", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-model-sha256", type=_digest, required=True)
    parser.add_argument("--vae-sha256", type=_digest, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--materialize-mode", choices=("auto", "reflink", "hardlink", "copy"), default="auto"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    payload = collect_v52c_response_curve(
        root=args.root,
        source_manifest_path=args.source_manifest,
        v52a_plan_path=args.v52a_plan,
        v52a_master_path=args.v52a_master,
        ace_config_path=args.ace_config,
        fingerprint_path=args.fingerprint,
        target_path=args.target,
        output_dir=args.output_dir,
        ace_model_sha256=args.ace_model_sha256,
        vae_sha256=args.vae_sha256,
        workers=args.workers,
        batch_size=args.batch_size,
        materialize_mode=args.materialize_mode,
        device_name=args.device,
        resume=not args.no_resume,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
