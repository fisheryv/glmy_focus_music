from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.tac_v52d import collect_repeatability


def _digest(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256 digest")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--v52c-plan", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-model-sha256", type=_digest, required=True)
    parser.add_argument("--vae-sha256", type=_digest, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--materialize-mode", choices=("auto", "reflink", "hardlink", "copy"), default="auto"
    )
    args = parser.parse_args(argv)
    payload = collect_repeatability(
        root=args.root,
        source_manifest_path=args.source_manifest,
        v52c_plan_path=args.v52c_plan,
        prompt_manifest_path=args.prompt_manifest,
        ace_config_path=args.ace_config,
        fingerprint_path=args.fingerprint,
        target_path=args.target,
        output_dir=args.output_dir,
        ace_model_sha256=args.ace_model_sha256,
        vae_sha256=args.vae_sha256,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        materialize_mode=args.materialize_mode,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
