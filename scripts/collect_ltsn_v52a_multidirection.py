from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v52a import collect_v52a_multidirection


def _digest(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256 digest")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect and exact-score the V5.2a orthogonal multi-direction experiment."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-model-sha256", type=_digest, required=True)
    parser.add_argument("--vae-sha256", type=_digest, required=True)
    parser.add_argument("--train-anchors", type=int, default=32)
    parser.add_argument("--unseen-anchors", type=int, default=16)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--train-directions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260910)
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
    payload = collect_v52a_multidirection(
        root=args.root,
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        source_evidence_path=args.source_evidence,
        ace_config_path=args.ace_config,
        fingerprint_path=args.fingerprint,
        output_dir=args.output_dir,
        ace_model_sha256=args.ace_model_sha256,
        vae_sha256=args.vae_sha256,
        train_anchor_count=args.train_anchors,
        unseen_anchor_count=args.unseen_anchors,
        direction_count=args.directions,
        train_direction_count=args.train_directions,
        seed=args.seed,
        workers=args.workers,
        exact_batch_size=args.exact_batch_size,
        materialize_mode=args.materialize_mode,
        cleanup_exact_batches=not args.keep_exact_batches,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
