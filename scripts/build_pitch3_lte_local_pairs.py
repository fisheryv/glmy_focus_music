from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_lte_data import (  # noqa: E402
    build_pitch3_lte_dataset,
    build_pitch3_lte_dataset_multigpu,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact V3-LTE local +/- pairs")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prompt-embeddings", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path, default=Path("configs/ace_rerank_180s.toml"))
    parser.add_argument(
        "--fingerprint", type=Path, default=Path("metadata/focus_pitch3_fingerprint_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-splits",
        nargs="+",
        choices=("train", "development"),
        default=("train", "development"),
    )
    parser.add_argument(
        "--local-splits",
        nargs="+",
        choices=("train", "development"),
        default=("train", "development"),
    )
    parser.add_argument("--workers", type=int, default=8, help="single-GPU exact workers")
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=4,
        help="multi-GPU exact workers started inside each isolated shard process",
    )
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--materialize-mode", default="auto")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", default=None)
    devices.add_argument("--devices", nargs="+", help="e.g. cuda:0 cuda:1 cuda:2")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    common = {
        "root": args.root,
        "source_manifest_path": args.source_manifest,
        "prompt_embedding_manifest_path": args.prompt_embeddings,
        "ace_config_path": args.ace_config,
        "fingerprint_path": args.fingerprint,
        "output_dir": args.output_dir,
        "include_splits": args.include_splits,
        "local_splits": args.local_splits,
        "exact_batch_size": args.exact_batch_size,
        "materialize_mode": args.materialize_mode,
        "resume": not args.no_resume,
    }
    if args.devices:
        result = build_pitch3_lte_dataset_multigpu(
            **common,
            devices=args.devices,
            workers_per_device=args.workers_per_device,
        )
    else:
        result = build_pitch3_lte_dataset(
            **common,
            workers=args.workers,
            device_name=args.device or "cuda:0",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
