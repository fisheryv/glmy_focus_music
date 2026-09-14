from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMPORT_ROOTS = (ROOT / "src", ROOT / "packages" / "pyglmy" / "src")
for local_root in reversed(LOCAL_IMPORT_ROOTS):
    if local_root.is_dir() and str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

from generation.pitch3_ood import build_pitch3_ood_augmentation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact-decoded Pitch-3 OOD data and merge it with ID labels"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument(
        "--ace-config", type=Path, default=Path("configs/ace_rerank_180s.toml")
    )
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ood-per-prompt", type=int, default=1)
    parser.add_argument("--evaluation-ood-per-prompt", type=int, default=1)
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
    args = parser.parse_args()
    result = build_pitch3_ood_augmentation(
        root=args.root,
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        ace_config_path=args.ace_config,
        fingerprint_path=args.fingerprint,
        output_dir=args.output_dir,
        ood_per_prompt=args.ood_per_prompt,
        evaluation_ood_per_prompt=args.evaluation_ood_per_prompt,
        duration_seconds=args.duration_seconds,
        workers=args.workers,
        exact_batch_size=args.exact_batch_size,
        materialize_mode=args.materialize_mode,
        cleanup_exact_batches=not args.keep_exact_batches,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
