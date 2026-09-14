from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_lte_evaluation import (  # noqa: E402
    finalize_pitch3_lte_quality,
    materialize_pitch3_lte_development_guidance,
    screen_pitch3_lte_development,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize or screen V3-LTE development guidance"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    guide = subparsers.add_parser("materialize-guidance")
    guide.add_argument("--root", type=Path, default=ROOT)
    guide.add_argument(
        "--fingerprint", type=Path, default=Path("metadata/focus_pitch3_fingerprint_v1.json")
    )
    guide.add_argument("--manifest", type=Path, required=True)
    guide.add_argument("--source-manifest", type=Path, required=True)
    guide.add_argument("--checkpoint", type=Path)
    guide.add_argument("--ensemble-manifest", type=Path)
    guide.add_argument("--checkpoint-sha256")
    guide.add_argument("--ace-config", type=Path, default=Path("configs/ace_rerank_180s.toml"))
    guide.add_argument("--prompts", type=Path, default=Path("metadata/ltsn_prompts.csv"))
    guide.add_argument("--output-dir", type=Path, required=True)
    guide.add_argument("--device", default="cuda:0")
    guide.add_argument("--workers", type=int, default=8)
    guide.add_argument("--exact-batch-size", type=int, default=32)
    guide.add_argument("--materialize-mode", default="auto")
    guide.add_argument("--discard-audio", action="store_true")
    screen = subparsers.add_parser("screen-development")
    screen.add_argument(
        "--fingerprint", type=Path, default=Path("metadata/focus_pitch3_fingerprint_v1.json")
    )
    screen.add_argument("--manifest", type=Path, required=True)
    screen.add_argument("--checkpoint", type=Path)
    screen.add_argument("--ensemble-manifest", type=Path)
    screen.add_argument("--checkpoint-sha256")
    screen.add_argument("--guidance-summary", type=Path)
    screen.add_argument("--quality-report", type=Path)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--device", default="cuda:0")
    quality = subparsers.add_parser("finalize-quality")
    quality.add_argument("--metrics", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)
    quality.add_argument("--bootstrap-resamples", type=int, default=2000)
    quality.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.stage == "materialize-guidance":
        result = materialize_pitch3_lte_development_guidance(
            root=args.root,
            fingerprint_path=args.fingerprint,
            dataset_manifest=args.manifest,
            source_manifest_path=args.source_manifest,
            checkpoint_path=args.checkpoint,
            ace_config_path=args.ace_config,
            prompt_manifest_path=args.prompts,
            output_dir=args.output_dir,
            checkpoint_sha256=args.checkpoint_sha256,
            ensemble_manifest_path=args.ensemble_manifest,
            device_name=args.device,
            workers=args.workers,
            exact_batch_size=args.exact_batch_size,
            materialize_mode=args.materialize_mode,
            retain_audio=not args.discard_audio,
        )
    elif args.stage == "screen-development":
        result = screen_pitch3_lte_development(
            fingerprint_path=args.fingerprint,
            dataset_manifest=args.manifest,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            checkpoint_sha256=args.checkpoint_sha256,
            ensemble_manifest_path=args.ensemble_manifest,
            guidance_summary_path=args.guidance_summary,
            quality_report_path=args.quality_report,
            device_name=args.device,
        )
    else:
        result = finalize_pitch3_lte_quality(
            metric_table_path=args.metrics,
            output_path=args.output,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
