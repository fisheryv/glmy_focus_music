from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_evaluation import (  # noqa: E402
    calibrate_pitch3_control_head,
    qualify_pitch3_control_head,
    screen_pitch3_development,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate or independently qualify the Pitch-3 latent control head"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    development = subparsers.add_parser(
        "screen-development",
        help="screen one checkpoint on development before calibration",
    )
    _common(development)
    calibration = subparsers.add_parser("calibrate", help="freeze calibration-split thresholds")
    _common(calibration)
    calibration.add_argument("--development-screen", type=Path, required=True)
    qualification = subparsers.add_parser(
        "qualify", help="evaluate the untouched qualification split once"
    )
    _common(qualification)
    qualification.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "fingerprint_path": args.fingerprint,
        "training_manifest": args.manifest,
        "checkpoint_path": args.checkpoint,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "device_name": args.device,
        "expected_checkpoint_sha256": args.checkpoint_sha256,
    }
    if args.stage == "screen-development":
        result = screen_pitch3_development(**common)
    elif args.stage == "calibrate":
        result = calibrate_pitch3_control_head(
            development_screen_path=args.development_screen,
            **common,
        )
    else:
        result = qualify_pitch3_control_head(calibration_path=args.calibration, **common)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
