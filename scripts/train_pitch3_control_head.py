from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_training import train_pitch3_control_head  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen Pitch-3 latent control head")
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pitch3_control_head_training.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_pitch3_control_head(
        fingerprint_path=args.fingerprint,
        training_manifest=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
