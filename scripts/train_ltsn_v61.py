from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v61_training import train_v61


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the no-new-audio V6.1 multitask screen.")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--tac-target", type=Path, required=True)
    parser.add_argument("--v6-preparation", type=Path, required=True)
    parser.add_argument("--v6-view", type=Path, required=True)
    parser.add_argument("--v6-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-view", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = train_v61(
        fingerprint_path=args.fingerprint,
        tac_target_path=args.tac_target,
        v6_preparation_path=args.v6_preparation,
        v6_view_path=args.v6_view,
        v6_checkpoint_path=args.v6_checkpoint,
        pair_view_path=args.pair_view,
        preparation_path=args.preparation,
        config_path=args.config,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
