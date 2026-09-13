from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v6_final_target import report_v6_final_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the V6 final-target diagnostic screen.")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--tac-target", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = report_v6_final_target(
        fingerprint_path=args.fingerprint,
        tac_target_path=args.tac_target,
        view_path=args.view,
        pair_manifest_path=args.pair_manifest,
        preparation_path=args.preparation,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
