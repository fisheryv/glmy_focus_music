from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_evaluation import create_development_ood_ablation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build-ltsn-calibration-ablation")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = create_development_ood_ablation(
        calibration_path=args.calibration,
        output_path=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
