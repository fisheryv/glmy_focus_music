"""Run actual PyTorch synthetic checks before next-stage training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite synthetic checks")
    from generation.ltsn_pipeline import write_json_atomic
    from generation.pitch3_lte_v39b_checks import run_checks

    result = run_checks(args.config, args.device)
    write_json_atomic(args.output_dir / "pitch3_lte_v39b_checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
