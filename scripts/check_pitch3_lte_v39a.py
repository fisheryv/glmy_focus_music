"""Run synthetic PyTorch network checks before real V3.9-A training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pitch3_lte_v39a.toml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checks: {args.output_dir}")
    from generation.ltsn_pipeline import write_json_atomic
    from generation.pitch3_lte_v39a_checks import run_checks

    result = run_checks(args.config, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "pitch3_lte_v39a_checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
