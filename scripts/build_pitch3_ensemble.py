from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_contract import load_pitch3_contract  # noqa: E402
from generation.pitch3_ensemble import build_pitch3_ensemble_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a weighted Pitch-3 control-head ensemble")
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--member",
        nargs=3,
        action="append",
        metavar=("NAME", "CHECKPOINT", "WEIGHT"),
        required=True,
    )
    parser.add_argument("--ensemble-id", default="ltch_pitch3_e2_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    members = [(name, Path(checkpoint), float(weight)) for name, checkpoint, weight in args.member]
    result = build_pitch3_ensemble_manifest(
        contract=load_pitch3_contract(args.fingerprint),
        training_manifest=args.manifest,
        members=members,
        output_path=args.output,
        ensemble_id=args.ensemble_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
