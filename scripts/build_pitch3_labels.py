from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_exact_scorer import ExactPitch3Scorer  # noqa: E402
from generation.pitch3_labeling import build_pitch3_label_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact Pitch-3 trajectory labels")
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--descriptor-table", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exact-label-table", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    result = build_pitch3_label_tables(
        trajectory_manifest=args.trajectory_manifest,
        descriptor_table=args.descriptor_table,
        output_manifest=args.output_manifest,
        exact_label_table=args.exact_label_table,
        split_manifest=args.split_manifest,
        scorer=ExactPitch3Scorer.from_json(args.fingerprint),
        engineering_smoke=args.engineering_smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
