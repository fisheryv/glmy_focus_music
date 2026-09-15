from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_lte_ensemble import (  # noqa: E402
    build_pitch3_lte_ensemble_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an equal-weight V3.3/V3.4 LTE ensemble")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_pitch3_lte_ensemble_manifest(
        manifest_paths=args.manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
