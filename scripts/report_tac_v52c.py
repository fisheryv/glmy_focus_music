from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.tac_v52c import write_response_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = write_response_report(
        response_points_path=args.response_points.resolve(), output_path=args.output.resolve()
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
