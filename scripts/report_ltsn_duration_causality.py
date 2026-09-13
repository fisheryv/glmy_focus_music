from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_duration_causality import build_duration_causality_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-180", type=Path, required=True)
    parser.add_argument("--report-60", type=Path, required=True)
    parser.add_argument("--report-30", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args(argv)
    payload = build_duration_causality_report(
        duration_report_paths={
            180: args.report_180.resolve(),
            60: args.report_60.resolve(),
            30: args.report_30.resolve(),
        },
        output_path=args.output.resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
