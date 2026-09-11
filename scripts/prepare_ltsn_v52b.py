from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v52b import prepare_v52b_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the frozen V5.2b probe suite.")
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--views-summary", type=Path, required=True)
    parser.add_argument("--v52a-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--operational-rms", type=float, default=0.005)
    parser.add_argument("--control-seed", type=int, default=20260911)
    args = parser.parse_args(argv)
    payload = prepare_v52b_probe(
        master_manifest_path=args.master_manifest,
        evidence_path=args.evidence,
        views_summary_path=args.views_summary,
        v52a_report_path=args.v52a_report,
        output_dir=args.output_dir,
        operational_rms=args.operational_rms,
        control_seed=args.control_seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
