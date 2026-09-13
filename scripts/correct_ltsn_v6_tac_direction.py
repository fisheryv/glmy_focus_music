from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v6_tac_correction import correct_v6_tac_direction_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct V6 held-out direction evidence to use exact TAC derivatives."
    )
    parser.add_argument("--legacy-report", type=Path, required=True)
    parser.add_argument("--legacy-outcomes", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--coordinate-manifest", type=Path, required=True)
    parser.add_argument("--tac-target", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-outcomes", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = correct_v6_tac_direction_evidence(
        legacy_report_path=args.legacy_report,
        legacy_outcomes_path=args.legacy_outcomes,
        pair_manifest_path=args.pair_manifest,
        coordinate_manifest_path=args.coordinate_manifest,
        tac_target_path=args.tac_target,
        output_report_path=args.output_report,
        output_outcomes_path=args.output_outcomes,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
