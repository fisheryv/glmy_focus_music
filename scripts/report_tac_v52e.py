from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import write_csv_atomic, write_json_atomic
from generation.tac_v52d import _read_csv
from generation.tac_v52e import analyze_actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeatability-report", type=Path, required=True)
    parser.add_argument("--action-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    repeatability = json.loads(args.repeatability_report.read_text(encoding="utf-8"))
    report, outcomes = analyze_actions(
        _read_csv(args.action_points), repeatability_report=repeatability
    )
    outcomes_path = args.output.with_name("v52e_action_outcomes.csv")
    write_csv_atomic(outcomes_path, outcomes)
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        expected = previous.get("points_sha256")
        if expected is not None and expected != sha256_file(args.action_points):
            raise LTSNContractError("V5.2e action points changed after collection")
        expected_repeatability = previous.get("v52d_repeatability_report_sha256")
        if expected_repeatability is not None and expected_repeatability != sha256_file(
            args.repeatability_report
        ):
            raise LTSNContractError("V5.2e repeatability report changed after collection")
        report = {**previous, **report}
    report["points_sha256"] = sha256_file(args.action_points)
    report["outcomes_sha256"] = sha256_file(outcomes_path)
    report["v52d_repeatability_report_sha256"] = sha256_file(args.repeatability_report)
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
