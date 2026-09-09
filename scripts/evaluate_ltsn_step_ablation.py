from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_step_ablation import evaluate_step4_gradient_diagnostic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate-ltsn-step-ablation")
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    payload = evaluate_step4_gradient_diagnostic(
        pair_table=args.pair_table,
        generation_plan=args.generation_plan,
        output_path=args.output,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
