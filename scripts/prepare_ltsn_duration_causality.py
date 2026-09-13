from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_duration_causality import prepare_duration_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2026091200)
    args = parser.parse_args(argv)
    payload = prepare_duration_plan(
        prompt_manifest_path=args.prompt_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        protocol_path=args.protocol.resolve(),
        seed_start=args.seed_start,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
