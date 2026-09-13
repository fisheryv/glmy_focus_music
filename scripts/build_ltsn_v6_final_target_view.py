from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v6_final_target import prepare_v6_final_target_view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the no-new-audio V6 final-target diagnostic view."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--tac-target", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare_v6_final_target_view(
        root=args.root,
        fingerprint_path=args.fingerprint,
        tac_target_path=args.tac_target,
        source_manifest_path=args.source_manifest,
        pair_manifest_path=args.pair_manifest,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
