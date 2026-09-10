from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v51c import build_v51c_ablation_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the V5.1c one-factor ablation suite.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--central-evidence", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_v51c_ablation_suite(
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        central_evidence_path=args.central_evidence,
        config_root=args.config_root,
        output_root=args.output_root,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
