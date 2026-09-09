from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v51a import build_v51a_overfit_view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the bounded V5.1a central-direction overfit diagnostic view."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--central-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step4-anchors", type=int, default=32)
    parser.add_argument("--step5-anchors", type=int, default=16)
    parser.add_argument("--step6-anchors", type=int, default=16)
    args = parser.parse_args(argv)
    payload = build_v51a_overfit_view(
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        central_evidence_path=args.central_evidence,
        output_dir=args.output_dir,
        step_quotas={
            4: args.step4_anchors,
            5: args.step5_anchors,
            6: args.step6_anchors,
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
