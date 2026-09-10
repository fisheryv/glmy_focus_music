from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v51b import (
    DEFAULT_DEVELOPMENT_ANCHORS,
    DEFAULT_RUNGS,
    build_v51b_memorization_ladder,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the nested deterministic V5.1b memorization ladder."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--central-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rungs", type=int, nargs="+", default=list(DEFAULT_RUNGS))
    parser.add_argument("--development-anchors", type=int, default=DEFAULT_DEVELOPMENT_ANCHORS)
    args = parser.parse_args(argv)
    payload = build_v51b_memorization_ladder(
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        central_evidence_path=args.central_evidence,
        output_root=args.output_root,
        rungs=args.rungs,
        development_anchor_count=args.development_anchors,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
