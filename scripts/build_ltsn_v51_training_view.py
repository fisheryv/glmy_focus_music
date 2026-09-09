from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v51 import build_v51_training_view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a stable-direction V5.1 training view without decoding new audio."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--central-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-abs-loss-separation", type=float, default=1e-5)
    args = parser.parse_args(argv)
    payload = build_v51_training_view(
        source_manifest_path=args.source_manifest,
        source_split_manifest_path=args.source_split_manifest,
        central_evidence_path=args.central_evidence,
        output_dir=args.output_dir,
        minimum_abs_loss_separation=args.minimum_abs_loss_separation,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
