"""Prepare/resume coordinate-verified frozen transition teachers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path)
    parser.add_argument("--trajectory-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from generation.pitch3_lte_transition_teacher import build_teacher

    result = build_teacher(
        root=args.root,
        dataset_manifest=args.manifest,
        fingerprint_path=args.fingerprint,
        output_dir=args.output_dir,
        ace_config=args.ace_config,
        trajectory_manifest=args.trajectory_manifest,
        device=args.device,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "samples": len(result["samples"]),
                "targets_sha256": result["targets_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
