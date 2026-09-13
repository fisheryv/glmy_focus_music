from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_duration_causality import (
    build_duration_reference_descriptors,
    build_duration_topology_contract,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, choices=(30, 60, 180), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--materialize-mode",
        choices=("auto", "reflink", "hardlink", "copy"),
        default="auto",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    descriptor_path = output_dir / "reference_descriptors.csv"
    extraction = build_duration_reference_descriptors(
        project_root=root,
        preprocess_manifest_path=args.preprocess_manifest.resolve(),
        work_dir=output_dir / "exact_work",
        output_path=descriptor_path,
        duration_seconds=args.duration_seconds,
        workers=args.workers,
        materialize_mode=args.materialize_mode,
    )
    contract = build_duration_topology_contract(
        root=root,
        descriptor_path=descriptor_path,
        output_dir=output_dir,
        duration_seconds=args.duration_seconds,
    )
    print(
        json.dumps(
            {"extraction": extraction, "contract": contract},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
