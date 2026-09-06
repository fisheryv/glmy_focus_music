#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.noninferiority_metrics import (
    TransformersClapBackend,
    generate_development_noninferiority_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build frozen CLAP and blind-quality evidence for LTSN development pairs."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-id", default="laion/clap-htsat-fused")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Immutable 40-hex Hugging Face commit SHA; moving tags are rejected.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--quality-table", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    run_root = args.run_root if args.run_root.is_absolute() else root / args.run_root
    run_root = run_root.resolve()
    output = (
        args.output
        if args.output is not None
        else run_root / "development_noninferiority_metrics.csv"
    )
    output = output if output.is_absolute() else root / output
    audit_output = args.audit_output or output.with_suffix(".audit.json")
    audit_output = audit_output if audit_output.is_absolute() else root / audit_output
    quality_table = args.quality_table
    if quality_table is not None and not quality_table.is_absolute():
        quality_table = root / quality_table

    backend = TransformersClapBackend(args.model_id, args.model_revision, args.device)
    audit = generate_development_noninferiority_metrics(
        run_root=run_root,
        output_path=output.resolve(),
        audit_path=audit_output.resolve(),
        backend=backend,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        segment_seconds=args.segment_seconds,
        quality_table_path=None if quality_table is None else quality_table.resolve(),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output.resolve()),
                "audit_output": str(audit_output.resolve()),
                "rows": audit["output"]["rows"],
                "prompts": audit["output"]["prompts"],
                "output_sha256": audit["output"]["sha256"],
                "quality_columns_complete": audit["output"]["quality_columns_complete"],
                "next_step": (
                    "run development-finalize"
                    if audit["output"]["quality_columns_complete"]
                    else "obtain blinded quality scores and rerun with --quality-table"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
