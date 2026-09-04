#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.noninferiority_metrics import (
    TransformersClapBackend,
    generate_noninferiority_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen CLAP prompt/diversity evidence for ACE reranking."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/ace_rerank/ace_rerank_18d_180s_v1"),
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=Path("generation/prompts/ace_rerank_formal.csv"),
    )
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
    return parser


def _under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    run_root = _under_root(root, args.run_root).resolve()
    prompt_manifest = _under_root(root, args.prompt_manifest).resolve()
    output = (
        _under_root(root, args.output).resolve()
        if args.output is not None
        else run_root / "noninferiority_metrics.csv"
    )
    audit_output = (
        _under_root(root, args.audit_output).resolve()
        if args.audit_output is not None
        else output.with_suffix(".audit.json")
    )
    quality_table = (
        _under_root(root, args.quality_table).resolve()
        if args.quality_table is not None
        else None
    )
    backend = TransformersClapBackend(args.model_id, args.model_revision, args.device)
    audit = generate_noninferiority_metrics(
        run_root=run_root,
        prompt_manifest_path=prompt_manifest,
        output_path=output,
        audit_path=audit_output,
        backend=backend,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        segment_seconds=args.segment_seconds,
        quality_table_path=quality_table,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "audit_output": str(audit_output),
                "rows": audit["output"]["rows"],
                "output_sha256": audit["output"]["sha256"],
                "quality_columns_complete": audit["output"]["quality_columns_complete"],
                "next_step": (
                    "run focus-ace-rerank evaluate-evidence"
                    if quality_table is not None
                    else "fill quality_baseline/quality_selected from blinded ratings first"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
