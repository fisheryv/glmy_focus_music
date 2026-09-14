from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pitch3_exact_scorer import ExactPitch3Scorer
from .pitch3_labeling import build_pitch3_label_tables
from .pitch3_training import train_pitch3_control_head


def labels_main() -> None:
    parser = argparse.ArgumentParser(description="Build exact Pitch-3 trajectory labels")
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--descriptor-table", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exact-label-table", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    result = build_pitch3_label_tables(
        trajectory_manifest=args.trajectory_manifest,
        descriptor_table=args.descriptor_table,
        output_manifest=args.output_manifest,
        exact_label_table=args.exact_label_table,
        split_manifest=args.split_manifest,
        scorer=ExactPitch3Scorer.from_json(args.fingerprint),
        engineering_smoke=args.engineering_smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the Pitch-3 latent control head")
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_pitch3_fingerprint_v1.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pitch3_control_head_training.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_pitch3_control_head(
        fingerprint_path=args.fingerprint,
        training_manifest=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
