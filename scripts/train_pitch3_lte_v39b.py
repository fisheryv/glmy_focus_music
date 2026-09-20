"""Run one predeclared next-stage LTE experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["budget", "distill-diagnose", "cv"], required=True)
    parser.add_argument("--variant", choices=["baseline", "transition", "distill"], required=True)
    parser.add_argument("--cv-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20260941)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from generation.pitch3_lte_v39b import train_v39b

    result = train_v39b(
        fingerprint_path=args.fingerprint,
        dataset_manifest=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        mode=args.mode,
        variant=args.variant,
        cv_fold=args.cv_fold,
        seed=args.seed,
        device_name=args.device,
        teacher_manifest=args.teacher_manifest,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "completed",
                    "mode",
                    "variant",
                    "epochs_completed",
                    "train_metrics",
                    "selection_metrics",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
