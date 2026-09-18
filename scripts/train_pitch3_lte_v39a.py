"""Fit one fixed-budget V3.9-A baseline/transition experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fingerprint", type=Path, default=ROOT / "metadata/focus_pitch3_fingerprint_v1.json"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pitch3_lte_v39a.toml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "transition"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cv-fold", type=int, choices=range(5))
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    from generation.pitch3_lte_v39a import train_v39a

    result = train_v39a(
        fingerprint_path=args.fingerprint,
        dataset_manifest=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        variant=args.variant,
        device_name=args.device,
        seed=args.seed,
        cv_fold=args.cv_fold,
        diagnostic=args.diagnostic,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "checkpoint",
                    "checkpoint_sha256",
                    "experiment_contract",
                    "completed",
                    "final_fit_metrics",
                )
                if k in result
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
