from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v61_data import prepare_v61_pair_view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the no-new-audio V6.1 TAC pair view.")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--tac-target", type=Path, required=True)
    parser.add_argument("--v6-preparation", type=Path, required=True)
    parser.add_argument("--v6-view", type=Path, required=True)
    parser.add_argument("--v6-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--coordinate-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-anchors-per-step", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=20260914)
    args = parser.parse_args(argv)
    payload = prepare_v61_pair_view(
        fingerprint_path=args.fingerprint,
        tac_target_path=args.tac_target,
        v6_preparation_path=args.v6_preparation,
        v6_view_path=args.v6_view,
        v6_checkpoint_path=args.v6_checkpoint,
        pair_manifest_path=args.pair_manifest,
        coordinate_manifest_path=args.coordinate_manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        validation_anchors_per_step=args.validation_anchors_per_step,
        split_seed=args.split_seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
