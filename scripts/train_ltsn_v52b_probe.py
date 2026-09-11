from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.ltsn_v52b import V52B_TARGET_MODES, V52B_VARIANTS
from generation.ltsn_v52b_training import train_v52b_ensemble


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one V5.2b diagnostic probe ensemble.")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=V52B_VARIANTS, required=True)
    parser.add_argument("--target-mode", choices=V52B_TARGET_MODES, required=True)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device")
    devices.add_argument("--devices", nargs="+")
    args = parser.parse_args(argv)
    payload = train_v52b_ensemble(
        fingerprint_path=args.fingerprint,
        pair_manifest_path=args.pair_manifest,
        preparation_path=args.preparation,
        config_path=args.config,
        output_dir=args.output_dir,
        variant=args.variant,
        target_mode=args.target_mode,
        device_name=args.device,
        device_names=args.devices,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
