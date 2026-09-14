from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.pitch3_lte_data import build_pitch3_lte_prompt_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen ACE prompt states for V3-LTE")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prompts", type=Path, default=Path("metadata/ltsn_prompts.csv"))
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--ace-config", type=Path, default=Path("configs/ace_rerank_180s.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-model-sha256", required=True)
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = build_pitch3_lte_prompt_embeddings(
        root=args.root,
        prompt_manifest_path=args.prompts,
        source_manifest_path=args.source_manifest,
        ace_config_path=args.ace_config,
        output_dir=args.output_dir,
        ace_model_sha256=args.ace_model_sha256,
        duration_seconds=args.duration_seconds,
        device_name=args.device,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
