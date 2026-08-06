"""Analyze one audio file with the optional audio dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

from focus_topology import analyze_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--view", choices=("pitch", "modulation"), default="pitch")
    parser.add_argument("--output", type=Path, default=Path("topology-result.json"))
    args = parser.parse_args()

    result = analyze_audio(args.audio, view=args.view)
    result.write_json(args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
