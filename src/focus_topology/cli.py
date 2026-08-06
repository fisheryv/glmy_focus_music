"""Command-line interface for the public library and legacy research helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .analysis import DEFAULT_THRESHOLDS, AnalysisConfig, analyze_states
from .audio import analyze_audio
from .data.manifest import validate_metadata
from .features.extractor import FeatureExtractionError
from .homology.glmy import compute_path_homology
from .pipeline import analyze_state_sequence


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _thresholds(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc


def _load_states(path: Path) -> list[Hashable | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("state JSON must be a top-level list")
    for index, state in enumerate(raw):
        if state is None:
            continue
        try:
            hash(state)
        except TypeError as exc:
            raise ValueError(f"state at index {index} is not a JSON scalar") from exc
    return raw


def _config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        thresholds=args.thresholds,
        top_k=args.top_k,
        include_self_loops=args.include_self_loops,
        tolerance=args.tolerance,
    )


def command_doctor(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    optional = ["librosa", "soundfile", "scipy", "pandas", "sklearn"]
    payload = {
        "package_version": __version__,
        "python": platform.python_version(),
        "root": str(root),
        "core_ready": importlib.util.find_spec("numpy") is not None,
        "audio_ready": all(
            importlib.util.find_spec(name) is not None
            for name in ("librosa", "soundfile", "scipy")
        ),
        "optional_dependencies": {
            name: importlib.util.find_spec(name) is not None for name in optional
        },
        "research_workspace": {
            name: (root / relative).exists()
            for name, relative in {
                "config": "configs/pipeline.toml",
                "metadata": "metadata/track_index.csv",
                "ace_step": "ACE-Step-1.5/pyproject.toml",
            }.items()
        },
    }
    _print_json(payload)
    return 0 if payload["core_ready"] else 1


def command_validate_manifest(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    report = validate_metadata(
        root / "metadata",
        root / "data_raw",
        check_files=args.check_files,
    )
    _print_json(
        {
            "ok": report.ok,
            "counts": report.counts,
            "warnings": report.warnings,
            "errors": report.errors,
        }
    )
    return 0 if report.ok else 1


def command_demo(_: argparse.Namespace) -> int:
    cycle = analyze_states(
        [0, 1, 2, 0, 1, 2, 0],
        config=AnalysisConfig(thresholds=(0.95, 0.5)),
        metadata={"name": "directed-cycle-demo"},
    )
    filled_triangle = compute_path_homology(
        [0, 1, 2],
        [(0, 1), (1, 2), (0, 2)],
        max_dimension=1,
    )
    _print_json(
        {
            "cycle_betti": {
                "h0": cycle.betti_curve(0),
                "h1": cycle.betti_curve(1),
            },
            "filled_triangle_betti": [group.betti for group in filled_triangle],
        }
    )
    return 0


def command_homology(args: argparse.Namespace) -> int:
    descriptors = analyze_state_sequence(
        _load_states(args.states),
        thresholds=args.thresholds,
        top_k=args.top_k,
        max_dimension=args.max_dimension,
    )
    serialized = json.dumps(descriptors, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


def command_states(args: argparse.Namespace) -> int:
    result = analyze_states(
        _load_states(args.states),
        config=_config(args),
        metadata={"source": str(args.states.resolve())},
    )
    if args.output:
        result.write_json(args.output, include_states=not args.omit_states)
    else:
        print(result.to_json(include_states=not args.omit_states))
    return 0


def command_audio(args: argparse.Namespace) -> int:
    result = analyze_audio(
        args.audio,
        view=args.view,
        track_id=args.track_id,
        config=_config(args),
    )
    if args.output:
        result.write_json(args.output, include_states=not args.omit_states)
    else:
        print(result.to_json(include_states=not args.omit_states))
    return 0


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thresholds",
        type=_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="comma-separated transition-probability thresholds",
    )
    parser.add_argument("--top-k", type=int, default=6, help="maximum outgoing edges per state")
    parser.add_argument("--include-self-loops", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, help="write result JSON to this path")
    parser.add_argument(
        "--omit-states",
        action="store_true",
        help="exclude the original state sequence from result JSON",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="focus-topology",
        description="Directed persistent path homology for musical state sequences",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    states = subparsers.add_parser("states", help="analyze a JSON state sequence")
    states.add_argument("states", type=Path)
    _add_analysis_options(states)
    states.set_defaults(handler=command_states)

    audio = subparsers.add_parser("audio", help="extract states and analyze one audio file")
    audio.add_argument("audio", type=Path)
    audio.add_argument(
        "--view", choices=("pitch", "modulation", "structure"), default="pitch"
    )
    audio.add_argument("--track-id")
    _add_analysis_options(audio)
    audio.set_defaults(handler=command_audio)

    homology = subparsers.add_parser(
        "homology",
        help="legacy descriptor-only analysis of a JSON state sequence",
    )
    homology.add_argument("states", type=Path)
    homology.add_argument(
        "--thresholds",
        type=_thresholds,
        default=(0.5, 0.6, 0.7, 0.8, 0.9),
    )
    homology.add_argument("--top-k", type=int, default=6)
    homology.add_argument("--max-dimension", type=int, default=1)
    homology.add_argument("--output", type=Path)
    homology.set_defaults(handler=command_homology)

    demo = subparsers.add_parser("demo", help="run a small topology smoke check")
    demo.set_defaults(handler=command_demo)

    doctor = subparsers.add_parser("doctor", help="inspect installed capabilities")
    doctor.add_argument("--root", type=Path, default=Path.cwd())
    doctor.set_defaults(handler=command_doctor)

    validate = subparsers.add_parser(
        "validate-manifest",
        help="validate this repository's research metadata",
    )
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--check-files", action="store_true")
    validate.set_defaults(handler=command_validate_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FeatureExtractionError, ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
