"""Small command-line interface for graph and point-cloud files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .graph import WeightedDiGraph
from .path_homology import path_homology, persistent_path_homology
from .tda import vietoris_rips


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _write(payload: Any, output: Path | None) -> None:
    text = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


def _load_graph(path: Path) -> WeightedDiGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "edges" not in payload:
        raise ValueError("graph JSON must be an object containing 'edges'")
    return WeightedDiGraph.from_edges(
        payload["edges"],
        vertices=payload.get("vertices"),
    )


def _levels(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(value.strip()) for value in raw.split(",") if value.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("levels must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one level is required")
    return values


def command_path(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    result = path_homology(
        graph.vertices,
        graph.edge_pairs,
        max_dimension=args.max_dimension,
        tolerance=args.tolerance,
    )
    _write(
        {
            "betti_numbers": result.betti_numbers,
            "groups": [asdict(group) for group in result.groups],
            "allowed_path_counts": {
                str(dimension): len(paths)
                for dimension, paths in result.complex.allowed_paths.items()
            },
        },
        args.output,
    )
    return 0


def command_pph(args: argparse.Namespace) -> int:
    graph = _load_graph(args.graph)
    result = persistent_path_homology(
        graph,
        args.levels,
        max_dimension=args.max_dimension,
        tolerance=args.tolerance,
        direction=args.direction,
    )
    _write(
        {
            "levels": result.thresholds,
            "direction": result.direction,
            "descriptors": result.descriptors,
            "intervals": [asdict(interval) for interval in result.intervals],
            "rank_invariants": [
                matrix.tolist() for matrix in result.rank_invariants
            ],
        },
        args.output,
    )
    return 0


def command_rips(args: argparse.Namespace) -> int:
    values = np.loadtxt(args.points, delimiter=args.delimiter, ndmin=2)
    result = vietoris_rips(
        values,
        max_dimension=args.max_dimension,
        coefficient=args.coefficient,
        distance_matrix=args.distance_matrix,
        max_points=args.max_points,
        normalize=args.normalize,
    )
    _write(
        {
            "point_count": result.point_count,
            "distance_scale": result.distance_scale,
            "coefficient": result.coefficient,
            "diagrams": [diagram.tolist() for diagram in result.diagrams],
        },
        args.output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathhom-tda",
        description="Path homology and Vietoris-Rips TDA",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    path = subparsers.add_parser("path", help="compute path homology of graph JSON")
    path.add_argument("graph", type=Path)
    path.add_argument("--max-dimension", type=int, default=1)
    path.add_argument("--tolerance", type=float, default=1e-9)
    path.add_argument("--output", type=Path)
    path.set_defaults(handler=command_path)

    pph = subparsers.add_parser(
        "pph",
        help="compute persistent path homology of weighted graph JSON",
    )
    pph.add_argument("graph", type=Path)
    pph.add_argument("--levels", type=_levels, required=True)
    pph.add_argument(
        "--direction",
        choices=("superlevel", "sublevel"),
        default="superlevel",
    )
    pph.add_argument("--max-dimension", type=int, default=1)
    pph.add_argument("--tolerance", type=float, default=1e-9)
    pph.add_argument("--output", type=Path)
    pph.set_defaults(handler=command_pph)

    rips = subparsers.add_parser(
        "rips",
        help="compute Vietoris-Rips persistence from CSV",
    )
    rips.add_argument("points", type=Path)
    rips.add_argument("--delimiter", default=",")
    rips.add_argument("--distance-matrix", action="store_true")
    rips.add_argument("--max-dimension", type=int, default=1)
    rips.add_argument("--coefficient", type=int, default=2)
    rips.add_argument("--max-points", type=int)
    rips.add_argument("--normalize", action="store_true")
    rips.add_argument("--output", type=Path)
    rips.set_defaults(handler=command_rips)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
