"""Compatibility adapter to the standalone :mod:`pyglmy` package."""

from __future__ import annotations

from collections.abc import Iterable

from pyglmy import (
    HomologyGroup,
    PathComplex,
    PathHomology,
    PathHomologyResult,
    PersistenceInterval,
    PersistentPathResult,
    WeightedDiGraph,
    build_path_complex,
    compute_path_homology,
    enumerate_allowed_paths,
    path_homology,
)
from pyglmy import filtration_descriptors as _filtration_descriptors
from pyglmy import persistent_path_homology as _persistent_path_homology

from graphs.transition import TransitionGraph


def _weighted_graph(graph: TransitionGraph) -> WeightedDiGraph:
    return WeightedDiGraph.from_edges(
        [
            (edge.source, edge.target, edge.weight)
            for edge in graph.edges
        ],
        vertices=graph.vertices,
    )


def filtration_descriptors(
    graph: TransitionGraph,
    thresholds: Iterable[float],
    *,
    max_dimension: int = 1,
    tolerance: float = 1e-9,
) -> list[dict[str, int | float]]:
    """Compute fixed-threshold descriptors with the standalone backend."""

    return _filtration_descriptors(
        _weighted_graph(graph),
        thresholds,
        max_dimension=max_dimension,
        tolerance=tolerance,
        direction="superlevel",
    )


def persistent_path_homology(
    graph: TransitionGraph,
    thresholds: Iterable[float],
    *,
    tolerance: float = 1e-9,
    max_dimension: int = 1,
) -> PersistentPathResult:
    """Compute persistent path homology with the standalone backend."""

    return _persistent_path_homology(
        _weighted_graph(graph),
        thresholds,
        max_dimension=max_dimension,
        tolerance=tolerance,
        direction="superlevel",
    )


__all__ = [
    "HomologyGroup",
    "PathComplex",
    "PathHomology",
    "PathHomologyResult",
    "PersistenceInterval",
    "PersistentPathResult",
    "build_path_complex",
    "compute_path_homology",
    "enumerate_allowed_paths",
    "filtration_descriptors",
    "path_homology",
    "persistent_path_homology",
]
