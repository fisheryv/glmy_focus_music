"""Minimal dependency-free weighted directed graph types."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FiltrationDirection = Literal["superlevel", "sublevel"]


@dataclass(frozen=True, slots=True)
class WeightedEdge:
    """One weighted directed edge."""

    source: Hashable
    target: Hashable
    weight: float

    def __post_init__(self) -> None:
        try:
            hash(self.source)
            hash(self.target)
        except TypeError as exc:
            raise ValueError("edge endpoints must be hashable") from exc
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError("edge weight must be finite")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class WeightedDiGraph:
    """Immutable weighted digraph with deterministic vertex and edge order."""

    vertices: tuple[Hashable, ...]
    edges: tuple[WeightedEdge, ...]

    def __post_init__(self) -> None:
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("vertices must be unique")
        vertex_set = set(self.vertices)
        pairs: set[tuple[Hashable, Hashable]] = set()
        for vertex in self.vertices:
            try:
                hash(vertex)
            except TypeError as exc:
                raise ValueError("vertices must be hashable") from exc
        for edge in self.edges:
            if edge.source not in vertex_set or edge.target not in vertex_set:
                raise ValueError(
                    f"edge {(edge.source, edge.target)!r} references an unknown vertex"
                )
            pair = (edge.source, edge.target)
            if pair in pairs:
                raise ValueError(f"duplicate directed edge: {pair!r}")
            pairs.add(pair)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[
            WeightedEdge
            | tuple[Hashable, Hashable]
            | tuple[Hashable, Hashable, float]
        ],
        *,
        vertices: Iterable[Hashable] | None = None,
        default_weight: float = 1.0,
    ) -> WeightedDiGraph:
        """Construct a graph from ``(source, target[, weight])`` records."""

        materialized: list[WeightedEdge] = []
        observed: set[Hashable] = set()
        for raw in edges:
            if isinstance(raw, WeightedEdge):
                edge = raw
            else:
                values = tuple(raw)
                if len(values) == 2:
                    edge = WeightedEdge(values[0], values[1], default_weight)
                elif len(values) == 3:
                    edge = WeightedEdge(values[0], values[1], float(values[2]))
                else:
                    raise ValueError("each edge must contain two or three values")
            materialized.append(edge)
            observed.update((edge.source, edge.target))

        if vertices is None:
            ordered_vertices = tuple(sorted(observed, key=repr))
        else:
            ordered_vertices = tuple(vertices)
        ordered_edges = tuple(
            sorted(
                materialized,
                key=lambda edge: (repr(edge.source), repr(edge.target)),
            )
        )
        return cls(ordered_vertices, ordered_edges)

    @classmethod
    def from_adjacency(
        cls,
        adjacency: ArrayLike,
        *,
        vertices: Sequence[Hashable] | None = None,
        zero_is_missing: bool = True,
    ) -> WeightedDiGraph:
        """Construct a graph from a square weighted adjacency matrix."""

        matrix = np.asarray(adjacency, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("adjacency must contain only finite values")
        labels = tuple(range(matrix.shape[0])) if vertices is None else tuple(vertices)
        if len(labels) != matrix.shape[0]:
            raise ValueError("vertices length must match adjacency size")
        edges = [
            WeightedEdge(labels[row], labels[column], float(matrix[row, column]))
            for row in range(matrix.shape[0])
            for column in range(matrix.shape[1])
            if (not zero_is_missing or matrix[row, column] != 0.0)
        ]
        return cls.from_edges(edges, vertices=labels)

    @property
    def edge_pairs(self) -> tuple[tuple[Hashable, Hashable], ...]:
        """Regular directed edges; self-loops are excluded."""

        return tuple(
            (edge.source, edge.target)
            for edge in self.edges
            if edge.source != edge.target
        )

    def threshold(
        self,
        level: float,
        *,
        direction: FiltrationDirection = "superlevel",
    ) -> WeightedDiGraph:
        """Return the graph at one filtration level.

        ``superlevel`` keeps weights greater than or equal to ``level``;
        ``sublevel`` keeps weights less than or equal to it.
        """

        value = float(level)
        if not math.isfinite(value):
            raise ValueError("filtration level must be finite")
        if direction == "superlevel":
            kept = tuple(edge for edge in self.edges if edge.weight >= value)
        elif direction == "sublevel":
            kept = tuple(edge for edge in self.edges if edge.weight <= value)
        else:
            raise ValueError("direction must be 'superlevel' or 'sublevel'")
        return WeightedDiGraph(self.vertices, kept)

    def adjacency_matrix(
        self,
        *,
        fill_value: float = 0.0,
    ) -> NDArray[np.float64]:
        """Return a dense adjacency matrix in ``vertices`` order."""

        matrix = np.full(
            (len(self.vertices), len(self.vertices)),
            float(fill_value),
            dtype=np.float64,
        )
        indices = {vertex: index for index, vertex in enumerate(self.vertices)}
        for edge in self.edges:
            matrix[indices[edge.source], indices[edge.target]] = edge.weight
        return matrix
