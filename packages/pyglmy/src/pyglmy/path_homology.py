"""GLMY path homology and finite persistent path homology."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .graph import FiltrationDirection, WeightedDiGraph

Path = tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class HomologyGroup:
    """Numerical summary of one path-homology group."""

    dimension: int
    betti: int
    allowed_path_count: int
    omega_dimension: int
    cycle_dimension: int
    boundary_rank: int


@dataclass(frozen=True, slots=True)
class PathComplex:
    """Allowed paths and GLMY chain-space bases through a maximum dimension."""

    allowed_paths: Mapping[int, tuple[Path, ...]]
    omega_bases: Mapping[int, NDArray[np.float64]]
    boundary_matrices: Mapping[int, NDArray[np.float64]]
    ambient_boundary_matrices: Mapping[int, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class PathHomologyResult:
    """Path-homology groups together with the computed chain complex."""

    groups: tuple[HomologyGroup, ...]
    complex: PathComplex

    @property
    def betti_numbers(self) -> tuple[int, ...]:
        return tuple(group.betti for group in self.groups)

    def group(self, dimension: int) -> HomologyGroup:
        if not 0 <= dimension < len(self.groups):
            raise ValueError(f"dimension must be between 0 and {len(self.groups) - 1}")
        return self.groups[dimension]


@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    """One interval in a finite path-homology filtration."""

    dimension: int
    birth_index: int
    death_index: int | None
    birth_threshold: float
    death_threshold: float | None
    lifetime: float
    multiplicity: int
    censored: bool


@dataclass(frozen=True, slots=True)
class PersistentPathResult:
    """Persistent path homology over a finite nested graph filtration."""

    thresholds: tuple[float, ...]
    descriptors: tuple[dict[str, int | float], ...]
    intervals: tuple[PersistenceInterval, ...]
    rank_invariants: tuple[NDArray[np.int64], ...]
    direction: FiltrationDirection = "superlevel"

    def betti_curve(self, dimension: int) -> tuple[int, ...]:
        if not 0 <= dimension < len(self.rank_invariants):
            raise ValueError(
                f"dimension must be between 0 and {len(self.rank_invariants) - 1}"
            )
        return tuple(
            int(row[f"h{dimension}_betti"]) for row in self.descriptors
        )

    def rank_invariant(self, dimension: int) -> NDArray[np.int64]:
        if not 0 <= dimension < len(self.rank_invariants):
            raise ValueError(
                f"dimension must be between 0 and {len(self.rank_invariants) - 1}"
            )
        return self.rank_invariants[dimension]

    @property
    def h0_rank_invariant(self) -> NDArray[np.int64]:
        return self.rank_invariant(0)

    @property
    def h1_rank_invariant(self) -> NDArray[np.int64]:
        return self.rank_invariant(1)


@dataclass(frozen=True, slots=True)
class _HomologySubspace:
    cycle_basis: NDArray[np.float64]
    boundary_basis: NDArray[np.float64]


class PathHomology:
    """Reusable object-oriented façade around the functional API."""

    def __init__(
        self,
        *,
        max_dimension: int = 1,
        tolerance: float = 1e-9,
    ) -> None:
        if max_dimension < 0:
            raise ValueError("max_dimension cannot be negative")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.max_dimension = max_dimension
        self.tolerance = tolerance

    def compute(
        self,
        vertices: Sequence[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
    ) -> PathHomologyResult:
        return path_homology(
            vertices,
            edges,
            max_dimension=self.max_dimension,
            tolerance=self.tolerance,
        )

    def persistent(
        self,
        graph: WeightedDiGraph,
        thresholds: Iterable[float],
        *,
        direction: FiltrationDirection = "superlevel",
    ) -> PersistentPathResult:
        return persistent_path_homology(
            graph,
            thresholds,
            max_dimension=self.max_dimension,
            tolerance=self.tolerance,
            direction=direction,
        )


def _matrix_rank(matrix: NDArray[np.float64], tolerance: float) -> int:
    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance))


def _null_space(
    matrix: NDArray[np.float64],
    tolerance: float,
) -> NDArray[np.float64]:
    column_count = matrix.shape[1]
    if column_count == 0:
        return np.zeros((0, 0), dtype=float)
    if matrix.shape[0] == 0:
        return np.eye(column_count, dtype=float)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return vh[rank:].T.copy()


def _is_regular(path: Path) -> bool:
    return all(left != right for left, right in zip(path, path[1:], strict=False))


def enumerate_allowed_paths(
    vertices: Sequence[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    max_dimension: int,
) -> dict[int, list[Path]]:
    """Enumerate regular allowed elementary paths through ``max_dimension``."""

    if max_dimension < 0:
        raise ValueError("max_dimension cannot be negative")
    vertex_set = set(vertices)
    if len(vertex_set) != len(vertices):
        raise ValueError("vertices must be unique")
    adjacency: dict[Hashable, list[Hashable]] = defaultdict(list)
    for source, target in set(edges):
        if source not in vertex_set or target not in vertex_set:
            raise ValueError(f"edge {(source, target)!r} references an unknown vertex")
        if source != target:
            adjacency[source].append(target)
    for source in adjacency:
        adjacency[source].sort(key=repr)

    paths: dict[int, list[Path]] = {
        0: [(vertex,) for vertex in sorted(vertex_set, key=repr)]
    }
    for dimension in range(1, max_dimension + 1):
        current: list[Path] = []
        for prefix in paths[dimension - 1]:
            current.extend(prefix + (target,) for target in adjacency.get(prefix[-1], []))
        paths[dimension] = current
    return paths


def _boundary_components(
    paths_p: Sequence[Path],
    paths_pm1: Sequence[Path],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    allowed_index = {path: index for index, path in enumerate(paths_pm1)}
    allowed = np.zeros((len(paths_pm1), len(paths_p)), dtype=float)
    nonallowed_rows: dict[Path, dict[int, float]] = defaultdict(dict)

    for column, path in enumerate(paths_p):
        for deleted_index in range(len(path)):
            face = path[:deleted_index] + path[deleted_index + 1 :]
            if not _is_regular(face):
                continue
            coefficient = float((-1) ** deleted_index)
            allowed_row = allowed_index.get(face)
            if allowed_row is not None:
                allowed[allowed_row, column] += coefficient
            else:
                nonallowed_rows[face][column] = (
                    nonallowed_rows[face].get(column, 0.0) + coefficient
                )

    nonallowed = np.zeros((len(nonallowed_rows), len(paths_p)), dtype=float)
    for row, face in enumerate(sorted(nonallowed_rows, key=repr)):
        for column, coefficient in nonallowed_rows[face].items():
            nonallowed[row, column] = coefficient
    return allowed, nonallowed


def build_path_complex(
    vertices: Sequence[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    max_dimension: int,
    tolerance: float = 1e-9,
) -> PathComplex:
    """Build the real-coefficient GLMY chain complex."""

    if max_dimension < 0:
        raise ValueError("max_dimension cannot be negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    paths = enumerate_allowed_paths(
        vertices,
        edges,
        max_dimension=max_dimension,
    )
    omega_bases: dict[int, NDArray[np.float64]] = {
        0: np.eye(len(paths[0]), dtype=float)
    }
    boundary_matrices: dict[int, NDArray[np.float64]] = {
        0: np.zeros((0, len(paths[0])), dtype=float)
    }
    ambient_boundary_matrices: dict[int, NDArray[np.float64]] = {
        0: np.zeros((0, len(paths[0])), dtype=float)
    }
    for dimension in range(1, max_dimension + 1):
        allowed, nonallowed = _boundary_components(
            paths[dimension],
            paths[dimension - 1],
        )
        omega_bases[dimension] = _null_space(nonallowed, tolerance)
        ambient_boundary = allowed @ omega_bases[dimension]
        ambient_boundary_matrices[dimension] = ambient_boundary
        boundary_matrices[dimension] = (
            omega_bases[dimension - 1].T @ ambient_boundary
        )
    return PathComplex(
        allowed_paths={
            dimension: tuple(values) for dimension, values in paths.items()
        },
        omega_bases=omega_bases,
        boundary_matrices=boundary_matrices,
        ambient_boundary_matrices=ambient_boundary_matrices,
    )


def path_homology(
    vertices: Sequence[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    max_dimension: int = 1,
    tolerance: float = 1e-9,
) -> PathHomologyResult:
    """Compute real-coefficient GLMY path homology."""

    edge_pairs = tuple(edges)
    complex_ = build_path_complex(
        vertices,
        edge_pairs,
        max_dimension=max_dimension + 1,
        tolerance=tolerance,
    )
    groups: list[HomologyGroup] = []
    for dimension in range(max_dimension + 1):
        boundary = complex_.boundary_matrices[dimension]
        omega_dimension = complex_.omega_bases[dimension].shape[1]
        outgoing_rank = _matrix_rank(boundary, tolerance)
        cycle_dimension = omega_dimension - outgoing_rank
        incoming_rank = _matrix_rank(
            complex_.boundary_matrices[dimension + 1],
            tolerance,
        )
        betti = cycle_dimension - incoming_rank
        if betti < 0:
            raise ArithmeticError("negative Betti number; check rank tolerance")
        groups.append(
            HomologyGroup(
                dimension=dimension,
                betti=int(betti),
                allowed_path_count=len(complex_.allowed_paths[dimension]),
                omega_dimension=int(omega_dimension),
                cycle_dimension=int(cycle_dimension),
                boundary_rank=int(incoming_rank),
            )
        )
    return PathHomologyResult(tuple(groups), complex_)


def compute_path_homology(
    vertices: Sequence[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    max_dimension: int = 1,
    tolerance: float = 1e-9,
) -> tuple[HomologyGroup, ...]:
    """Compatibility helper returning only homology-group summaries."""

    return path_homology(
        vertices,
        edges,
        max_dimension=max_dimension,
        tolerance=tolerance,
    ).groups


def filtration_descriptors(
    graph: WeightedDiGraph,
    thresholds: Iterable[float],
    *,
    max_dimension: int = 1,
    tolerance: float = 1e-9,
    direction: FiltrationDirection = "superlevel",
) -> list[dict[str, int | float]]:
    """Compute independent path-homology descriptors at fixed levels."""

    output: list[dict[str, int | float]] = []
    for threshold in thresholds:
        value = float(threshold)
        thresholded = graph.threshold(value, direction=direction)
        groups = compute_path_homology(
            thresholded.vertices,
            thresholded.edge_pairs,
            max_dimension=max_dimension,
            tolerance=tolerance,
        )
        row: dict[str, int | float] = {
            "threshold": value,
            "vertex_count": len(thresholded.vertices),
            "edge_count": len(thresholded.edge_pairs),
        }
        for group in groups:
            for key, item in asdict(group).items():
                if key != "dimension":
                    row[f"h{group.dimension}_{key}"] = item
        output.append(row)
    return output


def _embed_basis(
    basis: NDArray[np.float64],
    local_paths: Sequence[Path],
    ambient_paths: Sequence[Path],
) -> NDArray[np.float64]:
    embedded = np.zeros((len(ambient_paths), basis.shape[1]), dtype=float)
    ambient_index = {path: index for index, path in enumerate(ambient_paths)}
    for local_index, path in enumerate(local_paths):
        embedded[ambient_index[path]] = basis[local_index]
    return embedded


def _homology_subspaces(
    vertices: Sequence[Hashable],
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    max_dimension: int,
    ambient_paths: Mapping[int, Sequence[Path]],
    tolerance: float,
) -> tuple[tuple[HomologyGroup, ...], tuple[_HomologySubspace, ...]]:
    complex_ = build_path_complex(
        vertices,
        edges,
        max_dimension=max_dimension + 1,
        tolerance=tolerance,
    )
    groups: list[HomologyGroup] = []
    subspaces: list[_HomologySubspace] = []
    for dimension in range(max_dimension + 1):
        boundary = complex_.boundary_matrices[dimension]
        cycles_in_omega = _null_space(boundary, tolerance)
        cycles_local = complex_.omega_bases[dimension] @ cycles_in_omega
        boundaries_local = complex_.ambient_boundary_matrices[dimension + 1]
        cycle_dimension = cycles_local.shape[1]
        boundary_rank = _matrix_rank(boundaries_local, tolerance)
        betti = cycle_dimension - boundary_rank
        if betti < 0:
            raise ArithmeticError("negative Betti number; check rank tolerance")
        groups.append(
            HomologyGroup(
                dimension=dimension,
                betti=int(betti),
                allowed_path_count=len(complex_.allowed_paths[dimension]),
                omega_dimension=int(complex_.omega_bases[dimension].shape[1]),
                cycle_dimension=int(cycle_dimension),
                boundary_rank=int(boundary_rank),
            )
        )
        subspaces.append(
            _HomologySubspace(
                _embed_basis(
                    cycles_local,
                    complex_.allowed_paths[dimension],
                    ambient_paths[dimension],
                ),
                _embed_basis(
                    boundaries_local,
                    complex_.allowed_paths[dimension],
                    ambient_paths[dimension],
                ),
            )
        )
    return tuple(groups), tuple(subspaces)


def _persistent_rank(
    source_cycles: NDArray[np.float64],
    target_boundaries: NDArray[np.float64],
    tolerance: float,
) -> int:
    boundary_rank = _matrix_rank(target_boundaries, tolerance)
    combined = np.concatenate([source_cycles, target_boundaries], axis=1)
    return _matrix_rank(combined, tolerance) - boundary_rank


def _barcode_from_rank_invariant(
    ranks: NDArray[np.int64],
    thresholds: Sequence[float],
    *,
    dimension: int,
) -> list[PersistenceInterval]:
    level_count = len(thresholds)

    def rank(source: int, target: int) -> int:
        if source < 0 or target >= level_count:
            return 0
        return int(ranks[source, target])

    intervals: list[PersistenceInterval] = []
    for birth in range(level_count):
        for death in range(birth, level_count):
            multiplicity = (
                rank(birth, death)
                - rank(birth - 1, death)
                - rank(birth, death + 1)
                + rank(birth - 1, death + 1)
            )
            if multiplicity < 0:
                raise ArithmeticError("negative persistence multiplicity")
            if multiplicity == 0:
                continue
            censored = death == level_count - 1
            death_threshold = None if censored else float(thresholds[death + 1])
            terminal = (
                float(thresholds[-1])
                if censored
                else float(death_threshold)
            )
            intervals.append(
                PersistenceInterval(
                    dimension=dimension,
                    birth_index=birth,
                    death_index=None if censored else death + 1,
                    birth_threshold=float(thresholds[birth]),
                    death_threshold=death_threshold,
                    lifetime=abs(float(thresholds[birth]) - terminal),
                    multiplicity=multiplicity,
                    censored=censored,
                )
            )
    return intervals


def persistent_path_homology(
    graph: WeightedDiGraph,
    thresholds: Iterable[float],
    *,
    max_dimension: int = 1,
    tolerance: float = 1e-9,
    direction: FiltrationDirection = "superlevel",
) -> PersistentPathResult:
    """Compute exact finite persistent path homology in arbitrary requested degree.

    The implementation uses dense real linear algebra and explicit path
    enumeration. It is intended as a reference backend for small and medium
    sparse digraphs.
    """

    if max_dimension < 0:
        raise ValueError("max_dimension cannot be negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if direction not in ("superlevel", "sublevel"):
        raise ValueError("direction must be 'superlevel' or 'sublevel'")
    levels = tuple(
        sorted(
            {float(value) for value in thresholds},
            reverse=direction == "superlevel",
        )
    )
    if not levels:
        raise ValueError("at least one threshold is required")
    if not np.all(np.isfinite(levels)):
        raise ValueError("thresholds must be finite")

    ambient = graph.threshold(levels[-1], direction=direction)
    ambient_paths = enumerate_allowed_paths(
        ambient.vertices,
        ambient.edge_pairs,
        max_dimension=max_dimension,
    )
    descriptors: list[dict[str, int | float]] = []
    subspaces: list[tuple[_HomologySubspace, ...]] = []
    for level in levels:
        thresholded = graph.threshold(level, direction=direction)
        groups, level_subspaces = _homology_subspaces(
            thresholded.vertices,
            thresholded.edge_pairs,
            max_dimension=max_dimension,
            ambient_paths=ambient_paths,
            tolerance=tolerance,
        )
        row: dict[str, int | float] = {
            "threshold": level,
            "vertex_count": len(thresholded.vertices),
            "edge_count": len(thresholded.edge_pairs),
        }
        for group in groups:
            for key, value in asdict(group).items():
                if key != "dimension":
                    row[f"h{group.dimension}_{key}"] = value
        descriptors.append(row)
        subspaces.append(level_subspaces)

    rank_invariants: list[NDArray[np.int64]] = []
    intervals: list[PersistenceInterval] = []
    for dimension in range(max_dimension + 1):
        ranks = np.zeros((len(levels), len(levels)), dtype=np.int64)
        for source in range(len(levels)):
            for target in range(source, len(levels)):
                ranks[source, target] = _persistent_rank(
                    subspaces[source][dimension].cycle_basis,
                    subspaces[target][dimension].boundary_basis,
                    tolerance,
                )
        for level_index, row in enumerate(descriptors):
            if ranks[level_index, level_index] != row[f"h{dimension}_betti"]:
                raise ArithmeticError(
                    "rank invariant diagonal disagrees with Betti number"
                )
        rank_invariants.append(ranks)
        intervals.extend(
            _barcode_from_rank_invariant(
                ranks,
                levels,
                dimension=dimension,
            )
        )

    return PersistentPathResult(
        thresholds=levels,
        descriptors=tuple(descriptors),
        intervals=tuple(intervals),
        rank_invariants=tuple(rank_invariants),
        direction=direction,
    )
