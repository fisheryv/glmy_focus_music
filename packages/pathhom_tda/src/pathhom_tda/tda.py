"""Vietoris-Rips TDA helpers with an optional Ripser.py backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TDAError(RuntimeError):
    """Raised when a point cloud or persistence computation is invalid."""


@dataclass(frozen=True, slots=True)
class RipsResult:
    """Structured result returned by :func:`vietoris_rips`."""

    diagrams: tuple[NDArray[np.float64], ...]
    cocycles: tuple[tuple[NDArray[np.float64], ...], ...]
    distance_scale: float
    point_count: int
    coefficient: int
    max_dimension: int
    distance_matrix: bool

    def diagram(self, dimension: int) -> NDArray[np.float64]:
        if not 0 <= dimension < len(self.diagrams):
            raise ValueError(f"dimension must be between 0 and {len(self.diagrams) - 1}")
        return self.diagrams[dimension]

    def descriptors(self, *, prominent_lifetime: float = 0.1) -> dict[str, float]:
        return persistence_descriptors(
            self.diagrams,
            prominent_lifetime=prominent_lifetime,
        )


def finite_rows(values: ArrayLike) -> NDArray[np.float64]:
    """Return finite rows of a two-dimensional point cloud."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("point cloud must be a two-dimensional array")
    return array[np.all(np.isfinite(array), axis=1)]


def uniform_sample(
    values: ArrayLike,
    max_points: int,
    *,
    offset: float = 0.0,
) -> NDArray[np.float64]:
    """Select at most ``max_points`` deterministic, uniformly spaced rows."""

    if max_points < 1:
        raise ValueError("max_points must be positive")
    points = finite_rows(values)
    if len(points) <= max_points:
        return points
    positions = np.linspace(offset, len(points) - 1 + offset, max_points)
    indices = np.clip(np.rint(positions).astype(int), 0, len(points) - 1)
    return points[indices]


def normalize_distance_scale(
    values: ArrayLike,
    *,
    minimum_points: int = 2,
) -> tuple[NDArray[np.float64], float]:
    """Normalize a point cloud by its median positive pairwise distance."""

    points = finite_rows(values)
    if len(points) < minimum_points:
        raise TDAError(
            f"point cloud contains fewer than {minimum_points} finite points"
        )
    differences = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    upper = distances[np.triu_indices(len(points), k=1)]
    positive = upper[upper > np.finfo(float).eps]
    scale = float(np.median(positive)) if positive.size else 1.0
    return points / scale, scale


def normalize_distance_matrix(
    distances: ArrayLike,
) -> tuple[NDArray[np.float64], float]:
    """Normalize a square distance matrix by its positive upper-triangle median."""

    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance matrix must contain only finite values")
    if np.any(matrix < 0):
        raise ValueError("distance matrix cannot contain negative values")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("distance matrix must be symmetric")
    upper = matrix[np.triu_indices(len(matrix), k=1)]
    positive = upper[upper > np.finfo(float).eps]
    scale = float(np.median(positive)) if positive.size else 1.0
    return matrix / scale, scale


def delay_embedding(
    values: ArrayLike,
    *,
    dimension: int,
    lag: int,
) -> NDArray[np.float64]:
    """Create a standardized Takens delay embedding of a scalar series."""

    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if lag < 1:
        raise ValueError("lag must be positive")
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(series)
    if not np.all(finite):
        observed = np.flatnonzero(finite)
        if observed.size < 2:
            raise TDAError("delay series has insufficient finite values")
        series = np.interp(np.arange(series.size), observed, series[observed])
    scale = float(np.std(series))
    if scale > np.finfo(float).eps:
        series = (series - float(np.mean(series))) / scale
    else:
        series = series - float(np.mean(series))
    width = 1 + (dimension - 1) * lag
    if len(series) < width + 3:
        raise TDAError("delay series is too short")
    return np.column_stack(
        [
            series[offset : len(series) - width + 1 + offset]
            for offset in range(0, width, lag)
        ]
    )


def _entropy(lifetimes: NDArray[np.float64]) -> float:
    total = float(np.sum(lifetimes))
    if total <= np.finfo(float).eps:
        return 0.0
    probabilities = lifetimes / total
    entropy = -float(
        np.sum(probabilities * np.log(probabilities + np.finfo(float).eps))
    )
    maximum = math.log(len(lifetimes)) if len(lifetimes) > 1 else 1.0
    return entropy / maximum


def finite_lifetimes(diagram: ArrayLike) -> NDArray[np.float64]:
    """Return positive finite lifetimes from a persistence diagram."""

    values = np.asarray(diagram, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("a persistence diagram must have shape (n_intervals, 2)")
    finite = np.all(np.isfinite(values), axis=1)
    lifetimes = values[finite, 1] - values[finite, 0]
    return lifetimes[lifetimes > np.finfo(float).eps]


def diagram_descriptors(
    diagram: ArrayLike,
    *,
    prominent_lifetime: float = 0.1,
) -> dict[str, float]:
    """Summarize one persistence diagram without dimension-specific prefixes."""

    if prominent_lifetime < 0:
        raise ValueError("prominent_lifetime cannot be negative")
    lifetimes = finite_lifetimes(diagram)
    return {
        "count": float(lifetimes.size),
        "prominent_count": float(
            np.count_nonzero(lifetimes >= prominent_lifetime)
        ),
        "total_persistence": float(np.sum(lifetimes)),
        "max_persistence": float(np.max(lifetimes, initial=0.0)),
        "mean_persistence": (
            float(np.mean(lifetimes)) if lifetimes.size else 0.0
        ),
        "q75_persistence": (
            float(np.quantile(lifetimes, 0.75)) if lifetimes.size else 0.0
        ),
        "persistence_entropy": _entropy(lifetimes),
    }


def persistence_descriptors(
    diagrams: Sequence[ArrayLike],
    *,
    prominent_lifetime: float = 0.1,
) -> dict[str, float]:
    """Return stable H0/H1 descriptor names used by the research pipeline."""

    if len(diagrams) < 2:
        raise ValueError("H0 and H1 diagrams are required")
    h0 = diagram_descriptors(
        diagrams[0],
        prominent_lifetime=prominent_lifetime,
    )
    h1 = diagram_descriptors(
        diagrams[1],
        prominent_lifetime=prominent_lifetime,
    )
    return {
        "h0_total_persistence": h0["total_persistence"],
        "h0_max_persistence": h0["max_persistence"],
        "h0_q75_persistence": h0["q75_persistence"],
        "h0_persistence_entropy": h0["persistence_entropy"],
        "h1_count": h1["count"],
        "h1_prominent_count": h1["prominent_count"],
        "h1_total_persistence": h1["total_persistence"],
        "h1_max_persistence": h1["max_persistence"],
        "h1_mean_persistence": h1["mean_persistence"],
        "h1_persistence_entropy": h1["persistence_entropy"],
    }


def _ripser_backend() -> Any:
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError(
            "Vietoris-Rips persistence requires the 'tda' extra: "
            "install pathhom-tda[tda]"
        ) from exc
    return ripser


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


def vietoris_rips(
    values: ArrayLike,
    *,
    max_dimension: int = 1,
    coefficient: int = 2,
    distance_matrix: bool = False,
    threshold: float = np.inf,
    max_points: int | None = None,
    normalize: bool = False,
    do_cocycles: bool = False,
    n_perm: int | None = None,
) -> RipsResult:
    """Compute Vietoris-Rips persistence using Ripser.py.

    Point-cloud rows with non-finite values are removed. A distance matrix must
    instead be finite, square, symmetric, and non-negative.
    """

    if max_dimension < 0:
        raise ValueError("max_dimension cannot be negative")
    if isinstance(coefficient, bool) or not isinstance(coefficient, int):
        raise ValueError("coefficient must be a prime integer")
    if not _is_prime(coefficient):
        raise ValueError("coefficient must be a prime integer")
    if max_points is not None and max_points < 2:
        raise ValueError("max_points must be at least two")
    if math.isnan(float(threshold)) or float(threshold) < 0:
        raise ValueError("threshold must be non-negative")

    if distance_matrix:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("distance matrix must be square")
        if max_points is not None and len(data) > max_points:
            indices = np.unique(
                np.clip(
                    np.rint(np.linspace(0, len(data) - 1, max_points)).astype(int),
                    0,
                    len(data) - 1,
                )
            )
            data = data[np.ix_(indices, indices)]
        if normalize:
            data, distance_scale = normalize_distance_matrix(data)
        else:
            normalized, original_scale = normalize_distance_matrix(data)
            data = normalized * original_scale
            distance_scale = 1.0
    else:
        data = finite_rows(values)
        if max_points is not None:
            data = uniform_sample(data, max_points)
        if len(data) < 2:
            raise TDAError("point cloud contains fewer than two finite points")
        if normalize:
            data, distance_scale = normalize_distance_scale(data)
        else:
            distance_scale = 1.0

    ripser = _ripser_backend()
    kwargs: dict[str, Any] = {
        "maxdim": max_dimension,
        "coeff": coefficient,
        "distance_matrix": distance_matrix,
        "thresh": float(threshold),
        "do_cocycles": do_cocycles,
    }
    if n_perm is not None:
        kwargs["n_perm"] = n_perm
    raw = ripser(data, **kwargs)
    diagrams = tuple(
        np.asarray(diagram, dtype=np.float64)
        for diagram in raw["dgms"]
    )
    raw_cocycles = raw.get("cocycles", ())
    cocycles = tuple(
        tuple(np.asarray(cocycle, dtype=np.float64) for cocycle in dimension)
        for dimension in raw_cocycles
    )
    return RipsResult(
        diagrams=diagrams,
        cocycles=cocycles,
        distance_scale=distance_scale,
        point_count=len(data),
        coefficient=coefficient,
        max_dimension=max_dimension,
        distance_matrix=distance_matrix,
    )
