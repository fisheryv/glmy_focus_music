from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from focus_topology.generation.rerank import topology_distance


def paired_improvement(baseline_distances: ArrayLike, guided_distances: ArrayLike) -> np.ndarray:
    """Return positive fractional improvement for paired target distances."""

    baseline = np.asarray(baseline_distances, dtype=float)
    guided = np.asarray(guided_distances, dtype=float)
    if baseline.shape != guided.shape:
        raise ValueError("baseline and guided arrays must have the same shape")
    if np.any(baseline <= 0):
        raise ValueError("baseline distances must be positive")
    return (baseline - guided) / baseline


def relative_centroid_distances(
    descriptor: ArrayLike,
    centroids: Mapping[str, ArrayLike],
) -> dict[str, float]:
    """Measure a descriptor against the configured reference centroids."""

    if not centroids:
        raise ValueError("at least one centroid is required")
    return {
        name: topology_distance(descriptor, centroid)
        for name, centroid in sorted(centroids.items())
    }
