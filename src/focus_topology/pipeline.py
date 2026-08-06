"""Backward-compatible descriptor-only analysis helper."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

from .graphs.transition import build_transition_graph
from .homology.glmy import filtration_descriptors


def analyze_state_sequence(
    states: Sequence[Hashable | None],
    *,
    thresholds: Iterable[float],
    top_k: int | None = 6,
    max_dimension: int = 1,
) -> list[dict[str, int | float]]:
    """Run the legacy state-sequence -> graph -> fixed-threshold pipeline.

    New integrations should prefer :func:`focus_topology.analyze_states`,
    which also returns persistent intervals, rank invariants, and summary
    metrics.
    """

    graph = build_transition_graph(states, normalize=True, top_k=top_k)
    return filtration_descriptors(graph, thresholds, max_dimension=max_dimension)


__all__ = ["analyze_state_sequence"]
