import numpy as np
import pytest

from pyglmy import WeightedDiGraph


def test_graph_supports_superlevel_and_sublevel_filtrations() -> None:
    graph = WeightedDiGraph.from_edges(
        [(0, 1, 0.2), (1, 2, 0.8)],
        vertices=[0, 1, 2, 3],
    )

    assert graph.threshold(0.5).edge_pairs == ((1, 2),)
    assert graph.threshold(0.5, direction="sublevel").edge_pairs == ((0, 1),)
    assert graph.threshold(0.5).vertices == (0, 1, 2, 3)


def test_adjacency_round_trip_is_deterministic() -> None:
    graph = WeightedDiGraph.from_adjacency(
        np.asarray([[0.0, 0.5], [0.25, 0.0]])
    )

    np.testing.assert_allclose(
        graph.adjacency_matrix(),
        [[0.0, 0.5], [0.25, 0.0]],
    )


def test_duplicate_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        WeightedDiGraph.from_edges([(0, 1, 0.2), (0, 1, 0.3)])
