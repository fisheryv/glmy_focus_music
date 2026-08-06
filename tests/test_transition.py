import pytest

from focus_topology.graphs.transition import build_transition_graph


def test_transition_probabilities_are_normalized_per_source() -> None:
    graph = build_transition_graph([0, 1, 0, 2, 0, 1])
    from_zero = {edge.target: edge.weight for edge in graph.edges if edge.source == 0}

    assert from_zero == pytest.approx({1: 2 / 3, 2: 1 / 3})


def test_top_k_is_deterministic() -> None:
    graph = build_transition_graph([0, 2, 0, 1, 0, 2], top_k=1)

    assert [(edge.source, edge.target) for edge in graph.edges if edge.source == 0] == [(0, 2)]


def test_self_loops_are_excluded_by_default() -> None:
    graph = build_transition_graph([0, 0, 1])

    assert graph.edge_pairs == ((0, 1),)

