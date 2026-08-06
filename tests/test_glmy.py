import numpy as np

from focus_topology.graphs.transition import TransitionGraph, WeightedEdge
from focus_topology.homology.glmy import compute_path_homology, persistent_path_homology


def test_directed_cycle_has_one_dimensional_h1() -> None:
    groups = compute_path_homology(
        [0, 1, 2],
        [(0, 1), (1, 2), (2, 0)],
        max_dimension=1,
    )

    assert [group.betti for group in groups] == [1, 1]


def test_transitive_triangle_fills_the_one_cycle() -> None:
    groups = compute_path_homology(
        [0, 1, 2],
        [(0, 1), (1, 2), (0, 2)],
        max_dimension=1,
    )

    assert [group.betti for group in groups] == [1, 0]


def test_disconnected_vertices_are_retained() -> None:
    groups = compute_path_homology([0, 1, 2], [], max_dimension=1)

    assert [group.betti for group in groups] == [3, 0]


def test_persistent_path_homology_matches_level_betti_numbers() -> None:
    graph = TransitionGraph(
        vertices=(0, 1, 2),
        edges=(
            WeightedEdge(source=0, target=1, weight=0.9, count=9),
            WeightedEdge(source=1, target=2, weight=0.9, count=9),
            WeightedEdge(source=2, target=0, weight=0.9, count=9),
        ),
    )

    result = persistent_path_homology(graph, [0.95, 0.8])

    assert [row["h0_betti"] for row in result.descriptors] == [3, 1]
    assert [row["h1_betti"] for row in result.descriptors] == [0, 1]
    assert np.diag(result.h0_rank_invariant).tolist() == [3, 1]
    assert np.diag(result.h1_rank_invariant).tolist() == [0, 1]
    assert sum(
        interval.multiplicity
        for interval in result.intervals
        if interval.dimension == 1 and interval.censored
    ) == 1


def test_persistent_path_homology_is_threshold_order_invariant() -> None:
    graph = TransitionGraph(
        vertices=(0, 1),
        edges=(WeightedEdge(source=0, target=1, weight=0.75, count=3),),
    )

    descending = persistent_path_homology(graph, [0.9, 0.7, 0.5])
    shuffled = persistent_path_homology(graph, [0.5, 0.9, 0.7])

    assert descending.thresholds == shuffled.thresholds
    assert descending.intervals == shuffled.intervals
