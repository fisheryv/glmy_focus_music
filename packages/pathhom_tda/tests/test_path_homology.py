import numpy as np

from pathhom_tda import (
    PathHomology,
    WeightedDiGraph,
    build_path_complex,
    path_homology,
    persistent_path_homology,
)


def test_cycle_and_transitive_triangle_have_expected_h1() -> None:
    cycle = path_homology(
        [0, 1, 2],
        [(0, 1), (1, 2), (2, 0)],
    )
    triangle = path_homology(
        [0, 1, 2],
        [(0, 1), (1, 2), (0, 2)],
    )

    assert cycle.betti_numbers == (1, 1)
    assert triangle.betti_numbers == (1, 0)


def test_chain_complex_exposes_allowed_paths_and_omega_bases() -> None:
    complex_ = build_path_complex(
        [0, 1, 2],
        [(0, 1), (1, 2)],
        max_dimension=2,
    )

    assert complex_.allowed_paths[2] == ((0, 1, 2),)
    assert complex_.omega_bases[2].shape == (1, 0)
    assert complex_.boundary_matrices[2].shape == (2, 0)
    np.testing.assert_allclose(
        complex_.boundary_matrices[1] @ complex_.boundary_matrices[2],
        0.0,
        atol=1e-12,
    )


def test_object_oriented_facade_reuses_configuration() -> None:
    calculator = PathHomology(max_dimension=1)

    result = calculator.compute(
        [0, 1, 2],
        [(0, 1), (1, 2), (2, 0)],
    )

    assert result.betti_numbers == (1, 1)


def test_persistent_path_homology_supports_multiple_dimensions() -> None:
    graph = WeightedDiGraph.from_edges(
        [
            (0, 1, 0.9),
            (1, 2, 0.9),
            (2, 0, 0.9),
            (0, 2, 0.4),
        ]
    )

    result = persistent_path_homology(
        graph,
        [0.95, 0.8, 0.3],
        max_dimension=2,
    )

    assert result.thresholds == (0.95, 0.8, 0.3)
    assert result.betti_curve(1) == (0, 1, 0)
    assert len(result.rank_invariants) == 3
    for dimension, ranks in enumerate(result.rank_invariants):
        assert np.diag(ranks).tolist() == [
            row[f"h{dimension}_betti"] for row in result.descriptors
        ]


def test_sublevel_filtration_orders_levels_ascending() -> None:
    graph = WeightedDiGraph.from_edges(
        [(0, 1, 0.1), (1, 2, 0.2), (2, 0, 0.3)]
    )

    result = persistent_path_homology(
        graph,
        [0.35, 0.15, 0.25],
        direction="sublevel",
    )

    assert result.thresholds == (0.15, 0.25, 0.35)
    assert result.betti_curve(1) == (0, 0, 1)
