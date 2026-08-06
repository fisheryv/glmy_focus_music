from __future__ import annotations

import numpy as np
import pytest

from topology.multiview_fusion import (
    DiscoveryMahalanobisBlock,
    equal_block_fusion,
    hierarchical_fusion,
    paired_incremental_permutation,
    permutation_pseudo_f,
)


def test_discovery_block_is_rank_normalized_and_deterministic() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(80, 4))
    matrix[:, 3] = matrix[:, 0] + matrix[:, 1]
    transformer = DiscoveryMahalanobisBlock().fit(matrix)
    first = transformer.transform(matrix)
    second = transformer.transform(matrix.copy())
    assert transformer.effective_rank == 3
    assert first.shape == (80, 3)
    assert np.allclose(first, second)
    assert np.all(np.isfinite(first))


def test_equal_and_hierarchical_fusion_preserve_requested_weights() -> None:
    left = np.array([[1.0], [0.0]])
    right = np.array([[0.0], [1.0]])
    local = equal_block_fusion([left, right])
    assert np.allclose(np.sum(local**2, axis=1), 0.5)
    structure = np.ones((2, 1))
    combined = hierarchical_fusion(local, structure, structure_weight=0.25)
    assert np.allclose(combined[:, :2], local * np.sqrt(0.75))
    assert np.allclose(combined[:, 2:], structure * 0.5)


def test_hierarchical_fusion_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="structure_weight"):
        hierarchical_fusion(np.ones((2, 1)), np.ones((2, 1)), structure_weight=1.1)


def test_permutation_statistics_are_reproducible() -> None:
    labels = np.array(["a"] * 10 + ["b"] * 10)
    signal = np.concatenate([np.zeros((10, 1)), np.ones((10, 1))], axis=0)
    noise = np.zeros((20, 1))
    first = permutation_pseudo_f(signal, labels, permutations=99, seed=11)
    second = permutation_pseudo_f(signal, labels, permutations=99, seed=11)
    assert first == second
    assert first["pseudo_f"] > 0.0
    increment = paired_incremental_permutation(signal, noise, labels, permutations=99, seed=11)
    assert increment["delta_pseudo_f"] > 0.0
    assert increment["p_value_one_sided"] <= 0.05
