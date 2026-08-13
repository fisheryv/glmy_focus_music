import numpy as np
import pytest

from pyglmy import (
    delay_embedding,
    normalize_distance_scale,
    persistence_descriptors,
    uniform_sample,
    vietoris_rips,
)


def test_delay_embedding_and_sampling_are_deterministic() -> None:
    values = np.arange(10, dtype=float)
    embedded = delay_embedding(values, dimension=3, lag=2)
    sampled = uniform_sample(np.column_stack([values, values**2]), 4)

    assert embedded.shape == (6, 3)
    assert sampled[:, 0].tolist() == [0.0, 3.0, 6.0, 9.0]


def test_distance_normalization_sets_median_pair_distance_to_one() -> None:
    cloud = np.column_stack([np.arange(16), np.arange(16) ** 2])
    normalized, scale = normalize_distance_scale(cloud)
    distances = np.linalg.norm(
        normalized[:, None] - normalized[None, :],
        axis=2,
    )

    assert scale > 0
    assert np.isclose(
        np.median(distances[np.triu_indices(len(cloud), 1)]),
        1.0,
    )


def test_persistence_descriptors_ignore_infinite_h0_interval() -> None:
    result = persistence_descriptors(
        [
            np.asarray([[0.0, 0.2], [0.0, np.inf]]),
            np.asarray([[0.3, 0.6]]),
        ],
        prominent_lifetime=0.1,
    )

    assert result["h0_total_persistence"] == pytest.approx(0.2)
    assert result["h1_count"] == 1
    assert result["h1_max_persistence"] == pytest.approx(0.3)


def test_vietoris_rips_detects_circle_h1() -> None:
    pytest.importorskip("ripser")
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    result = vietoris_rips(circle, max_dimension=1, normalize=True)

    assert result.point_count == 32
    assert result.distance_scale > 0
    assert np.max(result.diagram(1)[:, 1] - result.diagram(1)[:, 0]) > 0.5
