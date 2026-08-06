from __future__ import annotations

import numpy as np

from tda.analysis import (
    _normalize_distance_scale,
    _uniform_sample,
    _wide,
    delay_embedding,
    persistence_descriptors,
)


def test_delay_embedding_preserves_requested_geometry() -> None:
    values = np.arange(10, dtype=float)
    embedded = delay_embedding(values, dimension=3, lag=2)
    assert embedded.shape == (6, 3)
    np.testing.assert_allclose(embedded[0, 1] - embedded[0, 0], 2 / np.std(values))


def test_uniform_sample_and_distance_normalization_are_deterministic() -> None:
    cloud = np.column_stack([np.arange(100), np.arange(100) ** 2])
    sampled = _uniform_sample(cloud, 16)
    normalized, scale = _normalize_distance_scale(sampled)
    assert sampled.shape == (16, 2)
    assert scale > 0
    distances = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=2)
    assert np.isclose(np.median(distances[np.triu_indices(16, 1)]), 1.0)


def test_persistence_descriptors_ignore_infinite_h0_class() -> None:
    diagrams = [
        np.array([[0.0, 0.2], [0.0, 0.5], [0.0, np.inf]]),
        np.array([[0.3, 0.6], [0.5, 0.55]]),
    ]
    result = persistence_descriptors(diagrams, prominent_lifetime=0.1)
    assert np.isclose(result["h0_total_persistence"], 0.7)
    assert result["h1_count"] == 2
    assert result["h1_prominent_count"] == 1
    assert np.isclose(result["h1_max_persistence"], 0.3)


def test_wide_pivots_descriptor_tuple_to_representation_columns() -> None:
    import pandas as pd

    identity = {
        "segment_id": "s1",
        "track_id": "t1",
        "group": "focus",
        "split": "discovery",
        "scale_seconds": 180.0,
    }
    descriptors = persistence_descriptors(
        [np.array([[0.0, 1.0], [0.0, np.inf]]), np.empty((0, 2))],
        prominent_lifetime=0.1,
    )
    frame = pd.DataFrame(
        [
            {**identity, "representation": "rhythm", **descriptors},
            {**identity, "representation": "chroma", **descriptors},
        ]
    )
    wide = _wide(frame)
    assert wide.shape == (1, 25)
    assert "rhythm__h0_total_persistence" in wide
