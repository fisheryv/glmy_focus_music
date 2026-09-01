from __future__ import annotations

import numpy as np


from features.structure import (
    abstract_structure_states,
    checkerboard_novelty,
    self_similarity_matrix,
    structural_features,
)


def test_ssm_novelty_recovers_a_clear_acoustic_boundary() -> None:
    first = np.tile([1.0, 0.0, 0.0], (20, 1))
    second = np.tile([0.0, 1.0, 0.0], (20, 1))
    vectors = np.vstack([first, second])

    similarity = self_similarity_matrix(vectors)
    novelty = checkerboard_novelty(similarity, kernel_size=5)

    assert similarity.shape == (40, 40)
    assert int(np.argmax(novelty)) == 20
    assert float(similarity[5, 10]) > float(similarity[5, 30])


def test_structural_features_turn_macro_sections_into_a_state_path() -> None:
    vectors = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (12, 1)),
            np.tile([0.0, 1.0, 0.0], (12, 1)),
            np.tile([1.0, 0.0, 0.0], (12, 1)),
        ]
    )
    features = structural_features(
        vectors,
        np.arange(36, dtype=float),
        np.ones(36, dtype=bool),
        duration=36.0,
        kernel_seconds=4.0,
        min_segment_seconds=6.0,
        max_segment_seconds=18.0,
        novelty_threshold=0.5,
    )
    states = abstract_structure_states(
        features.block_vectors,
        features.valid,
        similarity_threshold=0.7,
    )

    assert np.any(np.abs(features.boundary_times - 12.0) <= 1.0)
    assert np.any(np.abs(features.boundary_times - 24.0) <= 1.0)
    assert len(states) == features.block_vectors.shape[0]
    assert states[0] == states[-1]


def test_topology_loader_accepts_one_dimensional_structure_states(tmp_path) -> None:
    path = tmp_path / "structure.npz"
    np.savez(path, states=np.asarray([0, 1, -1, 0], dtype=np.int16))

    assert _load_state_sequence(path, "structure") == [0, 1, None, 0]
