from __future__ import annotations

import numpy as np

from features.pitch_v2 import (
    assign_codebook,
    chroma_to_tonnetz,
    normalize_chroma,
    tonnetz_similarity,
)


def test_tonnetz_projection_matches_librosa_definition() -> None:
    import librosa

    chroma = np.eye(12, dtype=float)
    expected = librosa.feature.tonnetz(chroma=chroma.T).T
    assert np.allclose(chroma_to_tonnetz(chroma), expected)


def test_pitch_v2_codebook_assignment_masks_invalid_steps() -> None:
    tonnetz = np.asarray([[0.0] * 6, [1.0] * 6, [0.1] * 6])
    centers = np.asarray([[0.0] * 6, [1.0] * 6])
    states = assign_codebook(tonnetz, centers, valid=[True, True, False])
    assert states.tolist() == [0, 1, -1]


def test_normalization_and_similarity_are_finite() -> None:
    chroma = np.vstack([np.zeros(12), np.eye(12)[0], np.eye(12)[7]])
    normalized = normalize_chroma(chroma)
    similarity = tonnetz_similarity(chroma_to_tonnetz(chroma))
    assert np.all(np.isfinite(normalized))
    assert np.all(np.isfinite(similarity))
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)
