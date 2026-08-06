from __future__ import annotations

import numpy as np

from features.modulation_smp import (
    assign_states,
    fit_codebook,
    fit_shared_transform,
)


def _spectra() -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.linspace(0.5, 8.0, 16)
    centers = (1.5, 4.0, 6.5)
    rows = []
    for center in centers:
        base = np.exp(-0.5 * ((frequencies - center) / 0.65) ** 2)
        for offset in np.linspace(-0.04, 0.04, 12):
            shifted = np.roll(base, 1 if offset > 0.02 else -1 if offset < -0.02 else 0)
            shifted = np.maximum(shifted + offset, 1e-6)
            rows.append(shifted / shifted.sum())
    return np.asarray(rows), frequencies


def test_smp_codebook_is_deterministic_and_ordered() -> None:
    spectra, frequencies = _spectra()
    transform, embedded = fit_shared_transform(spectra, frequencies, n_components=4, random_seed=17)
    first = fit_codebook(embedded, spectra, frequencies, state_count=3, random_seed=21)
    second = fit_codebook(embedded, spectra, frequencies, state_count=3, random_seed=21)
    np.testing.assert_allclose(first.centers, second.centers)
    np.testing.assert_array_equal(first.training_state_counts, second.training_state_counts)
    assert np.all(np.diff(first.spectral_centroids_hz) >= 0.0)
    assert first.training_state_counts.sum() == spectra.shape[0]


def test_assign_states_preserves_missing_windows() -> None:
    spectra, frequencies = _spectra()
    transform, embedded = fit_shared_transform(spectra, frequencies, n_components=4, random_seed=17)
    codebook = fit_codebook(embedded, spectra, frequencies, state_count=3, random_seed=21)
    valid = np.ones(spectra.shape[0], dtype=bool)
    valid[[2, 9]] = False
    states = assign_states(spectra, valid, transform, codebook)
    assert states.shape == valid.shape
    assert states[2] == states[9] == -1
    assert np.all(states[valid] >= 0)
    assert np.all(states[valid] < 3)
