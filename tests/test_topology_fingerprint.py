from __future__ import annotations

import numpy as np

from generation.topology_fingerprint import (
    CORE_FEATURES,
    SUPPORT_FEATURES,
    fit_topology_fingerprint,
    read_topology_fingerprint,
    write_topology_fingerprint,
)


def _fingerprint():
    rng = np.random.default_rng(20260716)
    core = rng.normal(size=(64, len(CORE_FEATURES)))
    support = rng.normal(size=(64, len(SUPPORT_FEATURES)))
    challenger = rng.normal(size=(64, 1))
    return fit_topology_fingerprint(
        core,
        support,
        challenger,
        fingerprint_id="test",
        reference_segment_ids=tuple(f"segment-{index}" for index in range(64)),
    )


def test_fingerprint_center_has_zero_distance_and_loss() -> None:
    fingerprint = _fingerprint()
    center = np.asarray(fingerprint.core_center)
    assert np.isclose(fingerprint.distance(center), 0.0)
    assert np.isclose(fingerprint.core_shell_loss(center), 0.0)


def test_fingerprint_support_band_penalizes_only_outside_values() -> None:
    fingerprint = _fingerprint()
    middle = (np.asarray(fingerprint.support_lower) + np.asarray(fingerprint.support_upper)) / 2
    assert np.isclose(fingerprint.support_band_loss(middle), 0.0)
    outside = np.asarray(fingerprint.support_upper) + 2.0 * np.asarray(
        fingerprint.support_scale
    )
    assert fingerprint.support_band_loss(outside) > 0.0


def test_fingerprint_round_trip(tmp_path) -> None:
    fingerprint = _fingerprint()
    path = tmp_path / "fingerprint.json"
    write_topology_fingerprint(path, fingerprint)
    loaded = read_topology_fingerprint(path)
    assert loaded == fingerprint
