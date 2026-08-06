from __future__ import annotations

import numpy as np

from repetition.analysis import (
    RepetitionConfig,
    _path_cycle_features,
    _sliding_window_features,
    transposition_invariant_chroma_distance,
)
from repetition.multiscale_delay import (
    MultiscaleDelayConfig,
    _analyze_scale,
    delay_embed_multivariate,
)
from repetition.strengthened import StrengthenedConfig, _phase_state_graph_features


def test_transposition_invariant_chroma_distance_ignores_key_shift() -> None:
    chroma = np.eye(12)
    shifted = np.roll(chroma, 5, axis=1)
    combined = np.vstack([chroma, shifted])
    distances = transposition_invariant_chroma_distance(combined)
    assert np.allclose(np.diag(distances[:12, 12:]), 0.0)


def test_sliding_window_h1_responds_to_periodic_signal() -> None:
    config = RepetitionConfig(max_landmarks=40)
    time = np.arange(360)
    periodic = np.sin(2 * np.pi * time / 24)
    shuffled = periodic[np.random.default_rng(7).permutation(len(periodic))]
    loop = _sliding_window_features(periodic, hop_seconds=0.5, config=config)
    null = _sliding_window_features(shuffled, hop_seconds=0.5, config=config)
    assert loop["h1_max_persistence"] > null["h1_max_persistence"]


def test_phase_lifted_path_graph_has_persistent_h1_for_repeated_cycle() -> None:
    config = RepetitionConfig(phase_bins=6)
    motif = np.column_stack(
        [np.sin(2 * np.pi * np.arange(12) / 12), np.cos(2 * np.pi * np.arange(12) / 12)]
    )
    repeated = np.tile(motif, (8, 1))
    result = _path_cycle_features(
        repeated,
        hop_seconds=2.0,
        transposition_invariant=False,
        config=config,
    )
    assert result["path_h1_betti_max"] == 1
    assert result["path_h1_cycle_strength"] > 0.9


def test_phase_state_path_homology_separates_loop_from_shuffle() -> None:
    config = StrengthenedConfig()
    time = np.arange(12)
    motif = np.column_stack(
        [
            np.sin(2 * np.pi * time / 12),
            np.cos(2 * np.pi * time / 12),
            np.sin(4 * np.pi * time / 12),
        ]
    )
    repeated = np.tile(motif, (10, 1))
    shuffled = repeated[np.random.default_rng(7).permutation(len(repeated))]
    loop = _phase_state_graph_features(repeated, hop_seconds=2.0, config=config)
    null = _phase_state_graph_features(shuffled, hop_seconds=2.0, config=config)
    assert loop["h1_max_lifetime"] > null["h1_max_lifetime"]
    assert loop["h1_interval_count"] < null["h1_interval_count"]
    assert loop["graph_vertices"] < null["graph_vertices"]


def test_multivariate_delay_embedding_spans_candidate_period() -> None:
    values = np.column_stack([np.arange(40), np.arange(40) ** 2])
    cloud, delay = delay_embed_multivariate(values, period_frames=16, dimension=8)
    assert delay == 2
    assert cloud.shape == (26, 16)


def test_multiscale_delay_h1_exceeds_block_shuffle_for_periodic_trajectory() -> None:
    config = MultiscaleDelayConfig(
        scales_bars=(4,),
        max_landmarks=48,
        max_landmark_candidates=96,
        surrogate_count=5,
    )
    time = np.arange(4 * 4 * 8)
    phase = 2 * np.pi * time / 16
    periodic = np.column_stack(
        [np.sin(phase), np.cos(phase), 0.5 * np.sin(2 * phase)]
    )
    result = _analyze_scale(
        periodic,
        frame_hop_seconds=0.5,
        bar_seconds=2.0,
        scale_bars=4,
        seed=7,
        config=config,
    )
    assert result["h1_surrogate_excess"] > 0.05
