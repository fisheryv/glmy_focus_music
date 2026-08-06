import numpy as np
import pytest

from focus_topology.generation.rerank import Candidate, rerank_candidates
from focus_topology.generation.steering import apply_linear_steering, steering_scale


def test_reranker_selects_nearest_candidate() -> None:
    candidates = [
        Candidate("far", np.array([2.0, 0.0])),
        Candidate("near", np.array([0.2, 0.0])),
    ]

    ranked = rerank_candidates(candidates, np.zeros(2))

    assert ranked[0].candidate.candidate_id == "near"


def test_steering_window_is_zero_outside_and_peaks_at_center() -> None:
    assert steering_scale(0.1) == 0.0
    assert steering_scale(0.9) == 0.0
    assert steering_scale((0.25 + 0.8) / 2) == pytest.approx(0.15)


def test_linear_steering_moves_in_target_direction() -> None:
    latent = np.zeros((1, 2, 3))
    direction = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    moved = apply_linear_steering(
        latent,
        direction,
        target_descriptor=[1.0, 2.0],
        current_descriptor=[0.0, 0.0],
        scale=0.5,
    )

    assert moved[0, 0].tolist() == pytest.approx([0.5, 1.0, 0.0])

