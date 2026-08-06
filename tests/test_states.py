import numpy as np

from focus_topology.features.states import dominant_pitch_states, modulation_states, quantize_1d


def test_dominant_pitch_uses_uncertain_state() -> None:
    chroma = np.zeros((2, 12))
    chroma[0, 4] = 1.0
    chroma[0, 5] = 0.1
    chroma[1, 4] = 1.0
    chroma[1, 5] = 0.99

    assert dominant_pitch_states(chroma) == [4, 12]


def test_quantization_preserves_missing_values() -> None:
    assert quantize_1d([0.0, float("nan"), 1.0], edges=[0.5]) == [0, -1, 1]


def test_modulation_states_are_hashable_tuples() -> None:
    states = modulation_states([[0.0, 1.0], [1.0, 0.0]], levels=2)

    assert all(isinstance(state, tuple) for state in states)
    assert len(set(states)) == 2

