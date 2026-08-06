from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dominant_pitch_states(
    chroma: ArrayLike,
    *,
    uncertainty_ratio: float = 1.15,
    silence_floor: float = 1e-8,
    uncertain_state: int = 12,
) -> list[int]:
    """Convert frame-wise 12-bin chroma to pitch-class states.

    A frame is uncertain when its strongest bin is too close to the runner-up,
    or when the frame contains essentially no energy. Input shape is
    ``(n_frames, 12)``.
    """

    values = np.asarray(chroma, dtype=float)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("chroma must have shape (n_frames, 12)")
    if uncertainty_ratio <= 1.0:
        raise ValueError("uncertainty_ratio must be greater than 1")

    states: list[int] = []
    for frame in values:
        if not np.all(np.isfinite(frame)) or float(np.max(frame)) <= silence_floor:
            states.append(uncertain_state)
            continue
        top_two = np.partition(frame, -2)[-2:]
        strongest = float(top_two[-1])
        runner_up = max(float(top_two[-2]), silence_floor)
        states.append(
            int(np.argmax(frame)) if strongest / runner_up >= uncertainty_ratio else uncertain_state
        )
    return states


def quantile_edges(values: ArrayLike, n_bins: int) -> NDArray[np.float64]:
    """Fit deterministic internal quantile edges for one-dimensional data."""

    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("cannot fit bins without finite values")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.unique(edges).astype(float)


def quantize_1d(
    values: ArrayLike,
    *,
    n_bins: int = 10,
    edges: ArrayLike | None = None,
    missing_state: int = -1,
) -> list[int]:
    """Quantize a scalar sequence using supplied or fitted quantile edges.

    Confirmation experiments should pass edges fitted on the discovery split;
    fitting separately on validation data would leak distribution information.
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    fitted_edges = (
        quantile_edges(array, n_bins) if edges is None else np.asarray(edges, dtype=float)
    )
    result = np.full(array.shape, missing_state, dtype=int)
    finite = np.isfinite(array)
    result[finite] = np.digitize(array[finite], fitted_edges, right=False)
    return result.tolist()


def modulation_states(
    band_energies: ArrayLike,
    *,
    levels: int = 3,
    edges_by_band: Sequence[ArrayLike] | None = None,
) -> list[tuple[int, ...]]:
    """Quantize modulation-band energies into hashable multiband states."""

    values = np.asarray(band_energies, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("band_energies must have shape (n_frames, n_bands)")
    if edges_by_band is not None and len(edges_by_band) != values.shape[1]:
        raise ValueError("edges_by_band must contain one edge array per band")

    per_band: list[list[int]] = []
    for band_index in range(values.shape[1]):
        edges = None if edges_by_band is None else edges_by_band[band_index]
        per_band.append(quantize_1d(values[:, band_index], n_bins=levels, edges=edges))
    return [tuple(band[frame] for band in per_band) for frame in range(values.shape[0])]
