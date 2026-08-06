from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def tonnetz_basis(n_chroma: int = 12) -> FloatArray:
    """Return the fixed six-dimensional Harte/librosa Tonnetz basis."""

    if n_chroma < 1:
        raise ValueError("n_chroma must be positive")
    pitch_classes = np.linspace(0.0, 12.0, num=n_chroma, endpoint=False)
    scales = np.asarray([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])
    phases = np.multiply.outer(scales, pitch_classes)
    phases[::2] -= 0.5
    radii = np.asarray([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    return radii[:, None] * np.cos(np.pi * phases)


def normalize_chroma(chroma: ArrayLike, *, epsilon: float = 1e-12) -> FloatArray:
    """L1-normalize row-wise chroma vectors without creating non-finite values."""

    values = np.asarray(chroma, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("chroma must have shape (n_steps, 12)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    totals = np.sum(np.maximum(values, 0.0), axis=1, keepdims=True)
    return np.divide(
        np.maximum(values, 0.0),
        totals,
        out=np.zeros_like(values),
        where=totals > epsilon,
    )


def chroma_to_tonnetz(chroma: ArrayLike) -> FloatArray:
    """Project row-wise 12-bin chroma vectors into six Tonnetz coordinates."""

    normalized = normalize_chroma(chroma)
    return normalized @ tonnetz_basis(normalized.shape[1]).T


def assign_codebook(
    tonnetz: ArrayLike,
    centers: ArrayLike,
    *,
    valid: ArrayLike | None = None,
    missing_state: int = -1,
) -> NDArray[np.int16]:
    """Assign Tonnetz vectors to frozen centers and mask invalid observations."""

    values = np.asarray(tonnetz, dtype=np.float64)
    codebook = np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("tonnetz must have shape (n_steps, 6)")
    if codebook.ndim != 2 or codebook.shape[1] != 6 or codebook.shape[0] < 2:
        raise ValueError("centers must have shape (n_states, 6) with n_states >= 2")
    mask = np.all(np.isfinite(values), axis=1)
    if valid is not None:
        supplied = np.asarray(valid, dtype=bool)
        if supplied.shape != (values.shape[0],):
            raise ValueError("valid must have shape (n_steps,)")
        mask &= supplied
    states = np.full(values.shape[0], missing_state, dtype=np.int16)
    if np.any(mask):
        distances = np.sum(
            (values[mask, None, :] - codebook[None, :, :]) ** 2,
            axis=2,
        )
        states[mask] = np.argmin(distances, axis=1).astype(np.int16)
    return states


def tonnetz_similarity(tonnetz: ArrayLike, *, scale: float | None = None) -> FloatArray:
    """Gaussian self-similarity matrix from Euclidean Tonnetz distances."""

    values = np.asarray(tonnetz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("tonnetz must have shape (n_steps, 6)")
    squared = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
    if scale is None:
        positive = squared[squared > np.finfo(float).eps]
        scale = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return np.exp(-squared / (2.0 * scale**2))


__all__ = [
    "assign_codebook",
    "chroma_to_tonnetz",
    "normalize_chroma",
    "tonnetz_basis",
    "tonnetz_similarity",
]
