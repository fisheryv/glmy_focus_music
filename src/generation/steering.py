from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def steering_scale(
    progress: float,
    *,
    start: float = 0.25,
    end: float = 0.80,
    maximum: float = 0.15,
) -> float:
    """Triangular steering window over normalized denoising progress."""

    if not 0.0 <= start < end <= 1.0:
        raise ValueError("expected 0 <= start < end <= 1")
    if maximum < 0:
        raise ValueError("maximum cannot be negative")
    if progress < start or progress > end:
        return 0.0
    center = (start + end) / 2.0
    half_width = (end - start) / 2.0
    return maximum * max(0.0, 1.0 - abs(progress - center) / half_width)


def apply_linear_steering(
    latent: ArrayLike,
    direction_matrix: ArrayLike,
    target_descriptor: ArrayLike,
    current_descriptor: ArrayLike,
    *,
    scale: float,
    max_update_norm: float | None = None,
) -> NDArray[np.float64]:
    """Apply model-independent direction steering to a latent tensor.

    ``direction_matrix`` has shape ``(latent_features, topology_features)``.
    The resulting feature update is broadcast across all leading latent axes.
    """

    latent_array = np.asarray(latent, dtype=float)
    matrix = np.asarray(direction_matrix, dtype=float)
    target = np.asarray(target_descriptor, dtype=float).reshape(-1)
    current = np.asarray(current_descriptor, dtype=float).reshape(-1)
    if target.shape != current.shape:
        raise ValueError("target and current descriptors must have the same shape")
    if matrix.ndim != 2 or matrix.shape != (latent_array.shape[-1], target.size):
        raise ValueError("direction_matrix shape must be (latent_features, topology_features)")
    if scale < 0:
        raise ValueError("scale cannot be negative")

    update = matrix @ (target - current)
    norm = float(np.linalg.norm(update))
    if max_update_norm is not None:
        if max_update_norm <= 0 or not math.isfinite(max_update_norm):
            raise ValueError("max_update_norm must be finite and positive")
        if norm > max_update_norm:
            update = update * (max_update_norm / norm)
    return latent_array + scale * update

