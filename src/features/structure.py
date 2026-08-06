from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contracts import StructureFeatures


def self_similarity_matrix(
    vectors: ArrayLike, valid: ArrayLike | None = None
) -> NDArray[np.float32]:
    """Build a cosine self-similarity matrix from a frame-wise feature sequence."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("vectors must have shape (n_frames, n_features)")
    frame_valid = (
        np.ones(values.shape[0], dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool).reshape(-1)
    )
    if frame_valid.size != values.shape[0]:
        raise ValueError("valid must contain one value per frame")

    finite = np.all(np.isfinite(values), axis=1) & frame_valid
    standardized = np.zeros_like(values)
    if np.any(finite):
        center = np.median(values[finite], axis=0)
        scale = np.median(np.abs(values[finite] - center), axis=0) * 1.4826
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        standardized[finite] = (values[finite] - center) / scale
    # A bias coordinate prevents frames exactly at the robust center from
    # collapsing to a zero vector (constant passages should remain self-similar).
    embedded = np.column_stack([standardized, finite.astype(np.float64)])
    norms = np.linalg.norm(embedded, axis=1, keepdims=True)
    normalized = np.divide(
        embedded,
        norms,
        out=np.zeros_like(embedded),
        where=norms > 1e-12,
    )
    similarity = normalized @ normalized.T
    similarity = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
    similarity[~finite, :] = 0.0
    similarity[:, ~finite] = 0.0
    similarity[np.diag_indices_from(similarity)] = finite.astype(float)
    return similarity.astype(np.float32)


def checkerboard_novelty(
    similarity: ArrayLike,
    *,
    kernel_size: int,
) -> NDArray[np.float32]:
    """Compute Foote-style boundary novelty along an SSM diagonal."""

    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    frame_count = matrix.shape[0]
    novelty = np.zeros(frame_count, dtype=np.float64)
    for index in range(kernel_size, frame_count - kernel_size):
        before = slice(index - kernel_size, index)
        after = slice(index, index + kernel_size)
        same = np.sum(matrix[before, before]) + np.sum(matrix[after, after])
        different = np.sum(matrix[before, after]) + np.sum(matrix[after, before])
        novelty[index] = (same - different) / (2.0 * kernel_size * kernel_size)
    novelty = np.maximum(novelty, 0.0)
    if frame_count >= 3:
        novelty = np.convolve(novelty, np.asarray([0.25, 0.5, 0.25]), mode="same")
    maximum = float(np.max(novelty)) if novelty.size else 0.0
    if maximum > 1e-12:
        novelty /= maximum
    return novelty.astype(np.float32)


def structural_boundaries(
    novelty: ArrayLike,
    *,
    min_segment_frames: int,
    max_segment_frames: int,
    threshold: float = 1.5,
) -> NDArray[np.int32]:
    """Pick novelty peaks while enforcing useful macro-section durations."""

    values = np.asarray(novelty, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("novelty must not be empty")
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be positive")
    if max_segment_frames < min_segment_frames:
        raise ValueError("max_segment_frames must be at least min_segment_frames")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    cutoff = median + threshold * max(mad, 1e-8)
    candidates = [
        index
        for index in range(1, values.size - 1)
        if values[index] >= cutoff
        and values[index] >= values[index - 1]
        and values[index] > values[index + 1]
        and min_segment_frames <= index <= values.size - min_segment_frames
    ]
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda item: (-values[item], item)):
        if all(abs(candidate - existing) >= min_segment_frames for existing in selected):
            selected.append(candidate)
    boundaries = [0, *sorted(selected), values.size]

    # Very homogeneous recordings may have no strong peak. Split overlong spans at
    # their best available local novelty so the structural path remains observable.
    output = [boundaries[0]]
    for right in boundaries[1:]:
        left = output[-1]
        while right - left > max_segment_frames:
            target = min(left + max_segment_frames, right - min_segment_frames)
            search_left = max(left + min_segment_frames, target - min_segment_frames)
            search_right = min(right - min_segment_frames, target + min_segment_frames)
            if search_right >= search_left:
                local = values[search_left : search_right + 1]
                split = search_left + int(np.argmax(local))
            else:
                split = target
            output.append(split)
            left = split
        output.append(right)
    return np.asarray(sorted(set(output)), dtype=np.int32)


def structural_features(
    vectors: ArrayLike,
    times: ArrayLike,
    valid: ArrayLike,
    *,
    duration: float,
    kernel_seconds: float = 8.0,
    min_segment_seconds: float = 8.0,
    max_segment_seconds: float = 45.0,
    novelty_threshold: float = 1.5,
) -> StructureFeatures:
    """Convert short-time acoustic features into SSM-delimited macro sections."""

    values = np.asarray(vectors, dtype=np.float32)
    frame_times = np.asarray(times, dtype=np.float64).reshape(-1)
    frame_valid = np.asarray(valid, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[0] != frame_times.size:
        raise ValueError("vectors and times must contain the same number of frames")
    if frame_valid.size != frame_times.size:
        raise ValueError("valid must contain one value per frame")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if frame_times.size > 1:
        hop = float(np.median(np.diff(frame_times)))
    else:
        hop = duration
    hop = max(hop, np.finfo(float).eps)
    similarity = self_similarity_matrix(values, frame_valid)
    novelty = checkerboard_novelty(
        similarity,
        kernel_size=max(1, int(round(kernel_seconds / hop))),
    )
    boundary_indices = structural_boundaries(
        novelty,
        min_segment_frames=max(1, int(round(min_segment_seconds / hop))),
        max_segment_frames=max(1, int(round(max_segment_seconds / hop))),
        threshold=novelty_threshold,
    )

    boundary_times = np.empty(boundary_indices.size, dtype=np.float32)
    boundary_times[0] = 0.0
    boundary_times[-1] = float(duration)
    for index in range(1, boundary_indices.size - 1):
        left = boundary_indices[index] - 1
        right = boundary_indices[index]
        boundary_times[index] = float((frame_times[left] + frame_times[right]) / 2.0)

    blocks: list[np.ndarray] = []
    block_valid: list[bool] = []
    for left, right in zip(boundary_indices[:-1], boundary_indices[1:], strict=True):
        mask = frame_valid[left:right] & np.all(np.isfinite(values[left:right]), axis=1)
        if np.any(mask):
            blocks.append(np.mean(values[left:right][mask], axis=0, dtype=np.float64))
            block_valid.append(True)
        else:
            blocks.append(np.zeros(values.shape[1], dtype=np.float64))
            block_valid.append(False)
    block_vectors = np.asarray(blocks, dtype=np.float32)
    block_times = ((boundary_times[:-1] + boundary_times[1:]) / 2.0).astype(np.float32)
    return StructureFeatures(
        times=block_times,
        boundary_times=boundary_times,
        boundary_indices=boundary_indices,
        self_similarity=similarity,
        novelty=novelty,
        block_vectors=block_vectors,
        valid=np.asarray(block_valid, dtype=np.bool_),
    )


def abstract_structure_states(
    block_vectors: ArrayLike,
    valid: ArrayLike | None = None,
    *,
    similarity_threshold: float = 0.75,
) -> list[int | None]:
    """Assign deterministic per-track recurrent labels to macro sections."""

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [0, 1]")
    values = np.asarray(block_vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("block_vectors must be two-dimensional")
    block_valid = (
        np.ones(values.shape[0], dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool).reshape(-1)
    )
    if block_valid.size != values.shape[0]:
        raise ValueError("valid must contain one value per block")
    usable = block_valid & np.all(np.isfinite(values), axis=1)
    if not np.any(usable):
        return [None] * values.shape[0]
    # Mean/std keeps a repeated (majority) section away from the origin; a
    # median/MAD transform would collapse that common section to a zero vector.
    center = np.mean(values[usable], axis=0)
    scale = np.std(values[usable], axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    normalized = (values - center) / scale
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = np.divide(normalized, norms, out=np.zeros_like(normalized), where=norms > 1e-12)

    centroids: list[np.ndarray] = []
    counts: list[int] = []
    states: list[int | None] = []
    for index, vector in enumerate(normalized):
        if not usable[index]:
            states.append(None)
            continue
        if centroids:
            similarities = np.asarray([float(vector @ centroid) for centroid in centroids])
            state = int(np.argmax(similarities))
        else:
            similarities = np.empty(0)
            state = 0
        if not centroids or float(similarities[state]) < similarity_threshold:
            state = len(centroids)
            centroids.append(vector.copy())
            counts.append(1)
        else:
            counts[state] += 1
            centroid = centroids[state] + (vector - centroids[state]) / counts[state]
            norm = float(np.linalg.norm(centroid))
            centroids[state] = centroid / norm if norm > 1e-12 else centroid
        states.append(state)
    return states


__all__ = [
    "abstract_structure_states",
    "checkerboard_novelty",
    "self_similarity_matrix",
    "structural_boundaries",
    "structural_features",
]
