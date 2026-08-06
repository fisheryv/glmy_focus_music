from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


@dataclass(frozen=True, slots=True)
class SharedSMPTransform:
    """Discovery-fitted transform shared by every SMP prototype codebook."""

    frequencies: np.ndarray
    robust_center: np.ndarray
    robust_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        values = np.asarray(spectra, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.frequencies.size:
            raise ValueError("spectra and shared SMP transform have incompatible shapes")
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("SMP spectra must be finite and non-negative")
        hellinger = np.sqrt(values)
        robust = (hellinger - self.robust_center) / self.robust_scale
        return (robust - self.pca_mean) @ self.pca_components.T


@dataclass(frozen=True, slots=True)
class SMPPrototypeCodebook:
    state_count: int
    centers: np.ndarray
    prototype_spectra: np.ndarray
    spectral_centroids_hz: np.ndarray
    training_state_counts: np.ndarray

    def predict(self, embedded: np.ndarray) -> np.ndarray:
        values = np.asarray(embedded, dtype=np.float64)
        distances = np.sum((values[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1).astype(np.int16)


def fit_shared_transform(
    spectra: np.ndarray,
    frequencies: np.ndarray,
    *,
    n_components: int,
    random_seed: int,
) -> tuple[SharedSMPTransform, np.ndarray]:
    values = np.asarray(spectra, dtype=np.float64)
    frequency_values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != frequency_values.size:
        raise ValueError("training spectra and frequencies have incompatible shapes")
    if not 1 <= n_components <= min(values.shape):
        raise ValueError("invalid PCA component count")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("training SMP spectra must be finite and non-negative")

    hellinger = np.sqrt(values)
    center = np.median(hellinger, axis=0)
    q1, q3 = np.quantile(hellinger, (0.25, 0.75), axis=0)
    scale = q3 - q1
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("one or more SMP frequency bins have zero discovery IQR")
    robust = (hellinger - center) / scale
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_seed,
    )
    embedded = pca.fit_transform(robust)
    transform = SharedSMPTransform(
        frequencies=frequency_values,
        robust_center=center,
        robust_scale=scale,
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )
    return transform, embedded


def fit_codebook(
    embedded: np.ndarray,
    original_spectra: np.ndarray,
    frequencies: np.ndarray,
    *,
    state_count: int,
    random_seed: int,
) -> SMPPrototypeCodebook:
    values = np.asarray(embedded, dtype=np.float64)
    spectra = np.asarray(original_spectra, dtype=np.float64)
    frequency_values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 2 or spectra.ndim != 2 or values.shape[0] != spectra.shape[0]:
        raise ValueError("embedded values and original spectra have incompatible shapes")
    if state_count < 2 or state_count > values.shape[0]:
        raise ValueError("invalid SMP state count")

    estimator = MiniBatchKMeans(
        n_clusters=state_count,
        batch_size=1024,
        n_init=20,
        max_iter=300,
        reassignment_ratio=0.0,
        random_state=random_seed,
    )
    raw_labels = estimator.fit_predict(values)
    prototypes = np.vstack(
        [np.mean(spectra[raw_labels == state], axis=0) for state in range(state_count)]
    )
    masses = np.sum(prototypes, axis=1)
    centroids = np.divide(
        prototypes @ frequency_values,
        masses,
        out=np.zeros(state_count, dtype=np.float64),
        where=masses > 0.0,
    )
    order = np.argsort(centroids, kind="stable")
    inverse = np.empty(state_count, dtype=np.int16)
    inverse[order] = np.arange(state_count, dtype=np.int16)
    labels = inverse[raw_labels]
    return SMPPrototypeCodebook(
        state_count=state_count,
        centers=np.asarray(estimator.cluster_centers_[order], dtype=np.float64),
        prototype_spectra=np.asarray(prototypes[order], dtype=np.float64),
        spectral_centroids_hz=np.asarray(centroids[order], dtype=np.float64),
        training_state_counts=np.bincount(labels, minlength=state_count).astype(np.int64),
    )


def assign_states(
    spectra: np.ndarray,
    valid: Iterable[bool],
    transform: SharedSMPTransform,
    codebook: SMPPrototypeCodebook,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    mask = np.asarray(list(valid), dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[0],):
        raise ValueError("spectra and valid mask have incompatible shapes")
    states = np.full(values.shape[0], -1, dtype=np.int16)
    usable = mask & np.all(np.isfinite(values), axis=1) & np.all(values >= 0.0, axis=1)
    if np.any(usable):
        states[usable] = codebook.predict(transform.transform(values[usable]))
    return states
