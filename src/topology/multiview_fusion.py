from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

from topology.statistics import _pseudo_f_statistic


@dataclass(slots=True)
class DiscoveryMahalanobisBlock:
    """Discovery-fitted, rank-normalized Mahalanobis coordinates for one view."""

    imputer: SimpleImputer | None = None
    keep: np.ndarray | None = None
    mean: np.ndarray | None = None
    whitening: np.ndarray | None = None
    effective_rank: int = 0

    def fit(self, matrix: np.ndarray) -> DiscoveryMahalanobisBlock:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("matrix must have at least two rows")
        self.imputer = SimpleImputer(strategy="median").fit(values)
        imputed = self.imputer.transform(values)
        variances = np.var(imputed, axis=0)
        self.keep = np.isfinite(variances) & (variances > np.finfo(float).eps)
        if not np.any(self.keep):
            raise ValueError("block has no non-constant dimensions")
        retained = imputed[:, self.keep]
        self.mean = np.mean(retained, axis=0)
        centered = retained - self.mean
        covariance = np.atleast_2d(np.cov(centered, rowvar=False))
        inverse = np.linalg.pinv(covariance, rcond=1e-10)
        eigenvalues, eigenvectors = np.linalg.eigh(inverse)
        positive = eigenvalues > np.finfo(float).eps
        self.whitening = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
        self.effective_rank = int(np.linalg.matrix_rank(covariance))
        if self.effective_rank < 1 or self.whitening.shape[1] < 1:
            raise ValueError("block covariance has zero effective rank")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.imputer, self.keep, self.mean, self.whitening)):
            raise RuntimeError("block transformer is not fitted")
        assert self.imputer is not None
        assert self.keep is not None
        assert self.mean is not None
        assert self.whitening is not None
        imputed = self.imputer.transform(np.asarray(matrix, dtype=float))
        centered = imputed[:, self.keep] - self.mean
        return (centered @ self.whitening) / np.sqrt(self.effective_rank)


def equal_block_fusion(blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenate blocks so each block has equal expected squared-distance weight."""

    if not blocks:
        raise ValueError("at least one block is required")
    rows = {np.asarray(block).shape[0] for block in blocks}
    if len(rows) != 1:
        raise ValueError("all blocks must contain the same rows")
    weight = 1.0 / np.sqrt(len(blocks))
    return np.concatenate([np.asarray(block, dtype=float) * weight for block in blocks], axis=1)


def hierarchical_fusion(
    local: np.ndarray,
    structure: np.ndarray,
    *,
    structure_weight: float = 0.5,
) -> np.ndarray:
    """Fuse an already normalized local block with a macro-structure block."""

    if not 0.0 <= structure_weight <= 1.0:
        raise ValueError("structure_weight must be in [0, 1]")
    local_values = np.asarray(local, dtype=float)
    structure_values = np.asarray(structure, dtype=float)
    if local_values.shape[0] != structure_values.shape[0]:
        raise ValueError("local and structure blocks must contain the same rows")
    parts: list[np.ndarray] = []
    if structure_weight < 1.0:
        parts.append(local_values * np.sqrt(1.0 - structure_weight))
    if structure_weight > 0.0:
        parts.append(structure_values * np.sqrt(structure_weight))
    return np.concatenate(parts, axis=1)


def permutation_pseudo_f(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    groups = np.asarray(labels)
    observed = _pseudo_f_statistic(values, groups)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(groups)
        exceedances += _pseudo_f_statistic(values, permuted) >= observed
    return {
        "pseudo_f": float(observed),
        "p_value": float((exceedances + 1) / (permutations + 1)),
    }


def paired_incremental_permutation(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """One-sided paired permutation test for a positive pseudo-F increment."""

    candidate_values = np.asarray(candidate, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    groups = np.asarray(labels)
    observed = _pseudo_f_statistic(candidate_values, groups) - _pseudo_f_statistic(
        baseline_values, groups
    )
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        permuted = rng.permutation(groups)
        null[index] = _pseudo_f_statistic(candidate_values, permuted) - _pseudo_f_statistic(
            baseline_values, permuted
        )
    return {
        "delta_pseudo_f": float(observed),
        "p_value_one_sided": float((np.count_nonzero(null >= observed) + 1) / (permutations + 1)),
        "null_ci_low": float(np.quantile(null, 0.025)),
        "null_ci_high": float(np.quantile(null, 0.975)),
    }


def stratified_bootstrap_differences(
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_predictions: np.ndarray,
    baseline_probabilities: np.ndarray,
    *,
    metric_functions: dict[str, Any],
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    groups = np.asarray(labels)
    strata = {label: np.flatnonzero(groups == label) for label in np.unique(groups)}
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(resamples, dtype=float) for name in metric_functions}
    for index in range(resamples):
        selected = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in strata.values()]
        )
        for name, function in metric_functions.items():
            candidate_score = function(
                groups[selected],
                candidate_predictions[selected],
                candidate_probabilities[selected],
            )
            baseline_score = function(
                groups[selected],
                baseline_predictions[selected],
                baseline_probabilities[selected],
            )
            samples[name][index] = candidate_score - baseline_score
    return {
        name: {
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "positive_fraction": float(np.mean(values > 0.0)),
        }
        for name, values in samples.items()
    }
