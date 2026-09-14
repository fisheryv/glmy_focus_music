from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from generation.pitch3_evaluation import (  # noqa: E402
    _auc,
    _calibration_threshold,
    _group_regression_gates,
    _qualification_gates,
)


def test_pitch3_auc_handles_ties() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert _auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert _auc(labels, np.full(4, 0.5)) == 0.5


def test_pitch3_calibration_threshold_preserves_each_correction_step() -> None:
    labels = np.asarray(([0.0] * 20 + [1.0] * 5) * 3)
    steps = np.repeat(np.asarray([4, 5, 6]), 25)
    probabilities = np.concatenate([np.linspace(0.01, 0.20, 20), np.linspace(0.8, 0.9, 5)] * 3)
    threshold, by_step = _calibration_threshold(labels, probabilities, steps)
    assert 0.18 <= threshold <= 0.20
    assert all(by_step[str(step)]["id_acceptance_rate"] >= 0.95 for step in (4, 5, 6))


def test_pitch3_qualification_gates_are_frozen() -> None:
    metrics = {
        "focus_logit_spearman": 0.70,
        "coordinate_spearman": [0.50, 0.60, 0.70],
        "coordinate_median_spearman": 0.60,
        "target_band_distance_spearman": 0.50,
        "quartile_ranking_accuracy": 0.65,
        "interval_90_coverage": 0.90,
        "ood_auroc": 0.80,
        "ood_sensitivity": 0.80,
    }
    by_step = {str(step): {"id_acceptance_rate": 0.95} for step in (4, 5, 6)}
    assert all(_qualification_gates(metrics, by_step).values())

    metrics["coordinate_spearman"] = [0.49, 0.60, 0.70]
    gates = _qualification_gates(metrics, by_step)
    assert gates["each_coordinate_spearman"] is False


def test_pitch3_group_regression_gates_reuse_frozen_thresholds() -> None:
    metrics = {
        "focus_logit_spearman": 0.70,
        "coordinate_spearman": [0.50, 0.60, 0.70],
        "coordinate_median_spearman": 0.60,
        "target_band_distance_spearman": 0.50,
        "quartile_ranking_accuracy": 0.65,
    }
    assert all(_group_regression_gates(metrics).values())
    metrics["target_band_distance_spearman"] = 0.49
    assert _group_regression_gates(metrics)["target_band_distance_spearman"] is False
