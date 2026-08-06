from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.batch import _sha256, _write_npz_atomic
from graphs.transition import build_transition_graph
from topology.batch import (
    TopologyConfig,
    TopologyJob,
    _graph_metrics,
    _load_state_sequence,
    _process_job,
)
from topology.hypothesis import (
    normalized_profile_dispersion,
    persistence_coordinates,
    run_hypothesis_tests,
    select_group_medoids,
)
from topology.statistics import (
    ANALYSIS_SETS,
    TOPOLOGY_METRICS,
    _bootstrap_rank_biserial_interval,
    _classification_metrics,
    _omnibus_and_pairwise,
    benjamini_hochberg,
    permanova_mahalanobis,
)


def test_modulation_tertile_loads_one_dimensional_states(tmp_path: Path) -> None:
    path = tmp_path / "modulation_tertile.npz"
    _write_npz_atomic(path, {"states": np.asarray([0, 1, -1, 2], dtype=np.int8)})

    assert _load_state_sequence(path, "modulation_tertile") == [0, 1, None, 2]


def test_classification_metrics_support_binary_probabilities() -> None:
    classes = np.asarray(["classical", "focus"])
    y_true = np.asarray(["classical", "classical", "focus", "focus"])
    y_pred = y_true.copy()
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=float)

    metrics = _classification_metrics(y_true, y_pred, probabilities, classes)

    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["macro_auroc_ovr"] == pytest.approx(1.0)
    assert metrics["macro_auprc"] == pytest.approx(1.0)


def test_benjamini_hochberg_is_monotone_in_rank_order() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])

    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_omnibus_and_pairwise_support_four_groups() -> None:
    rows = []
    for _, split, scale, _ in ANALYSIS_SETS:
        for group_index, group in enumerate(("a", "b", "c", "d")):
            for replicate in range(3):
                row = {
                    "segment_id": f"{split}-{scale}-{group}-{replicate}",
                    "track_id": f"{group}-{replicate}",
                    "group": group,
                    "split": split,
                    "scale_seconds": scale,
                    "view": "rhythm",
                }
                row.update(
                    {
                        metric: float(group_index + replicate / 10 + metric_index / 100)
                        for metric_index, metric in enumerate(TOPOLOGY_METRICS)
                    }
                )
                rows.append(row)

    omnibus, pairwise = _omnibus_and_pairwise(pd.DataFrame(rows))

    assert len(omnibus) == len(ANALYSIS_SETS) * len(TOPOLOGY_METRICS)
    assert len(pairwise) == len(ANALYSIS_SETS) * len(TOPOLOGY_METRICS) * 6
    assert {"n_a", "n_b", "n_c", "n_d"}.issubset(omnibus.columns)
    assert {
        "rank_biserial_ci95_low",
        "rank_biserial_ci95_high",
        "bootstrap_resamples",
        "bootstrap_seed",
    }.issubset(pairwise.columns)


def test_rank_biserial_bootstrap_interval_is_deterministic() -> None:
    first = np.asarray([3.0, 4.0, 5.0, 6.0])
    second = np.asarray([0.0, 1.0, 2.0, 3.0])

    first_interval = _bootstrap_rank_biserial_interval(
        first,
        second,
        resamples=500,
        seed=17,
    )
    second_interval = _bootstrap_rank_biserial_interval(
        first,
        second,
        resamples=500,
        seed=17,
    )

    assert first_interval == second_interval
    assert -1.0 <= first_interval[0] <= first_interval[1] <= 1.0


def test_permanova_is_deterministic_with_discovery_reference() -> None:
    reference = np.asarray([[0.0, 0.1], [0.2, 0.0], [1.8, 2.0], [2.1, 1.9]])
    values = np.asarray([[0.0, 0.0], [0.1, 0.2], [2.0, 2.0], [2.2, 1.8]])
    labels = ["a", "a", "b", "b"]

    first = permanova_mahalanobis(
        values, labels, permutations=99, seed=7, reference_matrix=reference
    )
    second = permanova_mahalanobis(
        values, labels, permutations=99, seed=7, reference_matrix=reference
    )

    assert first == second
    assert first["pseudo_f"] > 1
    assert 0 < first["p_value"] <= 1


def test_path_entropy_and_directed_recurrence_follow_preregistered_formulas() -> None:
    states = [0, 1, 0, 1]
    metrics = _graph_metrics(states, build_transition_graph(states, top_k=6))

    assert metrics["path_entropy"] == 0.0
    assert metrics["path_entropy_normalized"] == 0.0
    assert metrics["directed_recurrence"] == pytest.approx(5 / 9)
    assert metrics["directed_recurrence_unbiased"] == pytest.approx(1 / 3)


def test_profile_dispersion_is_threshold_span_normalized() -> None:
    thresholds = np.asarray([0.9, 0.7, 0.5])
    curve = np.asarray([0.0, 1.0, 0.0])
    center = np.zeros(3)

    assert normalized_profile_dispersion(curve, center, thresholds) == pytest.approx(0.5)


def test_superlevel_persistence_coordinates_are_conventional() -> None:
    assert persistence_coordinates(0.9, 0.5, terminal_threshold=0.05) == pytest.approx((0.1, 0.5))
    assert persistence_coordinates(0.9, None, terminal_threshold=0.05) == pytest.approx((0.1, 0.95))


def test_hypothesis_family_handles_direction_fdr_and_zero_inflation() -> None:
    rows = []
    for scale in (180.0, 300.0):
        for view in ("modulation", "pitch", "rhythm"):
            for group, entropy, recurrence, dispersion in (
                ("focus", 0.0, 1.0, 0.0),
                ("pop", 1.0, 0.2, 1.0),
                ("classical", 2.0, 0.1, 2.0),
            ):
                for index in range(12):
                    rows.append(
                        {
                            "segment_id": f"{group}_{view}_{int(scale)}_{index}",
                            "track_id": f"{group}_{index}",
                            "group": group,
                            "split": "validation",
                            "scale_seconds": scale,
                            "view": view,
                            "path_entropy": entropy,
                            "path_entropy_normalized": entropy,
                            "directed_recurrence": recurrence,
                            "directed_recurrence_unbiased": recurrence,
                            "primary_h1_profile_dispersion": dispersion,
                            "sensitivity_h1_profile_dispersion": dispersion,
                        }
                    )
    first, first_verdict = run_hypothesis_tests(
        pd.DataFrame(rows), fdr_q=0.1, bootstrap_resamples=100, seed=7
    )
    second, second_verdict = run_hypothesis_tests(
        pd.DataFrame(rows), fdr_q=0.1, bootstrap_resamples=100, seed=7
    )

    core = first[first["analysis_role"] == "confirmatory_core"]
    assert first_verdict == second_verdict == "supported"
    assert (core["verdict"] == "supported").all()
    pd.testing.assert_frame_equal(first, second)


def test_group_medoid_ties_are_broken_by_segment_id() -> None:
    rows = []
    curves = {}
    for group in ("focus", "pop", "classical"):
        for segment_id in (f"{group}_b", f"{group}_a"):
            row = {
                "segment_id": segment_id,
                "track_id": segment_id,
                "group": group,
                "split": "validation",
                "scale_seconds": 180.0,
                "view": "modulation",
                "path_entropy": 1.0,
                "directed_recurrence": 0.5,
            }
            rows.append(row)
            curves[(segment_id, segment_id, group, "validation", 180.0, "modulation")] = np.zeros(3)

    representatives = select_group_medoids(pd.DataFrame(rows), curves)

    assert [row["segment_id"] for row in representatives] == [
        "focus_a",
        "pop_a",
        "classical_a",
    ]


def test_topology_job_is_resumable_and_hash_verified(tmp_path: Path) -> None:
    feature_path = tmp_path / "features" / "chroma" / "states.npz"
    _write_npz_atomic(
        feature_path,
        {
            "states": np.asarray([0, 1, 2, 0, 1, 2, 0], dtype=np.int16),
            "valid": np.ones(7, dtype=bool),
        },
    )
    job = TopologyJob(
        segment_id="synthetic__180s",
        track_id="synthetic",
        group="focus",
        split="discovery",
        scale_seconds=180.0,
        view="pitch",
        feature_relative_path=feature_path.relative_to(tmp_path).as_posix(),
        feature_sha256=_sha256(feature_path),
    )
    config = TopologyConfig(thresholds=(0.95, 0.5))

    first = _process_job(
        job,
        root=tmp_path,
        config=config,
        config_sha256="config-hash",
        overwrite=False,
    )
    second = _process_job(
        job,
        root=tmp_path,
        config=config,
        config_sha256="config-hash",
        overwrite=False,
    )
    regenerated = _process_job(
        job,
        root=tmp_path,
        config=config,
        config_sha256="config-hash",
        overwrite=True,
    )

    assert first["status"] == "success"
    assert first["h1_betti_max"] == 1
    assert second["status"] == "verified_existing"
    assert second["persistence_sha256"] == first["persistence_sha256"]
    assert regenerated["persistence_sha256"] == first["persistence_sha256"]
    assert regenerated["sensitivity_persistence_sha256"] == first["sensitivity_persistence_sha256"]
    assert regenerated["graph_sha256"] == first["graph_sha256"]

    primary = np.load(tmp_path / first["persistence_relative_path"])
    sensitivity = np.load(tmp_path / first["sensitivity_persistence_relative_path"])
    sensitivity_lookup = {
        float(threshold): int(value)
        for threshold, value in zip(sensitivity["thresholds"], sensitivity["h1_betti"], strict=True)
    }
    assert [sensitivity_lookup[float(value)] for value in primary["thresholds"]] == list(
        primary["h1_betti"]
    )
