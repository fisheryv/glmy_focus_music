"""Correct V6 direction evidence from legacy Focus-band to TAC derivatives."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .tac_target import TACTopologyTarget

V6_EXPERIMENT = "ltsn_v6_final_target_screen"
HELDOUT_SPLITS = ("seen_anchor_heldout_direction", "unseen_anchor")
BLOCK_NAMES = ("pitch", "path_acoustic_phase", "path_chroma_phase")
TIE_TOLERANCE = 1e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"empty V6 correction input: {path}")
    return rows


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    informative = [row for row in rows if bool(row["exact_tac_informative"])]
    if not informative:
        raise LTSNContractError("V6 TAC correction contains no informative direction pairs")
    return {
        "pairs": len(informative),
        "all_pairs": len(rows),
        "exact_tac_tied_pairs": len(rows) - len(informative),
        "direction_agreement": (
            sum(int(row["direction_correct"]) for row in informative) / len(informative)
        ),
        "derivative_spearman": _spearman(
            [float(row["exact_tac_derivative"]) for row in informative],
            [float(row["predicted_tac_derivative"]) for row in informative],
        ),
    }


def _anchor_id(pair_id: str) -> str:
    marker = "__v52a_"
    return pair_id.split(marker, 1)[0] if marker in pair_id else pair_id


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 20260913, resamples: int = 5000
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_anchor_id(str(row["pair_id"]))].append(row)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    agreements = []
    correlations = []
    for _ in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        draw = [row for key in sampled for row in grouped[str(key)]]
        metrics = _metrics(draw)
        agreements.append(float(metrics["direction_agreement"]))
        correlations.append(float(metrics["derivative_spearman"]))
    return {
        "cluster": "antithetic_anchor",
        "clusters": len(keys),
        "resamples": resamples,
        "seed": seed,
        "direction_agreement_ci95": np.quantile(agreements, [0.025, 0.975]).tolist(),
        "derivative_spearman_ci95": np.quantile(
            correlations, [0.025, 0.975]
        ).tolist(),
    }


def _validate_legacy_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    outcomes_path: Path,
    pair_manifest_path: Path,
    target_path: Path,
) -> None:
    required_false = (
        "scientific_evidence",
        "qualification_eligible",
        "guidance_promotion_eligible",
        "production_authorization",
    )
    if (
        report.get("schema_version") != 1
        or report.get("experiment") != V6_EXPERIMENT
        or report.get("diagnostic_only") is not True
        or report.get("new_audio_files") != 0
        or any(report.get(key) is not False for key in required_false)
    ):
        raise LTSNContractError("input is not the bounded legacy V6 report")
    expected = {
        "pair_outcomes_sha256": sha256_file(outcomes_path),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "tac_target_sha256": sha256_file(target_path),
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise LTSNContractError(f"legacy V6 report {name} mismatch")
    if report_path.resolve() == outcomes_path.resolve():
        raise LTSNContractError("legacy V6 report and outcomes paths must differ")


def correct_v6_tac_direction_evidence(
    *,
    legacy_report_path: Path,
    legacy_outcomes_path: Path,
    pair_manifest_path: Path,
    coordinate_manifest_path: Path,
    tac_target_path: Path,
    output_report_path: Path,
    output_outcomes_path: Path,
) -> dict[str, Any]:
    """Recompute exact held-out directions under the same TAC target as V6."""

    legacy_report_path = legacy_report_path.resolve()
    legacy_outcomes_path = legacy_outcomes_path.resolve()
    pair_manifest_path = pair_manifest_path.resolve()
    coordinate_manifest_path = coordinate_manifest_path.resolve()
    tac_target_path = tac_target_path.resolve()
    output_report_path = output_report_path.resolve()
    output_outcomes_path = output_outcomes_path.resolve()
    if output_report_path == legacy_report_path or output_outcomes_path == legacy_outcomes_path:
        raise LTSNContractError("V6 TAC correction must preserve the legacy artifacts")

    report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    _validate_legacy_report(
        report,
        report_path=legacy_report_path,
        outcomes_path=legacy_outcomes_path,
        pair_manifest_path=pair_manifest_path,
        target_path=tac_target_path,
    )
    target = TACTopologyTarget.from_json(tac_target_path, verify_sources=False)
    legacy_rows = _read_csv(legacy_outcomes_path)
    pair_rows = _read_csv(pair_manifest_path)
    coordinate_rows = _read_csv(coordinate_manifest_path)
    legacy_by_id = {row["pair_id"]: row for row in legacy_rows}
    pair_by_id = {
        row["pair_id"]: row
        for row in pair_rows
        if row["evaluation_split"] in HELDOUT_SPLITS
    }
    coordinates_by_id = {row["sample_id"]: row for row in coordinate_rows}
    if len(legacy_by_id) != len(legacy_rows) or len(pair_by_id) != len(legacy_rows):
        raise LTSNContractError("V6 legacy outcomes or held-out pairs contain duplicate IDs")
    if set(legacy_by_id) != set(pair_by_id):
        raise LTSNContractError("V6 legacy outcomes do not match the held-out pair manifest")

    corrected: list[dict[str, Any]] = []
    fingerprint_hashes: set[str] = set()
    for pair_id in sorted(pair_by_id):
        pair = pair_by_id[pair_id]
        legacy = legacy_by_id[pair_id]
        minus = coordinates_by_id.get(pair["minus_sample_id"])
        plus = coordinates_by_id.get(pair["plus_sample_id"])
        if minus is None or plus is None:
            raise LTSNContractError(f"V6 pair coordinates are missing: {pair_id}")
        if (
            minus.get("latent_sha256") != pair.get("minus_latent_sha256")
            or plus.get("latent_sha256") != pair.get("plus_latent_sha256")
        ):
            raise LTSNContractError(f"V6 pair-to-coordinate latent hash mismatch: {pair_id}")
        fingerprint_hashes.update(
            (minus.get("fingerprint_json_sha256", ""), plus.get("fingerprint_json_sha256", ""))
        )
        rms = float(pair["rms_ratio"])
        if not math.isclose(rms, float(legacy["rms_ratio"]), rel_tol=1e-6, abs_tol=1e-12):
            raise LTSNContractError(f"V6 pair RMS mismatch: {pair_id}")
        minus_coordinates = json.loads(minus["coordinates_json"])
        plus_coordinates = json.loads(plus["coordinates_json"])
        minus_blocks = target.block_distances(minus_coordinates)
        plus_blocks = target.block_distances(plus_coordinates)
        minus_distance = float(target.distance(minus_coordinates)[0])
        plus_distance = float(target.distance(plus_coordinates)[0])
        exact_derivative = (minus_distance - plus_distance) / (2.0 * rms)
        predicted_derivative = float(legacy["predicted_derivative"])
        informative = abs(exact_derivative) > TIE_TOLERANCE
        corrected.append(
            {
                "pair_id": pair_id,
                "evaluation_split": pair["evaluation_split"],
                "step_number": int(pair["step_number"]),
                "rms_ratio": format(rms, ".17g"),
                "exact_tac_distance_minus": format(minus_distance, ".17g"),
                "exact_tac_distance_plus": format(plus_distance, ".17g"),
                "exact_tac_derivative": format(exact_derivative, ".17g"),
                "predicted_tac_derivative": format(predicted_derivative, ".17g"),
                "exact_tac_informative": informative,
                "direction_correct": int(
                    informative and predicted_derivative * exact_derivative > 0
                ),
                "legacy_focus_band_derivative": legacy["exact_derivative"],
                "exact_pitch_derivative": format(
                    (float(minus_blocks["pitch"][0]) - float(plus_blocks["pitch"][0]))
                    / (2.0 * rms),
                    ".17g",
                ),
                "exact_acoustic_phase_derivative": format(
                    (
                        float(minus_blocks["path_acoustic_phase"][0])
                        - float(plus_blocks["path_acoustic_phase"][0])
                    )
                    / (2.0 * rms),
                    ".17g",
                ),
                "exact_chroma_phase_derivative": format(
                    (
                        float(minus_blocks["path_chroma_phase"][0])
                        - float(plus_blocks["path_chroma_phase"][0])
                    )
                    / (2.0 * rms),
                    ".17g",
                ),
            }
        )
    if fingerprint_hashes != {str(report["fingerprint_sha256"])}:
        raise LTSNContractError("V6 coordinate manifest uses a different fingerprint")

    overall = _metrics(corrected)
    by_partition = {
        split: _metrics([row for row in corrected if row["evaluation_split"] == split])
        for split in HELDOUT_SPLITS
    }
    by_step = {
        str(step): _metrics([row for row in corrected if int(row["step_number"]) == step])
        for step in (4, 5, 6)
    }
    thresholds = dict(report["thresholds"])
    criteria = dict(report["criteria"])
    criteria.update(
        {
            "minimum_heldout_direction_pairs": (
                overall["pairs"] >= int(thresholds["minimum_heldout_direction_pairs"])
            ),
            "minimum_heldout_direction_agreement": (
                overall["direction_agreement"]
                >= float(thresholds["minimum_heldout_direction_agreement"])
            ),
            "minimum_heldout_derivative_spearman": (
                overall["derivative_spearman"]
                > float(thresholds["minimum_heldout_derivative_spearman"])
            ),
        }
    )
    write_csv_atomic(output_outcomes_path, corrected)
    supported = all(bool(value) for value in criteria.values())
    payload = {
        **report,
        "schema_version": 2,
        "direction_target": "tac_topology_distance_v1",
        "direction_formula": "(D_TAC(minus)-D_TAC(plus))/(2*rms_ratio)",
        "legacy_direction_target": "focus_band_loss",
        "legacy_direction_evidence_valid": False,
        "correction_reason": (
            "schema-v1 compared predicted TAC derivatives against exact Focus-band-loss "
            "derivatives; schema-v2 recomputes exact derivatives from the frozen TAC target"
        ),
        "heldout_direction_pairs": overall["pairs"],
        "heldout_exact_tac_tied_pairs": overall["exact_tac_tied_pairs"],
        "heldout_direction_agreement": overall["direction_agreement"],
        "heldout_derivative_spearman": overall["derivative_spearman"],
        "heldout_direction_by_partition": by_partition,
        "heldout_direction_by_step": by_step,
        "heldout_anchor_cluster_bootstrap": _cluster_bootstrap(corrected),
        "criteria": criteria,
        "final_target_signal_supported": supported,
        "status": "signal_supported" if supported else "signal_not_supported",
        "coordinate_manifest_sha256": sha256_file(coordinate_manifest_path),
        "legacy_report_sha256": sha256_file(legacy_report_path),
        "legacy_pair_outcomes_sha256": sha256_file(legacy_outcomes_path),
        "pair_outcomes": str(output_outcomes_path),
        "pair_outcomes_sha256": sha256_file(output_outcomes_path),
        "interpretation": (
            "schema-v2 tests whether the V6 final-target TAC scalar field agrees with exact "
            "TAC directions on existing held-out antithetic pairs; it remains diagnostic-only "
            "and does not establish perturbation-to-final causality or audio quality"
        ),
    }
    payload.pop("report_sha256", None)
    write_json_atomic(output_report_path, payload)
    payload["report_sha256"] = sha256_file(output_report_path)
    return payload
