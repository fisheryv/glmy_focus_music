"""Torch-free provenance and descriptive diagnostics for the V3.8-B experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REVISION = "v3.8b_ordinal_calibrated_direction"
SELECTION = "fixed_epoch_online_no_selection_v38b"
GLOBAL_VARIANTS = ("g0", "g1", "g37")
LOCAL_VARIANTS = ("l0", "l1", "l2")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def local_file(folder: Path, recorded: str) -> Path:
    """Resolve a sibling artifact after a server-to-desktop synchronization."""
    return folder / recorded.replace("\\", "/").rsplit("/", 1)[-1]


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"Missing or hash-mismatched artifact: {path}")


def validate_protocol(protocol: dict, rows: list[dict]) -> None:
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Dataset has duplicate sample IDs")
    train_rows = [r for r in rows if r["split"] == "train"]
    families = sorted({r["prompt_family"] for r in train_rows})
    if len(families) != 20 or {r["split"] for r in rows} != {"train", "development"}:
        raise ValueError("V3.8-B requires the original 20 train families and development split")
    if set(families) & {r["prompt_family"] for r in rows if r["split"] == "development"}:
        raise ValueError("Train/development families overlap")
    fold = protocol["cv_fold"]
    if fold is not None and (type(fold) is not int or fold not in range(5)):
        raise ValueError("Invalid CV fold")
    held = set(families[fold::5]) if fold is not None else set()
    expected_fit = {r["sample_id"] for r in train_rows if r["prompt_family"] not in held}
    expected_eval = {r["sample_id"] for r in train_rows if r["prompt_family"] in held}
    for key, expected in [
        ("train_sample_ids", expected_fit),
        ("selection_sample_ids", expected_eval),
    ]:
        values = protocol[key]
        if len(values) != len(set(values)) or set(values) != expected:
            raise ValueError(f"Protocol {key} differs from the original grouped split")
    if set(protocol["train_families"]) != set(families) - held:
        raise ValueError("Protocol train families differ")
    if set(protocol["selection_families"]) != held:
        raise ValueError("Protocol evaluation families differ")
    if protocol["checkpoint_selection"] != SELECTION or protocol["development_used"]:
        raise ValueError("V3.8-B forbids development/outer-fold checkpoint selection")
    for ids in (expected_fit, expected_eval):
        for sid in ids:
            r = by_id[sid]
            if r["source_kind"] == "local_finite_difference" and r["trajectory_id"] not in ids:
                raise ValueError("Local example is detached from its base anchor")


def ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def correlation(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) < 2:
        return None
    a, b = ranks(y), ranks(p)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def describe_predictions(
    rows: list[dict],
    truth: dict[str, Any],
    thresholds: list[float],
    derivative_scale: float,
    widths: list[float],
) -> dict:
    """No fitting or selection. Slices use fit-only thresholds, never the dev 0.1 slice."""
    families: dict[str, list[dict]] = defaultdict(list)
    directions: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        if row["sample_id"] not in truth:
            raise ValueError("Prediction sample is absent from the dataset")
        if row["source_kind"] == "base_step4_seed":
            families[row["prompt_family"]].append(row)
        elif row["direction_id"]:
            directions[row["direction_id"]][int(row["direction_sign"])] = row
    family_stats = {}
    for family, group in families.items():
        y = np.array([float(r["exact_energy"]) for r in group])
        p = np.array([float(r["predicted_energy"]) for r in group])
        positive = y > 0
        npos, nneg = int(positive.sum()), int((~positive).sum())
        auc = (
            float((ranks(p)[positive].sum() - npos * (npos - 1) / 2) / (npos * nneg))
            if npos and nneg
            else None
        )
        buckets = np.searchsorted(thresholds, y, side="left")
        q = np.array([truth[r["sample_id"]].coordinates for r in group])
        qp = np.array([json.loads(r["predicted_coordinates_json"]) for r in group])
        family_stats[family] = {
            "rho": correlation(y, p),
            "zero_positive_auc": auc,
            "positive_only_rho": correlation(y[positive], p[positive]),
            "energy_mae": float(np.mean(abs(y - p))),
            "coordinate_rho": [correlation(q[:, i], qp[:, i]) for i in range(3)],
            "coordinate_mae_width": np.mean(abs(q - qp) / np.array(widths), axis=0).tolist(),
            "train_defined_strata": [
                {
                    "stratum": i,
                    "n": int((buckets == i).sum()),
                    "rho": correlation(y[buckets == i], p[buckets == i]),
                }
                for i in range(len(thresholds) + 1)
            ],
            "zero_and_low_positive_rho": correlation(y[buckets <= 1], p[buckets <= 1]),
        }
        if all("predicted_global_logit" in r for r in group):
            scores = np.array([float(r["predicted_global_logit"]) for r in group])
            bands = np.array([float(r["exact_band"]) for r in group])
            strata_pairs: dict[tuple[int, int], list[float]] = defaultdict(list)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    delta = bands[i] - bands[j]
                    if abs(delta) <= 1e-6:
                        continue
                    key = tuple(sorted((int(buckets[i]), int(buckets[j]))))
                    strata_pairs[key].append(
                        float(np.logaddexp(0, -np.sign(delta) * (scores[i] - scores[j])))
                    )
            family_stats[family]["diagnostic_logit_ranknet_by_stratum_pair"] = [
                {"strata": list(key), "pairs": len(values), "mean": float(np.mean(values))}
                for key, values in sorted(strata_pairs.items())
            ]
    exact, predicted, global_derivative = [], [], []
    for pair in directions.values():
        if set(pair) != {-1, 1}:
            raise ValueError("Incomplete diagnostic direction pair")
        minus, plus = pair[-1], pair[1]
        eps = float(minus["epsilon"])
        if eps <= 0 or not np.isclose(eps, float(plus["epsilon"]), rtol=0, atol=1e-9):
            raise ValueError("Diagnostic pair epsilon mismatch")
        exact.append((float(plus["exact_energy"]) - float(minus["exact_energy"])) / (2 * eps))
        predicted.append(
            (float(plus["predicted_energy"]) - float(minus["predicted_energy"])) / (2 * eps)
        )
        global_derivative.append(
            (float(plus["predicted_global_energy"]) - float(minus["predicted_global_energy"]))
            / (2 * eps)
        )
    d, dp, dg = np.array(exact), np.array(predicted), np.array(global_derivative)
    nt = abs(d) > 1e-6
    return {
        "diagnostic_only": True,
        "slice_threshold_source": "fit_base_only",
        "thresholds": thresholds,
        "families": family_stats,
        "local": {
            "nontrivial_pairs": int(nt.sum()),
            "correct": int((np.where(dp[nt] < 0, -1, 1) == np.sign(d[nt])).sum()),
            "global_correct": int((np.where(dg[nt] < 0, -1, 1) == np.sign(d[nt])).sum()),
            "derivative_rho": correlation(d[nt], dp[nt]),
            "absolute_derivative_median": float(np.median(abs(dp[nt]))) if nt.any() else None,
            "asinh_derivative_mae": float(
                np.mean(
                    abs(
                        np.arcsinh(dp[nt] / derivative_scale) - np.arcsinh(d[nt] / derivative_scale)
                    )
                )
            )
            if nt.any()
            else None,
            "flat_derivative_mae": float(np.mean(abs(dp[~nt]))) if (~nt).any() else None,
        },
    }


def verify_manifest(path: Path, dataset_rows: list[dict] | None = None) -> tuple[dict, dict]:
    m = read_json(path)
    if m.get("architecture_revision") != REVISION or m.get("checkpoint_selection") != SELECTION:
        raise ValueError("Expected a fixed-epoch V3.8-B manifest")
    expected_scope = "train_family_cv" if m["cv_fold"] is not None else "development"
    if m["validation_scope"] != expected_scope:
        raise ValueError("CV fold and validation scope differ")
    for filename, key in [
        ("pitch3_lte_run_protocol.json", "run_protocol_sha256"),
        ("pitch3_lte_training_statistics.json", "training_statistics_sha256"),
        ("pitch3_lte_effective_config.json", "training_config_sha256"),
    ]:
        verify_file(path.parent / filename, m[key])
    verify_file(local_file(path.parent, m["checkpoint"]), m["checkpoint_sha256"])
    for filename, digest in m.get("diagnostic_artifacts_sha256", {}).items():
        verify_file(local_file(path.parent, filename), digest)
    protocol = read_json(path.parent / "pitch3_lte_run_protocol.json")
    for key in ["cv_fold", "validation_scope", "experiment_contract", "checkpoint_selection"]:
        if protocol[key] != m[key]:
            raise ValueError(f"Manifest detached from protocol: {key}")
    if protocol["dataset_manifest_sha256"] != m["training_manifest_sha256"]:
        raise ValueError("Manifest detached from dataset")
    stats = read_json(path.parent / "pitch3_lte_training_statistics.json")
    if stats["run_protocol_sha256"] != m["run_protocol_sha256"]:
        raise ValueError("Training statistics detached from protocol")
    if protocol["development_used"] or protocol["outer_evaluation_count"] != (
        1 if m["cv_fold"] is not None else 0
    ):
        raise ValueError("V3.8-B must evaluate outer folds once and never select on development")
    expected_mode = (
        "zero_global_only"
        if m["experiment_contract"]["local_variant"] == "l0"
        else "trained_anchored"
    )
    if m["local_residual_mode"] != expected_mode:
        raise ValueError("Local mode differs from experiment contract")
    if dataset_rows is not None:
        validate_protocol(protocol, dataset_rows)
    for split in ["train", "selection"]:
        key = f"{split}_predictions_sha256"
        if key in m:
            prediction_path = path.parent / f"pitch3_lte_{split}_predictions.csv"
            verify_file(prediction_path, m[key])
            prediction_ids = [r["sample_id"] for r in read_csv(prediction_path)]
            if len(prediction_ids) != len(set(prediction_ids)) or set(prediction_ids) != set(
                protocol[f"{split}_sample_ids"]
            ):
                raise ValueError(f"{split} predictions differ from protocol IDs")
        elif split == "train" or m["cv_fold"] is not None:
            raise ValueError(f"Missing {split} prediction artifact")
    return m, protocol
