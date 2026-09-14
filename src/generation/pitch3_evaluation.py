"""Calibration and independent qualification for the Pitch-3 control head."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .latent_topology_control_head import (
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
)
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .ltsn_training import spearman_correlation
from .pitch3_contract import (
    PITCH3_DIMENSIONS,
    Pitch3Contract,
    load_pitch3_contract,
    validate_pitch3_checkpoint_metadata,
)
from .pitch3_training import (
    Pitch3Dataset,
    Pitch3Snapshot,
    collate_pitch3,
    read_pitch3_manifest,
)

CORRECTION_STEPS = (4, 5, 6)
NORMAL_90_QUANTILE = 1.6448536269514722
MINIMUM_ID_ACCEPTANCE = 0.95
MINIMUM_FOCUS_SPEARMAN = 0.70
MINIMUM_COORDINATE_SPEARMAN = 0.50
MINIMUM_RANKING_ACCURACY = 0.65
MINIMUM_OOD_AUROC = 0.80
MINIMUM_OOD_SENSITIVITY = 0.80
MINIMUM_INTERVAL_COVERAGE = 0.85
MAXIMUM_INTERVAL_COVERAGE = 0.95


def _load_model(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    checkpoint_path: Path,
    device: torch.device,
    expected_checkpoint_sha256: str | None,
) -> tuple[Pitch3Contract, LatentTopologyControlHead, dict[str, Any], str]:
    contract = load_pitch3_contract(fingerprint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256.lower()
    ):
        raise LTSNContractError("Pitch-3 checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise LTSNContractError("Pitch-3 checkpoint payload is malformed")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise LTSNContractError("Pitch-3 checkpoint metadata is missing")
    validate_pitch3_checkpoint_metadata(metadata, contract)
    if metadata.get("model_family") != "latent_topology_control_head_pitch3":
        raise LTSNContractError("unexpected Pitch-3 checkpoint model family")
    if metadata.get("training_manifest_sha256") != sha256_file(training_manifest):
        raise LTSNContractError("evaluation manifest differs from the training manifest")
    model_config = payload.get("model_config")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise LTSNContractError("Pitch-3 checkpoint model config/state is missing")
    model = LatentTopologyControlHead(contract, Pitch3ControlHeadConfig(**dict(model_config)))
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return contract, model, dict(metadata), checkpoint_sha256


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.exp(-np.logaddexp(0.0, -values))


@torch.no_grad()
def _predict(
    model: LatentTopologyControlHead,
    records: Sequence[Pitch3Snapshot],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise LTSNContractError("selected Pitch-3 evaluation split is empty")
    loader = DataLoader(
        Pitch3Dataset(list(records)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pitch3,
    )
    collected: dict[str, list[np.ndarray]] = {
        "coordinate_mean": [],
        "coordinate_logvar": [],
        "predicted_focus_logit": [],
        "ood_logit": [],
        "coordinates": [],
        "focus_logit": [],
        "ood_label": [],
        "step_number": [],
    }
    sample_ids: list[str] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for raw in loader:
        latent = raw["latent"].to(device)
        timestep = raw["timestep"].to(device)
        step_number = raw["step_number"].to(device)
        attention_mask = raw["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = model(latent, timestep, step_number, attention_mask)
        sample_ids.extend(raw["sample_id"])
        tensors = {
            "coordinate_mean": output.coordinate_mean,
            "coordinate_logvar": output.coordinate_logvar,
            "predicted_focus_logit": output.focus_logit,
            "ood_logit": output.ood_logit,
            "coordinates": raw["coordinates"],
            "focus_logit": raw["focus_logit"],
            "ood_label": raw["ood_label"],
            "step_number": raw["step_number"],
        }
        for name, tensor in tensors.items():
            collected[name].append(tensor.detach().float().cpu().numpy())
    expected_ids = [record.sample_id for record in records]
    if sample_ids != expected_ids:
        raise RuntimeError("Pitch-3 prediction order differs from manifest order")
    prediction = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    prediction.update(
        {
            "sample_id": sample_ids,
            "prompt_id": [record.prompt_id for record in records],
            "trajectory_id": [record.trajectory_id for record in records],
            "split": [record.split for record in records],
            "timestep": np.asarray([record.timestep for record in records], dtype=float),
            "is_final": np.asarray([record.is_final for record in records], dtype=bool),
        }
    )
    prediction["ood_probability"] = _sigmoid(prediction["ood_logit"])
    prediction["coordinate_variance"] = np.exp(prediction["coordinate_logvar"])
    return prediction


def _prediction_rows(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(prediction["sample_id"]):
        exact = prediction["coordinates"][index]
        mean = prediction["coordinate_mean"][index]
        logvar = prediction["coordinate_logvar"][index]
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": prediction["prompt_id"][index],
                "trajectory_id": prediction["trajectory_id"][index],
                "split": prediction["split"][index],
                "step_number": int(prediction["step_number"][index]),
                "timestep": float(prediction["timestep"][index]),
                "is_final": str(bool(prediction["is_final"][index])).lower(),
                "exact_coordinates_json": json.dumps(exact.tolist(), separators=(",", ":")),
                "predicted_mean_json": json.dumps(mean.tolist(), separators=(",", ":")),
                "predicted_logvar_json": json.dumps(logvar.tolist(), separators=(",", ":")),
                "exact_focus_logit": float(prediction["focus_logit"][index]),
                "predicted_focus_logit": float(prediction["predicted_focus_logit"][index]),
                "ood_label": float(prediction["ood_label"][index]),
                "ood_probability": float(prediction["ood_probability"][index]),
            }
        )
    return rows


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels >= 0.5]
    negative = scores[labels < 0.5]
    if not len(positive) or not len(negative):
        return None
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _quartile_ranking_accuracy(exact: np.ndarray, predicted: np.ndarray) -> float:
    low = np.flatnonzero(exact <= np.quantile(exact, 0.25))
    high = np.flatnonzero(exact >= np.quantile(exact, 0.75))
    if not len(low) or not len(high):
        return 0.0
    return float(np.mean(predicted[high, None] > predicted[None, low]))


def _target_band_loss(coordinates: np.ndarray, contract: Pitch3Contract) -> np.ndarray:
    lower = np.asarray(contract.target_lower, dtype=float)
    upper = np.asarray(contract.target_upper, dtype=float)
    weights = np.asarray(contract.distance_weights, dtype=float)
    below = np.maximum(lower - coordinates, 0.0)
    above = np.maximum(coordinates - upper, 0.0)
    return (np.square(below) + np.square(above)) @ weights


def _ood_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    id_mask = labels < 0.5
    ood_mask = ~id_mask
    predicted_ood = probabilities > threshold
    id_acceptance = None if not np.any(id_mask) else float(np.mean(~predicted_ood[id_mask]))
    sensitivity = None if not np.any(ood_mask) else float(np.mean(predicted_ood[ood_mask]))
    return {
        "samples": int(len(labels)),
        "id_samples": int(np.count_nonzero(id_mask)),
        "ood_samples": int(np.count_nonzero(ood_mask)),
        "id_acceptance_rate": id_acceptance,
        "ood_sensitivity": sensitivity,
        "ood_auroc": _auc(labels, probabilities),
    }


def _calibration_threshold(
    labels: np.ndarray, probabilities: np.ndarray, steps: np.ndarray
) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    steps = np.asarray(steps, dtype=int)
    if not (labels.shape == probabilities.shape == steps.shape) or labels.ndim != 1:
        raise LTSNContractError("Pitch-3 OOD calibration arrays are malformed")
    by_step: dict[str, Any] = {}
    quantiles: list[float] = []
    for step in CORRECTION_STEPS:
        step_id = (steps == step) & (labels < 0.5)
        if not np.any(step_id):
            raise LTSNContractError(f"calibration has no ID samples at correction step {step}")
        quantile = float(np.quantile(probabilities[step_id], 0.95))
        quantiles.append(quantile)
        by_step[str(step)] = {
            "id_samples": int(np.count_nonzero(step_id)),
            "id_probability_q95": quantile,
        }
    threshold = max(quantiles)
    for step in CORRECTION_STEPS:
        step_mask = steps == step
        by_step[str(step)].update(
            _ood_metrics(labels[step_mask], probabilities[step_mask], threshold)
        )
    return threshold, by_step


def _metrics(
    prediction: Mapping[str, Any],
    *,
    contract: Pitch3Contract,
    variance_scale: np.ndarray,
    ood_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(prediction["ood_label"], dtype=float)
    id_mask = labels < 0.5
    if not np.any(id_mask):
        raise LTSNContractError("Pitch-3 qualification requires ID samples")
    exact = np.asarray(prediction["coordinates"], dtype=float)[id_mask]
    mean = np.asarray(prediction["coordinate_mean"], dtype=float)[id_mask]
    exact_focus = np.asarray(prediction["focus_logit"], dtype=float)[id_mask]
    predicted_focus = np.asarray(prediction["predicted_focus_logit"], dtype=float)[id_mask]
    coordinate_rhos = [
        spearman_correlation(mean[:, index], exact[:, index]) for index in range(PITCH3_DIMENSIONS)
    ]
    exact_band_loss = _target_band_loss(exact, contract)
    predicted_band_loss = _target_band_loss(mean, contract)
    variance = (
        np.asarray(prediction["coordinate_variance"], dtype=float)[id_mask]
        * variance_scale[None, :]
    )
    half_width = NORMAL_90_QUANTILE * np.sqrt(np.maximum(variance, 0.0))
    covered = (exact >= mean - half_width) & (exact <= mean + half_width)
    result = {
        "samples": int(len(labels)),
        "id_samples": int(np.count_nonzero(id_mask)),
        "focus_logit_spearman": spearman_correlation(predicted_focus, exact_focus),
        "coordinate_spearman": coordinate_rhos,
        "coordinate_median_spearman": float(np.median(coordinate_rhos)),
        "target_band_distance_spearman": spearman_correlation(predicted_band_loss, exact_band_loss),
        "quartile_ranking_accuracy": _quartile_ranking_accuracy(exact_focus, predicted_focus),
        "interval_90_coverage": float(np.mean(covered)),
        "coordinate_mae": np.mean(np.abs(mean - exact), axis=0).tolist(),
        "coordinate_rmse": np.sqrt(np.mean(np.square(mean - exact), axis=0)).tolist(),
        "focus_logit_mae": float(np.mean(np.abs(predicted_focus - exact_focus))),
    }
    result.update(_ood_metrics(labels, prediction["ood_probability"], ood_threshold))
    return result


def _qualification_gates(
    metrics: Mapping[str, Any], metrics_by_step: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    coordinate_rhos = metrics["coordinate_spearman"]
    return {
        "focus_logit_spearman": metrics["focus_logit_spearman"] >= MINIMUM_FOCUS_SPEARMAN,
        "each_coordinate_spearman": all(
            value >= MINIMUM_COORDINATE_SPEARMAN for value in coordinate_rhos
        ),
        "coordinate_median_spearman": (
            metrics["coordinate_median_spearman"] >= MINIMUM_COORDINATE_SPEARMAN
        ),
        "target_band_distance_spearman": (
            metrics["target_band_distance_spearman"] >= MINIMUM_COORDINATE_SPEARMAN
        ),
        "quartile_ranking_accuracy": (
            metrics["quartile_ranking_accuracy"] >= MINIMUM_RANKING_ACCURACY
        ),
        "interval_90_coverage": (
            MINIMUM_INTERVAL_COVERAGE
            <= metrics["interval_90_coverage"]
            <= MAXIMUM_INTERVAL_COVERAGE
        ),
        "correction_step_id_acceptance": all(
            str(step) in metrics_by_step
            and isinstance(metrics_by_step[str(step)].get("id_acceptance_rate"), (int, float))
            and metrics_by_step[str(step)]["id_acceptance_rate"] >= MINIMUM_ID_ACCEPTANCE
            for step in CORRECTION_STEPS
        ),
        "ood_auroc": isinstance(metrics.get("ood_auroc"), (int, float))
        and math.isfinite(metrics["ood_auroc"])
        and metrics["ood_auroc"] >= MINIMUM_OOD_AUROC,
        "ood_sensitivity": isinstance(metrics.get("ood_sensitivity"), (int, float))
        and metrics["ood_sensitivity"] >= MINIMUM_OOD_SENSITIVITY,
    }


def calibrate_pitch3_control_head(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 8,
    device_name: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze interval scaling and an ID-preserving OOD threshold on calibration only."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    contract, model, metadata, checkpoint_sha256 = _load_model(
        fingerprint_path=fingerprint_path,
        training_manifest=training_manifest,
        checkpoint_path=checkpoint_path,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    records = read_pitch3_manifest(training_manifest, contract)
    calibration_records = [record for record in records if record.split == "calibration"]
    prediction = _predict(model, calibration_records, batch_size=batch_size, device=device)
    id_mask = prediction["ood_label"] < 0.5
    if not np.any(id_mask) or not np.any(~id_mask):
        raise LTSNContractError("Pitch-3 calibration requires both ID and OOD samples")
    residual = np.abs(prediction["coordinates"][id_mask] - prediction["coordinate_mean"][id_mask])
    standard = np.sqrt(np.maximum(prediction["coordinate_variance"][id_mask], 1e-12))
    ratio = residual / standard
    quantile = np.quantile(ratio, 0.90, axis=0)
    variance_scale = np.maximum(np.square(quantile / NORMAL_90_QUANTILE), 1e-8)
    threshold, by_step = _calibration_threshold(
        prediction["ood_label"], prediction["ood_probability"], prediction["step_number"]
    )
    acceptance_passed = all(
        by_step[str(step)]["id_acceptance_rate"] >= MINIMUM_ID_ACCEPTANCE
        for step in CORRECTION_STEPS
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "pitch3_calibration_predictions.csv"
    write_csv_atomic(prediction_path, _prediction_rows(prediction))
    payload = {
        "schema_version": 1,
        "stage": "pitch3_calibration",
        "status": "frozen" if acceptance_passed else "failed",
        "qualification_eligible": acceptance_passed,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
        "training_config_sha256": metadata["training_config_sha256"],
        "prediction_csv": prediction_path.resolve().as_posix(),
        "prediction_csv_sha256": sha256_file(prediction_path),
        "sample_count": len(calibration_records),
        "id_sample_count": int(np.count_nonzero(id_mask)),
        "ood_sample_count": int(np.count_nonzero(~id_mask)),
        "variance_scale": variance_scale.tolist(),
        "ood_probability_threshold": threshold,
        "ood_by_step": by_step,
        "calibration_gate": {
            "correction_step_id_acceptance": acceptance_passed,
            "minimum_id_acceptance": MINIMUM_ID_ACCEPTANCE,
        },
        "ood_evidence_scope": "held_out_synthetic_transform_benchmark",
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    report_path = output_dir / "pitch3_calibration.json"
    write_json_atomic(report_path, payload)
    return {
        **payload,
        "report": report_path.resolve().as_posix(),
        "report_sha256": sha256_file(report_path),
    }


def qualify_pitch3_control_head(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    output_dir: Path,
    batch_size: int = 8,
    device_name: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one-shot independent surrogate qualification without authorizing guidance."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    contract, model, metadata, checkpoint_sha256 = _load_model(
        fingerprint_path=fingerprint_path,
        training_manifest=training_manifest,
        checkpoint_path=checkpoint_path,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("schema_version") != 1
        or calibration.get("stage") != "pitch3_calibration"
        or calibration.get("status") != "frozen"
        or calibration.get("qualification_eligible") is not True
    ):
        raise LTSNContractError("qualification requires a frozen Pitch-3 calibration")
    expected_bindings = {
        "fingerprint_json_sha256": contract.artifact_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
    }
    for name, value in expected_bindings.items():
        if calibration.get(name) != value:
            raise LTSNContractError(f"Pitch-3 calibration {name} binding mismatch")
    variance_scale = np.asarray(calibration.get("variance_scale"), dtype=float)
    if variance_scale.shape != (PITCH3_DIMENSIONS,) or not np.isfinite(variance_scale).all():
        raise LTSNContractError("Pitch-3 calibration variance scale is malformed")
    threshold = float(calibration.get("ood_probability_threshold"))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise LTSNContractError("Pitch-3 calibration OOD threshold is malformed")
    records = read_pitch3_manifest(training_manifest, contract)
    qualification_records = [record for record in records if record.split == "qualification"]
    prediction = _predict(model, qualification_records, batch_size=batch_size, device=device)
    overall = _metrics(
        prediction,
        contract=contract,
        variance_scale=variance_scale,
        ood_threshold=threshold,
    )
    by_step: dict[str, Any] = {}
    for step in sorted(np.unique(prediction["step_number"]).astype(int)):
        mask = prediction["step_number"] == step
        subset = {
            name: value[mask]
            if isinstance(value, np.ndarray)
            else [item for item, keep in zip(value, mask, strict=True) if keep]
            for name, value in prediction.items()
        }
        by_step[str(step)] = _metrics(
            subset,
            contract=contract,
            variance_scale=variance_scale,
            ood_threshold=threshold,
        )
    gates = _qualification_gates(overall, by_step)
    surrogate_passed = all(gates.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "pitch3_qualification_predictions.csv"
    write_csv_atomic(prediction_path, _prediction_rows(prediction))
    by_step_path = output_dir / "pitch3_qualification_by_step.csv"
    write_csv_atomic(
        by_step_path,
        [
            {
                "step_number": step,
                "metrics_json": json.dumps(metrics, sort_keys=True, separators=(",", ":")),
            }
            for step, metrics in by_step.items()
        ],
    )
    payload = {
        "schema_version": 1,
        "stage": "pitch3_surrogate_qualification",
        "status": "passed" if surrogate_passed else "failed",
        "surrogate_qualification_passed": surrogate_passed,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
        "training_config_sha256": metadata["training_config_sha256"],
        "calibration_sha256": sha256_file(calibration_path),
        "prediction_csv": prediction_path.resolve().as_posix(),
        "prediction_csv_sha256": sha256_file(prediction_path),
        "metrics_by_step_csv": by_step_path.resolve().as_posix(),
        "metrics_by_step_csv_sha256": sha256_file(by_step_path),
        "metrics": overall,
        "metrics_by_step": by_step,
        "gates": gates,
        "gate_thresholds": {
            "minimum_focus_spearman": MINIMUM_FOCUS_SPEARMAN,
            "minimum_coordinate_spearman": MINIMUM_COORDINATE_SPEARMAN,
            "minimum_ranking_accuracy": MINIMUM_RANKING_ACCURACY,
            "interval_coverage_range": [
                MINIMUM_INTERVAL_COVERAGE,
                MAXIMUM_INTERVAL_COVERAGE,
            ],
            "minimum_id_acceptance": MINIMUM_ID_ACCEPTANCE,
            "minimum_ood_auroc": MINIMUM_OOD_AUROC,
            "minimum_ood_sensitivity": MINIMUM_OOD_SENSITIVITY,
        },
        "variance_scale": variance_scale.tolist(),
        "ood_probability_threshold": threshold,
        "ood_evidence_scope": "held_out_synthetic_transform_benchmark",
        "decoded_exact_direction_agreement_evaluated": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    report_path = output_dir / "pitch3_qualification.json"
    write_json_atomic(report_path, payload)
    return {
        **payload,
        "report": report_path.resolve().as_posix(),
        "report_sha256": sha256_file(report_path),
    }
