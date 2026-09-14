"""Calibration and independent qualification for the Pitch-3 control head."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
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
from .pitch3_ensemble import load_pitch3_ensemble
from .pitch3_training import (
    FORMAL_STEPS,
    Pitch3Dataset,
    Pitch3Snapshot,
    collate_pitch3,
    pitch3_prompt_family,
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


def _load_predictor(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    device: torch.device,
    checkpoint_path: Path | None,
    ensemble_manifest_path: Path | None,
    expected_checkpoint_sha256: str | None,
    expected_ensemble_manifest_sha256: str | None,
) -> tuple[Pitch3Contract, nn.Module, dict[str, Any], dict[str, Any]]:
    if (checkpoint_path is None) == (ensemble_manifest_path is None):
        raise ValueError("select exactly one Pitch-3 checkpoint or ensemble manifest")
    if checkpoint_path is not None:
        contract, model, metadata, checkpoint_sha256 = _load_model(
            fingerprint_path=fingerprint_path,
            training_manifest=training_manifest,
            checkpoint_path=checkpoint_path,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        binding = {
            "model_artifact_kind": "single_checkpoint",
            "model_artifact_sha256": checkpoint_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "training_config_sha256": metadata["training_config_sha256"],
        }
        return contract, model, metadata, binding
    contract = load_pitch3_contract(fingerprint_path)
    model, metadata, ensemble_sha256 = load_pitch3_ensemble(
        manifest_path=ensemble_manifest_path,
        contract=contract,
        training_manifest=training_manifest,
        device=device,
        expected_manifest_sha256=expected_ensemble_manifest_sha256,
    )
    members = metadata["members"]
    binding = {
        "model_artifact_kind": "weighted_ensemble",
        "model_artifact_sha256": ensemble_sha256,
        "ensemble_manifest_sha256": ensemble_sha256,
        "ensemble_id": metadata["ensemble_id"],
        "ensemble_member_checkpoint_sha256s": [member["checkpoint_sha256"] for member in members],
        "ensemble_member_training_config_sha256s": [
            member["training_config_sha256"] for member in members
        ],
    }
    return contract, model, metadata, binding


def _validate_upstream_model_binding(
    report: Mapping[str, Any], binding: Mapping[str, Any], *, stage: str
) -> None:
    if "model_artifact_kind" in report or "model_artifact_sha256" in report:
        expected = {
            "model_artifact_kind": binding["model_artifact_kind"],
            "model_artifact_sha256": binding["model_artifact_sha256"],
        }
    else:
        expected = {"checkpoint_sha256": binding.get("checkpoint_sha256")}
    for name, value in expected.items():
        if report.get(name) != value:
            raise LTSNContractError(f"Pitch-3 {stage} {name} binding mismatch")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.exp(-np.logaddexp(0.0, -values))


@torch.no_grad()
def _predict(
    model: nn.Module,
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
            "ood_kind": [record.ood_kind for record in records],
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
                "ood_kind": prediction["ood_kind"][index],
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
        quantile = float(np.quantile(probabilities[step_id], 0.95, method="higher"))
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


def _development_gates(
    metrics: Mapping[str, Any], metrics_by_step: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    """Predeclared development screen without using calibrated interval coverage."""

    gates = _qualification_gates(metrics, metrics_by_step)
    gates.pop("interval_90_coverage")
    return gates


def _group_regression_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "focus_logit_spearman": metrics["focus_logit_spearman"] >= MINIMUM_FOCUS_SPEARMAN,
        "each_coordinate_spearman": all(
            value >= MINIMUM_COORDINATE_SPEARMAN for value in metrics["coordinate_spearman"]
        ),
        "coordinate_median_spearman": metrics["coordinate_median_spearman"]
        >= MINIMUM_COORDINATE_SPEARMAN,
        "target_band_distance_spearman": metrics["target_band_distance_spearman"]
        >= MINIMUM_COORDINATE_SPEARMAN,
        "quartile_ranking_accuracy": metrics["quartile_ranking_accuracy"]
        >= MINIMUM_RANKING_ACCURACY,
    }


def _subset_prediction(prediction: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        name: value[mask]
        if isinstance(value, np.ndarray)
        else [item for item, keep in zip(value, mask, strict=True) if keep]
        for name, value in prediction.items()
    }


def screen_pitch3_development(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    ensemble_manifest_path: Path | None = None,
    batch_size: int = 8,
    device_name: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_ensemble_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Screen one frozen predictor on development before calibration is consumed."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    contract, model, metadata, artifact_binding = _load_predictor(
        fingerprint_path=fingerprint_path,
        training_manifest=training_manifest,
        device=device,
        checkpoint_path=checkpoint_path,
        ensemble_manifest_path=ensemble_manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_ensemble_manifest_sha256=expected_ensemble_manifest_sha256,
    )
    records = read_pitch3_manifest(training_manifest, contract)
    development = [record for record in records if record.split == "development"]
    prediction = _predict(model, development, batch_size=batch_size, device=device)
    labels = prediction["ood_label"]
    if not np.any(labels < 0.5) or not np.any(labels >= 0.5):
        raise LTSNContractError("Pitch-3 development screen requires ID and OOD samples")
    threshold, threshold_by_step = _calibration_threshold(
        labels, prediction["ood_probability"], prediction["step_number"]
    )
    variance_scale = np.ones(PITCH3_DIMENSIONS, dtype=float)
    overall = _metrics(
        prediction,
        contract=contract,
        variance_scale=variance_scale,
        ood_threshold=threshold,
    )
    correction_mask = np.isin(prediction["step_number"].astype(int), CORRECTION_STEPS)
    ood_screen = _ood_metrics(
        labels[correction_mask],
        prediction["ood_probability"][correction_mask],
        threshold,
    )
    overall.update(
        {
            "regression_samples": int(len(labels)),
            "regression_id_samples": int(np.count_nonzero(labels < 0.5)),
            "ood_screen_samples": ood_screen["samples"],
            "ood_screen_id_samples": ood_screen["id_samples"],
            "ood_screen_ood_samples": ood_screen["ood_samples"],
            "id_acceptance_rate": ood_screen["id_acceptance_rate"],
            "ood_sensitivity": ood_screen["ood_sensitivity"],
            "ood_auroc": ood_screen["ood_auroc"],
        }
    )
    ood_kind = np.asarray(prediction["ood_kind"], dtype=object)
    ood_by_kind: dict[str, Any] = {}
    for kind in sorted({str(value) for value in ood_kind if value}):
        mask = correction_mask & (ood_kind == kind) & (labels >= 0.5)
        probabilities = prediction["ood_probability"][mask]
        if len(probabilities):
            ood_by_kind[kind] = {
                "samples": int(len(probabilities)),
                "sensitivity": float(np.mean(probabilities > threshold)),
                "probability_min": float(np.min(probabilities)),
                "probability_median": float(np.median(probabilities)),
                "probability_max": float(np.max(probabilities)),
            }
    by_step: dict[str, Any] = {}
    for step in CORRECTION_STEPS:
        mask = prediction["step_number"] == step
        subset = _subset_prediction(prediction, mask)
        by_step[str(step)] = _metrics(
            subset,
            contract=contract,
            variance_scale=variance_scale,
            ood_threshold=threshold,
        )
        by_step[str(step)]["id_probability_q95"] = threshold_by_step[str(step)][
            "id_probability_q95"
        ]
    gates = _development_gates(overall, by_step)
    raw_group_contract = metadata.get("group_robust_training", {})
    group_robust_required = isinstance(raw_group_contract, Mapping) and bool(
        raw_group_contract.get("development_screen_required", False)
    )
    family_values = np.asarray(
        [pitch3_prompt_family(value) for value in prediction["prompt_id"]], dtype=object
    )
    group_metrics_by_family: dict[str, Any] = {}
    group_gates_by_family: dict[str, Any] = {}
    for family in sorted(set(family_values)):
        family_mask = family_values == family
        if not np.any((labels < 0.5) & family_mask):
            continue
        family_metrics = _metrics(
            _subset_prediction(prediction, family_mask),
            contract=contract,
            variance_scale=variance_scale,
            ood_threshold=threshold,
        )
        group_metrics_by_family[str(family)] = family_metrics
        group_gates_by_family[str(family)] = _group_regression_gates(family_metrics)
    group_metrics_by_step: dict[str, Any] = {}
    group_gates_by_step: dict[str, Any] = {}
    for step in FORMAL_STEPS:
        step_mask = prediction["step_number"] == step
        if not np.any((labels < 0.5) & step_mask):
            continue
        step_metrics = _metrics(
            _subset_prediction(prediction, step_mask),
            contract=contract,
            variance_scale=variance_scale,
            ood_threshold=threshold,
        )
        group_metrics_by_step[str(step)] = step_metrics
        group_gates_by_step[str(step)] = _group_regression_gates(step_metrics)
    group_robust_passed = all(
        all(values.values())
        for collection in (group_gates_by_family, group_gates_by_step)
        for values in collection.values()
    )
    passed = all(gates.values()) and (not group_robust_required or group_robust_passed)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "pitch3_development_predictions.csv"
    write_csv_atomic(prediction_path, _prediction_rows(prediction))
    payload = {
        "schema_version": 2,
        "stage": "pitch3_development_screen",
        "status": "passed" if passed else "failed",
        "calibration_eligible": passed,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
        **artifact_binding,
        "ood_transform_version": metadata.get("ood_transform_version", ""),
        "ood_label_source": metadata.get("ood_label_source", ""),
        "prediction_csv": prediction_path.resolve().as_posix(),
        "prediction_csv_sha256": sha256_file(prediction_path),
        "ood_probability_threshold_diagnostic": threshold,
        "metrics": overall,
        "metrics_by_step": by_step,
        "ood_metrics_by_kind": ood_by_kind,
        "gates": gates,
        "group_robust_screen": {
            "required": group_robust_required,
            "passed": group_robust_passed,
            "metrics_by_prompt_family": group_metrics_by_family,
            "gates_by_prompt_family": group_gates_by_family,
            "metrics_by_step": group_metrics_by_step,
            "gates_by_step": group_gates_by_step,
            "thresholds_changed": False,
        },
        "selection_scope": "development_only",
        "qualification_split_consumed": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    report_path = output_dir / "pitch3_development_screen.json"
    write_json_atomic(report_path, payload)
    return {
        **payload,
        "report": report_path.resolve().as_posix(),
        "report_sha256": sha256_file(report_path),
    }


def calibrate_pitch3_control_head(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    development_screen_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    ensemble_manifest_path: Path | None = None,
    batch_size: int = 8,
    device_name: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_ensemble_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze interval scaling and an ID-preserving OOD threshold on calibration only."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    contract, model, metadata, artifact_binding = _load_predictor(
        fingerprint_path=fingerprint_path,
        training_manifest=training_manifest,
        device=device,
        checkpoint_path=checkpoint_path,
        ensemble_manifest_path=ensemble_manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_ensemble_manifest_sha256=expected_ensemble_manifest_sha256,
    )
    development_screen = json.loads(development_screen_path.read_text(encoding="utf-8"))
    if (
        development_screen.get("schema_version") != 2
        or development_screen.get("stage") != "pitch3_development_screen"
        or development_screen.get("status") != "passed"
        or development_screen.get("calibration_eligible") is not True
    ):
        raise LTSNContractError("calibration requires a passed Pitch-3 development screen")
    expected_screen_bindings = {
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
    }
    for name, value in expected_screen_bindings.items():
        if development_screen.get(name) != value:
            raise LTSNContractError(f"Pitch-3 development screen {name} binding mismatch")
    _validate_upstream_model_binding(
        development_screen, artifact_binding, stage="development screen"
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
    ood_metrics = _ood_metrics(prediction["ood_label"], prediction["ood_probability"], threshold)
    ood_auroc_passed = (
        isinstance(ood_metrics["ood_auroc"], (int, float))
        and math.isfinite(ood_metrics["ood_auroc"])
        and ood_metrics["ood_auroc"] >= MINIMUM_OOD_AUROC
    )
    ood_sensitivity_passed = (
        isinstance(ood_metrics["ood_sensitivity"], (int, float))
        and ood_metrics["ood_sensitivity"] >= MINIMUM_OOD_SENSITIVITY
    )
    calibration_passed = acceptance_passed and ood_auroc_passed and ood_sensitivity_passed
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "pitch3_calibration_predictions.csv"
    write_csv_atomic(prediction_path, _prediction_rows(prediction))
    payload = {
        "schema_version": 2,
        "stage": "pitch3_calibration",
        "status": "frozen" if calibration_passed else "failed",
        "qualification_eligible": calibration_passed,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
        **artifact_binding,
        "ood_transform_version": metadata.get("ood_transform_version", ""),
        "ood_label_source": metadata.get("ood_label_source", ""),
        "development_screen_sha256": sha256_file(development_screen_path),
        "prediction_csv": prediction_path.resolve().as_posix(),
        "prediction_csv_sha256": sha256_file(prediction_path),
        "sample_count": len(calibration_records),
        "id_sample_count": int(np.count_nonzero(id_mask)),
        "ood_sample_count": int(np.count_nonzero(~id_mask)),
        "variance_scale": variance_scale.tolist(),
        "ood_probability_threshold": threshold,
        "ood_metrics": ood_metrics,
        "ood_by_step": by_step,
        "calibration_gate": {
            "correction_step_id_acceptance": acceptance_passed,
            "ood_auroc": ood_auroc_passed,
            "ood_sensitivity": ood_sensitivity_passed,
            "minimum_id_acceptance": MINIMUM_ID_ACCEPTANCE,
            "minimum_ood_auroc": MINIMUM_OOD_AUROC,
            "minimum_ood_sensitivity": MINIMUM_OOD_SENSITIVITY,
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
    calibration_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    ensemble_manifest_path: Path | None = None,
    batch_size: int = 8,
    device_name: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_ensemble_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one-shot independent surrogate qualification without authorizing guidance."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    contract, model, metadata, artifact_binding = _load_predictor(
        fingerprint_path=fingerprint_path,
        training_manifest=training_manifest,
        device=device,
        checkpoint_path=checkpoint_path,
        ensemble_manifest_path=ensemble_manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_ensemble_manifest_sha256=expected_ensemble_manifest_sha256,
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("schema_version") != 2
        or calibration.get("stage") != "pitch3_calibration"
        or calibration.get("status") != "frozen"
        or calibration.get("qualification_eligible") is not True
    ):
        raise LTSNContractError("qualification requires a frozen Pitch-3 calibration")
    expected_bindings = {
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
    }
    for name, value in expected_bindings.items():
        if calibration.get(name) != value:
            raise LTSNContractError(f"Pitch-3 calibration {name} binding mismatch")
    _validate_upstream_model_binding(calibration, artifact_binding, stage="calibration")
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
        "training_manifest_sha256": sha256_file(training_manifest),
        **artifact_binding,
        "ood_transform_version": metadata.get("ood_transform_version", ""),
        "ood_label_source": metadata.get("ood_label_source", ""),
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
