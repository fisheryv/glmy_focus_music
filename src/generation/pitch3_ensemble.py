"""Hash-bound weighted ensembles for Pitch-3 latent control heads."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .latent_topology_control_head import (
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
    Pitch3ControlOutput,
)
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_json_atomic
from .pitch3_contract import Pitch3Contract, validate_pitch3_checkpoint_metadata

ENSEMBLE_STAGE = "pitch3_control_head_ensemble"
ENSEMBLE_AGGREGATION = {
    "coordinate_mean": "weighted_arithmetic_mean",
    "coordinate_variance": "weighted_law_of_total_variance",
    "focus_logit": "recomputed_from_ensemble_coordinate_mean",
    "ood_probability": "weighted_arithmetic_mean",
}


class Pitch3ControlHeadEnsemble(nn.Module):
    """Differentiable mixture of frozen Pitch-3 control heads."""

    def __init__(
        self,
        members: Sequence[LatentTopologyControlHead],
        weights: Sequence[float],
        contract: Pitch3Contract,
    ) -> None:
        super().__init__()
        if len(members) < 2 or len(members) != len(weights):
            raise ValueError("Pitch-3 ensemble requires two or more weighted members")
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("Pitch-3 ensemble weights must be finite and positive")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Pitch-3 ensemble weights must sum to one")
        self.members = nn.ModuleList(members)
        self.register_buffer("ensemble_weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer(
            "focus_coef", torch.tensor(contract.classifier_coef, dtype=torch.float32)
        )
        self.register_buffer(
            "focus_intercept", torch.tensor(contract.classifier_intercept, dtype=torch.float32)
        )

    def forward(
        self,
        latent: Tensor,
        timestep: Tensor | float,
        step_number: Tensor | int,
        attention_mask: Tensor | None = None,
    ) -> Pitch3ControlOutput:
        outputs = [member(latent, timestep, step_number, attention_mask) for member in self.members]
        weights = self.ensemble_weights[:, None, None]
        means = torch.stack([output.coordinate_mean.float() for output in outputs], dim=0)
        variances = torch.stack(
            [output.coordinate_logvar.float().exp() for output in outputs], dim=0
        )
        mean = (weights * means).sum(dim=0)
        second_moment = (weights * (variances + means.square())).sum(dim=0)
        variance = (second_moment - mean.square()).clamp_min(1e-12)
        probability_weights = self.ensemble_weights[:, None]
        probabilities = torch.stack(
            [output.ood_logit.float().sigmoid() for output in outputs], dim=0
        )
        ood_probability = (probability_weights * probabilities).sum(dim=0).clamp(1e-7, 1.0 - 1e-7)
        ood_logit = torch.logit(ood_probability)
        focus_logit = mean @ self.focus_coef + self.focus_intercept
        return Pitch3ControlOutput(mean, variance.log(), ood_logit, focus_logit)


def _checkpoint_payload(
    checkpoint_path: Path,
    *,
    contract: Pitch3Contract,
    training_manifest_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise LTSNContractError("Pitch-3 ensemble member payload is malformed")
    metadata = payload.get("metadata")
    model_config = payload.get("model_config")
    model_state = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping):
        raise LTSNContractError("Pitch-3 ensemble member metadata is missing")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise LTSNContractError("Pitch-3 ensemble member model config/state is missing")
    validate_pitch3_checkpoint_metadata(metadata, contract)
    if metadata.get("model_family") != "latent_topology_control_head_pitch3":
        raise LTSNContractError("unexpected Pitch-3 ensemble member model family")
    if metadata.get("training_manifest_sha256") != training_manifest_sha256:
        raise LTSNContractError("ensemble member training manifest binding mismatch")
    return payload, dict(metadata), dict(model_config)


def build_pitch3_ensemble_manifest(
    *,
    contract: Pitch3Contract,
    training_manifest: Path,
    members: Sequence[tuple[str, Path, float]],
    output_path: Path,
    ensemble_id: str = "ltch_pitch3_e2_v1",
) -> dict[str, Any]:
    """Freeze member hashes and aggregation rules without consuming holdout data."""

    if len(members) < 2:
        raise ValueError("Pitch-3 ensemble manifest requires at least two members")
    if not ensemble_id.strip():
        raise ValueError("Pitch-3 ensemble id must be non-empty")
    names = [name for name, _, _ in members]
    weights = [float(weight) for _, _, weight in members]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("Pitch-3 ensemble member names must be unique and non-empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("Pitch-3 ensemble weights must be finite and positive")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Pitch-3 ensemble weights must sum to one")
    manifest_sha256 = sha256_file(training_manifest)
    output_parent = output_path.resolve().parent
    frozen_members: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    ood_versions: set[str] = set()
    ood_sources: set[str] = set()
    for name, raw_path, weight in members:
        checkpoint_path = raw_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        _, metadata, model_config = _checkpoint_payload(
            checkpoint_path,
            contract=contract,
            training_manifest_sha256=manifest_sha256,
        )
        if reference_config is None:
            reference_config = model_config
        elif model_config != reference_config:
            raise LTSNContractError("Pitch-3 ensemble members must share one model config")
        ood_versions.add(str(metadata.get("ood_transform_version", "")))
        ood_sources.add(str(metadata.get("ood_label_source", "")))
        relative_path = Path(os.path.relpath(checkpoint_path, output_parent)).as_posix()
        frozen_members.append(
            {
                "name": name,
                "weight": float(weight),
                "checkpoint_path": relative_path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_config_sha256": metadata["training_config_sha256"],
                "architecture_revision": metadata.get("architecture_revision", ""),
                "band_training_objective": metadata.get("band_training_objective", {}),
            }
        )
    if len(ood_versions) != 1 or len(ood_sources) != 1:
        raise LTSNContractError("Pitch-3 ensemble members have incompatible OOD provenance")
    payload = {
        "schema_version": 1,
        "stage": ENSEMBLE_STAGE,
        "ensemble_id": ensemble_id,
        "model_family": "latent_topology_control_head_pitch3_ensemble",
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "feature_order": list(contract.feature_order),
        "classifier_sha256": contract.classifier_sha256,
        "training_manifest_sha256": manifest_sha256,
        "ood_transform_version": next(iter(ood_versions)),
        "ood_label_source": next(iter(ood_sources)),
        "aggregation": ENSEMBLE_AGGREGATION,
        "members": frozen_members,
        "selection_scope": "development_only",
        "qualification_split_consumed": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    write_json_atomic(output_path, payload)
    return {
        **payload,
        "ensemble_manifest": output_path.resolve().as_posix(),
        "ensemble_manifest_sha256": sha256_file(output_path),
    }


def load_pitch3_ensemble(
    *,
    manifest_path: Path,
    contract: Pitch3Contract,
    training_manifest: Path,
    device: torch.device,
    expected_manifest_sha256: str | None = None,
) -> tuple[Pitch3ControlHeadEnsemble, dict[str, Any], str]:
    """Load and strictly validate a frozen Pitch-3 ensemble artifact."""

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    ensemble_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and ensemble_sha256 != expected_manifest_sha256.lower():
        raise LTSNContractError("Pitch-3 ensemble manifest SHA-256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("stage") != ENSEMBLE_STAGE:
        raise LTSNContractError("Pitch-3 ensemble manifest schema/stage mismatch")
    expected = {
        "model_family": "latent_topology_control_head_pitch3_ensemble",
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "feature_order": list(contract.feature_order),
        "classifier_sha256": contract.classifier_sha256,
        "training_manifest_sha256": sha256_file(training_manifest),
        "aggregation": ENSEMBLE_AGGREGATION,
        "selection_scope": "development_only",
        "qualification_split_consumed": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise LTSNContractError(f"Pitch-3 ensemble {name} binding mismatch")
    raw_members = payload.get("members")
    if not isinstance(raw_members, list) or len(raw_members) < 2:
        raise LTSNContractError("Pitch-3 ensemble members are missing")
    if not isinstance(payload.get("ensemble_id"), str) or not payload["ensemble_id"].strip():
        raise LTSNContractError("Pitch-3 ensemble id is missing")
    names = [str(raw.get("name", "")) for raw in raw_members if isinstance(raw, Mapping)]
    if (
        len(names) != len(raw_members)
        or any(not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise LTSNContractError("Pitch-3 ensemble member names must be unique and non-empty")
    weights: list[float] = []
    models: list[LatentTopologyControlHead] = []
    reference_config: dict[str, Any] | None = None
    for raw in raw_members:
        try:
            weight = float(raw.get("weight")) if isinstance(raw, Mapping) else math.nan
        except (TypeError, ValueError) as _error:
            weight = math.nan
        weights.append(weight)
    if any(not math.isfinite(value) or value <= 0.0 for value in weights) or not math.isclose(
        sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise LTSNContractError("Pitch-3 ensemble weights must be positive and sum to one")
    weights = []
    for raw in raw_members:
        if not isinstance(raw, Mapping):
            raise LTSNContractError("Pitch-3 ensemble member entry is malformed")
        checkpoint_path = (manifest_path.parent / str(raw.get("checkpoint_path", ""))).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if checkpoint_sha256 != str(raw.get("checkpoint_sha256", "")).lower():
            raise LTSNContractError("Pitch-3 ensemble member SHA-256 mismatch")
        checkpoint, metadata, model_config = _checkpoint_payload(
            checkpoint_path,
            contract=contract,
            training_manifest_sha256=expected["training_manifest_sha256"],
        )
        if metadata.get("training_config_sha256") != raw.get("training_config_sha256"):
            raise LTSNContractError("ensemble member training config binding mismatch")
        if metadata.get("architecture_revision", "") != raw.get("architecture_revision", ""):
            raise LTSNContractError("ensemble member architecture revision binding mismatch")
        if metadata.get("band_training_objective", {}) != raw.get("band_training_objective", {}):
            raise LTSNContractError("ensemble member band objective binding mismatch")
        if metadata.get("ood_transform_version", "") != payload.get("ood_transform_version", ""):
            raise LTSNContractError("ensemble member OOD transform binding mismatch")
        if metadata.get("ood_label_source", "") != payload.get("ood_label_source", ""):
            raise LTSNContractError("ensemble member OOD label source binding mismatch")
        if reference_config is None:
            reference_config = model_config
        elif model_config != reference_config:
            raise LTSNContractError("Pitch-3 ensemble member model configs differ")
        model = LatentTopologyControlHead(contract, Pitch3ControlHeadConfig(**model_config))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        models.append(model)
        try:
            weights.append(float(raw.get("weight")))
        except (TypeError, ValueError) as error:
            raise LTSNContractError("Pitch-3 ensemble member weight is malformed") from error
    ensemble = Pitch3ControlHeadEnsemble(models, weights, contract).to(device).eval()
    return ensemble, dict(payload), ensemble_sha256
