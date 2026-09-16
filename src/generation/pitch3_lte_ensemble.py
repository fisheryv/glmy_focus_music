"""Auditable equal-weight scalar-energy ensembles for V3.3--V3.8 guidance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_json_atomic
from .pitch3_lte import Pitch3LTEEnergyEnsemble
from .pitch3_lte_data import LTE_MODEL_FAMILY
from .pitch3_lte_training import load_pitch3_lte_checkpoint

_ENSEMBLE_KINDS = {
    "v3.3_family_round_robin_dual_potential": "equal_weight_scalar_energy_v33",
    "v3.4_structured_coordinate_family_energy": "equal_weight_scalar_energy_v34",
    "v3.5_structured_anchored_coordinate_potential": ("equal_weight_anchored_scalar_energy_v35"),
    "v3.5r_minimal_global_anchored_direction": ("equal_weight_anchored_scalar_energy_v35r"),
    "v3.6_tail_calibrated_coordinate_regression": ("equal_weight_anchored_scalar_energy_v36_tcr"),
    "v3.7_direct_topology_energy": "equal_weight_direct_topology_energy_v37_dte",
    "v3.8a_logit_stratified_rank_energy": "equal_weight_direct_topology_energy_v38a",
    "v3.8b_ordinal_calibrated_direction": "equal_weight_direct_topology_energy_v38b",
}


def build_pitch3_lte_ensemble_manifest(
    *, manifest_paths: list[Path], output_path: Path
) -> dict[str, Any]:
    if len(manifest_paths) < 2:
        raise ValueError("V3-LTE ensemble requires at least two training manifests")
    members: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    seeds: set[int] = set()
    architecture_revision: str | None = None
    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("model_family") != LTE_MODEL_FAMILY:
            raise LTSNContractError("V3-LTE ensemble member has the wrong model family")
        if payload.get("validation_scope") == "train_family_cv":
            raise LTSNContractError("train-family CV checkpoints cannot form a guidance ensemble")
        member_revision = str(payload.get("architecture_revision", ""))
        if member_revision not in _ENSEMBLE_KINDS:
            raise LTSNContractError("V3-LTE ensemble member has an unsupported architecture")
        if architecture_revision is None:
            architecture_revision = member_revision
        elif member_revision != architecture_revision:
            raise LTSNContractError("V3-LTE ensemble cannot mix architecture revisions")
        seed = int(payload["seed"])
        if seed in seeds:
            raise LTSNContractError("V3-LTE ensemble seeds must be unique")
        seeds.add(seed)
        checkpoint = Path(payload["checkpoint"]).resolve()
        checkpoint_sha256 = str(payload["checkpoint_sha256"]).lower()
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha256:
            raise LTSNContractError("V3-LTE ensemble checkpoint is missing or hash-mismatched")
        contract = {
            "fingerprint_json_sha256": payload["fingerprint_json_sha256"],
            "training_manifest_sha256": payload["training_manifest_sha256"],
            "dataset_plan_sha256": payload["dataset_plan_sha256"],
            "guidance_steps": payload["guidance_steps"],
            "training_radius_ratio": payload["training_radius_ratio"],
            "maximum_guidance_update_ratio": payload["maximum_guidance_update_ratio"],
        }
        if member_revision == "v3.8a_logit_stratified_rank_energy":
            contract.update(
                {
                    "ranking_contract": payload["ranking_contract"],
                    "source_training_config_sha256": payload["source_training_config_sha256"],
                    "local_residual_mode": payload["local_residual_mode"],
                    "validation_scope": payload["validation_scope"],
                }
            )
        if member_revision == "v3.8b_ordinal_calibrated_direction":
            from .pitch3_lte_v38b_protocol import verify_manifest

            verify_manifest(manifest_path)
            contract.update(
                {
                    key: payload[key]
                    for key in (
                        "source_training_config_sha256",
                        "local_residual_mode",
                        "validation_scope",
                        "experiment_contract",
                        "checkpoint_selection",
                        "implementation_sha256",
                    )
                }
            )
        if shared is None:
            shared = contract
        elif contract != shared:
            raise LTSNContractError("V3-LTE ensemble members do not share one frozen contract")
        members.append(
            {
                "seed": seed,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "weight": 1.0 / len(manifest_paths),
            }
        )
    assert shared is not None and architecture_revision is not None
    ensemble_kind = _ENSEMBLE_KINDS[architecture_revision]
    report = {
        "schema_version": 1,
        "stage": (
            "pitch3_lte_v38b_ensemble"
            if ensemble_kind == "equal_weight_direct_topology_energy_v38b"
            else "pitch3_lte_v38a_ensemble"
            if ensemble_kind == "equal_weight_direct_topology_energy_v38a"
            else "pitch3_lte_v37_dte_ensemble"
            if ensemble_kind == "equal_weight_direct_topology_energy_v37_dte"
            else (
                "pitch3_lte_v36_tcr_ensemble"
                if ensemble_kind == "equal_weight_anchored_scalar_energy_v36_tcr"
                else (
                    "pitch3_lte_v35r_ensemble"
                    if ensemble_kind == "equal_weight_anchored_scalar_energy_v35r"
                    else (
                        "pitch3_lte_v35_ensemble"
                        if ensemble_kind == "equal_weight_anchored_scalar_energy_v35"
                        else (
                            "pitch3_lte_v34_ensemble"
                            if ensemble_kind == "equal_weight_scalar_energy_v34"
                            else "pitch3_lte_v33_ensemble"
                        )
                    )
                )
            )
        ),
        "model_family": LTE_MODEL_FAMILY,
        "architecture_revision": architecture_revision,
        "ensemble_kind": ensemble_kind,
        "member_count": len(members),
        "members": sorted(members, key=lambda item: item["seed"]),
        **shared,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    write_json_atomic(output_path, report)
    result = dict(report)
    result["ensemble_manifest"] = str(output_path.resolve())
    result["ensemble_manifest_sha256"] = sha256_file(output_path)
    return result


def load_pitch3_lte_ensemble(
    manifest_path: Path,
    *,
    device: torch.device,
    expected_sha256: str | None = None,
) -> tuple[Pitch3LTEEnergyEnsemble, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    if expected_sha256 is not None and sha256_file(manifest_path) != expected_sha256.lower():
        raise LTSNContractError("V3-LTE ensemble manifest SHA-256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = str(payload.get("architecture_revision", ""))
    if not revision and payload.get("ensemble_kind") == "equal_weight_scalar_energy_v33":
        # Preserve already-published V3.3 manifests created before the
        # architecture field became explicit in V3.4.
        revision = "v3.3_family_round_robin_dual_potential"
    expected_kind = _ENSEMBLE_KINDS.get(revision)
    if (
        payload.get("model_family") != LTE_MODEL_FAMILY
        or payload.get("ensemble_kind") != expected_kind
    ):
        raise LTSNContractError("invalid V3-LTE scalar-energy ensemble")
    members = payload.get("members", [])
    if len(members) < 2 or len(members) != int(payload.get("member_count", -1)):
        raise LTSNContractError("V3-LTE ensemble member count is invalid")
    models = []
    weights = []
    for member in members:
        checkpoint = Path(member["checkpoint"])
        model, metadata = load_pitch3_lte_checkpoint(
            checkpoint,
            device=device,
            expected_sha256=str(member["checkpoint_sha256"]),
        )
        if metadata.get("fingerprint_json_sha256") != payload.get(
            "fingerprint_json_sha256"
        ) or metadata.get("training_manifest_sha256") != payload.get("training_manifest_sha256"):
            raise LTSNContractError("V3-LTE ensemble member detached from ensemble contract")
        if metadata.get("architecture_revision") != revision:
            raise LTSNContractError("V3-LTE ensemble member architecture changed")
        if metadata.get("validation_scope") == "train_family_cv":
            raise LTSNContractError("train-family CV checkpoints are diagnostic only")
        if revision == "v3.8a_logit_stratified_rank_energy" and any(
            metadata.get(key) != payload.get(key)
            for key in (
                "ranking_contract",
                "source_training_config_sha256",
                "local_residual_mode",
                "validation_scope",
            )
        ):
            raise LTSNContractError("V3.8-A ensemble training contracts differ")
        if revision == "v3.8b_ordinal_calibrated_direction" and any(
            metadata.get(key) != payload.get(key)
            for key in (
                "experiment_contract",
                "checkpoint_selection",
                "implementation_sha256",
                "source_training_config_sha256",
                "local_residual_mode",
                "validation_scope",
            )
        ):
            raise LTSNContractError("V3.8-B ensemble training contracts differ")
        models.append(model)
        weights.append(float(member["weight"]))
    expected_weight = 1.0 / len(members)
    if any(abs(weight - expected_weight) > 1e-12 for weight in weights):
        raise LTSNContractError("V3-LTE ensemble weights are not the frozen equal average")
    ensemble = Pitch3LTEEnergyEnsemble(models, torch.tensor(weights)).to(device)
    ensemble.eval().requires_grad_(False)
    metadata = {
        **payload,
        "ensemble_manifest": str(manifest_path),
        "ensemble_manifest_sha256": sha256_file(manifest_path),
    }
    return ensemble, metadata
