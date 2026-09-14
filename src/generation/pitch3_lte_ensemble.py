"""Auditable equal-weight scalar-energy ensembles for V3.3 latent guidance."""

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


def build_pitch3_lte_ensemble_manifest(
    *, manifest_paths: list[Path], output_path: Path
) -> dict[str, Any]:
    if len(manifest_paths) < 2:
        raise ValueError("V3.3 ensemble requires at least two training manifests")
    members: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    seeds: set[int] = set()
    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("model_family") != LTE_MODEL_FAMILY:
            raise LTSNContractError("V3.3 ensemble member has the wrong model family")
        if payload.get("architecture_revision") != "v3.3_family_round_robin_dual_potential":
            raise LTSNContractError("V3.3 ensemble accepts only V3.3 checkpoints")
        seed = int(payload["seed"])
        if seed in seeds:
            raise LTSNContractError("V3.3 ensemble seeds must be unique")
        seeds.add(seed)
        checkpoint = Path(payload["checkpoint"]).resolve()
        checkpoint_sha256 = str(payload["checkpoint_sha256"]).lower()
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha256:
            raise LTSNContractError("V3.3 ensemble checkpoint is missing or hash-mismatched")
        contract = {
            "fingerprint_json_sha256": payload["fingerprint_json_sha256"],
            "training_manifest_sha256": payload["training_manifest_sha256"],
            "dataset_plan_sha256": payload["dataset_plan_sha256"],
            "guidance_steps": payload["guidance_steps"],
            "training_radius_ratio": payload["training_radius_ratio"],
            "maximum_guidance_update_ratio": payload["maximum_guidance_update_ratio"],
        }
        if shared is None:
            shared = contract
        elif contract != shared:
            raise LTSNContractError("V3.3 ensemble members do not share one frozen contract")
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
    assert shared is not None
    report = {
        "schema_version": 1,
        "stage": "pitch3_lte_v33_ensemble",
        "model_family": LTE_MODEL_FAMILY,
        "ensemble_kind": "equal_weight_scalar_energy_v33",
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
        raise LTSNContractError("V3.3 ensemble manifest SHA-256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("model_family") != LTE_MODEL_FAMILY
        or payload.get("ensemble_kind") != "equal_weight_scalar_energy_v33"
    ):
        raise LTSNContractError("invalid V3.3 scalar-energy ensemble")
    members = payload.get("members", [])
    if len(members) < 2 or len(members) != int(payload.get("member_count", -1)):
        raise LTSNContractError("V3.3 ensemble member count is invalid")
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
        ) or metadata.get("training_manifest_sha256") != payload.get(
            "training_manifest_sha256"
        ):
            raise LTSNContractError("V3.3 ensemble member detached from ensemble contract")
        models.append(model)
        weights.append(float(member["weight"]))
    expected_weight = 1.0 / len(members)
    if any(abs(weight - expected_weight) > 1e-12 for weight in weights):
        raise LTSNContractError("V3.3 ensemble weights are not the frozen equal average")
    ensemble = Pitch3LTEEnergyEnsemble(models, torch.tensor(weights)).to(device)
    ensemble.eval().requires_grad_(False)
    metadata = {
        **payload,
        "ensemble_manifest": str(manifest_path),
        "ensemble_manifest_sha256": sha256_file(manifest_path),
    }
    return ensemble, metadata
