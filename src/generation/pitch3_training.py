"""Training pipeline for the three-coordinate latent topology control head."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .latent_topology_control_head import (
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
    Pitch3ControlOutput,
)
from .ltsn_contract import LTSNContractError, sha256_file
from .pitch3_contract import PITCH3_DIMENSIONS, Pitch3Contract, load_pitch3_contract

SPLITS = {"train", "development", "calibration", "qualification"}


@dataclass(frozen=True, slots=True)
class Pitch3TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    micro_batch_size: int = 8
    max_epochs: int = 40
    minimum_epochs: int = 8
    early_stopping_patience: int = 7
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    use_bf16: bool = True
    seed: int = 20260913
    require_ood_both_classes: bool = True


@dataclass(frozen=True, slots=True)
class Pitch3LossWeights:
    coordinate: float = 1.0
    nll: float = 0.25
    focus: float = 0.25
    ood: float = 0.1


@dataclass(frozen=True, slots=True)
class Pitch3Snapshot:
    sample_id: str
    prompt_id: str
    trajectory_id: str
    split: str
    step_number: int
    timestep: float
    latent_path: Path
    latent_sha256: str
    coordinates: tuple[float, float, float]
    focus_logit: float
    ood_label: float
    is_final: bool


def load_pitch3_training_config(
    path: Path,
) -> tuple[Pitch3ControlHeadConfig, Pitch3TrainingConfig, Pitch3LossWeights]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model = Pitch3ControlHeadConfig(**payload.get("model", {}))
    training = Pitch3TrainingConfig(**payload.get("training", {}))
    loss = Pitch3LossWeights(**payload.get("loss", {}))
    if training.micro_batch_size < 1 or training.max_epochs < 1:
        raise ValueError("training batch size and epochs must be positive")
    if training.minimum_epochs < 1 or training.minimum_epochs > training.max_epochs:
        raise ValueError("minimum_epochs must be in [1,max_epochs]")
    if training.early_stopping_patience < 1 or training.gradient_clip_norm <= 0.0:
        raise ValueError("early stopping patience and gradient clip must be positive")
    if any(value < 0.0 for value in asdict(loss).values()) or loss.coordinate <= 0.0:
        raise ValueError("loss weights must be non-negative with coordinate > 0")
    return model, training, loss


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def read_pitch3_manifest(path: Path, contract: Pitch3Contract) -> list[Pitch3Snapshot]:
    records: list[Pitch3Snapshot] = []
    prompt_splits: dict[str, set[str]] = {}
    trajectory_splits: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("fingerprint_json_sha256", "").lower() != contract.artifact_sha256:
                raise LTSNContractError("Pitch-3 manifest fingerprint hash mismatch")
            if raw.get("label_scope") != "per_snapshot_exact":
                raise LTSNContractError("Pitch-3 snapshots require independent exact labels")
            feature_order = json.loads(raw.get("feature_order_json", "null"))
            if feature_order != list(contract.feature_order):
                raise LTSNContractError("Pitch-3 manifest feature order mismatch")
            values = tuple(float(value) for value in json.loads(raw["coordinates_json"]))
            if len(values) != PITCH3_DIMENSIONS or not all(math.isfinite(v) for v in values):
                raise LTSNContractError("Pitch-3 coordinates must contain three finite values")
            split = raw["split"].strip().lower()
            if split not in SPLITS:
                raise LTSNContractError(f"unsupported Pitch-3 split: {split}")
            latent_path = (path.parent / raw["latent_path"]).resolve()
            if not latent_path.is_file() or sha256_file(latent_path) != raw["latent_sha256"]:
                raise LTSNContractError("Pitch-3 latent is missing or hash-mismatched")
            step = int(raw["step_number"])
            is_final = _parse_bool(raw.get("is_final", "false"))
            if not is_final and step not in {4, 5, 6}:
                raise LTSNContractError("Pitch-3 non-final snapshots must be steps 4, 5, or 6")
            ood = float(raw.get("ood_label", 0.0))
            focus = float(raw["focus_logit"])
            if ood not in {0.0, 1.0} or not math.isfinite(focus):
                raise LTSNContractError("Pitch-3 OOD/focus labels are malformed")
            prompt_id = raw["prompt_id"]
            trajectory_id = raw["trajectory_id"]
            prompt_splits.setdefault(prompt_id, set()).add(split)
            trajectory_splits.setdefault(trajectory_id, set()).add(split)
            records.append(
                Pitch3Snapshot(
                    sample_id=raw["sample_id"],
                    prompt_id=prompt_id,
                    trajectory_id=trajectory_id,
                    split=split,
                    step_number=step,
                    timestep=float(raw["timestep"]),
                    latent_path=latent_path,
                    latent_sha256=raw["latent_sha256"].lower(),
                    coordinates=values,  # type: ignore[arg-type]
                    focus_logit=focus,
                    ood_label=ood,
                    is_final=is_final,
                )
            )
    if not records:
        raise LTSNContractError("Pitch-3 training manifest is empty")
    if any(len(values) != 1 for values in prompt_splits.values()):
        raise LTSNContractError("Pitch-3 prompt leakage detected")
    if any(len(values) != 1 for values in trajectory_splits.values()):
        raise LTSNContractError("Pitch-3 trajectory leakage detected")
    return records


class Pitch3Dataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[Pitch3Snapshot]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latent = np.load(record.latent_path, allow_pickle=False)
        if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
            raise LTSNContractError(f"invalid Pitch-3 latent array: {record.latent_path}")
        return {
            "sample_id": record.sample_id,
            "latent": torch.from_numpy(np.asarray(latent, dtype=np.float32)),
            "timestep": torch.tensor(record.timestep, dtype=torch.float32),
            "step_number": torch.tensor(record.step_number, dtype=torch.float32),
            "coordinates": torch.tensor(record.coordinates, dtype=torch.float32),
            "focus_logit": torch.tensor(record.focus_logit, dtype=torch.float32),
            "ood_label": torch.tensor(record.ood_label, dtype=torch.float32),
        }


def collate_pitch3(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty Pitch-3 batch")
    length = max(int(item["latent"].shape[0]) for item in items)
    latent = torch.zeros(len(items), length, 64, dtype=torch.float32)
    mask = torch.zeros(len(items), length, dtype=torch.bool)
    for index, item in enumerate(items):
        frames = int(item["latent"].shape[0])
        latent[index, :frames] = item["latent"]
        mask[index, :frames] = True
    return {
        "sample_id": [item["sample_id"] for item in items],
        "latent": latent,
        "attention_mask": mask,
        "timestep": torch.stack([item["timestep"] for item in items]),
        "step_number": torch.stack([item["step_number"] for item in items]),
        "coordinates": torch.stack([item["coordinates"] for item in items]),
        "focus_logit": torch.stack([item["focus_logit"] for item in items]),
        "ood_label": torch.stack([item["ood_label"] for item in items]),
    }


def pitch3_loss(
    output: Pitch3ControlOutput,
    coordinates: Tensor,
    focus_logit: Tensor,
    ood_label: Tensor,
    weights: Pitch3LossWeights,
    *,
    ood_positive_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if coordinates.ndim != 2 or coordinates.shape[1] != PITCH3_DIMENSIONS:
        raise ValueError("Pitch-3 targets must have shape [B,3]")
    id_mask = ood_label < 0.5
    if id_mask.any():
        error = output.coordinate_mean[id_mask] - coordinates[id_mask]
        coordinate = F.smooth_l1_loss(output.coordinate_mean[id_mask], coordinates[id_mask])
        nll = 0.5 * (
            torch.exp(-output.coordinate_logvar[id_mask]) * error.square()
            + output.coordinate_logvar[id_mask]
        ).mean()
        focus = F.smooth_l1_loss(output.focus_logit[id_mask], focus_logit[id_mask])
    else:
        zero = output.coordinate_mean.sum() * 0.0
        coordinate = zero
        nll = zero
        focus = zero
    positive_weight = torch.tensor(
        ood_positive_weight, device=ood_label.device, dtype=ood_label.dtype
    )
    ood = F.binary_cross_entropy_with_logits(
        output.ood_logit, ood_label, pos_weight=positive_weight
    )
    total = (
        weights.coordinate * coordinate
        + weights.nll * nll
        + weights.focus * focus
        + weights.ood * ood
    )
    return total, {"coordinate": coordinate, "nll": nll, "focus": focus, "ood": ood}


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch(
    model: LatentTopologyControlHead,
    loader: DataLoader[dict[str, Any]],
    weights: Pitch3LossWeights,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    use_bf16: bool,
    ood_positive_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "coordinate": 0.0, "nll": 0.0, "focus": 0.0, "ood": 0.0}
    samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw in loader:
            batch = _device_batch(raw, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(
                    batch["latent"],
                    batch["timestep"],
                    batch["step_number"],
                    batch["attention_mask"],
                )
                loss, parts = pitch3_loss(
                    output,
                    batch["coordinates"],
                    batch["focus_logit"],
                    batch["ood_label"],
                    weights,
                    ood_positive_weight=ood_positive_weight,
                )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            count = int(batch["latent"].shape[0])
            samples += count
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += float(value.detach()) * count
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_pitch3_control_head(
    *,
    fingerprint_path: Path,
    training_manifest: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    contract = load_pitch3_contract(fingerprint_path)
    model_config, training, weights = load_pitch3_training_config(config_path)
    records = read_pitch3_manifest(training_manifest, contract)
    train_records = [record for record in records if record.split == "train"]
    development = [record for record in records if record.split == "development"]
    if not train_records or not development:
        raise LTSNContractError("Pitch-3 training requires train and development splits")
    train_ood = np.asarray([record.ood_label for record in train_records], dtype=float)
    positives = int(np.count_nonzero(train_ood >= 0.5))
    negatives = int(len(train_ood) - positives)
    development_ood = np.asarray([record.ood_label for record in development], dtype=float)
    development_positives = int(np.count_nonzero(development_ood >= 0.5))
    development_negatives = int(len(development_ood) - development_positives)
    if training.require_ood_both_classes:
        if not positives or not negatives:
            raise LTSNContractError("Pitch-3 training requires ID and OOD train samples")
        if not development_positives or not development_negatives:
            raise LTSNContractError(
                "Pitch-3 training requires ID and OOD development samples"
            )
    positive_weight = 1.0 if not positives else max(1.0, negatives / positives)
    _set_seed(training.seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    model = LatentTopologyControlHead(contract, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    train_generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(
        Pitch3Dataset(train_records),
        batch_size=training.micro_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=training.num_workers,
        collate_fn=collate_pitch3,
    )
    development_loader = DataLoader(
        Pitch3Dataset(development),
        batch_size=training.micro_batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=collate_pitch3,
    )
    use_bf16 = (
        training.use_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, training.max_epochs + 1):
        train_metrics = _epoch(
            model,
            train_loader,
            weights,
            device,
            optimizer=optimizer,
            gradient_clip_norm=training.gradient_clip_norm,
            use_bf16=use_bf16,
            ood_positive_weight=positive_weight,
        )
        development_metrics = _epoch(
            model,
            development_loader,
            weights,
            device,
            optimizer=None,
            gradient_clip_norm=training.gradient_clip_norm,
            use_bf16=use_bf16,
            ood_positive_weight=positive_weight,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "development": development_metrics}
        )
        if development_metrics["loss"] < best_loss:
            best_loss = development_metrics["loss"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= training.minimum_epochs and stale >= training.early_stopping_patience:
            break
    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError("Pitch-3 training did not produce a finite checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"pitch3_control_head_seed_{training.seed}.pt"
    temporary = checkpoint.with_suffix(".pt.part")
    metadata = {
        "schema_version": 1,
        "model_family": "latent_topology_control_head_pitch3",
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": PITCH3_DIMENSIONS,
        "feature_order": list(contract.feature_order),
        "classifier_sha256": contract.classifier_sha256,
        "training_config_sha256": sha256_file(config_path),
        "training_manifest_sha256": sha256_file(training_manifest),
        "seed": training.seed,
        "device": str(device),
        "precision": "bf16_forward_fp32_loss" if use_bf16 else "fp32",
        "trainable_parameters": model.trainable_parameters,
        "best_development_loss": best_loss,
        "epochs_completed": len(history),
        "ood_class_counts": {
            "train": {"id": negatives, "ood": positives},
            "development": {
                "id": development_negatives,
                "ood": development_positives,
            },
        },
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    torch.save(
        {
            "metadata": metadata,
            "model_config": asdict(model_config),
            "training_config": asdict(training),
            "loss_weights": asdict(weights),
            "model_state_dict": best_state,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    result = {
        **metadata,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    manifest = output_dir / "pitch3_control_head_manifest.json"
    manifest_temporary = manifest.with_suffix(".json.part")
    manifest_temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest)
    return result
