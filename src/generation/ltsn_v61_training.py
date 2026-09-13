"""V6.1 multitask fine-tuning with existing exact TAC derivatives."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .ltsn_v6_final_target import (
    V6FinalTargetDataset,
    V6FinalTargetModel,
    V6HeadConfig,
    V6ScreenConfig,
    _normalization,
    _predict_view,
    _to_device,
    _view_metrics,
    _weighted_huber,
    collate_v6,
    load_v6_model,
    read_v6_view,
)
from .ltsn_v6_tac_correction import _cluster_bootstrap, _spearman
from .ltsn_v61_data import V61_EXPERIMENT, V61_PAIR_SPLITS
from .path_homology_surrogate import LTSNConfig


@dataclass(frozen=True, slots=True)
class V61TrainingConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_fraction: float = 0.05
    minimum_learning_rate: float = 1e-6
    micro_batch_size: int = 8
    direction_batch_size: int = 4
    max_epochs: int = 40
    minimum_epochs: int = 10
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 5.0
    huber_delta: float = 1.0
    final_target_loss_weight: float = 1.0
    direction_huber_weight: float = 1.0
    direction_sign_weight: float = 0.25
    selection_final_weight: float = 0.25
    selection_direction_huber_weight: float = 1.0
    selection_direction_agreement_weight: float = 0.25
    num_workers: int = 4
    seed: int = 20260716

    def validate(self) -> None:
        if self.micro_batch_size < 1 or self.direction_batch_size < 1:
            raise LTSNContractError("V6.1 batch sizes must be positive")
        if not 1 <= self.minimum_epochs <= self.max_epochs:
            raise LTSNContractError("invalid V6.1 epoch limits")
        positive = (
            self.learning_rate,
            self.minimum_learning_rate,
            self.gradient_clip_norm,
            self.huber_delta,
            self.final_target_loss_weight,
            self.direction_huber_weight,
            self.selection_final_weight,
            self.selection_direction_huber_weight,
        )
        if any(value <= 0.0 for value in positive):
            raise LTSNContractError("V6.1 positive training settings must exceed zero")
        if self.direction_sign_weight < 0.0 or self.selection_direction_agreement_weight < 0.0:
            raise LTSNContractError("V6.1 auxiliary weights cannot be negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise LTSNContractError("V6.1 warmup fraction must lie in [0,1)")


@dataclass(frozen=True, slots=True)
class V61PairRecord:
    pair_id: str
    anchor_sample_id: str
    split: str
    step_number: int
    timestep: float
    rms_ratio: float
    minus_path: Path
    plus_path: Path
    minus_sha256: str
    plus_sha256: str
    exact_derivative: float


def _values(cls: type[Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise LTSNContractError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return dict(raw)


def load_v61_config(path: Path) -> tuple[V61TrainingConfig, V6ScreenConfig]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    training = V61TrainingConfig(**_values(V61TrainingConfig, payload.get("training", {})))
    screen = V6ScreenConfig(**_values(V6ScreenConfig, payload.get("screen", {})))
    training.validate()
    screen.validate()
    return training, screen


def read_v61_pairs(path: Path) -> list[V61PairRecord]:
    rows: list[V61PairRecord] = []
    checked: dict[Path, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            minus = (path.parent / raw["minus_latent_path"]).resolve()
            plus = (path.parent / raw["plus_latent_path"]).resolve()
            for latent, expected in (
                (minus, raw["minus_latent_sha256"]),
                (plus, raw["plus_latent_sha256"]),
            ):
                if not latent.is_file():
                    raise LTSNContractError(f"V6.1 pair latent is missing: {latent}")
                actual = checked.setdefault(latent, sha256_file(latent))
                if actual != expected:
                    raise LTSNContractError(f"V6.1 pair latent hash mismatch: {latent}")
            split = raw["v61_split"]
            derivative = float(raw["exact_tac_derivative"])
            if split not in V61_PAIR_SPLITS or not math.isfinite(derivative):
                raise LTSNContractError("invalid V6.1 pair split or derivative")
            rows.append(
                V61PairRecord(
                    pair_id=raw["pair_id"],
                    anchor_sample_id=raw["anchor_sample_id"],
                    split=split,
                    step_number=int(raw["step_number"]),
                    timestep=float(raw["timestep"]),
                    rms_ratio=float(raw["rms_ratio"]),
                    minus_path=minus,
                    plus_path=plus,
                    minus_sha256=raw["minus_latent_sha256"],
                    plus_sha256=raw["plus_latent_sha256"],
                    exact_derivative=derivative,
                )
            )
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise LTSNContractError("V6.1 pair view is empty or contains duplicate IDs")
    if {row.split for row in rows} != set(V61_PAIR_SPLITS):
        raise LTSNContractError("V6.1 pair view is missing a frozen split")
    fit_anchors = {row.anchor_sample_id for row in rows if row.split == "direction_fit"}
    validation_anchors = {
        row.anchor_sample_id for row in rows if row.split == "direction_validation"
    }
    if fit_anchors & validation_anchors:
        raise LTSNContractError("V6.1 direction fit and validation anchors overlap")
    return rows


class V61PairDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[V61PairRecord], split: str) -> None:
        self.records = tuple(row for row in records if row.split == split)
        if not self.records:
            raise ValueError(f"no V6.1 pairs for split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        minus = np.load(row.minus_path, allow_pickle=False)
        plus = np.load(row.plus_path, allow_pickle=False)
        if (
            minus.ndim != 2
            or plus.shape != minus.shape
            or minus.shape[1] != 64
            or not np.isfinite(minus).all()
            or not np.isfinite(plus).all()
        ):
            raise LTSNContractError(f"invalid V6.1 pair latent: {row.pair_id}")
        return {
            "pair_id": row.pair_id,
            "anchor_sample_id": row.anchor_sample_id,
            "minus": torch.from_numpy(np.asarray(minus, dtype=np.float32)),
            "plus": torch.from_numpy(np.asarray(plus, dtype=np.float32)),
            "timestep": torch.tensor(row.timestep, dtype=torch.float32),
            "step_number": torch.tensor(row.step_number, dtype=torch.long),
            "rms_ratio": torch.tensor(row.rms_ratio, dtype=torch.float32),
            "exact_derivative": torch.tensor(row.exact_derivative, dtype=torch.float32),
        }


def collate_v61_pairs(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(int(item["minus"].shape[0]) for item in items)
    minus = torch.zeros(len(items), maximum, 64, dtype=torch.float32)
    plus = torch.zeros_like(minus)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["minus"].shape[0])
        minus[index, :length] = item["minus"]
        plus[index, :length] = item["plus"]
        mask[index, :length] = True
    return {
        "pair_id": [str(item["pair_id"]) for item in items],
        "anchor_sample_id": [str(item["anchor_sample_id"]) for item in items],
        "minus": minus,
        "plus": plus,
        "attention_mask": mask,
        **{
            key: torch.stack([item[key] for item in items])
            for key in ("timestep", "step_number", "rms_ratio", "exact_derivative")
        },
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    return device


def _schedule(update: int, warmup: int, total: int, minimum: float) -> float:
    if update < warmup:
        return max(minimum, (update + 1) / max(1, warmup))
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))


def _direction_transform(
    preparation: Mapping[str, Any], steps: Tensor, reference: Tensor
) -> tuple[Tensor, Tensor]:
    raw = preparation["direction_transforms_by_step"]
    centers = torch.zeros_like(steps, dtype=torch.float32, device=reference.device)
    scales = torch.zeros_like(centers)
    for step in (4, 5, 6):
        mask = steps == step
        centers[mask] = float(raw[str(step)]["center"])
        scales[mask] = float(raw[str(step)]["scale"])
    if torch.any(scales <= 0.0):
        raise LTSNContractError("V6.1 direction transform is missing a step")
    return centers, scales


def _predicted_derivative(
    model: V6FinalTargetModel,
    batch: Mapping[str, Tensor],
    target_center: Tensor,
    target_scale: Tensor,
) -> Tensor:
    minus = model(
        batch["minus"], batch["timestep"], batch["step_number"], batch["attention_mask"]
    )
    plus = model(
        batch["plus"], batch["timestep"], batch["step_number"], batch["attention_mask"]
    )
    minus = minus * target_scale + target_center
    plus = plus * target_scale + target_center
    weights = minus.new_tensor((0.5, 0.25, 0.25))
    return ((minus - plus) @ weights) / (2.0 * batch["rms_ratio"])


def _direction_loss(
    prediction: Tensor,
    exact: Tensor,
    steps: Tensor,
    preparation: Mapping[str, Any],
    config: V61TrainingConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    center, scale = _direction_transform(preparation, steps, prediction)
    predicted_normalized = (prediction.float() - center) / scale
    exact_normalized = (exact.float() - center) / scale
    huber = F.huber_loss(
        predicted_normalized, exact_normalized, delta=config.huber_delta, reduction="mean"
    )
    sign = torch.sign(exact.float())
    sign_loss = F.softplus(-sign * prediction.float() / scale).mean()
    total = config.direction_huber_weight * huber + config.direction_sign_weight * sign_loss
    return total, huber, sign_loss


def _pair_predictions(
    model: V6FinalTargetModel,
    loader: DataLoader[Any],
    device: torch.device,
    target_center: np.ndarray,
    target_scale: np.ndarray,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    center_tensor = torch.tensor(target_center, device=device)
    scale_tensor = torch.tensor(target_scale, device=device)
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            prediction = _predicted_derivative(model, batch, center_tensor, scale_tensor)
            for index, pair_id in enumerate(raw["pair_id"]):
                exact = float(raw["exact_derivative"][index])
                predicted = float(prediction[index].detach().cpu())
                rows.append(
                    {
                        "pair_id": pair_id,
                        "anchor_sample_id": raw["anchor_sample_id"][index],
                        "step_number": int(raw["step_number"][index]),
                        "rms_ratio": float(raw["rms_ratio"][index]),
                        "exact_tac_derivative": exact,
                        "predicted_tac_derivative": predicted,
                        "exact_tac_informative": True,
                        "direction_correct": int(exact * predicted > 0.0),
                    }
                )
    return rows


def _pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "direction_agreement": sum(int(row["direction_correct"]) for row in rows) / len(rows),
        "derivative_spearman": _spearman(
            [float(row["exact_tac_derivative"]) for row in rows],
            [float(row["predicted_tac_derivative"]) for row in rows],
        ),
    }


def _pair_huber(
    rows: Sequence[Mapping[str, Any]], preparation: Mapping[str, Any], delta: float
) -> float:
    losses = []
    for row in rows:
        transform = preparation["direction_transforms_by_step"][str(row["step_number"])]
        scale = float(transform["scale"])
        center = float(transform["center"])
        residual = (
            (float(row["predicted_tac_derivative"]) - center) / scale
            - (float(row["exact_tac_derivative"]) - center) / scale
        )
        absolute = abs(residual)
        losses.append(
            0.5 * residual * residual
            if absolute <= delta
            else delta * (absolute - 0.5 * delta)
        )
    return float(np.mean(losses))


def _validate_preparation(
    path: Path,
    *,
    fingerprint_path: Path,
    target_path: Path,
    v6_preparation_path: Path,
    v6_view_path: Path,
    v6_checkpoint_path: Path,
    pair_view_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != V61_EXPERIMENT
        or payload.get("diagnostic_only") is not True
        or payload.get("new_audio_files") != 0
        or payload.get("production_authorization") is not False
    ):
        raise LTSNContractError("invalid V6.1 preparation")
    expected = {
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(target_path),
        "v6_preparation_sha256": sha256_file(v6_preparation_path),
        "v6_view_sha256": sha256_file(v6_view_path),
        "v6_checkpoint_sha256": sha256_file(v6_checkpoint_path),
        "pair_view_sha256": sha256_file(pair_view_path),
        "config_sha256": sha256_file(config_path),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise LTSNContractError(f"stale V6.1 preparation: {name}")
    return payload


def train_v61(
    *,
    fingerprint_path: Path,
    tac_target_path: Path,
    v6_preparation_path: Path,
    v6_view_path: Path,
    v6_checkpoint_path: Path,
    pair_view_path: Path,
    preparation_path: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    paths = [
        fingerprint_path,
        tac_target_path,
        v6_preparation_path,
        v6_view_path,
        v6_checkpoint_path,
        pair_view_path,
        preparation_path,
        config_path,
    ]
    (
        fingerprint_path,
        tac_target_path,
        v6_preparation_path,
        v6_view_path,
        v6_checkpoint_path,
        pair_view_path,
        preparation_path,
        config_path,
    ) = [path.resolve() for path in paths]
    output_dir = output_dir.resolve()
    preparation = _validate_preparation(
        preparation_path,
        fingerprint_path=fingerprint_path,
        target_path=tac_target_path,
        v6_preparation_path=v6_preparation_path,
        v6_view_path=v6_view_path,
        v6_checkpoint_path=v6_checkpoint_path,
        pair_view_path=pair_view_path,
        config_path=config_path,
    )
    config, _ = load_v61_config(config_path)
    device = _device(device_name)
    _seed(config.seed)
    contract = load_fingerprint_contract(fingerprint_path)
    model, v6_checkpoint = load_v6_model(v6_checkpoint_path, contract, device)
    for name, expected in {
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "view_sha256": sha256_file(v6_view_path),
        "preparation_sha256": sha256_file(v6_preparation_path),
    }.items():
        if v6_checkpoint.get(name) != expected:
            raise LTSNContractError(f"V6.1 source checkpoint {name} mismatch")
    model.train()
    v6_preparation = json.loads(v6_preparation_path.read_text(encoding="utf-8"))
    target_center, target_scale = _normalization(v6_preparation)
    target_center_tensor = torch.tensor(target_center, device=device)
    target_scale_tensor = torch.tensor(target_scale, device=device)
    view_records = read_v6_view(v6_view_path)
    pair_records = read_v61_pairs(pair_view_path)
    generator = torch.Generator().manual_seed(config.seed)
    final_train_loader = DataLoader(
        V6FinalTargetDataset(view_records, "train"),
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=collate_v6,
        pin_memory=device.type == "cuda",
    )
    final_development_loader = DataLoader(
        V6FinalTargetDataset(view_records, "development"),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_v6,
        pin_memory=device.type == "cuda",
    )
    pair_fit_loader = DataLoader(
        V61PairDataset(pair_records, "direction_fit"),
        batch_size=config.direction_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 1),
        num_workers=config.num_workers,
        collate_fn=collate_v61_pairs,
        pin_memory=device.type == "cuda",
    )
    pair_validation_loader = DataLoader(
        V61PairDataset(pair_records, "direction_validation"),
        batch_size=config.direction_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_v61_pairs,
        pin_memory=device.type == "cuda",
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_updates = len(final_train_loader) * config.max_epochs
    warmup_updates = max(1, round(total_updates * config.warmup_fraction))
    minimum_ratio = config.minimum_learning_rate / config.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: _schedule(update, warmup_updates, total_updates, minimum_ratio),
    )
    best_objective = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history = []
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        pair_iterator = iter(pair_fit_loader)
        epoch_losses = []
        for final_raw in final_train_loader:
            try:
                pair_raw = next(pair_iterator)
            except StopIteration:
                pair_iterator = iter(pair_fit_loader)
                pair_raw = next(pair_iterator)
            final_batch = _to_device(final_raw, device)
            pair_batch = _to_device(pair_raw, device)
            final_prediction = model(
                final_batch["latent"],
                final_batch["timestep"],
                final_batch["step_number"],
                final_batch["attention_mask"],
            )
            final_target = (
                final_batch["targets"].float() - target_center_tensor
            ) / target_scale_tensor
            final_loss = _weighted_huber(final_prediction, final_target, config.huber_delta)
            if not torch.isfinite(final_loss):
                raise RuntimeError("non-finite V6.1 final-target loss")
            optimizer.zero_grad(set_to_none=True)
            (config.final_target_loss_weight * final_loss).backward()
            derivative = _predicted_derivative(
                model, pair_batch, target_center_tensor, target_scale_tensor
            )
            direction_loss, direction_huber, direction_sign = _direction_loss(
                derivative,
                pair_batch["exact_derivative"],
                pair_batch["step_number"],
                preparation,
                config,
            )
            if not torch.isfinite(direction_loss):
                raise RuntimeError("non-finite V6.1 direction loss")
            direction_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(
                (
                    float(final_loss.detach().cpu()),
                    float(direction_huber.detach().cpu()),
                    float(direction_sign.detach().cpu()),
                )
            )
        development_rows = _predict_view(
            model, final_development_loader, device, target_center, target_scale
        )
        development_metrics = _view_metrics(development_rows)
        validation_rows = _pair_predictions(
            model, pair_validation_loader, device, target_center, target_scale
        )
        validation_metrics = _pair_metrics(validation_rows)
        validation_huber = _pair_huber(validation_rows, preparation, config.huber_delta)
        objective = (
            config.selection_final_weight * development_metrics["total_distance_mae"]
            + config.selection_direction_huber_weight * validation_huber
            + config.selection_direction_agreement_weight
            * (1.0 - validation_metrics["direction_agreement"])
        )
        loss_array = np.asarray(epoch_losses, dtype=np.float64)
        history.append(
            {
                "epoch": epoch,
                "train_final_huber": float(np.mean(loss_array[:, 0])),
                "train_direction_huber": float(np.mean(loss_array[:, 1])),
                "train_direction_sign_loss": float(np.mean(loss_array[:, 2])),
                "development_total_distance_mae": development_metrics[
                    "total_distance_mae"
                ],
                "development_total_distance_spearman": development_metrics[
                    "total_distance_spearman"
                ],
                "validation_direction_huber": validation_huber,
                "validation_direction_agreement": validation_metrics[
                    "direction_agreement"
                ],
                "validation_derivative_spearman": validation_metrics[
                    "derivative_spearman"
                ],
                "selection_objective": objective,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if objective < best_objective - 1e-8:
            best_objective = objective
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= config.minimum_epochs and stale >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("V6.1 training produced no checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "v61_final_target.pt"
    summary_path = output_dir / "training_summary.json"
    if checkpoint_path.exists() or summary_path.exists():
        raise LTSNContractError("V6.1 output already exists; use a new model directory")
    temporary = checkpoint_path.with_suffix(".part.pt")
    torch.save(
        {
            "schema_version": 1,
            "experiment": V61_EXPERIMENT,
            "diagnostic_only": True,
            "scientific_evidence": False,
            "qualification_eligible": False,
            "guidance_promotion_eligible": False,
            "production_authorization": False,
            "new_audio_files": 0,
            "precision": "fp32",
            "state_dict": best_state,
            "model_config": v6_checkpoint["model_config"],
            "head_config": v6_checkpoint["head_config"],
            "training_config": asdict(config),
            "target_center": target_center.tolist(),
            "target_scale": target_scale.tolist(),
            "fingerprint_sha256": sha256_file(fingerprint_path),
            "tac_target_sha256": sha256_file(tac_target_path),
            "v6_checkpoint_sha256": sha256_file(v6_checkpoint_path),
            "v6_preparation_sha256": sha256_file(v6_preparation_path),
            "v6_view_sha256": sha256_file(v6_view_path),
            "pair_view_sha256": sha256_file(pair_view_path),
            "preparation_sha256": sha256_file(preparation_path),
            "config_sha256": sha256_file(config_path),
            "seed": config.seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_selection_objective": best_objective,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    payload = {
        "schema_version": 1,
        "experiment": V61_EXPERIMENT,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "new_audio_files": 0,
        "precision": "fp32",
        "seed": config.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_selection_objective": best_objective,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preparation_sha256": sha256_file(preparation_path),
        "config_sha256": sha256_file(config_path),
    }
    write_json_atomic(summary_path, payload)
    payload["training_summary_sha256"] = sha256_file(summary_path)
    return payload


def _load_v61_model(
    checkpoint_path: Path, fingerprint_path: Path, device: torch.device
) -> tuple[V6FinalTargetModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("experiment") != V61_EXPERIMENT
        or payload.get("diagnostic_only") is not True
        or payload.get("production_authorization") is not False
        or payload.get("new_audio_files") != 0
    ):
        raise LTSNContractError("invalid V6.1 checkpoint")
    contract = load_fingerprint_contract(fingerprint_path)
    model_raw = dict(payload["model_config"])
    model_raw["inactive_coordinate_indices"] = tuple(
        model_raw.get("inactive_coordinate_indices", ())
    )
    model = V6FinalTargetModel(
        contract,
        LTSNConfig(**model_raw),
        V6HeadConfig(**payload["head_config"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def report_v61(
    *,
    fingerprint_path: Path,
    tac_target_path: Path,
    v6_preparation_path: Path,
    v6_view_path: Path,
    v6_checkpoint_path: Path,
    pair_view_path: Path,
    preparation_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    fingerprint_path = fingerprint_path.resolve()
    tac_target_path = tac_target_path.resolve()
    v6_preparation_path = v6_preparation_path.resolve()
    v6_view_path = v6_view_path.resolve()
    v6_checkpoint_path = v6_checkpoint_path.resolve()
    pair_view_path = pair_view_path.resolve()
    preparation_path = preparation_path.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    preparation = _validate_preparation(
        preparation_path,
        fingerprint_path=fingerprint_path,
        target_path=tac_target_path,
        v6_preparation_path=v6_preparation_path,
        v6_view_path=v6_view_path,
        v6_checkpoint_path=v6_checkpoint_path,
        pair_view_path=pair_view_path,
        config_path=config_path,
    )
    config, screen = load_v61_config(config_path)
    device = _device(device_name)
    model, checkpoint = _load_v61_model(checkpoint_path, fingerprint_path, device)
    expected = {
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "v6_checkpoint_sha256": sha256_file(v6_checkpoint_path),
        "v6_preparation_sha256": sha256_file(v6_preparation_path),
        "v6_view_sha256": sha256_file(v6_view_path),
        "pair_view_sha256": sha256_file(pair_view_path),
        "preparation_sha256": sha256_file(preparation_path),
        "config_sha256": sha256_file(config_path),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise LTSNContractError(f"V6.1 checkpoint {name} mismatch")
    target_center = np.asarray(checkpoint["target_center"], dtype=np.float32)
    target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    v6_preparation = json.loads(v6_preparation_path.read_text(encoding="utf-8"))
    prepared_center, prepared_scale = _normalization(v6_preparation)
    if not np.array_equal(target_center, prepared_center) or not np.array_equal(
        target_scale, prepared_scale
    ):
        raise LTSNContractError("V6.1 checkpoint final-target transform mismatch")
    view_records = read_v6_view(v6_view_path)
    pair_records = read_v61_pairs(pair_view_path)
    final_metrics = {}
    for split in ("train", "development"):
        loader = DataLoader(
            V6FinalTargetDataset(view_records, split),
            batch_size=config.micro_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_v6,
            pin_memory=device.type == "cuda",
        )
        final_metrics[split] = _view_metrics(
            _predict_view(model, loader, device, target_center, target_scale)
        )
    pair_outcomes = []
    pair_metrics = {}
    for split in V61_PAIR_SPLITS:
        loader = DataLoader(
            V61PairDataset(pair_records, split),
            batch_size=config.direction_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_v61_pairs,
            pin_memory=device.type == "cuda",
        )
        rows = _pair_predictions(model, loader, device, target_center, target_scale)
        for row in rows:
            row["v61_split"] = split
        pair_outcomes.extend(rows)
        pair_metrics[split] = _pair_metrics(rows)
    heldout_rows = [
        row
        for row in pair_outcomes
        if row["v61_split"] in {"heldout_seen_anchor", "heldout_unseen_anchor"}
    ]
    heldout = _pair_metrics(heldout_rows)
    heldout_by_step = {
        str(step): _pair_metrics(
            [row for row in heldout_rows if int(row["step_number"]) == step]
        )
        for step in (4, 5, 6)
    }
    criteria = {
        "no_new_audio": True,
        "minimum_train_final_distance_spearman": (
            final_metrics["train"]["total_distance_spearman"]
            >= screen.minimum_train_final_distance_spearman
        ),
        "minimum_development_final_distance_spearman": (
            final_metrics["development"]["total_distance_spearman"]
            >= screen.minimum_development_final_distance_spearman
        ),
        "minimum_heldout_direction_pairs": (
            heldout["pairs"] >= screen.minimum_heldout_direction_pairs
        ),
        "minimum_heldout_direction_agreement": (
            heldout["direction_agreement"] >= screen.minimum_heldout_direction_agreement
        ),
        "minimum_heldout_derivative_spearman": (
            heldout["derivative_spearman"] > screen.minimum_heldout_derivative_spearman
        ),
    }
    outcomes_path = output_path.with_name("v61_pair_outcomes.csv")
    write_csv_atomic(outcomes_path, pair_outcomes)
    supported = all(criteria.values())
    payload = {
        "schema_version": 1,
        "experiment": V61_EXPERIMENT,
        "mode": "diagnostic_only",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "new_audio_files": 0,
        "initialization": "v6_final_target_checkpoint",
        "direction_training_target": "exact_tac_derivative",
        "direction_fit_pairs": preparation["pair_counts"]["direction_fit"],
        "direction_validation_pairs": preparation["pair_counts"]["direction_validation"],
        "heldout_direction_pairs": heldout["pairs"],
        "train_final_distance_spearman": final_metrics["train"][
            "total_distance_spearman"
        ],
        "development_final_distance_spearman": final_metrics["development"][
            "total_distance_spearman"
        ],
        "heldout_direction_agreement": heldout["direction_agreement"],
        "heldout_derivative_spearman": heldout["derivative_spearman"],
        "final_target_fit": final_metrics,
        "direction_fit_and_validation": pair_metrics,
        "heldout_direction_by_step": heldout_by_step,
        "heldout_anchor_cluster_bootstrap": _cluster_bootstrap(heldout_rows),
        "thresholds": asdict(screen),
        "criteria": criteria,
        "final_target_signal_supported": supported,
        "status": "signal_supported" if supported else "signal_not_supported",
        "interpretation": (
            "V6.1 adds exact TAC finite-difference supervision from fit-only anchors while "
            "keeping the original seen/unseen held-out pairs untouched; the result remains "
            "diagnostic-only and cannot establish audio quality or final causal improvement"
        ),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "v6_checkpoint_sha256": sha256_file(v6_checkpoint_path),
        "v6_preparation_sha256": sha256_file(v6_preparation_path),
        "v6_view_sha256": sha256_file(v6_view_path),
        "pair_view_sha256": sha256_file(pair_view_path),
        "preparation_sha256": sha256_file(preparation_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "pair_outcomes": str(outcomes_path),
        "pair_outcomes_sha256": sha256_file(outcomes_path),
    }
    write_json_atomic(output_path, payload)
    payload["report_sha256"] = sha256_file(output_path)
    return payload
