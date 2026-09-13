"""Diagnostic-only V6 final-target screen using existing LTSN artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .ltsn_contract import (
    FingerprintContract,
    LTSNContractError,
    load_fingerprint_contract,
    sha256_file,
)
from .ltsn_dataset import read_ltsn_manifest
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .ltsn_v52b_training import (
    V52BPair,
    V52BPairDataset,
    _spearman,
    collate_v52b_pairs,
    read_v52b_pairs,
)
from .path_homology_surrogate import LTSNConfig, PathHomologySurrogate
from .tac_target import TACTopologyTarget, robust_center_scale

V6_EXPERIMENT = "ltsn_v6_final_target_screen"
V6_BLOCKS = ("pitch", "path_acoustic_phase", "path_chroma_phase")
V6_WEIGHTS = (0.5, 0.25, 0.25)
V6_HELDOUT_SPLITS = ("seen_anchor_heldout_direction", "unseen_anchor")


@dataclass(frozen=True, slots=True)
class V6HeadConfig:
    hidden_dim: int = 256


@dataclass(frozen=True, slots=True)
class V6TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_fraction: float = 0.05
    minimum_learning_rate: float = 1e-6
    effective_batch_size: int = 32
    micro_batch_size: int = 8
    max_epochs: int = 80
    minimum_epochs: int = 20
    early_stopping_patience: int = 12
    gradient_clip_norm: float = 5.0
    huber_delta: float = 1.0
    num_workers: int = 4
    seed: int = 20260716

    def validate(self) -> None:
        if self.micro_batch_size < 1 or self.effective_batch_size < self.micro_batch_size:
            raise LTSNContractError("V6 effective batch must be at least the micro batch")
        if self.effective_batch_size % self.micro_batch_size:
            raise LTSNContractError("V6 effective batch must divide into micro batches")
        if not 1 <= self.minimum_epochs <= self.max_epochs:
            raise LTSNContractError("invalid V6 epoch limits")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise LTSNContractError("V6 warmup fraction must lie in [0,1)")
        if self.learning_rate <= 0.0 or self.minimum_learning_rate <= 0.0:
            raise LTSNContractError("V6 learning rates must be positive")
        if self.gradient_clip_norm <= 0.0 or self.huber_delta <= 0.0:
            raise LTSNContractError("V6 gradient clipping and Huber delta must be positive")


@dataclass(frozen=True, slots=True)
class V6ScreenConfig:
    minimum_train_final_distance_spearman: float = 0.70
    minimum_development_final_distance_spearman: float = 0.50
    minimum_heldout_direction_pairs: int = 128
    minimum_heldout_direction_agreement: float = 0.60
    minimum_heldout_derivative_spearman: float = 0.15

    def validate(self) -> None:
        correlations = (
            self.minimum_train_final_distance_spearman,
            self.minimum_development_final_distance_spearman,
            self.minimum_heldout_direction_agreement,
            self.minimum_heldout_derivative_spearman,
        )
        if any(not -1.0 <= value <= 1.0 for value in correlations):
            raise LTSNContractError("V6 screen correlations must lie in [-1,1]")
        if self.minimum_heldout_direction_pairs < 1:
            raise LTSNContractError("V6 held-out pair minimum must be positive")


@dataclass(frozen=True, slots=True)
class V6ViewRecord:
    sample_id: str
    prompt_id: str
    trajectory_id: str
    split: str
    step_number: int
    timestep: float
    latent_path: Path
    latent_sha256: str
    final_sample_id: str
    targets: tuple[float, float, float]
    total_distance: float


def _dataclass_values(cls: type[Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise LTSNContractError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return dict(raw)


def load_v6_config(
    path: Path,
) -> tuple[LTSNConfig, V6HeadConfig, V6TrainingConfig, V6ScreenConfig]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model_raw = _dataclass_values(LTSNConfig, payload.get("model", {}))
    if "inactive_coordinate_indices" in model_raw:
        model_raw["inactive_coordinate_indices"] = tuple(model_raw["inactive_coordinate_indices"])
    head = V6HeadConfig(**_dataclass_values(V6HeadConfig, payload.get("head", {})))
    training = V6TrainingConfig(
        **_dataclass_values(V6TrainingConfig, payload.get("training", {}))
    )
    screen = V6ScreenConfig(**_dataclass_values(V6ScreenConfig, payload.get("screen", {})))
    training.validate()
    screen.validate()
    if head.hidden_dim < 32:
        raise LTSNContractError("V6 head hidden dimension is too small")
    return LTSNConfig(**model_raw), head, training, screen


def _validate_target_binding(target: TACTopologyTarget, contract: FingerprintContract) -> None:
    expected = target.payload.get("source_sha256", {}).get(
        "metadata/focus_path_homology_fingerprint_v2.json"
    )
    if expected != contract.artifact_sha256:
        raise LTSNContractError("V6 TAC target uses a different frozen fingerprint")
    if tuple(target.payload["distance_definition"]["weights"]) != V6_WEIGHTS:
        raise LTSNContractError("V6 TAC block weights changed")


def _target_values(
    target: TACTopologyTarget, coordinates: Sequence[float]
) -> tuple[tuple[float, float, float], float]:
    blocks = target.block_distances(coordinates)
    values = tuple(float(blocks[name][0]) for name in V6_BLOCKS)
    total = float(sum(weight * value for weight, value in zip(V6_WEIGHTS, values, strict=True)))
    if not all(math.isfinite(value) and value >= 0.0 for value in (*values, total)):
        raise LTSNContractError("V6 final-target distance is not finite and non-negative")
    return values, total


def prepare_v6_final_target_view(
    *,
    root: Path,
    fingerprint_path: Path,
    tac_target_path: Path,
    source_manifest_path: Path,
    pair_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a reference-only final-target view without copying latent or audio files."""

    root = root.resolve()
    fingerprint_path = fingerprint_path.resolve()
    tac_target_path = tac_target_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    pair_manifest_path = pair_manifest_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    contract = load_fingerprint_contract(fingerprint_path)
    target = TACTopologyTarget.from_json(tac_target_path, root=root)
    _validate_target_binding(target, contract)
    _, _, _, screen = load_v6_config(config_path)
    records = read_ltsn_manifest(source_manifest_path, contract)
    pairs = read_v52b_pairs(pair_manifest_path)
    heldout_pairs = [row for row in pairs if row.evaluation_split in V6_HELDOUT_SPLITS]
    if len(heldout_pairs) < screen.minimum_heldout_direction_pairs:
        raise LTSNContractError(
            "V6 held-out operational pair count is below the frozen screen minimum"
        )

    by_trajectory: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_trajectory[record.trajectory_id].append(record)
    rows: list[dict[str, Any]] = []
    unique_train_targets: list[tuple[float, float, float]] = []
    split_trajectories: dict[str, int] = defaultdict(int)
    for trajectory_id, members in sorted(by_trajectory.items()):
        finals = [row for row in members if row.is_final]
        inputs = [row for row in members if not row.is_final]
        if len(finals) != 1:
            raise LTSNContractError(f"V6 trajectory requires one final snapshot: {trajectory_id}")
        if len(inputs) != 3 or {row.step_number for row in inputs} != {4, 5, 6}:
            raise LTSNContractError(
                f"V6 trajectory requires exactly one input at steps 4/5/6: {trajectory_id}"
            )
        final = finals[0]
        if final.step_number != 8 or any(
            (row.prompt_id, row.split) != (final.prompt_id, final.split) for row in inputs
        ):
            raise LTSNContractError("V6 final target differs in prompt, split, or final step")
        values, total = _target_values(target, final.coordinates)
        split_trajectories[final.split] += 1
        if final.split == "train":
            unique_train_targets.append(values)
        for row in sorted(inputs, key=lambda item: item.step_number):
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "prompt_id": row.prompt_id,
                    "trajectory_id": trajectory_id,
                    "split": row.split,
                    "step_number": row.step_number,
                    "timestep": format(row.timestep, ".17g"),
                    "latent_path": os.path.relpath(row.latent_path, output_dir),
                    "latent_sha256": row.latent_sha256,
                    "final_sample_id": final.sample_id,
                    "target_pitch_distance": format(values[0], ".17g"),
                    "target_acoustic_phase_distance": format(values[1], ".17g"),
                    "target_chroma_phase_distance": format(values[2], ".17g"),
                    "target_total_distance": format(total, ".17g"),
                }
            )
    if len(unique_train_targets) < 2 or split_trajectories.get("development", 0) < 1:
        raise LTSNContractError("V6 requires at least two train and one development trajectories")
    center, scale = robust_center_scale(unique_train_targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    view_path = output_dir / "v6_final_target_view.csv"
    write_csv_atomic(view_path, rows)
    exact_hashes = {row.exact_label_table_sha256 for row in records}
    if len(exact_hashes) != 1:
        raise LTSNContractError("V6 source manifest mixes exact label tables")
    payload = {
        "schema_version": 1,
        "experiment": V6_EXPERIMENT,
        "mode": "diagnostic_only",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "new_audio_files": 0,
        "copied_latent_files": 0,
        "reused_latents": len(rows),
        "trajectories": len(by_trajectory),
        "split_trajectories": dict(sorted(split_trajectories.items())),
        "input_steps": [4, 5, 6],
        "final_target_step": 8,
        "target_blocks": list(V6_BLOCKS),
        "target_weights": list(V6_WEIGHTS),
        "target_transform": {
            "kind": "train_split_robust_median_scale",
            "center": center.tolist(),
            "scale": scale.tolist(),
            "unique_train_final_targets": len(unique_train_targets),
        },
        "heldout_operational_pairs": len(heldout_pairs),
        "heldout_pair_splits": list(V6_HELDOUT_SPLITS),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_exact_label_table_sha256": next(iter(exact_hashes)),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "config_sha256": sha256_file(config_path),
        "view_path": str(view_path),
        "view_sha256": sha256_file(view_path),
    }
    preparation_path = output_dir / "v6_preparation.json"
    write_json_atomic(preparation_path, payload)
    payload["preparation_sha256"] = sha256_file(preparation_path)
    return payload


def read_v6_view(path: Path) -> list[V6ViewRecord]:
    rows: list[V6ViewRecord] = []
    checked: dict[Path, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            latent_path = (path.parent / raw["latent_path"]).resolve()
            if not latent_path.is_file():
                raise LTSNContractError(f"V6 latent is missing: {latent_path}")
            actual = checked.setdefault(latent_path, sha256_file(latent_path))
            if actual != raw["latent_sha256"]:
                raise LTSNContractError(f"V6 latent hash mismatch: {latent_path}")
            targets = (
                float(raw["target_pitch_distance"]),
                float(raw["target_acoustic_phase_distance"]),
                float(raw["target_chroma_phase_distance"]),
            )
            total = float(raw["target_total_distance"])
            if not all(math.isfinite(value) and value >= 0.0 for value in (*targets, total)):
                raise LTSNContractError("V6 view contains an invalid final target")
            expected_total = sum(
                weight * value for weight, value in zip(V6_WEIGHTS, targets, strict=True)
            )
            if not math.isclose(total, expected_total, rel_tol=1e-10, abs_tol=1e-12):
                raise LTSNContractError("V6 total distance differs from its three blocks")
            rows.append(
                V6ViewRecord(
                    sample_id=raw["sample_id"],
                    prompt_id=raw["prompt_id"],
                    trajectory_id=raw["trajectory_id"],
                    split=raw["split"],
                    step_number=int(raw["step_number"]),
                    timestep=float(raw["timestep"]),
                    latent_path=latent_path,
                    latent_sha256=raw["latent_sha256"],
                    final_sample_id=raw["final_sample_id"],
                    targets=targets,
                    total_distance=total,
                )
            )
    if not rows or len({row.sample_id for row in rows}) != len(rows):
        raise LTSNContractError("V6 view is empty or contains duplicate sample IDs")
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt_splits[row.prompt_id].add(row.split)
        trajectory_splits[row.trajectory_id].add(row.split)
    if any(len(value) != 1 for value in (*prompt_splits.values(), *trajectory_splits.values())):
        raise LTSNContractError("V6 view leaks prompts or trajectories across splits")
    return rows


class V6FinalTargetDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[V6ViewRecord], split: str) -> None:
        self.records = tuple(row for row in records if row.split == split)
        if not self.records:
            raise ValueError(f"no V6 records for split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        latent = np.load(row.latent_path, allow_pickle=False)
        if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
            raise LTSNContractError(f"invalid V6 latent: {row.latent_path}")
        return {
            "sample_id": row.sample_id,
            "trajectory_id": row.trajectory_id,
            "latent": torch.from_numpy(np.asarray(latent, dtype=np.float32)),
            "timestep": torch.tensor(row.timestep, dtype=torch.float32),
            "step_number": torch.tensor(row.step_number, dtype=torch.long),
            "targets": torch.tensor(row.targets, dtype=torch.float32),
        }


def collate_v6(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty V6 batch")
    maximum = max(int(item["latent"].shape[0]) for item in items)
    latent = torch.zeros(len(items), maximum, 64, dtype=torch.float32)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["latent"].shape[0])
        latent[index, :length] = item["latent"]
        mask[index, :length] = True
    return {
        "sample_id": [str(item["sample_id"]) for item in items],
        "trajectory_id": [str(item["trajectory_id"]) for item in items],
        "latent": latent,
        "attention_mask": mask,
        "timestep": torch.stack([item["timestep"] for item in items]),
        "step_number": torch.stack([item["step_number"] for item in items]),
        "targets": torch.stack([item["targets"] for item in items]),
    }


class V6FinalTargetModel(nn.Module):
    """Existing LTSN encoder plus a three-scalar final-target distance head."""

    def __init__(
        self,
        contract: FingerprintContract,
        model_config: LTSNConfig,
        head_config: V6HeadConfig,
    ) -> None:
        super().__init__()
        self.encoder = PathHomologySurrogate(contract, model_config)
        for module in (
            self.encoder.coordinate_mean_head,
            self.encoder.coordinate_logvar_head,
            self.encoder.ood_head,
        ):
            module.requires_grad_(False)
        self.final_target_head = nn.Sequential(
            nn.Linear(256, head_config.hidden_dim),
            nn.SiLU(),
            nn.Linear(head_config.hidden_dim, 3),
        )

    def forward(
        self,
        latent: Tensor,
        timestep: Tensor,
        step_number: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        shared = self.encoder.encode(latent, timestep, step_number, attention_mask)
        return self.final_target_head(shared.float())


def _seed_everything(seed: int) -> None:
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
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"explicit CUDA device is unavailable: {name}")
        torch.cuda.set_device(device)
    return device


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _schedule(update: int, warmup: int, total: int, minimum: float) -> float:
    if update < warmup:
        return max(minimum, (update + 1) / max(1, warmup))
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))


def _normalization(preparation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    transform = preparation.get("target_transform", {})
    center = np.asarray(transform.get("center"), dtype=np.float32)
    scale = np.asarray(transform.get("scale"), dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0.0):
        raise LTSNContractError("invalid V6 target transform")
    return center, scale


def _weighted_huber(prediction: Tensor, target: Tensor, delta: float) -> Tensor:
    per_block = F.huber_loss(prediction.float(), target.float(), delta=delta, reduction="none")
    weights = prediction.new_tensor(V6_WEIGHTS, dtype=torch.float32)
    return torch.mean(torch.sum(per_block * weights, dim=-1))


def _predict_view(
    model: V6FinalTargetModel,
    loader: DataLoader[Any],
    device: torch.device,
    center: np.ndarray,
    scale: np.ndarray,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    center_tensor = torch.tensor(center, device=device)
    scale_tensor = torch.tensor(scale, device=device)
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            normalized = model(
                batch["latent"],
                batch["timestep"],
                batch["step_number"],
                batch["attention_mask"],
            )
            prediction = normalized * scale_tensor + center_tensor
            for index, sample_id in enumerate(raw["sample_id"]):
                predicted = prediction[index].detach().cpu().numpy().astype(float)
                target = raw["targets"][index].numpy().astype(float)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "trajectory_id": raw["trajectory_id"][index],
                        "predicted_blocks": predicted.tolist(),
                        "target_blocks": target.tolist(),
                        "predicted_total": float(np.dot(predicted, V6_WEIGHTS)),
                        "target_total": float(np.dot(target, V6_WEIGHTS)),
                    }
                )
    return rows


def _view_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(row)
    aggregated = []
    for trajectory_id, members in grouped.items():
        target_blocks = np.asarray([row["target_blocks"] for row in members], dtype=float)
        if not np.allclose(target_blocks, target_blocks[0], rtol=0.0, atol=1e-7):
            raise LTSNContractError("V6 trajectory has inconsistent final targets across steps")
        predicted_blocks = np.mean(
            np.asarray([row["predicted_blocks"] for row in members], dtype=float), axis=0
        )
        aggregated.append(
            {
                "trajectory_id": trajectory_id,
                "target_blocks": target_blocks[0],
                "predicted_blocks": predicted_blocks,
                "target_total": float(np.dot(target_blocks[0], V6_WEIGHTS)),
                "predicted_total": float(np.dot(predicted_blocks, V6_WEIGHTS)),
            }
        )
    predicted = [float(row["predicted_total"]) for row in aggregated]
    target = [float(row["target_total"]) for row in aggregated]
    return {
        "samples": len(rows),
        "trajectories": len(aggregated),
        "aggregation": "mean_prediction_across_steps_4_5_6_per_trajectory",
        "total_distance_spearman": _spearman(target, predicted),
        "total_distance_mae": float(np.mean(np.abs(np.asarray(target) - predicted))),
        "block_spearman": {
            name: _spearman(
                [float(row["target_blocks"][index]) for row in aggregated],
                [float(row["predicted_blocks"][index]) for row in aggregated],
            )
            for index, name in enumerate(V6_BLOCKS)
        },
    }


def _validate_preparation(
    preparation_path: Path,
    *,
    fingerprint_path: Path,
    target_path: Path,
    view_path: Path,
    pair_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    payload = json.loads(preparation_path.read_text(encoding="utf-8"))
    required_false = (
        "scientific_evidence",
        "qualification_eligible",
        "guidance_promotion_eligible",
        "production_authorization",
    )
    if (
        payload.get("experiment") != V6_EXPERIMENT
        or payload.get("diagnostic_only") is not True
        or any(payload.get(key) is not False for key in required_false)
        or payload.get("new_audio_files") != 0
        or payload.get("copied_latent_files") != 0
    ):
        raise LTSNContractError("invalid or authorizing V6 preparation")
    expected = {
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(target_path),
        "view_sha256": sha256_file(view_path),
        "pair_manifest_sha256": sha256_file(pair_path),
        "config_sha256": sha256_file(config_path),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise LTSNContractError(f"stale V6 preparation: {name}")
    return payload


def train_v6_final_target(
    *,
    fingerprint_path: Path,
    tac_target_path: Path,
    view_path: Path,
    pair_manifest_path: Path,
    preparation_path: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Train one FP32 diagnostic model on final-output TAC block distances."""

    paths = [
        fingerprint_path,
        tac_target_path,
        view_path,
        pair_manifest_path,
        preparation_path,
        config_path,
    ]
    (
        fingerprint_path,
        tac_target_path,
        view_path,
        pair_manifest_path,
        preparation_path,
        config_path,
    ) = [path.resolve() for path in paths]
    output_dir = output_dir.resolve()
    preparation = _validate_preparation(
        preparation_path,
        fingerprint_path=fingerprint_path,
        target_path=tac_target_path,
        view_path=view_path,
        pair_path=pair_manifest_path,
        config_path=config_path,
    )
    contract = load_fingerprint_contract(fingerprint_path)
    model_config, head_config, training, _ = load_v6_config(config_path)
    records = read_v6_view(view_path)
    center, scale = _normalization(preparation)
    device = _device(device_name)
    _seed_everything(training.seed)
    loaders: dict[str, DataLoader[Any]] = {}
    for split in ("train", "development"):
        dataset = V6FinalTargetDataset(records, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=training.micro_batch_size,
            shuffle=split == "train",
            generator=(torch.Generator().manual_seed(training.seed) if split == "train" else None),
            num_workers=training.num_workers,
            collate_fn=collate_v6,
            pin_memory=device.type == "cuda",
        )
    model = V6FinalTargetModel(contract, model_config, head_config).to(device).float()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable, lr=training.learning_rate, weight_decay=training.weight_decay
    )
    accumulation = training.effective_batch_size // training.micro_batch_size
    updates_per_epoch = max(1, math.ceil(len(loaders["train"]) / accumulation))
    total_updates = max(1, updates_per_epoch * training.max_epochs)
    warmup_updates = max(1, round(total_updates * training.warmup_fraction))
    minimum_ratio = training.minimum_learning_rate / training.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: _schedule(update, warmup_updates, total_updates, minimum_ratio),
    )
    center_tensor = torch.tensor(center, device=device)
    scale_tensor = torch.tensor(scale, device=device)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history = []
    for epoch in range(1, training.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        for batch_index, raw in enumerate(loaders["train"], start=1):
            batch = _to_device(raw, device)
            normalized_target = (batch["targets"].float() - center_tensor) / scale_tensor
            prediction = model(
                batch["latent"],
                batch["timestep"],
                batch["step_number"],
                batch["attention_mask"],
            )
            loss = _weighted_huber(prediction, normalized_target, training.huber_delta)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V6 training loss")
            (loss / accumulation).backward()
            train_losses.append(float(loss.detach().cpu()))
            if batch_index % accumulation == 0 or batch_index == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(trainable, training.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        development_rows = _predict_view(
            model, loaders["development"], device, center, scale
        )
        development_metrics = _view_metrics(development_rows)
        development_loss = float(development_metrics["total_distance_mae"])
        history.append(
            {
                "epoch": epoch,
                "train_normalized_huber": float(np.mean(train_losses)),
                "development_total_distance_mae": development_loss,
                "development_total_distance_spearman": development_metrics[
                    "total_distance_spearman"
                ],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if development_loss < best_loss - 1e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= training.minimum_epochs and stale >= training.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("V6 training produced no checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "v6_final_target.pt"
    summary_path = output_dir / "training_summary.json"
    if checkpoint_path.exists() or summary_path.exists():
        raise LTSNContractError("V6 model output already exists; use a new output directory")
    temporary = checkpoint_path.with_suffix(".part.pt")
    torch.save(
        {
            "schema_version": 1,
            "experiment": V6_EXPERIMENT,
            "diagnostic_only": True,
            "scientific_evidence": False,
            "qualification_eligible": False,
            "guidance_promotion_eligible": False,
            "production_authorization": False,
            "precision": "fp32",
            "new_audio_files": 0,
            "state_dict": best_state,
            "model_config": asdict(model_config),
            "head_config": asdict(head_config),
            "training_config": asdict(training),
            "target_center": center.tolist(),
            "target_scale": scale.tolist(),
            "fingerprint_sha256": sha256_file(fingerprint_path),
            "tac_target_sha256": sha256_file(tac_target_path),
            "view_sha256": sha256_file(view_path),
            "pair_manifest_sha256": sha256_file(pair_manifest_path),
            "preparation_sha256": sha256_file(preparation_path),
            "config_sha256": sha256_file(config_path),
            "seed": training.seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_development_total_distance_mae": best_loss,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    payload = {
        "schema_version": 1,
        "experiment": V6_EXPERIMENT,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "new_audio_files": 0,
        "precision": "fp32",
        "seed": training.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preparation_sha256": sha256_file(preparation_path),
        "config_sha256": sha256_file(config_path),
    }
    write_json_atomic(summary_path, payload)
    payload["training_summary_sha256"] = sha256_file(summary_path)
    return payload


def load_v6_model(
    checkpoint_path: Path,
    contract: FingerprintContract,
    device: torch.device,
) -> tuple[V6FinalTargetModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_false = (
        "scientific_evidence",
        "qualification_eligible",
        "guidance_promotion_eligible",
        "production_authorization",
    )
    if (
        payload.get("experiment") != V6_EXPERIMENT
        or payload.get("diagnostic_only") is not True
        or payload.get("precision") != "fp32"
        or payload.get("new_audio_files") != 0
        or any(payload.get(key) is not False for key in required_false)
    ):
        raise LTSNContractError("checkpoint is not a bounded V6 final-target screen")
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


def _pair_rows(
    model: V6FinalTargetModel,
    records: Sequence[V52BPair],
    device: torch.device,
    center: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> list[dict[str, Any]]:
    selected = [row for row in records if row.evaluation_split in V6_HELDOUT_SPLITS]
    center_tensor = torch.tensor(center, device=device)
    scale_tensor = torch.tensor(scale, device=device)
    rows: list[dict[str, Any]] = []
    model.eval()
    for split in V6_HELDOUT_SPLITS:
        loader = DataLoader(
            V52BPairDataset(selected, split),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_v52b_pairs,
        )
        with torch.no_grad():
            for raw in loader:
                batch = _to_device(raw, device)
                minus = model(
                    batch["minus"],
                    batch["timestep"],
                    batch["step_number"],
                    batch["attention_mask"],
                )
                plus = model(
                    batch["plus"],
                    batch["timestep"],
                    batch["step_number"],
                    batch["attention_mask"],
                )
                minus = minus * scale_tensor + center_tensor
                plus = plus * scale_tensor + center_tensor
                weights = minus.new_tensor(V6_WEIGHTS)
                predicted_derivative = ((minus - plus) @ weights) / (
                    2.0 * batch["rms_ratio"]
                )
                for index, pair_id in enumerate(raw["pair_id"]):
                    exact = float(raw["exact_derivative"][index])
                    predicted = float(predicted_derivative[index].detach().cpu())
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "evaluation_split": split,
                            "step_number": int(raw["step_number"][index]),
                            "rms_ratio": float(raw["rms_ratio"][index]),
                            "exact_derivative": exact,
                            "predicted_derivative": predicted,
                            "direction_correct": int(predicted * exact > 0.0),
                        }
                    )
    return rows


def _flat_direction_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise LTSNContractError("V6 direction metrics require at least one pair")
    return {
        "pairs": len(rows),
        "direction_agreement": sum(int(row["direction_correct"]) for row in rows) / len(rows),
        "derivative_spearman": _spearman(
            [float(row["exact_derivative"]) for row in rows],
            [float(row["predicted_derivative"]) for row in rows],
        ),
    }


def report_v6_final_target(
    *,
    fingerprint_path: Path,
    tac_target_path: Path,
    view_path: Path,
    pair_manifest_path: Path,
    preparation_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate final-target fit and existing held-out antithetic directions."""

    fingerprint_path = fingerprint_path.resolve()
    tac_target_path = tac_target_path.resolve()
    view_path = view_path.resolve()
    pair_manifest_path = pair_manifest_path.resolve()
    preparation_path = preparation_path.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    preparation = _validate_preparation(
        preparation_path,
        fingerprint_path=fingerprint_path,
        target_path=tac_target_path,
        view_path=view_path,
        pair_path=pair_manifest_path,
        config_path=config_path,
    )
    contract = load_fingerprint_contract(fingerprint_path)
    _, _, training, screen = load_v6_config(config_path)
    records = read_v6_view(view_path)
    pairs = read_v52b_pairs(pair_manifest_path)
    device = _device(device_name)
    model, checkpoint = load_v6_model(checkpoint_path, contract, device)
    expected = {
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "view_sha256": sha256_file(view_path),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "preparation_sha256": sha256_file(preparation_path),
        "config_sha256": sha256_file(config_path),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise LTSNContractError(f"V6 checkpoint {name} mismatch")
    center = np.asarray(checkpoint["target_center"], dtype=np.float32)
    scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    prepared_center, prepared_scale = _normalization(preparation)
    if not np.array_equal(center, prepared_center) or not np.array_equal(scale, prepared_scale):
        raise LTSNContractError("V6 checkpoint target transform mismatch")
    view_metrics = {}
    for split in ("train", "development"):
        loader = DataLoader(
            V6FinalTargetDataset(records, split),
            batch_size=training.micro_batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            collate_fn=collate_v6,
            pin_memory=device.type == "cuda",
        )
        view_metrics[split] = _view_metrics(
            _predict_view(model, loader, device, center, scale)
        )
    pair_rows = _pair_rows(
        model, pairs, device, center, scale, training.micro_batch_size
    )
    heldout = _flat_direction_metrics(pair_rows)
    by_partition = {
        split: _flat_direction_metrics(
            [row for row in pair_rows if row["evaluation_split"] == split]
        )
        for split in V6_HELDOUT_SPLITS
    }
    by_step = {
        str(step): _flat_direction_metrics(
            [row for row in pair_rows if int(row["step_number"]) == step]
        )
        for step in (4, 5, 6)
    }
    criteria = {
        "no_new_audio": preparation["new_audio_files"] == 0,
        "minimum_train_final_distance_spearman": (
            view_metrics["train"]["total_distance_spearman"]
            >= screen.minimum_train_final_distance_spearman
        ),
        "minimum_development_final_distance_spearman": (
            view_metrics["development"]["total_distance_spearman"]
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
    outcomes_path = output_path.with_name("v6_final_target_pair_outcomes.csv")
    write_csv_atomic(outcomes_path, pair_rows)
    supported = all(criteria.values())
    payload = {
        "schema_version": 1,
        "experiment": V6_EXPERIMENT,
        "mode": "diagnostic_only",
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "new_audio_files": 0,
        "reused_final_targets": preparation["trajectories"],
        "reused_input_latents": preparation["reused_latents"],
        "reused_antithetic_pairs": heldout["pairs"],
        "train_final_distance_spearman": view_metrics["train"][
            "total_distance_spearman"
        ],
        "development_final_distance_spearman": view_metrics["development"][
            "total_distance_spearman"
        ],
        "heldout_direction_pairs": heldout["pairs"],
        "heldout_direction_agreement": heldout["direction_agreement"],
        "heldout_derivative_spearman": heldout["derivative_spearman"],
        "final_target_fit": view_metrics,
        "heldout_direction_by_partition": by_partition,
        "heldout_direction_by_step": by_step,
        "thresholds": asdict(screen),
        "criteria": criteria,
        "final_target_signal_supported": supported,
        "status": "signal_supported" if supported else "signal_not_supported",
        "interpretation": (
            "this no-new-audio screen tests whether existing intermediate latents predict "
            "same-trajectory final TAC distances and whether that scalar field aligns with "
            "existing held-out exact local directions; it does not establish audio quality, "
            "non-inferiority, qualification, or safe guidance"
        ),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "tac_target_sha256": sha256_file(tac_target_path),
        "view_sha256": sha256_file(view_path),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "preparation_sha256": sha256_file(preparation_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "pair_outcomes": str(outcomes_path),
        "pair_outcomes_sha256": sha256_file(outcomes_path),
    }
    write_json_atomic(output_path, payload)
    payload["report_sha256"] = sha256_file(output_path)
    return payload
