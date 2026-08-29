"""Training and checkpoint issuance for the frozen 18-D surrogate ensemble."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tomllib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from .ltsn_contract import (
    FingerprintContract,
    LTSNContractError,
    load_fingerprint_contract,
    sha256_file,
    validate_checkpoint_metadata,
)
from .ltsn_dataset import (
    LTSNSnapshot,
    LTSNSnapshotDataset,
    collate_ltsn_batch,
    manifest_identity,
    read_ltsn_manifest,
)
from .ltsn_losses import LTSNLossWeights, ltsn_loss, trajectory_delta_loss
from .ltsn_pipeline import (
    model_data_identity,
    require_ltsn_promotion_gate,
    write_json_atomic,
)
from .path_homology_surrogate import LTSNConfig, LTSNOutput, PathHomologySurrogate


@dataclass(frozen=True, slots=True)
class LTSNTrainingConfig:
    """Optimizer/runtime settings frozen into each checkpoint."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.05
    minimum_learning_rate: float = 1e-6
    effective_batch_size: int = 64
    micro_batch_size: int = 8
    max_epochs: int = 100
    minimum_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    seeds: tuple[int, ...] = (20260716, 20260717, 20260718)
    use_bf16: bool = True

    def validate(self, *, engineering_smoke: bool) -> None:
        if self.micro_batch_size < 1 or self.effective_batch_size < self.micro_batch_size:
            raise LTSNContractError("effective batch must be at least the micro batch")
        if self.effective_batch_size % self.micro_batch_size:
            raise LTSNContractError("effective batch must be divisible by micro batch")
        if self.max_epochs < 1 or self.minimum_epochs > self.max_epochs:
            raise LTSNContractError("invalid epoch limits")
        if not engineering_smoke:
            frozen = {
                "learning_rate": (self.learning_rate, 3e-4),
                "weight_decay": (self.weight_decay, 1e-2),
                "warmup_fraction": (self.warmup_fraction, 0.05),
                "effective_batch_size": (self.effective_batch_size, 64),
                "gradient_clip_norm": (self.gradient_clip_norm, 1.0),
            }
            changed = [name for name, (actual, expected) in frozen.items() if actual != expected]
            if changed:
                raise LTSNContractError(f"production training changed frozen defaults: {changed}")
            if len(self.seeds) != 3:
                raise LTSNContractError("production qualification requires exactly three seeds")


def _dataclass_values(cls: type[Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise LTSNContractError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return dict(raw)


def load_training_config(path: Path) -> tuple[LTSNConfig, LTSNTrainingConfig, LTSNLossWeights]:
    """Read architecture, optimizer, and loss sections from TOML."""

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model = LTSNConfig(**_dataclass_values(LTSNConfig, payload.get("model", {})))
    training_raw = _dataclass_values(LTSNTrainingConfig, payload.get("training", {}))
    if "seeds" in training_raw:
        training_raw["seeds"] = tuple(int(value) for value in training_raw["seeds"])
    training = LTSNTrainingConfig(**training_raw)
    losses = LTSNLossWeights(
        **_dataclass_values(LTSNLossWeights, payload.get("loss", {}))
    )
    return model, training, losses


class TrajectoryBatchSampler(Sampler[list[int]]):
    """Keep snapshots from shuffled trajectories together when practical."""

    def __init__(self, records: Sequence[LTSNSnapshot], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.trajectory_id].append(index)
        self.groups = tuple(tuple(values) for _, values in sorted(grouped.items()))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        groups = list(self.groups)
        rng.shuffle(groups)
        batch: list[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            if len(group) > self.batch_size:
                for start in range(0, len(group), self.batch_size):
                    yield list(group[start : start + self.batch_size])
            else:
                batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        return max(1, math.ceil(sum(len(group) for group in self.groups) / self.batch_size))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pair_indices(prompt_ids: Sequence[str], device: torch.device) -> Tensor | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        grouped[prompt_id].append(index)
    pairs = [
        (indices[left], indices[right])
        for indices in grouped.values()
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    ]
    if not pairs:
        return None
    return torch.tensor(pairs, dtype=torch.long, device=device)


def _trajectory_pairs(
    trajectory_ids: Sequence[str], step_numbers: Tensor, device: torch.device
) -> Tensor | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, trajectory_id in enumerate(trajectory_ids):
        grouped[trajectory_id].append(index)
    pairs: list[tuple[int, int]] = []
    steps = step_numbers.detach().cpu().tolist()
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: steps[index])
        pairs.extend(
            (left, right)
            for left, right in zip(ordered, ordered[1:], strict=False)
            if steps[right] > steps[left]
        )
    return None if not pairs else torch.tensor(pairs, dtype=torch.long, device=device)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _forward(model: PathHomologySurrogate, batch: Mapping[str, Any]) -> LTSNOutput:
    return model(
        batch["latent"],
        batch["timestep"],
        batch["step_number"],
        batch["attention_mask"],
    )


def _loss(
    output: LTSNOutput,
    batch: Mapping[str, Any],
    weights: LTSNLossWeights,
    device: torch.device,
) -> dict[str, Tensor]:
    result = ltsn_loss(
        output,
        batch["coordinates"],
        batch["focus_logit"],
        batch["ood_label"],
        pair_indices=_pair_indices(batch["prompt_id"], device),
        weights=weights,
    )
    trajectory_pairs = _trajectory_pairs(
        batch["trajectory_id"], batch["step_number"], device
    )
    if trajectory_pairs is not None:
        left, right = trajectory_pairs[:, 0], trajectory_pairs[:, 1]
        delta = trajectory_delta_loss(
            output.coordinate_mean[left],
            output.coordinate_mean[right],
            batch["coordinates"][left],
            batch["coordinates"][right],
        )
        result["total"] = result["total"] + weights.trajectory_delta * delta
        result["trajectory_delta"] = delta
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Dependency-light Spearman correlation with tie-aware average ranks."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return 0.0
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _schedule_factor(update: int, warmup: int, total: int, minimum: float) -> float:
    if update < warmup:
        return max((update + 1) / warmup, 1e-8)
    progress = (update - warmup) / max(total - warmup, 1)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def predict_dataset(
    model: PathHomologySurrogate,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Collect model outputs and exact targets in manifest order."""

    model.eval()
    collected: dict[str, list[Any]] = defaultdict(list)
    for raw in loader:
        batch = _to_device(raw, device)
        output = _forward(model, batch)
        for name, tensor in (
            ("coordinate_mean", output.coordinate_mean),
            ("coordinate_logvar", output.coordinate_logvar),
            ("ood_logit", output.ood_logit),
            ("predicted_focus_logit", output.focus_logit),
            ("coordinates", batch["coordinates"]),
            ("focus_logit", batch["focus_logit"]),
            ("ood_label", batch["ood_label"]),
            ("step_number", batch["step_number"]),
        ):
            collected[name].append(tensor.detach().float().cpu().numpy())
        for name in ("sample_id", "prompt_id", "trajectory_id"):
            collected[name].extend(raw[name])
    return {
        name: (
            np.concatenate(values, axis=0)
            if values and isinstance(values[0], np.ndarray)
            else values
        )
        for name, values in collected.items()
    }


def _development_objective(prediction: Mapping[str, Any]) -> dict[str, float]:
    score_error = float(
        np.mean(np.abs(prediction["predicted_focus_logit"] - prediction["focus_logit"]))
    )
    coordinate_rhos = [
        spearman_correlation(
            prediction["coordinate_mean"][:, index], prediction["coordinates"][:, index]
        )
        for index in range(18)
    ]
    pitch_rho = spearman_correlation(
        np.linalg.norm(prediction["coordinate_mean"][:, :16], axis=1),
        np.linalg.norm(prediction["coordinates"][:, :16], axis=1),
    )
    phase_rho = spearman_correlation(
        np.linalg.norm(prediction["coordinate_mean"][:, 16:], axis=1),
        np.linalg.norm(prediction["coordinates"][:, 16:], axis=1),
    )
    block_rho = 0.5 * (pitch_rho + phase_rho)
    objective = score_error + (1.0 - float(np.median(coordinate_rhos))) + (1.0 - block_rho)
    return {
        "objective": objective,
        "score_mae": score_error,
        "coordinate_median_spearman": float(np.median(coordinate_rhos)),
        "pitch_block_spearman": pitch_rho,
        "phase_block_spearman": phase_rho,
    }


def _metadata(
    *,
    contract: FingerprintContract,
    config_path: Path,
    manifest_path: Path,
    split_manifest_path: Path,
    records: Sequence[LTSNSnapshot],
    ace_model_sha256: str,
    vae_sha256: str,
    model_family: str,
    qualification_eligible: bool,
    reranking_gate_sha256: str,
) -> dict[str, Any]:
    identities = manifest_identity(manifest_path, records)
    return {
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dimensions": 18,
        "feature_order": list(contract.feature_order),
        "distance_weights": list(contract.distance_weights),
        "classifier_sha256": contract.classifier_sha256,
        "ltsn_config_sha256": sha256_file(config_path),
        **identities,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "model_family": model_family,
        "qualification_eligible": qualification_eligible,
        "reranking_gate_sha256": reranking_gate_sha256,
    }


def train_ensemble(
    *,
    fingerprint_path: Path,
    manifest_path: Path,
    split_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    reranking_gate_path: Path | None,
    engineering_smoke: bool,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Train three model-specific seeds and issue hash-bound checkpoints."""

    contract = load_fingerprint_contract(fingerprint_path)
    gate = require_ltsn_promotion_gate(
        reranking_gate_path, contract, engineering_smoke=engineering_smoke
    )
    model_config, training, loss_weights = load_training_config(config_path)
    training.validate(engineering_smoke=engineering_smoke)
    records = read_ltsn_manifest(manifest_path, contract)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    expected_assignments = dict(
        sorted({row["prompt_id"]: row["split"] for row in raw_rows}.items())
    )
    if split_payload.get("assignments") != expected_assignments:
        raise LTSNContractError("split manifest assignments differ from the training manifest")
    ace_hash, vae_hash, model_family = model_data_identity(raw_rows)
    eligible_values = {
        row.get("qualification_eligible", "false").lower() == "true" for row in raw_rows
    }
    if len(eligible_values) != 1:
        raise LTSNContractError("manifest mixes qualification eligibility")
    qualification_eligible = eligible_values.pop() and not engineering_smoke
    if not engineering_smoke and not qualification_eligible:
        raise LTSNContractError(
            "non-smoke training requires a qualification-eligible label manifest"
        )
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        contract=contract,
        config_path=config_path,
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
        records=records,
        ace_model_sha256=ace_hash,
        vae_sha256=vae_hash,
        model_family=model_family,
        qualification_eligible=qualification_eligible,
        reranking_gate_sha256="" if gate is None else gate.artifact_sha256,
    )
    validate_checkpoint_metadata(metadata, contract)
    train_records = [record for record in records if record.split == "train"]
    development_records = [record for record in records if record.split == "development"]
    if not train_records or not development_records:
        raise LTSNContractError("training requires non-empty train and development splits")
    development_loader = DataLoader(
        LTSNSnapshotDataset(records, "development"),
        batch_size=training.micro_batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=collate_ltsn_batch,
        pin_memory=device.type == "cuda",
    )
    checkpoint_rows: list[dict[str, Any]] = []
    for seed in training.seeds:
        _seed_everything(seed)
        train_dataset = LTSNSnapshotDataset(records, "train")
        sampler = TrajectoryBatchSampler(train_dataset.records, training.micro_batch_size, seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=training.num_workers,
            collate_fn=collate_ltsn_batch,
            pin_memory=device.type == "cuda",
        )
        model = PathHomologySurrogate(contract, model_config).to(device)
        optimizer = AdamW(
            model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
        )
        accumulation = training.effective_batch_size // training.micro_batch_size
        updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
        total_updates = max(1, updates_per_epoch * training.max_epochs)
        warmup_updates = max(1, round(total_updates * training.warmup_fraction))

        minimum_ratio = training.minimum_learning_rate / training.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda update, warmup=warmup_updates, total=total_updates, minimum=minimum_ratio: (
                _schedule_factor(update, warmup, total, minimum)
            ),
        )
        use_bf16 = training.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        best_objective = math.inf
        best_state: dict[str, Tensor] | None = None
        best_epoch = 0
        stale = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, training.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals: list[float] = []
            for batch_index, raw in enumerate(train_loader, start=1):
                batch = _to_device(raw, device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    output = _forward(model, batch)
                losses = _loss(output, batch, loss_weights, device)
                loss = losses["total"].float() / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite LTSN training loss")
                loss.backward()
                totals.append(float(losses["total"].detach().cpu()))
                if batch_index % accumulation == 0 or batch_index == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            development = _development_objective(
                predict_dataset(model, development_loader, device)
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(totals)),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **development,
                }
            )
            if development["objective"] < best_objective - 1e-8:
                best_objective = development["objective"]
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if epoch >= training.minimum_epochs and stale >= training.early_stopping_patience:
                break
        if best_state is None:
            raise RuntimeError("training produced no finite checkpoint")
        checkpoint_path = output_dir / f"ltsn_seed_{seed}.pt"
        temporary = checkpoint_path.with_suffix(".part.pt")
        torch.save(
            {
                "schema_version": 1,
                "state_dict": best_state,
                "model_config": asdict(model_config),
                "training_config": asdict(training),
                "loss_weights": asdict(loss_weights),
                "metadata": metadata,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_development_objective": best_objective,
                "history": history,
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        checkpoint_rows.append(
            {
                "seed": seed,
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "best_epoch": best_epoch,
                "best_development_objective": best_objective,
            }
        )
    ensemble = {
        "schema_version": 1,
        "status": "engineering_smoke_only" if engineering_smoke else "trained_pending_calibration",
        "qualification_eligible": qualification_eligible,
        "device": str(device),
        "precision": "bf16_forward_fp32_loss" if training.use_bf16 else "fp32",
        "metadata": metadata,
        "checkpoints": checkpoint_rows,
    }
    ensemble_path = output_dir / "ensemble_manifest.json"
    write_json_atomic(ensemble_path, ensemble)
    ensemble["ensemble_manifest_sha256"] = sha256_file(ensemble_path)
    return ensemble


def load_checkpoint_model(
    checkpoint_path: Path,
    contract: FingerprintContract,
    device: torch.device,
) -> tuple[PathHomologySurrogate, dict[str, Any]]:
    """Load one hash-bound model checkpoint for calibration or qualification."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    validate_checkpoint_metadata(metadata, contract)
    config = LTSNConfig(**payload["model_config"])
    model = PathHomologySurrogate(contract, config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload
