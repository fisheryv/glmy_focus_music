"""Training and checkpoint issuance for the frozen 18-D surrogate ensemble."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import os
import random
import tomllib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    canonical_json_sha256,
    model_data_identity,
    require_surrogate_training_gate,
    validate_snapshot_coverage,
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
    prompt_grouped_batches: bool = False
    central_direction_grouped_batches: bool = False
    qualification_aligned_early_stopping: bool = False
    coordinate_scale_floor: float = 1e-3
    require_ood_both_classes: bool = False
    ood_positive_weight_cap: float = 20.0
    use_band_improvement_local_loss: bool = False
    normalize_central_direction_by_rms: bool = True
    central_direction_exact_margin: float = 1e-4
    central_direction_primary_early_stopping: bool = False
    central_direction_classification_only: bool = False
    central_direction_overfit_diagnostic: bool = False

    def validate(self, *, engineering_smoke: bool) -> None:
        if self.micro_batch_size < 1 or self.effective_batch_size < self.micro_batch_size:
            raise LTSNContractError("effective batch must be at least the micro batch")
        if self.effective_batch_size % self.micro_batch_size:
            raise LTSNContractError("effective batch must be divisible by micro batch")
        if self.max_epochs < 1 or self.minimum_epochs > self.max_epochs:
            raise LTSNContractError("invalid epoch limits")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise LTSNContractError("training seeds must be non-empty and unique")
        if self.coordinate_scale_floor <= 0 or not math.isfinite(self.coordinate_scale_floor):
            raise LTSNContractError("coordinate_scale_floor must be finite and positive")
        if self.ood_positive_weight_cap < 1 or not math.isfinite(self.ood_positive_weight_cap):
            raise LTSNContractError("ood_positive_weight_cap must be finite and at least one")
        if self.central_direction_exact_margin <= 0 or not math.isfinite(
            self.central_direction_exact_margin
        ):
            raise LTSNContractError("central_direction_exact_margin must be finite and positive")
        if self.central_direction_overfit_diagnostic and not engineering_smoke:
            raise LTSNContractError("central-direction overfit mode is diagnostic-only")
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
    model_raw = _dataclass_values(LTSNConfig, payload.get("model", {}))
    if "inactive_coordinate_indices" in model_raw:
        model_raw["inactive_coordinate_indices"] = tuple(
            int(value) for value in model_raw["inactive_coordinate_indices"]
        )
    model = LTSNConfig(**model_raw)
    training_raw = _dataclass_values(LTSNTrainingConfig, payload.get("training", {}))
    if "seeds" in training_raw:
        training_raw["seeds"] = tuple(int(value) for value in training_raw["seeds"])
    training = LTSNTrainingConfig(**training_raw)
    losses = LTSNLossWeights(**_dataclass_values(LTSNLossWeights, payload.get("loss", {})))
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


class PromptGroupedBatchSampler(Sampler[list[int]]):
    """Pack same-prompt trajectories and local perturbations into auditable batches."""

    def __init__(self, records: Sequence[LTSNSnapshot], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        sample_index = {record.sample_id: index for index, record in enumerate(records)}
        if len(sample_index) != len(records):
            raise LTSNContractError("training records contain duplicate sample IDs")
        perturbations: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            if record.local_anchor_sample_id:
                perturbations[record.local_anchor_sample_id].append(index)
        used: set[int] = set()
        prompt_units: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        for anchor_id, perturbation_indices in sorted(perturbations.items()):
            if anchor_id not in sample_index:
                raise LTSNContractError(
                    f"local perturbation anchor is absent from training: {anchor_id}"
                )
            anchor_index = sample_index[anchor_id]
            unit = (anchor_index, *sorted(perturbation_indices))
            if len(unit) > batch_size:
                raise LTSNContractError(
                    f"local perturbation group exceeds micro batch size: {anchor_id}"
                )
            prompt_ids = {records[index].prompt_id for index in unit}
            if len(prompt_ids) != 1:
                raise LTSNContractError("local perturbation group crosses prompt boundaries")
            prompt_units[prompt_ids.pop()].append(unit)
            used.update(unit)
        trajectories: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            if index not in used:
                trajectories[(record.prompt_id, record.trajectory_id)].append(index)
        for (prompt_id, _), indices in sorted(trajectories.items()):
            ordered = tuple(sorted(indices, key=lambda index: records[index].step_number))
            if len(ordered) > batch_size:
                for start in range(0, len(ordered), batch_size):
                    prompt_units[prompt_id].append(ordered[start : start + batch_size])
            elif ordered:
                prompt_units[prompt_id].append(ordered)
        self.prompt_units = {
            prompt_id: tuple(units) for prompt_id, units in sorted(prompt_units.items())
        }
        if not self.prompt_units:
            raise LTSNContractError("prompt-grouped sampler received no training records")

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        prompt_ids = list(self.prompt_units)
        rng.shuffle(prompt_ids)
        output: list[list[int]] = []
        for prompt_id in prompt_ids:
            prompt_batches: list[list[int]] = []
            batch: list[int] = []
            for unit in self.prompt_units[prompt_id]:
                if batch and len(batch) + len(unit) > self.batch_size:
                    prompt_batches.append(batch)
                    batch = []
                batch.extend(unit)
            if batch:
                prompt_batches.append(batch)
            rng.shuffle(prompt_batches)
            output.extend(prompt_batches)
        return output

    def set_epoch(self, epoch: int) -> None:
        """Change only batch order while keeping deterministic prompt-local packing."""

        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


class CentralDirectionBatchSampler(Sampler[list[int]]):
    """Keep arbitrary central-direction pairs intact for diagnostic controls."""

    def __init__(self, records: Sequence[LTSNSnapshot], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        unpaired: list[int] = []
        for index, record in enumerate(records):
            if record.local_direction_group_id:
                grouped[record.local_direction_group_id].append(index)
            else:
                unpaired.append(index)
        units: list[tuple[int, ...]] = []
        for group_id, indices in sorted(grouped.items()):
            if len(indices) != 2:
                raise LTSNContractError(
                    f"central-direction sampler received an incomplete pair: {group_id}"
                )
            units.append(tuple(sorted(indices)))
        units.extend((index,) for index in unpaired)
        if not units:
            raise LTSNContractError("central-direction sampler received no training records")
        if any(len(unit) > batch_size for unit in units):
            raise LTSNContractError("central-direction group exceeds micro batch size")
        self.units = tuple(units)

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        units = list(self.units)
        rng.shuffle(units)
        output: list[list[int]] = []
        batch: list[int] = []
        for unit in units:
            if batch and len(batch) + len(unit) > self.batch_size:
                output.append(batch)
                batch = []
            batch.extend(unit)
        if batch:
            output.append(batch)
        return output

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pair_indices(
    prompt_ids: Sequence[str],
    device: torch.device,
    valid_mask: Tensor | None = None,
) -> Tensor | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        if valid_mask is not None and not bool(valid_mask[index].item()):
            continue
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


def _local_pair_indices(
    sample_ids: Sequence[str],
    local_anchor_sample_ids: Sequence[str],
    device: torch.device,
    valid_mask: Tensor | None = None,
) -> Tensor | None:
    index_by_sample = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    pairs: list[tuple[int, int]] = []
    for perturbed_index, anchor_id in enumerate(local_anchor_sample_ids):
        if not anchor_id:
            continue
        if valid_mask is not None and not bool(valid_mask[perturbed_index].item()):
            continue
        anchor_index = index_by_sample.get(anchor_id)
        if anchor_index is None:
            raise LTSNContractError(f"local perturbation batch is missing its anchor: {anchor_id}")
        if valid_mask is not None and not bool(valid_mask[anchor_index].item()):
            continue
        pairs.append((anchor_index, perturbed_index))
    return None if not pairs else torch.tensor(pairs, dtype=torch.long, device=device)


def _central_direction_pairs(
    group_ids: Sequence[str],
    signs: Tensor,
    rms_ratios: Tensor,
    device: torch.device,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    """Return ordered ``(-d,+d)`` pairs and their shared RMS values."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        if not group_id:
            continue
        grouped[group_id].append(index)
    pairs: list[tuple[int, int]] = []
    pair_rms: list[float] = []
    signs_cpu = signs.detach().float().cpu().tolist()
    rms_cpu = rms_ratios.detach().float().cpu().tolist()
    for group_id, indices in sorted(grouped.items()):
        if valid_mask is not None and not all(bool(valid_mask[index].item()) for index in indices):
            continue
        by_sign = {float(signs_cpu[index]): index for index in indices}
        if set(by_sign) != {-1.0, 1.0} or len(indices) != 2:
            raise LTSNContractError(
                f"central direction batch is missing a minus/plus member: {group_id}"
            )
        minus, plus = by_sign[-1.0], by_sign[1.0]
        rms = float(rms_cpu[minus])
        if rms <= 0 or not math.isclose(rms, float(rms_cpu[plus]), rel_tol=0, abs_tol=1e-9):
            raise LTSNContractError(f"central direction pair has inconsistent RMS: {group_id}")
        pairs.append((minus, plus))
        pair_rms.append(rms)
    if not pairs:
        return None, None
    return (
        torch.tensor(pairs, dtype=torch.long, device=device),
        torch.tensor(pair_rms, dtype=torch.float32, device=device),
    )


def _training_target_contract(
    records: Sequence[LTSNSnapshot],
    model_config: LTSNConfig,
    training: LTSNTrainingConfig,
    *,
    focus_band_threshold: float | None = None,
) -> dict[str, Any]:
    coordinates = np.asarray([record.coordinates for record in records], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 18:
        raise LTSNContractError("training target coordinates must have shape [N,18]")
    inactive = tuple(model_config.inactive_coordinate_indices)
    active_mask = np.ones(18, dtype=bool)
    active_mask[list(inactive)] = False
    ood = np.asarray([record.ood_label for record in records], dtype=float)
    id_coordinates = coordinates[ood < 0.5]
    if not len(id_coordinates):
        raise LTSNContractError("training target contract requires in-distribution records")
    if inactive and np.max(np.abs(id_coordinates[:, inactive]), initial=0.0) > 1e-10:
        raise LTSNContractError(
            "configured inactive coordinates are not exact-zero in the train split"
        )
    standard_deviation = np.std(id_coordinates, axis=0)
    scale = np.maximum(standard_deviation, training.coordinate_scale_floor)
    scale[~active_mask] = 1.0
    positives = int(np.count_nonzero(ood >= 0.5))
    negatives = int(len(ood) - positives)
    if training.require_ood_both_classes and (positives == 0 or negatives == 0):
        raise LTSNContractError("V2 training requires both ID and OOD samples in the train split")
    positive_weight = (
        1.0
        if positives == 0
        else min(training.ood_positive_weight_cap, max(1.0, negatives / positives))
    )
    return {
        "schema_version": 2 if focus_band_threshold is not None else 1,
        "source": "train_split_only",
        "coordinate_mean": np.mean(id_coordinates, axis=0).tolist(),
        "coordinate_standard_deviation": standard_deviation.tolist(),
        "coordinate_scale": scale.tolist(),
        "active_coordinate_mask": active_mask.tolist(),
        "inactive_coordinate_indices": list(inactive),
        "ood_positive_samples": positives,
        "ood_negative_samples": negatives,
        "ood_positive_weight": positive_weight,
        **(
            {"focus_band_threshold": float(focus_band_threshold)}
            if focus_band_threshold is not None
            else {}
        ),
        **(
            {
                "normalize_central_direction_by_rms": (training.normalize_central_direction_by_rms),
                "central_direction_exact_margin": training.central_direction_exact_margin,
                "central_direction_primary_early_stopping": (
                    training.central_direction_primary_early_stopping
                ),
                "central_direction_classification_only": (
                    training.central_direction_classification_only
                ),
                "central_direction_overfit_diagnostic": (
                    training.central_direction_overfit_diagnostic
                ),
            }
            if not training.normalize_central_direction_by_rms
            or training.central_direction_exact_margin != 1e-4
            or training.central_direction_primary_early_stopping
            or training.central_direction_classification_only
            or training.central_direction_overfit_diagnostic
            else {}
        ),
    }


def _trajectory_pairs(
    trajectory_ids: Sequence[str],
    step_numbers: Tensor,
    device: torch.device,
    valid_mask: Tensor | None = None,
) -> Tensor | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, trajectory_id in enumerate(trajectory_ids):
        if valid_mask is not None and not bool(valid_mask[index].item()):
            continue
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
    target_contract: Mapping[str, Any],
    use_band_improvement_local_loss: bool = False,
    normalize_central_direction_by_rms: bool = True,
    central_direction_exact_margin: float = 1e-4,
    central_direction_classification_only: bool = False,
) -> dict[str, Tensor]:
    pair_indices = _pair_indices(batch["prompt_id"], device, batch["ood_label"] < 0.5)
    central_pairs, central_rms = _central_direction_pairs(
        batch["local_direction_group_id"],
        batch["local_direction_sign"],
        batch["local_direction_rms_ratio"],
        device,
        batch["ood_label"] < 0.5,
    )
    result = ltsn_loss(
        output,
        batch["coordinates"],
        batch["focus_logit"],
        batch["ood_label"],
        pair_indices=pair_indices,
        local_pair_indices=_local_pair_indices(
            batch["sample_id"],
            batch["local_anchor_sample_id"],
            device,
            batch["ood_label"] < 0.5,
        ),
        central_pair_indices=central_pairs,
        central_pair_rms=central_rms,
        coordinate_scale=torch.tensor(
            target_contract["coordinate_scale"], device=device, dtype=torch.float32
        ),
        active_mask=torch.tensor(
            target_contract["active_coordinate_mask"], device=device, dtype=torch.bool
        ),
        ood_positive_weight=torch.tensor(
            target_contract["ood_positive_weight"], device=device, dtype=torch.float32
        ),
        focus_band_threshold=target_contract.get("focus_band_threshold"),
        use_band_improvement_local_loss=use_band_improvement_local_loss,
        normalize_central_direction_by_rms=normalize_central_direction_by_rms,
        central_direction_exact_margin=central_direction_exact_margin,
        central_direction_classification_only=central_direction_classification_only,
        weights=weights,
    )
    trajectory_pairs = _trajectory_pairs(
        batch["trajectory_id"],
        batch["step_number"],
        device,
        batch["ood_label"] < 0.5,
    )
    if trajectory_pairs is not None:
        left, right = trajectory_pairs[:, 0], trajectory_pairs[:, 1]
        delta = trajectory_delta_loss(
            output.coordinate_mean[left],
            output.coordinate_mean[right],
            batch["coordinates"][left],
            batch["coordinates"][right],
            coordinate_scale=torch.tensor(
                target_contract["coordinate_scale"], device=device, dtype=torch.float32
            ),
            active_mask=torch.tensor(
                target_contract["active_coordinate_mask"], device=device, dtype=torch.bool
            ),
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
            ("local_direction_sign", batch["local_direction_sign"]),
            ("local_direction_rms_ratio", batch["local_direction_rms_ratio"]),
        ):
            collected[name].append(tensor.detach().float().cpu().numpy())
        for name in (
            "sample_id",
            "prompt_id",
            "trajectory_id",
            "local_anchor_sample_id",
            "local_direction_group_id",
        ):
            collected[name].extend(raw[name])
    return {
        name: (
            np.concatenate(values, axis=0)
            if values and isinstance(values[0], np.ndarray)
            else values
        )
        for name, values in collected.items()
    }


def _quartile_ranking_accuracy(exact: np.ndarray, predicted: np.ndarray) -> float:
    low = np.flatnonzero(exact <= np.quantile(exact, 0.25))
    high = np.flatnonzero(exact >= np.quantile(exact, 0.75))
    if not len(low) or not len(high):
        return 0.0
    return float(np.mean(predicted[high, None] > predicted[None, low]))


def _development_objective(
    prediction: Mapping[str, Any],
    *,
    active_mask: Sequence[bool] | None = None,
    qualification_aligned: bool = False,
    focus_band_threshold: float | None = None,
    normalize_central_direction_by_rms: bool = True,
    central_direction_exact_margin: float = 1e-4,
    central_direction_classification_only: bool = False,
    central_direction_primary: bool = False,
) -> dict[str, float]:
    in_distribution = np.asarray(prediction["ood_label"], dtype=float) < 0.5
    if not np.any(in_distribution):
        raise LTSNContractError("development objective requires in-distribution samples")
    predicted_focus = prediction["predicted_focus_logit"][in_distribution]
    exact_focus = prediction["focus_logit"][in_distribution]
    predicted_coordinates = prediction["coordinate_mean"][in_distribution]
    exact_coordinates = prediction["coordinates"][in_distribution]
    score_error = float(np.mean(np.abs(predicted_focus - exact_focus)))
    coordinate_rhos = [
        spearman_correlation(predicted_coordinates[:, index], exact_coordinates[:, index])
        for index in range(18)
    ]
    mask = np.ones(18, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if mask.shape != (18,) or not mask.any():
        raise LTSNContractError("development active-coordinate mask is malformed")
    pitch_rho = spearman_correlation(
        np.linalg.norm(predicted_coordinates[:, :16], axis=1),
        np.linalg.norm(exact_coordinates[:, :16], axis=1),
    )
    phase_rho = spearman_correlation(
        np.linalg.norm(predicted_coordinates[:, 16:], axis=1),
        np.linalg.norm(exact_coordinates[:, 16:], axis=1),
    )
    block_rho = 0.5 * (pitch_rho + phase_rho)
    focus_rho = spearman_correlation(predicted_focus, exact_focus)
    coordinate_median = float(np.median(np.asarray(coordinate_rhos)[mask]))
    acoustic_rho = coordinate_rhos[16]
    chroma_rho = coordinate_rhos[17]
    quartile = _quartile_ranking_accuracy(exact_focus, predicted_focus)
    local_pair_count = 0
    local_direction_agreement = 0.0
    local_improvement_mae = 0.0
    central_pair_count = 0
    central_direction_agreement = 0.0
    central_derivative_mae = 0.0
    central_derivative_spearman = 0.0
    central_by_step: dict[int, tuple[int, float]] = {}
    if focus_band_threshold is not None:
        sample_ids = list(prediction.get("sample_id", ()))
        local_anchor_ids = list(prediction.get("local_anchor_sample_id", ()))
        index_by_sample = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        pairs = [
            (index_by_sample[anchor_id], perturbed_index)
            for perturbed_index, anchor_id in enumerate(local_anchor_ids)
            if anchor_id in index_by_sample
            and prediction["ood_label"][index_by_sample[anchor_id]] < 0.5
            and prediction["ood_label"][perturbed_index] < 0.5
        ]
        if pairs:
            anchor = np.asarray([pair[0] for pair in pairs], dtype=int)
            perturbed = np.asarray([pair[1] for pair in pairs], dtype=int)
            threshold = float(focus_band_threshold)
            exact_loss = np.maximum(0.0, threshold - prediction["focus_logit"]) ** 2
            predicted_loss = np.maximum(0.0, threshold - prediction["predicted_focus_logit"]) ** 2
            exact_improvement = exact_loss[anchor] - exact_loss[perturbed]
            predicted_improvement = predicted_loss[anchor] - predicted_loss[perturbed]
            informative = np.abs(exact_improvement) >= 1e-5
            if np.any(informative):
                exact_improvement = exact_improvement[informative]
                predicted_improvement = predicted_improvement[informative]
                local_pair_count = int(len(exact_improvement))
                local_direction_agreement = float(
                    np.mean(np.sign(exact_improvement) == np.sign(predicted_improvement))
                )
                local_improvement_mae = float(
                    np.mean(np.abs(exact_improvement - predicted_improvement))
                )
        group_ids = list(prediction.get("local_direction_group_id", ()))
        signs = np.asarray(prediction.get("local_direction_sign", ()), dtype=float)
        rms_values = np.asarray(prediction.get("local_direction_rms_ratio", ()), dtype=float)
        steps = np.asarray(prediction.get("step_number", ()), dtype=int)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, group_id in enumerate(group_ids):
            if group_id:
                grouped[group_id].append(index)
        exact_derivatives: list[float] = []
        predicted_derivatives: list[float] = []
        derivative_steps: list[int] = []
        threshold = float(focus_band_threshold)
        exact_loss = np.maximum(0.0, threshold - prediction["focus_logit"]) ** 2
        predicted_loss = np.maximum(0.0, threshold - prediction["predicted_focus_logit"]) ** 2
        for group_id, indices in sorted(grouped.items()):
            if not all(prediction["ood_label"][index] < 0.5 for index in indices):
                continue
            by_sign = {float(signs[index]): index for index in indices}
            if set(by_sign) != {-1.0, 1.0} or len(indices) != 2:
                raise LTSNContractError(
                    f"development central direction pair is incomplete: {group_id}"
                )
            minus, plus = by_sign[-1.0], by_sign[1.0]
            rms = float(rms_values[minus])
            if rms <= 0 or not math.isclose(rms, float(rms_values[plus]), rel_tol=0, abs_tol=1e-9):
                raise LTSNContractError(
                    f"development central direction RMS is inconsistent: {group_id}"
                )
            denominator = 2.0 * rms if normalize_central_direction_by_rms else 1.0
            exact_derivative = float((exact_loss[minus] - exact_loss[plus]) / denominator)
            if abs(exact_derivative) < central_direction_exact_margin:
                continue
            predicted_derivative = float(
                (
                    prediction["predicted_focus_logit"][plus]
                    - prediction["predicted_focus_logit"][minus]
                )
                / denominator
                if central_direction_classification_only
                else (predicted_loss[minus] - predicted_loss[plus]) / denominator
            )
            exact_derivatives.append(exact_derivative)
            predicted_derivatives.append(predicted_derivative)
            derivative_steps.append(int(steps[minus]))
        if exact_derivatives:
            exact_array = np.asarray(exact_derivatives, dtype=float)
            predicted_array = np.asarray(predicted_derivatives, dtype=float)
            central_pair_count = len(exact_derivatives)
            central_direction_agreement = float(
                np.mean(np.sign(exact_array) == np.sign(predicted_array))
            )
            central_derivative_mae = float(np.mean(np.abs(exact_array - predicted_array)))
            central_derivative_spearman = spearman_correlation(exact_array, predicted_array)
            for step in (4, 5, 6):
                mask_step = np.asarray(derivative_steps) == step
                if np.any(mask_step):
                    central_by_step[step] = (
                        int(np.count_nonzero(mask_step)),
                        float(
                            np.mean(
                                np.sign(exact_array[mask_step])
                                == np.sign(predicted_array[mask_step])
                            )
                        ),
                    )
    if qualification_aligned:
        thresholds = (
            (focus_rho, 0.70),
            (coordinate_median, 0.50),
            (pitch_rho, 0.50),
            (phase_rho, 0.50),
            (acoustic_rho, 0.50),
            (chroma_rho, 0.50),
            (quartile, 0.65),
        )
        objective = sum(max(0.0, threshold - value) for value, threshold in thresholds)
        objective += 0.05 * score_error
        if local_pair_count:
            objective += max(0.0, 0.65 - local_direction_agreement)
        if central_pair_count:
            objective += max(0.0, 0.65 - central_direction_agreement)
    else:
        objective = score_error + (1.0 - coordinate_median) + (1.0 - block_rho)
    if central_direction_primary:
        if central_pair_count == 0:
            raise LTSNContractError(
                "central-direction-primary early stopping requires informative development pairs"
            )
        objective = (
            (1.0 - central_direction_agreement)
            + 0.25 * (1.0 - central_derivative_spearman)
            + 0.05 * objective
        )
    result = {
        "objective": objective,
        "n_in_distribution": int(np.count_nonzero(in_distribution)),
        "score_mae": score_error,
        "focus_logit_spearman": focus_rho,
        "coordinate_median_spearman": coordinate_median,
        "pitch_block_spearman": pitch_rho,
        "phase_block_spearman": phase_rho,
        "acoustic_loop_coordinate_spearman": acoustic_rho,
        "chroma_loop_coordinate_spearman": chroma_rho,
        "quartile_ranking_accuracy": quartile,
        "local_improvement_pairs": local_pair_count,
        "local_direction_agreement": local_direction_agreement,
        "local_improvement_mae": local_improvement_mae,
        "central_direction_pairs": central_pair_count,
        "central_direction_agreement": central_direction_agreement,
        "central_derivative_mae": central_derivative_mae,
        "central_derivative_spearman": central_derivative_spearman,
    }
    for step in (4, 5, 6):
        count, agreement = central_by_step.get(step, (0, 0.0))
        result[f"central_direction_step_{step}_pairs"] = count
        result[f"central_direction_step_{step}_agreement"] = agreement
    return result


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
    surrogate_training_gate_sha256: str,
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
        "surrogate_training_gate_sha256": surrogate_training_gate_sha256,
        "guidance_promotion_eligible": False,
    }


def _resolve_training_devices(
    seeds: Sequence[int],
    device_name: str | None,
    device_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve either one sequential device or one explicit CUDA device per seed."""

    if device_name is not None and device_names is not None:
        raise LTSNContractError("--device and --devices are mutually exclusive")
    if device_names is None:
        return (device_name or ("cuda" if torch.cuda.is_available() else "cpu"),)

    devices = tuple(str(value).strip() for value in device_names)
    if not devices or any(not value for value in devices):
        raise LTSNContractError("--devices requires one non-empty device per seed")
    if len(devices) != len(seeds):
        raise LTSNContractError(
            f"parallel training requires {len(seeds)} devices for {len(seeds)} seeds"
        )
    if len(set(devices)) != len(devices):
        raise LTSNContractError("parallel training devices must be unique")
    for value in devices:
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise LTSNContractError(
                "parallel training requires explicit CUDA devices such as cuda:0"
            )
    return devices


def _validate_training_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device_name} is unavailable; found {torch.cuda.device_count()} GPUs"
            )
    return device


def _train_seed(
    *,
    seed: int,
    device_name: str,
    contract: FingerprintContract,
    model_config: LTSNConfig,
    training: LTSNTrainingConfig,
    loss_weights: LTSNLossWeights,
    records: Sequence[LTSNSnapshot],
    metadata: dict[str, Any],
    target_contract: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Train one independent ensemble member and write only its seed checkpoint."""

    device = _validate_training_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _seed_everything(seed)
    development_loader = DataLoader(
        LTSNSnapshotDataset(records, "development"),
        batch_size=training.micro_batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=collate_ltsn_batch,
        pin_memory=device.type == "cuda",
    )
    train_dataset = LTSNSnapshotDataset(records, "train")
    if training.central_direction_grouped_batches:
        sampler: Sampler[list[int]] = CentralDirectionBatchSampler(
            train_dataset.records, training.micro_batch_size, seed
        )
    elif training.prompt_grouped_batches:
        sampler = PromptGroupedBatchSampler(train_dataset.records, training.micro_batch_size, seed)
    else:
        sampler = TrajectoryBatchSampler(train_dataset.records, training.micro_batch_size, seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=training.num_workers,
        collate_fn=collate_ltsn_batch,
        pin_memory=device.type == "cuda",
    )
    overfit_loader = (
        DataLoader(
            train_dataset,
            batch_size=training.micro_batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            collate_fn=collate_ltsn_batch,
            pin_memory=device.type == "cuda",
        )
        if training.central_direction_overfit_diagnostic
        else None
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
    best_development_objective = math.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, training.max_epochs + 1):
        if isinstance(sampler, (PromptGroupedBatchSampler, CentralDirectionBatchSampler)):
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: list[float] = []
        component_totals: dict[str, list[float]] = defaultdict(list)
        local_pairs_seen = 0
        central_pairs_seen = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _to_device(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = _forward(model, batch)
            losses = _loss(
                output,
                batch,
                loss_weights,
                device,
                target_contract,
                use_band_improvement_local_loss=training.use_band_improvement_local_loss,
                normalize_central_direction_by_rms=(training.normalize_central_direction_by_rms),
                central_direction_exact_margin=training.central_direction_exact_margin,
                central_direction_classification_only=(
                    training.central_direction_classification_only
                ),
            )
            loss = losses["total"].float() / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite LTSN training loss")
            loss.backward()
            totals.append(float(losses["total"].detach().cpu()))
            for name, value in losses.items():
                component_totals[name].append(float(value.detach().cpu()))
            local_pairs_seen += sum(bool(value) for value in raw["local_anchor_sample_id"])
            central_pairs_seen += len({value for value in raw["local_direction_group_id"] if value})
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        development = _development_objective(
            predict_dataset(model, development_loader, device),
            active_mask=target_contract["active_coordinate_mask"],
            qualification_aligned=training.qualification_aligned_early_stopping,
            focus_band_threshold=target_contract.get("focus_band_threshold"),
            normalize_central_direction_by_rms=(training.normalize_central_direction_by_rms),
            central_direction_exact_margin=training.central_direction_exact_margin,
            central_direction_classification_only=(training.central_direction_classification_only),
            central_direction_primary=training.central_direction_primary_early_stopping,
        )
        overfit_training: dict[str, float] = {}
        selection_objective = development["objective"]
        if overfit_loader is not None:
            overfit_training = _development_objective(
                predict_dataset(model, overfit_loader, device),
                active_mask=target_contract["active_coordinate_mask"],
                focus_band_threshold=target_contract.get("focus_band_threshold"),
                normalize_central_direction_by_rms=(training.normalize_central_direction_by_rms),
                central_direction_exact_margin=training.central_direction_exact_margin,
                central_direction_classification_only=(
                    training.central_direction_classification_only
                ),
                central_direction_primary=True,
            )
            selection_objective = (
                1.0
                - overfit_training["central_direction_agreement"]
                + 0.25 * (1.0 - overfit_training["central_derivative_spearman"])
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(totals)),
                "train_loss_components": {
                    name: float(np.mean(values))
                    for name, values in sorted(component_totals.items())
                },
                "local_direction_pairs_seen": local_pairs_seen,
                "central_direction_pairs_seen": central_pairs_seen,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selection_objective": selection_objective,
                **development,
                **{f"overfit_train_{name}": value for name, value in overfit_training.items()},
            }
        )
        if selection_objective < best_objective - 1e-8:
            best_objective = selection_objective
            best_development_objective = development["objective"]
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
            "training_target_contract": dict(target_contract),
            "metadata": metadata,
            "seed": seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_selection_objective": best_objective,
            "best_development_objective": best_development_objective,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    return {
        "seed": seed,
        "device": str(device),
        "path": checkpoint_path.name,
        "sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "best_selection_objective": best_objective,
        "best_development_objective": best_development_objective,
    }


def train_ensemble(
    *,
    fingerprint_path: Path,
    manifest_path: Path,
    split_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    surrogate_training_gate_path: Path | None,
    engineering_smoke: bool,
    device_name: str | None = None,
    device_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train seed-independent members sequentially or on one explicit GPU each."""

    contract = load_fingerprint_contract(fingerprint_path)
    gate = require_surrogate_training_gate(
        surrogate_training_gate_path, contract, engineering_smoke=engineering_smoke
    )
    model_config, training, loss_weights = load_training_config(config_path)
    training.validate(engineering_smoke=engineering_smoke)
    loss_weights.validate()
    records = read_ltsn_manifest(manifest_path, contract)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not engineering_smoke:
        validate_snapshot_coverage(
            [row for row in raw_rows if not row.get("training_augmentation_kind", "").strip()]
        )
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
    expected_training_gate_sha256 = "" if gate is None else gate.artifact_sha256
    manifest_gate_values = {row.get("surrogate_training_gate_sha256", "") for row in raw_rows}
    if manifest_gate_values != {expected_training_gate_sha256}:
        raise LTSNContractError(
            "label manifest uses a different surrogate training gate: "
            f"expected {expected_training_gate_sha256 or '<none>'}, "
            f"manifest has {sorted(value or '<none>' for value in manifest_gate_values)}"
        )
    guidance_values = {row.get("guidance_promotion_eligible", "false").lower() for row in raw_rows}
    if guidance_values != {"false"}:
        raise LTSNContractError(
            "surrogate training manifest must not claim guidance promotion eligibility"
        )
    if loss_weights.local_direction > 0 and not any(
        row.get("local_anchor_sample_id", "").strip() for row in raw_rows
    ):
        raise LTSNContractError(
            "local_direction loss requires exact-labelled local perturbation records"
        )
    if loss_weights.central_direction > 0 and not any(
        row.get("local_direction_group_id", "").strip() for row in raw_rows
    ):
        raise LTSNContractError(
            "central_direction loss requires V5 symmetric finite-difference records"
        )
    if training.central_direction_primary_early_stopping and loss_weights.central_direction <= 0:
        raise LTSNContractError(
            "central-direction-primary early stopping requires positive central_direction loss"
        )
    if training.central_direction_overfit_diagnostic:
        other_weights = {
            name: value
            for name, value in asdict(loss_weights).items()
            if name != "central_direction" and value != 0
        }
        if loss_weights.central_direction <= 0 or other_weights:
            raise LTSNContractError(
                "V5.1a overfit diagnostic requires a central-direction-only loss"
            )
        if (
            training.use_bf16
            or not (training.prompt_grouped_batches or training.central_direction_grouped_batches)
            or not training.central_direction_classification_only
        ):
            raise LTSNContractError(
                "V5.1a overfit diagnostic requires FP32, grouped direction batches, "
                "and direction-classification loss"
            )
        if training.prompt_grouped_batches and training.central_direction_grouped_batches:
            raise LTSNContractError(
                "diagnostic training cannot enable both prompt and central-direction samplers"
            )
    devices = _resolve_training_devices(training.seeds, device_name, device_names)
    for value in devices:
        _validate_training_device(value)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_path = output_dir / "ensemble_manifest.json"
    if ensemble_path.exists():
        raise LTSNContractError(
            "output directory already contains an ensemble manifest; use a new run directory"
        )
    train_records = [record for record in records if record.split == "train"]
    development_records = [record for record in records if record.split == "development"]
    if not train_records or not development_records:
        raise LTSNContractError("training requires non-empty train and development splits")
    target_contract = _training_target_contract(
        train_records,
        model_config,
        training,
        focus_band_threshold=contract.focus_band_threshold,
    )
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
        surrogate_training_gate_sha256="" if gate is None else gate.artifact_sha256,
    )
    metadata["training_target_contract_sha256"] = canonical_json_sha256(target_contract)
    validate_checkpoint_metadata(metadata, contract)
    if len(devices) == 1:
        checkpoint_rows = [
            _train_seed(
                seed=seed,
                device_name=devices[0],
                contract=contract,
                model_config=model_config,
                training=training,
                loss_weights=loss_weights,
                records=records,
                metadata=metadata,
                target_contract=target_contract,
                output_dir=output_dir,
            )
            for seed in training.seeds
        ]
    else:
        context = multiprocessing.get_context("spawn")
        checkpoint_by_seed: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = {
                executor.submit(
                    _train_seed,
                    seed=seed,
                    device_name=device,
                    contract=contract,
                    model_config=model_config,
                    training=training,
                    loss_weights=loss_weights,
                    records=records,
                    metadata=metadata,
                    target_contract=target_contract,
                    output_dir=output_dir,
                ): (seed, device)
                for seed, device in zip(training.seeds, devices, strict=True)
            }
            for future in as_completed(futures):
                seed, device = futures[future]
                try:
                    checkpoint_by_seed[seed] = future.result()
                except Exception as error:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"parallel LTSN seed {seed} failed on {device}: {error}"
                    ) from error
        checkpoint_rows = [checkpoint_by_seed[seed] for seed in training.seeds]
    ensemble = {
        "schema_version": 1,
        "status": "engineering_smoke_only" if engineering_smoke else "trained_pending_calibration",
        "qualification_eligible": qualification_eligible,
        "device": devices[0] if len(devices) == 1 else "parallel",
        "devices": list(devices),
        "parallel_training": len(devices) > 1,
        "precision": "bf16_forward_fp32_loss" if training.use_bf16 else "fp32",
        "metadata": metadata,
        "training_target_contract": target_contract,
        "checkpoints": checkpoint_rows,
    }
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
    target_contract = payload.get("training_target_contract")
    expected_target_sha256 = metadata.get("training_target_contract_sha256")
    if expected_target_sha256 is not None:
        if (
            not isinstance(target_contract, dict)
            or canonical_json_sha256(target_contract) != expected_target_sha256
        ):
            raise LTSNContractError("checkpoint training target contract is hash-mismatched")
    config = LTSNConfig(**payload["model_config"])
    model = PathHomologySurrogate(contract, config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload
