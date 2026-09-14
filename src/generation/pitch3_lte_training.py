"""Training and development metrics for the independent V3-LTE contract."""

from __future__ import annotations

import csv
import json
import math
import random
import tomllib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_json_atomic
from .pitch3_contract import load_pitch3_contract
from .pitch3_lte import Pitch3LTEConfig, PromptConditionedTopologyEnergy
from .pitch3_lte_data import (
    LTE_MODEL_FAMILY,
    LTE_RADIUS_RATIO,
    validate_pitch3_lte_dataset_preflight,
)

LTE_COLLAPSE_PREDICTED_RANGE = 1e-3
LTE_COLLAPSE_EXACT_RANGE = 0.1
LTE_MAX_COLLAPSED_PROMPT_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class Pitch3LTETrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    minimum_epochs: int = 8
    early_stopping_patience: int = 7
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    use_bf16: bool = True
    seed: int = 20260920
    huber_delta: float = 1.0
    rank_min_delta: float = 1e-6
    local_objective: str = "central_difference_huber_v3"
    local_shape_weight: float = 0.25
    local_flat_weight: float = 0.25
    local_direction_weight: float = 1.0
    cross_prompt_rank_weight: float = 0.0
    prompt_groups_per_batch: int = 1
    prompt_dropout_probability: float = 0.0
    prompt_consistency_weight: float = 0.0
    prompt_frozen_epochs: int = 0
    energy_strata_positive_bins: int = 1
    energy_strata_weight_cap: float = 1.0

    def validate(self) -> None:
        if self.max_epochs < 1 or not 1 <= self.minimum_epochs <= self.max_epochs:
            raise ValueError("V3-LTE epoch limits are invalid")
        if self.early_stopping_patience < 1 or self.gradient_clip_norm <= 0:
            raise ValueError("V3-LTE patience and gradient clip must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("V3-LTE optimizer settings are invalid")
        if self.huber_delta <= 0 or self.rank_min_delta <= 0:
            raise ValueError("V3-LTE robust-loss thresholds must be positive")
        if self.local_objective not in {
            "central_difference_huber_v3",
            "robust_direction_v31",
            "decomposed_direction_v32",
        }:
            raise ValueError("unknown V3-LTE local objective")
        if min(
            self.local_direction_weight,
            self.local_shape_weight,
            self.local_flat_weight,
            self.cross_prompt_rank_weight,
        ) < 0:
            raise ValueError("V3-LTE local robust-loss weights must be non-negative")
        if self.prompt_groups_per_batch < 1:
            raise ValueError("V3-LTE prompt groups per batch must be positive")
        if not 0.0 <= self.prompt_dropout_probability < 1.0:
            raise ValueError("V3-LTE prompt dropout must lie in [0,1)")
        if self.prompt_consistency_weight < 0:
            raise ValueError("V3-LTE prompt consistency weight must be non-negative")
        if self.prompt_frozen_epochs < 0 or self.prompt_frozen_epochs >= self.max_epochs:
            raise ValueError("V3-LTE prompt frozen epochs are invalid")
        if self.energy_strata_positive_bins < 1:
            raise ValueError("V3-LTE energy strata bin count must be positive")
        if self.energy_strata_weight_cap < 1.0:
            raise ValueError("V3-LTE energy strata weight cap must be at least one")


@dataclass(frozen=True, slots=True)
class Pitch3LTEExample:
    sample_id: str
    prompt_id: str
    prompt_family: str
    trajectory_id: str
    split: str
    source_kind: str
    latent_path: Path
    latent_sha256: str
    prompt_embedding_path: Path
    prompt_embedding_sha256: str
    exact_band: float
    energy_target: float
    direction_id: str
    direction_sign: int
    epsilon: float
    timestep: float
    ace_model_sha256: str
    vae_sha256: str


def load_pitch3_lte_config(path: Path) -> tuple[Pitch3LTEConfig, Pitch3LTETrainingConfig]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model = Pitch3LTEConfig(**payload.get("model", {}))
    training = Pitch3LTETrainingConfig(**payload.get("training", {}))
    model.validate()
    training.validate()
    return model, training


def _read_examples(path: Path, fingerprint_sha256: str) -> list[Pitch3LTEExample]:
    records: list[Pitch3LTEExample] = []
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("fingerprint_json_sha256", "").lower() != fingerprint_sha256:
                raise LTSNContractError("V3-LTE dataset fingerprint hash mismatch")
            split = row["split"]
            if split not in {"train", "development"}:
                raise LTSNContractError("V3-LTE training accepts only train/development data")
            latent_path = (path.parent / row["latent_path"]).resolve()
            embedding_path = (path.parent / row["prompt_embedding_path"]).resolve()
            if not latent_path.is_file() or sha256_file(latent_path) != row["latent_sha256"]:
                raise LTSNContractError("V3-LTE latent is missing or hash-mismatched")
            if (
                not embedding_path.is_file()
                or sha256_file(embedding_path) != row["prompt_embedding_sha256"]
            ):
                raise LTSNContractError("V3-LTE prompt state is missing or hash-mismatched")
            target = float(row["energy_target"])
            band = float(row["exact_band"])
            epsilon = float(row["epsilon"])
            if not all(math.isfinite(value) for value in (target, band, epsilon)) or band < 0:
                raise LTSNContractError("V3-LTE target contains invalid values")
            if not math.isclose(target, math.log1p(band), abs_tol=1e-9):
                raise LTSNContractError("V3-LTE scalar target is not log1p(exact_band)")
            source_kind = row["source_kind"]
            direction_id = row.get("direction_id", "")
            direction_sign = int(float(row.get("direction_sign", 0)))
            if source_kind == "local_finite_difference":
                if not direction_id or direction_sign not in {-1, 1} or epsilon <= 0:
                    raise LTSNContractError("V3-LTE finite-difference row is malformed")
                if not math.isclose(float(row["radius_ratio"]), LTE_RADIUS_RATIO, abs_tol=1e-12):
                    raise LTSNContractError("V3-LTE local radius changed from 5%")
            elif source_kind != "base_step4_seed" or direction_id or direction_sign:
                raise LTSNContractError("V3-LTE base row is malformed")
            prompt_splits[row["prompt_id"]].add(split)
            records.append(
                Pitch3LTEExample(
                    sample_id=row["sample_id"],
                    prompt_id=row["prompt_id"],
                    prompt_family=row["prompt_family"],
                    trajectory_id=row["trajectory_id"],
                    split=split,
                    source_kind=source_kind,
                    latent_path=latent_path,
                    latent_sha256=row["latent_sha256"],
                    prompt_embedding_path=embedding_path,
                    prompt_embedding_sha256=row["prompt_embedding_sha256"],
                    exact_band=band,
                    energy_target=target,
                    direction_id=direction_id,
                    direction_sign=direction_sign,
                    epsilon=epsilon,
                    timestep=float(row["timestep"]),
                    ace_model_sha256=row["ace_model_sha256"],
                    vae_sha256=row["vae_sha256"],
                )
            )
    if not records or any(len(values) != 1 for values in prompt_splits.values()):
        raise LTSNContractError("V3-LTE dataset is empty or leaks prompts across splits")
    return records


class Pitch3LTEDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[Pitch3LTEExample]) -> None:
        self.records = records
        self._prompt_cache: dict[Path, tuple[Tensor, Tensor]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latent = np.load(record.latent_path, allow_pickle=False)
        if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
            raise LTSNContractError(f"invalid V3-LTE latent: {record.latent_path}")
        prompt = self._prompt_cache.get(record.prompt_embedding_path)
        if prompt is None:
            with np.load(record.prompt_embedding_path, allow_pickle=False) as payload:
                hidden = np.asarray(payload["hidden"], dtype=np.float32)
                mask = np.asarray(payload["mask"], dtype=np.bool_)
            if hidden.ndim != 2 or hidden.shape[1] != 1024 or mask.shape != hidden.shape[:1]:
                raise LTSNContractError("invalid frozen ACE prompt state")
            prompt = (torch.from_numpy(hidden), torch.from_numpy(mask))
            self._prompt_cache[record.prompt_embedding_path] = prompt
        return {
            "sample_id": record.sample_id,
            "prompt_id": record.prompt_id,
            "prompt_family": record.prompt_family,
            "source_kind": record.source_kind,
            "latent": torch.from_numpy(np.asarray(latent, dtype=np.float32)),
            "text_hidden": prompt[0],
            "text_mask": prompt[1],
            "exact_band": torch.tensor(record.exact_band, dtype=torch.float32),
            "energy_target": torch.tensor(record.energy_target, dtype=torch.float32),
            "direction_id": record.direction_id,
            "direction_sign": record.direction_sign,
            "epsilon": torch.tensor(record.epsilon, dtype=torch.float32),
        }


def collate_pitch3_lte(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty V3-LTE batch")
    frames = max(int(item["latent"].shape[0]) for item in items)
    tokens = max(int(item["text_hidden"].shape[0]) for item in items)
    latent = torch.zeros(len(items), frames, 64, dtype=torch.float32)
    latent_mask = torch.zeros(len(items), frames, dtype=torch.bool)
    text_hidden = torch.zeros(len(items), tokens, 1024, dtype=torch.float32)
    text_mask = torch.zeros(len(items), tokens, dtype=torch.bool)
    for index, item in enumerate(items):
        frame_count = int(item["latent"].shape[0])
        token_count = int(item["text_hidden"].shape[0])
        latent[index, :frame_count] = item["latent"]
        latent_mask[index, :frame_count] = True
        text_hidden[index, :token_count] = item["text_hidden"]
        text_mask[index, :token_count] = item["text_mask"]
    return {
        "sample_id": [item["sample_id"] for item in items],
        "prompt_id": [item["prompt_id"] for item in items],
        "prompt_family": [item["prompt_family"] for item in items],
        "source_kind": [item["source_kind"] for item in items],
        "latent": latent,
        "attention_mask": latent_mask,
        "text_hidden": text_hidden,
        "text_mask": text_mask,
        "exact_band": torch.stack([item["exact_band"] for item in items]),
        "energy_target": torch.stack([item["energy_target"] for item in items]),
        "direction_id": [item["direction_id"] for item in items],
        "direction_sign": torch.tensor([item["direction_sign"] for item in items]),
        "epsilon": torch.stack([item["epsilon"] for item in items]),
    }


class PromptBatchSampler(Sampler[list[int]]):
    """Keep complete prompt groups intact while optionally batching several."""

    def __init__(
        self,
        records: Sequence[Pitch3LTEExample],
        seed: int,
        shuffle: bool,
        groups_per_batch: int = 1,
    ) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.prompt_id].append(index)
        self.groups = []
        for prompt_id, indices in sorted(grouped.items()):
            base = [index for index in indices if records[index].source_kind == "base_step4_seed"]
            local = [
                index
                for index in indices
                if records[index].source_kind == "local_finite_difference"
            ]
            if len(base) != 4:
                raise LTSNContractError(
                    f"V3-LTE prompt does not contain four base seeds: {prompt_id}"
                )
            if local and (
                len(local) != 4
                or {records[index].direction_sign for index in local} != {-1, 1}
                or len({records[index].direction_id for index in local}) != 2
            ):
                raise LTSNContractError(f"V3-LTE prompt has incomplete local pairs: {prompt_id}")
            self.groups.append(base + local)
        self.seed = seed
        self.shuffle = shuffle
        if groups_per_batch < 1:
            raise ValueError("groups_per_batch must be positive")
        self.groups_per_batch = groups_per_batch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        groups = [values.copy() for values in self.groups]
        if self.shuffle:
            random.Random(self.seed + 1_000_003 * self.epoch).shuffle(groups)
        for start in range(0, len(groups), self.groups_per_batch):
            selected = groups[start : start + self.groups_per_batch]
            yield [index for group in selected for index in group]

    def __len__(self) -> int:
        return math.ceil(len(self.groups) / self.groups_per_batch)


def _pair_indices(batch: Mapping[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    base_by_prompt: dict[str, list[int]] = defaultdict(list)
    for index, (kind, prompt_id) in enumerate(
        zip(batch["source_kind"], batch["prompt_id"], strict=True)
    ):
        if kind == "base_step4_seed":
            base_by_prompt[str(prompt_id)].append(index)
    rank_pairs = [
        (left, right)
        for base in base_by_prompt.values()
        for offset, left in enumerate(base)
        for right in base[offset + 1 :]
    ]
    directions: dict[str, dict[int, int]] = defaultdict(dict)
    for index, (direction_id, sign) in enumerate(
        zip(batch["direction_id"], batch["direction_sign"].tolist(), strict=True)
    ):
        if direction_id:
            directions[direction_id][int(sign)] = index
    fd_pairs = []
    for direction_id, signed in sorted(directions.items()):
        if set(signed) != {-1, 1}:
            raise LTSNContractError(f"incomplete V3-LTE local pair: {direction_id}")
        fd_pairs.append((signed[-1], signed[1]))
    return rank_pairs, fd_pairs


def _cross_prompt_rank_pairs(batch: Mapping[str, Any]) -> list[tuple[int, int]]:
    base_by_prompt: dict[str, list[int]] = defaultdict(list)
    for index, (kind, prompt_id) in enumerate(
        zip(batch["source_kind"], batch["prompt_id"], strict=True)
    ):
        if kind == "base_step4_seed":
            base_by_prompt[str(prompt_id)].append(index)
    prompt_groups = list(base_by_prompt.values())
    return [
        (left, right)
        for group_index, left_group in enumerate(prompt_groups)
        for right_group in prompt_groups[group_index + 1 :]
        for left in left_group
        for right in right_group
    ]


def pitch3_lte_raw_losses(
    predicted: Tensor,
    batch: Mapping[str, Any],
    *,
    huber_delta: float,
    rank_min_delta: float,
    local_objective: str = "central_difference_huber_v3",
    local_derivative_scale: float = 1.0,
    local_delta_scale: float = 1.0,
    local_shape_weight: float = 0.25,
    local_flat_weight: float = 0.25,
    include_cross_prompt_rank: bool = False,
    energy_strata_thresholds: Sequence[float] = (),
    energy_strata_weights: Sequence[float] = (),
) -> dict[str, Tensor]:
    target = batch["energy_target"].float()
    value_terms = F.huber_loss(
        predicted.float(), target, delta=huber_delta, reduction="none"
    )
    if energy_strata_thresholds:
        if len(energy_strata_weights) != len(energy_strata_thresholds) + 1:
            raise ValueError("V3-LTE energy strata weights do not match thresholds")
        boundaries = target.new_tensor(tuple(energy_strata_thresholds))
        strata = torch.bucketize(target, boundaries, right=False)
        weights = target.new_tensor(tuple(energy_strata_weights))[strata]
        value = (value_terms * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        value = value_terms.mean()
    rank_terms = []
    cross_rank_terms = []
    fd_terms = []
    direction_terms = []
    shape_terms = []
    flat_terms = []
    zero = predicted.sum() * 0.0
    rank_pairs, fd_pairs = _pair_indices(batch)
    for left, right in rank_pairs:
        exact_delta = target[left] - target[right]
        if exact_delta.abs() >= rank_min_delta:
            rank_terms.append(
                F.softplus(-torch.sign(exact_delta) * (predicted[left] - predicted[right]))
            )
    if include_cross_prompt_rank:
        for left, right in _cross_prompt_rank_pairs(batch):
            exact_delta = target[left] - target[right]
            if exact_delta.abs() >= rank_min_delta:
                cross_rank_terms.append(
                    F.softplus(-torch.sign(exact_delta) * (predicted[left] - predicted[right]))
                )
    for minus, plus in fd_pairs:
        epsilon = batch["epsilon"][minus].float()
        if not torch.isclose(epsilon, batch["epsilon"][plus].float(), atol=1e-9, rtol=0):
            raise LTSNContractError("V3-LTE finite-difference epsilon differs within a pair")
        exact_derivative = (target[plus] - target[minus]) / (2.0 * epsilon)
        predicted_derivative = (predicted[plus] - predicted[minus]) / (2.0 * epsilon)
        if local_objective == "central_difference_huber_v3":
            fd_terms.append(
                F.huber_loss(predicted_derivative, exact_derivative, delta=huber_delta)
            )
            continue
        if local_objective not in {"robust_direction_v31", "decomposed_direction_v32"}:
            raise ValueError(f"unknown V3-LTE local objective: {local_objective}")
        predicted_delta = predicted[plus] - predicted[minus]
        if exact_derivative.abs() <= rank_min_delta:
            flat_terms.append(F.smooth_l1_loss(predicted_delta / local_delta_scale, zero))
            continue
        direction_terms.append(
            F.softplus(-torch.sign(exact_derivative) * predicted_delta / local_delta_scale)
        )
        exact_robust = torch.asinh(exact_derivative / local_derivative_scale)
        predicted_robust = torch.asinh(predicted_derivative / local_derivative_scale)
        shape_terms.append(F.huber_loss(predicted_robust, exact_robust, delta=huber_delta))
    losses = {
        "value": value,
        "prompt_rank": torch.stack(rank_terms).mean() if rank_terms else zero,
    }
    if include_cross_prompt_rank:
        losses["cross_prompt_rank"] = (
            torch.stack(cross_rank_terms).mean() if cross_rank_terms else zero
        )
    if local_objective == "central_difference_huber_v3":
        losses["local_fd"] = torch.stack(fd_terms).mean() if fd_terms else zero
    elif local_objective == "robust_direction_v31":
        direction = torch.stack(direction_terms).mean() if direction_terms else zero
        shape = torch.stack(shape_terms).mean() if shape_terms else zero
        flat = torch.stack(flat_terms).mean() if flat_terms else zero
        losses["local_robust"] = (
            direction + local_shape_weight * shape + local_flat_weight * flat
        )
    else:
        losses["local_direction"] = (
            torch.stack(direction_terms).mean() if direction_terms else zero
        )
        losses["local_shape"] = torch.stack(shape_terms).mean() if shape_terms else zero
        losses["local_flat"] = torch.stack(flat_terms).mean() if flat_terms else zero
    return losses


def _local_training_scales(
    records: Sequence[Pitch3LTEExample],
    minimum_delta: float,
) -> dict[str, float]:
    signed: dict[str, dict[int, Pitch3LTEExample]] = defaultdict(dict)
    for record in records:
        if record.direction_id:
            signed[record.direction_id][record.direction_sign] = record
    derivatives: list[float] = []
    deltas: list[float] = []
    for pair in signed.values():
        if set(pair) != {-1, 1}:
            raise LTSNContractError("V3-LTE training scale found an incomplete local pair")
        minus, plus = pair[-1], pair[1]
        if not math.isclose(minus.epsilon, plus.epsilon, abs_tol=1e-9):
            raise LTSNContractError("V3-LTE training scale found mismatched epsilon")
        delta = plus.energy_target - minus.energy_target
        derivative = delta / (2.0 * minus.epsilon)
        if abs(derivative) > minimum_delta:
            derivatives.append(abs(derivative))
            deltas.append(abs(delta))
    if not derivatives or not deltas:
        raise LTSNContractError("V3-LTE training split has no nonzero local directions")
    derivative_scale = float(np.median(derivatives))
    delta_scale = float(np.median(deltas))
    if min(derivative_scale, delta_scale) <= 0:
        raise LTSNContractError("V3-LTE robust local scales are degenerate")
    return {
        "local_derivative_scale": derivative_scale,
        "local_delta_scale": delta_scale,
        "nonzero_local_directions": float(len(derivatives)),
    }


def _energy_strata(
    records: Sequence[Pitch3LTEExample],
    positive_bins: int,
    weight_cap: float,
) -> dict[str, Any]:
    values = np.asarray([record.energy_target for record in records], dtype=np.float64)
    positive = values[values > 0]
    if not len(positive):
        raise LTSNContractError("V3-LTE energy stratification found no positive targets")
    quantiles = [index / positive_bins for index in range(1, positive_bins)]
    positive_boundaries = [float(value) for value in np.quantile(positive, quantiles)]
    thresholds = [0.0, *sorted(set(positive_boundaries))]
    strata = np.searchsorted(np.asarray(thresholds), values, side="left")
    counts = np.bincount(strata, minlength=len(thresholds) + 1)
    if np.any(counts <= 0):
        raise LTSNContractError("V3-LTE energy stratification produced an empty stratum")
    raw = len(values) / (len(counts) * counts.astype(np.float64))
    capped = np.minimum(raw, weight_cap)
    normalization = float(np.sum(capped * counts) / len(values))
    weights = capped / normalization
    return {
        "thresholds": thresholds,
        "weights": [float(value) for value in weights],
        "counts": [int(value) for value in counts],
        "source": "train_split_only",
    }


def _raw_loss_kwargs(
    training: Pitch3LTETrainingConfig,
    local_scales: Mapping[str, float],
    energy_strata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "huber_delta": training.huber_delta,
        "rank_min_delta": training.rank_min_delta,
        "local_objective": training.local_objective,
        "local_derivative_scale": float(local_scales["local_derivative_scale"]),
        "local_delta_scale": float(local_scales["local_delta_scale"]),
        "local_shape_weight": training.local_shape_weight,
        "local_flat_weight": training.local_flat_weight,
        "include_cross_prompt_rank": training.cross_prompt_rank_weight > 0,
        "energy_strata_thresholds": tuple(energy_strata["thresholds"]),
        "energy_strata_weights": tuple(energy_strata["weights"]),
    }


def _loss_component_weights(
    training: Pitch3LTETrainingConfig,
    names: Sequence[str],
) -> dict[str, float]:
    configured = {
        "value": 1.0,
        "prompt_rank": 1.0,
        "cross_prompt_rank": training.cross_prompt_rank_weight,
        "local_fd": 1.0,
        "local_robust": 1.0,
        "local_direction": training.local_direction_weight,
        "local_shape": training.local_shape_weight,
        "local_flat": training.local_flat_weight,
    }
    weights = {name: float(configured[name]) for name in names}
    if any(value <= 0 for value in weights.values()):
        raise LTSNContractError("V3-LTE enabled loss component has non-positive weight")
    return weights


def _prompt_permutation(prompt_ids: Sequence[str], device: torch.device) -> Tensor | None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        grouped[str(prompt_id)].append(index)
    groups = list(grouped.values())
    if len(groups) < 2:
        return None
    permutation = list(range(len(prompt_ids)))
    for group_index, indices in enumerate(groups):
        replacement = groups[(group_index + 1) % len(groups)]
        if len(indices) != len(replacement):
            raise LTSNContractError("V3-LTE shuffled prompt groups have different sizes")
        for index, replacement_index in zip(indices, replacement, strict=True):
            permutation[index] = replacement_index
    return torch.tensor(permutation, dtype=torch.long, device=device)


def _training_forward(
    model: PromptConditionedTopologyEnergy,
    batch: Mapping[str, Any],
    training: Pitch3LTETrainingConfig,
    prompt_regularization_enabled: bool,
) -> tuple[Tensor, Tensor]:
    latent_state = model.encode_latent(batch["latent"], batch["attention_mask"])
    prompt_state = model.encode_prompt(batch["text_hidden"], batch["text_mask"])
    predicted = model.energy_from_states(latent_state, prompt_state).energy
    zero = predicted.sum() * 0.0
    if training.prompt_consistency_weight <= 0 or not prompt_regularization_enabled:
        return predicted, zero
    permutation = _prompt_permutation(batch["prompt_id"], prompt_state.device)
    if permutation is None:
        raise LTSNContractError(
            "V3.1 prompt consistency requires at least two prompt groups per batch"
        )
    shuffled = model.energy_from_states(latent_state, prompt_state[permutation]).energy
    dropout_state = prompt_state.clone()
    prompt_ids = list(dict.fromkeys(str(value) for value in batch["prompt_id"]))
    drop_group = torch.rand(len(prompt_ids), device=prompt_state.device) < float(
        training.prompt_dropout_probability
    )
    for group_index, prompt_id in enumerate(prompt_ids):
        if bool(drop_group[group_index]):
            indices = [
                index for index, value in enumerate(batch["prompt_id"]) if str(value) == prompt_id
            ]
            dropout_state[indices] = 0.0
    dropped = model.energy_from_states(latent_state, dropout_state).energy
    reference = predicted.detach()
    consistency = 0.5 * (
        F.huber_loss(shuffled, reference, delta=training.huber_delta)
        + F.huber_loss(dropped, reference, delta=training.huber_delta)
    )
    return predicted, consistency


def _set_prompt_interaction_trainable(
    model: PromptConditionedTopologyEnergy,
    enabled: bool,
) -> None:
    if model.config.fusion_mode != "latent_primary_residual_v31":
        if not enabled:
            raise LTSNContractError(
                "prompt freezing requires the V3.1+ latent-primary residual architecture"
            )
        return
    modules = (
        model.text_input_norm,
        model.text_projection,
        model.joint,
        model.interaction_energy_head,
    )
    for module in modules:
        module.requires_grad_(enabled)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return 0.0
    left, right = _rank(np.asarray(first, dtype=float)), _rank(np.asarray(second, dtype=float))
    if np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _prediction_rows(
    model: PromptConditionedTopologyEnergy,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _to_device(raw, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                predicted = model(
                    batch["latent"],
                    batch["attention_mask"],
                    batch["text_hidden"],
                    batch["text_mask"],
                ).energy
            for index, sample_id in enumerate(raw["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "prompt_id": raw["prompt_id"][index],
                        "prompt_family": raw["prompt_family"][index],
                        "source_kind": raw["source_kind"][index],
                        "direction_id": raw["direction_id"][index],
                        "direction_sign": int(raw["direction_sign"][index]),
                        "epsilon": float(raw["epsilon"][index]),
                        "exact_band": float(raw["exact_band"][index]),
                        "exact_energy": float(raw["energy_target"][index]),
                        "predicted_energy": float(predicted[index].float().cpu()),
                    }
                )
    return rows


def pitch3_lte_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = [row for row in rows if row["source_kind"] == "base_step4_seed"]
    if not base:
        raise LTSNContractError("V3-LTE metrics require base Step-4 rows")
    exact_band = np.asarray([float(row["exact_band"]) for row in base])
    predicted = np.asarray([float(row["predicted_energy"]) for row in base])
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base:
        by_prompt[str(row["prompt_id"])].append(row)
        by_family[str(row["prompt_family"])].append(row)
    collapsed_prompt_ids = []
    for prompt_id, values in by_prompt.items():
        exact_values = [float(row["exact_energy"]) for row in values]
        predicted_values = [float(row["predicted_energy"]) for row in values]
        exact_range = max(exact_values) - min(exact_values)
        predicted_range = max(predicted_values) - min(predicted_values)
        if (
            exact_range > LTE_COLLAPSE_EXACT_RANGE
            and predicted_range < LTE_COLLAPSE_PREDICTED_RANGE
        ):
            collapsed_prompt_ids.append(prompt_id)
    collapsed_fraction = len(collapsed_prompt_ids) / len(by_prompt)
    rank_correct = 0
    rank_total = 0
    for values in by_prompt.values():
        for offset, left in enumerate(values):
            for right in values[offset + 1 :]:
                exact_delta = float(left["exact_band"]) - float(right["exact_band"])
                if abs(exact_delta) <= 1e-6:
                    continue
                predicted_delta = float(left["predicted_energy"]) - float(right["predicted_energy"])
                rank_correct += int(
                    math.copysign(1, exact_delta) == math.copysign(1, predicted_delta)
                )
                rank_total += 1
    family_rho = {
        family: spearman(
            np.asarray([float(row["exact_band"]) for row in values]),
            np.asarray([float(row["predicted_energy"]) for row in values]),
        )
        for family, values in sorted(by_family.items())
    }
    local: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["direction_id"]:
            local[str(row["direction_id"])][int(row["direction_sign"])] = row
    direction_correct = 0
    direction_total = 0
    derivative_exact: list[float] = []
    derivative_predicted: list[float] = []
    for signed in local.values():
        if set(signed) != {-1, 1}:
            continue
        minus, plus = signed[-1], signed[1]
        epsilon = float(minus["epsilon"])
        exact = (float(plus["exact_energy"]) - float(minus["exact_energy"])) / (2 * epsilon)
        estimate = (float(plus["predicted_energy"]) - float(minus["predicted_energy"])) / (
            2 * epsilon
        )
        if abs(exact) <= 1e-6:
            continue
        direction_correct += int(math.copysign(1, exact) == math.copysign(1, estimate))
        direction_total += 1
        derivative_exact.append(exact)
        derivative_predicted.append(estimate)
    pooled = spearman(exact_band, predicted)
    ranking = rank_correct / rank_total if rank_total else 0.0
    direction = direction_correct / direction_total if direction_total else 0.0
    gates = {
        "direct_energy_spearman": pooled >= 0.50,
        "every_prompt_family_spearman": bool(family_rho) and min(family_rho.values()) >= 0.50,
        "same_prompt_ranking_accuracy": rank_total > 0 and ranking >= 0.65,
        "local_direction_sign_accuracy": direction_total > 0 and direction >= 0.65,
        "prompt_latent_sensitivity": collapsed_fraction <= LTE_MAX_COLLAPSED_PROMPT_FRACTION,
    }
    deficits = {
        "direct_energy_spearman": max(0.0, 0.50 - pooled),
        "every_prompt_family_spearman": max(0.0, 0.50 - min(family_rho.values(), default=0.0)),
        "same_prompt_ranking_accuracy": max(0.0, 0.65 - ranking),
        "local_direction_sign_accuracy": max(0.0, 0.65 - direction) if direction_total else 0.65,
        "prompt_latent_sensitivity": max(
            0.0, collapsed_fraction - LTE_MAX_COLLAPSED_PROMPT_FRACTION
        ),
    }
    return {
        "samples": len(rows),
        "base_samples": len(base),
        "direct_energy_spearman": pooled,
        "prompt_family_spearman": family_rho,
        "minimum_prompt_family_spearman": min(family_rho.values(), default=0.0),
        "same_prompt_rank_pairs": rank_total,
        "same_prompt_ranking_accuracy": ranking,
        "local_direction_pairs": direction_total,
        "local_direction_sign_accuracy": direction,
        "local_derivative_spearman": spearman(
            np.asarray(derivative_exact), np.asarray(derivative_predicted)
        )
        if direction_total >= 2
        else 0.0,
        "collapsed_prompt_count": len(collapsed_prompt_ids),
        "collapsed_prompt_fraction": collapsed_fraction,
        "collapsed_prompt_ids": sorted(collapsed_prompt_ids),
        "prompt_collapse_definition": {
            "maximum_predicted_range": LTE_COLLAPSE_PREDICTED_RANGE,
            "minimum_exact_range": LTE_COLLAPSE_EXACT_RANGE,
            "maximum_fraction": LTE_MAX_COLLAPSED_PROMPT_FRACTION,
        },
        "gates": gates,
        "gate_deficits": deficits,
        "total_gate_deficit": sum(deficits.values()),
        "all_gates_passed": all(gates.values()),
    }


def _loss_normalizers(
    model: PromptConditionedTopologyEnergy,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    training: Pitch3LTETrainingConfig,
    local_scales: Mapping[str, float],
    energy_strata: Mapping[str, Any],
) -> dict[str, float]:
    names = ["value", "prompt_rank"]
    if training.cross_prompt_rank_weight > 0:
        names.append("cross_prompt_rank")
    if training.local_objective == "central_difference_huber_v3":
        names.append("local_fd")
    elif training.local_objective == "robust_direction_v31":
        names.append("local_robust")
    else:
        names.extend(("local_direction", "local_shape", "local_flat"))
    values: dict[str, list[float]] = {name: [] for name in names}
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _to_device(raw, device)
            predicted = model(
                batch["latent"], batch["attention_mask"], batch["text_hidden"], batch["text_mask"]
            ).energy
            losses = pitch3_lte_raw_losses(
                predicted,
                batch,
                **_raw_loss_kwargs(training, local_scales, energy_strata),
            )
            for name, loss in losses.items():
                value = float(loss.detach().cpu())
                if math.isfinite(value) and value > 0:
                    values[name].append(value)
    medians = {name: float(np.median(items)) if items else 0.0 for name, items in values.items()}
    if any(value <= 0 or not math.isfinite(value) for value in medians.values()):
        raise LTSNContractError("V3-LTE first-epoch loss normalization found an empty component")
    return medians


def _normalized_dataset_loss(
    model: PromptConditionedTopologyEnergy,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    training: Pitch3LTETrainingConfig,
    normalizers: Mapping[str, float],
    local_scales: Mapping[str, float],
    energy_strata: Mapping[str, Any],
    component_weights: Mapping[str, float],
) -> float:
    totals: list[float] = []
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _to_device(raw, device)
            predicted = model(
                batch["latent"],
                batch["attention_mask"],
                batch["text_hidden"],
                batch["text_mask"],
            ).energy
            losses = pitch3_lte_raw_losses(
                predicted,
                batch,
                **_raw_loss_kwargs(training, local_scales, energy_strata),
            )
            totals.append(
                sum(
                    component_weights[name] * float(losses[name].cpu()) / normalizers[name]
                    for name in normalizers
                )
            )
    return float(np.mean(totals)) if totals else float("inf")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pitch3_lte_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_sha256: str | None = None,
) -> tuple[PromptConditionedTopologyEnergy, dict[str, Any]]:
    if expected_sha256 is not None and sha256_file(checkpoint_path) != expected_sha256.lower():
        raise LTSNContractError("V3-LTE checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    if metadata.get("model_family") != LTE_MODEL_FAMILY or metadata.get("guidance_steps") != [4]:
        raise LTSNContractError("invalid V3-LTE checkpoint contract")
    model = PromptConditionedTopologyEnergy(Pitch3LTEConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval().requires_grad_(False)
    return model, metadata


def train_pitch3_lte(
    *,
    fingerprint_path: Path,
    dataset_manifest: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    dataset_manifest = dataset_manifest.resolve()
    contract = load_pitch3_contract(fingerprint_path)
    model_config, training = load_pitch3_lte_config(config_path)
    dataset_summary_path = dataset_manifest.parent / "pitch3_lte_dataset_summary.json"
    if dataset_summary_path.is_file():
        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        dataset_preflight_source = "published_dataset_summary"
    else:
        dataset_summary = validate_pitch3_lte_dataset_preflight(dataset_manifest)
        dataset_preflight_source = str(dataset_summary["preflight_source"])
    if dataset_summary.get("local_preflight_passed") is not True or dataset_summary.get(
        "dataset_manifest_sha256"
    ) != sha256_file(dataset_manifest):
        raise LTSNContractError("V3-LTE dataset failed or is detached from its preflight")
    records = _read_examples(dataset_manifest, contract.artifact_sha256)
    train_records = [record for record in records if record.split == "train"]
    development_records = [record for record in records if record.split == "development"]
    if not train_records or not development_records:
        raise LTSNContractError("V3-LTE requires isolated train and development splits")
    prompt_counts = {
        "train": len({record.prompt_id for record in train_records}),
        "development": len({record.prompt_id for record in development_records}),
    }
    if prompt_counts != {"train": 320, "development": 64}:
        raise LTSNContractError(
            "formal V3-LTE training requires 320 train and 64 development prompts"
        )
    if not any(record.source_kind == "local_finite_difference" for record in train_records):
        raise LTSNContractError("V3-LTE training split has no exact local finite differences")
    if not any(record.source_kind == "local_finite_difference" for record in development_records):
        raise LTSNContractError("V3-LTE checkpoint selection requires held-out local pairs")
    local_scales = _local_training_scales(train_records, training.rank_min_delta)
    energy_strata = _energy_strata(
        train_records,
        training.energy_strata_positive_bins,
        training.energy_strata_weight_cap,
    )
    _seed_everything(training.seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for V3-LTE but is unavailable")
    model = PromptConditionedTopologyEnergy(model_config).to(device)
    _set_prompt_interaction_trainable(model, training.prompt_frozen_epochs == 0)
    train_sampler = PromptBatchSampler(
        train_records,
        training.seed,
        True,
        groups_per_batch=training.prompt_groups_per_batch,
    )
    development_sampler = PromptBatchSampler(development_records, training.seed, False)
    train_loader = DataLoader(
        Pitch3LTEDataset(train_records),
        batch_sampler=train_sampler,
        collate_fn=collate_pitch3_lte,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
    )
    development_loader = DataLoader(
        Pitch3LTEDataset(development_records),
        batch_sampler=development_sampler,
        collate_fn=collate_pitch3_lte,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
    )
    normalizers = _loss_normalizers(
        model,
        train_loader,
        device,
        training,
        local_scales,
        energy_strata,
    )
    component_weights = _loss_component_weights(training, tuple(normalizers))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"pitch3_lte_seed_{training.seed}.pt"
    candidate_paths = {
        "gate_deficit": checkpoint_path,
        "global_energy": output_dir / f"pitch3_lte_seed_{training.seed}_best_energy.pt",
        "local_direction": output_dir / f"pitch3_lte_seed_{training.seed}_best_direction.pt",
    }
    candidate_keys: dict[str, tuple[float, ...]] = {
        name: (float("inf"),) for name in candidate_paths
    }
    candidate_records: dict[str, dict[str, Any]] = {}
    patience = 0
    history: list[dict[str, Any]] = []
    metadata_base = {
        "schema_version": 1,
        "model_family": LTE_MODEL_FAMILY,
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_config_sha256": sha256_file(config_path),
        "training_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_plan_sha256": dataset_summary.get("dataset_plan_sha256"),
        "dataset_preflight_source": dataset_preflight_source,
        "seed": training.seed,
        "device": device_name,
        "precision": "bf16_forward_fp32_loss" if training.use_bf16 else "fp32",
        "architecture_revision": (
            "v3.2_dual_rms_latent_primary_prompt_residual"
            if model_config.latent_stem_mode == "dual_rms_v32"
            else (
                "v3.1_latent_primary_prompt_residual"
                if model_config.fusion_mode == "latent_primary_residual_v31"
                else "v3_joint_fusion"
            )
        ),
        "energy_target": "log1p(exact_pitch3_target_band_loss)",
        "prompt_condition": "frozen_ace_step_text_hidden_state",
        "prompt_id_embedding_used": False,
        "guidance_steps": [4],
        "training_radius_ratio": LTE_RADIUS_RATIO,
        "maximum_guidance_update_ratio": LTE_RADIUS_RATIO / 2.0,
        "loss_components": list(normalizers),
        "loss_component_weights": component_weights,
        "local_objective": training.local_objective,
        "local_training_scales": local_scales,
        "energy_stratification": energy_strata,
        "prompt_regularization": {
            "groups_per_batch": training.prompt_groups_per_batch,
            "dropout_probability": training.prompt_dropout_probability,
            "consistency_weight": training.prompt_consistency_weight,
            "frozen_epochs": training.prompt_frozen_epochs,
        },
        "prompt_collapse_gate": {
            "maximum_predicted_range": LTE_COLLAPSE_PREDICTED_RANGE,
            "minimum_exact_range": LTE_COLLAPSE_EXACT_RANGE,
            "maximum_fraction": LTE_MAX_COLLAPSED_PROMPT_FRACTION,
        },
        "first_epoch_loss_medians": normalizers,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }

    def save_candidate(
        name: str,
        key: tuple[float, ...],
        epoch: int,
        development_loss: float,
        metrics: Mapping[str, Any],
    ) -> None:
        metadata = {
            **metadata_base,
            "best_epoch": epoch,
            "best_development_gate_deficit": float(metrics["total_gate_deficit"]),
            "best_development_loss": development_loss,
            "best_development_metrics": dict(metrics),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_candidate_kind": name,
            "checkpoint_candidate_key": list(key),
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "training_config": asdict(training),
                "metadata": metadata,
            },
            candidate_paths[name],
        )
        candidate_keys[name] = key
        candidate_records[name] = {
            "epoch": epoch,
            "selection_key": list(key),
            "development_gate_deficit": float(metrics["total_gate_deficit"]),
            "all_gates_passed": bool(metrics["all_gates_passed"]),
        }

    for epoch in range(1, training.max_epochs + 1):
        train_sampler.set_epoch(epoch)
        prompt_interaction_enabled = epoch > training.prompt_frozen_epochs
        _set_prompt_interaction_trainable(model, prompt_interaction_enabled)
        model.train()
        sums = {name: 0.0 for name in normalizers}
        sums.update({"prompt_consistency": 0.0, "total": 0.0})
        batches = 0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=training.use_bf16 and device.type == "cuda",
            ):
                predicted, prompt_consistency = _training_forward(
                    model,
                    batch,
                    training,
                    prompt_regularization_enabled=prompt_interaction_enabled,
                )
            raw_losses = pitch3_lte_raw_losses(
                predicted.float(),
                batch,
                **_raw_loss_kwargs(training, local_scales, energy_strata),
            )
            total = sum(
                component_weights[name] * raw_losses[name] / normalizers[name]
                for name in normalizers
            )
            total = total + (
                training.prompt_consistency_weight
                * prompt_consistency.float()
                / normalizers["value"]
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()
            for name, loss in raw_losses.items():
                sums[name] += float(loss.detach().cpu())
            sums["prompt_consistency"] += float(prompt_consistency.detach().cpu())
            sums["total"] += float(total.detach().cpu())
            batches += 1
        development_rows = _prediction_rows(model, development_loader, device, use_bf16=False)
        metrics = pitch3_lte_metrics(development_rows)
        development_loss = _normalized_dataset_loss(
            model,
            development_loader,
            device,
            training,
            normalizers,
            local_scales,
            energy_strata,
            component_weights,
        )
        epoch_losses = {name: value / max(batches, 1) for name, value in sums.items()}
        selection = (float(metrics["total_gate_deficit"]), development_loss)
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_losses,
                "development_normalized_loss": development_loss,
                "development_metrics": metrics,
                "selection_key": list(selection),
                "prompt_interaction_enabled": prompt_interaction_enabled,
            }
        )
        energy_selection = (
            -float(metrics["direct_energy_spearman"]),
            -float(metrics["minimum_prompt_family_spearman"]),
            development_loss,
        )
        direction_selection = (
            -float(metrics["local_direction_sign_accuracy"]),
            -float(metrics["local_derivative_spearman"]),
            development_loss,
        )
        if selection < candidate_keys["gate_deficit"]:
            patience = 0
            save_candidate("gate_deficit", selection, epoch, development_loss, metrics)
        else:
            patience += 1
        if energy_selection < candidate_keys["global_energy"]:
            save_candidate(
                "global_energy", energy_selection, epoch, development_loss, metrics
            )
        if direction_selection < candidate_keys["local_direction"]:
            save_candidate(
                "local_direction", direction_selection, epoch, development_loss, metrics
            )
        if epoch >= training.minimum_epochs and patience >= training.early_stopping_patience:
            break
    _, best_metadata = load_pitch3_lte_checkpoint(checkpoint_path, device=device)
    for name, path in candidate_paths.items():
        candidate_records[name].update(
            {
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": sha256_file(path),
            }
        )
    manifest = {
        **best_metadata,
        "epochs_completed": len(history),
        "checkpoint_selection": "fp32_development_gate_deficit_then_normalized_training_loss",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_candidates": candidate_records,
        "training_history": history,
    }
    manifest_path = output_dir / "pitch3_lte_manifest.json"
    write_json_atomic(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
