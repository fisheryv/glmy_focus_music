"""Training pipeline for the three-coordinate latent topology control head."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tomllib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .latent_topology_control_head import (
    PITCH3_OOD_STAT_FEATURES,
    LatentTopologyControlHead,
    Pitch3ControlHeadConfig,
    Pitch3ControlOutput,
)
from .ltsn_contract import LTSNContractError, sha256_file
from .pitch3_contract import PITCH3_DIMENSIONS, Pitch3Contract, load_pitch3_contract

SPLITS = {"train", "development", "calibration", "qualification"}
CORRECTION_STEPS = (4, 5, 6)
FORMAL_STEPS = (4, 5, 6, 8)


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
    id_samples_per_batch: int = 0
    ood_samples_per_batch: int = 0
    gate_aligned_selection: bool = False
    id_band_stratified: bool = False
    group_robust_sampling: bool = False
    group_robust_selection: bool = False
    group_robust_sampler_kind: str = "trajectory_anchor_v26"
    pooled_band_spearman_floor: float = 0.0
    scale_high_training_target_scales: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Pitch3LossWeights:
    coordinate: float = 1.0
    nll: float = 0.25
    focus: float = 0.25
    ood: float = 0.1
    band: float = 0.0
    band_rank: float = 0.0
    band_rank_min_delta: float = 1e-4
    local_band_rank: float = 0.0
    coordinate2_rank: float = 0.0
    coordinate2_rank_min_delta: float = 1e-4
    trajectory_delta: float = 0.0
    band_region: float = 0.0
    band_zero: float = 0.0
    band_region_positive_only: bool = False
    band_smooth_temperature_fraction: float = 0.0
    ood_margin: float = 0.0
    ood_margin_value: float = 1.0


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
    ood_kind: str
    ood_transform_version: str
    ood_label_source: str


def load_pitch3_training_config(
    path: Path,
) -> tuple[Pitch3ControlHeadConfig, Pitch3TrainingConfig, Pitch3LossWeights]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model = Pitch3ControlHeadConfig(**payload.get("model", {}))
    training_payload = dict(payload.get("training", {}))
    if "scale_high_training_target_scales" in training_payload:
        training_payload["scale_high_training_target_scales"] = tuple(
            float(value) for value in training_payload["scale_high_training_target_scales"]
        )
    training = Pitch3TrainingConfig(**training_payload)
    loss = Pitch3LossWeights(**payload.get("loss", {}))
    if training.micro_batch_size < 1 or training.max_epochs < 1:
        raise ValueError("training batch size and epochs must be positive")
    if training.minimum_epochs < 1 or training.minimum_epochs > training.max_epochs:
        raise ValueError("minimum_epochs must be in [1,max_epochs]")
    if training.early_stopping_patience < 1 or training.gradient_clip_norm <= 0.0:
        raise ValueError("early stopping patience and gradient clip must be positive")
    balanced = training.id_samples_per_batch + training.ood_samples_per_batch
    if balanced not in {0, training.micro_batch_size}:
        raise ValueError("balanced Pitch-3 batch counts must sum to micro_batch_size")
    if (training.id_samples_per_batch == 0) != (training.ood_samples_per_batch == 0):
        raise ValueError("balanced Pitch-3 batches require positive ID and OOD counts")
    if training.id_band_stratified and (
        training.id_samples_per_batch < 3 or training.id_samples_per_batch % 3
    ):
        raise ValueError("band-stratified ID batches require an ID count divisible by three")
    if training.group_robust_sampling and (
        training.id_samples_per_batch != 6 or training.ood_samples_per_batch != 2
    ):
        raise ValueError("group-robust Pitch-3 sampling requires a 6 ID / 2 OOD batch")
    if training.group_robust_selection and not training.gate_aligned_selection:
        raise ValueError("group-robust selection requires gate-aligned selection")
    sampler_kinds = {"trajectory_anchor_v26", "snapshot_pair_v26_grr"}
    if training.group_robust_sampler_kind not in sampler_kinds:
        raise ValueError(
            "group_robust_sampler_kind must be trajectory_anchor_v26 or snapshot_pair_v26_grr"
        )
    if not training.group_robust_sampling and (
        training.group_robust_sampler_kind != "trajectory_anchor_v26"
    ):
        raise ValueError("a non-default group sampler requires group-robust sampling")
    if (
        training.group_robust_sampler_kind == "snapshot_pair_v26_grr"
        and not training.id_band_stratified
    ):
        raise ValueError("the snapshot-pair group sampler requires ID Band stratification")
    if not math.isfinite(training.pooled_band_spearman_floor) or not (
        0.0 <= training.pooled_band_spearman_floor <= 1.0
    ):
        raise ValueError("pooled_band_spearman_floor must be finite and in [0,1]")
    if training.pooled_band_spearman_floor > 0.0 and not training.group_robust_selection:
        raise ValueError("a pooled Band floor requires group-robust selection")
    if (
        any(not math.isfinite(value) or value < 0.0 for value in asdict(loss).values())
        or loss.coordinate <= 0.0
    ):
        raise ValueError("loss settings must be finite and non-negative with coordinate > 0")
    if loss.ood_margin > 0.0 and loss.ood_margin_value <= 0.0:
        raise ValueError("positive OOD margin loss requires ood_margin_value > 0")
    if (loss.band_rank > 0.0 or loss.local_band_rank > 0.0) and (loss.band_rank_min_delta <= 0.0):
        raise ValueError("positive band ranking loss requires band_rank_min_delta > 0")
    if loss.coordinate2_rank > 0.0 and loss.coordinate2_rank_min_delta <= 0.0:
        raise ValueError("positive coordinate-2 ranking loss requires a positive minimum delta")
    group_loss_enabled = (
        loss.local_band_rank > 0.0 or loss.coordinate2_rank > 0.0 or loss.trajectory_delta > 0.0
    )
    if training.group_robust_sampling != group_loss_enabled:
        raise ValueError("group-robust sampling and group-robust losses must be enabled together")
    if training.group_robust_selection and not training.group_robust_sampling:
        raise ValueError("group-robust selection requires group-robust sampling")
    if loss.band_smooth_temperature_fraction > 0.5:
        raise ValueError("band smoothing temperature fraction must be in [0,0.5]")
    if loss.band_smooth_temperature_fraction > 0.0 and not (
        loss.band > 0.0 or loss.band_rank > 0.0 or loss.local_band_rank > 0.0
    ):
        raise ValueError("band smoothing requires a positive band or band-rank weight")
    if loss.band_region_positive_only and loss.band_region <= 0.0:
        raise ValueError("positive-only band regions require a positive band-region weight")
    scales = training.scale_high_training_target_scales
    if scales and (
        len(set(scales)) != len(scales)
        or any(not math.isfinite(value) or value <= 1.0 or value > 4.0 for value in scales)
    ):
        raise ValueError("training scale targets must be unique finite values in (1,4]")
    return model, training, loss


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def pitch3_prompt_family(prompt_id: str) -> str:
    """Return the frozen base-family identifier without the prompt variant suffix."""

    marker = prompt_id.rfind("__v")
    if marker <= 0 or not prompt_id[marker + 3 :].isdigit():
        return prompt_id
    return prompt_id[:marker]


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
            ood_kind = raw.get("ood_kind", "").strip()
            ood_transform_version = raw.get("ood_transform_version", "").strip()
            ood_label_source = raw.get("ood_label_source", "").strip()
            if ood >= 0.5 and not all((ood_kind, ood_transform_version, ood_label_source)):
                raise LTSNContractError("Pitch-3 OOD rows require transform provenance")
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
                    ood_kind=ood_kind,
                    ood_transform_version=ood_transform_version,
                    ood_label_source=ood_label_source,
                )
            )
    if not records:
        raise LTSNContractError("Pitch-3 training manifest is empty")
    if any(len(values) != 1 for values in prompt_splits.values()):
        raise LTSNContractError("Pitch-3 prompt leakage detected")
    if any(len(values) != 1 for values in trajectory_splits.values()):
        raise LTSNContractError("Pitch-3 trajectory leakage detected")
    return records


def _virtual_scale_assignments(
    records: list[Pitch3Snapshot], targets: tuple[float, ...]
) -> dict[str, float]:
    eligible = [
        record
        for record in records
        if targets and record.ood_label >= 0.5 and record.ood_kind == "ood_scale_high"
    ]
    eligible.sort(
        key=lambda record: hashlib.sha256(f"pitch3-scale-v22|{record.sample_id}".encode()).digest()
    )
    return {
        record.sample_id: targets[index % len(targets)] for index, record in enumerate(eligible)
    }


def _virtual_scale_summary(
    records: list[Pitch3Snapshot], targets: tuple[float, ...]
) -> dict[str, Any]:
    scale_assignments = _virtual_scale_assignments(records, targets)
    assignments = [
        {
            "sample_id": record.sample_id,
            "source_latent_sha256": record.latent_sha256,
            "source_scale": 4.0,
            "target_scale": scale_assignments[record.sample_id],
        }
        for record in records
        if targets and record.ood_label >= 0.5 and record.ood_kind == "ood_scale_high"
    ]
    encoded = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode("utf-8")
    counts = Counter(item["target_scale"] for item in assignments)
    return {
        "schema_version": 1,
        "kind": "hash_deterministic_scale_high_rescaling",
        "assignment_salt": "pitch3-scale-v22",
        "source_kind": "ood_scale_high",
        "source_scale": 4.0,
        "target_scales": list(targets),
        "assignment_count": len(assignments),
        "assignment_counts_by_target": {str(scale): counts[scale] for scale in sorted(counts)},
        "assignment_sha256": hashlib.sha256(encoded).hexdigest(),
        "coordinate_targets_used": False,
    }


class Pitch3Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[Pitch3Snapshot],
        *,
        scale_high_training_target_scales: tuple[float, ...] = (),
    ) -> None:
        self.records = records
        self.scale_high_training_target_scales = scale_high_training_target_scales
        self.virtual_scale_assignments = _virtual_scale_assignments(
            records, scale_high_training_target_scales
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latent = np.load(record.latent_path, allow_pickle=False)
        if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
            raise LTSNContractError(f"invalid Pitch-3 latent array: {record.latent_path}")
        virtual_scale = 0.0
        if (
            self.scale_high_training_target_scales
            and record.split == "train"
            and record.ood_label >= 0.5
            and record.ood_kind == "ood_scale_high"
        ):
            virtual_scale = self.virtual_scale_assignments[record.sample_id]
            latent = np.asarray(latent, dtype=np.float32) * np.float32(virtual_scale / 4.0)
        return {
            "sample_id": record.sample_id,
            "prompt_id": record.prompt_id,
            "prompt_family": pitch3_prompt_family(record.prompt_id),
            "trajectory_id": record.trajectory_id,
            "latent": torch.from_numpy(np.asarray(latent, dtype=np.float32)),
            "timestep": torch.tensor(record.timestep, dtype=torch.float32),
            "step_number": torch.tensor(record.step_number, dtype=torch.float32),
            "coordinates": torch.tensor(record.coordinates, dtype=torch.float32),
            "focus_logit": torch.tensor(record.focus_logit, dtype=torch.float32),
            "ood_label": torch.tensor(record.ood_label, dtype=torch.float32),
            "ood_kind": record.ood_kind,
            "virtual_ood_scale": virtual_scale,
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
        "prompt_id": [item["prompt_id"] for item in items],
        "prompt_family": [item["prompt_family"] for item in items],
        "trajectory_id": [item["trajectory_id"] for item in items],
        "latent": latent,
        "attention_mask": mask,
        "timestep": torch.stack([item["timestep"] for item in items]),
        "step_number": torch.stack([item["step_number"] for item in items]),
        "coordinates": torch.stack([item["coordinates"] for item in items]),
        "focus_logit": torch.stack([item["focus_logit"] for item in items]),
        "ood_label": torch.stack([item["ood_label"] for item in items]),
        "ood_kind": [item["ood_kind"] for item in items],
        "virtual_ood_scale": [item["virtual_ood_scale"] for item in items],
    }


class Pitch3BalancedBatchSampler(Sampler[list[int]]):
    """Deterministic 6:2-style ID/OOD batches with OOD-family rotation."""

    def __init__(
        self,
        records: list[Pitch3Snapshot],
        *,
        id_per_batch: int,
        ood_per_batch: int,
        seed: int,
        contract: Pitch3Contract | None = None,
        id_band_stratified: bool = False,
    ) -> None:
        if id_per_batch < 1 or ood_per_batch < 1:
            raise ValueError("balanced batch counts must be positive")
        self.id_indices = [index for index, record in enumerate(records) if record.ood_label < 0.5]
        by_kind: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if record.ood_label >= 0.5:
                by_kind.setdefault(record.ood_kind, []).append(index)
        if not self.id_indices or not by_kind or any(not values for values in by_kind.values()):
            raise LTSNContractError("balanced Pitch-3 batches require ID and OOD families")
        self.ood_by_kind = dict(sorted(by_kind.items()))
        self.id_per_batch = id_per_batch
        self.ood_per_batch = ood_per_batch
        self.seed = seed
        self.epoch = 0
        self.id_by_stratum: dict[str, list[int]] = {}
        self.id_strata_summary: dict[str, Any] = {}
        if id_band_stratified:
            if contract is None or id_per_batch % 3:
                raise ValueError("band-stratified batches require a contract and 3-way ID count")
            if len(self.id_indices) < 3:
                raise LTSNContractError("band-stratified batches require at least three ID samples")
            lower = np.asarray(contract.target_lower, dtype=float)
            upper = np.asarray(contract.target_upper, dtype=float)
            weights = np.asarray(contract.distance_weights, dtype=float)

            def band_loss(index: int) -> float:
                coordinate = np.asarray(records[index].coordinates, dtype=float)
                below = np.maximum(lower - coordinate, 0.0)
                above = np.maximum(coordinate - upper, 0.0)
                return float(((np.square(below) + np.square(above)) * weights).sum())

            ordered = sorted(
                self.id_indices,
                key=lambda index: (
                    band_loss(index),
                    hashlib.sha256(records[index].sample_id.encode("utf-8")).digest(),
                ),
            )
            chunks = np.array_split(np.asarray(ordered, dtype=int), 3)
            names = ("low", "middle", "high")
            self.id_by_stratum = {
                name: [int(value) for value in chunk]
                for name, chunk in zip(names, chunks, strict=True)
            }
            self.id_strata_summary = {
                name: {
                    "samples": len(indices),
                    "minimum_band_loss": min(band_loss(index) for index in indices),
                    "maximum_band_loss": max(band_loss(index) for index in indices),
                }
                for name, indices in self.id_by_stratum.items()
            }
        self.batch_count = max(
            math.ceil(len(self.id_indices) / id_per_batch),
            math.ceil(sum(len(values) for values in by_kind.values()) / ood_per_batch),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _cycle(values: list[int], count: int, rng: random.Random) -> list[int]:
        result: list[int] = []
        pool: list[int] = []
        while len(result) < count:
            if not pool:
                pool = values.copy()
                rng.shuffle(pool)
            take = min(count - len(result), len(pool))
            result.extend(pool[:take])
            del pool[:take]
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        id_stream = (
            self._cycle(self.id_indices, self.batch_count * self.id_per_batch, rng)
            if not self.id_by_stratum
            else []
        )
        id_streams = (
            {
                name: self._cycle(
                    indices,
                    self.batch_count * (self.id_per_batch // len(self.id_by_stratum)),
                    rng,
                )
                for name, indices in self.id_by_stratum.items()
            }
            if self.id_by_stratum
            else {}
        )
        kind_names = list(self.ood_by_kind)
        needed_by_kind = {name: 0 for name in kind_names}
        assignments: list[str] = []
        for index in range(self.batch_count * self.ood_per_batch):
            name = kind_names[index % len(kind_names)]
            assignments.append(name)
            needed_by_kind[name] += 1
        ood_streams = {
            name: self._cycle(self.ood_by_kind[name], needed, rng)
            for name, needed in needed_by_kind.items()
        }
        ood_cursors = {name: 0 for name in kind_names}
        for batch_index in range(self.batch_count):
            start = batch_index * self.id_per_batch
            if id_streams:
                per_stratum = self.id_per_batch // len(id_streams)
                stratum_start = batch_index * per_stratum
                batch = [
                    index
                    for stream in id_streams.values()
                    for index in stream[stratum_start : stratum_start + per_stratum]
                ]
            else:
                batch = id_stream[start : start + self.id_per_batch]
            for offset in range(self.ood_per_batch):
                name = assignments[batch_index * self.ood_per_batch + offset]
                batch.append(ood_streams[name][ood_cursors[name]])
                ood_cursors[name] += 1
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batch_count


class Pitch3GroupRobustBatchSampler(Sampler[list[int]]):
    """Build 6:2 batches around one prompt-family trajectory anchor.

    Four ID rows are the complete formal trajectory. Two companion ID rows
    come from other trajectories in the same prompt family at rotating steps.
    This makes within-family/step ranking and within-trajectory delta losses
    observable while retaining deterministic OOD-family rotation.
    """

    _COMPANION_STEP_PAIRS = ((4, 5), (6, 8), (4, 6), (5, 8))

    def __init__(
        self,
        records: list[Pitch3Snapshot],
        *,
        id_per_batch: int,
        ood_per_batch: int,
        seed: int,
        contract: Pitch3Contract,
    ) -> None:
        if id_per_batch != 6 or ood_per_batch != 2:
            raise ValueError("Pitch-3 group-robust batches require exactly 6 ID and 2 OOD rows")
        self.records = records
        self.id_per_batch = id_per_batch
        self.ood_per_batch = ood_per_batch
        self.seed = seed
        self.epoch = 0
        by_trajectory: dict[str, list[int]] = {}
        by_kind: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if record.ood_label < 0.5:
                by_trajectory.setdefault(record.trajectory_id, []).append(index)
            else:
                by_kind.setdefault(record.ood_kind, []).append(index)
        if not by_trajectory or not by_kind or any(not values for values in by_kind.values()):
            raise LTSNContractError("group-robust batches require ID trajectories and OOD families")
        self.ood_by_kind = dict(sorted(by_kind.items()))
        self.trajectory_indices: dict[str, tuple[int, int, int, int]] = {}
        self.trajectory_family: dict[str, str] = {}
        for trajectory_id, indices in by_trajectory.items():
            by_step = {records[index].step_number: index for index in indices}
            families = {pitch3_prompt_family(records[index].prompt_id) for index in indices}
            if set(by_step) != set(FORMAL_STEPS) or len(indices) != len(FORMAL_STEPS):
                raise LTSNContractError(
                    "group-robust training requires one ID row per formal step in every trajectory"
                )
            if len(families) != 1:
                raise LTSNContractError("one trajectory cannot span multiple prompt families")
            self.trajectory_indices[trajectory_id] = tuple(by_step[step] for step in FORMAL_STEPS)
            self.trajectory_family[trajectory_id] = next(iter(families))
        self.families = sorted(set(self.trajectory_family.values()))
        self.companions: dict[tuple[str, int], list[int]] = {}
        for trajectory_id, indices in self.trajectory_indices.items():
            family = self.trajectory_family[trajectory_id]
            for index in indices:
                self.companions.setdefault((family, records[index].step_number), []).append(index)
        for family in self.families:
            for step in FORMAL_STEPS:
                trajectories = {
                    records[index].trajectory_id
                    for index in self.companions.get((family, step), [])
                }
                if len(trajectories) < 2:
                    raise LTSNContractError(
                        "group-robust training needs two trajectories per family and formal step"
                    )

        lower = np.asarray(contract.target_lower, dtype=float)
        upper = np.asarray(contract.target_upper, dtype=float)
        weights = np.asarray(contract.distance_weights, dtype=float)

        def trajectory_band_loss(trajectory_id: str) -> float:
            values = []
            for index in self.trajectory_indices[trajectory_id]:
                coordinate = np.asarray(records[index].coordinates, dtype=float)
                below = np.maximum(lower - coordinate, 0.0)
                above = np.maximum(coordinate - upper, 0.0)
                values.append(float(((np.square(below) + np.square(above)) * weights).sum()))
            return float(np.mean(values))

        ordered = sorted(
            self.trajectory_indices,
            key=lambda trajectory_id: (
                trajectory_band_loss(trajectory_id),
                hashlib.sha256(trajectory_id.encode("utf-8")).digest(),
            ),
        )
        chunks = np.array_split(np.asarray(ordered, dtype=object), 3)
        self.trajectory_stratum: dict[str, str] = {}
        self.trajectories_by_family_stratum: dict[tuple[str, str], list[str]] = {}
        self.trajectory_strata_summary: dict[str, Any] = {}
        for name, chunk in zip(("low", "middle", "high"), chunks, strict=True):
            trajectory_ids = [str(value) for value in chunk]
            for trajectory_id in trajectory_ids:
                self.trajectory_stratum[trajectory_id] = name
                key = (self.trajectory_family[trajectory_id], name)
                self.trajectories_by_family_stratum.setdefault(key, []).append(trajectory_id)
            self.trajectory_strata_summary[name] = {
                "trajectories": len(trajectory_ids),
                "minimum_mean_band_loss": min(
                    trajectory_band_loss(value) for value in trajectory_ids
                ),
                "maximum_mean_band_loss": max(
                    trajectory_band_loss(value) for value in trajectory_ids
                ),
            }
        self.batch_count = max(
            math.ceil(len(by_trajectory) * len(FORMAL_STEPS) / id_per_batch),
            math.ceil(sum(len(values) for values in by_kind.values()) / ood_per_batch),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _take(pools: dict[Any, list[Any]], key: Any, values: list[Any], rng: random.Random) -> Any:
        if not pools.get(key):
            pools[key] = values.copy()
            rng.shuffle(pools[key])
        return pools[key].pop()

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pools: dict[Any, list[Any]] = {}
        family_stream = Pitch3BalancedBatchSampler._cycle(
            list(range(len(self.families))), self.batch_count, rng
        )
        family_counts = Counter()
        kind_names = list(self.ood_by_kind)
        ood_assignments = [
            kind_names[index % len(kind_names)]
            for index in range(self.batch_count * self.ood_per_batch)
        ]
        rng.shuffle(ood_assignments)
        for batch_index, family_index in enumerate(family_stream):
            family = self.families[family_index]
            family_count = family_counts[family]
            family_counts[family] += 1
            desired_stratum = ("low", "middle", "high")[family_count % 3]
            available_strata = [
                name
                for name in (desired_stratum, "low", "middle", "high")
                if self.trajectories_by_family_stratum.get((family, name))
            ]
            stratum = available_strata[0]
            trajectory_values = self.trajectories_by_family_stratum[(family, stratum)]
            trajectory_id = self._take(
                pools, ("trajectory", family, stratum), trajectory_values, rng
            )
            batch = list(self.trajectory_indices[trajectory_id])
            steps = self._COMPANION_STEP_PAIRS[family_count % len(self._COMPANION_STEP_PAIRS)]
            for step in steps:
                candidates = [
                    index
                    for index in self.companions[(family, step)]
                    if self.records[index].trajectory_id != trajectory_id
                ]
                batch.append(
                    self._take(pools, ("companion", family, step, trajectory_id), candidates, rng)
                )
            for offset in range(self.ood_per_batch):
                kind = ood_assignments[batch_index * self.ood_per_batch + offset]
                batch.append(self._take(pools, ("ood", kind), self.ood_by_kind[kind], rng))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batch_count


class Pitch3SnapshotPairBatchSampler(Sampler[list[int]]):
    """Build family-aware 6:2 batches with frozen snapshot-level Band balance.

    Each batch contains two low-, two middle-, and two high-Band ID snapshots.
    The six snapshots form three same-family/same-step comparison pairs with
    low-high, low-middle, and middle-high contrasts.  Three of the four formal
    steps are used per batch and the omitted step and pair assignment rotate
    deterministically across batches for each prompt family.
    """

    _PAIR_PATTERNS = (("low", "high"), ("low", "middle"), ("middle", "high"))

    def __init__(
        self,
        records: list[Pitch3Snapshot],
        *,
        id_per_batch: int,
        ood_per_batch: int,
        seed: int,
        contract: Pitch3Contract,
    ) -> None:
        if id_per_batch != 6 or ood_per_batch != 2:
            raise ValueError("Pitch-3 snapshot-pair batches require exactly 6 ID and 2 OOD rows")
        self.records = records
        self.id_per_batch = id_per_batch
        self.ood_per_batch = ood_per_batch
        self.seed = seed
        self.epoch = 0
        self.id_indices = [index for index, record in enumerate(records) if record.ood_label < 0.5]
        by_kind: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if record.ood_label >= 0.5:
                by_kind.setdefault(record.ood_kind, []).append(index)
        if not self.id_indices or not by_kind or any(not values for values in by_kind.values()):
            raise LTSNContractError("snapshot-pair batches require ID samples and OOD families")
        self.ood_by_kind = dict(sorted(by_kind.items()))

        lower = np.asarray(contract.target_lower, dtype=float)
        upper = np.asarray(contract.target_upper, dtype=float)
        weights = np.asarray(contract.distance_weights, dtype=float)

        def band_loss(index: int) -> float:
            coordinate = np.asarray(records[index].coordinates, dtype=float)
            below = np.maximum(lower - coordinate, 0.0)
            above = np.maximum(coordinate - upper, 0.0)
            return float(((np.square(below) + np.square(above)) * weights).sum())

        ordered = sorted(
            self.id_indices,
            key=lambda index: (
                band_loss(index),
                hashlib.sha256(records[index].sample_id.encode("utf-8")).digest(),
            ),
        )
        chunks = np.array_split(np.asarray(ordered, dtype=int), 3)
        self.id_by_stratum = {
            name: [int(value) for value in chunk]
            for name, chunk in zip(("low", "middle", "high"), chunks, strict=True)
        }
        self.id_stratum_by_index = {
            index: name for name, indices in self.id_by_stratum.items() for index in indices
        }
        self.id_strata_summary = {
            name: {
                "samples": len(indices),
                "minimum_band_loss": min(band_loss(index) for index in indices),
                "maximum_band_loss": max(band_loss(index) for index in indices),
            }
            for name, indices in self.id_by_stratum.items()
        }
        self.snapshots: dict[tuple[str, int, str], list[int]] = {}
        for index in self.id_indices:
            record = records[index]
            family = pitch3_prompt_family(record.prompt_id)
            stratum = self.id_stratum_by_index[index]
            self.snapshots.setdefault((family, record.step_number, stratum), []).append(index)
        self.families = sorted({key[0] for key in self.snapshots})
        for family in self.families:
            for step in FORMAL_STEPS:
                for first_stratum, second_stratum in self._PAIR_PATTERNS:
                    first = self.snapshots.get((family, step, first_stratum), [])
                    second = self.snapshots.get((family, step, second_stratum), [])
                    if (
                        not first
                        or not second
                        or not any(
                            records[left].trajectory_id != records[right].trajectory_id
                            for left in first
                            for right in second
                        )
                    ):
                        raise LTSNContractError(
                            "snapshot-pair training requires cross-trajectory low/middle/high "
                            f"coverage for family={family}, step={step}"
                        )
        self.batch_count = max(
            math.ceil(len(self.id_indices) / id_per_batch),
            math.ceil(sum(len(values) for values in by_kind.values()) / ood_per_batch),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _take(pools: dict[Any, list[Any]], key: Any, values: list[Any], rng: random.Random) -> Any:
        if not pools.get(key):
            pools[key] = values.copy()
            rng.shuffle(pools[key])
        return pools[key].pop()

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pools: dict[Any, list[Any]] = {}
        family_stream = Pitch3BalancedBatchSampler._cycle(
            list(range(len(self.families))), self.batch_count, rng
        )
        family_counts = Counter()
        kind_names = list(self.ood_by_kind)
        ood_assignments = [
            kind_names[index % len(kind_names)]
            for index in range(self.batch_count * self.ood_per_batch)
        ]
        rng.shuffle(ood_assignments)
        for batch_index, family_index in enumerate(family_stream):
            family = self.families[family_index]
            family_count = family_counts[family]
            family_counts[family] += 1
            omitted = family_count % len(FORMAL_STEPS)
            selected_steps = [step for index, step in enumerate(FORMAL_STEPS) if index != omitted]
            rotation = (family_count // len(FORMAL_STEPS)) % len(selected_steps)
            selected_steps = selected_steps[rotation:] + selected_steps[:rotation]
            batch: list[int] = []
            for step, (first_stratum, second_stratum) in zip(
                selected_steps, self._PAIR_PATTERNS, strict=True
            ):
                first_values = self.snapshots[(family, step, first_stratum)]
                second_values = self.snapshots[(family, step, second_stratum)]
                viable_first = [
                    index
                    for index in first_values
                    if any(
                        self.records[index].trajectory_id != self.records[other].trajectory_id
                        for other in second_values
                    )
                ]
                first = self._take(
                    pools,
                    ("pair-first", family, step, first_stratum, second_stratum),
                    viable_first,
                    rng,
                )
                second_candidates = [
                    index
                    for index in second_values
                    if self.records[index].trajectory_id != self.records[first].trajectory_id
                ]
                second = self._take(
                    pools,
                    (
                        "pair-second",
                        family,
                        step,
                        second_stratum,
                        self.records[first].trajectory_id,
                    ),
                    second_candidates,
                    rng,
                )
                batch.extend((first, second))
            for offset in range(self.ood_per_batch):
                kind = ood_assignments[batch_index * self.ood_per_batch + offset]
                batch.append(self._take(pools, ("ood", kind), self.ood_by_kind[kind], rng))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batch_count


def _same_group_mask(values: Sequence[str], reference: Tensor) -> Tensor:
    return torch.tensor(
        [[first == second for second in values] for first in values],
        device=reference.device,
        dtype=torch.bool,
    )


def pitch3_loss(
    output: Pitch3ControlOutput,
    coordinates: Tensor,
    focus_logit: Tensor,
    ood_label: Tensor,
    weights: Pitch3LossWeights,
    *,
    ood_positive_weight: float = 1.0,
    target_lower: Tensor | None = None,
    target_upper: Tensor | None = None,
    distance_weights: Tensor | None = None,
    prompt_family: Sequence[str] | None = None,
    trajectory_id: Sequence[str] | None = None,
    step_number: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if coordinates.ndim != 2 or coordinates.shape[1] != PITCH3_DIMENSIONS:
        raise ValueError("Pitch-3 targets must have shape [B,3]")
    id_mask = ood_label < 0.5
    if id_mask.any():
        error = output.coordinate_mean[id_mask] - coordinates[id_mask]
        coordinate = F.smooth_l1_loss(output.coordinate_mean[id_mask], coordinates[id_mask])
        nll = (
            0.5
            * (
                torch.exp(-output.coordinate_logvar[id_mask]) * error.square()
                + output.coordinate_logvar[id_mask]
            ).mean()
        )
        focus = F.smooth_l1_loss(output.focus_logit[id_mask], focus_logit[id_mask])
        group_losses_enabled = (
            weights.local_band_rank > 0.0
            or weights.coordinate2_rank > 0.0
            or weights.trajectory_delta > 0.0
        )
        if group_losses_enabled:
            batch_size = len(ood_label)
            if (
                prompt_family is None
                or trajectory_id is None
                or step_number is None
                or len(prompt_family) != batch_size
                or len(trajectory_id) != batch_size
                or step_number.shape != ood_label.shape
            ):
                raise ValueError("group-robust losses require batch-aligned family/trajectory/step")
            id_keep = id_mask.detach().cpu().tolist()
            id_families = [
                value for value, keep in zip(prompt_family, id_keep, strict=True) if keep
            ]
            id_trajectories = [
                value for value, keep in zip(trajectory_id, id_keep, strict=True) if keep
            ]
            id_steps = step_number[id_mask]
            upper_triangle = torch.triu(
                torch.ones(
                    (len(id_families), len(id_families)),
                    device=coordinates.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            local_pair_mask = (
                upper_triangle
                & _same_group_mask(id_families, coordinates)
                & (id_steps[:, None] == id_steps[None, :])
            )
            trajectory_pair_mask = (
                upper_triangle
                & _same_group_mask(id_trajectories, coordinates)
                & (id_steps[:, None] != id_steps[None, :])
            )
        else:
            local_pair_mask = None
            trajectory_pair_mask = None
        if (
            weights.band > 0.0
            or weights.band_rank > 0.0
            or weights.local_band_rank > 0.0
            or weights.band_region > 0.0
            or weights.band_zero > 0.0
        ):
            if target_lower is None or target_upper is None or distance_weights is None:
                raise ValueError("Pitch-3 band loss requires frozen target tensors")
            predicted_coordinates = output.coordinate_mean[id_mask]
            exact_coordinates = coordinates[id_mask]
            hard_predicted_below = torch.relu(target_lower - predicted_coordinates)
            hard_predicted_above = torch.relu(predicted_coordinates - target_upper)
            if weights.band_smooth_temperature_fraction > 0.0:
                temperature = (target_upper - target_lower) * float(
                    weights.band_smooth_temperature_fraction
                )
                predicted_below = temperature * F.softplus(
                    (target_lower - predicted_coordinates) / temperature
                )
                predicted_above = temperature * F.softplus(
                    (predicted_coordinates - target_upper) / temperature
                )
            else:
                predicted_below = hard_predicted_below
                predicted_above = hard_predicted_above
            exact_below = torch.relu(target_lower - exact_coordinates)
            exact_above = torch.relu(exact_coordinates - target_upper)
            predicted_band = (
                (predicted_below.square() + predicted_above.square()) * distance_weights
            ).sum(dim=1)
            exact_band = ((exact_below.square() + exact_above.square()) * distance_weights).sum(
                dim=1
            )
            if weights.band_zero > 0.0:
                exact_zero_band = exact_band == 0.0
                if exact_zero_band.any():
                    hard_linear_excursion = (
                        (hard_predicted_below + hard_predicted_above) * distance_weights
                    ).sum(dim=1)
                    band_zero = hard_linear_excursion[exact_zero_band].mean()
                else:
                    band_zero = output.coordinate_mean.sum() * 0.0
            else:
                band_zero = output.coordinate_mean.sum() * 0.0
            predicted_log_band = torch.log1p(predicted_band)
            exact_log_band = torch.log1p(exact_band)
            band = F.smooth_l1_loss(predicted_log_band, exact_log_band)
            if weights.band_rank > 0.0 and len(predicted_log_band) >= 2:
                exact_delta = exact_log_band[:, None] - exact_log_band[None, :]
                predicted_delta = predicted_log_band[:, None] - predicted_log_band[None, :]
                upper_triangle = torch.triu(
                    torch.ones_like(exact_delta, dtype=torch.bool), diagonal=1
                )
                informative = upper_triangle & (exact_delta.abs() >= weights.band_rank_min_delta)
                if informative.any():
                    signs = torch.sign(exact_delta[informative])
                    band_rank = F.softplus(-signs * predicted_delta[informative]).mean()
                else:
                    band_rank = output.coordinate_mean.sum() * 0.0
            else:
                band_rank = output.coordinate_mean.sum() * 0.0
            if weights.local_band_rank > 0.0 and local_pair_mask is not None:
                exact_delta = exact_log_band[:, None] - exact_log_band[None, :]
                predicted_delta = predicted_log_band[:, None] - predicted_log_band[None, :]
                informative = local_pair_mask & (exact_delta.abs() >= weights.band_rank_min_delta)
                if informative.any():
                    signs = torch.sign(exact_delta[informative])
                    local_band_rank = F.softplus(-signs * predicted_delta[informative]).mean()
                else:
                    local_band_rank = output.coordinate_mean.sum() * 0.0
            else:
                local_band_rank = output.coordinate_mean.sum() * 0.0
            if weights.band_region > 0.0:
                exact_is_below = exact_coordinates < target_lower
                exact_is_above = exact_coordinates > target_upper
                below_region_error = torch.relu(predicted_coordinates - target_lower)
                above_region_error = torch.relu(target_upper - predicted_coordinates)
                inside_region_error = torch.relu(target_lower - predicted_coordinates) + torch.relu(
                    predicted_coordinates - target_upper
                )
                if weights.band_region_positive_only:
                    region_error = torch.where(
                        exact_is_below,
                        below_region_error,
                        torch.where(
                            exact_is_above,
                            above_region_error,
                            torch.zeros_like(inside_region_error),
                        ),
                    )
                    exact_positive_band = exact_band > 0.0
                    if exact_positive_band.any():
                        band_region = (
                            (region_error * distance_weights).sum(dim=1)[exact_positive_band].mean()
                        )
                    else:
                        band_region = output.coordinate_mean.sum() * 0.0
                else:
                    region_error = torch.where(
                        exact_is_below,
                        below_region_error,
                        torch.where(exact_is_above, above_region_error, inside_region_error),
                    )
                    band_region = (region_error * distance_weights).sum(dim=1).mean()
            else:
                band_region = output.coordinate_mean.sum() * 0.0
        else:
            band = output.coordinate_mean.sum() * 0.0
            band_rank = output.coordinate_mean.sum() * 0.0
            local_band_rank = output.coordinate_mean.sum() * 0.0
            band_region = output.coordinate_mean.sum() * 0.0
            band_zero = output.coordinate_mean.sum() * 0.0
        if weights.coordinate2_rank > 0.0 and local_pair_mask is not None:
            exact_coordinate2 = coordinates[id_mask, 1]
            predicted_coordinate2 = output.coordinate_mean[id_mask, 1]
            exact_delta = exact_coordinate2[:, None] - exact_coordinate2[None, :]
            predicted_delta = predicted_coordinate2[:, None] - predicted_coordinate2[None, :]
            informative = local_pair_mask & (
                exact_delta.abs() >= weights.coordinate2_rank_min_delta
            )
            if informative.any():
                signs = torch.sign(exact_delta[informative])
                coordinate2_rank = F.softplus(-signs * predicted_delta[informative]).mean()
            else:
                coordinate2_rank = output.coordinate_mean.sum() * 0.0
        else:
            coordinate2_rank = output.coordinate_mean.sum() * 0.0
        if weights.trajectory_delta > 0.0 and trajectory_pair_mask is not None:
            exact_id_coordinates = coordinates[id_mask]
            predicted_id_coordinates = output.coordinate_mean[id_mask]
            exact_delta = exact_id_coordinates[:, None, :] - exact_id_coordinates[None, :, :]
            predicted_delta = (
                predicted_id_coordinates[:, None, :] - predicted_id_coordinates[None, :, :]
            )
            if trajectory_pair_mask.any():
                trajectory_delta = F.smooth_l1_loss(
                    predicted_delta[trajectory_pair_mask],
                    exact_delta[trajectory_pair_mask],
                )
            else:
                trajectory_delta = output.coordinate_mean.sum() * 0.0
        else:
            trajectory_delta = output.coordinate_mean.sum() * 0.0
    else:
        zero = output.coordinate_mean.sum() * 0.0
        coordinate = zero
        nll = zero
        focus = zero
        band = zero
        band_rank = zero
        local_band_rank = zero
        band_region = zero
        band_zero = zero
        coordinate2_rank = zero
        trajectory_delta = zero
    positive_weight = torch.tensor(
        ood_positive_weight, device=ood_label.device, dtype=ood_label.dtype
    )
    ood = F.binary_cross_entropy_with_logits(
        output.ood_logit, ood_label, pos_weight=positive_weight
    )
    ood_logits = output.ood_logit
    id_ood_logits = ood_logits[ood_label < 0.5]
    positive_ood_logits = ood_logits[ood_label >= 0.5]
    if weights.ood_margin > 0.0 and len(id_ood_logits) and len(positive_ood_logits):
        separation = positive_ood_logits[:, None] - id_ood_logits[None, :]
        ood_margin = torch.relu(weights.ood_margin_value - separation).mean()
    else:
        ood_margin = output.ood_logit.sum() * 0.0
    total = (
        weights.coordinate * coordinate
        + weights.nll * nll
        + weights.focus * focus
        + weights.ood * ood
        + weights.band * band
        + weights.band_rank * band_rank
        + weights.local_band_rank * local_band_rank
        + weights.band_region * band_region
        + weights.band_zero * band_zero
        + weights.coordinate2_rank * coordinate2_rank
        + weights.trajectory_delta * trajectory_delta
        + weights.ood_margin * ood_margin
    )
    return total, {
        "coordinate": coordinate,
        "nll": nll,
        "focus": focus,
        "ood": ood,
        "band": band,
        "band_rank": band_rank,
        "local_band_rank": local_band_rank,
        "band_region": band_region,
        "band_zero": band_zero,
        "coordinate2_rank": coordinate2_rank,
        "trajectory_delta": trajectory_delta,
        "ood_margin": ood_margin,
    }


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
    target_lower: Tensor,
    target_upper: Tensor,
    distance_weights: Tensor,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "coordinate": 0.0,
        "nll": 0.0,
        "focus": 0.0,
        "ood": 0.0,
        "band": 0.0,
        "band_rank": 0.0,
        "local_band_rank": 0.0,
        "band_region": 0.0,
        "band_zero": 0.0,
        "coordinate2_rank": 0.0,
        "trajectory_delta": 0.0,
        "ood_margin": 0.0,
    }
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
                    target_lower=target_lower,
                    target_upper=target_upper,
                    distance_weights=distance_weights,
                    prompt_family=batch["prompt_family"],
                    trajectory_id=batch["trajectory_id"],
                    step_number=batch["step_number"],
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


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return 0.0
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels >= 0.5]
    negative = scores[labels < 0.5]
    if not len(positive) or not len(negative):
        return 0.0
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _quartile_accuracy(exact: np.ndarray, predicted: np.ndarray) -> float:
    low = np.flatnonzero(exact <= np.quantile(exact, 0.25))
    high = np.flatnonzero(exact >= np.quantile(exact, 0.75))
    if not len(low) or not len(high):
        return 0.0
    return float(np.mean(predicted[high, None] > predicted[None, low]))


def _band_loss_numpy(coordinates: np.ndarray, contract: Pitch3Contract) -> np.ndarray:
    lower = np.asarray(contract.target_lower, dtype=float)
    upper = np.asarray(contract.target_upper, dtype=float)
    weights = np.asarray(contract.distance_weights, dtype=float)
    below = np.maximum(lower - coordinates, 0.0)
    above = np.maximum(coordinates - upper, 0.0)
    return ((np.square(below) + np.square(above)) * weights).sum(axis=1)


def _regression_gate_metrics(
    exact: np.ndarray,
    mean: np.ndarray,
    exact_focus: np.ndarray,
    predicted_focus: np.ndarray,
    contract: Pitch3Contract,
) -> dict[str, Any]:
    coordinate_rhos = [
        _spearman(mean[:, index], exact[:, index]) for index in range(PITCH3_DIMENSIONS)
    ]
    return {
        "samples": int(len(exact)),
        "focus_logit_spearman": _spearman(predicted_focus, exact_focus),
        "coordinate_spearman": coordinate_rhos,
        "coordinate_median_spearman": float(np.median(coordinate_rhos)),
        "target_band_distance_spearman": _spearman(
            _band_loss_numpy(mean, contract), _band_loss_numpy(exact, contract)
        ),
        "quartile_ranking_accuracy": _quartile_accuracy(exact_focus, predicted_focus),
    }


def _regression_gate_deficits(metrics: dict[str, Any]) -> dict[str, float]:
    thresholds = {
        "focus_logit_spearman": 0.70,
        "coordinate_median_spearman": 0.50,
        "target_band_distance_spearman": 0.50,
        "quartile_ranking_accuracy": 0.65,
    }
    deficits = {
        name: max(0.0, threshold - float(metrics[name])) for name, threshold in thresholds.items()
    }
    deficits["each_coordinate_spearman"] = sum(
        max(0.0, 0.50 - value) for value in metrics["coordinate_spearman"]
    )
    return deficits


@torch.no_grad()
def _development_gate_screen(
    model: LatentTopologyControlHead,
    loader: DataLoader[dict[str, Any]],
    *,
    contract: Pitch3Contract,
    device: torch.device,
    use_bf16: bool,
    group_robust: bool = False,
) -> dict[str, Any]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "mean": [],
        "predicted_focus": [],
        "ood_logit": [],
        "coordinates": [],
        "focus": [],
        "ood_label": [],
        "step": [],
    }
    prompt_families: list[str] = []
    for raw in loader:
        batch = _device_batch(raw, device)
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
        tensors = {
            "mean": output.coordinate_mean,
            "predicted_focus": output.focus_logit,
            "ood_logit": output.ood_logit,
            "coordinates": batch["coordinates"],
            "focus": batch["focus_logit"],
            "ood_label": batch["ood_label"],
            "step": batch["step_number"],
        }
        for name, tensor in tensors.items():
            collected[name].append(tensor.detach().float().cpu().numpy())
        prompt_families.extend(batch["prompt_family"])
    values = {name: np.concatenate(items, axis=0) for name, items in collected.items()}
    labels = values["ood_label"]
    id_mask = labels < 0.5
    correction_mask = np.isin(values["step"].astype(int), CORRECTION_STEPS)
    if not np.any(id_mask) or not np.any((labels >= 0.5) & correction_mask):
        raise LTSNContractError("gate-aligned selection requires development ID and OOD")
    probabilities = np.exp(-np.logaddexp(0.0, -values["ood_logit"]))
    quantiles = []
    id_acceptance_by_step: dict[str, float] = {}
    for step in CORRECTION_STEPS:
        step_id = (values["step"] == step) & id_mask
        if not np.any(step_id):
            raise LTSNContractError(f"development selection lacks ID step {step}")
        quantiles.append(float(np.quantile(probabilities[step_id], 0.95, method="higher")))
    threshold = max(quantiles)
    for step in CORRECTION_STEPS:
        step_id = (values["step"] == step) & id_mask
        id_acceptance_by_step[str(step)] = float(np.mean(probabilities[step_id] <= threshold))
    exact = values["coordinates"][id_mask]
    mean = values["mean"][id_mask]
    exact_focus = values["focus"][id_mask]
    predicted_focus = values["predicted_focus"][id_mask]
    regression_metrics = _regression_gate_metrics(
        exact, mean, exact_focus, predicted_focus, contract
    )
    screen_labels = labels[correction_mask]
    screen_probabilities = probabilities[correction_mask]
    sensitivity = float(np.mean(screen_probabilities[screen_labels >= 0.5] > threshold))
    metrics = {
        **regression_metrics,
        "minimum_correction_step_id_acceptance": min(id_acceptance_by_step.values()),
        "ood_auroc": _auc(screen_labels, screen_probabilities),
        "ood_sensitivity": sensitivity,
        "ood_probability_threshold": threshold,
        "id_acceptance_by_step": id_acceptance_by_step,
    }
    thresholds = {
        "focus_logit_spearman": 0.70,
        "coordinate_median_spearman": 0.50,
        "target_band_distance_spearman": 0.50,
        "quartile_ranking_accuracy": 0.65,
        "minimum_correction_step_id_acceptance": 0.95,
        "ood_auroc": 0.80,
        "ood_sensitivity": 0.80,
    }
    deficits: dict[str, float] = {
        name: max(0.0, threshold_value - float(metrics[name]))
        for name, threshold_value in thresholds.items()
    }
    deficits["each_coordinate_spearman"] = sum(
        max(0.0, 0.50 - value) for value in metrics["coordinate_spearman"]
    )
    total_deficit = float(sum(deficits.values()))
    group_screen: dict[str, Any] = {
        "enabled": group_robust,
        "prompt_family_metrics": {},
        "step_metrics": {},
        "prompt_family_deficits": {},
        "step_deficits": {},
        "worst_prompt_family_deficit": 0.0,
        "worst_step_deficit": 0.0,
        "total_deficit": 0.0,
        "all_group_gates_passed": True,
    }
    if group_robust:
        family_array = np.asarray(prompt_families, dtype=object)[id_mask]
        id_steps = values["step"][id_mask].astype(int)
        groups = {
            "prompt_family": (family_array, sorted(set(family_array))),
            "step": (id_steps, list(FORMAL_STEPS)),
        }
        for group_name, (group_values, names) in groups.items():
            metric_output = group_screen[f"{group_name}_metrics"]
            deficit_output = group_screen[f"{group_name}_deficits"]
            for name in names:
                mask = group_values == name
                if np.count_nonzero(mask) < 2:
                    raise LTSNContractError(
                        f"group-robust development screen lacks {group_name} samples: {name}"
                    )
                group_metrics = _regression_gate_metrics(
                    exact[mask],
                    mean[mask],
                    exact_focus[mask],
                    predicted_focus[mask],
                    contract,
                )
                group_deficits = _regression_gate_deficits(group_metrics)
                key = str(name)
                metric_output[key] = group_metrics
                deficit_output[key] = {
                    **group_deficits,
                    "total_deficit": float(sum(group_deficits.values())),
                }
        group_screen["worst_prompt_family_deficit"] = max(
            value["total_deficit"] for value in group_screen["prompt_family_deficits"].values()
        )
        group_screen["worst_step_deficit"] = max(
            value["total_deficit"] for value in group_screen["step_deficits"].values()
        )
        group_screen["total_deficit"] = (
            group_screen["worst_prompt_family_deficit"] + group_screen["worst_step_deficit"]
        )
        group_screen["all_group_gates_passed"] = group_screen["total_deficit"] <= 1e-12
    return {
        "metrics": metrics,
        "deficits": deficits,
        "total_deficit": total_deficit,
        "all_gates_passed": total_deficit <= 1e-12,
        "group_robust": group_screen,
    }


def _pitch3_checkpoint_selection_key(
    training: Pitch3TrainingConfig,
    *,
    gate_deficit: float,
    group_gate_deficit: float,
    development_loss: float,
    pooled_band_spearman: float,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    band_floor_deficit = max(0.0, training.pooled_band_spearman_floor - pooled_band_spearman)
    guardrail_passed = gate_deficit <= 1e-12 and band_floor_deficit <= 1e-12
    guardrail = {
        "pooled_band_spearman_floor": training.pooled_band_spearman_floor,
        "pooled_band_floor_deficit": band_floor_deficit,
        "pooled_gate_deficit": gate_deficit,
        "passed": guardrail_passed,
    }
    if training.group_robust_selection and training.pooled_band_spearman_floor > 0.0:
        key: tuple[Any, ...] = (
            not guardrail_passed,
            gate_deficit + band_floor_deficit,
            group_gate_deficit,
            development_loss,
        )
    elif training.group_robust_selection:
        key = (group_gate_deficit, gate_deficit, development_loss)
    else:
        key = (gate_deficit, development_loss)
    return key, guardrail


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
            raise LTSNContractError("Pitch-3 training requires ID and OOD development samples")
    if training.id_samples_per_batch and training.ood_samples_per_batch:
        positive_weight = training.id_samples_per_batch / training.ood_samples_per_batch
    else:
        positive_weight = 1.0 if not positives else max(1.0, negatives / positives)
    ood_versions = sorted(
        {record.ood_transform_version for record in records if record.ood_label >= 0.5}
    )
    ood_label_sources = sorted(
        {record.ood_label_source for record in records if record.ood_label >= 0.5}
    )
    if len(ood_versions) != 1 or len(ood_label_sources) != 1:
        raise LTSNContractError("Pitch-3 training requires one frozen OOD provenance contract")
    _set_seed(training.seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    model = LatentTopologyControlHead(contract, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    train_generator = torch.Generator().manual_seed(training.seed)
    train_dataset = Pitch3Dataset(
        train_records,
        scale_high_training_target_scales=training.scale_high_training_target_scales,
    )
    train_batch_sampler: (
        Pitch3BalancedBatchSampler
        | Pitch3GroupRobustBatchSampler
        | Pitch3SnapshotPairBatchSampler
        | None
    ) = None
    if training.id_samples_per_batch and training.ood_samples_per_batch:
        if training.group_robust_sampling:
            sampler_class = (
                Pitch3SnapshotPairBatchSampler
                if training.group_robust_sampler_kind == "snapshot_pair_v26_grr"
                else Pitch3GroupRobustBatchSampler
            )
            train_batch_sampler = sampler_class(
                train_records,
                id_per_batch=training.id_samples_per_batch,
                ood_per_batch=training.ood_samples_per_batch,
                seed=training.seed,
                contract=contract,
            )
        else:
            train_batch_sampler = Pitch3BalancedBatchSampler(
                train_records,
                id_per_batch=training.id_samples_per_batch,
                ood_per_batch=training.ood_samples_per_batch,
                seed=training.seed,
                contract=contract,
                id_band_stratified=training.id_band_stratified,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=training.num_workers,
            collate_fn=collate_pitch3,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
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
    use_bf16 = training.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    virtual_scale_augmentation = _virtual_scale_summary(
        train_records, training.scale_high_training_target_scales
    )
    target_lower = torch.tensor(contract.target_lower, device=device, dtype=torch.float32)
    target_upper = torch.tensor(contract.target_upper, device=device, dtype=torch.float32)
    distance_weights = torch.tensor(contract.distance_weights, device=device, dtype=torch.float32)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_gate_deficit = float("inf")
    best_group_gate_deficit = float("inf")
    best_pooled_band_floor_deficit = float("inf")
    best_pooled_guardrail_passed = False
    best_selection_key: tuple[Any, ...] | None = None
    best_gate_screen: dict[str, Any] | None = None
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, training.max_epochs + 1):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        train_metrics = _epoch(
            model,
            train_loader,
            weights,
            device,
            optimizer=optimizer,
            gradient_clip_norm=training.gradient_clip_norm,
            use_bf16=use_bf16,
            ood_positive_weight=positive_weight,
            target_lower=target_lower,
            target_upper=target_upper,
            distance_weights=distance_weights,
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
            target_lower=target_lower,
            target_upper=target_upper,
            distance_weights=distance_weights,
        )
        gate_screen = (
            _development_gate_screen(
                model,
                development_loader,
                contract=contract,
                device=device,
                use_bf16=use_bf16,
                group_robust=training.group_robust_selection,
            )
            if training.gate_aligned_selection
            else None
        )
        gate_deficit = 0.0 if gate_screen is None else float(gate_screen["total_deficit"])
        group_gate_deficit = (
            0.0 if gate_screen is None else float(gate_screen["group_robust"]["total_deficit"])
        )
        pooled_band_spearman = (
            training.pooled_band_spearman_floor
            if gate_screen is None
            else float(gate_screen["metrics"]["target_band_distance_spearman"])
        )
        current_key, selection_guardrail = _pitch3_checkpoint_selection_key(
            training,
            gate_deficit=gate_deficit,
            group_gate_deficit=group_gate_deficit,
            development_loss=development_metrics["loss"],
            pooled_band_spearman=pooled_band_spearman,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "development": development_metrics,
                "development_gate_screen": gate_screen,
                "selection_guardrail": selection_guardrail,
            }
        )
        improved = best_selection_key is None or current_key < best_selection_key
        if improved:
            best_selection_key = current_key
            best_loss = development_metrics["loss"]
            best_gate_deficit = gate_deficit
            best_group_gate_deficit = group_gate_deficit
            best_pooled_band_floor_deficit = selection_guardrail["pooled_band_floor_deficit"]
            best_pooled_guardrail_passed = selection_guardrail["passed"]
            best_gate_screen = gate_screen
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
        "architecture_revision": (
            "pitch3_ltch_v23_raw_ood_stats" if model_config.ood_stats_dim else "pitch3_ltch_legacy"
        ),
        "ood_stats_projection_dim": model_config.ood_stats_dim,
        "ood_stats_feature_order": (
            list(PITCH3_OOD_STAT_FEATURES) if model_config.ood_stats_dim else []
        ),
        "ood_stats_source": (
            "pre_input_layernorm_masked_latent" if model_config.ood_stats_dim else ""
        ),
        "band_training_objective": {
            "kind": (
                "balanced_zero_positive_region_v25"
                if weights.band_zero > 0.0 and weights.band_region_positive_only
                else (
                    "region_consistency_smooth_excursion_v24"
                    if weights.band_region > 0.0 and weights.band_smooth_temperature_fraction > 0.0
                    else (
                        "region_consistency_v24"
                        if weights.band_region > 0.0
                        else (
                            "smooth_excursion_v24"
                            if weights.band_smooth_temperature_fraction > 0.0
                            else "hard_excursion_v23_compatible"
                        )
                    )
                )
            ),
            "region_consistency_weight": weights.band_region,
            "zero_distance_suppression_weight": weights.band_zero,
            "region_positive_samples_only": weights.band_region_positive_only,
            "boundary_auxiliary_weight_sum": weights.band_region + weights.band_zero,
            "smooth_temperature_fraction_of_band_width": (weights.band_smooth_temperature_fraction),
            "evaluation_band_formula_changed": False,
        },
        "group_robust_training": {
            "enabled": training.group_robust_sampling and training.group_robust_selection,
            "sampler": (
                "prompt_family_snapshot_band_pair_rotation"
                if isinstance(train_batch_sampler, Pitch3SnapshotPairBatchSampler)
                else (
                    "prompt_family_trajectory_anchor_band_rotation"
                    if training.group_robust_sampling
                    else "disabled"
                )
            ),
            "local_band_rank_weight": weights.local_band_rank,
            "coordinate2_rank_weight": weights.coordinate2_rank,
            "trajectory_delta_weight": weights.trajectory_delta,
            "same_prompt_family_and_step_pairs": True,
            "local_pairs_per_batch": (
                3 if isinstance(train_batch_sampler, Pitch3SnapshotPairBatchSampler) else 2
            ),
            "snapshot_band_counts_per_batch": (
                {"low": 2, "middle": 2, "high": 2}
                if isinstance(train_batch_sampler, Pitch3SnapshotPairBatchSampler)
                else {}
            ),
            "development_screen_required": training.group_robust_selection,
            "qualification_split_used_for_training_or_selection": False,
        },
        "best_development_loss": best_loss,
        "best_development_gate_deficit": best_gate_deficit,
        "best_development_group_gate_deficit": best_group_gate_deficit,
        "best_development_pooled_band_floor_deficit": best_pooled_band_floor_deficit,
        "best_development_pooled_guardrail_passed": best_pooled_guardrail_passed,
        "pooled_selection_guardrail": {
            "target_band_distance_spearman_floor": training.pooled_band_spearman_floor,
            "pooled_gates_must_pass": training.pooled_band_spearman_floor > 0.0,
            "thresholds_changed": False,
        },
        "best_development_gate_screen": best_gate_screen,
        "epochs_completed": len(history),
        "ood_class_counts": {
            "train": {"id": negatives, "ood": positives},
            "development": {
                "id": development_negatives,
                "ood": development_positives,
            },
        },
        "ood_transform_version": ood_versions[0],
        "ood_label_source": ood_label_sources[0],
        "batch_policy": {
            "kind": (
                "prompt_family_snapshot_band_pair_rotation"
                if isinstance(train_batch_sampler, Pitch3SnapshotPairBatchSampler)
                else (
                    "prompt_family_trajectory_anchor_band_rotation"
                    if isinstance(train_batch_sampler, Pitch3GroupRobustBatchSampler)
                    else (
                        "balanced_id_ood_family_rotation"
                        if train_batch_sampler is not None
                        else "random_shuffle"
                    )
                )
            ),
            "id_samples_per_batch": training.id_samples_per_batch,
            "ood_samples_per_batch": training.ood_samples_per_batch,
            "ood_positive_weight": positive_weight,
            "id_band_stratified": training.id_band_stratified,
            "id_strata": (
                train_batch_sampler.id_strata_summary
                if isinstance(
                    train_batch_sampler,
                    (Pitch3BalancedBatchSampler, Pitch3SnapshotPairBatchSampler),
                )
                else {}
            ),
            "trajectory_strata": (
                train_batch_sampler.trajectory_strata_summary
                if isinstance(train_batch_sampler, Pitch3GroupRobustBatchSampler)
                else {}
            ),
        },
        "checkpoint_selection": (
            "development_pooled_guardrail_then_worst_group_then_loss"
            if training.group_robust_selection and training.pooled_band_spearman_floor > 0.0
            else (
                "development_worst_group_then_pooled_gate_deficit_then_loss"
                if training.group_robust_selection
                else (
                    "development_gate_deficit_then_loss"
                    if training.gate_aligned_selection
                    else "development_loss"
                )
            )
        ),
        "training_virtual_ood_augmentation": virtual_scale_augmentation,
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
