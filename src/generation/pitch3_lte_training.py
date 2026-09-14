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
from .pitch3_lte_data import LTE_MODEL_FAMILY, LTE_RADIUS_RATIO


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

    def validate(self) -> None:
        if self.max_epochs < 1 or not 1 <= self.minimum_epochs <= self.max_epochs:
            raise ValueError("V3-LTE epoch limits are invalid")
        if self.early_stopping_patience < 1 or self.gradient_clip_norm <= 0:
            raise ValueError("V3-LTE patience and gradient clip must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("V3-LTE optimizer settings are invalid")
        if self.huber_delta <= 0 or self.rank_min_delta <= 0:
            raise ValueError("V3-LTE robust-loss thresholds must be positive")


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
    """One complete prompt group per batch: four seeds plus two +/- pairs."""

    def __init__(self, records: Sequence[Pitch3LTEExample], seed: int, shuffle: bool) -> None:
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
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        groups = [values.copy() for values in self.groups]
        if self.shuffle:
            random.Random(self.seed + 1_000_003 * self.epoch).shuffle(groups)
        yield from groups

    def __len__(self) -> int:
        return len(self.groups)


def _pair_indices(batch: Mapping[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    base = [index for index, kind in enumerate(batch["source_kind"]) if kind == "base_step4_seed"]
    rank_pairs = [(left, right) for offset, left in enumerate(base) for right in base[offset + 1 :]]
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


def pitch3_lte_raw_losses(
    predicted: Tensor,
    batch: Mapping[str, Any],
    *,
    huber_delta: float,
    rank_min_delta: float,
) -> dict[str, Tensor]:
    target = batch["energy_target"].float()
    value = F.huber_loss(predicted.float(), target, delta=huber_delta)
    rank_terms = []
    fd_terms = []
    rank_pairs, fd_pairs = _pair_indices(batch)
    for left, right in rank_pairs:
        exact_delta = target[left] - target[right]
        if exact_delta.abs() >= rank_min_delta:
            rank_terms.append(
                F.softplus(-torch.sign(exact_delta) * (predicted[left] - predicted[right]))
            )
    for minus, plus in fd_pairs:
        epsilon = batch["epsilon"][minus].float()
        if not torch.isclose(epsilon, batch["epsilon"][plus].float(), atol=1e-9, rtol=0):
            raise LTSNContractError("V3-LTE finite-difference epsilon differs within a pair")
        exact_derivative = (target[plus] - target[minus]) / (2.0 * epsilon)
        predicted_derivative = (predicted[plus] - predicted[minus]) / (2.0 * epsilon)
        fd_terms.append(F.huber_loss(predicted_derivative, exact_derivative, delta=huber_delta))
    zero = predicted.sum() * 0.0
    return {
        "value": value,
        "prompt_rank": torch.stack(rank_terms).mean() if rank_terms else zero,
        "local_fd": torch.stack(fd_terms).mean() if fd_terms else zero,
    }


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
    }
    deficits = {
        "direct_energy_spearman": max(0.0, 0.50 - pooled),
        "every_prompt_family_spearman": max(0.0, 0.50 - min(family_rho.values(), default=0.0)),
        "same_prompt_ranking_accuracy": max(0.0, 0.65 - ranking),
        "local_direction_sign_accuracy": max(0.0, 0.65 - direction) if direction_total else 0.65,
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
) -> dict[str, float]:
    values: dict[str, list[float]] = {"value": [], "prompt_rank": [], "local_fd": []}
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
                huber_delta=training.huber_delta,
                rank_min_delta=training.rank_min_delta,
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
                huber_delta=training.huber_delta,
                rank_min_delta=training.rank_min_delta,
            )
            totals.append(
                sum(float(losses[name].cpu()) / normalizers[name] for name in normalizers)
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
    contract = load_pitch3_contract(fingerprint_path)
    model_config, training = load_pitch3_lte_config(config_path)
    dataset_summary_path = dataset_manifest.parent / "pitch3_lte_dataset_summary.json"
    if not dataset_summary_path.is_file():
        raise LTSNContractError("V3-LTE dataset preflight summary is missing")
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
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
    _seed_everything(training.seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for V3-LTE but is unavailable")
    model = PromptConditionedTopologyEnergy(model_config).to(device)
    train_sampler = PromptBatchSampler(train_records, training.seed, True)
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
    normalizers = _loss_normalizers(model, train_loader, device, training)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"pitch3_lte_seed_{training.seed}.pt"
    best_key = (float("inf"), float("inf"))
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
        "seed": training.seed,
        "device": device_name,
        "precision": "bf16_forward_fp32_loss" if training.use_bf16 else "fp32",
        "energy_target": "log1p(exact_pitch3_target_band_loss)",
        "prompt_condition": "frozen_ace_step_text_hidden_state",
        "prompt_id_embedding_used": False,
        "guidance_steps": [4],
        "training_radius_ratio": LTE_RADIUS_RATIO,
        "maximum_guidance_update_ratio": LTE_RADIUS_RATIO / 2.0,
        "loss_components": ["value_huber", "same_prompt_rank", "local_central_difference"],
        "loss_component_weights": [1.0, 1.0, 1.0],
        "first_epoch_loss_medians": normalizers,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    for epoch in range(1, training.max_epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        sums = {"value": 0.0, "prompt_rank": 0.0, "local_fd": 0.0, "total": 0.0}
        batches = 0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=training.use_bf16 and device.type == "cuda",
            ):
                predicted = model(
                    batch["latent"],
                    batch["attention_mask"],
                    batch["text_hidden"],
                    batch["text_mask"],
                ).energy
            raw_losses = pitch3_lte_raw_losses(
                predicted.float(),
                batch,
                huber_delta=training.huber_delta,
                rank_min_delta=training.rank_min_delta,
            )
            total = sum(raw_losses[name] / normalizers[name] for name in normalizers)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()
            for name, loss in raw_losses.items():
                sums[name] += float(loss.detach().cpu())
            sums["total"] += float(total.detach().cpu())
            batches += 1
        development_rows = _prediction_rows(model, development_loader, device, training.use_bf16)
        metrics = pitch3_lte_metrics(development_rows)
        development_loss = _normalized_dataset_loss(
            model, development_loader, device, training, normalizers
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
            }
        )
        if selection < best_key:
            best_key = selection
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "training_config": asdict(training),
                    "metadata": {
                        **metadata_base,
                        "best_epoch": epoch,
                        "best_development_gate_deficit": selection[0],
                        "best_development_loss": selection[1],
                        "best_development_metrics": metrics,
                        "trainable_parameters": model.trainable_parameters,
                    },
                },
                checkpoint_path,
            )
        else:
            patience += 1
        if epoch >= training.minimum_epochs and patience >= training.early_stopping_patience:
            break
    _, best_metadata = load_pitch3_lte_checkpoint(checkpoint_path, device=device)
    manifest = {
        **best_metadata,
        "epochs_completed": len(history),
        "checkpoint_selection": "development_gate_deficit_then_normalized_training_loss",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history": history,
    }
    manifest_path = output_dir / "pitch3_lte_manifest.json"
    write_json_atomic(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
