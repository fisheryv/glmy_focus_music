"""GPU training for the diagnostic-only V5.2b direction probes."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import os
import random
import tomllib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .ltsn_pipeline import write_json_atomic
from .ltsn_v52b import (
    PRIMARY_VARIANT,
    V52B_EXPERIMENT,
    V52B_TARGET_MODES,
    V52B_VARIANTS,
)
from .path_homology_surrogate import LTSNConfig, PathHomologySurrogate


@dataclass(frozen=True, slots=True)
class V52BProbeConfig:
    hidden_dim: int = 256


@dataclass(frozen=True, slots=True)
class V52BTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_fraction: float = 0.05
    minimum_learning_rate: float = 1e-6
    effective_batch_size: int = 32
    micro_batch_size: int = 16
    max_epochs: int = 100
    minimum_epochs: int = 100
    early_stopping_patience: int = 100
    gradient_clip_norm: float = 10.0
    num_workers: int = 4
    seeds: tuple[int, ...] = (20260716, 20260717, 20260718)

    def validate(self) -> None:
        if self.micro_batch_size < 1 or self.effective_batch_size < self.micro_batch_size:
            raise LTSNContractError("V5.2b effective batch must be at least the micro batch")
        if self.effective_batch_size % self.micro_batch_size:
            raise LTSNContractError("V5.2b effective batch must divide into micro batches")
        if self.minimum_epochs < 1 or self.minimum_epochs > self.max_epochs:
            raise LTSNContractError("invalid V5.2b epoch limits")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise LTSNContractError("V5.2b requires exactly three unique seeds")
        if self.weight_decay != 0.0:
            raise LTSNContractError("V5.2b freezes weight decay at zero")
        if self.gradient_clip_norm <= 0.0:
            raise LTSNContractError("V5.2b gradient clipping must be positive")


@dataclass(frozen=True, slots=True)
class V52BPair:
    pair_id: str
    evaluation_split: str
    step_number: int
    timestep: float
    rms_ratio: float
    minus_path: Path
    plus_path: Path
    true_target: float
    matched_control_target: float
    exact_derivative: float


def _dataclass_values(cls: type[Any], raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise LTSNContractError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return dict(raw)


def load_v52b_config(
    path: Path,
) -> tuple[LTSNConfig, V52BProbeConfig, V52BTrainingConfig]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model_raw = _dataclass_values(LTSNConfig, payload.get("model", {}))
    if "inactive_coordinate_indices" in model_raw:
        model_raw["inactive_coordinate_indices"] = tuple(
            int(value) for value in model_raw["inactive_coordinate_indices"]
        )
    probe = V52BProbeConfig(**_dataclass_values(V52BProbeConfig, payload.get("probe", {})))
    training_raw = _dataclass_values(V52BTrainingConfig, payload.get("training", {}))
    if "seeds" in training_raw:
        training_raw["seeds"] = tuple(int(value) for value in training_raw["seeds"])
    training = V52BTrainingConfig(**training_raw)
    training.validate()
    if probe.hidden_dim < 32:
        raise LTSNContractError("V5.2b probe hidden dimension is too small")
    return LTSNConfig(**model_raw), probe, training


def read_v52b_pairs(path: Path) -> list[V52BPair]:
    rows: list[V52BPair] = []
    checked_hashes: dict[Path, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            minus = (path.parent / raw["minus_latent_path"]).resolve()
            plus = (path.parent / raw["plus_latent_path"]).resolve()
            for latent, expected in (
                (minus, raw["minus_latent_sha256"]),
                (plus, raw["plus_latent_sha256"]),
            ):
                if not latent.is_file():
                    raise LTSNContractError(f"V5.2b latent is missing: {latent}")
                actual = checked_hashes.setdefault(latent, sha256_file(latent))
                if actual != expected:
                    raise LTSNContractError(f"V5.2b latent hash mismatch: {latent}")
            true_target = float(raw["true_target"])
            control_target = float(raw["matched_control_target"])
            if true_target not in {0.0, 1.0} or control_target not in {0.0, 1.0}:
                raise LTSNContractError("V5.2b targets must be binary")
            rows.append(
                V52BPair(
                    pair_id=raw["pair_id"],
                    evaluation_split=raw["evaluation_split"],
                    step_number=int(raw["step_number"]),
                    timestep=float(raw["timestep"]),
                    rms_ratio=float(raw["rms_ratio"]),
                    minus_path=minus,
                    plus_path=plus,
                    true_target=true_target,
                    matched_control_target=control_target,
                    exact_derivative=float(raw["exact_derivative"]),
                )
            )
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise LTSNContractError("V5.2b pair manifest is empty or contains duplicate IDs")
    expected_splits = {
        "train",
        "seen_anchor_heldout_direction",
        "unseen_anchor",
        "train_rms_sensitivity",
        "seen_anchor_heldout_direction_rms_sensitivity",
        "unseen_anchor_rms_sensitivity",
    }
    if {row.evaluation_split for row in rows} != expected_splits:
        raise LTSNContractError("V5.2b pair manifest has unexpected evaluation splits")
    return rows


class V52BPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, records: Sequence[V52BPair], split: str, *, target_mode: str = "true"
    ) -> None:
        if target_mode not in V52B_TARGET_MODES:
            raise ValueError(f"unsupported V5.2b target mode: {target_mode}")
        self.records = tuple(row for row in records if row.evaluation_split == split)
        self.target_mode = target_mode
        if not self.records:
            raise ValueError(f"no V5.2b pairs for split {split}")

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
            raise LTSNContractError(f"invalid V5.2b pair latents: {row.pair_id}")
        target = (
            row.matched_control_target
            if self.target_mode == "matched_control" and row.evaluation_split == "train"
            else row.true_target
        )
        return {
            "pair_id": row.pair_id,
            "minus": torch.from_numpy(np.asarray(minus, dtype=np.float32)),
            "plus": torch.from_numpy(np.asarray(plus, dtype=np.float32)),
            "timestep": torch.tensor(row.timestep, dtype=torch.float32),
            "step_number": torch.tensor(row.step_number, dtype=torch.long),
            "rms_ratio": torch.tensor(row.rms_ratio, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "true_target": torch.tensor(row.true_target, dtype=torch.float32),
            "exact_derivative": torch.tensor(row.exact_derivative, dtype=torch.float32),
        }


def collate_v52b_pairs(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty V5.2b batch")
    maximum = max(int(item["minus"].shape[0]) for item in items)
    minus = torch.zeros(len(items), maximum, 64, dtype=torch.float32)
    plus = torch.zeros_like(minus)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["minus"].shape[0])
        minus[index, :length] = item["minus"]
        plus[index, :length] = item["plus"]
        mask[index, :length] = True
    output = {"pair_id": [str(item["pair_id"]) for item in items]}
    output.update(
        {
            "minus": minus,
            "plus": plus,
            "attention_mask": mask,
            **{
                key: torch.stack([item[key] for item in items])
                for key in (
                    "timestep",
                    "step_number",
                    "rms_ratio",
                    "target",
                    "true_target",
                    "exact_derivative",
                )
            },
        }
    )
    return output


class V52BDirectionProbe(nn.Module):
    """Shared LTSN encoder with scalar, antisymmetric pair, or direction-field heads."""

    def __init__(
        self,
        contract: FingerprintContract,
        model_config: LTSNConfig,
        probe_config: V52BProbeConfig,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in V52B_VARIANTS:
            raise ValueError(f"unsupported V5.2b variant: {variant}")
        self.variant = variant
        self.encoder = PathHomologySurrogate(contract, model_config)
        hidden = probe_config.hidden_dim
        self.pair_head = nn.Sequential(nn.Linear(256 * 4, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.field_head = nn.Sequential(
            nn.Linear(256 * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 256),
        )

    def _raw_pair(self, left: Tensor, right: Tensor) -> Tensor:
        return self.pair_head(
            torch.cat((left, right, right - left, (right - left).abs()), dim=-1)
        ).squeeze(-1)

    def forward(
        self,
        minus: Tensor,
        plus: Tensor,
        timestep: Tensor,
        step_number: Tensor,
        rms_ratio: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        minus_features = self.encoder.encode(minus, timestep, step_number, attention_mask)
        plus_features = self.encoder.encode(plus, timestep, step_number, attention_mask)
        if self.variant == "scalar":
            minus_score = self.encoder.readout(minus_features).focus_logit
            plus_score = self.encoder.readout(plus_features).focus_logit
            return plus_score - minus_score
        if self.variant == "pair":
            return 0.5 * (
                self._raw_pair(minus_features, plus_features)
                - self._raw_pair(plus_features, minus_features)
            )
        delta = plus_features - minus_features
        midpoint = 0.5 * (plus_features + minus_features)
        rms_feature = torch.log10(rms_ratio.float().clamp_min(1e-8)).unsqueeze(-1)
        field = self.field_head(torch.cat((midpoint, delta.abs(), rms_feature), dim=-1))
        normalized_delta = F.layer_norm(delta, (delta.shape[-1],))
        return torch.sum(field * normalized_delta, dim=-1) / math.sqrt(delta.shape[-1])


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
        if device.index is None or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"explicit CUDA device is unavailable: {name}")
        torch.cuda.set_device(device)
    return device


def _schedule(update: int, warmup: int, total: int, minimum: float) -> float:
    if update < warmup:
        return max(minimum, (update + 1) / max(1, warmup))
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))


def _rank_numpy(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _rank_numpy(np.asarray(left, dtype=np.float64))
    right_rank = _rank_numpy(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def predict_v52b(
    model: V52BDirectionProbe, loader: DataLoader[Any], device: torch.device
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            logits = model(
                batch["minus"],
                batch["plus"],
                batch["timestep"],
                batch["step_number"],
                batch["rms_ratio"],
                batch["attention_mask"],
            )
            for index, pair_id in enumerate(raw["pair_id"]):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "logit": float(logits[index].detach().cpu()),
                        "target": float(raw["true_target"][index]),
                        "training_target": float(raw["target"][index]),
                        "exact_derivative": float(raw["exact_derivative"][index]),
                        "rms_ratio": float(raw["rms_ratio"][index]),
                        "step_number": int(raw["step_number"][index]),
                    }
                )
    return rows


def _training_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    successes = sum((row["logit"] > 0.0) == (row["training_target"] > 0.5) for row in rows)
    return {
        "pairs": float(len(rows)),
        "successes": float(successes),
        "agreement": successes / len(rows),
        "derivative_spearman": _spearman(
            [float(row["exact_derivative"]) for row in rows],
            [float(row["logit"]) / (2.0 * float(row["rms_ratio"])) for row in rows],
        ),
    }


def _train_seed(
    *,
    seed: int,
    device_name: str,
    fingerprint_path: Path,
    pair_manifest_path: Path,
    preparation_path: Path,
    config_path: Path,
    output_dir: Path,
    variant: str,
    target_mode: str,
) -> dict[str, Any]:
    device = _device(device_name)
    _seed_everything(seed)
    contract = load_fingerprint_contract(fingerprint_path)
    model_config, probe_config, training = load_v52b_config(config_path)
    records = read_v52b_pairs(pair_manifest_path)
    train_dataset = V52BPairDataset(records, "train", target_mode=target_mode)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=training.num_workers,
        collate_fn=collate_v52b_pairs,
        pin_memory=device.type == "cuda",
    )
    audit_loader = DataLoader(
        train_dataset,
        batch_size=training.micro_batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=collate_v52b_pairs,
        pin_memory=device.type == "cuda",
    )
    model = V52BDirectionProbe(contract, model_config, probe_config, variant).to(device)
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
        lambda update: _schedule(update, warmup_updates, total_updates, minimum_ratio),
    )
    best_objective = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history = []
    for epoch in range(1, training.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _to_device(raw, device)
            logits = model(
                batch["minus"],
                batch["plus"],
                batch["timestep"],
                batch["step_number"],
                batch["rms_ratio"],
                batch["attention_mask"],
            )
            loss = F.binary_cross_entropy_with_logits(logits.float(), batch["target"].float())
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V5.2b training loss")
            (loss / accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        metrics = _training_metrics(predict_v52b(model, audit_loader, device))
        objective = 1.0 - metrics["agreement"] + 0.25 * (1.0 - metrics["derivative_spearman"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selection_objective": objective,
                **{f"train_{key}": value for key, value in metrics.items()},
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
        if epoch >= training.minimum_epochs and stale >= training.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("V5.2b training produced no checkpoint")
    checkpoint_path = output_dir / f"v52b_seed_{seed}.pt"
    temporary = checkpoint_path.with_suffix(".part.pt")
    torch.save(
        {
            "schema_version": 1,
            "experiment": V52B_EXPERIMENT,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "guidance_promotion_eligible": False,
            "variant": variant,
            "target_mode": target_mode,
            "state_dict": best_state,
            "model_config": asdict(model_config),
            "probe_config": asdict(probe_config),
            "training_config": asdict(training),
            "fingerprint_sha256": sha256_file(fingerprint_path),
            "pair_manifest_sha256": sha256_file(pair_manifest_path),
            "preparation_sha256": sha256_file(preparation_path),
            "config_sha256": sha256_file(config_path),
            "seed": seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_selection_objective": best_objective,
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
    }


def train_v52b_ensemble(
    *,
    fingerprint_path: Path,
    pair_manifest_path: Path,
    preparation_path: Path,
    config_path: Path,
    output_dir: Path,
    variant: str,
    target_mode: str,
    device_name: str | None = None,
    device_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if variant not in V52B_VARIANTS or target_mode not in V52B_TARGET_MODES:
        raise LTSNContractError("unknown V5.2b variant or target mode")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if (
        preparation.get("experiment") != V52B_EXPERIMENT
        or preparation.get("diagnostic_only") is not True
        or preparation.get("qualification_eligible") is not False
        or preparation.get("guidance_promotion_eligible") is not False
        or preparation.get("pair_manifest_sha256") != sha256_file(pair_manifest_path)
    ):
        raise LTSNContractError("invalid or stale V5.2b preparation")
    _, _, training = load_v52b_config(config_path)
    if device_name is not None and device_names is not None:
        raise LTSNContractError("--device and --devices are mutually exclusive")
    if device_names is None:
        devices = (device_name or ("cuda" if torch.cuda.is_available() else "cpu"),)
    else:
        devices = tuple(str(value) for value in device_names)
        if len(devices) != len(training.seeds) or len(set(devices)) != len(devices):
            raise LTSNContractError("parallel V5.2b training requires one unique device per seed")
        if any(torch.device(value).type != "cuda" for value in devices):
            raise LTSNContractError("parallel V5.2b devices must be CUDA devices")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_path = output_dir / "ensemble_manifest.json"
    if ensemble_path.exists():
        raise LTSNContractError("V5.2b ensemble already exists; use a new output directory")
    common = {
        "fingerprint_path": fingerprint_path.resolve(),
        "pair_manifest_path": pair_manifest_path.resolve(),
        "preparation_path": preparation_path.resolve(),
        "config_path": config_path.resolve(),
        "output_dir": output_dir.resolve(),
        "variant": variant,
        "target_mode": target_mode,
    }
    if len(devices) == 1:
        checkpoints = [
            _train_seed(seed=seed, device_name=devices[0], **common) for seed in training.seeds
        ]
    else:
        context = multiprocessing.get_context("spawn")
        by_seed: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = {
                executor.submit(_train_seed, seed=seed, device_name=device, **common): seed
                for seed, device in zip(training.seeds, devices, strict=True)
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    by_seed[seed] = future.result()
                except Exception as error:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"parallel V5.2b seed {seed} failed: {error}") from error
        checkpoints = [by_seed[seed] for seed in training.seeds]
    ensemble = {
        "schema_version": 1,
        "experiment": V52B_EXPERIMENT,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "primary_variant": PRIMARY_VARIANT,
        "variant": variant,
        "target_mode": target_mode,
        "precision": "fp32",
        "device": devices[0] if len(devices) == 1 else "parallel",
        "devices": list(devices),
        "parallel_training": len(devices) > 1,
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "preparation_sha256": sha256_file(preparation_path),
        "config_sha256": sha256_file(config_path),
        "checkpoints": checkpoints,
    }
    write_json_atomic(ensemble_path, ensemble)
    ensemble["ensemble_manifest_sha256"] = sha256_file(ensemble_path)
    return ensemble


def load_v52b_model(
    checkpoint_path: Path, contract: FingerprintContract, device: torch.device
) -> tuple[V52BDirectionProbe, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("experiment") != V52B_EXPERIMENT
        or payload.get("diagnostic_only") is not True
        or payload.get("qualification_eligible") is not False
        or payload.get("guidance_promotion_eligible") is not False
    ):
        raise LTSNContractError("checkpoint is not a bounded V5.2b diagnostic")
    model_raw = dict(payload["model_config"])
    model_raw["inactive_coordinate_indices"] = tuple(
        model_raw.get("inactive_coordinate_indices", ())
    )
    model = V52BDirectionProbe(
        contract,
        LTSNConfig(**model_raw),
        V52BProbeConfig(**payload["probe_config"]),
        str(payload["variant"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
