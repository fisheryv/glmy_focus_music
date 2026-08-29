"""Hashed per-snapshot LTSN dataset with prompt/trajectory leakage checks."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .ltsn_contract import FingerprintContract, LTSNContractError, sha256_file

_SPLITS = {"train", "development", "calibration", "qualification"}


@dataclass(frozen=True, slots=True)
class LTSNSnapshot:
    """One independently decoded and exact-labelled clean-latent snapshot."""

    sample_id: str
    prompt_id: str
    trajectory_id: str
    split: str
    step_number: int
    timestep: float
    latent_path: Path
    latent_sha256: str
    coordinates: tuple[float, ...]
    focus_logit: float
    ood_label: float
    is_final: bool
    exact_label_table_sha256: str


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def read_ltsn_manifest(path: Path, contract: FingerprintContract) -> list[LTSNSnapshot]:
    """Read a CSV manifest and reject stale hashes, 51-D labels, and copied final labels."""

    rows: list[LTSNSnapshot] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("fingerprint_json_sha256", "").lower() != contract.artifact_sha256:
                raise LTSNContractError("snapshot fingerprint JSON SHA-256 mismatch")
            if raw.get("label_scope") != "per_snapshot_exact":
                raise LTSNContractError("every x0_hat snapshot requires its own exact label")
            feature_order = json.loads(raw.get("feature_order_json", "null"))
            if feature_order != list(contract.feature_order):
                raise LTSNContractError("snapshot feature order differs from the scorer")
            coordinates = tuple(float(value) for value in json.loads(raw["coordinates_json"]))
            if len(coordinates) != 18 or not all(math.isfinite(value) for value in coordinates):
                raise LTSNContractError("snapshot coordinates must be 18 finite values")
            split = raw["split"].strip().lower()
            if split not in _SPLITS:
                raise LTSNContractError(f"unsupported LTSN split: {split}")
            step_number = int(raw["step_number"])
            is_final = _parse_bool(raw.get("is_final", "false"))
            if not is_final and step_number not in {4, 5, 6}:
                raise LTSNContractError("non-final training snapshots must be steps 4, 5, or 6")
            latent_path = (path.parent / raw["latent_path"]).resolve()
            if not latent_path.is_file() or sha256_file(latent_path) != raw.get(
                "latent_sha256", ""
            ):
                raise LTSNContractError("snapshot latent is missing or hash-mismatched")
            exact_label_path = (path.parent / raw["exact_label_table_path"]).resolve()
            if not exact_label_path.is_file() or sha256_file(exact_label_path) != raw.get(
                "exact_label_table_sha256", ""
            ):
                raise LTSNContractError("exact label table is missing or hash-mismatched")
            rows.append(
                LTSNSnapshot(
                    sample_id=raw["sample_id"],
                    prompt_id=raw["prompt_id"],
                    trajectory_id=raw["trajectory_id"],
                    split=split,
                    step_number=step_number,
                    timestep=float(raw["timestep"]),
                    latent_path=latent_path,
                    latent_sha256=raw["latent_sha256"].lower(),
                    coordinates=coordinates,
                    focus_logit=float(raw["focus_logit"]),
                    ood_label=float(raw["ood_label"]),
                    is_final=is_final,
                    exact_label_table_sha256=raw["exact_label_table_sha256"].lower(),
                )
            )
    if not rows:
        raise LTSNContractError("LTSN manifest is empty")
    validate_group_splits(rows)
    if len({row.exact_label_table_sha256 for row in rows}) != 1:
        raise LTSNContractError("manifest mixes exact label tables")
    return rows


def validate_group_splits(records: Iterable[LTSNSnapshot]) -> None:
    """Ensure each prompt and trajectory belongs to exactly one partition."""

    prompt_splits: dict[str, set[str]] = {}
    trajectory_splits: dict[str, set[str]] = {}
    for record in records:
        prompt_splits.setdefault(record.prompt_id, set()).add(record.split)
        trajectory_splits.setdefault(record.trajectory_id, set()).add(record.split)
    leaking_prompts = [key for key, values in prompt_splits.items() if len(values) != 1]
    leaking_trajectories = [key for key, values in trajectory_splits.items() if len(values) != 1]
    if leaking_prompts or leaking_trajectories:
        raise LTSNContractError("prompt or trajectory leakage detected across LTSN splits")


class LTSNSnapshotDataset(Dataset[dict[str, Tensor | str]]):
    """Load precomputed ``[T,64]`` x0_hat arrays and their exact 18-D labels."""

    def __init__(self, records: Sequence[LTSNSnapshot], split: str) -> None:
        self.records = tuple(record for record in records if record.split == split)
        if not self.records:
            raise ValueError(f"no LTSN records for split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        latent = np.load(record.latent_path, allow_pickle=False)
        if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
            raise LTSNContractError(f"invalid ACE latent: {record.latent_path}")
        return {
            "sample_id": record.sample_id,
            "prompt_id": record.prompt_id,
            "trajectory_id": record.trajectory_id,
            "latent": torch.from_numpy(np.asarray(latent, dtype=np.float32)),
            "timestep": torch.tensor(record.timestep, dtype=torch.float32),
            "step_number": torch.tensor(record.step_number, dtype=torch.long),
            "coordinates": torch.tensor(record.coordinates, dtype=torch.float32),
            "focus_logit": torch.tensor(record.focus_logit, dtype=torch.float32),
            "ood_label": torch.tensor(record.ood_label, dtype=torch.float32),
        }


def collate_ltsn_batch(items: Sequence[dict[str, Tensor | str]]) -> dict[str, object]:
    """Pad variable-duration latents and create the required valid-frame mask."""

    if not items:
        raise ValueError("cannot collate an empty LTSN batch")
    lengths = [int(item["latent"].shape[0]) for item in items]  # type: ignore[union-attr]
    maximum = max(lengths)
    batch = torch.zeros(len(items), maximum, 64, dtype=torch.float32)
    mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        latent = item["latent"]
        assert isinstance(latent, Tensor)
        batch[index, : latent.shape[0]] = latent
        mask[index, : latent.shape[0]] = True
    tensor_keys = ("timestep", "step_number", "coordinates", "focus_logit", "ood_label")
    result: dict[str, object] = {"latent": batch, "attention_mask": mask}
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in items])  # type: ignore[list-item]
    for key in ("sample_id", "prompt_id", "trajectory_id"):
        result[key] = [str(item[key]) for item in items]
    return result


def manifest_identity(path: Path, records: Sequence[LTSNSnapshot]) -> dict[str, str]:
    """Return checkpoint-ready manifest and exact-label table digests."""

    return {
        "training_manifest_sha256": sha256_file(path),
        "exact_label_table_sha256": records[0].exact_label_table_sha256,
    }
