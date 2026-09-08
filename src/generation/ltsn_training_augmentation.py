"""Exact-labelled local and held-out OOD augmentation for LTSN V3."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import AceStepAdapter
from .experiment import load_experiment_config
from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_dataset import LTSNSnapshot, read_ltsn_manifest
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import canonical_json_sha256, write_csv_atomic, write_json_atomic
from .path_homology_exact_scorer import ExactPathHomologyScorer


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary, path)


def _smooth_direction(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape, dtype=np.float32)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(raw, ((2, 2), (0, 0)), mode="edge")
    smooth = sum(
        weight * padded[offset : offset + shape[0]] for offset, weight in enumerate(kernel)
    )
    return np.asarray(smooth, dtype=np.float32)


def _local_perturbation(
    latent: np.ndarray, *, seed: int, rms_ratio: float, sign: float
) -> np.ndarray:
    direction = _smooth_direction(latent.shape, seed)
    latent_rms = float(np.sqrt(np.mean(np.square(latent, dtype=np.float64))))
    direction_rms = float(np.sqrt(np.mean(np.square(direction, dtype=np.float64))))
    if latent_rms <= 0 or direction_rms <= 0:
        raise LTSNContractError("local perturbation requires non-zero latent and direction RMS")
    update = direction * (sign * rms_ratio * latent_rms / direction_rms)
    result = latent + update
    if not np.isfinite(result).all():
        raise LTSNContractError("local perturbation produced NaN or Inf")
    return result.astype(np.float32, copy=False)


class _OnPolicyDirectionProvider:
    """Generate local candidates from the actual frozen ensemble gradient."""

    def __init__(
        self,
        *,
        ensemble_manifest: Path,
        fingerprint_path: Path,
        device_name: str,
    ) -> None:
        import torch
        from torch.nn import functional as functional

        from .ltsn_evaluation import _load_ensemble

        self.torch = torch
        self.functional = functional
        self.device = torch.device(device_name)
        self.contract, self.ensemble, self.models = _load_ensemble(
            ensemble_manifest, fingerprint_path, self.device
        )
        self.ensemble_manifest = ensemble_manifest
        self._cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    def _low_pass(self, gradient: Any) -> Any:
        torch = self.torch
        kernel = torch.tensor(
            [1.0, 2.0, 3.0, 2.0, 1.0], device=gradient.device, dtype=gradient.dtype
        )
        kernel /= kernel.sum()
        channels = gradient.shape[-1]
        weights = kernel.reshape(1, 1, -1).expand(channels, 1, -1)
        channel_first = gradient.transpose(1, 2)
        padded = self.functional.pad(channel_first, (2, 2), mode="replicate")
        return self.functional.conv1d(padded, weights, groups=channels).transpose(1, 2)

    def describe(self, anchor: LTSNSnapshot) -> dict[str, Any]:
        if anchor.sample_id in self._cache:
            return self._cache[anchor.sample_id][1]
        torch = self.torch
        latent = np.load(anchor.latent_path, allow_pickle=False).astype(np.float32, copy=False)
        clean = torch.from_numpy(latent).to(self.device).unsqueeze(0).detach().clone()
        clean.requires_grad_(True)
        mask = torch.ones(clean.shape[:2], device=self.device, dtype=torch.bool)
        outputs = [model(clean, anchor.timestep, anchor.step_number, mask) for model in self.models]
        scores = torch.stack([output.focus_logit.float() for output in outputs])
        mean_score = scores.mean(dim=0)
        threshold = float(self.contract.focus_band_threshold)
        ensemble_energy = self.functional.relu(threshold - mean_score).square()
        gradient = torch.autograd.grad(
            ensemble_energy.sum(), clean, retain_graph=True, allow_unused=False
        )[0]
        member_energy = self.functional.relu(threshold - scores).square()
        member_gradients = torch.stack(
            [
                torch.autograd.grad(
                    member_energy[index].sum(),
                    clean,
                    retain_graph=index + 1 < len(outputs),
                    allow_unused=False,
                )[0]
                for index in range(len(outputs))
            ]
        )
        direction = -self._low_pass(gradient)
        latent_rms = torch.sqrt(clean.square().mean())
        direction_rms = torch.sqrt(direction.square().mean())
        usable = bool(direction_rms.item() > 1e-12 and torch.isfinite(direction).all())
        base_update = (
            direction * (latent_rms / direction_rms.clamp_min(1e-12))
            if usable
            else torch.zeros_like(direction)
        )
        if len(outputs) < 2:
            minimum_cosine = 1.0
        else:
            cosines: list[float] = []
            flattened = member_gradients.flatten(2)
            for left in range(len(outputs)):
                for right in range(left + 1, len(outputs)):
                    left_value = flattened[left, 0]
                    right_value = flattened[right, 0]
                    denominator = left_value.norm() * right_value.norm()
                    cosines.append(
                        -1.0
                        if denominator.item() <= 1e-12
                        else float((left_value @ right_value / denominator).item())
                    )
            minimum_cosine = min(cosines)
        description = {
            "exact_focus_logit": float(anchor.focus_logit),
            "proxy_focus_logit": float(mean_score.item()),
            "member_focus_logits": [float(value) for value in scores[:, 0].detach().cpu()],
            "exact_out_of_band": bool(anchor.focus_logit < threshold),
            "proxy_out_of_band": bool(mean_score.item() < threshold),
            "all_members_out_of_band": bool((scores[:, 0] < threshold).all().item()),
            "member_gradient_cosine": minimum_cosine,
            "gradient_usable": usable,
        }
        self._cache[anchor.sample_id] = (
            base_update[0].detach().cpu().numpy().astype(np.float32, copy=False),
            description,
        )
        return description

    def perturb(self, anchor: LTSNSnapshot, rms_ratio: float) -> tuple[np.ndarray, dict[str, Any]]:
        description = self.describe(anchor)
        base_update = self._cache[anchor.sample_id][0]
        latent = np.load(anchor.latent_path, allow_pickle=False).astype(np.float32, copy=False)
        result = latent + float(rms_ratio) * base_update
        if not np.isfinite(result).all():
            raise LTSNContractError("on-policy augmentation produced NaN or Inf")
        torch = self.torch
        with torch.inference_mode():
            values = torch.from_numpy(result).to(self.device).unsqueeze(0)
            mask = torch.ones(values.shape[:2], device=self.device, dtype=torch.bool)
            after = torch.stack(
                [
                    model(values, anchor.timestep, anchor.step_number, mask).focus_logit.float()
                    for model in self.models
                ]
            )
        threshold = float(self.contract.focus_band_threshold)
        before_loss = max(0.0, threshold - float(description["proxy_focus_logit"])) ** 2
        after_score = float(after.mean().item())
        diagnostics = {
            **description,
            "rms_ratio": float(rms_ratio),
            "proxy_focus_logit_after": after_score,
            "proxy_band_loss_improvement": (before_loss - max(0.0, threshold - after_score) ** 2),
        }
        return result.astype(np.float32, copy=False), diagnostics

    def retain(self, sample_ids: set[str]) -> None:
        """Release rejected anchor directions before loading the ACE decoder."""

        self._cache = {
            sample_id: value for sample_id, value in self._cache.items() if sample_id in sample_ids
        }


def _select_on_policy_anchors(
    records: list[LTSNSnapshot],
    provider: _OnPolicyDirectionProvider,
    trajectories_per_prompt: int,
) -> tuple[list[LTSNSnapshot], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, list[LTSNSnapshot]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if (
            record.split in {"train", "development"}
            and not record.is_final
            and record.step_number in {4, 5, 6}
            and not record.local_anchor_sample_id
        ):
            grouped[(record.split, record.prompt_id)][record.trajectory_id].append(record)
    selected: list[LTSNSnapshot] = []
    descriptions: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        candidates: list[LTSNSnapshot] = []
        for trajectory_id in sorted(grouped[key])[:trajectories_per_prompt]:
            candidates.extend(grouped[key][trajectory_id])
        scored = []
        for candidate in candidates:
            description = provider.describe(candidate)
            descriptions[candidate.sample_id] = description
            if description["proxy_out_of_band"] and description["gradient_usable"]:
                scored.append((candidate, description))
        if not scored:
            continue
        # Prefer true out-of-band anchors; within each category stay near the
        # frozen boundary where local direction is most consequential.
        candidate, _ = min(
            scored,
            key=lambda item: (
                not item[1]["exact_out_of_band"],
                abs(float(item[1]["exact_focus_logit"]) - provider.contract.focus_band_threshold),
                item[0].step_number,
                item[0].sample_id,
            ),
        )
        selected.append(candidate)
    if not selected:
        raise LTSNContractError("no proxy-out-of-band on-policy anchors are available")
    provider.retain({anchor.sample_id for anchor in selected})
    return selected, descriptions


def _on_policy_plan(
    anchors: list[LTSNSnapshot],
    descriptions: dict[str, dict[str, Any]],
    rms_ratios: tuple[float, ...],
) -> list[dict[str, Any]]:
    planned = []
    for anchor in anchors:
        for rms_ratio in rms_ratios:
            tag = int(round(rms_ratio * 10_000))
            planned.append(
                {
                    "sample_id": f"{anchor.sample_id}__onpolicy_r{tag:04d}",
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": anchor.prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": "on_policy_direction",
                    "split": anchor.split,
                    "seed": 0,
                    "rms_ratio": rms_ratio,
                    "sign": -1.0,
                    "is_final": False,
                    "anchor_diagnostics": descriptions[anchor.sample_id],
                }
            )
    return planned


def _select_anchors(
    records: list[LTSNSnapshot], trajectories_per_prompt: int
) -> list[LTSNSnapshot]:
    grouped: dict[str, dict[str, list[LTSNSnapshot]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.split == "train" and not record.is_final and record.step_number in {4, 5, 6}:
            grouped[record.prompt_id][record.trajectory_id].append(record)
    selected: list[LTSNSnapshot] = []
    for prompt_id in sorted(grouped):
        trajectories = grouped[prompt_id]
        for trajectory_id in sorted(trajectories)[:trajectories_per_prompt]:
            rows = sorted(trajectories[trajectory_id], key=lambda item: item.step_number)
            if {row.step_number for row in rows} != {4, 5, 6}:
                raise LTSNContractError(
                    f"selected augmentation trajectory lacks steps 4/5/6: {trajectory_id}"
                )
            selected.extend(rows)
    if not selected:
        raise LTSNContractError("no train step 4/5/6 anchors are available for augmentation")
    return selected


def _augmentation_plan(
    anchors: list[LTSNSnapshot],
    *,
    perturbations_per_anchor: int,
    rms_ratio: float,
    ood_per_prompt: int,
    seed: int,
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    prompt_step_ood: dict[tuple[str, int], int] = defaultdict(int)
    prompt_rank = {
        prompt_id: index
        for index, prompt_id in enumerate(sorted({anchor.prompt_id for anchor in anchors}))
    }
    for anchor_index, anchor in enumerate(anchors):
        for perturbation_index in range(perturbations_per_anchor):
            sign = -1.0 if perturbation_index % 2 else 1.0
            item_seed = seed + anchor_index * max(perturbations_per_anchor, 1) + perturbation_index
            sample_id = f"{anchor.sample_id}__local{perturbation_index:02d}"
            planned.append(
                {
                    "sample_id": sample_id,
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": anchor.prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": "local_direction",
                    "split": "train",
                    "seed": item_seed,
                    "rms_ratio": rms_ratio,
                    "sign": sign,
                    "is_final": False,
                }
            )
        ood_key = (anchor.prompt_id, anchor.step_number)
        if prompt_step_ood[ood_key] < ood_per_prompt:
            ood_index = prompt_step_ood[ood_key]
            prompt_step_ood[ood_key] += 1
            planned.append(
                {
                    "sample_id": f"{anchor.sample_id}__ood{ood_index:02d}",
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": anchor.prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": (
                        "ood_zero"
                        if (prompt_rank[anchor.prompt_id] + ood_index) % 2 == 0
                        else "ood_scale_high"
                    ),
                    "split": "train",
                    "seed": seed + 10_000_000 + anchor_index,
                    "rms_ratio": 0.0,
                    "sign": 0.0,
                    "is_final": False,
                }
            )
    sample_ids = [item["sample_id"] for item in planned]
    if len(set(sample_ids)) != len(sample_ids):
        raise LTSNContractError("augmentation plan contains duplicate sample IDs")
    return planned


def _evaluation_ood_plan(
    records: list[LTSNSnapshot],
    *,
    ood_per_prompt: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create prompt-held-out OOD examples with transforms unseen in training."""

    if ood_per_prompt < 0:
        raise ValueError("evaluation OOD count must be non-negative")
    grouped: dict[tuple[str, str, int], list[LTSNSnapshot]] = defaultdict(list)
    for record in records:
        if record.split in {"calibration", "qualification"} and record.step_number in {
            4,
            5,
            6,
            8,
        }:
            grouped[(record.split, record.prompt_id, record.step_number)].append(record)
    planned: list[dict[str, Any]] = []
    for prompt_rank, ((split, prompt_id, step_number), candidates) in enumerate(
        sorted(grouped.items())
    ):
        ordered = sorted(candidates, key=lambda item: (item.trajectory_id, item.sample_id))
        if len(ordered) < ood_per_prompt:
            raise LTSNContractError(
                f"{split} prompt lacks {ood_per_prompt} step-{step_number} OOD anchors: {prompt_id}"
            )
        for ood_index, anchor in enumerate(ordered[:ood_per_prompt]):
            kind = "ood_time_reverse" if (prompt_rank + ood_index) % 2 == 0 else "ood_channel_roll"
            planned.append(
                {
                    "sample_id": f"{anchor.sample_id}__heldout_ood{ood_index:02d}",
                    "anchor_sample_id": anchor.sample_id,
                    "prompt_id": prompt_id,
                    "step_number": anchor.step_number,
                    "timestep": anchor.timestep,
                    "kind": kind,
                    "split": split,
                    "seed": seed + 20_000_000 + prompt_rank * max(ood_per_prompt, 1) + ood_index,
                    "rms_ratio": 0.0,
                    "sign": 0.0,
                    "is_final": anchor.is_final,
                }
            )
    evaluation_prompts = {
        (record.split, record.prompt_id)
        for record in records
        if record.split in {"calibration", "qualification"}
    }
    expected_groups = {
        (split, prompt_id, step) for split, prompt_id in evaluation_prompts for step in (4, 5, 6, 8)
    }
    if ood_per_prompt and set(grouped) != expected_groups:
        raise LTSNContractError(
            "every calibration/qualification prompt must provide step 4/5/6/8 OOD anchors"
        )
    return planned


def _validate_receipt(path: Path, output_dir: Path, plan_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_sha256") != plan_sha256:
        raise LTSNContractError("augmentation receipt belongs to a different plan")
    for name in ("latent", "audio"):
        artifact = output_dir / payload[f"{name}_path"]
        if not artifact.is_file() or sha256_file(artifact) != payload[f"{name}_sha256"]:
            raise LTSNContractError(f"augmentation receipt {name} is hash-mismatched")
    return payload


def _materialize_augmentation(
    *,
    item: dict[str, Any],
    anchor: LTSNSnapshot,
    adapter: AceStepAdapter,
    output_dir: Path,
    plan_sha256: str,
    on_policy_provider: _OnPolicyDirectionProvider | None = None,
) -> dict[str, Any]:
    receipt_path = output_dir / "receipts" / f"{item['sample_id']}.json"
    if receipt_path.is_file():
        receipt = _validate_receipt(receipt_path, output_dir, plan_sha256)
        if receipt.get("sample_id") != item["sample_id"]:
            raise LTSNContractError("augmentation receipt sample binding mismatch")
        return receipt
    latent = np.load(anchor.latent_path, allow_pickle=False).astype(np.float32, copy=False)
    if latent.ndim != 2 or latent.shape[1] != 64 or not np.isfinite(latent).all():
        raise LTSNContractError(f"invalid augmentation anchor latent: {anchor.sample_id}")
    on_policy_diagnostics: dict[str, Any] | None = None
    if item["kind"] == "local_direction":
        augmented = _local_perturbation(
            latent,
            seed=int(item["seed"]),
            rms_ratio=float(item["rms_ratio"]),
            sign=float(item["sign"]),
        )
    elif item["kind"] == "on_policy_direction":
        if on_policy_provider is None:
            raise LTSNContractError("on-policy augmentation requires a frozen ensemble")
        augmented, on_policy_diagnostics = on_policy_provider.perturb(
            anchor, float(item["rms_ratio"])
        )
    elif item["kind"] == "ood_zero":
        augmented = np.zeros_like(latent)
    elif item["kind"] == "ood_scale_high":
        augmented = latent * 4.0
    elif item["kind"] == "ood_time_reverse":
        augmented = np.ascontiguousarray(latent[::-1])
    elif item["kind"] == "ood_channel_roll":
        augmented = np.roll(latent, shift=17, axis=1).copy()
    else:
        raise LTSNContractError(f"unknown augmentation kind: {item['kind']}")
    latent_path = output_dir / "latents" / f"{item['sample_id']}.npy"
    audio_path = output_dir / "data_raw" / "local_augmentation" / f"{item['sample_id']}.wav"
    if latent_path.is_file():
        existing = np.load(latent_path, allow_pickle=False)
        if existing.dtype != np.float32 or not np.array_equal(existing, augmented):
            raise LTSNContractError(
                f"unreceipted augmentation latent is mismatched: {item['sample_id']}"
            )
    elif latent_path.exists():
        raise LTSNContractError(f"augmentation latent path is not a file: {item['sample_id']}")
    else:
        _save_npy_atomic(latent_path, augmented)
    if audio_path.exists() and not audio_path.is_file():
        raise LTSNContractError(f"augmentation audio path is not a file: {item['sample_id']}")
    # Re-decode any unreceipted audio atomically. This recovers a crash between
    # derived-artifact creation and the receipt without trusting stale audio.
    adapter.decode_latent_to_audio(augmented, audio_path)
    receipt = {
        "schema_version": 1,
        "sample_id": item["sample_id"],
        "anchor_sample_id": anchor.sample_id,
        "kind": item["kind"],
        "latent_path": latent_path.relative_to(output_dir).as_posix(),
        "latent_sha256": sha256_file(latent_path),
        "audio_path": audio_path.relative_to(output_dir).as_posix(),
        "audio_sha256": sha256_file(audio_path),
        "plan_sha256": plan_sha256,
        "on_policy_diagnostics": on_policy_diagnostics,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _rectangular(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def _write_augmentation_trajectory_manifest(
    *,
    path: Path,
    planned: list[dict[str, Any]],
    source_by_sample: dict[str, dict[str, str]],
    receipt_by_id: dict[str, dict[str, Any]],
) -> None:
    """Issue the minimal hash-bound manifest consumed by exact batch labeling."""

    rows: list[dict[str, Any]] = []
    for item in planned:
        anchor = source_by_sample[item["anchor_sample_id"]]
        receipt = receipt_by_id[item["sample_id"]]
        rows.append(
            {
                "sample_id": item["sample_id"],
                "prompt_id": item["prompt_id"],
                "trajectory_id": item["sample_id"],
                "split": item["split"],
                "model_family": anchor["model_family"],
                "step_number": item["step_number"],
                "timestep": item["timestep"],
                "latent_path": receipt["latent_path"],
                "latent_sha256": receipt["latent_sha256"],
                "audio_path": receipt["audio_path"],
                "audio_sha256": receipt["audio_sha256"],
                "is_final": str(bool(item.get("is_final", False))).lower(),
                "ace_model_sha256": anchor["ace_model_sha256"],
                "vae_sha256": anchor["vae_sha256"],
                "training_augmentation_kind": item["kind"],
                "local_anchor_sample_id": (
                    item["anchor_sample_id"]
                    if item["kind"] in {"local_direction", "on_policy_direction"}
                    else ""
                ),
            }
        )
    write_csv_atomic(path, rows)


def build_ltsn_training_augmentation(
    *,
    root: Path,
    source_manifest_path: Path,
    source_split_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    trajectories_per_prompt: int = 1,
    perturbations_per_anchor: int = 2,
    rms_ratio: float = 0.005,
    ood_per_prompt: int = 1,
    evaluation_ood_per_prompt: int = 1,
    seed: int = 2026071600,
    duration_seconds: float = 180.0,
    workers: int = 8,
    exact_batch_size: int = 256,
    materialize_mode: str = "auto",
    cleanup_exact_batches: bool = True,
    device_name: str = "cuda:0",
    resume: bool = False,
    local_mode: str = "random",
    on_policy_ensemble_manifest: Path | None = None,
    on_policy_rms_ratios: tuple[float, ...] = (0.0025, 0.005, 0.01),
) -> dict[str, Any]:
    """Append train-only local/OOD and prompt-held-out evaluation OOD records."""

    if (
        trajectories_per_prompt < 1
        or perturbations_per_anchor < 1
        or ood_per_prompt < 0
        or evaluation_ood_per_prompt < 0
    ):
        raise ValueError("augmentation counts are invalid")
    if rms_ratio not in {0.0025, 0.005, 0.01}:
        raise ValueError("rms_ratio must be 0.0025, 0.005, or 0.01")
    if local_mode not in {"random", "on_policy"}:
        raise ValueError("local_mode must be random or on_policy")
    if not on_policy_rms_ratios or any(
        value not in {0.0025, 0.005, 0.01} for value in on_policy_rms_ratios
    ):
        raise ValueError("on-policy RMS ratios must use 0.0025, 0.005, or 0.01")
    if duration_seconds != 180.0:
        raise ValueError("exact LTSN training augmentation is frozen to 180 seconds")
    root = root.resolve()
    output_dir = output_dir.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    source_manifest_path = resolve(source_manifest_path)
    source_split_manifest_path = resolve(source_split_manifest_path)
    ace_config_path = resolve(ace_config_path)
    fingerprint_path = resolve(fingerprint_path)
    if on_policy_ensemble_manifest is not None:
        on_policy_ensemble_manifest = resolve(on_policy_ensemble_manifest)
    contract = load_fingerprint_contract(fingerprint_path)
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    source_records = read_ltsn_manifest(source_manifest_path, contract)
    source_rows = _read_csv(source_manifest_path)
    if {row.get("ace_model_sha256", "") for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("augmentation ACE model hash differs from the source manifest")
    if {row.get("vae_sha256", "") for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("augmentation VAE hash differs from the source manifest")
    anchors = _select_anchors(source_records, trajectories_per_prompt)
    on_policy_provider: _OnPolicyDirectionProvider | None = None
    if local_mode == "on_policy":
        if on_policy_ensemble_manifest is None or not on_policy_ensemble_manifest.is_file():
            raise LTSNContractError("on-policy augmentation requires --on-policy-ensemble-manifest")
        on_policy_provider = _OnPolicyDirectionProvider(
            ensemble_manifest=on_policy_ensemble_manifest,
            fingerprint_path=fingerprint_path,
            device_name=device_name,
        )
        ensemble_metadata = on_policy_provider.ensemble.get("metadata", {})
        if ensemble_metadata.get("ace_model_sha256") != ace_model_sha256:
            raise LTSNContractError("on-policy ensemble uses a different ACE model")
        if ensemble_metadata.get("vae_sha256") != vae_sha256:
            raise LTSNContractError("on-policy ensemble uses a different VAE")
        source_families = {row.get("model_family", "") for row in source_rows}
        if source_families != {ensemble_metadata.get("model_family")}:
            raise LTSNContractError("on-policy ensemble uses a different model family")
        on_policy_anchors, descriptions = _select_on_policy_anchors(
            source_records, on_policy_provider, trajectories_per_prompt
        )
        planned = _on_policy_plan(
            on_policy_anchors, descriptions, tuple(sorted(set(on_policy_rms_ratios)))
        )
        # OOD examples are independent of the local-gradient source.  Retain
        # deterministic train OOD coverage at every correction step.
        planned.extend(
            item
            for item in _augmentation_plan(
                anchors,
                perturbations_per_anchor=1,
                rms_ratio=rms_ratio,
                ood_per_prompt=ood_per_prompt,
                seed=seed,
            )
            if item["kind"].startswith("ood_")
        )
    else:
        planned = _augmentation_plan(
            anchors,
            perturbations_per_anchor=perturbations_per_anchor,
            rms_ratio=rms_ratio,
            ood_per_prompt=ood_per_prompt,
            seed=seed,
        )
    planned.extend(
        _evaluation_ood_plan(
            source_records,
            ood_per_prompt=evaluation_ood_per_prompt,
            seed=seed,
        )
    )
    artifact_version = 4 if local_mode == "on_policy" else 3
    plan_path = output_dir / "training_augmentation_plan.json"
    plan = {
        "schema_version": artifact_version,
        "scope": (
            "on_policy_exact_local_and_step_matched_ood_v4"
            if local_mode == "on_policy"
            else "train_local_ood_and_heldout_evaluation_ood_v3"
        ),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "trajectories_per_prompt": trajectories_per_prompt,
        "perturbations_per_anchor": perturbations_per_anchor,
        "rms_ratio": rms_ratio,
        "local_mode": local_mode,
        "on_policy_rms_ratios": list(on_policy_rms_ratios),
        "on_policy_ensemble_manifest_sha256": (
            "" if on_policy_ensemble_manifest is None else sha256_file(on_policy_ensemble_manifest)
        ),
        "ood_per_prompt": ood_per_prompt,
        "evaluation_ood_per_prompt": evaluation_ood_per_prompt,
        "seed": seed,
        "duration_seconds": duration_seconds,
        "exact_batch_size": exact_batch_size,
        "planned": planned,
    }
    if plan_path.is_file():
        if not resume:
            raise FileExistsError("augmentation plan exists; pass --resume or use a new output dir")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("training augmentation plan changed; use a new output dir")
    else:
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    anchor_by_id = {record.sample_id: record for record in source_records}
    receipts = [
        _materialize_augmentation(
            item=item,
            anchor=anchor_by_id[item["anchor_sample_id"]],
            adapter=adapter,
            output_dir=output_dir,
            plan_sha256=plan_sha256,
            on_policy_provider=on_policy_provider,
        )
        for item in planned
    ]
    source_by_sample = {row["sample_id"]: row for row in source_rows}
    receipt_by_id = {receipt["sample_id"]: receipt for receipt in receipts}
    trajectory_manifest_path = output_dir / "training_augmentation_trajectories.csv"
    _write_augmentation_trajectory_manifest(
        path=trajectory_manifest_path,
        planned=planned,
        source_by_sample=source_by_sample,
        receipt_by_id=receipt_by_id,
    )
    descriptor_path = output_dir / "training_augmentation_descriptors.csv"
    storage_summary = build_exact_snapshot_descriptors(
        project_root=root,
        trajectory_manifest=trajectory_manifest_path,
        work_dir=output_dir / "exact_work",
        output_path=descriptor_path,
        workers=workers,
        batch_size=exact_batch_size,
        materialize_mode=materialize_mode,
        cleanup_batches=cleanup_exact_batches,
        resume=resume,
    )
    descriptors = _read_csv(descriptor_path)
    descriptor_by_id = {row["sample_id"]: row for row in descriptors}

    old_label_path = (
        source_manifest_path.parent / source_rows[0]["exact_label_table_path"]
    ).resolve()
    old_labels = _read_csv(old_label_path)
    label_columns = list(old_labels[0])
    new_labels: list[dict[str, Any]] = []
    augmented_rows: list[dict[str, Any]] = []
    local_evidence: list[dict[str, Any]] = []
    for item in planned:
        descriptor = descriptor_by_id[item["sample_id"]]
        receipt = receipt_by_id[item["sample_id"]]
        if descriptor.get("label_source") != "decoded_snapshot_exact_v1":
            raise LTSNContractError("training augmentation requires decoded exact labels")
        if descriptor.get("audio_sha256") != receipt["audio_sha256"]:
            raise LTSNContractError(
                f"augmentation descriptor audio hash mismatch: {item['sample_id']}"
            )
        forced_ood = item["kind"].startswith("ood_")
        technical_ood = float(descriptor["ood_label"]) >= 0.5
        ood_label = int(forced_ood or technical_ood)
        pitch = json.loads(descriptor["pitch_descriptors_json"])
        score = scorer.score(
            pitch,
            [float(descriptor["acoustic_loop_score"])],
            [float(descriptor["chroma_loop_score"])],
        )
        coordinates_json = json.dumps(score.coordinates[0].tolist(), separators=(",", ":"))
        if item["kind"] in {"local_direction", "on_policy_direction"}:
            anchor_focus = float(source_by_sample[item["anchor_sample_id"]]["focus_logit"])
            threshold = float(contract.focus_band_threshold)
            exact_before = max(0.0, threshold - anchor_focus) ** 2
            exact_after = max(0.0, threshold - float(score.focus_logit[0])) ** 2
            local_evidence.append(
                {
                    "sample_id": item["sample_id"],
                    "anchor_sample_id": item["anchor_sample_id"],
                    "split": item["split"],
                    "step_number": item["step_number"],
                    "rms_ratio": item["rms_ratio"],
                    "exact_band_loss_before": exact_before,
                    "exact_band_loss_after": exact_after,
                    "exact_band_loss_improvement": exact_before - exact_after,
                    "proxy_band_loss_improvement": (
                        (receipt.get("on_policy_diagnostics") or {}).get(
                            "proxy_band_loss_improvement"
                        )
                    ),
                }
            )
        new_labels.append(
            {
                "sample_id": item["sample_id"],
                "pitch_descriptors_json": json.dumps(pitch, separators=(",", ":")),
                "acoustic_loop_score": descriptor["acoustic_loop_score"],
                "chroma_loop_score": descriptor["chroma_loop_score"],
                "coordinates_json": coordinates_json,
                "focus_logit": float(score.focus_logit[0]),
                "focus_probability": float(score.focus_probability[0]),
                "focus_band_loss": float(score.focus_band_loss[0]),
                "pitch_block_l2_norm": float(score.pitch_block_l2_norm[0]),
                "phase_block_l2_norm": float(score.phase_block_l2_norm[0]),
                "ood_label": ood_label,
                "label_scope": "per_snapshot_exact",
                "fingerprint_json_sha256": contract.artifact_sha256,
            }
        )
        anchor = source_by_sample[item["anchor_sample_id"]]
        augmented = dict(anchor)
        augmented.update(
            {
                "sample_id": item["sample_id"],
                "trajectory_id": item["sample_id"],
                "split": item["split"],
                "step_number": item["step_number"],
                "timestep": item["timestep"],
                "latent_path": os.path.relpath(
                    output_dir / receipt["latent_path"], output_dir
                ).replace("\\", "/"),
                "latent_sha256": receipt["latent_sha256"],
                "coordinates_json": coordinates_json,
                "focus_logit": float(score.focus_logit[0]),
                "ood_label": ood_label,
                "is_final": str(bool(item.get("is_final", False))).lower(),
                "local_anchor_sample_id": (
                    item["anchor_sample_id"]
                    if item["kind"] in {"local_direction", "on_policy_direction"}
                    else ""
                ),
                "training_augmentation_kind": item["kind"],
            }
        )
        augmented_rows.append(augmented)

    exact_label_path = output_dir / f"exact_snapshot_labels_v{artifact_version}.csv"
    write_csv_atomic(
        exact_label_path,
        _rectangular([*old_labels, *new_labels], label_columns),
    )
    label_sha256 = sha256_file(exact_label_path)
    manifest_path = output_dir / f"ltsn_manifest_v{artifact_version}.csv"
    manifest_columns = list(source_rows[0])
    for column in ("local_anchor_sample_id", "training_augmentation_kind"):
        if column not in manifest_columns:
            manifest_columns.append(column)
    for row in source_rows:
        source_latent = (source_manifest_path.parent / row["latent_path"]).resolve()
        row["latent_path"] = os.path.relpath(source_latent, output_dir).replace("\\", "/")
    combined = [*source_rows, *augmented_rows]
    for row in combined:
        row["exact_label_table_path"] = exact_label_path.name
        row["exact_label_table_sha256"] = label_sha256
    write_csv_atomic(manifest_path, _rectangular(combined, manifest_columns))
    split_payload = json.loads(source_split_manifest_path.read_text(encoding="utf-8"))
    split_payload.update(
        {
            "schema_version": artifact_version,
            "training_augmentation_scope": plan["scope"],
            "source_split_manifest_sha256": sha256_file(source_split_manifest_path),
            "augmentation_plan_sha256": plan_sha256,
            "training_manifest_sha256": sha256_file(manifest_path),
        }
    )
    split_path = output_dir / f"split_manifest_v{artifact_version}.json"
    write_json_atomic(split_path, split_payload)
    summary = {
        "schema_version": artifact_version,
        "scope": plan["scope"],
        "source_samples": len(source_rows),
        "augmentation_samples": len(augmented_rows),
        "local_direction_samples": sum(
            item["kind"] in {"local_direction", "on_policy_direction"} for item in planned
        ),
        "ood_samples": sum(item["kind"].startswith("ood_") for item in planned),
        "evaluation_ood_samples": sum(
            item["split"] in {"calibration", "qualification"} for item in planned
        ),
        "ood_samples_by_split": {
            split: sum(
                item["split"] == split and item["kind"].startswith("ood_") for item in planned
            )
            for split in ("train", "calibration", "qualification")
        },
        "prompts": len({item["prompt_id"] for item in planned}),
        "local_exact_improved": sum(
            row["exact_band_loss_improvement"] > 0 for row in local_evidence
        ),
        "local_exact_tied": sum(row["exact_band_loss_improvement"] == 0 for row in local_evidence),
        "local_exact_worsened": sum(
            row["exact_band_loss_improvement"] < 0 for row in local_evidence
        ),
        "local_exact_out_of_band_anchors": sum(
            row["exact_band_loss_before"] > 0 for row in local_evidence
        ),
        "plan_sha256": plan_sha256,
        "trajectory_manifest_sha256": sha256_file(trajectory_manifest_path),
        "descriptor_table_sha256": sha256_file(descriptor_path),
        "exact_storage": storage_summary,
        "exact_label_table_sha256": label_sha256,
        "training_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "configuration_sha256": canonical_json_sha256(
            {
                "trajectories_per_prompt": trajectories_per_prompt,
                "perturbations_per_anchor": perturbations_per_anchor,
                "rms_ratio": rms_ratio,
                "local_mode": local_mode,
                "on_policy_rms_ratios": list(on_policy_rms_ratios),
                "ood_per_prompt": ood_per_prompt,
                "evaluation_ood_per_prompt": evaluation_ood_per_prompt,
                "seed": seed,
                "exact_batch_size": exact_batch_size,
            }
        ),
    }
    if local_evidence:
        write_csv_atomic(output_dir / "local_direction_exact_evidence.csv", local_evidence)
        summary["local_direction_exact_evidence_sha256"] = sha256_file(
            output_dir / "local_direction_exact_evidence.csv"
        )
    write_json_atomic(output_dir / "training_augmentation_summary.json", summary)
    return summary
