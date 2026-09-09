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
                    # Do not use ``left_value @ right_value`` here.  On some
                    # CUDA/cuBLAS combinations Sdot rejects long strided
                    # gradient vectors with CUBLAS_STATUS_NOT_SUPPORTED.  Pure
                    # elementwise reductions are layout-independent and this
                    # value is diagnostic only.
                    left_value = flattened[left, 0].float()
                    right_value = flattened[right, 0].float()
                    dot = (left_value * right_value).sum(dtype=torch.float32)
                    denominator = torch.sqrt(
                        left_value.square().sum(dtype=torch.float32)
                        * right_value.square().sum(dtype=torch.float32)
                    )
                    valid = bool(
                        torch.isfinite(dot).item()
                        and torch.isfinite(denominator).item()
                        and denominator.item() > 1e-12
                    )
                    cosines.append(-1.0 if not valid else float((dot / denominator).item()))
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

    def perturb(
        self, anchor: LTSNSnapshot, rms_ratio: float, *, sign: float = 1.0
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if sign not in {-1.0, 1.0}:
            raise ValueError("on-policy perturbation sign must be -1 or +1")
        description = self.describe(anchor)
        base_update = self._cache[anchor.sample_id][0]
        latent = np.load(anchor.latent_path, allow_pickle=False).astype(np.float32, copy=False)
        result = latent + float(sign) * float(rms_ratio) * base_update
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
            "sign": float(sign),
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


def _v5_step_quotas(total: int) -> dict[int, int]:
    if total < 4 or total % 4:
        raise ValueError("V5 anchor counts must be positive multiples of four")
    return {4: total // 2, 5: total // 4, 6: total // 4}


def _select_v5_symmetric_anchors(
    records: list[LTSNSnapshot],
    provider: _OnPolicyDirectionProvider,
    *,
    trajectories_per_prompt: int,
    train_anchor_count: int,
    development_anchor_count: int,
) -> tuple[list[LTSNSnapshot], dict[str, dict[str, Any]]]:
    """Select exact-OOB anchors with frozen 50/25/25 correction-step quotas."""

    quotas = {
        "train": _v5_step_quotas(train_anchor_count),
        "development": _v5_step_quotas(development_anchor_count),
    }
    threshold = float(provider.contract.focus_band_threshold)
    grouped: dict[tuple[str, int, str], list[LTSNSnapshot]] = defaultdict(list)
    trajectories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in sorted(records, key=lambda item: (item.split, item.prompt_id, item.sample_id)):
        if (
            record.split not in quotas
            or record.is_final
            or record.step_number not in {4, 5, 6}
            or record.local_anchor_sample_id
            or record.focus_logit >= threshold
        ):
            continue
        trajectory_key = (record.split, record.prompt_id)
        allowed = trajectories[trajectory_key]
        if record.trajectory_id not in allowed:
            if len(allowed) >= trajectories_per_prompt:
                continue
            allowed.add(record.trajectory_id)
        grouped[(record.split, record.step_number, record.prompt_id)].append(record)

    selected: list[LTSNSnapshot] = []
    descriptions: dict[str, dict[str, Any]] = {}
    for split in ("train", "development"):
        prompt_ids = sorted(
            {prompt_id for candidate_split, _, prompt_id in grouped if candidate_split == split}
        )
        if not prompt_ids:
            raise LTSNContractError(f"V5 has no exact-OOB candidate prompts in {split}")
        for step, quota in quotas[split].items():
            candidates = {
                prompt_id: sorted(
                    grouped.get((split, step, prompt_id), ()),
                    key=lambda item: (
                        abs(float(item.focus_logit) - threshold),
                        item.trajectory_id,
                        item.sample_id,
                    ),
                )
                for prompt_id in prompt_ids
            }
            offsets = {prompt_id: 0 for prompt_id in prompt_ids}
            accepted: list[LTSNSnapshot] = []
            while len(accepted) < quota:
                progressed = False
                for prompt_id in prompt_ids:
                    values = candidates[prompt_id]
                    while offsets[prompt_id] < len(values):
                        candidate = values[offsets[prompt_id]]
                        offsets[prompt_id] += 1
                        progressed = True
                        description = provider.describe(candidate)
                        descriptions[candidate.sample_id] = description
                        if description["gradient_usable"]:
                            accepted.append(candidate)
                            break
                    if len(accepted) == quota:
                        break
                if not progressed:
                    raise LTSNContractError(
                        f"V5 cannot satisfy {split} step-{step} anchor quota {quota}; "
                        f"accepted {len(accepted)}"
                    )
            selected.extend(accepted)
    if len({anchor.sample_id for anchor in selected}) != len(selected):
        raise LTSNContractError("V5 selected a duplicate symmetric anchor")
    provider.retain({anchor.sample_id for anchor in selected})
    return selected, descriptions


def _v5_symmetric_plan(
    anchors: list[LTSNSnapshot],
    descriptions: dict[str, dict[str, Any]],
    rms_ratios: tuple[float, ...],
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for anchor in anchors:
        for rms_ratio in rms_ratios:
            tag = int(round(rms_ratio * 10_000))
            group_id = f"{anchor.sample_id}__v5_fd_r{tag:04d}"
            for sign, sign_tag in ((-1.0, "minus"), (1.0, "plus")):
                planned.append(
                    {
                        "sample_id": f"{group_id}__{sign_tag}",
                        "anchor_sample_id": anchor.sample_id,
                        "prompt_id": anchor.prompt_id,
                        "step_number": anchor.step_number,
                        "timestep": anchor.timestep,
                        "kind": "on_policy_symmetric",
                        "split": anchor.split,
                        "seed": 0,
                        "rms_ratio": rms_ratio,
                        "sign": sign,
                        "direction_group_id": group_id,
                        "is_final": False,
                        "anchor_diagnostics": descriptions[anchor.sample_id],
                    }
                )
    return planned


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _central_direction_evidence(
    local_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair V5 minus/plus exact labels into auditable central derivatives."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in local_evidence:
        group_id = str(row.get("direction_group_id", ""))
        if group_id:
            grouped[group_id].append(row)
    output: list[dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items()):
        by_sign = {float(row["sign"]): row for row in rows}
        if set(by_sign) != {-1.0, 1.0} or len(rows) != 2:
            raise LTSNContractError(f"V5 direction group is not a minus/plus pair: {group_id}")
        minus, plus = by_sign[-1.0], by_sign[1.0]
        rms_ratio = float(minus["rms_ratio"])
        if rms_ratio <= 0 or float(plus["rms_ratio"]) != rms_ratio:
            raise LTSNContractError(f"V5 direction group has inconsistent RMS: {group_id}")
        exact_derivative = (
            float(minus["exact_band_loss_after"]) - float(plus["exact_band_loss_after"])
        ) / (2.0 * rms_ratio)
        minus_proxy_improvement = minus.get("proxy_band_loss_improvement")
        plus_proxy_improvement = plus.get("proxy_band_loss_improvement")
        proxy_derivative = None
        if minus_proxy_improvement is not None and plus_proxy_improvement is not None:
            # before_loss cancels: L(-d)-L(+d) = I(+d)-I(-d).
            proxy_derivative = (float(plus_proxy_improvement) - float(minus_proxy_improvement)) / (
                2.0 * rms_ratio
            )
        output.append(
            {
                "direction_group_id": group_id,
                "anchor_sample_id": minus["anchor_sample_id"],
                "split": minus["split"],
                "step_number": minus["step_number"],
                "rms_ratio": rms_ratio,
                "minus_sample_id": minus["sample_id"],
                "plus_sample_id": plus["sample_id"],
                "exact_loss_minus": minus["exact_band_loss_after"],
                "exact_loss_plus": plus["exact_band_loss_after"],
                "exact_derivative": exact_derivative,
                "proxy_derivative": proxy_derivative,
                "direction_agrees": (
                    None
                    if proxy_derivative is None or abs(exact_derivative) < 1e-12
                    else bool(np.sign(proxy_derivative) == np.sign(exact_derivative))
                ),
            }
        )
    return output


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
        # Preserve the V4 contract: the provider's descent direction is +d;
        # the historical plan sign field was metadata and was not applied.
        augmented, on_policy_diagnostics = on_policy_provider.perturb(
            anchor, float(item["rms_ratio"])
        )
    elif item["kind"] == "on_policy_symmetric":
        if on_policy_provider is None:
            raise LTSNContractError("symmetric on-policy augmentation requires a frozen ensemble")
        augmented, on_policy_diagnostics = on_policy_provider.perturb(
            anchor, float(item["rms_ratio"]), sign=float(item["sign"])
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
                    if item["kind"]
                    in {"local_direction", "on_policy_direction", "on_policy_symmetric"}
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
    v5_train_anchor_count: int = 512,
    v5_development_anchor_count: int = 128,
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
    if local_mode not in {"random", "on_policy", "symmetric_on_policy"}:
        raise ValueError("local_mode must be random, on_policy, or symmetric_on_policy")
    if not on_policy_rms_ratios or any(
        value not in {0.0025, 0.005, 0.01} for value in on_policy_rms_ratios
    ):
        raise ValueError("on-policy RMS ratios must use 0.0025, 0.005, or 0.01")
    if local_mode == "symmetric_on_policy":
        if tuple(sorted(set(on_policy_rms_ratios))) != (0.0025, 0.005):
            raise ValueError("V5 symmetric on-policy RMS ratios must be 0.0025 and 0.005")
        _v5_step_quotas(v5_train_anchor_count)
        _v5_step_quotas(v5_development_anchor_count)
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
    if local_mode in {"on_policy", "symmetric_on_policy"}:
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
        if local_mode == "symmetric_on_policy":
            on_policy_anchors, descriptions = _select_v5_symmetric_anchors(
                source_records,
                on_policy_provider,
                trajectories_per_prompt=trajectories_per_prompt,
                train_anchor_count=v5_train_anchor_count,
                development_anchor_count=v5_development_anchor_count,
            )
            planned = _v5_symmetric_plan(
                on_policy_anchors,
                descriptions,
                tuple(sorted(set(on_policy_rms_ratios))),
            )
        else:
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
    planned_sample_ids = [str(item["sample_id"]) for item in planned]
    if len(set(planned_sample_ids)) != len(planned_sample_ids):
        raise LTSNContractError("training augmentation plan contains duplicate sample IDs")
    artifact_version = {"random": 3, "on_policy": 4, "symmetric_on_policy": 5}[local_mode]
    plan_path = output_dir / "training_augmentation_plan.json"
    plan = {
        "schema_version": artifact_version,
        "scope": {
            "random": "train_local_ood_and_heldout_evaluation_ood_v3",
            "on_policy": "on_policy_exact_local_and_step_matched_ood_v4",
            "symmetric_on_policy": "symmetric_exact_finite_difference_v5",
        }[local_mode],
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
        **(
            {
                "v5_train_anchor_count": v5_train_anchor_count,
                "v5_development_anchor_count": v5_development_anchor_count,
            }
            if local_mode == "symmetric_on_policy"
            else {}
        ),
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
        if item["kind"] in {
            "local_direction",
            "on_policy_direction",
            "on_policy_symmetric",
        }:
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
                    **(
                        {
                            "direction_group_id": item["direction_group_id"],
                            "sign": item["sign"],
                        }
                        if item["kind"] == "on_policy_symmetric"
                        else {}
                    ),
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
                    if item["kind"]
                    in {"local_direction", "on_policy_direction", "on_policy_symmetric"}
                    else ""
                ),
                "local_direction_group_id": item.get("direction_group_id", ""),
                "local_direction_sign": item.get("sign", 0.0),
                "local_direction_rms_ratio": item.get("rms_ratio", 0.0),
                "training_augmentation_kind": item["kind"],
            }
        )
        augmented_rows.append(augmented)

    central_evidence = (
        _central_direction_evidence(local_evidence) if local_mode == "symmetric_on_policy" else []
    )

    exact_label_path = output_dir / f"exact_snapshot_labels_v{artifact_version}.csv"
    write_csv_atomic(
        exact_label_path,
        _rectangular([*old_labels, *new_labels], label_columns),
    )
    label_sha256 = sha256_file(exact_label_path)
    manifest_path = output_dir / f"ltsn_manifest_v{artifact_version}.csv"
    manifest_columns = list(source_rows[0])
    augmentation_columns = ["local_anchor_sample_id", "training_augmentation_kind"]
    if local_mode == "symmetric_on_policy":
        augmentation_columns[1:1] = [
            "local_direction_group_id",
            "local_direction_sign",
            "local_direction_rms_ratio",
        ]
    for column in augmentation_columns:
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
            item["kind"] in {"local_direction", "on_policy_direction", "on_policy_symmetric"}
            for item in planned
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
        **(
            {
                "symmetric_direction_groups": len(central_evidence),
                "symmetric_anchors_by_split": {
                    split: len(
                        {
                            row["anchor_sample_id"]
                            for row in central_evidence
                            if row["split"] == split
                        }
                    )
                    for split in ("train", "development")
                },
                "symmetric_groups_by_split": {
                    split: sum(row["split"] == split for row in central_evidence)
                    for split in ("train", "development")
                },
                "symmetric_groups_by_step": {
                    str(step): sum(int(row["step_number"]) == step for row in central_evidence)
                    for step in (4, 5, 6)
                },
                "symmetric_anchors_by_step_and_split": {
                    split: {
                        str(step): len(
                            {
                                row["anchor_sample_id"]
                                for row in central_evidence
                                if row["split"] == split and int(row["step_number"]) == step
                            }
                        )
                        for step in (4, 5, 6)
                    }
                    for split in ("train", "development")
                },
                "exact_derivative_positive": sum(
                    float(row["exact_derivative"]) > 1e-12 for row in central_evidence
                ),
                "exact_derivative_tied": sum(
                    abs(float(row["exact_derivative"])) <= 1e-12 for row in central_evidence
                ),
                "exact_derivative_negative": sum(
                    float(row["exact_derivative"]) < -1e-12 for row in central_evidence
                ),
            }
            if local_mode == "symmetric_on_policy"
            else {}
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
                **(
                    {
                        "v5_train_anchor_count": v5_train_anchor_count,
                        "v5_development_anchor_count": v5_development_anchor_count,
                    }
                    if local_mode == "symmetric_on_policy"
                    else {}
                ),
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
    if central_evidence:
        central_path = output_dir / "central_direction_exact_evidence.csv"
        write_csv_atomic(central_path, central_evidence)
        informative = [
            row
            for row in central_evidence
            if abs(float(row["exact_derivative"])) > 1e-12 and row["proxy_derivative"] is not None
        ]
        summary["central_direction_exact_evidence_sha256"] = sha256_file(central_path)
        summary["proxy_exact_derivative_direction_agreement"] = (
            0.0
            if not informative
            else float(np.mean([bool(row["direction_agrees"]) for row in informative]))
        )
        if len(informative) >= 2:
            exact_values = np.asarray(
                [float(row["exact_derivative"]) for row in informative], dtype=np.float64
            )
            proxy_values = np.asarray(
                [float(row["proxy_derivative"]) for row in informative], dtype=np.float64
            )
            exact_ranks = _rank_average(exact_values)
            proxy_ranks = _rank_average(proxy_values)
            if np.std(exact_ranks) > 0 and np.std(proxy_ranks) > 0:
                summary["proxy_exact_derivative_spearman"] = float(
                    np.corrcoef(exact_ranks, proxy_ranks)[0, 1]
                )
            else:
                summary["proxy_exact_derivative_spearman"] = 0.0
        else:
            summary["proxy_exact_derivative_spearman"] = 0.0
    write_json_atomic(output_dir / "training_augmentation_summary.json", summary)
    return summary
