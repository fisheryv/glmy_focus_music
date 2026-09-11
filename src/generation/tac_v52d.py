"""TAC V5.2d repeatability and on-manifold rollout audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import LTSNContractError, load_fingerprint_contract, sha256_file
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .ltsn_storage import remove_generated_audio
from .path_homology_exact_scorer import ExactPathHomologyScorer
from .tac_target import TACTopologyTarget
from .tac_v52c import wilson_interval

V52D_EXPERIMENT = "tac_v5_2d_on_manifold_rollout"
V52D_SCALES = (0.25, 0.5, 1.0)
V52D_REFERENCE_SCALE = 0.5
V52D_ON_MANIFOLD_BASES = ("scheduler_update", "history_pca_1", "history_pca_2")
V52D_RANDOM_BASES = ("random_control_1", "random_control_2", "random_control_3")
V52D_ALL_BASES = (*V52D_ON_MANIFOLD_BASES, *V52D_RANDOM_BASES)
V52D_CFG_STATUS = "unavailable_xl_turbo_cfg_distilled_no_unconditional_branch"
V52D_EXACT_MARGIN = 1e-8


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _unit_rms(candidate: np.ndarray, previous: Sequence[np.ndarray]) -> np.ndarray | None:
    value = np.asarray(candidate, dtype=np.float64).copy()
    for _ in range(2):
        for basis in previous:
            denominator = float(np.sum(basis * basis, dtype=np.float64))
            value -= basis * (float(np.sum(value * basis, dtype=np.float64)) / denominator)
    rms = float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 1e-8:
        return None
    value /= rms
    pivot = int(np.argmax(np.abs(value)))
    if value.reshape(-1)[pivot] < 0.0:
        value *= -1.0
    return value


def _smooth_random(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal(shape)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    padded = np.pad(raw, ((2, 2), (0, 0)), mode="edge")
    return sum(weight * padded[offset : offset + shape[0]] for offset, weight in enumerate(kernel))


def build_action_bases(
    update_history: np.ndarray, *, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct deterministic unit-RMS rollout and matched random bases."""

    history = np.asarray(update_history, dtype=np.float64)
    if history.ndim != 3 or history.shape[0] < 3 or history.shape[2] != 64:
        raise LTSNContractError("V5.2d update history must have shape [K,T,64], K >= 3")
    if not np.isfinite(history).all():
        raise LTSNContractError("V5.2d update history contains NaN or Inf")
    current = history[-1]
    centered = (history - np.mean(history, axis=0, keepdims=True)).reshape(history.shape[0], -1)
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    candidates = [current, *(vector.reshape(current.shape) for vector in vectors)]
    on_manifold: list[np.ndarray] = []
    for candidate in candidates:
        normalized = _unit_rms(candidate, on_manifold)
        if normalized is not None:
            on_manifold.append(normalized)
        if len(on_manifold) == len(V52D_ON_MANIFOLD_BASES):
            break
    if len(on_manifold) != len(V52D_ON_MANIFOLD_BASES):
        raise LTSNContractError("V5.2d rollout history has fewer than three stable directions")

    rng = np.random.default_rng(seed)
    controls: list[np.ndarray] = []
    attempts = 0
    while len(controls) < len(V52D_RANDOM_BASES) and attempts < 128:
        attempts += 1
        normalized = _unit_rms(_smooth_random(current.shape, rng), [*on_manifold, *controls])
        if normalized is not None:
            controls.append(normalized)
    if len(controls) != len(V52D_RANDOM_BASES):
        raise LTSNContractError("V5.2d could not construct matched random controls")
    values = [*on_manifold, *controls]
    matrix = np.stack([value.reshape(-1) for value in values])
    cosine = matrix @ matrix.T
    norms = np.sqrt(np.sum(matrix * matrix, axis=1))
    cosine /= norms[:, None] * norms[None, :]
    maximum_cosine = float(np.max(np.abs(cosine - np.eye(len(values)))))
    if maximum_cosine > 1e-5:
        raise LTSNContractError(f"V5.2d basis is not orthogonal: {maximum_cosine}")
    bases = dict(zip(V52D_ALL_BASES, values, strict=True))
    return bases, {
        "basis_names": list(V52D_ALL_BASES),
        "history_steps": int(history.shape[0]),
        "history_singular_values": singular_values.tolist(),
        "maximum_absolute_pairwise_cosine": maximum_cosine,
        "random_attempts": attempts,
        "basis_sha256": {
            name: hashlib.sha256(value.astype(np.float32).tobytes(order="C")).hexdigest()
            for name, value in bases.items()
        },
        "cfg_residual": V52D_CFG_STATUS,
    }


class OnManifoldActionHook:
    """Inject one frozen action after a selected scheduler update."""

    def __init__(
        self,
        *,
        anchor_id: str,
        target_step: int,
        basis_name: str | None,
        scale: float = 0.0,
        sign: float = 0.0,
        seed: int = 20260912,
    ) -> None:
        if target_step not in {4, 5, 6}:
            raise ValueError("V5.2d target step must be 4, 5, or 6")
        if basis_name is not None and basis_name not in V52D_ALL_BASES:
            raise ValueError(f"unsupported V5.2d basis: {basis_name}")
        if basis_name is not None and (scale not in V52D_SCALES or sign not in {-1.0, 1.0}):
            raise ValueError("V5.2d action scale or sign changed")
        self.anchor_id = anchor_id
        self.target_step = target_step
        self.basis_name = basis_name
        self.scale = float(scale)
        self.sign = float(sign)
        self.seed = int(seed)
        self.history: list[np.ndarray] = []
        self.audit: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        xt_next: Any,
        xt_before_step: Any,
        velocity: Any,
        timestep: float,
        next_timestep: float,
        step_index: int,
        attention_mask: Any,
        repaint_mask: Any | None = None,
    ) -> Any:
        del velocity, timestep, next_timestep, repaint_mask
        step_number = int(step_index) + 1
        if xt_next.shape[0] != 1:
            raise LTSNContractError("V5.2d requires generation batch_size=1")
        valid = attention_mask[0].detach().to(device="cpu", dtype=bool).numpy()
        update = (xt_next[0] - xt_before_step[0]).detach().float().cpu().numpy()[valid]
        if update.ndim != 2 or update.shape[1] != 64 or not np.isfinite(update).all():
            raise LTSNContractError("V5.2d observed an invalid scheduler update")
        self.history.append(update.astype(np.float64, copy=False))
        if step_number != self.target_step:
            return xt_next
        bases, basis_audit = build_action_bases(
            np.stack(self.history), seed=_seed(self.seed, self.anchor_id)
        )
        update_rms = float(np.sqrt(np.mean(np.square(update), dtype=np.float64)))
        if update_rms <= 0.0:
            raise LTSNContractError("V5.2d target scheduler update has zero RMS")
        self.audit = {
            **basis_audit,
            "anchor_id": self.anchor_id,
            "target_step": self.target_step,
            "basis_name": self.basis_name or "no_op",
            "scale": self.scale,
            "sign": self.sign,
            "scheduler_update_rms": update_rms,
            "injected": self.basis_name is not None,
            "applied": False,
        }
        if self.basis_name is None:
            return xt_next
        delta = self.sign * self.scale * update_rms * bases[self.basis_name]
        corrected = xt_next.clone()
        import torch

        mask = attention_mask[0].to(device=corrected.device, dtype=torch.bool)
        corrected[0, mask] += torch.from_numpy(delta.astype(np.float32)).to(
            device=corrected.device, dtype=corrected.dtype
        )
        self.audit["action_rms"] = float(np.sqrt(np.mean(np.square(delta))))
        self.audit["applied"] = True
        return corrected


def _mcnemar(true_values: Sequence[bool], control_values: Sequence[bool]) -> dict[str, Any]:
    if len(true_values) != len(control_values) or not true_values:
        raise ValueError("McNemar inputs must be paired and non-empty")
    true_only = sum(t and not c for t, c in zip(true_values, control_values, strict=True))
    control_only = sum(c and not t for t, c in zip(true_values, control_values, strict=True))
    discordant = true_only + control_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value) for value in range(min(true_only, control_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "on_manifold_only_stable": true_only,
        "random_only_stable": control_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p": p_value,
    }


def analyze_repeatability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["anchor_sample_id"]), []).append(row)
    if not grouped:
        raise LTSNContractError("V5.2d repeatability table is empty")
    anchor_rows = []
    for anchor_id, members in sorted(grouped.items()):
        by_repeat: dict[int, list[Mapping[str, Any]]] = {}
        for row in members:
            by_repeat.setdefault(int(row["rollout_repeat"]), []).append(row)
        if set(by_repeat) != {0, 1, 2} or any(len(value) != 2 for value in by_repeat.values()):
            raise LTSNContractError(f"incomplete V5.2d repeatability rows: {anchor_id}")
        extraction_equal = True
        representatives = []
        for _repeat, values in sorted(by_repeat.items()):
            if {int(row["extraction_repeat"]) for row in values} != {0, 1}:
                raise LTSNContractError(f"invalid exact extraction replicas: {anchor_id}")
            coordinates = [
                np.asarray(json.loads(row["coordinates_json"]), dtype=float) for row in values
            ]
            extraction_equal &= (
                np.array_equal(coordinates[0], coordinates[1])
                and len({str(row["descriptor_signature_sha256"]) for row in values}) == 1
            )
            representatives.append(values[0])
        latent_equal = len({str(row["final_latent_sha256"]) for row in representatives}) == 1
        audio_equal = len({str(row["audio_sha256"]) for row in representatives}) == 1
        coordinate_matrix = np.stack(
            [
                np.asarray(json.loads(row["coordinates_json"]), dtype=float)
                for row in representatives
            ]
        )
        distance = np.asarray([float(row["target_distance"]) for row in representatives])
        anchor_rows.append(
            {
                "anchor_sample_id": anchor_id,
                "latent_bitwise_equal": latent_equal,
                "audio_bitwise_equal": audio_equal,
                "exact_extraction_bitwise_equal": extraction_equal,
                "maximum_coordinate_range": float(np.max(np.ptp(coordinate_matrix, axis=0))),
                "target_distance_range": float(np.ptp(distance)),
                "mean_target_distance": float(np.mean(distance)),
                "all_in_distribution": all(
                    not bool(float(row.get("ood_label", 0.0))) for row in representatives
                ),
            }
        )
    noise = np.asarray([row["target_distance_range"] for row in anchor_rows], dtype=float)
    noise_q95 = float(np.quantile(noise, 0.95))
    median_distance = float(np.median([row["mean_target_distance"] for row in anchor_rows]))
    relative_noise = noise_q95 / max(median_distance, np.finfo(float).eps)
    passed = bool(
        all(
            row["exact_extraction_bitwise_equal"] and row["all_in_distribution"]
            for row in anchor_rows
        )
        and relative_noise <= 0.01
    )
    return {
        "schema_version": 1,
        "experiment": f"{V52D_EXPERIMENT}_repeatability",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "anchors": len(anchor_rows),
        "rollout_repeats": 3,
        "exact_extraction_repeats": 2,
        "target_distance_noise_q95": noise_q95,
        "median_baseline_target_distance": median_distance,
        "relative_target_distance_noise_q95": relative_noise,
        "maximum_target_distance_range": float(np.max(noise)),
        "anchor_results": anchor_rows,
        "diagnostics": {
            "all_final_latents_bitwise_equal": all(
                row["latent_bitwise_equal"] for row in anchor_rows
            ),
            "all_decoded_audio_bitwise_equal": all(
                row["audio_bitwise_equal"] for row in anchor_rows
            ),
        },
        "gates": {
            "all_exact_extractions_bitwise_equal": all(
                row["exact_extraction_bitwise_equal"] for row in anchor_rows
            ),
            "relative_target_distance_noise_q95_at_most_0_01": bool(relative_noise <= 0.01),
            "all_repeats_in_distribution": all(row["all_in_distribution"] for row in anchor_rows),
        },
        "repeatability_status": "passed" if passed else "failed",
        "action_collection_authorized": passed,
    }


def analyze_actions(
    rows: Sequence[Mapping[str, Any]], *, repeatability_report: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        repeatability_report.get("repeatability_status") != "passed"
        or repeatability_report.get("action_collection_authorized") is not True
    ):
        raise LTSNContractError("V5.2d action analysis requires passed repeatability")
    grouped: dict[tuple[str, str, float], dict[float, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["anchor_sample_id"]),
            str(row["basis_name"]),
            float(row["action_scale"]),
        )
        grouped.setdefault(key, {})[float(row["sign"])] = row
    outcomes = []
    for (anchor_id, basis_name, scale), signed in sorted(grouped.items()):
        if set(signed) != {-1.0, 1.0}:
            raise LTSNContractError(
                f"incomplete V5.2d action pair: {(anchor_id, basis_name, scale)}"
            )
        minus, plus = signed[-1.0], signed[1.0]
        difference = float(minus["target_distance"]) - float(plus["target_distance"])
        outcomes.append(
            {
                "anchor_sample_id": anchor_id,
                "basis_name": basis_name,
                "basis_role": "on_manifold"
                if basis_name in V52D_ON_MANIFOLD_BASES
                else "matched_random_control",
                "action_scale": scale,
                "target_response": difference / (2.0 * scale),
                "response_sign": int(np.sign(difference))
                if abs(difference) >= V52D_EXACT_MARGIN
                else 0,
                "absolute_half_pair_effect": abs(difference) / 2.0,
                "pair_in_distribution": not bool(float(minus["ood_label"]))
                and not bool(float(plus["ood_label"])),
            }
        )
    outcome_by_key = {
        (row["anchor_sample_id"], row["basis_name"], row["action_scale"]): row for row in outcomes
    }
    stability: dict[tuple[str, str], list[bool]] = {}
    for anchor_id in sorted({str(row["anchor_sample_id"]) for row in outcomes}):
        for basis in V52D_ALL_BASES:
            reference = outcome_by_key[(anchor_id, basis, V52D_REFERENCE_SCALE)]["response_sign"]
            stability[(anchor_id, basis)] = []
            for scale in V52D_SCALES:
                if scale == V52D_REFERENCE_SCALE:
                    continue
                tested = outcome_by_key[(anchor_id, basis, scale)]["response_sign"]
                stability[(anchor_id, basis)].append(reference != 0 and tested == reference)
    on_values = [
        value
        for (anchor, basis), values in stability.items()
        if basis in V52D_ON_MANIFOLD_BASES
        for value in values
    ]
    random_values = [
        value
        for (anchor, basis), values in stability.items()
        if basis in V52D_RANDOM_BASES
        for value in values
    ]
    successes = sum(on_values)
    interval = wilson_interval(successes, len(on_values))
    agreement = successes / len(on_values)
    paired_on = []
    paired_random = []
    for anchor_id in sorted({key[0] for key in stability}):
        for on_basis, random_basis in zip(V52D_ON_MANIFOLD_BASES, V52D_RANDOM_BASES, strict=True):
            paired_on.extend(stability[(anchor_id, on_basis)])
            paired_random.extend(stability[(anchor_id, random_basis)])
    comparison = _mcnemar(paired_on, paired_random)
    effects = [
        float(row["absolute_half_pair_effect"])
        for row in outcomes
        if row["basis_role"] == "on_manifold"
    ]
    noise = float(repeatability_report["target_distance_noise_q95"])
    median_effect = float(np.median(effects))
    all_id = all(bool(row["pair_in_distribution"]) for row in outcomes)
    gates = {
        "on_manifold_cross_scale_agreement_at_least_0_80": agreement >= 0.80,
        "on_manifold_wilson_lower_above_0_50": interval[0] > 0.50,
        "on_manifold_better_than_random_mcnemar_p_below_0_05": (
            comparison["on_manifold_only_stable"] > comparison["random_only_stable"]
            and comparison["two_sided_exact_p"] < 0.05
        ),
        "median_effect_above_three_times_repeatability_noise": median_effect > 3.0 * noise,
        "all_pairs_in_distribution": all_id,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "experiment": V52D_EXPERIMENT,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "cfg_residual": V52D_CFG_STATUS,
        "action_bases": {
            "on_manifold": list(V52D_ON_MANIFOLD_BASES),
            "matched_random_control": list(V52D_RANDOM_BASES),
        },
        "scales": list(V52D_SCALES),
        "reference_scale": V52D_REFERENCE_SCALE,
        "exact_margin": V52D_EXACT_MARGIN,
        "pairs": len(outcomes),
        "on_manifold_cross_scale": {
            "successes": successes,
            "comparisons": len(on_values),
            "agreement": agreement,
            "wilson95": interval,
        },
        "random_cross_scale_agreement": sum(random_values) / len(random_values),
        "matched_control_mcnemar": comparison,
        "repeatability_noise_q95": noise,
        "median_on_manifold_absolute_half_pair_effect": median_effect,
        "effect_to_noise_ratio": median_effect / max(noise, np.finfo(float).eps),
        "gates": gates,
        "stage1d_status": "supported_for_critic_collection" if passed else "not_supported",
        "next_stage_authorized": passed,
    }
    return report, outcomes


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary, path)


def _final_latent(values: Any) -> np.ndarray:
    value = values
    for method in ("detach", "cpu", "float"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[1] != 64 or not np.isfinite(matrix).all():
        raise LTSNContractError("V5.2d final latent must have finite shape [T,64]")
    return matrix


def _load_anchor_specs(
    *,
    source_manifest_path: Path,
    v52c_plan_path: Path,
    prompt_manifest_path: Path,
    anchor_count: int,
) -> list[dict[str, Any]]:
    if anchor_count != 8:
        raise ValueError("V5.2d pilot is frozen to eight anchors")
    source = {row["sample_id"]: row for row in _read_csv(source_manifest_path)}
    plan = json.loads(v52c_plan_path.read_text(encoding="utf-8"))
    candidates = list(plan.get("anchor_ids", []))
    if len(candidates) != 16 or len(set(candidates)) != 16:
        raise LTSNContractError("V5.2d requires the completed 16-anchor V5.2c plan")
    prompts = {row["prompt_id"]: row for row in _read_csv(prompt_manifest_path)}
    quotas = {4: 3, 5: 3, 6: 2}
    selected = []
    for step, quota in quotas.items():
        available = [anchor for anchor in candidates if int(source[anchor]["step_number"]) == step]
        ordered = sorted(available, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        if len(ordered) < quota:
            raise LTSNContractError(f"V5.2d cannot fill step-{step} anchor quota")
        selected.extend(ordered[:quota])
    specs = []
    for anchor_id in selected:
        row = source[anchor_id]
        prompt = prompts.get(row["prompt_id"])
        if prompt is None:
            raise LTSNContractError(f"V5.2d prompt is missing: {row['prompt_id']}")
        trajectory_id = row["trajectory_id"]
        if "__seed" not in trajectory_id:
            raise LTSNContractError(f"V5.2d trajectory has no frozen seed: {trajectory_id}")
        seed = int(trajectory_id.rsplit("__seed", 1)[1])
        specs.append(
            {
                "anchor_sample_id": anchor_id,
                "prompt_id": row["prompt_id"],
                "caption": prompt["caption"],
                "seed": seed,
                "bpm": int(prompt["bpm"]) if prompt.get("bpm", "").strip() else None,
                "keyscale": prompt.get("keyscale", ""),
                "timesignature": prompt.get("timesignature", ""),
                "step_number": int(row["step_number"]),
                "model_family": row["model_family"],
                "split": "development",
            }
        )
    return sorted(specs, key=lambda row: row["anchor_sample_id"])


def _completed_batch(
    batch_dir: Path, *, plan_sha256: str, logical_ids: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]] | None:
    descriptor_path = batch_dir / "descriptors.csv"
    records_path = batch_dir / "generated_records.json"
    completion_path = batch_dir / "completion.json"
    if not completion_path.exists():
        return None
    if not all(path.is_file() for path in (descriptor_path, records_path, completion_path)):
        raise LTSNContractError(f"incomplete V5.2d batch checkpoint: {batch_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
    if (
        completion.get("plan_sha256") != plan_sha256
        or completion.get("logical_ids") != logical_ids
        or completion.get("descriptor_sha256") != sha256_file(descriptor_path)
        or completion.get("records_sha256") != sha256_file(records_path)
        or completion.get("wav_retained") != 0
    ):
        raise LTSNContractError(f"mismatched V5.2d batch checkpoint: {batch_dir}")
    if list(batch_dir.rglob("*.wav")):
        raise LTSNContractError(f"completed V5.2d batch retained WAV: {batch_dir}")
    return _read_csv(descriptor_path), records


def _exact_batch(
    *,
    root: Path,
    output_dir: Path,
    batch_index: int,
    plan_sha256: str,
    records: list[dict[str, Any]],
    exact_repeats: int,
    workers: int,
    materialize_mode: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    batch_dir = output_dir / "batches" / f"batch_{batch_index:05d}"
    logical_ids = [str(row["sample_id"]) for row in records]
    completed = _completed_batch(batch_dir, plan_sha256=plan_sha256, logical_ids=logical_ids)
    if completed is not None:
        return completed
    batch_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows = []
    for record in records:
        for extraction_repeat in range(exact_repeats):
            exact_id = f"{record['sample_id']}__extract{extraction_repeat:02d}"
            trajectory_rows.append(
                {
                    "sample_id": exact_id,
                    "prompt_id": record["prompt_id"],
                    "trajectory_id": exact_id,
                    "split": "development",
                    "model_family": record["model_family"],
                    "step_number": record["step_number"],
                    "timestep": "0.0",
                    "latent_path": os.path.relpath(record["latent_path"], batch_dir).replace(
                        "\\", "/"
                    ),
                    "latent_sha256": record["final_latent_sha256"],
                    "audio_path": os.path.relpath(record["audio_path"], batch_dir).replace(
                        "\\", "/"
                    ),
                    "audio_sha256": record["audio_sha256"],
                    "is_final": "true",
                    "ace_model_sha256": record["ace_model_sha256"],
                    "vae_sha256": record["vae_sha256"],
                    "training_augmentation_kind": V52D_EXPERIMENT,
                    "local_anchor_sample_id": "",
                }
            )
    trajectory_path = batch_dir / "trajectories.csv"
    write_csv_atomic(trajectory_path, trajectory_rows)
    descriptor_path = batch_dir / "descriptors.csv"
    try:
        build_exact_snapshot_descriptors(
            project_root=root,
            trajectory_manifest=trajectory_path,
            work_dir=batch_dir / "exact_work",
            output_path=descriptor_path,
            workers=workers,
            batch_size=len(trajectory_rows),
            materialize_mode=materialize_mode,
            cleanup_batches=True,
            resume=True,
        )
    finally:
        for record in records:
            Path(record["audio_path"]).unlink(missing_ok=True)
    descriptors = _read_csv(descriptor_path)
    expected = sorted(row["sample_id"] for row in trajectory_rows)
    if sorted(row["sample_id"] for row in descriptors) != expected:
        raise LTSNContractError("V5.2d exact descriptor identities changed")
    records_path = batch_dir / "generated_records.json"
    write_json_atomic(records_path, {"schema_version": 1, "records": records})
    write_json_atomic(
        batch_dir / "completion.json",
        {
            "schema_version": 1,
            "plan_sha256": plan_sha256,
            "logical_ids": logical_ids,
            "descriptor_sha256": sha256_file(descriptor_path),
            "records_sha256": sha256_file(records_path),
            "wav_retained": 0,
        },
    )
    return descriptors, records


def _score_rows(
    *,
    descriptors: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, Any]],
    scorer: ExactPathHomologyScorer,
    target: TACTopologyTarget,
) -> list[dict[str, Any]]:
    record_by_id = {str(row["sample_id"]): row for row in records}
    output = []
    for descriptor in descriptors:
        exact_id = descriptor["sample_id"]
        logical_id, extraction_text = exact_id.rsplit("__extract", 1)
        record = record_by_id[logical_id]
        score = scorer.score(
            json.loads(descriptor["pitch_descriptors_json"]),
            [float(descriptor["acoustic_loop_score"])],
            [float(descriptor["chroma_loop_score"])],
        )
        coordinates = score.coordinates[0]
        row = {
            **{
                key: value
                for key, value in record.items()
                if key not in {"latent_path", "audio_path", "action_audit"}
            },
            "exact_sample_id": exact_id,
            "extraction_repeat": int(extraction_text),
            "coordinates_json": json.dumps(coordinates.tolist(), separators=(",", ":")),
            "target_distance": float(target.distance(coordinates)[0]),
            "ood_label": descriptor["ood_label"],
            "descriptor_signature_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "pitch": descriptor["pitch_descriptors_json"],
                        "acoustic": descriptor["acoustic_loop_score"],
                        "chroma": descriptor["chroma_loop_score"],
                        "ood": descriptor["ood_label"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        output.append(row)
    return output


def _generate_record(
    *,
    adapter: Any,
    config: Any,
    spec: Mapping[str, Any],
    sample_id: str,
    hook: OnManifoldActionHook,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    from .ace_adapter import GenerationRequest

    adapter.set_topology_corrector(hook)
    generator_root = output_dir / "generator_audio" / sample_id
    generated = adapter.generate(
        GenerationRequest(
            prompt=str(spec["caption"]),
            seed=int(spec["seed"]),
            duration_seconds=180.0,
            output_dir=generator_root,
            inference_steps=int(config.ace.inference_steps),
            bpm=spec["bpm"],
            keyscale=str(spec["keyscale"]),
            timesignature=str(spec["timesignature"]),
        )
    )
    if hook.audit is None:
        raise LTSNContractError(f"V5.2d hook did not observe target step: {sample_id}")
    if hook.basis_name is not None and hook.audit.get("applied") is not True:
        raise LTSNContractError(f"V5.2d action was not applied: {sample_id}")
    latent = _final_latent(generated.final_latent)
    latent_path = output_dir / "latents" / f"{sample_id}.npy"
    if latent_path.is_file():
        existing = np.load(latent_path, allow_pickle=False)
        if not np.array_equal(existing, latent):
            raise LTSNContractError(f"V5.2d resumed latent changed: {sample_id}")
    else:
        _save_npy_atomic(latent_path, latent)
    audio_path = output_dir / "ephemeral_audio" / f"{sample_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.unlink(missing_ok=True)
    adapter.decode_latent_to_audio(latent, audio_path)
    remove_generated_audio(generated.audio_path, generator_root)
    return {
        "sample_id": sample_id,
        "anchor_sample_id": spec["anchor_sample_id"],
        "prompt_id": spec["prompt_id"],
        "seed": spec["seed"],
        "step_number": spec["step_number"],
        "model_family": spec["model_family"],
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "latent_path": str(latent_path),
        "final_latent_sha256": sha256_file(latent_path),
        "audio_path": str(audio_path),
        "audio_sha256": sha256_file(audio_path),
        "action_audit": hook.audit,
        **dict(extra),
    }


def _context(
    *, root: Path, ace_config_path: Path, fingerprint_path: Path, target_path: Path
) -> tuple[Any, ExactPathHomologyScorer, TACTopologyTarget]:
    from .experiment import load_experiment_config

    config = load_experiment_config(root, ace_config_path)
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)
    target = TACTopologyTarget.from_json(target_path, root=root)
    contract = load_fingerprint_contract(fingerprint_path)
    if target.payload["fingerprint_sha256"] != contract.artifact_sha256:
        raise LTSNContractError("V5.2d target is bound to another fingerprint")
    if config.ace.model != "acestep-v15-xl-turbo" or config.ace.guidance_scale != 1.0:
        raise LTSNContractError("V5.2d requires frozen CFG-distilled XL-Turbo")
    return config, scorer, target


def _implementation_hashes(root: Path, config: Any) -> dict[str, str]:
    paths = {
        "src/generation/tac_v52d.py": Path(__file__).resolve(),
        "src/generation/ace_adapter.py": root / "src" / "generation" / "ace_adapter.py",
        "ace_xl_turbo_sampler": root
        / config.ace.checkout
        / "acestep"
        / "models"
        / "xl_turbo"
        / "modeling_acestep_v15_xl_turbo.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def collect_repeatability(
    *,
    root: Path,
    source_manifest_path: Path,
    v52c_plan_path: Path,
    prompt_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    target_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    device_name: str = "cuda:0",
    workers: int = 8,
    batch_size: int = 4,
    materialize_mode: str = "auto",
) -> dict[str, Any]:
    """Run three no-op rollouts and two exact extractions for eight anchors."""

    root = root.resolve()
    paths = [
        source_manifest_path,
        v52c_plan_path,
        prompt_manifest_path,
        ace_config_path,
        fingerprint_path,
        target_path,
        output_dir,
    ]
    (
        source_manifest_path,
        v52c_plan_path,
        prompt_manifest_path,
        ace_config_path,
        fingerprint_path,
        target_path,
        output_dir,
    ) = [path.resolve() if path.is_absolute() else (root / path).resolve() for path in paths]
    config, scorer, target = _context(
        root=root,
        ace_config_path=ace_config_path,
        fingerprint_path=fingerprint_path,
        target_path=target_path,
    )
    specs = _load_anchor_specs(
        source_manifest_path=source_manifest_path,
        v52c_plan_path=v52c_plan_path,
        prompt_manifest_path=prompt_manifest_path,
        anchor_count=8,
    )
    source_rows = _read_csv(source_manifest_path)
    if {row["ace_model_sha256"] for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("V5.2d ACE hash differs from source manifest")
    if {row["vae_sha256"] for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("V5.2d VAE hash differs from source manifest")
    planned = [
        {
            **spec,
            "rollout_repeat": repeat,
            "sample_id": f"{spec['anchor_sample_id']}__v52d_rep{repeat:02d}",
        }
        for spec in specs
        for repeat in range(3)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "experiment": f"{V52D_EXPERIMENT}_repeatability",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "v52c_plan_sha256": sha256_file(v52c_plan_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "implementation_sha256": _implementation_hashes(root, config),
        "anchor_count": 8,
        "rollout_repeats": 3,
        "exact_extraction_repeats": 2,
        "cfg_residual": V52D_CFG_STATUS,
        "planned": planned,
    }
    plan_path = output_dir / "v52d_repeatability_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise LTSNContractError("V5.2d repeatability plan changed; use a new output directory")
    if not plan_path.exists():
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    from .ace_adapter import AceStepAdapter

    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    all_descriptors: list[dict[str, str]] = []
    all_records: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(planned), batch_size)):
        items = planned[start : start + batch_size]
        batch_dir = output_dir / "batches" / f"batch_{batch_index:05d}"
        logical_ids = [str(row["sample_id"]) for row in items]
        completed = _completed_batch(batch_dir, plan_sha256=plan_sha256, logical_ids=logical_ids)
        if completed is not None:
            descriptors, records = completed
        else:
            records = []
            for item in items:
                hook = OnManifoldActionHook(
                    anchor_id=item["anchor_sample_id"],
                    target_step=int(item["step_number"]),
                    basis_name=None,
                )
                records.append(
                    _generate_record(
                        adapter=adapter,
                        config=config,
                        spec=item,
                        sample_id=item["sample_id"],
                        hook=hook,
                        output_dir=output_dir,
                        ace_model_sha256=ace_model_sha256,
                        vae_sha256=vae_sha256,
                        extra={"rollout_repeat": item["rollout_repeat"]},
                    )
                )
            descriptors, records = _exact_batch(
                root=root,
                output_dir=output_dir,
                batch_index=batch_index,
                plan_sha256=plan_sha256,
                records=records,
                exact_repeats=2,
                workers=workers,
                materialize_mode=materialize_mode,
            )
        all_descriptors.extend(descriptors)
        all_records.extend(records)
    points = _score_rows(
        descriptors=all_descriptors, records=all_records, scorer=scorer, target=target
    )
    points_path = output_dir / "v52d_repeatability_points.csv"
    write_csv_atomic(points_path, points)
    report = analyze_repeatability(points)
    report.update(
        {
            "plan_sha256": plan_sha256,
            "ace_model_sha256": ace_model_sha256,
            "vae_sha256": vae_sha256,
            "fingerprint_sha256": sha256_file(fingerprint_path),
            "target_sha256": sha256_file(target_path),
            "implementation_sha256": plan["implementation_sha256"],
            "points_sha256": sha256_file(points_path),
            "retained_wav_files": len(list(output_dir.rglob("*.wav"))),
            "retained_latent_files": len(list((output_dir / "latents").glob("*.npy"))),
        }
    )
    if report["retained_wav_files"] != 0:
        raise LTSNContractError("V5.2d repeatability cleanup retained WAV")
    report_path = output_dir / "v52d_repeatability_report.json"
    write_json_atomic(report_path, report)
    return report


def collect_actions(
    *,
    root: Path,
    source_manifest_path: Path,
    v52c_plan_path: Path,
    prompt_manifest_path: Path,
    ace_config_path: Path,
    fingerprint_path: Path,
    target_path: Path,
    repeatability_report_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    device_name: str = "cuda:0",
    workers: int = 8,
    batch_size: int = 12,
    materialize_mode: str = "auto",
) -> dict[str, Any]:
    """Regenerate, inject one action, complete denoising, and exact-score final audio."""

    root = root.resolve()
    raw_paths = [
        source_manifest_path,
        v52c_plan_path,
        prompt_manifest_path,
        ace_config_path,
        fingerprint_path,
        target_path,
        repeatability_report_path,
        output_dir,
    ]
    resolved = [
        path.resolve() if path.is_absolute() else (root / path).resolve() for path in raw_paths
    ]
    (
        source_manifest_path,
        v52c_plan_path,
        prompt_manifest_path,
        ace_config_path,
        fingerprint_path,
        target_path,
        repeatability_report_path,
        output_dir,
    ) = resolved
    repeatability = json.loads(repeatability_report_path.read_text(encoding="utf-8"))
    if (
        repeatability.get("repeatability_status") != "passed"
        or repeatability.get("action_collection_authorized") is not True
    ):
        raise LTSNContractError("V5.2d repeatability failed; action collection is blocked")
    expected_repeatability = {
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "implementation_sha256": _implementation_hashes(
            root,
            _context(
                root=root,
                ace_config_path=ace_config_path,
                fingerprint_path=fingerprint_path,
                target_path=target_path,
            )[0],
        ),
    }
    for name, expected in expected_repeatability.items():
        if repeatability.get(name) != expected:
            raise LTSNContractError(f"V5.2d repeatability provenance mismatch: {name}")
    source_rows = _read_csv(source_manifest_path)
    if {row["ace_model_sha256"] for row in source_rows} != {ace_model_sha256}:
        raise LTSNContractError("V5.2d ACE hash differs from source manifest")
    if {row["vae_sha256"] for row in source_rows} != {vae_sha256}:
        raise LTSNContractError("V5.2d VAE hash differs from source manifest")
    config, scorer, target = _context(
        root=root,
        ace_config_path=ace_config_path,
        fingerprint_path=fingerprint_path,
        target_path=target_path,
    )
    specs = _load_anchor_specs(
        source_manifest_path=source_manifest_path,
        v52c_plan_path=v52c_plan_path,
        prompt_manifest_path=prompt_manifest_path,
        anchor_count=8,
    )
    planned = []
    for spec in specs:
        for basis in V52D_ALL_BASES:
            for scale in V52D_SCALES:
                for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                    planned.append(
                        {
                            **spec,
                            "basis_name": basis,
                            "action_scale": scale,
                            "sign": sign,
                            "sample_id": (
                                f"{spec['anchor_sample_id']}__v52d_{basis}_"
                                f"s{int(scale * 100):03d}_{sign_name}"
                            ),
                        }
                    )
    planned.sort(key=lambda row: row["sample_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "experiment": V52D_EXPERIMENT,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "v52c_plan_sha256": sha256_file(v52c_plan_path),
        "repeatability_report_sha256": sha256_file(repeatability_report_path),
        "ace_config_sha256": sha256_file(ace_config_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "target_sha256": sha256_file(target_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "implementation_sha256": _implementation_hashes(root, config),
        "anchor_count": 8,
        "on_manifold_bases": list(V52D_ON_MANIFOLD_BASES),
        "matched_random_bases": list(V52D_RANDOM_BASES),
        "scales": list(V52D_SCALES),
        "cfg_residual": V52D_CFG_STATUS,
        "planned_samples": len(planned),
        "planned": planned,
    }
    plan_path = output_dir / "v52d_action_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise LTSNContractError("V5.2d action plan changed; use a new output directory")
    if not plan_path.exists():
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    from .ace_adapter import AceStepAdapter

    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    all_descriptors: list[dict[str, str]] = []
    all_records: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(planned), batch_size)):
        items = planned[start : start + batch_size]
        batch_dir = output_dir / "batches" / f"batch_{batch_index:05d}"
        logical_ids = [str(row["sample_id"]) for row in items]
        completed = _completed_batch(batch_dir, plan_sha256=plan_sha256, logical_ids=logical_ids)
        if completed is not None:
            descriptors, records = completed
        else:
            records = []
            for item in items:
                hook = OnManifoldActionHook(
                    anchor_id=item["anchor_sample_id"],
                    target_step=int(item["step_number"]),
                    basis_name=item["basis_name"],
                    scale=float(item["action_scale"]),
                    sign=float(item["sign"]),
                )
                records.append(
                    _generate_record(
                        adapter=adapter,
                        config=config,
                        spec=item,
                        sample_id=item["sample_id"],
                        hook=hook,
                        output_dir=output_dir,
                        ace_model_sha256=ace_model_sha256,
                        vae_sha256=vae_sha256,
                        extra={
                            "basis_name": item["basis_name"],
                            "basis_role": "on_manifold"
                            if item["basis_name"] in V52D_ON_MANIFOLD_BASES
                            else "matched_random_control",
                            "action_scale": item["action_scale"],
                            "sign": item["sign"],
                        },
                    )
                )
            descriptors, records = _exact_batch(
                root=root,
                output_dir=output_dir,
                batch_index=batch_index,
                plan_sha256=plan_sha256,
                records=records,
                exact_repeats=1,
                workers=workers,
                materialize_mode=materialize_mode,
            )
        all_descriptors.extend(descriptors)
        all_records.extend(records)
    basis_audits: dict[str, set[str]] = {}
    for record in all_records:
        encoded = json.dumps(
            record["action_audit"]["basis_sha256"], sort_keys=True, separators=(",", ":")
        )
        basis_audits.setdefault(record["anchor_sample_id"], set()).add(encoded)
    if any(len(values) != 1 for values in basis_audits.values()):
        raise LTSNContractError("V5.2d action basis changed across matched rollouts")
    points = _score_rows(
        descriptors=all_descriptors, records=all_records, scorer=scorer, target=target
    )
    points_path = output_dir / "v52d_action_points.csv"
    write_csv_atomic(points_path, points)
    report, outcomes = analyze_actions(points, repeatability_report=repeatability)
    outcomes_path = output_dir / "v52d_action_outcomes.csv"
    write_csv_atomic(outcomes_path, outcomes)
    report.update(
        {
            "plan_sha256": plan_sha256,
            "points_sha256": sha256_file(points_path),
            "outcomes_sha256": sha256_file(outcomes_path),
            "repeatability_report_sha256": sha256_file(repeatability_report_path),
            "retained_wav_files": len(list(output_dir.rglob("*.wav"))),
            "retained_latent_files": len(list((output_dir / "latents").glob("*.npy"))),
        }
    )
    if report["retained_wav_files"] != 0:
        raise LTSNContractError("V5.2d action cleanup retained WAV")
    report_path = output_dir / "v52d_action_report.json"
    write_json_atomic(report_path, report)
    return report
