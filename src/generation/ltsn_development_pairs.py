"""Scope-limited baseline/guided pair generation for LTSN development evidence."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ace_adapter import AceStepAdapter, GenerationRequest
from .exact_features import (
    compute_frozen_18d_descriptors,
    extract_candidate_features,
    preprocess_candidates,
    write_descriptor_csv,
)
from .experiment import CandidateRecord, load_experiment_config
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_evaluation import _load_ensemble
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .path_homology_exact_scorer import ExactPathHomologyScorer
from .topology_corrector import TopologyCorrector, TopologyCorrectorConfig

AUTHORIZATION_SCOPE = "development_only"
GENERATION_MANIFEST = "development_generation_manifest.csv"
RAW_PAIR_TABLE = "development_pairs_raw.csv"
FINAL_PAIR_TABLE = "development_pairs.csv"
PAIR_DECODER_CONTRACT = "ace_vae_decode_latent_to_audio_v1"


def _relative(path: Path, parent: Path) -> str:
    return path.resolve().relative_to(parent.resolve()).as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_development_plan(
    prompt_manifest: Path,
    *,
    seed_start: int,
    seeds_per_prompt: int,
    expected_development_prompts: int,
) -> list[dict[str, Any]]:
    """Expand only development prompts while preserving full-manifest seed indices."""

    if seeds_per_prompt < 1 or expected_development_prompts < 1:
        raise ValueError("seed and prompt counts must be positive")
    source = _read_csv(prompt_manifest)
    if not source:
        raise LTSNContractError("prompt manifest is empty")
    prompt_ids = [row.get("prompt_id", "").strip() for row in source]
    if any(not value for value in prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise LTSNContractError("prompt manifest has empty or duplicate prompt IDs")
    output: list[dict[str, Any]] = []
    development_count = 0
    for prompt_index, row in enumerate(source):
        if row.get("split", "").strip() != "development":
            continue
        development_count += 1
        if not row.get("caption", "").strip():
            raise LTSNContractError("development prompt has an empty caption")
        if row.get("seed", "").strip():
            raise LTSNContractError(
                "formal development pairs require globally indexed generated seeds"
            )
        for seed_index in range(seeds_per_prompt):
            seed = seed_start + prompt_index * seeds_per_prompt + seed_index
            output.append(
                {
                    "pair_id": f"{row['prompt_id']}__seed{seed}",
                    "prompt_id": row["prompt_id"],
                    "prompt_index": prompt_index,
                    "caption": row["caption"],
                    "seed": seed,
                    "bpm": int(row["bpm"]) if row.get("bpm", "").strip() else None,
                    "keyscale": row.get("keyscale", "").strip(),
                    "timesignature": row.get("timesignature", "").strip(),
                }
            )
    if development_count != expected_development_prompts:
        raise LTSNContractError(
            "development prompt count changed: "
            f"expected {expected_development_prompts}, found {development_count}"
        )
    pair_ids = [row["pair_id"] for row in output]
    seeds = [row["seed"] for row in output]
    if len(set(pair_ids)) != len(pair_ids) or len(set(seeds)) != len(seeds):
        raise LTSNContractError("development pair plan contains duplicate IDs or seeds")
    return output


def _validate_runtime_bindings(
    *,
    ensemble: Mapping[str, Any],
    calibration: Mapping[str, Any],
    ensemble_manifest: Path,
    fingerprint_sha256: str,
    ace_model_sha256: str,
    vae_sha256: str,
) -> Mapping[str, Any]:
    metadata = ensemble.get("metadata")
    if not isinstance(metadata, dict):
        raise LTSNContractError("ensemble metadata is missing")
    if ensemble.get("qualification_eligible") is not True:
        raise LTSNContractError("development pairs require a qualification-eligible ensemble")
    if metadata.get("qualification_eligible") is not True:
        raise LTSNContractError("ensemble metadata is not qualification eligible")
    if metadata.get("guidance_promotion_eligible") is not False:
        raise LTSNContractError("ensemble metadata must not claim guidance promotion")
    expected = {
        "fingerprint_json_sha256": fingerprint_sha256,
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "model_family": "acestep-v15-xl-turbo",
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise LTSNContractError(f"ensemble {name} binding mismatch")
    if calibration.get("status") != "frozen":
        raise LTSNContractError("development correction requires frozen calibration")
    if calibration.get("qualification_eligible") is not True:
        raise LTSNContractError("calibration is not qualification eligible")
    calibration_bindings = {
        "fingerprint_json_sha256": fingerprint_sha256,
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest),
    }
    for name, value in calibration_bindings.items():
        if calibration.get(name) != value:
            raise LTSNContractError(f"calibration {name} binding mismatch")
    return metadata


def _corrector_config(calibration: Mapping[str, Any]) -> TopologyCorrectorConfig:
    return TopologyCorrectorConfig(
        enabled=True,
        qualification_passed=False,
        authorization_scope=AUTHORIZATION_SCOPE,
        guidance_scale=1.0,
        rms_clip_ratio=0.005,
        step_weights={4: 0.5, 5: 1.0, 6: 0.5},
        ood_probability_threshold=float(calibration["ood_probability_threshold"]),
        max_aleatoric_variance=float(calibration["max_aleatoric_variance"]),
        max_epistemic_variance=float(calibration["max_epistemic_variance"]),
        max_interval_width=float(calibration["max_interval_width"]),
        variance_scale=tuple(float(value) for value in calibration["variance_scale"]),
    )


def _latent_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 64 or not np.isfinite(array).all():
        raise LTSNContractError("ACE final pred_latents must have finite shape [1,T,64]")
    return array


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _proxy_focus_loss(
    latent: np.ndarray,
    models: Sequence[Any],
    contract: Any,
    device: torch.device,
) -> tuple[float, float]:
    tensor = torch.from_numpy(latent).unsqueeze(0).to(device)
    mask = torch.ones(tensor.shape[:2], dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits = [model(tensor, 0.0, 8, mask).focus_logit.float() for model in models]
    focus_logit = float(torch.stack(logits).mean().cpu())
    if not math.isfinite(focus_logit):
        raise LTSNContractError("LTSN ensemble produced a non-finite final-latent score")
    loss = max(0.0, float(contract.focus_band_threshold) - focus_logit) ** 2
    return focus_logit, loss


def _materialize_identical_audio(source: Path, target: Path) -> None:
    """Atomically reuse one decoded WAV for a latent-identical paired arm."""

    if not source.is_file():
        raise LTSNContractError(f"canonical baseline audio is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.canonical.part")
    if temporary.exists():
        temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonicalize_noop_pair_audio(
    baseline: Mapping[str, Any],
    guided: dict[str, Any],
    *,
    output_dir: Path,
    receipt_path: Path | None = None,
) -> bool:
    """Reuse baseline audio when the guided arm has an identical final latent."""

    latent_changed = baseline["latent_sha256"] != guided["latent_sha256"]
    if latent_changed:
        return True
    baseline_audio = output_dir / str(baseline["audio_path"])
    guided_audio = output_dir / str(guided["audio_path"])
    baseline_sha256 = sha256_file(baseline_audio)
    if baseline_sha256 != baseline["audio_sha256"]:
        raise LTSNContractError("canonical baseline audio is hash-mismatched")
    if not guided_audio.is_file() or sha256_file(guided_audio) != baseline_sha256:
        _materialize_identical_audio(baseline_audio, guided_audio)
    guided["audio_sha256"] = baseline_sha256
    guided["audio_derivation"] = "baseline_reuse_for_identical_latent"
    guided["canonical_audio_source_candidate_id"] = baseline["candidate_id"]
    if receipt_path is not None:
        write_json_atomic(receipt_path, guided)
    _validate_pair_decode_identity(baseline, guided)
    return False


def _resume_arm(
    receipt_path: Path,
    output_dir: Path,
    plan_sha256: str,
    reference_arm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("plan_sha256") != plan_sha256:
        raise LTSNContractError("development arm receipt belongs to a different plan")
    if receipt.get("authorization_scope") != AUTHORIZATION_SCOPE:
        raise LTSNContractError("development arm receipt has an invalid scope")
    if receipt.get("decoder_contract") != PAIR_DECODER_CONTRACT:
        raise LTSNContractError("development arm used a different decoder contract")
    latent_path = output_dir / receipt["latent_path"]
    if not latent_path.is_file() or sha256_file(latent_path) != receipt["latent_sha256"]:
        raise LTSNContractError("development arm latent is missing or hash-mismatched")
    if reference_arm is not None:
        _canonicalize_noop_pair_audio(
            reference_arm,
            receipt,
            output_dir=output_dir,
            receipt_path=receipt_path,
        )
    audio_path = output_dir / receipt["audio_path"]
    if not audio_path.is_file() or sha256_file(audio_path) != receipt["audio_sha256"]:
        raise LTSNContractError("development arm audio is missing or hash-mismatched")
    return receipt


def _generate_arm(
    *,
    adapter: AceStepAdapter,
    row: Mapping[str, Any],
    arm: str,
    output_dir: Path,
    plan_sha256: str,
    models: Sequence[Any],
    contract: Any,
    device: torch.device,
    corrector: TopologyCorrector,
    duration_seconds: float,
    inference_steps: int,
    reference_arm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = f"{row['pair_id']}__{arm}"
    receipt_path = output_dir / "receipts" / f"{candidate_id}.json"
    if receipt_path.is_file():
        receipt = _resume_arm(
            receipt_path,
            output_dir,
            plan_sha256,
            reference_arm=reference_arm,
        )
        expected = {
            "pair_id": row["pair_id"],
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "arm": arm,
            "candidate_id": candidate_id,
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise LTSNContractError(f"development arm receipt binding mismatch: {candidate_id}")
        return receipt
    audio_path = output_dir / "data_raw" / "development" / arm / f"{candidate_id}.wav"
    latent_path = output_dir / "latents" / "development" / arm / f"{candidate_id}.npy"
    if audio_path.exists() or latent_path.exists():
        raise LTSNContractError(
            f"unreceipted development artifact exists for {candidate_id}; use a new output dir"
        )
    adapter.set_topology_corrector(None if arm == "baseline" else corrector)
    result = adapter.generate(
        GenerationRequest(
            prompt=str(row["caption"]),
            seed=int(row["seed"]),
            duration_seconds=duration_seconds,
            output_dir=output_dir / "generator_output" / str(row["pair_id"]) / arm,
            inference_steps=inference_steps,
            bpm=row["bpm"],
            keyscale=str(row["keyscale"]),
            timesignature=str(row["timesignature"]),
        )
    )
    if result.seed != int(row["seed"]):
        raise LTSNContractError("ACE returned a different seed for a development arm")
    latent = _latent_array(result.final_latent)
    _save_npy_atomic(latent_path, latent)
    latent_sha256 = sha256_file(latent_path)
    reuse_reference_audio = (
        reference_arm is not None
        and reference_arm["latent_sha256"] == latent_sha256
    )
    if reuse_reference_audio:
        _materialize_identical_audio(
            output_dir / str(reference_arm["audio_path"]), audio_path
        )
    else:
        adapter.decode_latent_to_audio(latent, audio_path)
    generated_audio = result.audio_path.resolve()
    if generated_audio != audio_path.resolve() and generated_audio.is_file():
        generated_audio.unlink()
    focus_logit, focus_loss = _proxy_focus_loss(latent, models, contract, device)
    receipt = {
        "schema_version": 2,
        "pair_id": row["pair_id"],
        "prompt_id": row["prompt_id"],
        "seed": row["seed"],
        "arm": arm,
        "candidate_id": candidate_id,
        "audio_path": _relative(audio_path, output_dir),
        "audio_sha256": sha256_file(audio_path),
        "latent_path": _relative(latent_path, output_dir),
        "latent_sha256": latent_sha256,
        "proxy_focus_logit": focus_logit,
        "proxy_focus_band_loss": focus_loss,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "decoder_contract": PAIR_DECODER_CONTRACT,
        "audio_derivation": (
            "baseline_reuse_for_identical_latent"
            if reuse_reference_audio
            else "ace_vae_decode_latent_to_audio"
        ),
        "canonical_audio_source_candidate_id": (
            reference_arm["candidate_id"] if reuse_reference_audio else ""
        ),
        "plan_sha256": plan_sha256,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _validate_pair_decode_identity(
    baseline: Mapping[str, Any], guided: Mapping[str, Any]
) -> bool:
    """Require deterministic shared decoding whenever guidance is a latent no-op."""

    latent_changed = baseline["latent_sha256"] != guided["latent_sha256"]
    if not latent_changed and baseline["audio_sha256"] != guided["audio_sha256"]:
        raise LTSNContractError(
            "identical baseline/guided latents decoded to different audio; "
            "the paired decoder contract is not deterministic"
        )
    return latent_changed


def _validate_partial_generation_manifest(
    path: Path,
    *,
    plan_rows: Sequence[Mapping[str, Any]],
    plan_sha256: str,
    output_dir: Path,
) -> None:
    rows = _read_csv(path)
    if len(rows) > len(plan_rows) or [row.get("pair_id") for row in rows] != [
        row["pair_id"] for row in plan_rows[: len(rows)]
    ]:
        raise LTSNContractError("resumed generation manifest is not a valid plan prefix")
    for row, planned in zip(rows, plan_rows, strict=False):
        if (
            row.get("authorization_scope") != AUTHORIZATION_SCOPE
            or row.get("plan_sha256") != plan_sha256
            or row.get("decoder_contract") != PAIR_DECODER_CONTRACT
            or row.get("prompt_id") != planned["prompt_id"]
            or int(row["seed"]) != planned["seed"]
        ):
            raise LTSNContractError("resumed generation manifest binding mismatch")
        for arm in ("baseline", "guided"):
            for kind in ("audio", "latent"):
                artifact = output_dir / row[f"{arm}_{kind}_path"]
                if (
                    not artifact.is_file()
                    or sha256_file(artifact) != row[f"{arm}_{kind}_sha256"]
                ):
                    raise LTSNContractError(
                        f"resumed generation artifact hash mismatch: {planned['pair_id']}"
                    )
        _validate_pair_decode_identity(
            {
                "latent_sha256": row["baseline_latent_sha256"],
                "audio_sha256": row["baseline_audio_sha256"],
            },
            {
                "latent_sha256": row["guided_latent_sha256"],
                "audio_sha256": row["guided_audio_sha256"],
            },
        )


def generate_development_pairs(
    *,
    root: Path,
    ace_config: Path,
    prompt_manifest: Path,
    fingerprint_path: Path,
    ensemble_manifest: Path,
    calibration_path: Path,
    output_dir: Path,
    ace_model_sha256: str,
    vae_sha256: str,
    seed_start: int = 2026071600,
    seeds_per_prompt: int = 4,
    expected_development_prompts: int = 64,
    duration_seconds: float = 180.0,
    device_name: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    """Generate same-prompt/seed baseline and development-only guided arms."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    ace_config = ace_config if ace_config.is_absolute() else root / ace_config
    prompt_manifest = (
        prompt_manifest if prompt_manifest.is_absolute() else root / prompt_manifest
    )
    fingerprint_path = (
        fingerprint_path if fingerprint_path.is_absolute() else root / fingerprint_path
    )
    ensemble_manifest = (
        ensemble_manifest if ensemble_manifest.is_absolute() else root / ensemble_manifest
    )
    calibration_path = (
        calibration_path if calibration_path.is_absolute() else root / calibration_path
    )
    if (
        seeds_per_prompt != 4
        or expected_development_prompts != 64
        or not math.isclose(duration_seconds, 180.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise LTSNContractError(
            "formal development-only generation requires 64 prompts x 4 seeds at 180s"
        )
    config = load_experiment_config(root, ace_config)
    if config.ace.inference_steps != 8:
        raise LTSNContractError("formal development pairs require 8 ACE inference steps")
    plan_rows = build_development_plan(
        prompt_manifest,
        seed_start=seed_start,
        seeds_per_prompt=seeds_per_prompt,
        expected_development_prompts=expected_development_prompts,
    )
    device = torch.device(device_name)
    contract, ensemble, models = _load_ensemble(ensemble_manifest, fingerprint_path, device)
    if len(models) != 3:
        raise LTSNContractError("formal development pairs require the frozen three-seed ensemble")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    metadata = _validate_runtime_bindings(
        ensemble=ensemble,
        calibration=calibration,
        ensemble_manifest=ensemble_manifest,
        fingerprint_sha256=contract.artifact_sha256,
        ace_model_sha256=ace_model_sha256,
        vae_sha256=vae_sha256,
    )
    corrector_config = _corrector_config(calibration)
    corrector = TopologyCorrector(
        models,
        contract,
        [metadata for _ in models],
        corrector_config,
    )
    plan_path = output_dir / "development_generation_plan.json"
    plan = {
        "schema_version": 2,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
        "ace_config_sha256": sha256_file(ace_config),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "ensemble_manifest_sha256": sha256_file(ensemble_manifest),
        "calibration_sha256": sha256_file(calibration_path),
        "ace_model_sha256": ace_model_sha256,
        "vae_sha256": vae_sha256,
        "seed_start": seed_start,
        "seeds_per_prompt": seeds_per_prompt,
        "expected_development_prompts": expected_development_prompts,
        "duration_seconds": duration_seconds,
        "inference_steps": config.ace.inference_steps,
        "decoder_contract": PAIR_DECODER_CONTRACT,
        "corrector": {
            "guidance_scale": corrector_config.guidance_scale,
            "rms_clip_ratio": corrector_config.rms_clip_ratio,
            "step_weights": {
                str(step): weight for step, weight in corrector_config.step_weights.items()
            },
        },
        "planned_pairs": plan_rows,
    }
    if plan_path.is_file():
        if not resume:
            raise FileExistsError(
                "development generation plan already exists; pass --resume or use a new output dir"
            )
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise LTSNContractError("development generation plan changed; use a new output dir")
    else:
        if (output_dir / GENERATION_MANIFEST).exists():
            raise LTSNContractError("generation manifest exists without its frozen plan")
        write_json_atomic(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    manifest_path = output_dir / GENERATION_MANIFEST
    if manifest_path.is_file():
        _validate_partial_generation_manifest(
            manifest_path,
            plan_rows=plan_rows,
            plan_sha256=plan_sha256,
            output_dir=output_dir,
        )
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    completed: list[dict[str, Any]] = []
    for row in plan_rows:
        baseline = _generate_arm(
            adapter=adapter,
            row=row,
            arm="baseline",
            output_dir=output_dir,
            plan_sha256=plan_sha256,
            models=models,
            contract=contract,
            device=device,
            corrector=corrector,
            duration_seconds=duration_seconds,
            inference_steps=config.ace.inference_steps,
        )
        guided = _generate_arm(
            adapter=adapter,
            row=row,
            arm="guided",
            output_dir=output_dir,
            plan_sha256=plan_sha256,
            models=models,
            contract=contract,
            device=device,
            corrector=corrector,
            duration_seconds=duration_seconds,
            inference_steps=config.ace.inference_steps,
            reference_arm=baseline,
        )
        latent_changed = _canonicalize_noop_pair_audio(
            baseline,
            guided,
            output_dir=output_dir,
            receipt_path=(
                output_dir / "receipts" / f"{guided['candidate_id']}.json"
            ),
        )
        arms = {"baseline": baseline, "guided": guided}
        completed.append(
            {
                "pair_id": row["pair_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "fingerprint_json_sha256": contract.artifact_sha256,
                "baseline_candidate_id": arms["baseline"]["candidate_id"],
                "guided_candidate_id": arms["guided"]["candidate_id"],
                "baseline_audio_path": arms["baseline"]["audio_path"],
                "guided_audio_path": arms["guided"]["audio_path"],
                "baseline_audio_sha256": arms["baseline"]["audio_sha256"],
                "guided_audio_sha256": arms["guided"]["audio_sha256"],
                "baseline_latent_path": arms["baseline"]["latent_path"],
                "guided_latent_path": arms["guided"]["latent_path"],
                "baseline_latent_sha256": arms["baseline"]["latent_sha256"],
                "guided_latent_sha256": arms["guided"]["latent_sha256"],
                "latent_changed": str(latent_changed).lower(),
                "decoder_contract": PAIR_DECODER_CONTRACT,
                "proxy_focus_band_loss_before": arms["baseline"][
                    "proxy_focus_band_loss"
                ],
                "proxy_focus_band_loss_after": arms["guided"]["proxy_focus_band_loss"],
                "authorization_scope": AUTHORIZATION_SCOPE,
                "plan_sha256": plan_sha256,
            }
        )
        write_csv_atomic(manifest_path, completed)
    return {
        "authorization_scope": AUTHORIZATION_SCOPE,
        "pairs": len(completed),
        "prompts": expected_development_prompts,
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": sha256_file(manifest_path),
        "plan_sha256": plan_sha256,
    }


def _validate_generation_rows(
    rows: Sequence[dict[str, str]], output_dir: Path, plan: Mapping[str, Any]
) -> None:
    expected = {row["pair_id"]: row for row in plan["planned_pairs"]}
    observed = {row.get("pair_id", ""): row for row in rows}
    if len(observed) != len(rows) or set(observed) != set(expected):
        raise LTSNContractError("generation manifest is incomplete or has duplicate pairs")
    plan_sha256 = sha256_file(output_dir / "development_generation_plan.json")
    for pair_id, row in observed.items():
        planned = expected[pair_id]
        if row.get("authorization_scope") != AUTHORIZATION_SCOPE:
            raise LTSNContractError("generation manifest contains a non-development scope")
        if row.get("prompt_id") != planned["prompt_id"] or int(row["seed"]) != planned["seed"]:
            raise LTSNContractError(f"generation pair binding mismatch: {pair_id}")
        if row.get("plan_sha256") != plan_sha256:
            raise LTSNContractError("generation row belongs to a different plan")
        if row.get("decoder_contract") != PAIR_DECODER_CONTRACT:
            raise LTSNContractError("generation row used a different decoder contract")
        for arm in ("baseline", "guided"):
            for kind in ("audio", "latent"):
                path = output_dir / row[f"{arm}_{kind}_path"]
                if not path.is_file() or sha256_file(path) != row[f"{arm}_{kind}_sha256"]:
                    raise LTSNContractError(f"{pair_id} {arm} {kind} hash mismatch")
        _validate_pair_decode_identity(
            {
                "latent_sha256": row["baseline_latent_sha256"],
                "audio_sha256": row["baseline_audio_sha256"],
            },
            {
                "latent_sha256": row["guided_latent_sha256"],
                "audio_sha256": row["guided_audio_sha256"],
            },
        )


def score_development_pairs(
    *,
    root: Path,
    ace_config: Path,
    fingerprint_path: Path,
    output_dir: Path,
    workers: int = 4,
) -> dict[str, Any]:
    """Run the frozen exact 18-D scorer on both decoded arms."""

    output_dir = output_dir.resolve()
    root = root.resolve()
    ace_config = ace_config if ace_config.is_absolute() else root / ace_config
    fingerprint_path = (
        fingerprint_path if fingerprint_path.is_absolute() else root / fingerprint_path
    )
    plan_path = output_dir / "development_generation_plan.json"
    manifest_path = output_dir / GENERATION_MANIFEST
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = _read_csv(manifest_path)
    _validate_generation_rows(rows, output_dir, plan)
    config = load_experiment_config(root, ace_config)
    scorer = ExactPathHomologyScorer.from_json(
        fingerprint_path, expected_sha256=plan["fingerprint_json_sha256"]
    )
    planned = {row["pair_id"]: row for row in plan["planned_pairs"]}
    records: list[CandidateRecord] = []
    for row in rows:
        prompt = planned[row["pair_id"]]
        for index, arm in enumerate(("baseline", "guided")):
            records.append(
                CandidateRecord(
                    experiment_id="ltsn_development_only",
                    prompt_id=row["prompt_id"],
                    caption=prompt["caption"],
                    candidate_index=index,
                    candidate_id=row[f"{arm}_candidate_id"],
                    seed=int(row["seed"]),
                    duration_seconds=float(plan["duration_seconds"]),
                    bpm=prompt["bpm"],
                    keyscale=prompt["keyscale"],
                    timesignature=prompt["timesignature"],
                    status="generated",
                    audio_relative_path=row[f"{arm}_audio_path"],
                    audio_sha256=row[f"{arm}_audio_sha256"],
                    latent_relative_path=row[f"{arm}_latent_path"],
                    latent_sha256=row[f"{arm}_latent_sha256"],
                )
            )
    processed = preprocess_candidates(root, output_dir, records, workers=workers)
    features = extract_candidate_features(root, output_dir, processed, workers=workers)
    descriptors = compute_frozen_18d_descriptors(
        root, output_dir, records, features, scorer
    )
    descriptor_path = output_dir / "development_exact_descriptors.csv"
    write_descriptor_csv(descriptor_path, descriptors)
    descriptor_sha256 = sha256_file(descriptor_path)
    generation_manifest_sha256 = sha256_file(manifest_path)
    plan_sha256 = sha256_file(plan_path)
    descriptor_by_id = {row["candidate_id"]: row for row in descriptors}
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        baseline = descriptor_by_id[row["baseline_candidate_id"]]
        guided = descriptor_by_id[row["guided_candidate_id"]]
        baseline_eligible = (
            float(baseline["raw_clip_fraction"]) <= config.scoring.maximum_clip_fraction
            and float(baseline["raw_rms"]) >= config.scoring.minimum_rms
            and float(baseline["raw_dc_offset"]) <= config.scoring.maximum_dc_offset
        )
        guided_eligible = (
            float(guided["raw_clip_fraction"]) <= config.scoring.maximum_clip_fraction
            and float(guided["raw_rms"]) >= config.scoring.minimum_rms
            and float(guided["raw_dc_offset"]) <= config.scoring.maximum_dc_offset
        )
        raw_rows.append(
            {
                "pair_id": row["pair_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "generation_manifest_sha256": generation_manifest_sha256,
                "generation_plan_sha256": plan_sha256,
                "exact_descriptor_table_sha256": descriptor_sha256,
                "baseline_candidate_id": row["baseline_candidate_id"],
                "guided_candidate_id": row["guided_candidate_id"],
                "baseline_audio_sha256": row["baseline_audio_sha256"],
                "guided_audio_sha256": row["guided_audio_sha256"],
                "baseline_latent_sha256": row["baseline_latent_sha256"],
                "guided_latent_sha256": row["guided_latent_sha256"],
                "latent_changed": row["latent_changed"],
                "decoder_contract": row["decoder_contract"],
                "exact_focus_band_loss_before": baseline["focus_band_loss"],
                "exact_focus_band_loss_after": guided["focus_band_loss"],
                "proxy_focus_band_loss_before": row["proxy_focus_band_loss_before"],
                "proxy_focus_band_loss_after": row["proxy_focus_band_loss_after"],
                "baseline_technical_quality_eligible": str(baseline_eligible).lower(),
                "guided_technical_quality_eligible": str(guided_eligible).lower(),
                "authorization_scope": AUTHORIZATION_SCOPE,
            }
        )
    raw_path = output_dir / RAW_PAIR_TABLE
    write_csv_atomic(raw_path, raw_rows)
    return {
        "authorization_scope": AUTHORIZATION_SCOPE,
        "pairs": len(raw_rows),
        "descriptor_table_sha256": descriptor_sha256,
        "raw_pair_table_sha256": sha256_file(raw_path),
    }


def _cluster_bootstrap_mean(
    differences: np.ndarray,
    prompt_ids: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    prompts = np.unique(prompt_ids)
    if not len(prompts) or resamples < 1:
        raise ValueError("cluster bootstrap requires prompts and resamples")
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        chosen = rng.choice(prompts, len(prompts), replace=True)
        values = np.concatenate([differences[prompt_ids == prompt] for prompt in chosen])
        samples[index] = np.mean(values)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def finalize_development_pairs(
    *,
    raw_pair_table: Path,
    evidence_table: Path,
    protocol_path: Path,
    output_dir: Path,
    bootstrap_resamples: int = 2000,
    seed: int = 20260716,
) -> dict[str, Any]:
    """Bind external evidence and issue a non-promotable development pair table."""

    raw_rows = _read_csv(raw_pair_table)
    evidence_rows = _read_csv(evidence_table)
    if not raw_rows or not evidence_rows:
        raise LTSNContractError("raw pairs and numeric non-inferiority evidence are required")
    raw = {row.get("pair_id", ""): row for row in raw_rows}
    evidence = {row.get("pair_id", ""): row for row in evidence_rows}
    if len(raw) != len(raw_rows) or len(evidence) != len(evidence_rows):
        raise LTSNContractError("raw or evidence pair IDs are empty or duplicated")
    if set(raw) != set(evidence):
        raise LTSNContractError("evidence table must contain exactly the generated pair IDs")
    fingerprints = {row.get("fingerprint_json_sha256", "") for row in raw_rows}
    if len(fingerprints) != 1 or len(next(iter(fingerprints))) != 64:
        raise LTSNContractError("raw pairs do not share one frozen fingerprint")
    prompt_counts: dict[str, int] = {}
    for row in raw_rows:
        prompt_id = row.get("prompt_id", "")
        prompt_counts[prompt_id] = prompt_counts.get(prompt_id, 0) + 1
    seeds = {int(row["seed"]) for row in raw_rows}
    if (
        len(raw_rows) != 256
        or len(prompt_counts) != 64
        or set(prompt_counts.values()) != {4}
        or len(seeds) != 256
    ):
        raise LTSNContractError(
            "formal development evidence requires 64 prompts x 4 paired seeds"
        )
    for name in (
        "generation_manifest_sha256",
        "generation_plan_sha256",
        "exact_descriptor_table_sha256",
    ):
        values = {row.get(name, "") for row in raw_rows}
        if len(values) != 1 or len(next(iter(values))) != 64:
            raise LTSNContractError(f"raw pairs do not share one {name}")
    for pair_id in raw:
        left, right = raw[pair_id], evidence[pair_id]
        if left.get("authorization_scope") != AUTHORIZATION_SCOPE:
            raise LTSNContractError("raw pair table is not development-only")
        if right.get("prompt_id") != left.get("prompt_id") or int(right["seed"]) != int(
            left["seed"]
        ):
            raise LTSNContractError(f"evidence pair binding mismatch: {pair_id}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_version = protocol.get("schema_version")
    if protocol_version not in {1, 2} or protocol.get("status") != "frozen_before_generation":
        raise LTSNContractError("non-inferiority protocol is not frozen-before-generation")
    specifications = protocol.get("criteria")
    if not isinstance(specifications, dict) or set(specifications) != {
        "quality",
        "prompt",
        "diversity",
    }:
        raise LTSNContractError("protocol must define quality, prompt, and diversity")
    if protocol_version == 2 and protocol.get("gate_contract") != "latent_guidance_promotion_v2":
        raise LTSNContractError("V2 protocol has an invalid guidance gate contract")
    ordered_ids = sorted(raw)
    prompt_ids = np.asarray([raw[pair_id]["prompt_id"] for pair_id in ordered_ids])
    criteria: dict[str, Any] = {}
    for index, (name, specification) in enumerate(specifications.items()):
        baseline_column = f"{name}_baseline"
        guided_column = f"{name}_guided"
        evidence_required = bool(specification.get("evidence_required", True))
        gate_modes = specification.get(
            "gate_modes", ["development", "confirmation"] if protocol_version == 1 else []
        )
        if not isinstance(gate_modes, list) or any(
            value not in {"development", "confirmation"} for value in gate_modes
        ):
            raise LTSNContractError(f"invalid protocol gate_modes for {name}")
        try:
            baseline_text = [evidence[pair_id].get(baseline_column, "") for pair_id in ordered_ids]
            guided_text = [evidence[pair_id].get(guided_column, "") for pair_id in ordered_ids]
            evidence_available = all(
                str(value).strip() for value in (*baseline_text, *guided_text)
            )
            if not evidence_available:
                if evidence_required:
                    raise ValueError
                if any(str(value).strip() for value in (*baseline_text, *guided_text)):
                    raise ValueError
                criteria[name] = {
                    "passed": None,
                    "evidence_available": False,
                    "required_for_gate": False,
                    "gate_modes": gate_modes,
                    "metric": specification["metric"],
                    "direction": specification["direction"],
                    "margin": float(specification["margin"]),
                    "estimate": None,
                    "ci95": None,
                }
                continue
            baseline = np.asarray([float(value) for value in baseline_text])
            guided = np.asarray([float(value) for value in guided_text])
        except (KeyError, TypeError, ValueError) as exc:
            raise LTSNContractError(f"numeric {name} evidence is missing or malformed") from exc
        if not np.isfinite(baseline).all() or not np.isfinite(guided).all():
            raise LTSNContractError(f"numeric {name} evidence contains NaN or Inf")
        direction = specification.get("direction")
        if direction not in {"higher_is_better", "lower_is_better"}:
            raise LTSNContractError(f"invalid protocol direction for {name}")
        difference = guided - baseline
        if direction == "lower_is_better":
            difference = -difference
        low, high = _cluster_bootstrap_mean(
            difference,
            prompt_ids,
            resamples=bootstrap_resamples,
            seed=seed + index,
        )
        margin = float(specification["margin"])
        if margin < 0 or not math.isfinite(margin):
            raise LTSNContractError(f"invalid non-inferiority margin for {name}")
        criteria[name] = {
            "passed": low >= -margin,
            "evidence_available": True,
            "required_for_gate": bool(gate_modes),
            "gate_modes": gate_modes,
            "metric": specification["metric"],
            "direction": direction,
            "margin": margin,
            "estimate": float(np.mean(difference)),
            "ci95": [low, high],
        }
    guided_technical_quality = all(
        raw[pair_id].get("guided_technical_quality_eligible", "").lower() == "true"
        for pair_id in ordered_ids
    )
    quality_noninferior = criteria["quality"]["passed"]
    if protocol_version == 1:
        quality_noninferior = bool(quality_noninferior and guided_technical_quality)
    prompt_noninferior = criteria["prompt"]["passed"]
    diversity_preserved = criteria["diversity"]["passed"]
    final_rows: list[dict[str, Any]] = []
    for pair_id in ordered_ids:
        final_rows.append(
            {
                **raw[pair_id],
                "quality_baseline": evidence[pair_id].get("quality_baseline", ""),
                "quality_guided": evidence[pair_id].get("quality_guided", ""),
                "prompt_baseline": evidence[pair_id]["prompt_baseline"],
                "prompt_guided": evidence[pair_id]["prompt_guided"],
                "diversity_baseline": evidence[pair_id]["diversity_baseline"],
                "diversity_guided": evidence[pair_id]["diversity_guided"],
                "quality_noninferior": (
                    "not_evaluated"
                    if quality_noninferior is None
                    else str(quality_noninferior).lower()
                ),
                "prompt_noninferior": str(prompt_noninferior).lower(),
                "diversity_preserved": str(diversity_preserved).lower(),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / FINAL_PAIR_TABLE
    write_csv_atomic(final_path, final_rows)
    report = {
        "schema_version": 2 if protocol_version == 2 else 1,
        "mode": "development",
        "authorization_scope": AUTHORIZATION_SCOPE,
        "guidance_promotion_eligible": False,
        "pairs": len(final_rows),
        "prompts": len(np.unique(prompt_ids)),
        "fingerprint_json_sha256": next(iter(raw.values()))[
            "fingerprint_json_sha256"
        ],
        "raw_pair_table_sha256": sha256_file(raw_pair_table),
        "generation_manifest_sha256": next(iter(raw.values()))[
            "generation_manifest_sha256"
        ],
        "generation_plan_sha256": next(iter(raw.values()))["generation_plan_sha256"],
        "exact_descriptor_table_sha256": next(iter(raw.values()))[
            "exact_descriptor_table_sha256"
        ],
        "evidence_table_sha256": sha256_file(evidence_table),
        "protocol_sha256": sha256_file(protocol_path),
        "criteria": criteria,
        "all_guided_technical_quality_eligible": guided_technical_quality,
        "blind_quality_is_gate": False if protocol_version == 2 else True,
        "blind_quality_evidence_available": criteria["quality"]["evidence_available"],
        "quality_noninferior": quality_noninferior,
        "prompt_noninferior": prompt_noninferior,
        "diversity_preserved": diversity_preserved,
        "development_pairs_sha256": sha256_file(final_path),
    }
    report_path = output_dir / "development_noninferiority_report.json"
    write_json_atomic(report_path, report)
    return {**report, "report_sha256": sha256_file(report_path)}
