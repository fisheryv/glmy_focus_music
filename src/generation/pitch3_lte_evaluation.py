"""Held-out development guidance and go/no-go screen for V3-LTE."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ace_adapter import AceStepAdapter, GenerationRequest
from .experiment import load_experiment_config
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .pitch3_contract import load_pitch3_contract
from .pitch3_exact_scorer import ExactPitch3Scorer
from .pitch3_lte import Pitch3LTECorrector, Pitch3LTEGuidanceConfig
from .pitch3_lte_ensemble import load_pitch3_lte_ensemble
from .pitch3_lte_training import (
    Pitch3LTEDataset,
    PromptBatchSampler,
    _prediction_rows,
    _read_examples,
    collate_pitch3_lte,
    load_pitch3_lte_checkpoint,
    pitch3_lte_metrics,
)


def _load_lte_model_artifact(
    *,
    checkpoint_path: Path | None,
    ensemble_manifest_path: Path | None,
    device: torch.device,
    expected_sha256: str | None,
) -> tuple[Any, dict[str, Any], Path, str]:
    if (checkpoint_path is None) == (ensemble_manifest_path is None):
        raise ValueError("select exactly one V3-LTE checkpoint or ensemble manifest")
    if ensemble_manifest_path is not None:
        model, metadata = load_pitch3_lte_ensemble(
            ensemble_manifest_path,
            device=device,
            expected_sha256=expected_sha256,
        )
        return model, metadata, ensemble_manifest_path, "equal_weight_ensemble"
    assert checkpoint_path is not None
    model, metadata = load_pitch3_lte_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=expected_sha256,
    )
    return model, metadata, checkpoint_path, "checkpoint"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path, parent).replace("\\", "/")


def _prompt_tensors(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        hidden = torch.from_numpy(np.asarray(payload["hidden"], dtype=np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.asarray(payload["mask"], dtype=np.bool_)).unsqueeze(0)
    return hidden.to(device), mask.to(device)


def _latent_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 64 or not np.isfinite(array).all():
        raise LTSNContractError("ACE final pred_latents must have finite shape [1,T,64]")
    return array


def materialize_pitch3_lte_development_guidance(
    *,
    root: Path,
    fingerprint_path: Path,
    dataset_manifest: Path,
    source_manifest_path: Path,
    checkpoint_path: Path | None,
    ace_config_path: Path,
    prompt_manifest_path: Path,
    output_dir: Path,
    checkpoint_sha256: str | None = None,
    ensemble_manifest_path: Path | None = None,
    device_name: str = "cuda:0",
    workers: int = 8,
    exact_batch_size: int = 32,
    materialize_mode: str = "auto",
    retain_audio: bool = True,
) -> dict[str, Any]:
    """Run one Step-4 hook and complete Steps 5--8 for held-out evidence."""

    root = root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    fingerprint_path = resolve(fingerprint_path)
    dataset_manifest = resolve(dataset_manifest)
    source_manifest_path = resolve(source_manifest_path)
    checkpoint_path = resolve(checkpoint_path) if checkpoint_path is not None else None
    ensemble_manifest_path = (
        resolve(ensemble_manifest_path) if ensemble_manifest_path is not None else None
    )
    ace_config_path = resolve(ace_config_path)
    prompt_manifest_path = resolve(prompt_manifest_path)
    output_dir = resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_pitch3_contract(fingerprint_path)
    dataset_plan_path = dataset_manifest.parent / "pitch3_lte_dataset_plan.json"
    if not dataset_plan_path.is_file():
        raise LTSNContractError("V3-LTE dataset plan is missing")
    dataset_plan = json.loads(dataset_plan_path.read_text(encoding="utf-8"))
    if dataset_plan.get("source_manifest_sha256") != sha256_file(source_manifest_path):
        raise LTSNContractError("V3-LTE final guidance source differs from the training plan")
    source_rows = _read_csv(source_manifest_path)
    source_final = {
        row["trajectory_id"]: row
        for row in source_rows
        if row.get("split") == "development"
        and row.get("is_final", "false").lower() == "true"
        and float(row.get("ood_label", 0.0)) < 0.5
    }
    records = [
        record
        for record in _read_examples(dataset_manifest, contract.artifact_sha256)
        if record.split == "development" and record.source_kind == "base_step4_seed"
    ]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.prompt_id].append(record)
    if not grouped or any(len(values) != 4 for values in grouped.values()):
        raise LTSNContractError("V3-LTE guidance requires four held-out seeds per prompt")
    if {record.trajectory_id for record in records} != set(source_final):
        raise LTSNContractError(
            "V3-LTE final-latent baselines do not match development trajectories"
        )
    device = torch.device(device_name)
    model, metadata, model_artifact_path, model_artifact_kind = _load_lte_model_artifact(
        checkpoint_path=checkpoint_path,
        ensemble_manifest_path=ensemble_manifest_path,
        device=device,
        expected_sha256=checkpoint_sha256,
    )
    if metadata["fingerprint_json_sha256"] != contract.artifact_sha256:
        raise LTSNContractError("V3-LTE checkpoint fingerprint mismatch")
    if metadata["training_manifest_sha256"] != sha256_file(dataset_manifest):
        raise LTSNContractError("V3-LTE guidance dataset differs from checkpoint selection data")
    config = load_experiment_config(root, ace_config_path)
    os.environ["ACESTEP_DEVICE"] = device_name
    adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
    selected = sorted(records, key=lambda record: (record.prompt_id, record.sample_id))
    if len(grouped) != 64 or len(selected) != 256:
        raise LTSNContractError("formal V3-LTE guidance requires 64 prompts x 4 seeds")
    prompts = {row["prompt_id"]: row for row in _read_csv(prompt_manifest_path)}
    if not set(grouped).issubset(prompts):
        raise LTSNContractError("V3-LTE guidance prompt text is incomplete")

    def seed_of(record: Any) -> int:
        match = re.search(r"__seed(\d+)", record.trajectory_id)
        if match is None:
            raise LTSNContractError(
                f"V3-LTE trajectory has no auditable seed: {record.trajectory_id}"
            )
        return int(match.group(1))

    planned_pairs = [
        {
            "pair_id": record.sample_id,
            "prompt_id": record.prompt_id,
            "caption": prompts[record.prompt_id]["caption"],
            "seed": seed_of(record),
        }
        for record in selected
    ]
    plan = {
        "schema_version": 1,
        "stage": "pitch3_lte_development_guidance",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "production_authorization": False,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "model_artifact_kind": model_artifact_kind,
        "model_artifact_sha256": sha256_file(model_artifact_path),
        "checkpoint_sha256": (
            sha256_file(model_artifact_path) if model_artifact_kind == "checkpoint" else None
        ),
        "ensemble_manifest_sha256": (
            sha256_file(model_artifact_path)
            if model_artifact_kind == "equal_weight_ensemble"
            else None
        ),
        "ace_config_sha256": sha256_file(ace_config_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest_path),
        "authorization_scope": "development_only",
        "selection": "complete_64_prompt_x_4_seed_final_generation_cohort",
        "planned_pairs": planned_pairs,
        "guidance_step": 4,
        "training_radius_ratio": 0.05,
        "maximum_update_ratio": 0.025,
        "backtracking_attempts": 1,
        "sampler_contract": "one_step4_hook_then_complete_steps5_to8",
        "retain_audio_for_quality": retain_audio,
    }
    plan_path = output_dir / "pitch3_lte_guidance_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise LTSNContractError("V3-LTE guidance plan changed; use a new output directory")
    if not plan_path.is_file():
        write_json_atomic(plan_path, plan)
    development_plan_path = output_dir / "development_generation_plan.json"
    if (
        development_plan_path.is_file()
        and json.loads(development_plan_path.read_text(encoding="utf-8")) != plan
    ):
        raise LTSNContractError("V3-LTE development generation plan changed")
    if not development_plan_path.is_file():
        write_json_atomic(development_plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for record in selected:
        pair_id = record.sample_id
        baseline_id = f"{pair_id}__baseline"
        guided_id = f"{pair_id}__guided"
        source = source_final[record.trajectory_id]
        baseline_latent = (source_manifest_path.parent / source["latent_path"]).resolve()
        if not baseline_latent.is_file() or sha256_file(baseline_latent) != source["latent_sha256"]:
            raise LTSNContractError("V3-LTE final baseline latent is missing or mismatched")
        baseline_np = np.load(baseline_latent, allow_pickle=False).astype(np.float32, copy=False)
        guided_latent = output_dir / "latents" / f"{guided_id}.npy"
        baseline_audio = output_dir / "audio" / f"{baseline_id}.wav"
        guided_audio = output_dir / "audio" / f"{guided_id}.wav"
        if not baseline_audio.is_file():
            adapter.decode_latent_to_audio(baseline_np, baseline_audio)
        receipt_path = output_dir / "receipts" / f"{guided_id}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("plan_sha256") != plan_sha256:
                raise LTSNContractError("V3-LTE guided receipt belongs to another plan")
            if (
                not guided_latent.is_file()
                or sha256_file(guided_latent) != receipt.get("guided_latent_sha256")
                or not guided_audio.is_file()
                or sha256_file(guided_audio) != receipt.get("guided_audio_sha256")
            ):
                raise LTSNContractError("V3-LTE guided receipt artifact mismatch")
        else:
            prompt_hidden, prompt_mask = _prompt_tensors(record.prompt_embedding_path, device)
            corrector = Pitch3LTECorrector(
                model,
                prompt_hidden,
                prompt_mask,
                metadata,
                Pitch3LTEGuidanceConfig(enabled=True, authorization_scope="development_only"),
            )
            adapter.set_topology_corrector(corrector)
            prompt = prompts[record.prompt_id]
            result = adapter.generate(
                GenerationRequest(
                    prompt=prompt["caption"],
                    seed=seed_of(record),
                    duration_seconds=180.0,
                    output_dir=output_dir / "generator_output" / pair_id / "guided",
                    inference_steps=8,
                    bpm=int(prompt["bpm"]) if prompt.get("bpm", "").strip() else None,
                    keyscale=prompt.get("keyscale", ""),
                    timesignature=prompt.get("timesignature", ""),
                )
            )
            if result.seed != seed_of(record):
                raise LTSNContractError("ACE returned a different V3-LTE development seed")
            telemetry = corrector.drain_telemetry()
            if len(telemetry) != 1 or telemetry[0].get("step_number") != 4:
                raise LTSNContractError("V3-LTE must invoke exactly one Step-4 correction")
            guided_np = _latent_array(result.final_latent)
            _save_npy_atomic(guided_latent, guided_np)
            adapter.decode_latent_to_audio(guided_np, guided_audio)
            generated_audio = result.audio_path.resolve()
            if generated_audio != guided_audio.resolve() and generated_audio.is_file():
                generated_audio.unlink()
            receipt = {
                "schema_version": 1,
                "pair_id": pair_id,
                "guided_sample_id": guided_id,
                "plan_sha256": plan_sha256,
                "guided_latent_sha256": sha256_file(guided_latent),
                "guided_audio_sha256": sha256_file(guided_audio),
                "topology_correction_telemetry": telemetry,
            }
            write_json_atomic(receipt_path, receipt)
        telemetry = receipt["topology_correction_telemetry"]
        diagnostic = telemetry[0]
        coordinates = np.asarray(json.loads(source["coordinates_json"]), dtype=float)
        lower = np.asarray(contract.target_lower, dtype=float)
        upper = np.asarray(contract.target_upper, dtype=float)
        weights = np.asarray(contract.distance_weights, dtype=float)
        baseline_band = float(
            (
                (np.maximum(lower - coordinates, 0.0) ** 2)
                + (np.maximum(coordinates - upper, 0.0) ** 2)
            )
            @ weights
        )
        row = {
            "pair_id": pair_id,
            "prompt_id": record.prompt_id,
            "prompt_family": record.prompt_family,
            "seed": seed_of(record),
            "authorization_scope": "development_only",
            "baseline_candidate_id": baseline_id,
            "guided_candidate_id": guided_id,
            "baseline_sample_id": record.sample_id,
            "guided_sample_id": guided_id,
            "baseline_latent_path": _relative(baseline_latent, output_dir),
            "baseline_latent_sha256": source["latent_sha256"],
            "guided_latent_path": _relative(guided_latent, output_dir),
            "guided_latent_sha256": sha256_file(guided_latent),
            "baseline_audio_path": _relative(baseline_audio, output_dir),
            "baseline_audio_sha256": sha256_file(baseline_audio),
            "guided_audio_path": _relative(guided_audio, output_dir),
            "guided_audio_sha256": sha256_file(guided_audio),
            "baseline_exact_band": baseline_band,
            "guided_exact_band": "",
            "exact_band_improved": "",
            "applied": bool(diagnostic["applied"][0]),
            "energy_before": float(diagnostic["energy_before"][0]),
            "energy_after": float(diagnostic["energy_after"][0]),
            "gradient_rms": float(diagnostic["gradient_rms"][0]),
            "clean_update_rms": float(diagnostic["clean_update_rms"][0]),
            "backtracked": bool(diagnostic["backtracked"][0]),
            "no_op_reason_code": int(diagnostic["no_op_reason_code"][0]),
            "plan_sha256": plan_sha256,
        }
        rows.append(row)
        trajectory_rows.append(
            {
                "sample_id": guided_id,
                "prompt_id": record.prompt_id,
                "trajectory_id": guided_id,
                "split": "development",
                "model_family": "acestep-v15-xl-turbo",
                "step_number": 8,
                "timestep": 0.0,
                "latent_path": _relative(guided_latent, output_dir),
                "latent_sha256": sha256_file(guided_latent),
                "audio_path": _relative(guided_audio, output_dir),
                "audio_sha256": sha256_file(guided_audio),
                "is_final": "true",
                "ace_model_sha256": record.ace_model_sha256,
                "vae_sha256": record.vae_sha256,
                "local_anchor_sample_id": record.sample_id,
            }
        )
    trajectory_path = output_dir / "pitch3_lte_guided_trajectories.csv"
    write_csv_atomic(trajectory_path, trajectory_rows)
    descriptor_path = output_dir / "pitch3_lte_guided_exact_descriptors.csv"
    build_exact_snapshot_descriptors(
        project_root=root,
        trajectory_manifest=trajectory_path,
        work_dir=output_dir / "exact_work",
        output_path=descriptor_path,
        workers=workers,
        batch_size=exact_batch_size,
        materialize_mode=materialize_mode,
        cleanup_batches=True,
        resume=True,
    )
    descriptors = {row["sample_id"]: row for row in _read_csv(descriptor_path)}
    scorer = ExactPitch3Scorer(contract)
    for row in rows:
        descriptor = descriptors.get(str(row["guided_sample_id"]))
        if descriptor is None:
            raise LTSNContractError("V3-LTE guided exact descriptor is missing")
        score = scorer.score(json.loads(descriptor["pitch_descriptors_json"]))
        guided_band = float(score.target_band_loss[0])
        row["guided_exact_band"] = guided_band
        row["exact_band_improved"] = guided_band < float(row["baseline_exact_band"])
    pairs_path = output_dir / "pitch3_lte_guided_pairs.csv"
    write_csv_atomic(pairs_path, rows)
    write_csv_atomic(output_dir / "development_generation_manifest.csv", rows)
    quality_template_path = output_dir / "pitch3_lte_quality_template.csv"
    write_csv_atomic(
        quality_template_path,
        [
            {
                "pair_id": row["pair_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "baseline_candidate_id": row["baseline_candidate_id"],
                "guided_candidate_id": row["guided_candidate_id"],
                "quality_baseline": "",
                "quality_guided": "",
            }
            for row in rows
        ],
    )
    applied = [row for row in rows if row["applied"]]
    improvement_rate = sum(bool(row["exact_band_improved"]) for row in rows) / len(rows)
    if not retain_audio:
        for row in rows:
            (output_dir / str(row["baseline_audio_path"])).unlink(missing_ok=True)
            (output_dir / str(row["guided_audio_path"])).unlink(missing_ok=True)
    summary = {
        "schema_version": 1,
        "stage": "pitch3_lte_development_guidance",
        "status": "passed" if applied and improvement_rate > 0.50 else "failed",
        "pairs": len(rows),
        "applied_pairs": len(applied),
        "backtracked_pairs": sum(bool(row["backtracked"]) for row in rows),
        "exact_band_improved_pairs": sum(bool(row["exact_band_improved"]) for row in rows),
        "exact_band_improvement_rate": improvement_rate,
        "gate_threshold": ">0.50",
        "gate_passed": improvement_rate > 0.50,
        "audio_retained_for_quality": retain_audio,
        "quality_gate_pending": retain_audio,
        "quality_template": str(quality_template_path),
        "quality_template_sha256": sha256_file(quality_template_path),
        "guided_pairs": str(pairs_path),
        "guided_pairs_sha256": sha256_file(pairs_path),
        "model_artifact_kind": model_artifact_kind,
        "model_artifact_sha256": sha256_file(model_artifact_path),
        "checkpoint_sha256": (
            sha256_file(model_artifact_path) if model_artifact_kind == "checkpoint" else None
        ),
        "ensemble_manifest_sha256": (
            sha256_file(model_artifact_path)
            if model_artifact_kind == "equal_weight_ensemble"
            else None
        ),
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    write_json_atomic(output_dir / "pitch3_lte_guidance_summary.json", summary)
    return summary


def finalize_pitch3_lte_quality(
    *,
    metric_table_path: Path,
    output_path: Path,
    bootstrap_resamples: int = 2000,
    seed: int = 20260920,
) -> dict[str, Any]:
    """Freeze cluster-bootstrap non-inferiority for the 256 guided pairs."""

    rows = _read_csv(metric_table_path)
    prompt_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        prompt_counts[row["prompt_id"]] += 1
    if len(rows) != 256 or len(prompt_counts) != 64 or set(prompt_counts.values()) != {4}:
        raise LTSNContractError("V3-LTE quality evidence requires 64 prompts x 4 seeds")
    if bootstrap_resamples < 1000:
        raise ValueError("V3-LTE quality bootstrap requires at least 1000 resamples")
    prompt_ids = np.asarray([row["prompt_id"] for row in rows], dtype=object)
    prompts = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)

    def criterion(name: str, required: bool) -> dict[str, Any]:
        baseline_key = f"{name}_baseline"
        guided_key = f"{name}_guided"
        raw = [(row.get(baseline_key, "").strip(), row.get(guided_key, "").strip()) for row in rows]
        complete = all(left and right for left, right in raw)
        if not complete:
            return {
                "metric": name,
                "evidence_complete": False,
                "required": required,
                "passed": False if required else None,
            }
        baseline = np.asarray([float(left) for left, _ in raw], dtype=float)
        guided = np.asarray([float(right) for _, right in raw], dtype=float)
        if not np.isfinite(baseline).all() or not np.isfinite(guided).all():
            raise LTSNContractError(f"V3-LTE {name} evidence contains NaN or Inf")
        differences = guided - baseline
        bootstrap = np.empty(bootstrap_resamples, dtype=float)
        for index in range(bootstrap_resamples):
            chosen = rng.choice(prompts, len(prompts), replace=True)
            values = np.concatenate([differences[prompt_ids == prompt] for prompt in chosen])
            bootstrap[index] = float(values.mean())
        interval = np.quantile(bootstrap, [0.025, 0.975])
        return {
            "metric": name,
            "evidence_complete": True,
            "required": required,
            "baseline_mean": float(baseline.mean()),
            "guided_mean": float(guided.mean()),
            "paired_mean_difference": float(differences.mean()),
            "cluster_bootstrap_95_interval": [float(interval[0]), float(interval[1])],
            "margin": 0.0,
            "passed": bool(interval[0] >= 0.0),
        }

    criteria = {
        "quality": criterion("quality", True),
        "prompt": criterion("prompt", True),
        "diversity": criterion("diversity", True),
    }
    report = {
        "schema_version": 1,
        "stage": "pitch3_lte_quality_noninferiority",
        "status": "passed"
        if all(item["passed"] is True for item in criteria.values())
        else "failed",
        "criteria": criteria,
        "quality_noninferior": criteria["quality"]["passed"],
        "prompt_noninferior": criteria["prompt"]["passed"],
        "diversity_noninferior": criteria["diversity"]["passed"],
        "metric_table_sha256": sha256_file(metric_table_path),
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "qualification_eligible": False,
        "production_authorization": False,
    }
    write_json_atomic(output_path, report)
    return report


def screen_pitch3_lte_development(
    *,
    fingerprint_path: Path,
    dataset_manifest: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    checkpoint_sha256: str | None = None,
    ensemble_manifest_path: Path | None = None,
    guidance_summary_path: Path | None = None,
    quality_report_path: Path | None = None,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run the pre-registered V3-LTE held-out go/no-go screen."""

    contract = load_pitch3_contract(fingerprint_path)
    records = [
        record
        for record in _read_examples(dataset_manifest, contract.artifact_sha256)
        if record.split == "development"
    ]
    sampler = PromptBatchSampler(records, seed=0, shuffle=False)
    loader = DataLoader(
        Pitch3LTEDataset(records),
        batch_sampler=sampler,
        collate_fn=collate_pitch3_lte,
        num_workers=0,
    )
    device = torch.device(device_name)
    model, metadata, model_artifact_path, model_artifact_kind = _load_lte_model_artifact(
        checkpoint_path=checkpoint_path,
        ensemble_manifest_path=ensemble_manifest_path,
        device=device,
        expected_sha256=checkpoint_sha256,
    )
    if metadata["fingerprint_json_sha256"] != contract.artifact_sha256:
        raise LTSNContractError("V3-LTE development checkpoint fingerprint mismatch")
    rows = _prediction_rows(model, loader, device, use_bf16=False)
    metrics = pitch3_lte_metrics(rows)
    guidance = (
        json.loads(guidance_summary_path.read_text(encoding="utf-8"))
        if guidance_summary_path is not None
        else None
    )
    quality = (
        json.loads(quality_report_path.read_text(encoding="utf-8"))
        if quality_report_path is not None
        else None
    )
    guidance_passed = bool(guidance and guidance.get("gate_passed") is True)
    quality_passed = bool(
        quality
        and quality.get("quality_noninferior") is True
        and quality.get("prompt_noninferior", True) is True
        and quality.get("diversity_noninferior", True) is True
    )
    gates = {
        **metrics["gates"],
        "step4_exact_band_improvement_rate": guidance_passed,
        "audio_quality_noninferiority": quality_passed,
    }
    non_quality_passed = all(
        value for name, value in gates.items() if name != "audio_quality_noninferiority"
    )
    status = (
        "passed"
        if all(gates.values())
        else ("pending_quality" if quality is None and non_quality_passed else "failed")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "pitch3_lte_development_predictions.csv"
    write_csv_atomic(predictions_path, rows)
    report = {
        "schema_version": 1,
        "stage": "pitch3_lte_development_screen",
        "status": status,
        "calibration_eligible": status == "passed",
        "gates": gates,
        "metrics": metrics,
        "guidance_summary": guidance,
        "quality_report": quality,
        "model_artifact_kind": model_artifact_kind,
        "model_artifact_sha256": sha256_file(model_artifact_path),
        "checkpoint_sha256": (
            sha256_file(model_artifact_path) if model_artifact_kind == "checkpoint" else None
        ),
        "ensemble_manifest_sha256": (
            sha256_file(model_artifact_path)
            if model_artifact_kind == "equal_weight_ensemble"
            else None
        ),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "predictions_sha256": sha256_file(predictions_path),
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    write_json_atomic(output_dir / "pitch3_lte_development_screen.json", report)
    return report
