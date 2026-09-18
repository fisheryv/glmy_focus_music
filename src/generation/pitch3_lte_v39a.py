"""Matched G0/L0 versus high-resolution transition training, without dev selection."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_json_atomic
from .pitch3_contract import load_pitch3_contract
from .pitch3_lte import PromptConditionedTopologyEnergy
from .pitch3_lte_data import LTE_MODEL_FAMILY, validate_pitch3_lte_dataset_preflight
from .pitch3_lte_training import (
    _prediction_rows,
    _read_examples,
    _seed_everything,
    _set_training_stage_trainable,
    _to_device,
    load_pitch3_lte_config,
    pitch3_lte_metrics,
)
from .pitch3_lte_v38b import (
    export_predictions,
    forward_losses,
    gradient_snapshot,
    initialize_statistics,
    loader,
)
from .pitch3_lte_v38b_protocol import describe_predictions, read_csv, read_json, verify_file
from .pitch3_lte_v39a_protocol import (
    REVISION,
    SELECTION,
    VARIANTS,
    content_hash,
    grouped_split,
    validate_config,
    validate_protocol,
    verify_manifest,
)


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_matched_models(config, seed):
    """Explicit copy, independent of layer registration/RNG consumption order."""
    _seed_everything(seed)
    baseline = PromptConditionedTopologyEnergy(replace(config, transition_branch=False))
    candidate = PromptConditionedTopologyEnergy(replace(config, transition_branch=True))
    shared = baseline.state_dict()
    missing, unexpected = candidate.load_state_dict(shared, strict=False)
    expected_missing = {k for k in candidate.state_dict() if k.startswith("transition_branch.")}
    if set(missing) != expected_missing or unexpected:
        raise LTSNContractError("New architecture changed the shared parameter contract")
    for name, tensor in shared.items():
        if not torch.equal(tensor, candidate.state_dict()[name]):
            raise LTSNContractError(f"Initial shared tensor differs: {name}")
    added = sum(p.numel() for p in candidate.transition_branch.parameters())
    if added > 150_000:
        raise LTSNContractError("Transition branch exceeds the declared parameter budget")
    return (
        baseline,
        candidate,
        {
            "shared_state_sha256": state_hash(shared),
            "initialization_policy": "explicit_shared_state_copy_then_zero_transition_output",
            "seed": seed,
            "baseline_parameters": sum(p.numel() for p in baseline.parameters()),
            "transition_parameters_added": added,
            "lags_latent_frames": list(candidate.transition_branch.lags),
            "transition_summary_features": 260,
            "new_branch_dropout": 0.0,
        },
    )


def check_initial_identity(baseline, candidate, sample, training, stats, weights):
    outputs, losses = [], []
    with torch.no_grad():
        for model in (baseline, candidate):
            model.eval()
            state = model.encode_latent(sample["latent"], sample["attention_mask"])
            prompt = model.encode_prompt(sample["text_hidden"], sample["text_mask"])
            parts = model.potential_components_from_states(state, prompt, state.detach())
            outputs.append((state, parts.global_energy, parts.coordinates))
            losses.append(forward_losses(model, sample, "global", training, stats))
    for left, right in zip(*outputs, strict=True):
        if not torch.equal(left, right):
            raise LTSNContractError("Matched models do not have identical initial outputs")
    for name in weights:
        if not torch.equal(losses[0][name], losses[1][name]):
            raise LTSNContractError(f"Initial matched loss differs: {name}")
    return {
        "outputs_and_losses_equal": True,
        "sample_ids": sample["sample_id"],
        "initial_raw_losses": {name: float(losses[0][name]) for name in weights},
        "normalizer_source": "identical_baseline_initialization_fit_records_only",
    }


def prepare_data(fingerprint_path, dataset_manifest, config, fold, diagnostic):
    contract = load_pitch3_contract(fingerprint_path)
    for key, target in (
        ("coordinate_lower", contract.target_lower),
        ("coordinate_upper", contract.target_upper),
        ("coordinate_center", contract.target_center),
        ("coordinate_distance_weights", contract.distance_weights),
    ):
        if not np.allclose(getattr(config, key), target, rtol=0, atol=1e-12):
            raise LTSNContractError(f"Model {key} differs from frozen teacher")
    summary_path = dataset_manifest.parent / "pitch3_lte_dataset_summary.json"
    summary = (
        read_json(summary_path)
        if summary_path.is_file()
        else validate_pitch3_lte_dataset_preflight(dataset_manifest)
    )
    dataset_hash = sha256_file(dataset_manifest)
    if (
        summary.get("local_preflight_passed") is not True
        or summary.get("dataset_manifest_sha256") != dataset_hash
    ):
        raise LTSNContractError("Dataset preflight missing or detached")
    verify_file(
        dataset_manifest.parent / "pitch3_lte_dataset_plan.json", summary["dataset_plan_sha256"]
    )
    records = _read_examples(dataset_manifest, contract.artifact_sha256)
    for record in records:
        band = sum(
            w * (max(lo - q, 0) ** 2 + max(q - hi, 0) ** 2)
            for q, lo, hi, w in zip(
                record.coordinates,
                contract.target_lower,
                contract.target_upper,
                contract.distance_weights,
                strict=True,
            )
        )
        if not math.isclose(band, record.exact_band, rel_tol=1e-9, abs_tol=1e-9):
            raise LTSNContractError(f"Detached coordinate teacher: {record.sample_id}")
    for split, expected in (("train", 320), ("development", 64)):
        if len({r.prompt_id for r in records if r.split == split}) != expected:
            raise LTSNContractError("Expected original 320/64 prompt split")
    rows = read_csv(dataset_manifest)
    fit_rows, outer_rows = grouped_split(rows, fold, diagnostic)
    by_id = {r.sample_id: r for r in records}
    fit = [by_id[r["sample_id"]] for r in fit_rows]
    outer = [by_id[r["sample_id"]] for r in outer_rows]
    return contract, summary, records, rows, fit, outer


def parameter_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("transition_branch."):
            group = (
                "transition_output"
                if name.startswith("transition_branch.output.")
                else "transition_features"
            )
        elif name.startswith(
            ("latent_energy_head.", "coordinate_head.", "interaction_energy_head.")
        ):
            group = "readouts"
        elif name.startswith(("text_", "joint.")):
            group = "prompt"
        else:
            group = "latent_backbone"
        groups.setdefault(group, []).append(parameter)
    return groups


def gradient_norms(groups):
    # Avoid CUDA dot/Sdot; diagnostic reductions have no effect on training.
    return {
        name: float(
            sum(
                (p.grad.detach().square().sum() for p in params if p.grad is not None),
                params[0].new_zeros(()),
            ).sqrt()
        )
        for name, params in groups.items()
    }


def fit_diagnostics(model, fit, device, training, stats):
    model.eval()
    rows = _prediction_rows(model, loader(fit, training.seed, "local"), device, use_bf16=False)
    result = describe_predictions(
        rows,
        {r.sample_id: r for r in fit},
        stats["energy_stratification"]["thresholds"],
        stats["local_training_scales"]["local_derivative_scale"],
        stats["coordinate_auxiliary"]["coordinate_widths"],
    )
    return {"metrics": pitch3_lte_metrics(rows), "diagnostics": result}


def train_v39a(
    *,
    fingerprint_path: Path,
    dataset_manifest: Path,
    config_path: Path,
    output_dir: Path,
    variant="transition",
    cv_fold=None,
    seed=None,
    device_name="cpu",
    diagnostic=False,
):
    if variant not in VARIANTS:
        raise ValueError("Unknown V3.9-A variant")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {output_dir}")
    config_payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = validate_config(config_payload)
    config, training = load_pitch3_lte_config(config_path)
    training = replace(training, seed=training.seed if seed is None else seed)
    if type(training.seed) is not int or training.seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    training.validate()
    dataset_manifest = dataset_manifest.resolve()
    contract, summary, records, rows, fit, outer = prepare_data(
        fingerprint_path, dataset_manifest, config, cv_fold, diagnostic
    )
    mode = "diagnostic" if diagnostic else "cv" if cv_fold is not None else "full"
    epochs = experiment["diagnostic_epochs" if diagnostic else "global_epochs"]
    experiment_contract = {
        **experiment,
        "variant": variant,
        "local_variant": "l0",
        "ordinal_weight": 0.0,
        "shape_weight": 0.0,
        "ensemble_readout": "mean_member_energies",
        "ranking_objective": training.ranking_objective,
        "initialization_policy": "explicit_shared_state_copy_then_zero_transition_output",
    }
    protocol = {
        "run_mode": mode,
        "cv_fold": cv_fold,
        "seed": training.seed,
        "validation_scope": {
            "cv": "train_family_cv",
            "full": "development",
            "diagnostic": "train_fit_diagnostic",
        }[mode],
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "development_used": False,
        "checkpoint_selection": SELECTION,
        "outer_evaluation_count": int(mode == "cv"),
        "train_sample_ids": sorted(r.sample_id for r in fit),
        "selection_sample_ids": sorted(r.sample_id for r in outer),
        "train_families": sorted({r.prompt_family for r in fit}),
        "selection_families": sorted({r.prompt_family for r in outer}),
        "experiment_contract": experiment_contract,
        "production_authorization": False,
    }
    validate_protocol(protocol, rows)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
    baseline, candidate, initial = build_matched_models(config, training.seed)
    baseline.to(device)
    candidate.to(device)
    stats = initialize_statistics(baseline, fit, training, device, "g0")
    weights = {
        "value": 1.0,
        "prompt_rank": 1.0,
        "coordinate": training.coordinate_loss_weight,
        "family_stratified_rank": 1.0,
    }
    sample = _to_device(next(iter(loader(fit, training.seed, "global"))), device)
    initial.update(check_initial_identity(baseline, candidate, sample, training, stats, weights))
    model = candidate if variant == "transition" else baseline
    del baseline, candidate
    config = model.config
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "pitch3_lte_run_protocol.json"
    write_json_atomic(protocol_path, protocol)
    stats["run_protocol_sha256"] = sha256_file(protocol_path)
    write_json_atomic(output_dir / "pitch3_lte_initialization.json", initial)
    stats_path = output_dir / "pitch3_lte_training_statistics.json"
    write_json_atomic(stats_path, stats)
    effective_path = output_dir / "pitch3_lte_effective_config.json"
    write_json_atomic(
        effective_path,
        {"model": asdict(config), "training": asdict(training), "v39a": experiment_contract},
    )
    write_json_atomic(
        output_dir / "pitch3_lte_gradient_initial.json",
        gradient_snapshot(model, sample, "global", training, stats, weights),
    )
    _set_training_stage_trainable(model, training, "global")
    frozen_local = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith("local_energy_head.")
    }
    groups = parameter_groups(model)
    parameters = [p for params in groups.values() for p in params]
    optimizer = torch.optim.AdamW(
        parameters, lr=training.learning_rate, weight_decay=training.weight_decay
    )
    train_loader = loader(fit, training.seed, "global", shuffle=True, workers=training.num_workers)
    _seed_everything(training.seed)
    rng_devices = [device.index] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    with torch.random.fork_rng(devices=rng_devices):
        epoch_zero = fit_diagnostics(model, fit, device, training, stats)
    write_json_atomic(output_dir / "pitch3_lte_fit_epoch_zero.json", epoch_zero)
    for epoch in range(1, epochs + 1):
        _set_training_stage_trainable(model, training, "global")
        train_loader.batch_sampler.set_epoch(epoch)
        epoch_start = {
            name: [p.detach().clone() for p in params] for name, params in groups.items()
        }
        sums = dict.fromkeys(weights, 0.0)
        norms, block_norms = [], []
        started = time.perf_counter()
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses = forward_losses(model, batch, "global", training, stats)
            contributions = {
                k: w * losses[k] / stats["loss_normalizers"][k] for k, w in weights.items()
            }
            total = sum(contributions.values())
            if not bool(torch.isfinite(total)):
                raise LTSNContractError("Non-finite V3.9-A loss")
            total.backward()
            before = gradient_norms(groups)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, training.gradient_clip_norm, error_if_nonfinite=True
            )
            after = gradient_norms(groups)
            optimizer.step()
            norms.append(float(norm))
            block_norms.append(
                {k: {"before_clip": before[k], "after_clip": after[k]} for k in groups}
            )
            for key, value in contributions.items():
                sums[key] += float(value.detach())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        updates = {}
        with torch.no_grad():
            for name, params in groups.items():
                old = epoch_start[name]
                delta = float(
                    sum((p - q).square().sum() for p, q in zip(params, old, strict=True)).sqrt()
                )
                scale = float(sum(q.square().sum() for q in old).sqrt())
                updates[name] = {
                    "epoch_net_update_l2": delta,
                    "relative_to_epoch_start": delta / scale if scale else None,
                }
        del epoch_start
        # Evaluation loaders also draw a DataLoader base seed. Restore RNG so
        # diagnostics cannot change subsequent training dropout for either arm.
        with torch.random.fork_rng(devices=rng_devices):
            fit_report = fit_diagnostics(model, fit, device, training, stats)
        item = {
            "epoch": epoch,
            "optimizer_steps": len(norms),
            "training_stage": "global",
            "normalized_training_contributions": {k: v / len(norms) for k, v in sums.items()},
            "gradient_norm_mean": float(np.mean(norms)),
            "gradient_norm_max": max(norms),
            "clipping_fraction": float(np.mean(np.array(norms) > training.gradient_clip_norm)),
            "block_gradient_norms_mean": {
                k: {
                    phase: float(np.mean([x[k][phase] for x in block_norms]))
                    for phase in ("before_clip", "after_clip")
                }
                for k in groups
            },
            "block_parameter_updates": updates,
            "training_seconds": train_seconds,
            "fit_evaluation": fit_report,
            "selection_evaluated": False,
        }
        history.append(item)
        write_json_atomic(output_dir / "pitch3_lte_training_history.json", {"history": history})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "fit_min_family_rho": fit_report["metrics"]["minimum_prompt_family_spearman"],
                    "clipping_fraction": item["clipping_fraction"],
                    "training_seconds": train_seconds,
                }
            ),
            flush=True,
        )
        if (
            epoch == 1
            and training.maximum_initial_component_contribution > 0
            and any(
                v > training.maximum_initial_component_contribution
                for v in item["normalized_training_contributions"].values()
            )
        ):
            raise LTSNContractError("Initial loss contribution exceeds the matched limit")
    for name, before in frozen_local.items():
        if not torch.equal(before, dict(model.named_parameters())[name].detach()):
            raise LTSNContractError("V3.9-A global fitting changed local parameters")
    write_json_atomic(
        output_dir / "pitch3_lte_gradient_final.json",
        gradient_snapshot(model, sample, "global", training, stats, weights),
    )
    train_path, train_diag = export_predictions(
        model, fit, records, device, training, stats, output_dir, "train"
    )
    result_fields = {
        "train_predictions_sha256": sha256_file(train_path),
        "train_metrics": train_diag["metrics"],
    }
    if outer:
        path, diag = export_predictions(
            model, outer, records, device, training, stats, output_dir, "selection"
        )
        result_fields.update(
            selection_predictions_sha256=sha256_file(path), best_development_metrics=diag["metrics"]
        )
    if diagnostic:
        result = {
            "stage": "pitch3_lte_v39a_fit_diagnostic",
            "completed": True,
            "diagnostic_only": True,
            "development_used": False,
            "outer_evaluation_count": 0,
            "production_authorization": False,
            "epochs_completed": epochs,
            "variant": variant,
            "cv_fold": cv_fold,
            "fit_families": protocol["train_families"],
            "initialization": initial,
            "final_fit_metrics": train_diag["metrics"],
            "artifacts_sha256": {
                p.name: sha256_file(p) for p in output_dir.iterdir() if p.is_file()
            },
        }
        write_json_atomic(output_dir / "pitch3_lte_diagnostic_complete.json", result)
        return result
    metadata = {
        **{k: v for k, v in protocol.items() if not k.endswith("sample_ids")},
        **result_fields,
        "schema_version": 1,
        "model_family": LTE_MODEL_FAMILY,
        "architecture_revision": REVISION,
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_plan_sha256": summary["dataset_plan_sha256"],
        "source_training_config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "pitch3_lte_v39a.py",
                "pitch3_lte_v39a_protocol.py",
                "pitch3_lte_transition.py",
                "pitch3_lte.py",
                "pitch3_lte_v38b.py",
                "pitch3_lte_v38b_protocol.py",
                "pitch3_lte_training.py",
                "pitch3_lte_metrics.py",
                "path_homology_surrogate.py",
            )
        },
        "training_config_sha256": sha256_file(effective_path),
        "effective_training_config": str(effective_path.resolve()),
        "run_protocol_sha256": sha256_file(protocol_path),
        "training_statistics_sha256": sha256_file(stats_path),
        "shared_initial_state_sha256": initial["shared_state_sha256"],
        "initial_normalizers_sha256": content_hash(stats["loss_normalizers"]),
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "optimized_parameters": sum(p.numel() for p in parameters),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "diagnostic_artifacts_sha256": {
            p.name: sha256_file(p) for p in sorted(output_dir.glob("pitch3_lte_*.json"))
        },
        "precision": "fp32",
        "guidance_steps": [4],
        "training_radius_ratio": 0.05,
        "maximum_guidance_update_ratio": 0.025,
        "energy_target": "log1p(exact_pitch3_target_band_loss)",
        "local_residual_mode": "zero_global_only",
        "loss_components": list(weights),
        "loss_component_weights": weights,
        "loss_normalizers": stats["loss_normalizers"],
        "energy_stratification": stats["energy_stratification"],
        "local_training_scales": stats["local_training_scales"],
        "best_epoch": epochs,
        "epochs_completed": epochs,
        "checkpoint_parameter_source": "online",
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    checkpoint = output_dir / f"pitch3_lte_seed_{training.seed}.pt"
    temporary = checkpoint.with_suffix(".pt.part")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "training_config": asdict(training),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(checkpoint)
    manifest = {
        **metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    manifest_path = output_dir / "pitch3_lte_manifest.json"
    write_json_atomic(manifest_path, manifest)
    verify_manifest(manifest_path, rows)
    return manifest
