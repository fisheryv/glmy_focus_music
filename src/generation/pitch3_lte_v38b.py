"""Fixed-epoch global/ordinal and anchored derivative ablations for V3.8-B.

Outer train-family folds are evaluated once after fitting. Full-train runs never
evaluate development; the existing screen is a separate downstream operation.
"""

from __future__ import annotations

import json
import math
import platform
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import write_csv_atomic, write_json_atomic
from .pitch3_contract import load_pitch3_contract
from .pitch3_lte import PromptConditionedTopologyEnergy
from .pitch3_lte_data import LTE_MODEL_FAMILY, validate_pitch3_lte_dataset_preflight
from .pitch3_lte_training import (
    Pitch3LTEDataset,
    PromptBatchSampler,
    _direct_energy_coordinate_contract,
    _energy_strata,
    _local_training_scales,
    _pair_indices,
    _prediction_rows,
    _raw_loss_kwargs,
    _read_examples,
    _seed_everything,
    _set_training_stage_trainable,
    _to_device,
    _train_family_cv_split,
    collate_pitch3_lte,
    load_pitch3_lte_checkpoint,
    load_pitch3_lte_config,
    pitch3_lte_metrics,
    pitch3_lte_raw_losses,
)
from .pitch3_lte_v38b_protocol import (
    GLOBAL_VARIANTS,
    LOCAL_VARIANTS,
    REVISION,
    SELECTION,
    describe_predictions,
    local_file,
    read_csv,
    read_json,
    validate_protocol,
    verify_file,
    verify_manifest,
)


def ordinal_loss(logits, batch, thresholds):
    if logits is None or logits.shape != (len(batch["source_kind"]), 3):
        raise LTSNContractError("V3.8-B requires three cumulative ordinal logits")
    if len(thresholds) != 3 or thresholds[0] != 0 or not 0 < thresholds[1] < thresholds[2]:
        raise LTSNContractError("Ordinal thresholds must be zero and fit-positive tertiles")
    base = torch.tensor(
        [k == "base_step4_seed" for k in batch["source_kind"]], device=logits.device
    )
    if not bool(base.any()):
        return logits.sum() * 0
    targets = (batch["energy_target"][base, None] > logits.new_tensor(thresholds)).float()
    return F.binary_cross_entropy_with_logits(logits[base].float(), targets)


def derivative_shape_loss(energy, batch, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Derivative scale must be finite and positive")
    terms = []
    for minus, plus in _pair_indices(batch)[1]:
        eps = batch["epsilon"][minus]
        if eps <= 0 or not torch.isclose(eps, batch["epsilon"][plus], rtol=0, atol=1e-9):
            raise LTSNContractError("Shape loss requires a shared positive epsilon")
        exact = (batch["energy_target"][plus] - batch["energy_target"][minus]) / (2 * eps)
        if exact.abs() <= 1e-6:
            continue
        predicted = (energy[plus] - energy[minus]) / (2 * eps)
        terms.append(F.huber_loss(torch.asinh(predicted / scale), torch.asinh(exact / scale)))
    return torch.stack(terms).mean() if terms else energy.sum() * 0


def loader(records, seed, kind, *, shuffle=False, workers=0):
    full = kind == "global"
    sampler = PromptBatchSampler(
        records,
        seed,
        shuffle,
        groups_per_batch=16 if full else 2,
        pairing_mode="family_full_base_v34" if full else "family_round_robin_v33",
    )
    return DataLoader(
        Pitch3LTEDataset(records),
        batch_sampler=sampler,
        collate_fn=collate_pitch3_lte,
        num_workers=workers,
    )


def forward_losses(model, batch, stage, training, stats):
    latent = model.encode_latent(batch["latent"], batch["attention_mask"])
    prompt = model.encode_prompt(batch["text_hidden"], batch["text_mask"])
    anchor = (
        latent.detach()
        if stage == "global"
        else model.encode_latent(batch["anchor_latent"], batch["anchor_attention_mask"]).detach()
    )
    parts = model.potential_components_from_states(latent, prompt, anchor_latent_state=anchor)
    losses = pitch3_lte_raw_losses(
        parts.global_energy if stage == "global" else parts.energy,
        batch,
        predicted_coordinates=parts.coordinates,
        ranking_logits=parts.global_logit,
        **_raw_loss_kwargs(
            training,
            stats["local_training_scales"],
            stats["energy_stratification"],
            stats["coordinate_auxiliary"],
            include_structured=stage == "global",
        ),
    )
    if model.config.ordinal_auxiliary and stage == "global":
        losses["ordinal"] = ordinal_loss(
            parts.ordinal_logits, batch, stats["energy_stratification"]["thresholds"]
        )
    if stage == "local":
        losses["local_shape"] = derivative_shape_loss(
            parts.energy, batch, stats["local_training_scales"]["local_derivative_scale"]
        )
    return losses


def weights_for(training, experiment, global_variant, local_variant, stage):
    if stage == "local":
        weights = {"local_direction": 1.0, "local_flat": 0.25}
        if local_variant == "l2":
            weights["local_shape"] = experiment["shape_weight"]
        return weights
    weights = {"value": 1.0, "prompt_rank": 1.0, "coordinate": training.coordinate_loss_weight}
    weights["family_listwise" if global_variant == "g37" else "family_stratified_rank"] = 1.0
    if global_variant == "g1":
        weights["ordinal"] = experiment["ordinal_weight"]
    return weights


def gradient_snapshot(model, batch, stage, training, stats, weights):
    _set_training_stage_trainable(model, training, stage)
    model.eval()  # Reproducible diagnostic, no dropout or optimizer update.
    losses = forward_losses(model, batch, stage, training, stats)
    params = [p for p in model.parameters() if p.requires_grad]
    vectors = {}
    for name, weight in weights.items():
        grads = torch.autograd.grad(
            weight * losses[name] / stats["loss_normalizers"][name],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        vectors[name] = torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                for p, g in zip(params, grads, strict=True)
            ]
        )
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    cosine = {
        a: {
            b: float(torch.dot(va, vb) / (va.norm() * vb.norm()))
            if norms[a] > 0 and norms[b] > 0
            else None
            for b, vb in vectors.items()
        }
        for a, va in vectors.items()
    }
    return {
        "sample_ids": batch["sample_id"],
        "mode": "eval",
        "stage": stage,
        "weighted_normalized_gradient_norms": norms,
        "gradient_cosine": cosine,
    }


def initialize_statistics(model, train_records, training, device, global_variant):
    stats = {
        "source": "fit_records_only",
        "energy_stratification": _energy_strata(
            [r for r in train_records if r.source_kind == "base_step4_seed"], 3, 4
        ),
        "local_training_scales": _local_training_scales(train_records, 1e-6),
        "coordinate_auxiliary": _direct_energy_coordinate_contract(train_records, model.config),
        "global_loss_batching": "full_family_base_64",
    }
    names = {
        "global": [
            "value",
            "prompt_rank",
            "coordinate",
            "family_listwise" if global_variant == "g37" else "family_stratified_rank",
        ],
        "local": ["local_direction"],
    }
    normalizers = {"ordinal": 1.0, "local_shape": 1.0, "local_flat": 1.0}
    model.eval()
    with torch.inference_mode():
        for stage, keys in names.items():
            values = {key: [] for key in keys}
            for raw in loader(train_records, training.seed, stage):
                losses = forward_losses(model, _to_device(raw, device), stage, training, stats)
                for key in keys:
                    value = float(losses[key])
                    if math.isfinite(value) and value > 0:
                        values[key].append(value)
            for key, items in values.items():
                if not items:
                    raise LTSNContractError(f"No positive initial loss for {key}")
                normalizers[key] = float(np.median(items))
    stats["loss_normalizers"] = normalizers
    stats["loss_normalizer_policy"] = {
        key: "fixed_dimensionless_scale"
        if key in {"ordinal", "local_shape", "local_flat"}
        else "initial_fit_batch_median"
        for key in normalizers
    }
    return stats


def export_predictions(model, records, all_records, device, training, stats, output_dir, label):
    rows = _prediction_rows(model, loader(records, training.seed, "local"), device, use_bf16=False)
    path = output_dir / f"pitch3_lte_{label}_predictions.csv"
    write_csv_atomic(path, rows)
    diagnostics = describe_predictions(
        rows,
        {r.sample_id: r for r in all_records},
        stats["energy_stratification"]["thresholds"],
        stats["local_training_scales"]["local_derivative_scale"],
        stats["coordinate_auxiliary"]["coordinate_widths"],
    )
    diagnostics["metrics"] = pitch3_lte_metrics(rows)
    write_json_atomic(output_dir / f"pitch3_lte_{label}_diagnostics.json", diagnostics)
    return path, diagnostics


def train_v38b(
    *,
    fingerprint_path: Path,
    dataset_manifest: Path,
    config_path: Path,
    output_dir: Path,
    global_variant="g1",
    local_variant="l0",
    cv_fold=None,
    seed=None,
    device_name="cpu",
    global_manifest: Path | None = None,
):
    if global_variant not in GLOBAL_VARIANTS or local_variant not in LOCAL_VARIANTS:
        raise ValueError("Unknown V3.8-B ablation")
    if (local_variant != "l0") != (global_manifest is not None):
        raise ValueError("L1/L2 require the same frozen global manifest; L0 trains global only")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an experiment: {output_dir}")
    model_config, training = load_pitch3_lte_config(config_path)
    experiment = tomllib.loads(config_path.read_text(encoding="utf-8"))["v38b"]
    if not model_config.ordinal_auxiliary or model_config.potential_mode != "direct_anchored_v37":
        raise ValueError("V3.8-B requires an ordinal-capable direct anchored model")
    if set(experiment) != {
        "global_epochs",
        "local_epochs",
        "parameter_source",
        "ordinal_weight",
        "shape_weight",
        "ordinal_normalizer",
        "shape_normalizer",
    }:
        raise ValueError("Unknown or missing V3.8-B experiment setting")
    if any(
        type(experiment[k]) is not int or experiment[k] <= 0
        for k in ["global_epochs", "local_epochs"]
    ):
        raise ValueError("Fixed epoch counts must be positive integers")
    if experiment["parameter_source"] != "online" or any(
        experiment[k] != 1.0 for k in ["ordinal_normalizer", "shape_normalizer"]
    ):
        raise ValueError("V3.8-B uses fixed online checkpoints and dimensionless auxiliary scales")
    if any(
        not math.isfinite(experiment[k]) or experiment[k] <= 0
        for k in ["ordinal_weight", "shape_weight"]
    ):
        raise ValueError("Auxiliary weights must be finite and positive")
    training = replace(training, seed=training.seed if seed is None else seed)
    if training.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if training.use_bf16 or training.rank_min_delta != 1e-6 or training.local_flat_weight != 0.25:
        raise ValueError("V3.8-B retains fp32, frozen direction threshold and flat weight")
    if global_variant == "g37":
        training = replace(
            training,
            training_schedule="direct_energy_anchored_global_then_local_v37_dte",
            ranking_objective="energy_rank_legacy",
            family_stratified_rank_weight=0,
            family_listwise_weight=1,
        )
    training.validate()
    contract = load_pitch3_contract(fingerprint_path)
    for name, target in [
        ("coordinate_lower", contract.target_lower),
        ("coordinate_upper", contract.target_upper),
        ("coordinate_center", contract.target_center),
        ("coordinate_distance_weights", contract.distance_weights),
    ]:
        if not np.allclose(getattr(model_config, name), target, rtol=0, atol=1e-12):
            raise LTSNContractError(f"Model {name} differs from the frozen fingerprint")
    dataset_manifest = dataset_manifest.resolve()
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
        raise LTSNContractError("Dataset preflight is missing or detached")
    plan = dataset_manifest.parent / "pitch3_lte_dataset_plan.json"
    verify_file(plan, summary["dataset_plan_sha256"])
    records = _read_examples(dataset_manifest, contract.artifact_sha256)
    for r in records:
        band = sum(
            w * (max(lo - q, 0) ** 2 + max(q - hi, 0) ** 2)
            for q, lo, hi, w in zip(
                r.coordinates,
                contract.target_lower,
                contract.target_upper,
                contract.distance_weights,
                strict=True,
            )
        )
        if not math.isclose(band, r.exact_band, rel_tol=1e-9, abs_tol=1e-9):
            raise LTSNContractError(f"Detached exact coordinate label: {r.sample_id}")
    fit = [r for r in records if r.split == "train"]
    if (
        len({r.prompt_id for r in fit}) != 320
        or len({r.prompt_id for r in records if r.split == "development"}) != 64
    ):
        raise LTSNContractError("V3.8-B requires the original 320/64 prompt split")
    evaluation = []
    if cv_fold is not None:
        fit, evaluation = _train_family_cv_split(fit, cv_fold)
    experiment_contract = {
        **experiment,
        "global_variant": global_variant,
        "local_variant": local_variant,
        "ranking_objective": training.ranking_objective,
        "ordinal_weight": experiment["ordinal_weight"] if global_variant == "g1" else 0.0,
        "shape_weight": experiment["shape_weight"] if local_variant == "l2" else 0.0,
        "ensemble_readout": "mean_member_energies",
    }
    protocol = {
        "validation_scope": "train_family_cv" if cv_fold is not None else "development",
        "cv_fold": cv_fold,
        "cv_assignment": "sorted_train_families_stride_5" if cv_fold is not None else None,
        "dataset_manifest_sha256": dataset_hash,
        "development_used": False,
        "checkpoint_selection": SELECTION,
        "outer_evaluation_count": 1 if evaluation else 0,
        "train_sample_ids": sorted(r.sample_id for r in fit),
        "selection_sample_ids": sorted(r.sample_id for r in evaluation),
        "train_families": sorted({r.prompt_family for r in fit}),
        "selection_families": sorted({r.prompt_family for r in evaluation}),
        "experiment_contract": experiment_contract,
        "production_authorization": False,
    }
    dataset_rows = read_csv(dataset_manifest)
    validate_protocol(protocol, dataset_rows)
    _seed_everything(training.seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    parent_fields = {}
    if global_manifest is None:
        model = PromptConditionedTopologyEnergy(model_config).to(device)
        stats = initialize_statistics(model, fit, training, device, global_variant)
        stage = "global"
    else:
        parent, parent_protocol = verify_manifest(global_manifest, dataset_rows)
        if parent["experiment_contract"]["local_variant"] != "l0":
            raise LTSNContractError("Local branches must start from a global-only checkpoint")
        for key, expected in [
            ("seed", training.seed),
            ("training_manifest_sha256", dataset_hash),
            ("source_training_config_sha256", sha256_file(config_path)),
            ("fingerprint_json_sha256", contract.artifact_sha256),
            ("cv_fold", cv_fold),
        ]:
            if parent.get(key) != expected:
                raise LTSNContractError(f"Global parent mismatch: {key}")
        if parent["experiment_contract"]["global_variant"] != global_variant:
            raise LTSNContractError("Global parent variant mismatch")
        if parent_protocol["train_sample_ids"] != protocol["train_sample_ids"]:
            raise LTSNContractError("Global parent fit IDs mismatch")
        model, checkpoint_metadata = load_pitch3_lte_checkpoint(
            local_file(global_manifest.parent, parent["checkpoint"]),
            device=device,
            expected_sha256=parent["checkpoint_sha256"],
        )
        if checkpoint_metadata["experiment_contract"] != parent["experiment_contract"]:
            raise LTSNContractError("Global checkpoint metadata differs from manifest")
        for key in (
            "run_protocol_sha256",
            "training_manifest_sha256",
            "fingerprint_json_sha256",
            "seed",
            "cv_fold",
            "validation_scope",
        ):
            if checkpoint_metadata.get(key) != parent.get(key):
                raise LTSNContractError(f"Global checkpoint metadata mismatch: {key}")
        if asdict(model.config) != asdict(model_config):
            # JSON tuples are lists on reload, compare their canonical serialization.
            if json.dumps(asdict(model.config), sort_keys=True) != json.dumps(
                asdict(model_config), sort_keys=True
            ):
                raise LTSNContractError("Global parent model configuration differs")
        stats = read_json(global_manifest.parent / "pitch3_lte_training_statistics.json")
        parent_fields = {
            "source_global_manifest": str(global_manifest.resolve()),
            "source_global_manifest_sha256": sha256_file(global_manifest),
            "source_global_checkpoint_sha256": parent["checkpoint_sha256"],
        }
        stage = "local"
    stats = {**stats, "run_protocol_sha256": None}
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "pitch3_lte_run_protocol.json"
    write_json_atomic(protocol_path, {**protocol, **parent_fields})
    stats["run_protocol_sha256"] = sha256_file(protocol_path)
    stats_path = output_dir / "pitch3_lte_training_statistics.json"
    write_json_atomic(stats_path, stats)
    effective_path = output_dir / "pitch3_lte_effective_config.json"
    write_json_atomic(
        effective_path,
        {"model": asdict(model_config), "training": asdict(training), "v38b": experiment_contract},
    )
    weights = weights_for(training, experiment, global_variant, local_variant, stage)
    train_loader = loader(fit, training.seed, stage, shuffle=True, workers=training.num_workers)
    sample = _to_device(next(iter(loader(fit, training.seed, stage))), device)
    before = gradient_snapshot(model, sample, stage, training, stats, weights)
    write_json_atomic(output_dir / "pitch3_lte_gradient_initial.json", before)
    frozen = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if stage == "local" and not n.startswith("local_energy_head.")
    }
    _set_training_stage_trainable(model, training, stage)
    if stage == "global" and global_variant != "g1":
        model.ordinal_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training.learning_rate if stage == "global" else training.local_learning_rate,
        weight_decay=training.weight_decay,
    )
    # Diagnostics and initialization must not change training dropout draws across G0/G1.
    _seed_everything(training.seed)
    history = []
    epochs = experiment[f"{stage}_epochs"]
    for epoch in range(1, epochs + 1):
        _set_training_stage_trainable(model, training, stage)
        if stage == "global" and global_variant != "g1":
            model.ordinal_head.requires_grad_(False)
        train_loader.batch_sampler.set_epoch(epoch)
        sums = dict.fromkeys(weights, 0.0)
        norms = []
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses = forward_losses(model, batch, stage, training, stats)
            contributions = {
                k: w * losses[k] / stats["loss_normalizers"][k] for k, w in weights.items()
            }
            total = sum(contributions.values())
            if not bool(torch.isfinite(total)):
                raise LTSNContractError("Non-finite V3.8-B loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            norms.append(float(norm))
            for key, value in contributions.items():
                sums[key] += float(value.detach())
        item = {
            "epoch": epoch,
            "training_stage": stage,
            "normalized_training_contributions": {k: v / len(norms) for k, v in sums.items()},
            "gradient_norm_mean": float(np.mean(norms)),
            "gradient_norm_max": max(norms),
            "clipping_fraction": float(np.mean(np.array(norms) > training.gradient_clip_norm)),
            "selection_evaluated": False,
        }
        history.append(item)
        write_json_atomic(output_dir / "pitch3_lte_training_history.json", {"history": history})
        print(json.dumps(item), flush=True)
        if epoch == 1 and training.maximum_initial_component_contribution > 0:
            excessive = {
                k: v
                for k, v in item["normalized_training_contributions"].items()
                if v > training.maximum_initial_component_contribution
            }
            if excessive:
                write_json_atomic(output_dir / "pitch3_lte_loss_balance_failure.json", excessive)
                raise LTSNContractError("V3.8-B initial loss contribution exceeds frozen limit")
    for name, value in frozen.items():
        if not torch.equal(dict(model.named_parameters())[name].detach().cpu(), value):
            raise LTSNContractError(f"Local training changed frozen global parameter: {name}")
    write_json_atomic(
        output_dir / "pitch3_lte_gradient_final.json",
        gradient_snapshot(model, sample, stage, training, stats, weights),
    )
    train_path, train_diag = export_predictions(
        model, fit, records, device, training, stats, output_dir, "train"
    )
    result_fields = {
        "train_predictions_sha256": sha256_file(train_path),
        "train_metrics": train_diag["metrics"],
    }
    if evaluation:
        path, diag = export_predictions(
            model, evaluation, records, device, training, stats, output_dir, "selection"
        )
        result_fields.update(
            {
                "selection_predictions_sha256": sha256_file(path),
                "best_development_metrics": diag["metrics"],
            }
        )
    metadata = {
        **{k: v for k, v in protocol.items() if not k.endswith("sample_ids")},
        **parent_fields,
        **result_fields,
        "schema_version": 1,
        "model_family": LTE_MODEL_FAMILY,
        "architecture_revision": REVISION,
        "fingerprint_id": contract.fingerprint_id,
        "fingerprint_spec_revision": contract.spec_revision,
        "fingerprint_json_sha256": contract.artifact_sha256,
        "training_manifest_sha256": dataset_hash,
        "dataset_plan_sha256": summary["dataset_plan_sha256"],
        "source_training_config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "pitch3_lte_v38b.py",
                "pitch3_lte_v38b_protocol.py",
                "pitch3_lte.py",
                "pitch3_lte_training.py",
                "pitch3_lte_metrics.py",
            )
        },
        "training_config_sha256": sha256_file(effective_path),
        "effective_training_config": str(effective_path.resolve()),
        "run_protocol_sha256": sha256_file(protocol_path),
        "training_statistics_sha256": sha256_file(stats_path),
        "seed": training.seed,
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "optimized_parameters": sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        ),
        "diagnostic_artifacts_sha256": {
            p.name: sha256_file(p)
            for p in sorted(output_dir.glob("pitch3_lte_*.json"))
            if p.name
            in {
                "pitch3_lte_gradient_initial.json",
                "pitch3_lte_gradient_final.json",
                "pitch3_lte_train_diagnostics.json",
                "pitch3_lte_selection_diagnostics.json",
                "pitch3_lte_training_history.json",
            }
        },
        "precision": "fp32",
        "guidance_steps": [4],
        "training_radius_ratio": 0.05,
        "maximum_guidance_update_ratio": 0.025,
        "energy_target": "log1p(exact_pitch3_target_band_loss)",
        "local_residual_mode": "zero_global_only" if stage == "global" else "trained_anchored",
        "loss_components": list(weights),
        "loss_component_weights": weights,
        "loss_normalizers": stats["loss_normalizers"],
        "energy_stratification": stats["energy_stratification"],
        "local_training_scales": stats["local_training_scales"],
        "best_epoch": epochs,
        "epochs_completed": epochs,
        "checkpoint_parameter_source": "online",
        "global_parameters_unchanged_during_local": stage == "local",
        "training_history": history,
        "qualification_eligible": False,
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    checkpoint = output_dir / f"pitch3_lte_seed_{training.seed}.pt"
    temporary = checkpoint.with_suffix(".pt.part")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
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
    write_json_atomic(output_dir / "pitch3_lte_manifest.json", manifest)
    return manifest
