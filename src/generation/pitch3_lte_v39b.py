"""Fixed-budget fit probes and matched frozen-transition distillation."""

from __future__ import annotations

import json
import platform
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .ltsn_contract import sha256_file
from .ltsn_pipeline import write_json_atomic
from .pitch3_lte_training import (
    _raw_loss_kwargs,
    _seed_everything,
    _set_training_stage_trainable,
    _to_device,
    load_pitch3_lte_config,
    pitch3_lte_raw_losses,
)
from .pitch3_lte_v38b import export_predictions, initialize_statistics, loader
from .pitch3_lte_v38b_protocol import correlation
from .pitch3_lte_v39a import (
    build_matched_models,
    check_initial_identity,
    fit_diagnostics,
    gradient_norms,
    parameter_groups,
    prepare_data,
    state_hash,
)
from .pitch3_lte_v39a_protocol import content_hash
from .pitch3_lte_v39b_protocol import (
    REVISION,
    SELECTION,
    load_teacher,
    split_rows,
    validate_config,
    validate_protocol,
    variants_for,
    verify_run,
)


def forward_bundle(model, head, batch, training, stats, teacher, contract):
    captured = []
    hook = (
        model.transition_branch.summary.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        if head is not None
        else None
    )
    try:
        latent = model.encode_latent(batch["latent"], batch["attention_mask"])
    finally:
        if hook is not None:
            hook.remove()
    prompt = model.encode_prompt(batch["text_hidden"], batch["text_mask"])
    parts = model.potential_components_from_states(latent, prompt, latent.detach())
    losses = pitch3_lte_raw_losses(
        parts.global_energy,
        batch,
        predicted_coordinates=parts.coordinates,
        ranking_logits=parts.global_logit,
        **_raw_loss_kwargs(
            training,
            stats["local_training_scales"],
            stats["energy_stratification"],
            stats["coordinate_auxiliary"],
            include_structured=True,
        ),
    )
    if head is not None:
        if len(captured) != 1:
            raise RuntimeError("Transition summary was not captured exactly once")
        logp = head(captured[0]).log_softmax(-1)
        p = logp.exp().reshape(-1, 16, 16)
        target = torch.as_tensor(
            np.stack([teacher[sid] for sid in batch["sample_id"]]), dtype=p.dtype, device=p.device
        )
        losses["transition_kl"] = (
            (target.flatten(1) * (target.flatten(1).clamp_min(1e-30).log() - logp)).sum(1).mean()
        )
        raw = torch.stack((p.diagonal(dim1=1, dim2=2).sum(1), p.square().sum((1, 2))), 1)
        center = p.new_tensor(contract.transform_center[1:])
        scale = p.new_tensor(contract.transform_scale[1:])
        width = p.new_tensor(contract.target_upper[1:]) - p.new_tensor(contract.target_lower[1:])
        q = (raw - center) / scale
        losses["transition_coordinate"] = F.smooth_l1_loss(
            q / width, batch["coordinates"][:, 1:] / width
        )
    return losses, parts


def family_gradient_probe(model, head, fit, training, stats, weights, teacher, contract, device):
    """Eval-only probes; no optimization, always called inside an RNG fork."""
    model.eval()
    groups = parameter_groups(model)
    if head is not None and any(p.requires_grad for p in head.parameters()):
        groups["transition_teacher_head"] = list(head.parameters())
    parameters = [p for group in groups.values() for p in group]
    out = {}
    for raw in loader(fit, training.seed, "global"):
        batch = _to_device(raw, device)
        losses, parts = forward_bundle(model, head, batch, training, stats, teacher, contract)
        selected = batch["energy_target"] <= stats["energy_stratification"]["thresholds"][1]
        y, pred = batch["energy_target"][selected], parts.global_energy[selected]
        z = parts.global_logit[selected]
        band = batch["exact_band"][selected]
        delta = band[:, None] - band[None, :]
        valid = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1) & (
            delta.abs() > training.rank_min_delta
        )
        terms = {
            name: weight * losses[name] / stats["loss_normalizers"][name]
            for name, weight in weights.items()
            if weight
        }
        terms["low_value_probe"] = (
            F.huber_loss(pred, y) if len(y) else parts.global_energy.sum() * 0
        )
        terms["low_rank_probe"] = (
            F.softplus(-delta.sign() * (z[:, None] - z[None, :]))[valid].mean()
            if valid.any()
            else parts.global_logit.sum() * 0
        )
        terms["optimized_total"] = sum(
            value for name, value in terms.items() if not name.endswith("_probe")
        )
        norms = {}
        for name, value in terms.items():
            grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            offset = 0
            norms[name] = {}
            for group, params in groups.items():
                values = grads[offset : offset + len(params)]
                norms[name][group] = float(
                    np.sqrt(
                        sum(
                            float(g.detach().double().square().sum().cpu())
                            for g in values
                            if g is not None
                        )
                    )
                )
                offset += len(params)
        exact = batch["coordinates"][selected].detach().cpu().numpy()
        estimated = parts.coordinates[selected].detach().cpu().numpy()
        family = batch["prompt_family"][0]
        out[family] = {
            "low_samples": int(selected.sum()),
            "low_rank_pairs": int(valid.sum()),
            "losses": {k: float(v.detach()) for k, v in terms.items()},
            "block_gradient_norms": norms,
            "low_coordinate_rho": [correlation(exact[:, i], estimated[:, i]) for i in range(3)],
        }
    return {"mode": "eval_fit_only", "diagnostic_only": True, "families": out}


def train_v39b(
    *,
    fingerprint_path,
    dataset_manifest,
    config_path,
    output_dir,
    mode,
    variant,
    cv_fold,
    seed=20260941,
    device_name="cpu",
    teacher_manifest=None,
):
    output_dir, dataset_manifest = Path(output_dir), Path(dataset_manifest).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {output_dir}")
    if variant not in variants_for(mode) or type(seed) is not int or seed < 0:
        raise ValueError("Invalid variant or seed")
    payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    experiment = validate_config(payload)
    config, training = load_pitch3_lte_config(config_path)
    training = replace(training, seed=seed)
    contract, _, records, rows, _, _ = prepare_data(
        fingerprint_path, dataset_manifest, config, cv_fold, False
    )
    fit_rows, outer_rows = split_rows(rows, cv_fold, mode)
    by_id = {r.sample_id: r for r in records}
    fit = [by_id[r["sample_id"]] for r in fit_rows]
    outer = [by_id[r["sample_id"]] for r in outer_rows]
    epochs = experiment[
        {"budget": "budget_epochs", "distill-diagnose": "diagnostic_epochs", "cv": "cv_epochs"}[
            mode
        ]
    ]
    teacher, teacher_meta = {}, None
    if mode != "budget":
        if teacher_manifest is None:
            raise ValueError("Distillation comparisons require a verified --teacher-manifest")
        teacher, teacher_meta = load_teacher(
            Path(teacher_manifest),
            dataset_manifest,
            fingerprint_path,
            [r.sample_id for r in fit if r.source_kind == "base_step4_seed"],
        )
    protocol = {
        "mode": mode,
        "variant": variant,
        "cv_fold": cv_fold,
        "seed": seed,
        "epochs": epochs,
        "checkpoint_selection": SELECTION,
        "development_used": False,
        "outer_evaluation_count": int(mode == "cv"),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "source_config_sha256": sha256_file(config_path),
        "fingerprint_json_sha256": contract.artifact_sha256,
        "teacher_sha256": sha256_file(teacher_manifest) if teacher_meta else None,
        "teacher_fit_sample_ids": sorted(teacher),
        "train_sample_ids": sorted(r.sample_id for r in fit),
        "selection_sample_ids": sorted(r.sample_id for r in outer),
        "train_families": sorted({r.prompt_family for r in fit}),
        "selection_families": sorted({r.prompt_family for r in outer}),
        "experiment": experiment,
    }
    validate_protocol(protocol, rows)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
    rng_devices = [device.index] if device.type == "cuda" else []
    baseline, candidate, initial = build_matched_models(config, seed)
    baseline.to(device)
    candidate.to(device)
    stats = initialize_statistics(baseline, fit, training, device, "g0")
    weights = {
        "value": 1.0,
        "prompt_rank": 1.0,
        "family_stratified_rank": 1.0,
        "coordinate": training.coordinate_loss_weight,
    }
    sample = _to_device(next(iter(loader(fit, seed, "global"))), device)
    initial.update(check_initial_identity(baseline, candidate, sample, training, stats, weights))
    model = baseline if variant == "baseline" else candidate
    if mode != "budget":
        # Both arms have the identical teacher head for normalization/evaluation.
        # Its construction does not consume the common training RNG stream.
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed + 390_000)
            head = torch.nn.Linear(config.model_dim, 256).to(device)
        initial["teacher_head_state_sha256"] = state_hash(head.state_dict())
        initial["transition_initial_state_sha256"] = state_hash(candidate.state_dict())
        values = {"transition_kl": [], "transition_coordinate": []}
        candidate.eval()
        with torch.no_grad():
            for raw in loader(fit, seed, "global"):
                losses, _ = forward_bundle(
                    candidate, head, _to_device(raw, device), training, stats, teacher, contract
                )
                for name in values:
                    values[name].append(float(losses[name]))
        stats["loss_normalizers"].update(
            {k: max(float(np.mean(v)), 1e-6) for k, v in values.items()}
        )
        weights.update(
            transition_kl=experiment["transition_kl_weight"] if variant == "distill" else 0.0,
            transition_coordinate=experiment["transition_coordinate_weight"]
            if variant == "distill"
            else 0.0,
        )
        for p in head.parameters():
            p.requires_grad_(variant == "distill")
    else:
        head = None
    del baseline, candidate
    _set_training_stage_trainable(model, training, "global")
    groups = parameter_groups(model)
    if head is not None and variant == "distill":
        groups["transition_teacher_head"] = list(head.parameters())
    parameters = [p for group in groups.values() for p in group]
    frozen_local = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith("local_energy_head.")
    }
    optimizer = torch.optim.AdamW(
        parameters, lr=training.learning_rate, weight_decay=training.weight_decay
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name, data):
        write_json_atomic(output_dir / name, data)

    write("pitch3_lte_run_protocol.json", protocol)
    write("pitch3_lte_initialization.json", initial)
    write("pitch3_lte_training_statistics.json", stats)
    write(
        "pitch3_lte_effective_config.json",
        {
            "model": asdict(model.config),
            "training": asdict(training),
            "v39b": experiment,
            "weights": weights,
        },
    )
    _seed_everything(seed)
    train_loader = loader(fit, seed, "global", shuffle=True, workers=training.num_workers)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.random.fork_rng(devices=rng_devices):
        write(
            "pitch3_lte_fit_epoch_zero.json", fit_diagnostics(model, fit, device, training, stats)
        )
        write(
            "pitch3_lte_family_gradient_epoch_000.json",
            family_gradient_probe(
                model, head, fit, training, stats, weights, teacher, contract, device
            ),
        )
    history = []
    for epoch in range(1, epochs + 1):
        _set_training_stage_trainable(model, training, "global")
        train_loader.batch_sampler.set_epoch(epoch)
        old = {g: [p.detach().clone() for p in ps] for g, ps in groups.items()}
        norms, blocks, families = [], [], {}
        sums = dict.fromkeys(weights, 0.0)
        started = time.perf_counter()
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses, _ = forward_bundle(model, head, batch, training, stats, teacher, contract)
            contributions = {
                k: w * losses[k] / stats["loss_normalizers"][k] for k, w in weights.items()
            }
            total = sum(contributions.values())
            if not torch.isfinite(total):
                raise ValueError("Non-finite training loss")
            total.backward()
            before = gradient_norms(groups)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, training.gradient_clip_norm, error_if_nonfinite=True
            )
            after = gradient_norms(groups)
            optimizer.step()
            norms.append(float(norm))
            blocks.append({g: {"before_clip": before[g], "after_clip": after[g]} for g in groups})
            families[batch["prompt_family"][0]] = {
                "normalized_losses": {k: float(v.detach()) for k, v in contributions.items()},
                "gradient_norm_before_clip": float(norm),
                "block_gradient_norms": blocks[-1],
            }
            for k, v in contributions.items():
                sums[k] += float(v.detach())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        updates = {}
        with torch.no_grad():
            for g, ps in groups.items():
                delta = float(
                    sum((p - q).square().sum() for p, q in zip(ps, old[g], strict=True)).sqrt()
                )
                norm = float(sum(q.square().sum() for q in old[g]).sqrt())
                updates[g] = {
                    "epoch_net_update_l2": delta,
                    "relative_to_epoch_start": delta / norm if norm else None,
                }
        del old
        with torch.random.fork_rng(devices=rng_devices):
            fit_report = fit_diagnostics(model, fit, device, training, stats)
            if epoch == epochs or (mode != "cv" and epoch in (12, 24)):
                write(
                    f"pitch3_lte_family_gradient_epoch_{epoch:03d}.json",
                    family_gradient_probe(
                        model, head, fit, training, stats, weights, teacher, contract, device
                    ),
                )
        item = {
            "epoch": epoch,
            "optimizer_steps": len(norms),
            "training_stage": "global",
            "normalized_training_contributions": {k: v / len(norms) for k, v in sums.items()},
            "gradient_norm_mean": float(np.mean(norms)),
            "gradient_norm_max": max(norms),
            "clipping_fraction": float(np.mean(np.array(norms) > training.gradient_clip_norm)),
            "block_gradient_norms_mean": {
                g: {
                    phase: float(np.mean([b[g][phase] for b in blocks]))
                    for phase in ("before_clip", "after_clip")
                }
                for g in groups
            },
            "block_parameter_updates": updates,
            "families_training": families,
            "training_seconds": seconds,
            "fit_evaluation": fit_report,
            "selection_evaluated": False,
        }
        history.append(item)
        write("pitch3_lte_training_history.json", {"history": history})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "mode": mode,
                    "fit_min_family_rho": fit_report["metrics"]["minimum_prompt_family_spearman"],
                    "clipping_fraction": item["clipping_fraction"],
                }
            ),
            flush=True,
        )
        if (
            epoch == 1
            and max(item["normalized_training_contributions"].values())
            > training.maximum_initial_component_contribution
        ):
            raise ValueError("Initial loss contribution exceeds declared limit")
    if any(
        not torch.equal(p, dict(model.named_parameters())[n].detach())
        for n, p in frozen_local.items()
    ):
        raise ValueError("Global experiment modified the local head")
    _, train_report = export_predictions(
        model, fit, records, device, training, stats, output_dir, "train"
    )
    outer_report = None
    if outer:
        _, outer_report = export_predictions(
            model, outer, records, device, training, stats, output_dir, "selection"
        )
        checkpoint = output_dir / f"pitch3_lte_seed_{seed}.pt"
        temporary = checkpoint.with_suffix(".pt.part")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model.config),
                "training_config": asdict(training),
                "teacher_head_state_dict": head.state_dict(),
                "metadata": {
                    "architecture_revision": REVISION,
                    "internal_cv_only": True,
                    "protocol": protocol,
                },
            },
            temporary,
        )
        temporary.replace(checkpoint)
    result = {
        **protocol,
        "architecture_revision": REVISION,
        "completed": True,
        "diagnostic_only": True,
        "epochs_completed": epochs,
        "protocol_content_sha256": content_hash(protocol),
        "shared_initial_state_sha256": initial["shared_state_sha256"],
        "initial_normalizers_sha256": content_hash(stats["loss_normalizers"]),
        "initial_teacher_head_sha256": initial.get("teacher_head_state_sha256"),
        "train_metrics": train_report["metrics"],
        "selection_metrics": outer_report["metrics"] if outer_report else None,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "implementation_sha256": {
            p.name: sha256_file(p)
            for p in [
                *Path(__file__).parent.glob("pitch3_lte*.py"),
                Path(__file__).with_name("path_homology_surrogate.py"),
                Path(__file__).with_name("pitch3_contract.py"),
            ]
        },
        "artifacts_sha256": {p.name: sha256_file(p) for p in output_dir.iterdir() if p.is_file()},
        "guidance_promotion_eligible": False,
        "production_authorization": False,
    }
    write("pitch3_lte_v39b_complete.json", result)
    verify_run(output_dir / "pitch3_lte_v39b_complete.json", rows)
    return result
