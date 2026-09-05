"""Command-line entry points for the auditable LTSN training pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .ace_adapter import AceStepAdapter, GenerationRequest
from .experiment import load_experiment_config
from .ltsn_contract import load_fingerprint_contract
from .ltsn_evaluation import (
    calibrate_ensemble,
    evaluate_guidance_pairs,
    qualify_ensemble,
)
from .ltsn_exact_labeling import build_exact_snapshot_descriptors
from .ltsn_pipeline import (
    TrajectoryRecorder,
    build_exact_label_tables,
    merge_trajectory_shard_manifests,
    require_surrogate_training_gate,
    synthetic_descriptor_rows,
    validate_snapshot_coverage,
    write_csv_atomic,
    write_json_atomic,
)
from .ltsn_storage import MATERIALIZE_MODES, remove_generated_audio
from .ltsn_training import train_ensemble
from .path_homology_exact_scorer import ExactPathHomologyScorer


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _digest(value: str, name: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_rows(path: Path, seed_start: int, seeds_per_prompt: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    output: list[dict[str, Any]] = []
    for prompt_index, row in enumerate(source):
        if not row.get("prompt_id") or not row.get("caption") or not row.get("split"):
            raise ValueError("prompt manifest requires prompt_id, caption, and split")
        supplied_seed = row.get("seed", "").strip()
        seeds = (
            [int(supplied_seed)]
            if supplied_seed
            else [
                seed_start + prompt_index * seeds_per_prompt + index
                for index in range(seeds_per_prompt)
            ]
        )
        for seed in seeds:
            output.append({**row, "seed": seed})
    if not output:
        raise ValueError("prompt manifest is empty")
    return output


def _prompt_shard(
    rows: list[dict[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must lie in [0, shard count)")
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if not selected:
        raise ValueError(f"trajectory shard {shard_index} is empty")
    return selected


def _trajectory_id(row: dict[str, Any]) -> str:
    return f"{row['prompt_id']}__seed{row['seed']}"


def collect_main(argv: list[str] | None = None) -> int:
    """Collect no-op trajectories and optionally VAE-decode every selected snapshot."""

    parser = argparse.ArgumentParser(prog="focus-ltsn-collect")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ace-config", type=Path, default=Path("configs/ace_rerank_180s.toml"))
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("ace", "synthetic"), default="ace")
    parser.add_argument("--ace-model-sha256", required=True)
    parser.add_argument("--vae-sha256", required=True)
    parser.add_argument("--seed-start", type=int, default=2026071600)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only fully decoded trajectories from the atomic shard manifest",
    )
    parser.add_argument("--decode-snapshots", action="store_true")
    parser.add_argument(
        "--discard-generator-final-audio",
        action="store_true",
        help=(
            "delete the unreferenced generator WAV after all signed step snapshots are decoded; "
            "the per-snapshot WAVs and manifest remain unchanged"
        ),
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.discard_generator_final_audio and (
        args.backend != "ace" or not args.decode_snapshots
    ):
        parser.error(
            "--discard-generator-final-audio requires --backend ace and --decode-snapshots"
        )
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error("--shard-index and --shard-count must be supplied together")
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    config = load_experiment_config(root, args.ace_config)
    model_family = config.ace.model
    recorder = TrajectoryRecorder(
        output_dir,
        model_family=model_family,
        ace_model_sha256=_digest(args.ace_model_sha256, "ace_model_sha256"),
        vae_sha256=_digest(args.vae_sha256, "vae_sha256"),
        inference_steps=config.ace.inference_steps,
        engineering_smoke=args.engineering_smoke,
    )
    all_prompts = _prompt_rows(args.prompt_manifest, args.seed_start, args.seeds_per_prompt)
    prompts = (
        all_prompts
        if args.shard_index is None
        else _prompt_shard(all_prompts, args.shard_index, args.shard_count)
    )
    planned_trajectories = {
        _trajectory_id(row): (str(row["prompt_id"]), str(row["split"]))
        for row in prompts
    }
    planned_trajectory_ids = set(planned_trajectories)
    if len(planned_trajectories) != len(prompts):
        raise ValueError("collection plan contains duplicate trajectory IDs")
    manifest = output_dir / "trajectory_manifest.csv"
    plan_path = output_dir / "collection_plan.json"
    config_path = (
        args.ace_config.resolve()
        if args.ace_config.is_absolute()
        else (root / args.ace_config).resolve()
    )
    prompt_manifest = args.prompt_manifest.resolve()
    plan_payload = {
        "schema_version": 1,
        "model_family": model_family,
        "ace_model_sha256": recorder.ace_model_sha256,
        "vae_sha256": recorder.vae_sha256,
        "ace_config_sha256": _file_sha256(config_path),
        "prompt_manifest_sha256": _file_sha256(prompt_manifest),
        "seed_start": args.seed_start,
        "seeds_per_prompt": args.seeds_per_prompt,
        "duration_seconds": args.duration_seconds,
        "inference_steps": config.ace.inference_steps,
        "snapshot_steps": sorted(recorder.selected_steps),
        "decode_snapshots": args.decode_snapshots,
        "discard_generator_final_audio": args.discard_generator_final_audio,
        "engineering_smoke": args.engineering_smoke,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "planned_trajectories": [
            {
                "trajectory_id": _trajectory_id(row),
                "prompt_id": str(row["prompt_id"]),
                "split": str(row["split"]),
                "seed": int(row["seed"]),
            }
            for row in prompts
        ],
    }
    if plan_path.exists():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan_payload:
            raise ValueError(
                f"collection plan changed; use a new shard directory: {plan_path}"
            )
    else:
        if manifest.exists():
            raise ValueError(
                f"cannot safely resume a manifest without collection_plan.json: {manifest}"
            )
        write_json_atomic(plan_path, plan_payload)
    completed_trajectory_ids: frozenset[str] = frozenset()
    if manifest.exists():
        if not args.resume:
            raise FileExistsError(
                "trajectory manifest already exists; pass --resume or use a new "
                f"directory: {manifest}"
            )
        completed_trajectory_ids = recorder.resume_from_manifest(
            manifest,
            require_audio=args.decode_snapshots,
        )
        unexpected = completed_trajectory_ids - planned_trajectory_ids
        if unexpected:
            raise ValueError(
                "resumed shard contains trajectories outside the current plan: "
                f"{sorted(unexpected)[:5]}"
            )
        for record in recorder.records:
            if planned_trajectories.get(record.trajectory_id) != (
                record.prompt_id,
                record.split,
            ):
                raise ValueError(
                    "resumed trajectory prompt/split differs from the current plan: "
                    f"{record.trajectory_id}"
                )
    adapter = None
    if args.backend == "ace":
        if model_family != "acestep-v15-xl-turbo":
            raise ValueError("the pinned sampler hook currently supports ACE-Step XL-Turbo only")
        adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
        adapter.set_topology_corrector(recorder)
    discarded_generator_audio = 0
    newly_collected = 0
    for row in prompts:
        trajectory_id = _trajectory_id(row)
        if trajectory_id in completed_trajectory_ids:
            continue
        recorder.begin(
            prompt_id=row["prompt_id"], trajectory_id=trajectory_id, split=row["split"]
        )
        before = len(recorder.records)
        generated_audio = None
        try:
            if adapter is not None:
                generated = adapter.generate(
                    GenerationRequest(
                        prompt=row["caption"],
                        seed=int(row["seed"]),
                        duration_seconds=args.duration_seconds,
                        output_dir=output_dir / "final_audio" / trajectory_id,
                        inference_steps=config.ace.inference_steps,
                        bpm=int(row["bpm"]) if row.get("bpm", "").strip() else None,
                        keyscale=row.get("keyscale", ""),
                        timesignature=row.get("timesignature", ""),
                    )
                )
                generated_audio = generated.audio_path
                if generated.final_latent is None:
                    raise RuntimeError(
                        "ACE-Step returned no final pred_latents; disable save-memory mode"
                    )
                recorder.record_final_latent(generated.final_latent)
            else:
                rng = np.random.default_rng(int(row["seed"]))
                import torch

                xt = torch.from_numpy(rng.normal(size=(1, 96, 64)).astype(np.float32))
                mask = torch.ones(1, 96, dtype=torch.bool)
                for step in range(config.ace.inference_steps):
                    velocity = torch.from_numpy(
                        rng.normal(scale=0.1, size=(1, 96, 64)).astype(np.float32)
                    )
                    recorder(
                        xt_next=xt,
                        xt_before_step=xt,
                        velocity=velocity,
                        timestep=max(0.05, 1.0 - step / config.ace.inference_steps),
                        next_timestep=max(0.0, 1.0 - (step + 1) / config.ace.inference_steps),
                        step_index=step,
                        attention_mask=mask,
                    )
                    xt = xt - velocity / config.ace.inference_steps
            validate_snapshot_coverage(
                recorder.records[before:],
                expected_steps=recorder.selected_steps,
            )
        finally:
            recorder.end()
        if len(recorder.records) == before:
            raise RuntimeError(f"sampler hook recorded no snapshots for {trajectory_id}")
        if args.decode_snapshots:
            if adapter is None:
                raise ValueError("--decode-snapshots requires --backend ace")
            for record in recorder.records[before:]:
                latent = np.load(output_dir / record.latent_path, allow_pickle=False)
                audio_path = output_dir / "snapshot_audio" / f"{record.sample_id}.wav"
                adapter.decode_latent_to_audio(latent, audio_path)
                recorder.attach_audio(record.sample_id, audio_path)
        if args.discard_generator_final_audio:
            if generated_audio is None:
                raise RuntimeError("ACE-Step returned no generator audio to discard")
            remove_generated_audio(generated_audio, output_dir / "final_audio")
            discarded_generator_audio += 1
        recorder.write_manifest(manifest)
        newly_collected += 1
    validate_snapshot_coverage(
        recorder.records,
        expected_steps=recorder.selected_steps,
    )
    actual_trajectory_ids = {record.trajectory_id for record in recorder.records}
    if actual_trajectory_ids != planned_trajectory_ids:
        raise RuntimeError("completed shard differs from its deterministic trajectory plan")
    recorder.write_manifest(manifest)
    _print(
        {
            "status": "engineering_smoke_only" if args.engineering_smoke else "shadow_collected",
            "trajectory_manifest": str(manifest),
            "trajectory_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "trajectories": len(prompts),
            "snapshots": len(recorder.records),
            "decoded_snapshots": sum(bool(record.audio_path) for record in recorder.records),
            "resumed_trajectories": len(completed_trajectory_ids),
            "newly_collected_trajectories": newly_collected,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "collection_plan": str(plan_path),
            "collection_plan_sha256": _file_sha256(plan_path),
            "discarded_generator_final_audio": discarded_generator_audio,
        }
    )
    return 0


def merge_main(argv: list[str] | None = None) -> int:
    """Validate and atomically merge independently collected GPU shards."""

    parser = argparse.ArgumentParser(prog="focus-ltsn-merge-shards")
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2026071600)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    parser.add_argument("--require-audio", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    prompts = _prompt_rows(args.prompt_manifest, args.seed_start, args.seeds_per_prompt)
    expected_plan = {
        _trajectory_id(row): (str(row["prompt_id"]), str(row["split"]))
        for row in prompts
    }
    if len(expected_plan) != len(prompts):
        raise ValueError("collection plan contains duplicate trajectory IDs")
    shards_root = args.shards_root.resolve()
    shard_manifests = [
        shards_root / f"shard_{index:02d}" / "trajectory_manifest.csv"
        for index in range(args.shard_count)
    ]
    missing = [str(path) for path in shard_manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shard manifests: {missing}")
    prompt_manifest_sha256 = _file_sha256(args.prompt_manifest.resolve())
    shard_plan_sha256: list[str] = []
    common_plan_identity: dict[str, Any] | None = None
    for index, shard_manifest in enumerate(shard_manifests):
        plan_path = shard_manifest.parent / "collection_plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"missing shard collection plan: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        expected_shard = _prompt_shard(prompts, index, args.shard_count)
        expected_shard_plan = [
            {
                "trajectory_id": _trajectory_id(row),
                "prompt_id": str(row["prompt_id"]),
                "split": str(row["split"]),
                "seed": int(row["seed"]),
            }
            for row in expected_shard
        ]
        if (
            plan.get("schema_version") != 1
            or plan.get("shard_index") != index
            or plan.get("shard_count") != args.shard_count
            or plan.get("prompt_manifest_sha256") != prompt_manifest_sha256
            or plan.get("seed_start") != args.seed_start
            or plan.get("seeds_per_prompt") != args.seeds_per_prompt
            or plan.get("planned_trajectories") != expected_shard_plan
            or (args.require_audio and not plan.get("decode_snapshots"))
        ):
            raise ValueError(f"shard collection plan differs from merge request: {plan_path}")
        identity_keys = (
            "model_family",
            "ace_model_sha256",
            "vae_sha256",
            "ace_config_sha256",
            "prompt_manifest_sha256",
            "seed_start",
            "seeds_per_prompt",
            "duration_seconds",
            "inference_steps",
            "snapshot_steps",
            "decode_snapshots",
            "discard_generator_final_audio",
            "engineering_smoke",
            "shard_count",
        )
        identity = {key: plan.get(key) for key in identity_keys}
        if common_plan_identity is None:
            common_plan_identity = identity
        elif identity != common_plan_identity:
            raise ValueError("collection shard plans do not share one frozen configuration")
        shard_plan_sha256.append(_file_sha256(plan_path))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = merge_trajectory_shard_manifests(
        shard_manifests,
        output_manifest=output_dir / "trajectory_manifest.csv",
        expected_trajectory_plan=expected_plan,
        require_audio=args.require_audio,
    )
    summary["shard_collection_plan_sha256"] = shard_plan_sha256
    summary_path = output_dir / "trajectory_merge_summary.json"
    write_json_atomic(summary_path, summary)
    summary["merge_summary"] = str(summary_path)
    summary["merge_summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    _print(summary)
    return 0


def labels_main(argv: list[str] | None = None) -> int:
    """Build per-snapshot exact labels and the leakage-checked training manifest."""

    parser = argparse.ArgumentParser(prog="focus-ltsn-labels")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("metadata/focus_path_homology_fingerprint_v2.json"),
    )
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--descriptor-table", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surrogate-training-gate", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="exact-label snapshots per bounded work batch",
    )
    parser.add_argument(
        "--materialize-mode",
        choices=MATERIALIZE_MODES,
        default="auto",
        help="auto tries reflink, then hardlink, then a verified physical copy",
    )
    parser.add_argument(
        "--keep-batch-work",
        action="store_true",
        help="retain reconstructible per-batch audio/features for debugging",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore and replace completed batch descriptor checkpoints",
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    fingerprint = (
        (root / args.fingerprint).resolve()
        if not args.fingerprint.is_absolute()
        else args.fingerprint
    )
    scorer = ExactPathHomologyScorer.from_json(fingerprint)
    gate = require_surrogate_training_gate(
        args.surrogate_training_gate,
        scorer.contract,
        engineering_smoke=args.engineering_smoke,
    )
    if not args.engineering_smoke:
        with args.trajectory_manifest.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            validate_snapshot_coverage(list(csv.DictReader(handle)))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_table = args.descriptor_table
    storage_summary = None
    if descriptor_table is None:
        descriptor_table = output_dir / "exact_snapshot_descriptors.csv"
        if args.engineering_smoke:
            rows = synthetic_descriptor_rows(
                args.trajectory_manifest,
                pitch_dimensions=len(scorer.transforms["pitch"]["input_features"]),
            )
            write_csv_atomic(descriptor_table, rows)
        else:
            storage_summary = build_exact_snapshot_descriptors(
                project_root=root,
                trajectory_manifest=args.trajectory_manifest,
                work_dir=(args.work_dir or output_dir / "exact_work").resolve(),
                output_path=descriptor_table,
                workers=args.workers,
                batch_size=args.batch_size,
                materialize_mode=args.materialize_mode,
                cleanup_batches=not args.keep_batch_work,
                resume=not args.no_resume,
            )
    summary = build_exact_label_tables(
        trajectory_manifest=args.trajectory_manifest,
        descriptor_table=descriptor_table,
        output_manifest=output_dir / "ltsn_manifest.csv",
        exact_label_table=output_dir / "exact_labels.csv",
        split_manifest=output_dir / "split_manifest.json",
        scorer=scorer,
        gate=gate,
        engineering_smoke=args.engineering_smoke,
    )
    if storage_summary is not None:
        summary["storage"] = storage_summary
    _print(summary)
    return 0


def train_main(argv: list[str] | None = None) -> int:
    """Train the model-specific ensemble under the Stage-1 gate."""

    parser = argparse.ArgumentParser(prog="focus-ltsn-train")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surrogate-training-gate", type=Path)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device")
    devices.add_argument(
        "--devices",
        nargs="+",
        help="one explicit CUDA device per configured seed, for example cuda:0 cuda:1 cuda:2",
    )
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args(argv)
    _print(
        train_ensemble(
            fingerprint_path=args.fingerprint,
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            config_path=args.config,
            output_dir=args.output_dir,
            surrogate_training_gate_path=args.surrogate_training_gate,
            engineering_smoke=args.engineering_smoke,
            device_name=args.device,
            device_names=args.devices,
        )
    )
    return 0


def qualification_main(argv: list[str] | None = None) -> int:
    """Calibrate or independently qualify a frozen ensemble."""

    parser = argparse.ArgumentParser(prog="focus-ltsn-qualify")
    parser.add_argument("command", choices=("calibrate", "qualify"))
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--guidance-development-report", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        payload = calibrate_ensemble(
            fingerprint_path=args.fingerprint,
            manifest_path=args.manifest,
            ensemble_manifest=args.ensemble_manifest,
            output_path=args.output,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    else:
        if args.calibration is None:
            parser.error("qualify requires --calibration")
        payload = qualify_ensemble(
            fingerprint_path=args.fingerprint,
            manifest_path=args.manifest,
            ensemble_manifest=args.ensemble_manifest,
            calibration_path=args.calibration,
            output_path=args.output,
            guidance_development_report=args.guidance_development_report,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    _print(payload)
    return 0 if payload.get("status") != "failed" else 1


def guidance_main(argv: list[str] | None = None) -> int:
    """Evaluate paired decoded exact/proxy changes for development or confirmation."""

    parser = argparse.ArgumentParser(prog="focus-ltsn-guidance-eval")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--qualification-report", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args(argv)
    contract = load_fingerprint_contract(args.fingerprint)
    payload = evaluate_guidance_pairs(
        pair_table=args.pair_table,
        output_path=args.output,
        fingerprint_sha256=contract.artifact_sha256,
        mode=args.mode,
        qualification_report=args.qualification_report,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    _print(payload)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(train_main())
