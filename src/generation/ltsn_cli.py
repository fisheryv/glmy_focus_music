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
    require_surrogate_training_gate,
    synthetic_descriptor_rows,
    write_csv_atomic,
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
    prompts = _prompt_rows(args.prompt_manifest, args.seed_start, args.seeds_per_prompt)
    adapter = None
    if args.backend == "ace":
        if model_family != "acestep-v15-xl-turbo":
            raise ValueError("the pinned sampler hook currently supports ACE-Step XL-Turbo only")
        adapter = AceStepAdapter(root / config.ace.checkout, config.ace)
        adapter.set_topology_corrector(recorder)
    discarded_generator_audio = 0
    for row in prompts:
        trajectory_id = f"{row['prompt_id']}__seed{row['seed']}"
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
    manifest = output_dir / "trajectory_manifest.csv"
    recorder.write_manifest(manifest)
    _print(
        {
            "status": "engineering_smoke_only" if args.engineering_smoke else "shadow_collected",
            "trajectory_manifest": str(manifest),
            "trajectory_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "trajectories": len(prompts),
            "snapshots": len(recorder.records),
            "decoded_snapshots": sum(bool(record.audio_path) for record in recorder.records),
            "discarded_generator_final_audio": discarded_generator_audio,
        }
    )
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
