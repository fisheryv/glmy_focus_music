"""V3.9-A bounded experiment queues; matched pairs share one GPU, no auto-selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_pitch3_lte_v38b import execute_jobs, preflight_devices  # noqa: E402

DEVICE_STAGES = {"check", "diagnose", "cv-global", "train-global", "model-screen"}


def build_jobs(args):
    for name in ("devices", "folds"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Nonempty unique --{name} required")
    variants = args.variants or ["baseline", "transition"]
    seeds = args.seeds or (
        [20260941, 20260942, 20260943]
        if args.stage in {"train-global", "ensemble", "model-screen"}
        else [20260941]
    )
    if (
        len(variants) != len(set(variants))
        or len(seeds) != len(set(seeds))
        or any(s < 0 for s in seeds)
    ):
        raise ValueError("Variants/seeds must be unique; seeds nonnegative")
    if args.stage in {"train-global", "ensemble", "model-screen"} and len(variants) != 1:
        raise ValueError("Explicitly select one --variants value after internal CV review")
    if args.stage in {"train-global", "ensemble", "model-screen"} and len(seeds) != 3:
        raise ValueError("Final V3.9-A uses three predeclared seeds")
    jobs = []

    def add(command, output, device, require_empty=True):
        if args.stage in DEVICE_STAGES:
            command = [*command, "--device", device]
        jobs.append(
            {
                "command": [args.python, *map(str, command)],
                "output_dir": str(output),
                "device": device,
                "require_empty": require_empty,
            }
        )

    root, runs = args.root, args.run_root
    common = ["--fingerprint", args.fingerprint, "--manifest", args.dataset_manifest]
    if args.stage == "check":
        for index, device in enumerate(args.devices):
            add(
                [
                    root / "scripts/check_pitch3_lte_v39a.py",
                    "--config",
                    args.config,
                    "--output-dir",
                    runs / "checks" / f"device_{index}",
                ],
                runs / "checks" / f"device_{index}",
                device,
            )
    elif args.stage in {"diagnose", "cv-global", "train-global"}:
        full = args.stage == "train-global"
        for fi, fold in enumerate([None] if full else args.folds):
            for si, seed in enumerate(seeds):
                # Both variants of each fold/seed stay sequentially on one GPU.
                device = args.devices[(fi * len(seeds) + si) % len(args.devices)]
                for variant in variants:
                    base = (
                        runs
                        / ("diagnostics" if args.stage == "diagnose" else "final" if full else "cv")
                        / variant
                    )
                    if fold is not None:
                        base /= f"fold_{fold}"
                    output = (
                        base
                        / f"seed_{seed}"
                        / ("diagnostic" if args.stage == "diagnose" else "models")
                    )
                    command = [
                        root / "scripts/train_pitch3_lte_v39a.py",
                        *common,
                        "--config",
                        args.config,
                        "--variant",
                        variant,
                        "--seed",
                        seed,
                        "--output-dir",
                        output,
                    ]
                    if fold is not None:
                        command += ["--cv-fold", fold]
                    if args.stage == "diagnose":
                        command += ["--diagnostic"]
                    add(command, output, device)
    elif args.stage == "summary":
        add(
            [
                root / "scripts/summarize_pitch3_lte_v39a.py",
                "--run-root",
                runs,
                "--dataset-manifest",
                args.dataset_manifest,
                "--seeds",
                *seeds,
                "--variants",
                *variants,
            ],
            runs / "summary",
            args.devices[0],
            False,
        )
    elif args.stage in {"ensemble", "model-screen"}:
        base = runs / "final" / variants[0]
        ensemble = base / "pitch3_lte_ensemble.json"
        if args.stage == "ensemble":
            command = [root / "scripts/build_pitch3_lte_ensemble.py", "--output", ensemble]
            for seed in seeds:
                command += ["--manifest", base / f"seed_{seed}/models/pitch3_lte_manifest.json"]
            add(command, base, args.devices[0], False)
        else:
            output = base / "development_screen_model_only_fp32"
            add(
                [
                    root / "scripts/evaluate_pitch3_lte.py",
                    "screen-development",
                    *common,
                    "--ensemble-manifest",
                    ensemble,
                    "--output-dir",
                    output,
                ],
                output,
                args.devices[0],
            )
    return jobs


def verify_run(output, stage):
    from generation.pitch3_lte_v39a_protocol import read_json, verify_diagnostic, verify_manifest

    if stage in {"cv-global", "train-global"}:
        verify_manifest(output / "pitch3_lte_manifest.json")
    elif stage == "diagnose":
        verify_diagnostic(output / "pitch3_lte_diagnostic_complete.json")
    elif stage == "check":
        if not read_json(output / "pitch3_lte_v39a_checks.json").get("all_checks_passed"):
            raise ValueError("Network checks did not complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "check",
            "diagnose",
            "cv-global",
            "summary",
            "train-global",
            "ensemble",
            "model-screen",
        ],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--min-free-gpu-gib", type=float, default=8.0)
    parser.add_argument("--skip-busy-gpus", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--folds", type=int, choices=range(5), nargs="+", default=list(range(5)))
    parser.add_argument("--variants", choices=["baseline", "transition"], nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.run_root = (args.run_root or args.root / "runs/pitch3_lte_v39a").resolve()
    args.dataset_manifest = (
        args.dataset_manifest
        or args.root / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv"
    ).resolve()
    args.fingerprint = (
        args.fingerprint or args.root / "metadata/focus_pitch3_fingerprint_v1.json"
    ).resolve()
    args.config = (args.config or args.root / "configs/pitch3_lte_v39a.toml").resolve()
    build_jobs(args)  # Validate choices before probing any devices.
    preflight_devices(args, device_stages=DEVICE_STAGES)
    execute_jobs(build_jobs(args), args, verify_run=verify_run)


if __name__ == "__main__":
    main()
