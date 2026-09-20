"""Next-stage queues: budget first, explicit teacher/distillation stages later."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from generation.pitch3_lte_v39b_protocol import variants_for, verify_run  # noqa: E402
from scripts.run_pitch3_lte_v38b import execute_jobs, preflight_devices  # noqa: E402

DEVICE_STAGES = {"check", "budget", "teacher", "distill-diagnose", "cv"}


def build_jobs(args):
    for name in ("devices", "folds", "seeds"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Unique nonempty --{name} required")
    if any(type(s) is not int or s < 0 for s in args.seeds):
        raise ValueError("Nonnegative seeds required")
    if any(type(f) is not int or f not in range(5) for f in args.folds):
        raise ValueError("Folds must be integers from 0 through 4")
    jobs = []

    def add(command, output, device, empty=True):
        if args.stage in DEVICE_STAGES:
            command += ["--device", device]
        jobs.append(
            {
                "command": [args.python, *map(str, command)],
                "output_dir": str(output),
                "device": device,
                "require_empty": empty,
            }
        )

    common = ["--manifest", args.dataset_manifest, "--fingerprint", args.fingerprint]
    if args.stage in {"budget", "distill-diagnose", "cv"}:
        variants = args.variants or list(variants_for(args.stage))
        if len(set(variants)) != len(variants) or not set(variants) <= set(
            variants_for(args.stage)
        ):
            raise ValueError("Variant does not belong to this experiment stage")
        for fi, fold in enumerate(args.folds):
            for si, seed in enumerate(args.seeds):
                device = args.devices[(fi * len(args.seeds) + si) % len(args.devices)]
                for variant in variants:
                    output = args.run_root / args.stage / variant / f"fold_{fold}" / f"seed_{seed}"
                    command = [
                        args.root / "scripts/train_pitch3_lte_v39b.py",
                        *common,
                        "--config",
                        args.config,
                        "--mode",
                        args.stage,
                        "--variant",
                        variant,
                        "--cv-fold",
                        fold,
                        "--seed",
                        seed,
                        "--output-dir",
                        output,
                    ]
                    if args.stage != "budget":
                        command += ["--teacher-manifest", args.teacher_manifest]
                    add(command, output, device)
    elif args.stage == "teacher":
        output = args.teacher_manifest.parent
        command = [
            args.root / "scripts/prepare_pitch3_lte_v39b_teacher.py",
            *common,
            "--root",
            args.root,
            "--output-dir",
            output,
            "--workers",
            args.workers,
        ]
        if args.ace_config:
            command += ["--ace-config", args.ace_config]
        if args.trajectory_manifest:
            command += ["--trajectory-manifest", args.trajectory_manifest]
        add(command, output, args.devices[0], False)
    elif args.stage == "check":
        for i, device in enumerate(args.devices):
            output = args.run_root / "checks" / f"device_{i}"
            add(
                [
                    args.root / "scripts/check_pitch3_lte_v39b.py",
                    "--config",
                    args.config,
                    "--output-dir",
                    output,
                ],
                output,
                device,
            )
    else:
        stage = args.stage.removesuffix("-summary")
        output = args.run_root / "summary"
        add(
            [
                args.root / "scripts/summarize_pitch3_lte_v39b.py",
                "--mode",
                stage,
                "--run-root",
                args.run_root,
                "--dataset-manifest",
                args.dataset_manifest,
                "--folds",
                *args.folds,
                "--seeds",
                *args.seeds,
            ],
            output,
            args.devices[0],
            False,
        )
    return jobs


def verify_output(output, stage):
    if stage in {"budget", "distill-diagnose", "cv"}:
        verify_run(output / "pitch3_lte_v39b_complete.json")
    elif stage == "check":
        from generation.pitch3_lte_v38b_protocol import read_json

        if not read_json(output / "pitch3_lte_v39b_checks.json")["all_checks_passed"]:
            raise ValueError("Synthetic checks incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "check",
            "budget",
            "budget-summary",
            "teacher",
            "distill-diagnose",
            "distill-diagnose-summary",
            "cv",
            "cv-summary",
        ],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--trajectory-manifest", type=Path)
    parser.add_argument("--ace-config", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260941])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--variants", nargs="+", choices=["baseline", "transition", "distill"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gpu-gib", type=float, default=8.0)
    parser.add_argument("--skip-busy-gpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    for name, default in (
        ("run_root", args.root / "runs/pitch3_lte_v39b"),
        ("config", args.root / "configs/pitch3_lte_v39b.toml"),
        (
            "dataset_manifest",
            args.root / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv",
        ),
        ("fingerprint", args.root / "metadata/focus_pitch3_fingerprint_v1.json"),
    ):
        setattr(args, name, (getattr(args, name) or default).resolve())
    args.teacher_manifest = (
        args.teacher_manifest or args.run_root / "teacher/teacher_manifest.json"
    ).resolve()
    if args.teacher_manifest.name != "teacher_manifest.json":
        raise ValueError("Teacher manifest must be named teacher_manifest.json")
    build_jobs(args)
    preflight_devices(args, device_stages=DEVICE_STAGES)
    execute_jobs(build_jobs(args), args, verify_run=verify_output)


if __name__ == "__main__":
    main()
