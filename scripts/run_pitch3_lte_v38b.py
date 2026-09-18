"""Run bounded V3.8-B experiments with one sequential queue per GPU.

No stage automatically chooses a variant or proceeds to development screening.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE_STAGES = {
    "diagnose",
    "cv-control",
    "cv-global",
    "cv-local",
    "train-global",
    "train-local",
    "model-screen",
}


def build_jobs(args):
    if not args.devices or len(args.devices) != len(set(args.devices)):
        raise ValueError("Each GPU queue requires a unique device")
    if len(args.folds) != len(set(args.folds)):
        raise ValueError("Duplicate folds would overwrite the same run")
    root, runs = args.root, args.run_root
    dataset = args.dataset_manifest
    variants = args.global_variants or (
        ["g0", "g1"] if args.stage in {"cv-global", "summary"} else []
    )
    locals_ = args.local_variants or (["l1", "l2"] if args.stage == "cv-local" else ["l0"])
    seeds = args.seeds or (
        [20260941]
        if args.stage.startswith("cv-") or args.stage == "summary"
        else [20260941, 20260942, 20260943]
    )
    if (
        len(set(seeds)) != len(seeds)
        or len(set(variants)) != len(variants)
        or len(set(locals_)) != len(locals_)
    ):
        raise ValueError("Duplicate seed or variant")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative")
    if (
        args.stage in {"cv-local", "train-global", "train-local", "ensemble", "model-screen"}
        and len(variants) != 1
    ):
        raise ValueError("Explicitly choose one --global-variants value using internal CV evidence")
    if args.stage in {"train-local", "ensemble", "model-screen"} and len(locals_) != 1:
        raise ValueError("Full-training/screening requires one frozen local variant")
    if args.stage in {"cv-local", "train-local"} and any(v == "l0" for v in locals_):
        raise ValueError("L0 is the existing global run; local fitting supports L1/L2")
    python = args.python
    jobs = []

    def add(command, output, model=True):
        jobs.append(
            {
                "command": [python, *map(str, command)],
                "output_dir": str(output),
                "require_empty": model,
            }
        )

    common = ["--fingerprint", args.fingerprint, "--manifest", dataset]
    if args.stage == "diagnose":
        source = args.source_run_root
        paths = sorted(source.glob("seed_*/models/pitch3_lte_manifest.json"))
        paths += sorted(source.glob("cv/fold_*/models/pitch3_lte_manifest.json"))
        if len(paths) != 8:
            raise ValueError("Expected all three V3.8-A seeds and five CV manifests")
        for path in paths:
            output = runs / "diagnostics_v38a" / path.parent.parent.name
            add(
                [
                    root / "scripts/diagnose_pitch3_lte_checkpoint.py",
                    "--model-manifest",
                    path,
                    "--dataset-manifest",
                    dataset,
                    "--fingerprint",
                    args.fingerprint,
                    "--output-dir",
                    output,
                ],
                output,
            )
    elif args.stage == "cv-control":
        if len(seeds) != 1:
            raise ValueError("The legacy V3.7 control uses one explicit seed per output root")
        for fold in args.folds:
            output = runs / "cv_v37_control" / f"fold_{fold}" / "models"
            add(
                [
                    root / "scripts/train_pitch3_lte.py",
                    *common,
                    "--config",
                    root / "configs/pitch3_lte_v37_dte.toml",
                    "--global-only",
                    "--seed",
                    seeds[0],
                    "--cv-fold",
                    fold,
                    "--output-dir",
                    output,
                ],
                output,
            )
    elif args.stage in {"cv-global", "cv-local", "train-global", "train-local"}:
        cv = args.stage.startswith("cv-")
        local = args.stage.endswith("local")
        for variant in variants:
            for lv in locals_ if local else ["l0"]:
                for fold in args.folds if cv else [None]:
                    for seed in seeds:
                        relative = (Path(f"fold_{fold}") if cv else Path()) / f"seed_{seed}/models"
                        base = runs / ("cv" if cv else "final") / variant
                        output = base / lv / relative
                        command = [
                            root / "scripts/train_pitch3_lte_v38b.py",
                            *common,
                            "--config",
                            args.config,
                            "--seed",
                            seed,
                            "--global-variant",
                            variant,
                            "--local-variant",
                            lv,
                            "--output-dir",
                            output,
                        ]
                        if cv:
                            command += ["--cv-fold", fold]
                        if local:
                            command += [
                                "--global-manifest",
                                base / "l0" / relative / "pitch3_lte_manifest.json",
                            ]
                        add(command, output)
    elif args.stage == "summary":
        for variant in variants:
            for lv in locals_:
                output = runs / "cv" / variant / lv
                add(
                    [
                        root / "scripts/summarize_pitch3_lte_v38b.py",
                        "--cv-root",
                        output,
                        "--dataset-manifest",
                        dataset,
                        "--seeds",
                        *seeds,
                    ],
                    output,
                    False,
                )
    elif args.stage in {"ensemble", "model-screen"}:
        if len(seeds) != 3:
            raise ValueError("Final V3.8-B uses three predeclared seeds")
        base = runs / "final" / variants[0] / locals_[0]
        ensemble = base / "pitch3_lte_ensemble.json"
        if args.stage == "ensemble":
            command = [root / "scripts/build_pitch3_lte_ensemble.py", "--output", ensemble]
            for seed in seeds:
                command += ["--manifest", base / f"seed_{seed}/models/pitch3_lte_manifest.json"]
            add(command, base, False)
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
            )
    for index, job in enumerate(jobs):
        job["device"] = args.devices[index % len(args.devices)]
        if args.stage in DEVICE_STAGES:
            job["command"] += ["--device", job["device"]]
    return jobs


def _child_env(root):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _query_cuda_memory(devices, args):
    # Probe in a short-lived child using exactly the training interpreter/env.
    # torch logical indices respect CUDA_VISIBLE_DEVICES (including UUID masks).
    code = """
import json
import sys
import torch

rows = []
for name in sys.argv[1:]:
    device = torch.device(name)
    index = torch.cuda.current_device() if device.index is None else device.index
    free, total = torch.cuda.mem_get_info(index)
    rows.append(dict(device=name, index=index, name=torch.cuda.get_device_name(index),
                     free_bytes=free, total_bytes=total))
print(json.dumps(rows))
"""
    result = subprocess.run(
        [args.python, "-c", code, *devices],
        cwd=args.root,
        env=_child_env(args.root),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(
            f"CUDA memory preflight failed using {args.python}:\n{result.stderr[-16384:]}"
        )
    return json.loads(result.stdout)


def preflight_devices(args, *, device_stages=DEVICE_STAGES):
    """Filter only the requested devices; preserve the full experiment job list."""
    if not math.isfinite(args.min_free_gpu_gib) or args.min_free_gpu_gib <= 0:
        raise ValueError("--min-free-gpu-gib must be finite and positive")
    if args.dry_run or args.stage not in device_stages:
        return
    cuda_devices = [d for d in args.devices if d == "cuda" or d.startswith("cuda:")]
    if not cuda_devices:
        return
    rows = _query_cuda_memory(cuda_devices, args)
    if [row["device"] for row in rows] != cuda_devices:
        raise RuntimeError("CUDA memory probe returned different devices")
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError("CUDA aliases refer to the same GPU; each queue needs a unique device")
    busy = []
    for row in rows:
        free, total = row["free_bytes"] / 2**30, row["total_bytes"] / 2**30
        print(
            f"GPU preflight {row['device']} ({row['name']}): {free:.2f}/{total:.2f} GiB free",
            flush=True,
        )
        if free < args.min_free_gpu_gib:
            busy.append(row["device"])
    if busy and not args.skip_busy_gpus:
        raise RuntimeError(
            f"Insufficient free GPU memory on {', '.join(busy)} "
            f"(minimum {args.min_free_gpu_gib:g} GiB). No experiment started. "
            "Inspect nvidia-smi; choose free --devices or use --skip-busy-gpus "
            "to queue all jobs on the remaining requested devices."
        )
    selected = [device for device in args.devices if device not in busy]
    if not selected:
        raise RuntimeError(
            f"No requested device has {args.min_free_gpu_gib:g} GiB free. "
            "No experiment started; inspect nvidia-smi and rerun when a GPU is available."
        )
    if busy:
        print(
            f"Skipping busy GPUs {', '.join(busy)}; all jobs will use "
            f"{', '.join(selected)} (one sequential queue per device).",
            flush=True,
        )
    args.devices = selected


def _log_tail(path: Path, max_bytes: int = 16384, max_lines: int = 80) -> str:
    """Read a bounded tail even when a training log is large or partly encoded."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError as exc:
        return f"Unable to read log: {exc}"
    return "\n".join(text.splitlines()[-max_lines:]) or "(empty log)"


def execute_jobs(jobs, args, *, verify_run=None):
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "jobs": jobs}, indent=2))
        return
    # Preflight every target before starting any process; never truncate an old log.
    for job in jobs:
        output = Path(job["output_dir"])
        if job["require_empty"] and output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Existing run: {output}; choose a new root or unrun --folds/--seeds"
            )
    queues = [[] for _ in args.devices]
    for job in jobs:
        queues[args.devices.index(job["device"])].append(job)

    def worker(queue):
        for job in queue:
            output = Path(job["output_dir"])
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.parent / f"{output.name}_{args.stage}.log"
            # Reports may be regenerated, but their logs are also kept per invocation.
            suffix = 1
            while log.exists():
                log = output.parent / f"{output.name}_{args.stage}_{suffix}.log"
                suffix += 1
            env = _child_env(args.root)
            with log.open("x", encoding="utf-8") as handle:
                print(f"START {args.stage} {output} [{job['device']}]", flush=True)
                result = subprocess.run(
                    job["command"],
                    cwd=args.root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            if result.returncode:
                raise RuntimeError(
                    f"Experiment failed ({result.returncode}) [{job['device']}]: {output}\n"
                    f"Full log: {log}\n"
                    f"Command: {json.dumps(job['command'], ensure_ascii=False)}\n"
                    f"Last log lines:\n{_log_tail(log)}"
                )
            if verify_run is not None:
                verify_run(output, args.stage)
            elif args.stage in {"cv-global", "cv-local", "train-global", "train-local"}:
                sys.path.insert(0, str(args.root / "src"))
                from generation.pitch3_lte_v38b_protocol import verify_manifest

                verify_manifest(output / "pitch3_lte_manifest.json")
            print(f"DONE {output}", flush=True)

    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = {pool.submit(worker, q): q[0]["device"] for q in queues if q}
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                message = f"Queue [{futures[future]}]: {type(exc).__name__}: {exc}"
                print(message, file=sys.stderr, flush=True)
                failures.append(message)
        if failures:
            raise RuntimeError(
                f"{len(failures)} experiment queue(s) failed:\n\n" + "\n\n".join(failures)
            ) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "diagnose",
            "cv-control",
            "cv-global",
            "cv-local",
            "summary",
            "train-global",
            "train-local",
            "ensemble",
            "model-screen",
        ],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument(
        "--min-free-gpu-gib",
        type=float,
        default=8.0,
        help="Minimum free GPU memory before launch (default: 8 GiB; not a peak-memory estimate)",
    )
    parser.add_argument(
        "--skip-busy-gpus",
        action="store_true",
        help="Queue all jobs on requested GPUs meeting the memory floor; no probe in --dry-run",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--folds", type=int, choices=range(5), nargs="+", default=list(range(5)))
    parser.add_argument("--global-variants", choices=["g0", "g1", "g37"], nargs="+")
    parser.add_argument("--local-variants", choices=["l0", "l1", "l2"], nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.run_root = (args.run_root or args.root / "runs/pitch3_lte_v38b").resolve()
    args.source_run_root = (args.source_run_root or args.root / "runs/pitch3_lte_v38a").resolve()
    args.dataset_manifest = (
        args.dataset_manifest
        or args.root / "runs/pitch3_lte_v3/exact_local_dataset/pitch3_lte_examples.csv"
    ).resolve()
    args.fingerprint = (
        args.fingerprint or args.root / "metadata/focus_pitch3_fingerprint_v1.json"
    ).resolve()
    args.config = (args.config or args.root / "configs/pitch3_lte_v38b.toml").resolve()
    jobs = build_jobs(args)
    preflight_devices(args)
    if jobs and any(job["device"] not in args.devices for job in jobs):
        jobs = build_jobs(args)
    execute_jobs(jobs, args)


if __name__ == "__main__":
    main()
