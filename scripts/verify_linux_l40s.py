from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _memory_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    first = path.read_text(encoding="utf-8").splitlines()[0].split()
    return int(first[1]) / 1024**2


def verify(root: Path, *, allow_missing_data: bool) -> dict[str, Any]:
    root = root.resolve()
    release = tomllib.loads((root / "reproducibility/release_manifest.toml").read_text())
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, observed: Any, expected: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "observed": observed, "expected": expected})

    check("platform", sys.platform.startswith("linux"), sys.platform, "linux")
    check("architecture", platform.machine() == "x86_64", platform.machine(), "x86_64")
    check("python", sys.version_info[:2] == (3, 12), platform.python_version(), "3.12.x")
    check("cpu_threads", (os.cpu_count() or 0) >= 32, os.cpu_count(), ">=32")
    memory = _memory_gib()
    check("memory_gib", memory is not None and memory >= 240, memory, ">=240")

    try:
        import torch

        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        check("cuda_available", torch.cuda.is_available(), torch.cuda.is_available(), True)
        check("gpu_count", len(devices) >= 2, len(devices), ">=2")
        check(
            "gpu_model",
            len(devices) >= 2 and all("L40S" in name.upper() for name in devices[:2]),
            devices,
            "2 x NVIDIA L40S",
        )
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        check("bf16", bf16, bf16, True)
        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda, "devices": devices}
    except ImportError:
        check("torch", False, "not installed", "CUDA-enabled torch")
        torch_info = None

    check("project_git", _git_head(root) is not None, _git_head(root), "recorded commit")
    ace_revision = release["sources"]["ace_step"]["revision"]
    pyglmy_revision = release["sources"]["pyglmy"]["revision"]
    ace_head = _git_head(root / "ACE-Step-1.5")
    pyglmy_head = _git_head(root / "packages/pyglmy")
    check("ace_revision", ace_head == ace_revision, ace_head, ace_revision)
    check("pyglmy_revision", pyglmy_head == pyglmy_revision, pyglmy_head, pyglmy_revision)

    scorer_release_path = root / release["scorer"]["release_manifest"]
    profile_path = root / "metadata/focus_path_homology_fingerprint_v2.json"
    expected_profile_sha256 = release["scorer"]["profile_sha256"]
    check(
        "scorer_release_manifest",
        scorer_release_path.is_file(),
        scorer_release_path.relative_to(root).as_posix()
        if scorer_release_path.is_file()
        else "missing",
        release["scorer"]["release_manifest"],
    )
    check(
        "scorer_profile",
        profile_path.is_file(),
        profile_path.relative_to(root).as_posix() if profile_path.is_file() else "missing",
        "metadata/focus_path_homology_fingerprint_v2.json",
    )

    scorer_payload: dict[str, Any] | None = None
    if scorer_release_path.is_file():
        try:
            loaded = json.loads(scorer_release_path.read_text(encoding="utf-8"))
            scorer_payload = loaded if isinstance(loaded, dict) else None
            check(
                "scorer_release_json",
                scorer_payload is not None,
                type(loaded).__name__,
                "JSON object",
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            check("scorer_release_json", False, f"{type(error).__name__}: {error}", "valid JSON")

    if scorer_payload is not None:
        dimensions = scorer_payload.get("dimensions")
        legacy_status = scorer_payload.get("runtime_status", {}).get("legacy_51d_scorer")
        check("scorer_dimensions", dimensions == 18, dimensions, 18)
        check("legacy_51d", legacy_status == "reject", legacy_status, "reject")

    if profile_path.is_file():
        profile_sha256 = _sha256(profile_path)
        check(
            "scorer_profile_sha256",
            profile_sha256 == expected_profile_sha256,
            profile_sha256,
            expected_profile_sha256,
        )

    dataset_receipt = root / "runs/reproducibility/dataset.json"
    if dataset_receipt.is_file():
        dataset = json.loads(dataset_receipt.read_text(encoding="utf-8"))
        dataset_status = dataset.get("status")
        audio_files = dataset.get("audio_files")
        sums_hash = dataset.get("sha256s_sha256")
        expected_sums_hash = release["sources"]["dataset"]["sha256s_sha256"]
        check("dataset_status", dataset_status == "verified", dataset_status, "verified")
        check("dataset_audio_files", audio_files == 600, audio_files, 600)
        check("dataset_sha256s", sums_hash == expected_sums_hash, sums_hash, expected_sums_hash)
    else:
        check("dataset_receipt", allow_missing_data, "missing", "runs/reproducibility/dataset.json")

    return {
        "schema_version": 1,
        "status": "passed" if all(item["ok"] for item in checks) else "failed",
        "root": str(root),
        "torch": torch_info,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-missing-data", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/reproducibility/preflight.json"))
    args = parser.parse_args(argv)
    payload = verify(args.root, allow_missing_data=args.allow_missing_data)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
