from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_pitch3_lte_v38b as runner
from scripts.run_pitch3_lte_v38b import _log_tail, execute_jobs


def test_runner_exposes_all_child_tracebacks_and_preserves_logs(tmp_path, capsys):
    args = SimpleNamespace(
        dry_run=False, devices=["cuda:1", "cuda:2"], stage="cv-global", root=tmp_path
    )
    jobs = []
    for fold, device in enumerate(args.devices):
        output = tmp_path / f"fold_{fold}" / "models"
        output.parent.mkdir()
        (output.parent / "models_cv-global.log").write_text("previous attempt", encoding="utf-8")
        jobs.append(
            {
                "command": [
                    sys.executable,
                    "-c",
                    f"raise ValueError('child failure in fold {fold}')",
                ],
                "output_dir": str(output),
                "device": device,
                "require_empty": True,
            }
        )
    with pytest.raises(RuntimeError, match=r"2 experiment queue\(s\) failed") as raised:
        execute_jobs(jobs, args)
    terminal = capsys.readouterr().err
    for fold, job in enumerate(jobs):
        output = Path(job["output_dir"])
        log = output.parent / "models_cv-global_1.log"
        assert (output.parent / "models_cv-global.log").read_text() == "previous attempt"
        assert f"ValueError: child failure in fold {fold}" in log.read_text()
        assert f"ValueError: child failure in fold {fold}" in str(raised.value)
        assert f"ValueError: child failure in fold {fold}" in terminal
        assert str(log) in str(raised.value)
        assert job["device"] in str(raised.value)


def test_log_tail_is_bounded_and_handles_invalid_utf8(tmp_path):
    log = tmp_path / "training.log"
    log.write_bytes(b"old output\n" * 10000 + b"invalid: \xff\nRuntimeError: final cause\n")
    tail = _log_tail(log)
    assert tail.endswith("RuntimeError: final cause")
    assert "invalid: \ufffd" in tail
    assert len(tail.splitlines()) == 80
    # A single long line must also be bounded, regardless of line count.
    log.write_bytes(b"x" * 100000 + b"final cause")
    tail = _log_tail(log)
    assert len(tail) == 16384 and tail.endswith("final cause")
    log.write_bytes(b"")
    assert _log_tail(log) == "(empty log)"
    assert _log_tail(tmp_path / "absent.log").startswith("Unable to read log:")


def test_runner_preflights_all_outputs_before_starting_any_child(tmp_path, monkeypatch):
    args = SimpleNamespace(dry_run=False, devices=["cpu"], stage="cv-global", root=tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    artifact = existing / "checkpoint.pt"
    artifact.write_bytes(b"original checkpoint")
    jobs = [
        {"output_dir": str(p), "require_empty": True, "device": "cpu", "command": []}
        for p in [tmp_path / "fresh", existing]
    ]

    def unexpected_process(*_args, **_kwargs):
        pytest.fail("A child was launched before every output passed preflight")

    monkeypatch.setattr("scripts.run_pitch3_lte_v38b.subprocess.run", unexpected_process)
    with pytest.raises(FileExistsError, match="Existing run"):
        execute_jobs(jobs, args)
    assert artifact.read_bytes() == b"original checkpoint"
    assert not (tmp_path / "fresh").exists()


def test_runner_success_still_finishes_without_false_failure(tmp_path):
    args = SimpleNamespace(dry_run=False, devices=["cpu"], stage="summary", root=tmp_path)
    output = tmp_path / "report"
    execute_jobs(
        [
            {
                "output_dir": str(output),
                "require_empty": False,
                "device": "cpu",
                "command": [sys.executable, "-c", "print('completed report')"],
            }
        ],
        args,
    )
    assert (tmp_path / "report_summary.log").read_text().strip() == "completed report"


def _gpu_args(tmp_path, **overrides):
    fields = dict(
        root=tmp_path,
        run_root=tmp_path / "runs",
        dataset_manifest=tmp_path / "dataset.csv",
        fingerprint=tmp_path / "fingerprint.json",
        config=tmp_path / "config.toml",
        python=sys.executable,
        stage="cv-global",
        global_variants=None,
        local_variants=None,
        seeds=None,
        folds=list(range(5)),
        devices=["cuda:1", "cuda:2", "cuda:3"],
        dry_run=False,
        min_free_gpu_gib=8.0,
        skip_busy_gpus=False,
    )
    return SimpleNamespace(**(fields | overrides))


def _memory_rows(devices, free_gib):
    return [
        dict(
            device=device,
            index=int(device.split(":")[1]),
            name="GPU fixture",
            free_bytes=int(free * 2**30),
            total_bytes=int(44.52 * 2**30),
        )
        for device, free in zip(devices, free_gib, strict=True)
    ]


def test_gpu_preflight_skips_occupied_card_without_dropping_or_changing_experiments(
    tmp_path, monkeypatch
):
    args = _gpu_args(tmp_path, skip_busy_gpus=True)
    original = runner.build_jobs(args)
    # The reported external 41.21 GiB allocation leaves about 3.31 GiB before LTE starts.
    monkeypatch.setattr(
        runner, "_query_cuda_memory", lambda devices, _args: _memory_rows(devices, [40, 3.31, 40])
    )
    runner.preflight_devices(args)
    assert args.devices == ["cuda:1", "cuda:3"]
    revised = runner.build_jobs(args)
    assert len(revised) == len(original) == 10
    for old, new in zip(original, revised, strict=True):
        assert new["command"][:-2] == old["command"][:-2]
        assert new["output_dir"] == old["output_dir"]
        assert new["command"][-2:] == ["--device", new["device"]]
    assert [job["device"] for job in revised] == ["cuda:1", "cuda:3"] * 5
    assert not args.run_root.exists()


@pytest.mark.parametrize("skip_busy, free_gib", [(False, [40, 3.31, 40]), (True, [2, 3.31, 1])])
def test_gpu_preflight_refuses_insufficient_memory_before_any_experiment(
    tmp_path, monkeypatch, skip_busy, free_gib
):
    args = _gpu_args(tmp_path, skip_busy_gpus=skip_busy)
    monkeypatch.setattr(
        runner, "_query_cuda_memory", lambda devices, _args: _memory_rows(devices, free_gib)
    )
    with pytest.raises(RuntimeError, match="No experiment started"):
        runner.preflight_devices(args)
    assert args.devices == ["cuda:1", "cuda:2", "cuda:3"]
    assert not args.run_root.exists()


@pytest.mark.parametrize(
    "overrides", [{"dry_run": True}, {"stage": "summary"}, {"devices": ["cpu"]}]
)
def test_gpu_preflight_does_not_import_torch_for_dry_run_reports_or_cpu(
    tmp_path, monkeypatch, overrides
):
    def unexpected_probe(*_args):
        pytest.fail("This invocation must remain Torch-free")

    monkeypatch.setattr(runner, "_query_cuda_memory", unexpected_probe)
    runner.preflight_devices(_gpu_args(tmp_path, **overrides))


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), float("inf")])
def test_gpu_preflight_rejects_invalid_memory_floor(tmp_path, minimum):
    with pytest.raises(ValueError, match="finite and positive"):
        runner.preflight_devices(_gpu_args(tmp_path, min_free_gpu_gib=minimum))


def test_gpu_probe_uses_training_python_and_preserves_visible_device_mapping(tmp_path, monkeypatch):
    args = _gpu_args(tmp_path, python="training-python", devices=["cuda:0", "cuda:1"])
    rows = _memory_rows(args.devices, [40, 3.31])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-physical-2,GPU-physical-3")
    calls = []

    def probe(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")

    monkeypatch.setattr(runner.subprocess, "run", probe)
    assert runner._query_cuda_memory(args.devices, args) == rows
    command, options = calls[0]
    assert command[0] == "training-python"
    assert command[-2:] == args.devices
    assert options["cwd"] == tmp_path
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-physical-2,GPU-physical-3"
    assert options["env"]["PYTHONPATH"].startswith(str(tmp_path / "src"))


def test_gpu_preflight_rejects_two_aliases_for_one_gpu(tmp_path, monkeypatch):
    args = _gpu_args(tmp_path, devices=["cuda", "cuda:0"])
    rows = _memory_rows(["cuda:0", "cuda:0"], [40, 40])
    rows[0]["device"] = "cuda"
    monkeypatch.setattr(runner, "_query_cuda_memory", lambda *_args: rows)
    with pytest.raises(ValueError, match="same GPU"):
        runner.preflight_devices(args)


def test_main_rebuilds_gpu_queues_after_memory_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "cv-global", "--root", str(tmp_path), "--skip-busy-gpus"]
    )
    monkeypatch.setattr(
        runner, "_query_cuda_memory", lambda devices, _args: _memory_rows(devices, [40, 3.31, 40])
    )
    launched = []
    monkeypatch.setattr(runner, "execute_jobs", lambda jobs, args: launched.append((jobs, args)))
    runner.main()
    jobs, args = launched[0]
    assert args.devices == ["cuda:1", "cuda:3"]
    assert len(jobs) == 10
    assert [job["device"] for job in jobs] == ["cuda:1", "cuda:3"] * 5
