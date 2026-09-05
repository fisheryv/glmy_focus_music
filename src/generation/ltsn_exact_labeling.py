"""Storage-bounded exact descriptor extraction for collected snapshot audio."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from features.batch import _read_npz
from features.pitch_v2 import assign_codebook, chroma_to_tonnetz
from graphs.transition import build_transition_graph
from homology.glmy import persistent_path_homology
from topology.batch import _graph_metrics, _topology_metrics, load_topology_config
from topology.metrics import TOPOLOGY_METRICS

from .exact_features import (
    _copy_frozen_model,
    _technical_audio_metrics,
    extract_candidate_features,
    preprocess_candidates,
)
from .experiment import CandidateRecord
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import (
    canonical_json_sha256,
    read_trajectory_manifest,
    write_csv_atomic,
    write_json_atomic,
)
from .ltsn_storage import MATERIALIZE_MODES, materialize_file, remove_tree_within

BATCH_RECEIPT_SCHEMA_VERSION = 1
EXACT_DESCRIPTOR_EXECUTOR = "process_pool_spawn"
EXACT_WORKER_BLAS_THREADS = 1
_EXACT_WORKER_STATE: dict[str, Any] | None = None
_EXACT_THREADPOOL_LIMITER: Any | None = None


def _initialize_exact_descriptor_worker(
    project_root: str,
    work_dir: str,
    centers: np.ndarray,
    codebook_sha256: str,
) -> None:
    """Load immutable exact-label state once and clamp native thread pools."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(EXACT_WORKER_BLAS_THREADS)
    from threadpoolctl import threadpool_limits

    from repetition.analysis import _load_model as load_repetition_model
    from repetition.analysis import load_config as load_repetition_config

    global _EXACT_THREADPOOL_LIMITER, _EXACT_WORKER_STATE
    _EXACT_THREADPOOL_LIMITER = threadpool_limits(limits=EXACT_WORKER_BLAS_THREADS)
    project_path = Path(project_root)
    work_path = Path(work_dir)
    _EXACT_WORKER_STATE = {
        "project_root": project_path,
        "work_dir": work_path,
        "centers": np.asarray(centers, dtype=np.float64),
        "codebook_sha256": codebook_sha256,
        "repetition_model": load_repetition_model(work_path),
        "repetition_config": load_repetition_config(project_path),
        "topology_config": load_topology_config(project_path),
    }


def _copy_snapshot_audio(
    trajectory_manifest: Path,
    work_dir: Path,
    rows: list[dict[str, str]],
    *,
    materialize_mode: str,
) -> tuple[list[CandidateRecord], Counter[str]]:
    records: list[CandidateRecord] = []
    methods: Counter[str] = Counter()
    for row in rows:
        relative = row.get("audio_path", "")
        source = (trajectory_manifest.parent / relative).resolve()
        if not relative or not source.is_file():
            raise LTSNContractError(f"snapshot has not been VAE-decoded: {row['sample_id']}")
        expected = row.get("audio_sha256", "")
        if not expected:
            raise LTSNContractError(f"snapshot has no signed audio SHA-256: {row['sample_id']}")
        target = work_dir / "data_raw" / "snapshots" / f"{row['sample_id']}.wav"
        method = materialize_file(
            source,
            target,
            mode=materialize_mode,
            expected_sha256=expected,
        )
        methods[method] += 1
        duration = float(_technical_audio_metrics(target)["raw_duration_seconds"])
        if duration + 0.5 < 180.0:
            raise LTSNContractError(
                f"exact LTSN teacher requires a 180 s snapshot audio: {source} ({duration:.3f}s)"
            )
        records.append(
            CandidateRecord(
                experiment_id="ltsn_exact_labels",
                prompt_id=row["prompt_id"],
                caption="LTSN snapshot exact teacher",
                candidate_index=int(row["step_number"]),
                candidate_id=row["sample_id"],
                seed=0,
                duration_seconds=180.0,
                status="generated",
                audio_relative_path=target.relative_to(work_dir).as_posix(),
                audio_sha256=expected,
            )
        )
    return records, methods


def _pitch_descriptors(
    project_root: Path,
    work_dir: Path,
    feature_row: dict[str, Any],
    centers: np.ndarray,
    topology_config: Any | None = None,
) -> list[float]:
    arrays = _read_npz(work_dir / feature_row["chroma_relative_path"])
    chroma = np.asarray(arrays["chroma"], dtype=np.float64)
    tonnetz = chroma_to_tonnetz(chroma)
    valid = np.asarray(arrays["valid"], dtype=bool)
    valid &= np.all(np.isfinite(tonnetz), axis=1) & (np.sum(chroma, axis=1) > 1e-8)
    raw_states = assign_codebook(tonnetz, centers, valid=valid)
    states = [int(value) if value >= 0 else None for value in raw_states]
    config = topology_config or load_topology_config(project_root)
    graph = build_transition_graph(
        states,
        normalize=True,
        top_k=config.top_k,
        include_self_loops=config.include_self_loops,
    )
    persistence = persistent_path_homology(
        graph, config.thresholds, tolerance=config.rank_tolerance
    )
    metrics = {**_graph_metrics(states, graph), **_topology_metrics(persistence)}
    return [float(metrics[name]) for name in TOPOLOGY_METRICS]


def _compute_exact_descriptor(
    task: tuple[dict[str, Any], dict[str, str], str],
) -> dict[str, Any]:
    """Compute one snapshot descriptor inside an isolated CPU worker."""

    if _EXACT_WORKER_STATE is None:
        raise RuntimeError("exact descriptor worker was not initialized")
    from repetition.analysis import _compute_segment as compute_repetition_segment

    feature_row, trajectory_row, audio_relative_path = task
    state = _EXACT_WORKER_STATE
    work_dir = state["work_dir"]
    sample_id = str(feature_row["segment_id"])
    phase_rows = compute_repetition_segment(
        work_dir,
        feature_row,
        state["repetition_model"],
        state["repetition_config"],
        ("path_acoustic_phase", "path_chroma_phase"),
        False,
    )
    phase = {row["representation"]: row for row in phase_rows}
    technical = _technical_audio_metrics(work_dir / audio_relative_path)
    ood = technical["raw_rms"] < 1e-4 or technical["raw_clip_fraction"] > 0.01
    return {
        "sample_id": sample_id,
        "pitch_descriptors_json": json.dumps(
            _pitch_descriptors(
                state["project_root"],
                work_dir,
                feature_row,
                state["centers"],
                state["topology_config"],
            ),
            separators=(",", ":"),
        ),
        "acoustic_loop_score": float(phase["path_acoustic_phase"]["loop_score"]),
        "chroma_loop_score": float(phase["path_chroma_phase"]["loop_score"]),
        "ood_label": int(ood),
        "label_source": "decoded_snapshot_exact_v1",
        "audio_sha256": trajectory_row["audio_sha256"],
        "pitch_v2_codebook_sha256": state["codebook_sha256"],
    }


def _batch_input_sha256(rows: list[dict[str, str]]) -> str:
    return canonical_json_sha256(
        [
            {
                "sample_id": row["sample_id"],
                "audio_sha256": row.get("audio_sha256", ""),
                "latent_sha256": row["latent_sha256"],
            }
            for row in rows
        ]
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_resumable_batch(
    *,
    descriptor_path: Path,
    receipt_path: Path,
    trajectory_manifest_sha256: str,
    batch_input_sha256: str,
    expected_sample_ids: list[str],
) -> tuple[list[dict[str, str]], Counter[str]] | None:
    if not descriptor_path.exists() and not receipt_path.exists():
        return None
    if not descriptor_path.is_file() or not receipt_path.is_file():
        raise LTSNContractError(
            f"incomplete exact-label batch checkpoint: {descriptor_path.parent}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": BATCH_RECEIPT_SCHEMA_VERSION,
        "trajectory_manifest_sha256": trajectory_manifest_sha256,
        "batch_input_sha256": batch_input_sha256,
        "descriptor_table_sha256": sha256_file(descriptor_path),
        "samples": len(expected_sample_ids),
    }
    for name, expected in required.items():
        if receipt.get(name) != expected:
            raise LTSNContractError(
                f"exact-label batch checkpoint mismatch for {name}: {descriptor_path}"
            )
    rows = _read_csv(descriptor_path)
    if sorted(row["sample_id"] for row in rows) != sorted(expected_sample_ids):
        raise LTSNContractError(f"exact-label batch sample identities changed: {descriptor_path}")
    methods = Counter(
        {str(name): int(count) for name, count in receipt["materialization_counts"].items()}
    )
    return rows, methods


def _extract_batch(
    *,
    project_root: Path,
    trajectory_manifest: Path,
    work_dir: Path,
    rows: list[dict[str, str]],
    workers: int,
    materialize_mode: str,
    codebook_path: Path,
    centers: np.ndarray,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records, methods = _copy_snapshot_audio(
        trajectory_manifest,
        work_dir,
        rows,
        materialize_mode=materialize_mode,
    )
    processed = preprocess_candidates(project_root, work_dir, records, workers=workers)
    feature_rows = extract_candidate_features(
        project_root, work_dir, processed, workers=workers
    )
    _copy_frozen_model(project_root, work_dir)
    trajectory_by_id = {row["sample_id"]: row for row in rows}
    record_by_id = {record.candidate_id: record for record in records}
    feature_ids = [str(row["segment_id"]) for row in feature_rows]
    expected_ids = sorted(trajectory_by_id)
    if sorted(feature_ids) != expected_ids or len(set(feature_ids)) != len(feature_ids):
        raise LTSNContractError("exact feature rows do not match the batch trajectory IDs")
    tasks = [
        (
            feature_row,
            trajectory_by_id[str(feature_row["segment_id"])],
            record_by_id[str(feature_row["segment_id"])].audio_relative_path,
        )
        for feature_row in feature_rows
    ]
    worker_count = min(workers, len(tasks))
    codebook_sha256 = sha256_file(codebook_path)
    output: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_exact_descriptor_worker,
        initargs=(
            str(project_root),
            str(work_dir),
            centers,
            codebook_sha256,
        ),
    ) as executor:
        futures = {
            executor.submit(_compute_exact_descriptor, task): str(task[0]["segment_id"])
            for task in tasks
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                output.append(future.result())
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"exact Path Homology failed for snapshot {sample_id}: {error}"
                ) from error
    output.sort(key=lambda item: item["sample_id"])
    if [str(item["sample_id"]) for item in output] != expected_ids:
        raise LTSNContractError("parallel exact descriptors do not match the batch plan")
    return output, methods


def build_exact_snapshot_descriptors(
    *,
    project_root: Path,
    trajectory_manifest: Path,
    work_dir: Path,
    output_path: Path,
    workers: int = 2,
    batch_size: int = 256,
    materialize_mode: str = "auto",
    cleanup_batches: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Run exact extraction in resumable, storage-bounded batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if materialize_mode not in MATERIALIZE_MODES:
        raise ValueError(f"unsupported materialization mode: {materialize_mode}")
    rows = sorted(read_trajectory_manifest(trajectory_manifest), key=lambda row: row["sample_id"])
    manifest_sha256 = sha256_file(trajectory_manifest)
    codebook_path = project_root / "features" / "models" / "pitch_v2_codebook.npz"
    if not codebook_path.is_file():
        raise FileNotFoundError(codebook_path)
    with np.load(codebook_path, allow_pickle=False) as archive:
        centers = np.asarray(archive["centers"], dtype=np.float64)

    work_dir.mkdir(parents=True, exist_ok=True)
    batches_root = work_dir / "batches"
    checkpoints_root = work_dir / "batch_descriptors"
    output: list[dict[str, Any] | dict[str, str]] = []
    materialization_counts: Counter[str] = Counter()
    resumed_batches = 0
    batch_count = (len(rows) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        batch_rows = rows[start : start + batch_size]
        batch_id = f"batch_{batch_index:05d}"
        batch_work_dir = batches_root / batch_id
        descriptor_path = checkpoints_root / f"{batch_id}.csv"
        receipt_path = checkpoints_root / f"{batch_id}.json"
        input_sha256 = _batch_input_sha256(batch_rows)
        resumed = None
        if resume:
            resumed = _load_resumable_batch(
                descriptor_path=descriptor_path,
                receipt_path=receipt_path,
                trajectory_manifest_sha256=manifest_sha256,
                batch_input_sha256=input_sha256,
                expected_sample_ids=[row["sample_id"] for row in batch_rows],
            )
        else:
            descriptor_path.unlink(missing_ok=True)
            receipt_path.unlink(missing_ok=True)
        if resumed is not None:
            descriptor_rows, methods = resumed
            resumed_batches += 1
        else:
            remove_tree_within(batch_work_dir, work_dir)
            descriptor_rows, methods = _extract_batch(
                project_root=project_root,
                trajectory_manifest=trajectory_manifest,
                work_dir=batch_work_dir,
                rows=batch_rows,
                workers=workers,
                materialize_mode=materialize_mode,
                codebook_path=codebook_path,
                centers=centers,
            )
            write_csv_atomic(descriptor_path, descriptor_rows)
            write_json_atomic(
                receipt_path,
                {
                    "schema_version": BATCH_RECEIPT_SCHEMA_VERSION,
                    "batch_id": batch_id,
                    "samples": len(descriptor_rows),
                    "trajectory_manifest_sha256": manifest_sha256,
                    "batch_input_sha256": input_sha256,
                    "descriptor_table_sha256": sha256_file(descriptor_path),
                    "materialization_counts": dict(sorted(methods.items())),
                    "exact_descriptor_executor": EXACT_DESCRIPTOR_EXECUTOR,
                    "requested_workers": workers,
                    "effective_workers": min(workers, len(batch_rows)),
                    "worker_blas_threads": EXACT_WORKER_BLAS_THREADS,
                },
            )
        output.extend(descriptor_rows)
        materialization_counts.update(methods)
        if cleanup_batches:
            remove_tree_within(batch_work_dir, work_dir)

    output.sort(key=lambda item: item["sample_id"])
    expected_ids = [row["sample_id"] for row in rows]
    output_ids = [str(row["sample_id"]) for row in output]
    if output_ids != expected_ids or len(set(output_ids)) != len(output_ids):
        raise LTSNContractError("merged exact descriptors do not match the trajectory manifest")
    write_csv_atomic(output_path, output)
    storage_report_path = output_path.with_name(f"{output_path.stem}_storage.json")
    report = {
        "schema_version": 1,
        "samples": len(output),
        "batches": batch_count,
        "resumed_batches": resumed_batches,
        "batch_size": batch_size,
        "requested_workers": workers,
        "maximum_effective_workers": min(workers, batch_size, len(rows)),
        "exact_descriptor_executor": EXACT_DESCRIPTOR_EXECUTOR,
        "worker_blas_threads": EXACT_WORKER_BLAS_THREADS,
        "materialize_mode": materialize_mode,
        "materialization_counts": dict(sorted(materialization_counts.items())),
        "cleanup_batches": cleanup_batches,
        "trajectory_manifest_sha256": manifest_sha256,
        "descriptor_table": str(output_path),
        "descriptor_table_sha256": sha256_file(output_path),
        "pitch_v2_codebook_sha256": sha256_file(codebook_path),
    }
    write_json_atomic(storage_report_path, report)
    return {**report, "storage_report": str(storage_report_path)}
