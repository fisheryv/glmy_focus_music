from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from data.preprocess import (
    AudioPreprocessConfig,
    SegmentPlan,
    choose_segment_window,
    process_plan,
    read_preprocess_manifest,
    write_manifest,
)
from features.batch import (
    SegmentJob,
    extract_batch,
)
from features.batch import (
    _load_config as load_feature_config,
)
from repetition.analysis import (
    _compute_segment as compute_repetition_segment,
)
from repetition.analysis import (
    _load_model as load_repetition_model,
)
from repetition.analysis import (
    load_config as load_repetition_config,
)
from tda.analysis import (
    _compute_segment as compute_tda_segment,
)
from tda.analysis import (
    _load_model as load_tda_model,
)
from tda.analysis import (
    load_config as load_tda_config,
)

from .experiment import CandidateRecord
from .target_profile import CORE_FEATURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preprocess_config(project_root: Path, scale_seconds: float) -> AudioPreprocessConfig:
    with (project_root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle).get("preprocessing", {})
    return AudioPreprocessConfig(
        sample_rate=int(raw.get("analysis_sample_rate", 22_050)),
        channels=int(raw.get("analysis_channels", 1)),
        target_lufs=float(raw.get("target_lufs", -15.0)),
        peak_ceiling_dbfs=float(raw.get("peak_ceiling_dbfs", -1.0)),
        scales_seconds=(float(scale_seconds),),
    )


def _copy_frozen_model(project_root: Path, run_root: Path) -> None:
    source_dir = project_root / "features" / "models"
    target_dir = run_root / "features" / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("state_model.npz", "state_model.json"):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"frozen state model is missing: {source}")
        target = target_dir / name
        if target.is_file():
            if _sha256(source) != _sha256(target):
                raise ValueError(
                    "Frozen state model differs from the current source. "
                    f"Start a new run_id instead of mutating {target}."
                )
            continue
        shutil.copy2(source, target)


def _preprocess_one(
    record: CandidateRecord,
    *,
    run_root: Path,
    config: AudioPreprocessConfig,
    previous: dict[str, str] | None,
) -> dict[str, Any]:
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError("exact scoring requires soundfile; install .[audio,stats,tda]") from exc
    raw_path = run_root / record.audio_relative_path
    info = soundfile.info(str(raw_path))
    if info.duration + 0.5 < record.duration_seconds:
        raise RuntimeError(
            f"candidate audio is shorter than requested: {info.duration:.3f}s "
            f"< {record.duration_seconds:.3f}s ({raw_path})"
        )
    start, duration, used_full = choose_segment_window(info.duration, record.duration_seconds)
    source_relative = raw_path.relative_to(run_root / "data_raw").as_posix()
    token = f"{int(record.duration_seconds)}s"
    plan = SegmentPlan(
        segment_id=record.candidate_id,
        track_id=record.candidate_id,
        group="generated",
        split="experiment",
        scale_seconds=record.duration_seconds,
        source_relative_path=source_relative,
        output_relative_path=(
            f"features/audio/{token}/generated/experiment/{record.candidate_id}.wav"
        ),
        source_duration_seconds=float(info.duration),
        start_seconds=start,
        requested_duration_seconds=duration,
        used_full_track=used_full,
    )
    return process_plan(plan, root=run_root, config=config, previous_row=previous)


def preprocess_candidates(
    project_root: Path,
    run_root: Path,
    records: list[CandidateRecord],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    generated = [record for record in records if record.status in {"generated", "scored"}]
    if not generated:
        raise ValueError("no generated candidates are available for exact scoring")
    scale_values = {record.duration_seconds for record in generated}
    if len(scale_values) != 1:
        raise ValueError("all candidates in one run must have the same duration")
    config = _preprocess_config(project_root, scale_values.pop())
    manifest_path = run_root / "manifests" / "preprocessed_candidates.csv"
    previous = read_preprocess_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _preprocess_one,
                record,
                run_root=run_root,
                config=config,
                previous=previous.get(record.candidate_id),
            ): record
            for record in generated
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "segment_id": record.candidate_id,
                        "track_id": record.candidate_id,
                        "group": "generated",
                        "split": "experiment",
                        "scale_seconds": record.duration_seconds,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    write_manifest(manifest_path, rows)
    failures = [row for row in rows if row.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"preprocessing failed for {len(failures)} candidate(s)")
    return rows


def extract_candidate_features(
    project_root: Path,
    run_root: Path,
    processed_rows: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        SegmentJob(
            segment_id=str(row["segment_id"]),
            track_id=str(row["track_id"]),
            group="generated",
            split="experiment",
            scale_seconds=float(row["scale_seconds"]),
            input_relative_path=str(row["output_relative_path"]),
            input_sha256=str(row["sha256"]),
        )
        for row in processed_rows
    ]
    rows = extract_batch(
        jobs,
        root=run_root,
        config=load_feature_config(project_root),
        workers=workers,
        overwrite=False,
        manifest_path=run_root / "manifests" / "feature_candidates.csv",
    )
    failures = [row for row in rows if row.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"feature extraction failed for {len(failures)} candidate(s)")
    return rows


def _technical_audio_metrics(path: Path) -> dict[str, float]:
    import soundfile

    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    absolute = np.abs(audio)
    return {
        "raw_sample_rate": float(sample_rate),
        "raw_duration_seconds": float(audio.shape[0] / sample_rate),
        "raw_peak": float(np.max(absolute, initial=0.0)),
        "raw_rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "raw_clip_fraction": float(np.mean(absolute >= 0.999)),
        "raw_dc_offset": float(np.max(np.abs(np.mean(audio, axis=0)), initial=0.0)),
    }


def compute_exact_descriptors(
    project_root: Path,
    run_root: Path,
    records: list[CandidateRecord],
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _copy_frozen_model(project_root, run_root)
    tda_model = load_tda_model(run_root)
    repetition_model = load_repetition_model(run_root)
    tda_config = load_tda_config(project_root)
    repetition_config = load_repetition_config(project_root)
    record_by_id = {record.candidate_id: record for record in records}
    output: list[dict[str, Any]] = []
    for row in feature_rows:
        tda_rows = compute_tda_segment(
            run_root,
            row,
            tda_model,
            tda_config,
            ("acoustic_novelty_delay", "rhythm"),
        )
        repetition_rows = compute_repetition_segment(
            run_root,
            row,
            repetition_model,
            repetition_config,
            ("path_acoustic_phase", "path_rhythm_phase"),
            False,
        )
        tda_lookup = {item["representation"]: item for item in tda_rows}
        repetition_lookup = {item["representation"]: item for item in repetition_rows}
        descriptor = {
            CORE_FEATURES[0]: tda_lookup["acoustic_novelty_delay"]["h0_max_persistence"],
            CORE_FEATURES[1]: tda_lookup["rhythm"]["h0_total_persistence"],
            CORE_FEATURES[2]: repetition_lookup["path_acoustic_phase"]["loop_score"],
            CORE_FEATURES[3]: repetition_lookup["path_rhythm_phase"]["loop_score"],
        }
        record = record_by_id[str(row["segment_id"])]
        technical = _technical_audio_metrics(run_root / record.audio_relative_path)
        output.append(
            {
                "candidate_id": record.candidate_id,
                "prompt_id": record.prompt_id,
                "candidate_index": record.candidate_index,
                "seed": record.seed,
                **descriptor,
                **technical,
            }
        )
    return sorted(output, key=lambda item: (item["prompt_id"], item["candidate_index"]))


def write_descriptor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty descriptor table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
