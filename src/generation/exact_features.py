from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    _read_npz,
    extract_batch,
)
from features.batch import (
    _load_config as load_feature_config,
)
from features.pitch_v2 import assign_codebook, chroma_to_tonnetz
from graphs.transition import build_transition_graph
from homology.glmy import persistent_path_homology
from topology.batch import _graph_metrics, _topology_metrics, load_topology_config
from topology.metrics import TOPOLOGY_METRICS

from .experiment import CandidateRecord

if TYPE_CHECKING:
    from .path_homology_exact_scorer import ExactPathHomologyScorer


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


def _pitch_path_homology_descriptors(
    project_root: Path,
    run_root: Path,
    feature_row: dict[str, Any],
    centers: np.ndarray,
) -> list[float]:
    arrays = _read_npz(run_root / feature_row["chroma_relative_path"])
    chroma = np.asarray(arrays["chroma"], dtype=np.float64)
    tonnetz = chroma_to_tonnetz(chroma)
    valid = np.asarray(arrays["valid"], dtype=bool)
    valid &= np.all(np.isfinite(tonnetz), axis=1) & (np.sum(chroma, axis=1) > 1e-8)
    raw_states = assign_codebook(tonnetz, centers, valid=valid)
    states = [int(value) if value >= 0 else None for value in raw_states]
    config = load_topology_config(project_root)
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


def compute_frozen_18d_descriptors(
    project_root: Path,
    run_root: Path,
    records: list[CandidateRecord],
    feature_rows: list[dict[str, Any]],
    scorer: ExactPathHomologyScorer,
) -> list[dict[str, Any]]:
    """Compute the signed Pitch/Acoustic/Chroma teacher for final candidate audio."""

    import json

    from repetition.analysis import _compute_segment as compute_repetition_segment
    from repetition.analysis import _load_model as load_repetition_model
    from repetition.analysis import load_config as load_repetition_config

    codebook_path = project_root / "features" / "models" / "pitch_v2_codebook.npz"
    if not codebook_path.is_file():
        raise FileNotFoundError(codebook_path)
    with np.load(codebook_path, allow_pickle=False) as archive:
        centers = np.asarray(archive["centers"], dtype=np.float64)
    _copy_frozen_model(project_root, run_root)
    repetition_model = load_repetition_model(run_root)
    repetition_config = load_repetition_config(project_root)
    record_by_id = {record.candidate_id: record for record in records}
    output: list[dict[str, Any]] = []
    for feature_row in feature_rows:
        candidate_id = str(feature_row["segment_id"])
        if candidate_id not in record_by_id:
            raise ValueError(f"feature row has no candidate record: {candidate_id}")
        phase_rows = compute_repetition_segment(
            run_root,
            feature_row,
            repetition_model,
            repetition_config,
            ("path_acoustic_phase", "path_chroma_phase"),
            False,
        )
        phase = {row["representation"]: row for row in phase_rows}
        pitch = _pitch_path_homology_descriptors(
            project_root, run_root, feature_row, centers
        )
        acoustic = float(phase["path_acoustic_phase"]["loop_score"])
        chroma = float(phase["path_chroma_phase"]["loop_score"])
        score = scorer.score(pitch, [acoustic], [chroma])
        record = record_by_id[candidate_id]
        audio_path = run_root / record.audio_relative_path
        if _sha256(audio_path) != record.audio_sha256:
            raise ValueError(f"candidate audio hash changed before exact scoring: {candidate_id}")
        technical = _technical_audio_metrics(audio_path)
        output.append(
            {
                "candidate_id": candidate_id,
                "prompt_id": record.prompt_id,
                "candidate_index": record.candidate_index,
                "seed": record.seed,
                "audio_sha256": record.audio_sha256,
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "feature_order_json": json.dumps(
                    list(scorer.contract.feature_order), separators=(",", ":")
                ),
                "pitch_descriptors_json": json.dumps(pitch, separators=(",", ":")),
                "acoustic_loop_score": acoustic,
                "chroma_loop_score": chroma,
                "coordinates_json": json.dumps(
                    score.coordinates[0].tolist(), separators=(",", ":")
                ),
                "focus_logit": float(score.focus_logit[0]),
                "focus_probability": float(score.focus_probability[0]),
                "focus_band_loss": float(score.focus_band_loss[0]),
                "pitch_block_l2_norm": float(score.pitch_block_l2_norm[0]),
                "phase_block_l2_norm": float(score.phase_block_l2_norm[0]),
                "pitch_v2_codebook_sha256": _sha256(codebook_path),
                "label_source": "decoded_candidate_exact_18d_v1",
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
