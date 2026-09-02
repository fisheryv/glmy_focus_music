from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from data.hf_release import DEFAULT_DATASET_ROOT, DatasetReleaseError, verify_release_dataset
from focus_topology.data.manifest import load_split, load_tracks, validate_metadata
from focus_topology.data.schema import SplitName, TrackGroup, TrackRecord

DEFAULT_SCALES = (180.0, 300.0)
MANIFEST_COLUMNS = (
    "segment_id",
    "track_id",
    "group",
    "split",
    "scale_seconds",
    "source_relative_path",
    "output_relative_path",
    "source_duration_seconds",
    "start_seconds",
    "requested_duration_seconds",
    "actual_duration_seconds",
    "sample_rate",
    "channels",
    "dtype",
    "target_lufs",
    "measured_lufs_before",
    "measured_lufs_after",
    "peak_before",
    "peak_after",
    "peak_limited",
    "limiter_gain_db",
    "limited_sample_fraction",
    "used_full_track",
    "padding_applied",
    "sha256",
    "status",
    "processed_at",
    "config_sha256",
    "error",
)


class PreprocessError(RuntimeError):
    pass


class SilenceError(PreprocessError):
    pass


@dataclass(frozen=True, slots=True)
class AudioPreprocessConfig:
    sample_rate: int = 22_050
    channels: int = 1
    target_lufs: float = -15.0
    peak_ceiling_dbfs: float = -1.0
    scales_seconds: tuple[float, ...] = DEFAULT_SCALES
    normalization_backend: str = "ffmpeg_loudnorm_two_pass_soft_limiter"

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 1:
            raise ValueError("the preregistered analysis copy must be mono")
        if not self.scales_seconds or any(value <= 0 for value in self.scales_seconds):
            raise ValueError("all scales must be positive")
        if not -30.0 <= self.target_lufs <= -5.0:
            raise ValueError("target_lufs is outside the supported research range")
        if not -6.0 <= self.peak_ceiling_dbfs <= 0.0:
            raise ValueError("peak_ceiling_dbfs must be between -6 and 0 dBFS")


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    segment_id: str
    track_id: str
    group: str
    split: str
    scale_seconds: float
    source_relative_path: str
    output_relative_path: str
    source_duration_seconds: float
    start_seconds: float
    requested_duration_seconds: float
    used_full_track: bool


def choose_segment_window(
    source_seconds: float, requested_seconds: float
) -> tuple[float, float, bool]:
    """Return a deterministic center crop, or the whole track when it is shorter.

    Center crops reduce the systematic intro/outro bias called out in the research
    report. Short tracks are never looped, padded, or joined to another source.
    """

    if not math.isfinite(source_seconds) or source_seconds <= 0:
        raise ValueError("source duration must be a positive finite value")
    if not math.isfinite(requested_seconds) or requested_seconds <= 0:
        raise ValueError("requested duration must be a positive finite value")
    actual = min(source_seconds, requested_seconds)
    start = max(0.0, (source_seconds - actual) / 2.0)
    return start, actual, source_seconds <= requested_seconds


def fallback_segment_starts(
    source_seconds: float, requested_seconds: float, preferred_start: float
) -> tuple[float, ...]:
    """List content-independent starts used only when a candidate is silent."""

    maximum = max(0.0, source_seconds - requested_seconds)
    candidates = (
        preferred_start,
        max(0.0, preferred_start - requested_seconds),
        min(maximum, preferred_start + requested_seconds),
        0.0,
        maximum,
    )
    result: list[float] = []
    for value in candidates:
        bounded = min(maximum, max(0.0, value))
        if not any(math.isclose(bounded, prior, abs_tol=1e-6) for prior in result):
            result.append(bounded)
    return tuple(result)


def _seconds_token(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else str(seconds).replace(".", "p")


def _config_hash(config: AudioPreprocessConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assignments(metadata_dir: Path) -> dict[str, SplitName]:
    assignments: dict[str, SplitName] = {}
    for split in SplitName:
        members = load_split(metadata_dir / f"split_{split.value}.csv")
        for track_id in members:
            if track_id in assignments:
                raise PreprocessError(f"track {track_id!r} occurs in more than one split")
            assignments[track_id] = split
    return assignments


def build_segment_plans(
    tracks: Iterable[TrackRecord],
    assignments: dict[str, SplitName],
    *,
    scales_seconds: Sequence[float] = DEFAULT_SCALES,
    output_root: Path = Path("features/audio"),
) -> list[SegmentPlan]:
    plans: list[SegmentPlan] = []
    for track in sorted(tracks, key=lambda item: (item.group.value, item.track_id)):
        if track.duration_seconds is None:
            raise PreprocessError(f"track {track.track_id!r} has no audited duration")
        split = assignments.get(track.track_id)
        if split is None:
            raise PreprocessError(f"track {track.track_id!r} has no split assignment")
        for scale in scales_seconds:
            start, _, full_track = choose_segment_window(track.duration_seconds, scale)
            token = _seconds_token(scale)
            segment_id = f"{track.track_id}__{token}s"
            relative = (
                output_root
                / f"{token}s"
                / track.group.value
                / split.value
                / f"{segment_id}.wav"
            )
            plans.append(
                SegmentPlan(
                    segment_id=segment_id,
                    track_id=track.track_id,
                    group=track.group.value,
                    split=split.value,
                    scale_seconds=float(scale),
                    source_relative_path=track.relative_path.as_posix(),
                    output_relative_path=relative.as_posix(),
                    source_duration_seconds=float(track.duration_seconds),
                    start_seconds=start,
                    requested_duration_seconds=min(float(scale), float(track.duration_seconds)),
                    used_full_track=full_track,
                )
            )
    return plans


def _load_optional_audio_modules() -> tuple[Any, Any, Any]:
    try:
        import imageio_ffmpeg
        import pyloudnorm
        import soundfile
    except ImportError as exc:
        raise PreprocessError(
            "audio dependencies are missing; install the project with .[audio]"
        ) from exc
    return imageio_ffmpeg, pyloudnorm, soundfile


def _decode_f32le(
    source: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    ffmpeg_exe: str,
    allow_short: bool = False,
) -> np.ndarray:
    command = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PreprocessError(f"ffmpeg decode failed for {source.name}: {detail}")
    audio = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise PreprocessError(f"decoder returned no audio for {source.name}")
    expected = int(round(duration_seconds * sample_rate))
    # Public catalogs commonly round duration to whole seconds and compressed
    # formats add encoder delay. Two seconds is strict enough to catch a bad
    # seek while accepting those documented container-level differences.
    tolerance = max(2, int(sample_rate * 2.0))
    if audio.size > expected + tolerance:
        audio = audio[:expected]
    if audio.size < expected - tolerance and not allow_short:
        raise PreprocessError(
            f"decoded duration is short for {source.name}: {audio.size / sample_rate:.3f}s "
            f"< {duration_seconds:.3f}s"
        )
    return audio


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float,
    peak_ceiling_dbfs: float,
    pyloudnorm_module: Any,
    ffmpeg_exe: str,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise PreprocessError("audio is empty or contains non-finite samples")
    peak_before = float(np.max(np.abs(values)))
    if peak_before <= 1e-10:
        raise SilenceError("audio is effectively silent")

    meter = pyloudnorm_module.Meter(sample_rate)
    lufs_before = float(meter.integrated_loudness(values.astype(np.float64)))
    if not math.isfinite(lufs_before):
        raise SilenceError("integrated loudness is not finite")

    analysis_filter = (
        f"loudnorm=I={target_lufs}:TP={peak_ceiling_dbfs}:LRA=11:print_format=json"
    )
    common_input = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    analysis_command = [
        *common_input,
        "-af",
        analysis_filter,
        "-f",
        "null",
        "-",
    ]
    analysis = subprocess.run(
        analysis_command,
        input=values.astype("<f4", copy=False).tobytes(),
        check=False,
        capture_output=True,
    )
    if analysis.returncode:
        detail = analysis.stderr.decode("utf-8", errors="replace").strip()
        raise PreprocessError(f"ffmpeg loudnorm analysis failed: {detail}")
    stats = parse_loudnorm_stats(analysis.stderr.decode("utf-8", errors="replace"))
    loudnorm_filter = (
        f"loudnorm=I={target_lufs}:TP={peak_ceiling_dbfs}:LRA=11:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true:print_format=none"
    )
    render_command = [
        *common_input,
        "-loglevel",
        "error",
        "-af",
        loudnorm_filter,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        render_command,
        input=values.astype("<f4", copy=False).tobytes(),
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PreprocessError(f"ffmpeg loudnorm failed: {detail}")
    normalized = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if normalized.size < values.size:
        raise PreprocessError("ffmpeg loudnorm returned truncated audio")
    normalized = normalized[: values.size]
    peak_limit = float(10.0 ** (peak_ceiling_dbfs / 20.0))
    normalized, lufs_after, limiter_gain_db, limited_fraction = calibrate_final_loudness(
        normalized,
        meter=meter,
        target_lufs=target_lufs,
        peak_limit=peak_limit,
    )
    peak_after = float(np.max(np.abs(normalized)))
    return normalized, {
        "measured_lufs_before": lufs_before,
        "measured_lufs_after": lufs_after,
        "peak_before": peak_before,
        "peak_after": peak_after,
        "peak_limited": limited_fraction > 0.0,
        "limiter_gain_db": limiter_gain_db,
        "limited_sample_fraction": limited_fraction,
    }


def parse_loudnorm_stats(stderr: str) -> dict[str, str]:
    matches = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    if not matches:
        raise PreprocessError("ffmpeg loudnorm did not return JSON measurements")
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise PreprocessError("ffmpeg loudnorm returned malformed JSON") from exc
    required = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    missing = required - payload.keys()
    if missing:
        raise PreprocessError(f"ffmpeg loudnorm measurements are missing {sorted(missing)}")
    return {key: str(payload[key]) for key in required}


def soft_peak_limit(audio: np.ndarray, peak_limit: float) -> tuple[np.ndarray, float]:
    """Apply a smooth, deterministic knee below the final sample-peak ceiling."""

    values = np.asarray(audio, dtype=np.float32)
    knee = peak_limit * 0.85
    magnitude = np.abs(values).astype(np.float64)
    affected = magnitude > knee
    if not np.any(affected):
        return values.astype(np.float32, copy=True), 0.0
    width = peak_limit - knee
    limited_magnitude = magnitude.copy()
    limited_magnitude[affected] = knee + width * np.tanh(
        (magnitude[affected] - knee) / width
    )
    result = np.copysign(limited_magnitude, values).astype(np.float32)
    return result, float(np.mean(affected))


def calibrate_final_loudness(
    audio: np.ndarray,
    *,
    meter: Any,
    target_lufs: float,
    peak_limit: float,
    tolerance_lu: float = 0.2,
    max_iterations: int = 6,
) -> tuple[np.ndarray, float, float, float]:
    """Calibrate pyloudnorm LUFS after resampling while enforcing sample peak."""

    base = np.asarray(audio, dtype=np.float32)
    gain_db = 0.0
    result = base
    lufs = float("nan")
    affected_fraction = 0.0
    for _ in range(max_iterations):
        gained = base * np.float32(10.0 ** (gain_db / 20.0))
        result, affected_fraction = soft_peak_limit(gained, peak_limit)
        lufs = float(meter.integrated_loudness(result.astype(np.float64)))
        if not math.isfinite(lufs):
            raise SilenceError("post-normalization loudness is not finite")
        error = target_lufs - lufs
        if abs(error) <= tolerance_lu:
            break
        gain_db = min(30.0, max(-30.0, gain_db + error))
    return result, lufs, gain_db, affected_fraction


def _existing_result(
    plan: SegmentPlan,
    output: Path,
    *,
    config: AudioPreprocessConfig,
    config_sha256: str,
    soundfile_module: Any,
    previous_row: dict[str, str] | None,
) -> dict[str, Any] | None:
    if not output.is_file() or previous_row is None:
        return None
    if previous_row.get("config_sha256") != config_sha256:
        return None
    if previous_row.get("output_relative_path") != plan.output_relative_path:
        return None
    try:
        same_source_duration = math.isclose(
            float(previous_row["source_duration_seconds"]),
            plan.source_duration_seconds,
            abs_tol=1e-6,
        )
        same_requested_duration = math.isclose(
            float(previous_row["requested_duration_seconds"]),
            plan.requested_duration_seconds,
            abs_tol=1e-6,
        )
    except (KeyError, ValueError):
        return None
    if not same_source_duration or not same_requested_duration:
        return None
    if previous_row.get("status") == "failed":
        return None
    info = soundfile_module.info(str(output))
    if info.samplerate != config.sample_rate or info.channels != 1 or info.subtype != "FLOAT":
        return None
    try:
        expected_output_seconds = float(previous_row["actual_duration_seconds"])
    except (KeyError, ValueError):
        return None
    if not math.isclose(
        info.frames / info.samplerate,
        expected_output_seconds,
        abs_tol=1.0 / config.sample_rate,
    ):
        return None
    output_sha256 = _sha256(output)
    if previous_row.get("sha256") != output_sha256:
        return None
    return {**previous_row, "status": "verified_existing", "error": ""}


def _upgrade_legacy_result(
    plan: SegmentPlan,
    output: Path,
    *,
    config: AudioPreprocessConfig,
    config_sha256: str,
    previous_row: dict[str, str] | None,
    pyloudnorm_module: Any,
    soundfile_module: Any,
) -> dict[str, Any] | None:
    """Upgrade this pipeline's pre-limiter outputs without decoding the source again."""

    if previous_row is None or previous_row.get("status") == "failed":
        return None
    if previous_row.get("config_sha256") == config_sha256:
        return None
    if previous_row.get("limiter_gain_db") or previous_row.get("limited_sample_fraction"):
        return None
    if previous_row.get("output_relative_path") != plan.output_relative_path:
        return None
    try:
        if not math.isclose(
            float(previous_row["source_duration_seconds"]),
            plan.source_duration_seconds,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(previous_row["requested_duration_seconds"]),
            plan.requested_duration_seconds,
            abs_tol=1e-6,
        ):
            return None
    except (KeyError, ValueError):
        return None
    if not output.is_file() or previous_row.get("sha256") != _sha256(output):
        return None
    info = soundfile_module.info(str(output))
    if info.samplerate != config.sample_rate or info.channels != 1 or info.subtype != "FLOAT":
        return None

    audio, sample_rate = soundfile_module.read(str(output), dtype="float32", always_2d=False)
    meter = pyloudnorm_module.Meter(sample_rate)
    peak_limit = float(10.0 ** (config.peak_ceiling_dbfs / 20.0))
    calibrated, lufs_after, gain_db, limited_fraction = calibrate_final_loudness(
        audio,
        meter=meter,
        target_lufs=config.target_lufs,
        peak_limit=peak_limit,
    )
    temporary = output.with_suffix(".part.wav")
    try:
        soundfile_module.write(str(temporary), calibrated, sample_rate, subtype="FLOAT")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **previous_row,
        "actual_duration_seconds": calibrated.size / sample_rate,
        "measured_lufs_after": lufs_after,
        "peak_after": float(np.max(np.abs(calibrated))),
        "peak_limited": limited_fraction > 0.0,
        "limiter_gain_db": gain_db,
        "limited_sample_fraction": limited_fraction,
        "sha256": _sha256(output),
        "status": "verified_upgraded",
        "processed_at": date.today().isoformat(),
        "config_sha256": config_sha256,
        "error": "",
    }


def process_plan(
    plan: SegmentPlan,
    *,
    root: Path,
    source_path: Path | None = None,
    config: AudioPreprocessConfig,
    overwrite: bool = False,
    previous_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    imageio_ffmpeg, pyloudnorm, soundfile = _load_optional_audio_modules()
    source = source_path or root / "data_raw" / Path(plan.source_relative_path)
    output = root / Path(plan.output_relative_path)
    config_sha256 = _config_hash(config)
    if not source.is_file():
        raise PreprocessError(f"missing source audio: {source}")
    if not overwrite:
        existing = _existing_result(
            plan,
            output,
            config=config,
            config_sha256=config_sha256,
            soundfile_module=soundfile,
            previous_row=previous_row,
        )
        if existing is not None:
            return existing
        upgraded = _upgrade_legacy_result(
            plan,
            output,
            config=config,
            config_sha256=config_sha256,
            previous_row=previous_row,
            pyloudnorm_module=pyloudnorm,
            soundfile_module=soundfile,
        )
        if upgraded is not None:
            return upgraded

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    silence_errors: list[str] = []
    for selected_start in fallback_segment_starts(
        plan.source_duration_seconds,
        plan.requested_duration_seconds,
        plan.start_seconds,
    ):
        audio = _decode_f32le(
            source,
            start_seconds=selected_start,
            duration_seconds=plan.requested_duration_seconds,
            sample_rate=config.sample_rate,
            ffmpeg_exe=ffmpeg_exe,
            allow_short=plan.used_full_track,
        )
        try:
            audio, metrics = normalize_loudness(
                audio,
                config.sample_rate,
                target_lufs=config.target_lufs,
                peak_ceiling_dbfs=config.peak_ceiling_dbfs,
                pyloudnorm_module=pyloudnorm,
                ffmpeg_exe=ffmpeg_exe,
            )
            break
        except SilenceError as exc:
            silence_errors.append(f"{selected_start:.3f}s: {exc}")
    else:
        raise SilenceError("all deterministic windows were silent: " + "; ".join(silence_errors))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.wav")
    try:
        soundfile.write(str(temporary), audio, config.sample_rate, subtype="FLOAT")
        info = soundfile.info(str(temporary))
        if info.samplerate != config.sample_rate or info.channels != 1 or info.subtype != "FLOAT":
            raise PreprocessError(f"output verification failed for {plan.segment_id}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        **asdict(plan),
        "start_seconds": selected_start,
        "actual_duration_seconds": audio.size / config.sample_rate,
        "sample_rate": config.sample_rate,
        "channels": 1,
        "dtype": "float32",
        "target_lufs": config.target_lufs,
        **metrics,
        "padding_applied": False,
        "sha256": _sha256(output),
        "status": "verified",
        "processed_at": date.today().isoformat(),
        "config_sha256": config_sha256,
        "error": "",
    }


def _failure_row(
    plan: SegmentPlan, exc: BaseException, config: AudioPreprocessConfig
) -> dict[str, Any]:
    return {
        **asdict(plan),
        "actual_duration_seconds": "",
        "sample_rate": config.sample_rate,
        "channels": 1,
        "dtype": "float32",
        "target_lufs": config.target_lufs,
        "measured_lufs_before": "",
        "measured_lufs_after": "",
        "peak_before": "",
        "peak_after": "",
        "peak_limited": "",
        "limiter_gain_db": "",
        "limited_sample_fraction": "",
        "padding_applied": False,
        "sha256": "",
        "status": "failed",
        "processed_at": date.today().isoformat(),
        "config_sha256": _config_hash(config),
        "error": str(exc),
    }


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (row["group"], row["track_id"], float(row["scale_seconds"])),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_preprocess_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["segment_id"]: {key: value or "" for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
            if row.get("segment_id")
        }


def _parse_scales(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scales must be comma-separated seconds") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("scales must contain positive seconds")
    return values


def _load_pipeline_defaults(root: Path) -> dict[str, Any]:
    with (root / "configs" / "pipeline.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    return raw.get("preprocessing", raw.get("project", {}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focus-preprocess")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("FOCUS_DATASET_ROOT", DEFAULT_DATASET_ROOT)),
        help="verified HF dataset root (default: datasets/open-focus-classical-600)",
    )
    source.add_argument(
        "--data-root",
        type=Path,
        help="explicit legacy canonical-path tree; bypasses the HF layout adapter",
    )
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--output-root", type=Path, default=Path("features/audio"))
    parser.add_argument("--scales", type=_parse_scales, default=DEFAULT_SCALES)
    parser.add_argument("--groups", default="focus,classical")
    parser.add_argument("--track-ids", help="optional comma-separated internal track IDs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-per-group", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("metadata/preprocessed_segments.csv"))
    parser.add_argument("--summary", type=Path, default=Path("metadata/preprocessing_summary.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    metadata_dir = (
        args.metadata_dir if args.metadata_dir.is_absolute() else root / args.metadata_dir
    )
    data_root = None
    dataset_summary: dict[str, Any] | None = None
    if args.data_root is not None:
        data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
        report = validate_metadata(metadata_dir, data_root, check_files=True)
        source_by_track = {
            track.track_id: data_root / track.relative_path
            for track in load_tracks(metadata_dir / "track_index.csv")
        }
        source_mode = "legacy_data_root"
    else:
        dataset_root = (
            args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
        )
        report = validate_metadata(metadata_dir, root, check_files=False)
        try:
            verification = verify_release_dataset(
                dataset_root=dataset_root,
                project_metadata=metadata_dir,
                verify_audio=True,
            )
        except DatasetReleaseError as exc:
            raise PreprocessError(f"HF dataset validation failed: {exc}") from exc
        source_by_track = verification.source_by_track
        dataset_summary = verification.summary
        source_mode = "huggingface_release"
    if not report.ok:
        raise PreprocessError("metadata validation failed: " + "; ".join(report.errors))
    defaults = _load_pipeline_defaults(root)
    config = AudioPreprocessConfig(
        sample_rate=int(defaults.get("analysis_sample_rate", 22_050)),
        channels=int(defaults.get("analysis_channels", 1)),
        target_lufs=float(defaults.get("target_lufs", -15.0)),
        peak_ceiling_dbfs=float(defaults.get("peak_ceiling_dbfs", -1.0)),
        scales_seconds=tuple(args.scales),
    )
    config.validate()

    try:
        groups = {TrackGroup(value.strip()) for value in args.groups.split(",") if value.strip()}
    except ValueError as exc:
        raise PreprocessError(f"invalid group in {args.groups!r}") from exc
    all_tracks = load_tracks(metadata_dir / "track_index.csv")
    tracks = [
        track
        for track in all_tracks
        if track.group in groups
    ]
    assignments = load_assignments(metadata_dir)
    if dataset_summary is not None:
        release_rows = {row["track_id"]: row for row in verification.rows}
        for track in all_tracks:
            row = release_rows[track.track_id]
            if row["split"] != assignments[track.track_id].value:
                raise PreprocessError(f"HF/project split mismatch for {track.track_id}")
    if args.track_ids:
        requested_ids = {value.strip() for value in args.track_ids.split(",") if value.strip()}
        available_ids = {track.track_id for track in tracks}
        missing_ids = requested_ids - available_ids
        if missing_ids:
            raise PreprocessError(f"unknown or excluded track IDs: {sorted(missing_ids)}")
        tracks = [track for track in tracks if track.track_id in requested_ids]
    if args.limit_per_group is not None:
        if args.limit_per_group <= 0:
            raise PreprocessError("limit-per-group must be positive")
        limited: list[TrackRecord] = []
        per_group: Counter[str] = Counter()
        for track in tracks:
            if per_group[track.group.value] < args.limit_per_group:
                limited.append(track)
                per_group[track.group.value] += 1
        tracks = limited

    plans = build_segment_plans(
        tracks,
        assignments,
        scales_seconds=config.scales_seconds,
        output_root=args.output_root,
    )
    estimated_bytes = sum(
        int(plan.requested_duration_seconds * config.sample_rate * config.channels * 4)
        for plan in plans
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tracks": len(tracks),
                    "segments": len(plans),
                    "estimated_gib": round(estimated_bytes / (1024**3), 2),
                    "full_track_segments": sum(plan.used_full_track for plan in plans),
                    "groups": dict(Counter(plan.group for plan in plans)),
                    "splits": dict(Counter(plan.split for plan in plans)),
                    "config_sha256": _config_hash(config),
                    "source_mode": source_mode,
                    "dataset_identity": dataset_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.workers <= 0:
        raise PreprocessError("workers must be positive")
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    previous_rows = read_preprocess_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_plan,
                plan,
                root=root,
                source_path=source_by_track[plan.track_id],
                config=config,
                overwrite=args.overwrite,
                previous_row=previous_rows.get(plan.segment_id),
            ): plan
            for plan in plans
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            plan = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep the batch auditable and resumable
                row = _failure_row(plan, exc, config)
            rows.append(row)
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                failures = sum(item["status"] == "failed" for item in rows)
                print(
                    f"preprocess: {completed}/{len(futures)} complete; failures={failures}",
                    flush=True,
                )

    write_manifest(manifest_path, rows)
    status_counts = Counter(row["status"] for row in rows)
    successful = [row for row in rows if row["status"] != "failed"]
    plan_by_id = {plan.segment_id: plan for plan in plans}
    lufs_after = [
        float(row["measured_lufs_after"])
        for row in successful
        if str(row.get("measured_lufs_after", "")).strip()
    ]
    peaks_after = [
        float(row["peak_after"])
        for row in successful
        if str(row.get("peak_after", "")).strip()
    ]
    actual_durations = [float(row["actual_duration_seconds"]) for row in successful]
    peak_ceiling = float(10.0 ** (config.peak_ceiling_dbfs / 20.0))
    output_bytes = sum(
        (root / Path(row["output_relative_path"])).stat().st_size for row in successful
    )
    summary = {
        "generated_at": date.today().isoformat(),
        "ok": status_counts["failed"] == 0,
        "manifest": str(manifest_path),
        "tracks": len({row["track_id"] for row in rows}),
        "segments": len(rows),
        "status_counts": dict(status_counts),
        "group_counts": dict(Counter(row["group"] for row in successful)),
        "split_counts": dict(Counter(row["split"] for row in successful)),
        "scale_counts": dict(
            Counter(f"{_seconds_token(float(row['scale_seconds']))}s" for row in successful)
        ),
        "full_track_segments": sum(
            str(row["used_full_track"]).strip().lower() in {"true", "1"}
            for row in successful
        ),
        "silence_fallback_segments": sum(
            not math.isclose(
                float(row["start_seconds"]),
                plan_by_id[row["segment_id"]].start_seconds,
                abs_tol=1e-6,
            )
            for row in successful
        ),
        "loudness_within_minus16_minus14": sum(-16.0 <= value <= -14.0 for value in lufs_after),
        "loudness_measured_segments": len(lufs_after),
        "loudness_after_min": min(lufs_after, default=None),
        "loudness_after_max": max(lufs_after, default=None),
        "peak_within_ceiling_segments": sum(value <= peak_ceiling for value in peaks_after),
        "peak_measured_segments": len(peaks_after),
        "maximum_sample_peak": max(peaks_after, default=None),
        "output_audio_hours": round(sum(actual_durations) / 3600.0, 2),
        "output_bytes": output_bytes,
        "output_gib": round(output_bytes / (1024**3), 2),
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "manifest_sha256": _sha256(manifest_path),
        "source_mode": source_mode,
        "dataset_identity": dataset_summary,
    }
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
