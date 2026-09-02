import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest

from data.preprocess import (
    PreprocessError,
    _decode_f32le,
    build_parser,
    build_segment_plans,
    choose_segment_window,
    fallback_segment_starts,
    parse_loudnorm_stats,
    soft_peak_limit,
    write_manifest,
)
from focus_topology.data.schema import SplitName, TrackGroup, TrackRecord


def test_parser_accepts_isolated_metadata_and_output_roots() -> None:
    args = build_parser().parse_args(
        [
            "--metadata-dir",
            "metadata/open",
            "--output-root",
            "features/audio_open",
        ]
    )

    assert args.metadata_dir == Path("metadata/open")
    assert args.output_root == Path("features/audio_open")
    assert args.dataset_root == Path("datasets/open-focus-classical-600")
    assert args.data_root is None


def test_parser_keeps_legacy_data_root_explicit() -> None:
    args = build_parser().parse_args(["--data-root", "data_raw"])

    assert args.data_root == Path("data_raw")


def test_choose_segment_window_uses_center_crop() -> None:
    assert choose_segment_window(600.0, 180.0) == (210.0, 180.0, False)


def test_choose_segment_window_preserves_short_track_without_padding() -> None:
    assert choose_segment_window(120.0, 180.0) == (0.0, 120.0, True)


def test_fallback_starts_are_content_independent_and_bounded() -> None:
    assert fallback_segment_starts(712.0, 180.0, 266.0) == (266.0, 86.0, 446.0, 0.0, 532.0)


def test_parse_loudnorm_stats_extracts_last_json_block() -> None:
    stderr = '''noise
    {
      "input_i": "-20.10", "input_tp": "-2.00", "input_lra": "3.20",
      "input_thresh": "-30.20", "output_i": "-15.00", "target_offset": "0.01"
    }
    '''
    assert parse_loudnorm_stats(stderr) == {
        "input_i": "-20.10",
        "input_tp": "-2.00",
        "input_lra": "3.20",
        "input_thresh": "-30.20",
        "target_offset": "0.01",
    }


def test_soft_peak_limit_preserves_small_samples_and_caps_peaks() -> None:
    audio = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    limited, affected = soft_peak_limit(audio, 0.9)
    assert limited[1:4].tolist() == audio[1:4].tolist()
    assert float(np.max(np.abs(limited))) <= 0.9
    assert affected == pytest.approx(0.4)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf")])
def test_choose_segment_window_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        choose_segment_window(duration, 180.0)


def test_plans_inherit_source_track_split() -> None:
    track = TrackRecord(
        track_id="focus_001",
        group=TrackGroup.FOCUS,
        relative_path=Path("focus_music/focus_001.mp3"),
        duration_seconds=600.0,
        instrumental=True,
        restricted=True,
    )
    plans = build_segment_plans(
        [track],
        {track.track_id: SplitName.HOLDOUT},
        scales_seconds=(180.0, 300.0),
    )
    assert {plan.split for plan in plans} == {"holdout"}
    assert {plan.scale_seconds for plan in plans} == {180.0, 300.0}
    assert all("/holdout/" in plan.output_relative_path for plan in plans)


def test_decode_accepts_catalog_duration_overestimate_only_for_full_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = np.ones(50, dtype="<f4")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=decoded.tobytes(), stderr=b""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = Path("catalog-duration-overestimate.mp3")

    with pytest.raises(PreprocessError, match="decoded duration is short"):
        _decode_f32le(
            source,
            start_seconds=0.0,
            duration_seconds=10.0,
            sample_rate=10,
            ffmpeg_exe="ffmpeg",
        )

    accepted = _decode_f32le(
        source,
        start_seconds=0.0,
        duration_seconds=10.0,
        sample_rate=10,
        ffmpeg_exe="ffmpeg",
        allow_short=True,
    )
    assert accepted.size == 50


def test_manifest_sort_accepts_resumed_string_and_fresh_float_scales(tmp_path: Path) -> None:
    path = tmp_path / "preprocessed_segments.csv"
    rows = [
        {
            "segment_id": "track__300s",
            "track_id": "track",
            "group": "focus",
            "scale_seconds": 300.0,
        },
        {
            "segment_id": "track__180s",
            "track_id": "track",
            "group": "focus",
            "scale_seconds": "180.0",
        },
    ]

    write_manifest(path, rows)

    with path.open(encoding="utf-8", newline="") as handle:
        segment_ids = [row["segment_id"] for row in csv.DictReader(handle)]
    assert segment_ids == ["track__180s", "track__300s"]
