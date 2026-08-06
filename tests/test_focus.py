from pathlib import Path

import pytest

from data.focus import _allocate_splits, assign_splits, scan_mp3


def _mpeg1_layer3_frame() -> bytes:
    header = bytes.fromhex("fffb9000")
    frame_bytes = 144 * 128_000 // 44_100
    return header + bytes(frame_bytes - len(header))


def test_frame_scan_ignores_large_non_audio_tail(tmp_path: Path) -> None:
    frames = _mpeg1_layer3_frame() * 100
    path = tmp_path / "padded.mp3"
    path.write_bytes(frames + bytes(5 * 1024 * 1024))

    result = scan_mp3(path)

    assert result.mpeg_frame_count == 100
    assert result.duration_seconds == pytest.approx(100 * 1152 / 44100)
    assert result.sample_rate == 44100
    assert result.channels == 2
    assert result.duration_discrepancy
    assert result.valid_audio_bytes == len(frames)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (115, {"discovery": 75, "validation": 23, "holdout": 17}),
        (85, {"discovery": 55, "validation": 17, "holdout": 13}),
    ],
)
def test_focus_split_allocation(count: int, expected: dict[str, int]) -> None:
    assert _allocate_splits(count) == expected


def test_category_stratified_split_keeps_global_target_counts() -> None:
    rows = [
        {
            "track_id": f"deep_{index:03d}",
            "category": "focus_deepwork",
            "sha256": f"{index:064x}",
        }
        for index in range(116)
    ]
    rows.extend(
        {
            "track_id": f"learning_{index:03d}",
            "category": "focus_learning",
            "sha256": f"{index + 1000:064x}",
        }
        for index in range(84)
    )

    assignments = assign_splits(rows)

    assert len(assignments) == 200
    assert list(assignments.values()).count("discovery") == 130
    assert list(assignments.values()).count("validation") == 40
    assert list(assignments.values()).count("holdout") == 30
