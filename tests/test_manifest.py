from pathlib import Path

from focus_topology.data.manifest import validate_metadata

TRACK_HEADER = (
    "track_id,group,relative_path,sha256,duration_seconds,sample_rate,channels,"
    "artist_key,album_key,composer_key,instrumental,restricted\n"
)
LICENSE_HEADER = (
    "track_id,group,source_url,license_type,downloaded_at,redistribution_allowed,notes\n"
)


def _write_metadata(root: Path, track_row: str, license_row: str) -> None:
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "track_index.csv").write_text(TRACK_HEADER + track_row, encoding="utf-8")
    (metadata / "licenses.csv").write_text(LICENSE_HEADER + license_row, encoding="utf-8")
    (metadata / "split_discovery.csv").write_text("track_id,group\nt1,focus\n", encoding="utf-8")
    (metadata / "split_validation.csv").write_text("track_id,group\n", encoding="utf-8")
    (metadata / "split_holdout.csv").write_text("track_id,group\n", encoding="utf-8")


def test_open_focus_track_requires_verified_license_and_source(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        "t1,focus,focus_music/t1.wav,,10,22050,1,a,b,,true,false\n",
        "t1,focus,,unknown,2026-07-16,false,missing provenance\n",
    )

    report = validate_metadata(tmp_path / "metadata", tmp_path / "data_raw")

    assert not report.ok
    assert any("no verified license" in error for error in report.errors)
    assert any("no source_url" in error for error in report.errors)


def test_valid_open_focus_record_passes(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        "t1,focus,focus_music/t1.wav,,10,22050,1,a,b,,true,false\n",
        "t1,focus,https://example.test/t1,CC_BY,2026-08-01,true,open\n",
    )

    report = validate_metadata(tmp_path / "metadata", tmp_path / "data_raw")

    assert report.ok


def test_valid_restricted_focus_record_passes(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        "t1,focus,focus_music/t1.wav,,10,22050,1,a,b,,true,true\n",
        "t1,focus,,research_authorization,2026-07-16,false,restricted\n",
    )

    report = validate_metadata(tmp_path / "metadata", tmp_path / "data_raw")

    assert report.ok


def test_restricted_record_cannot_allow_redistribution(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        "t1,focus,focus_music/t1.wav,,10,22050,1,a,b,,true,true\n",
        "t1,focus,,research_authorization,2026-07-16,true,restricted\n",
    )

    report = validate_metadata(tmp_path / "metadata", tmp_path / "data_raw")

    assert not report.ok
    assert any("restricted track" in error for error in report.errors)


def test_classical_track_is_allowed_in_holdout(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path,
        "t1,classical,classical_music/t1.wav,,10,22050,1,,,bach,true,false\n",
        "t1,classical,https://example.test/t1,PDM,2026-08-01,true,open\n",
    )
    metadata = tmp_path / "metadata"
    (metadata / "split_discovery.csv").write_text("track_id,group\n", encoding="utf-8")
    (metadata / "split_holdout.csv").write_text("track_id,group\nt1,classical\n", encoding="utf-8")

    report = validate_metadata(metadata, tmp_path / "data_raw")

    assert report.ok
