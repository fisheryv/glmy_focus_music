import csv
import urllib.parse
from pathlib import Path

from data.focus_open import (
    OPEN_FOCUS_COLUMNS,
    _download_one,
    rebuild_local_style_selection,
    select_fma_fallback,
    select_jamendo_focus,
)


def _result(
    track_id: int,
    artist_id: int,
    *,
    genre: str = "ambient",
    album_name: str = "",
) -> dict:
    return {
        "id": str(track_id),
        "name": f"Track {track_id}",
        "duration": 180,
        "artist_id": str(artist_id),
        "artist_name": f"Artist {artist_id}",
        "album_id": str(track_id),
        "album_name": album_name,
        "license_ccurl": "https://creativecommons.org/licenses/by/4.0/",
        "shareurl": f"https://www.jamendo.com/track/{track_id}",
        "audiodownload_allowed": True,
        "musicinfo": {
            "tags": {"genres": [genre], "instruments": ["piano"], "vartags": []}
        },
    }


def test_select_jamendo_focus_applies_filters_caps_and_reserve(tmp_path: Path) -> None:
    results = [_result(index, index // 2) for index in range(1, 11)]

    def request_json(url: str) -> dict:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["vocalinstrumental"] == ["instrumental"]
        assert query["audioformat"] == ["mp32"]
        assert query["speed"][0] in {"verylow", "low"}
        mood = query["tags"][0]
        page = int(query.get("offset", ["0"])[0])
        page_results = results if mood == "study" and page == 0 else []
        return {"headers": {"status": "success"}, "results": page_results}

    rows = select_jamendo_focus(
        tmp_path / "focus.csv",
        client_id="test",
        target_count=3,
        reserve_count=2,
        max_per_artist=1,
        max_per_album=1,
        max_pages=1,
        request_json=request_json,
    )

    assert len(rows) == 5
    assert sum(row["selection_status"] == "selected" for row in rows) == 3
    assert sum(row["selection_status"] == "reserve" for row in rows) == 2
    assert len({row["artist_key"] for row in rows}) == 5
    assert all(set(row) == set(OPEN_FOCUS_COLUMNS) for row in rows)


def test_select_jamendo_focus_preserves_album_name(tmp_path: Path) -> None:
    result = _result(1, 1, album_name="Quiet Meditation")

    def request_json(url: str) -> dict:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        results = [result] if query["tags"] == ["study"] else []
        return {"headers": {"status": "success"}, "results": results}

    rows = select_jamendo_focus(
        tmp_path / "focus.csv",
        client_id="test",
        target_count=1,
        reserve_count=0,
        max_pages=1,
        request_json=request_json,
    )

    assert rows[0]["album_name"] == "Quiet Meditation"


def _mpeg1_layer3_frame() -> bytes:
    header = bytes.fromhex("fffb9000")
    frame_bytes = 144 * 128_000 // 44_100
    return header + bytes(frame_bytes - len(header))


def test_download_rejects_audio_below_actual_bitrate_threshold(tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(_mpeg1_layer3_frame() * 100)
    row = {column: "" for column in OPEN_FOCUS_COLUMNS}
    row.update(
        {
            "track_id": "focus_fma_000001",
            "relative_path": "focus_open_music/focus_fma_000001.mp3",
            "download_url": source.as_uri(),
        }
    )

    result = _download_one(row, tmp_path / "data", "unused", 127.8)

    assert result["download_status"] == "rejected_bitrate"
    assert float(result["average_bitrate_kbps"]) == 127.706


def test_fma_fallback_requires_instrumental_slow_low_speech_audio(tmp_path: Path) -> None:
    metadata = tmp_path / "fma_metadata"
    audio_root = tmp_path / "fma_medium"
    metadata.mkdir()
    target = audio_root / "000" / "000001.mp3"
    target.parent.mkdir(parents=True)
    target.write_bytes(_mpeg1_layer3_frame() * 100)
    (metadata / "tracks.csv").write_text(
        ",album,artist,track,track,track,track,track\n"
        ",id,name,duration,genre_top,license,title,tags\n"
        "1,10,Quiet Artist,180,Ambient,CC BY,Still Air,ambient background\n"
        "2,11,Loud Artist,180,Rock,CC BY,Noise,rock\n",
        encoding="utf-8",
    )
    (metadata / "echonest.csv").write_text(
        ",echonest,echonest,echonest,echonest\n"
        ",audio_features,audio_features,audio_features,audio_features\n"
        "track_id,instrumentalness,tempo,speechiness,energy\n"
        "1,0.98,72,0.02,0.20\n"
        "2,0.99,140,0.02,0.90\n",
        encoding="utf-8",
    )

    rows = select_fma_fallback(metadata, audio_root, count=1)

    assert [row["source_track_id"] for row in rows] == ["1"]
    assert rows[0]["matched_genre_tags"] == "ambient"
    assert "instrumentalness=0.980" in rows[0]["instrumental_evidence"]


def test_rebuild_local_style_keeps_only_allowed_genres(tmp_path: Path) -> None:
    selected_dir = tmp_path / "focus_open_music"
    excluded_dir = tmp_path / "focus_open_music_excluded"
    selected_dir.mkdir()
    excluded_dir.mkdir()
    rows = []
    for index, genre in enumerate(("ambient", "ambient newage", "lofi", "newage"), start=1):
        track_id = f"focus_jamendo_{index:07d}"
        row = {column: "" for column in OPEN_FOCUS_COLUMNS}
        row.update(
            {
                "track_id": track_id,
                "artist_key": f"artist_{index}",
                "album_key": f"album_{index}",
                "matched_genre_tags": genre,
                "matched_mood_tags": "meditation",
                "speed_evidence": "Jamendo API speed=verylow",
                "download_status": "verified",
                "relative_path": f"focus_open_music/{track_id}.mp3",
            }
        )
        rows.append(row)
        (selected_dir / f"{track_id}.mp3").write_bytes(b"audio")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPEN_FOCUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = rebuild_local_style_selection(
        manifest,
        selected_dir,
        excluded_dir,
        target_count=2,
        max_per_artist=1,
        max_per_album=1,
    )

    rebuilt = list(csv.DictReader(manifest.open(encoding="utf-8")))
    kept = [row for row in rebuilt if row["download_status"] == "verified"]
    assert report["selected"] == 2
    assert len(list(selected_dir.glob("*.mp3"))) == 2
    assert len(list(excluded_dir.glob("*.mp3"))) == 2
    assert all({"ambient", "lofi"} & set(row["matched_genre_tags"].split()) for row in kept)


def test_rebuild_prefers_accented_meditation_text_and_honors_forced_ids(
    tmp_path: Path,
) -> None:
    selected_dir = tmp_path / "focus_open_music"
    excluded_dir = tmp_path / "focus_open_music_excluded"
    selected_dir.mkdir()
    excluded_dir.mkdir()
    rows = []
    for index, title in enumerate(
        ("Ordinary", "Meditación profunda", "Other", "Required"), start=1
    ):
        track_id = f"focus_jamendo_{index:07d}"
        row = {column: "" for column in OPEN_FOCUS_COLUMNS}
        row.update(
            {
                "track_id": track_id,
                "title": title,
                "artist_key": f"artist_{index}",
                "album_key": f"album_{index}",
                "matched_mood_tags": "meditation",
                "speed_evidence": "Jamendo API speed=verylow",
                "download_status": "verified",
                "relative_path": f"focus_open_music/{track_id}.mp3",
            }
        )
        rows.append(row)
        (selected_dir / f"{track_id}.mp3").write_bytes(b"audio")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPEN_FOCUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = rebuild_local_style_selection(
        manifest,
        selected_dir,
        excluded_dir,
        allowed_genres=(),
        allowed_moods=("meditation",),
        prefer_title_album_stem="medit",
        force_include=("focus_jamendo_0000004",),
        force_exclude=("focus_jamendo_0000001",),
        target_count=2,
    )

    rebuilt = list(csv.DictReader(manifest.open(encoding="utf-8")))
    kept = {row["track_id"] for row in rebuilt if row["download_status"] == "verified"}
    assert kept == {"focus_jamendo_0000002", "focus_jamendo_0000004"}
    assert report["preferred_title_album_selected"] == 1
    assert report["forced_included"] == ["focus_jamendo_0000004"]
    assert report["forced_excluded"] == ["focus_jamendo_0000001"]
