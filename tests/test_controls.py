import csv
import hashlib
import io
import tarfile
import wave
from collections import Counter
from pathlib import Path

from data.controls import (
    CANDIDATE_COLUMNS,
    _assign_classical_split,
    catalog_classical_candidates,
    download_candidates,
    extend_classical_with_musicnet,
    extract_musicnet_archive,
    parse_jamendo_licenses,
    select_pop_candidates,
)


def test_parse_jamendo_license_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "audio_licenses.txt"
    ledger.write_text(
        "01/101.mp3\n"
        "Track by Artist from Jamendo: http://www.jamendo.com/track/101\n"
        "Available under a Creative Commons Attribution license: "
        "http://creativecommons.org/licenses/by/3.0/\n\n",
        encoding="utf-8",
    )

    parsed = parse_jamendo_licenses(ledger)

    assert parsed["01/101.mp3"].license_url.endswith("/by/3.0/")


def test_pop_selection_requires_pop_and_unanimous_instrumental(tmp_path: Path) -> None:
    source = tmp_path / "mtg"
    (source / "data").mkdir(parents=True)
    annotations = source / "derived/music-classification-annotations"
    annotations.mkdir(parents=True)
    (source / "data/raw_30s_cleantags_50artists.tsv").write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "track_0000101\tartist_a\talbum_a\t01/101.mp3\t180\tgenre---pop\tinstrument---piano\n"
        "track_0000102\tartist_b\talbum_b\t02/102.mp3\t180\tgenre---rock\n",
        encoding="utf-8",
    )
    (annotations / "music-classification-annotations-clean.tsv").write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tANNOTATIONS\n"
        "track_0000101\tartist_a\talbum_a\t01/101.mp3\t180\t"
        "voice_instrumental---instrumental,instrumental,instrumental\n"
        "track_0000102\tartist_b\talbum_b\t02/102.mp3\t180\t"
        "voice_instrumental---instrumental,instrumental,instrumental\n",
        encoding="utf-8",
    )
    (source / "data/raw.meta.tsv").write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME\tALBUM_NAME\tRELEASEDATE\tURL\n"
        "track_0000101\tartist_a\talbum_a\tOne\tArtist A\tAlbum A\t2020-01-01\thttp://jamendo/101\n",
        encoding="utf-8",
    )
    (source / "audio_licenses.txt").write_text(
        "01/101.mp3\nTrack by Artist from Jamendo: http://www.jamendo.com/track/101\n"
        "Available under a Creative Commons Attribution license: "
        "http://creativecommons.org/licenses/by/3.0/\n\n",
        encoding="utf-8",
    )

    rows = select_pop_candidates(source, tmp_path / "pop.csv")

    assert [row["source_track_id"] for row in rows] == ["101"]
    assert rows[0]["instrumental_evidence"].startswith("MTG unanimous")

    rows[0]["download_status"] = "verified"
    rows[0]["sha256"] = "saved-checksum"
    with (tmp_path / "pop.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    preserved = select_pop_candidates(source, tmp_path / "pop.csv")
    excluded = select_pop_candidates(
        source,
        tmp_path / "pop.csv",
        excluded_source_ids=["101"],
    )

    assert preserved[0]["download_status"] == "verified"
    assert preserved[0]["sha256"] == "saved-checksum"
    assert excluded == []


def test_classical_catalog_keeps_only_piano_and_quartet(tmp_path: Path) -> None:
    prefix = "https://archive.org/download/musopen-lossless-dvd/Musopen-Lossless-DVD.zip/"
    html = (
        f'<a href="{prefix}Musopen%20DVD%20%28lossless%29/Goldberg%20Variations/'
        'Variation%201.m4a">piano</a>'
        f'<a href="{prefix}Musopen%20DVD%20%28lossless%29/String%20Quartets/'
        'Haydn%20Quartet%20in%20D%20Major%20Op.64/Movement%20I.m4a">quartet</a>'
        f'<a href="{prefix}Musopen%20DVD%20%28lossless%29/Brahms%20-%20Symphony/'
        'Movement%20I.m4a">orchestra</a>'
    )

    rows = catalog_classical_candidates(tmp_path / "classical.csv", html=html)

    assert {row["subpool"] for row in rows} == {"piano_solo", "string_quartet"}
    assert {row["composer_key"] for row in rows} == {"bach", "haydn"}


def test_classical_split_is_exact_symmetric_and_composer_isolated() -> None:
    rows = []
    groups = (
        ("bach", 5, ("piano_solo",)),
        ("mozart", 4, ("piano_solo", "string_quartet")),
        ("brahms", 3, ("mixed_chamber",)),
        ("schubert", 3, ("piano_solo",)),
        ("dvorak", 2, ("string_quartet",)),
        ("faure", 2, ("mixed_chamber",)),
        ("suk", 1, ("string_quartet",)),
    )
    for composer, count, subpools in groups:
        for index in range(count):
            rows.append(
                {
                    "track_id": f"{composer}_{index}",
                    "composer_key": composer,
                    "subpool": subpools[index % len(subpools)],
                    "split": "",
                }
            )

    _assign_classical_split(rows, 0.20, 20260716, 0.15)

    assert Counter(row["split"] for row in rows) == {
        "discovery": 13,
        "validation": 4,
        "holdout": 3,
    }
    composer_splits: dict[str, set[str]] = {}
    for row in rows:
        composer_splits.setdefault(row["composer_key"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in composer_splits.values())


def test_force_download_replaces_a_verified_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"ID3new-audio")
    data_root = tmp_path / "data_raw"
    target = data_root / "classical_music/track.mp3"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"ID3old-audio")
    manifest = tmp_path / "manifest.csv"
    row = {column: "" for column in CANDIDATE_COLUMNS}
    row.update(
        {
            "track_id": "track",
            "group": "classical",
            "download_url": source.as_uri(),
            "relative_path": "classical_music/track.mp3",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "download_status": "verified",
        }
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    counts = download_candidates(
        manifest,
        data_root,
        workers=1,
        force_track_ids=["track"],
    )

    assert counts["verified"] == 1
    assert target.read_bytes() == source.read_bytes()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        updated = next(csv.DictReader(handle))
    assert updated["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_musicnet_extension_excludes_museopen_and_reaches_target(tmp_path: Path) -> None:
    manifest = tmp_path / "classical.csv"
    base_rows = []
    for track_id, composer, subpool in (
        ("piano", "bach", "piano_solo"),
        ("quartet", "beethoven", "string_quartet"),
    ):
        row = {column: "" for column in CANDIDATE_COLUMNS}
        row.update(
            {
                "track_id": track_id,
                "group": "classical",
                "source_dataset": "musopen-lossless-dvd",
                "composer_key": composer,
                "subpool": subpool,
                "download_status": "verified",
            }
        )
        base_rows.append(row)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(base_rows)

    metadata = tmp_path / "musicnet_metadata.csv"
    columns = [
        "id",
        "composer",
        "composition",
        "movement",
        "ensemble",
        "source",
        "transcriber",
        "catalog_name",
        "seconds",
    ]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "1",
                    "composer": "Brahms",
                    "composition": "Cello Sonata",
                    "movement": "I",
                    "ensemble": "Accompanied Cello",
                    "source": "European Archive",
                    "transcriber": "",
                    "catalog_name": "OP38",
                    "seconds": "120",
                },
                {
                    "id": "2",
                    "composer": "Mozart",
                    "composition": "Piano Trio",
                    "movement": "II",
                    "ensemble": "Piano Trio",
                    "source": "European Archive",
                    "transcriber": "",
                    "catalog_name": "K1",
                    "seconds": "100",
                },
                {
                    "id": "3",
                    "composer": "Schubert",
                    "composition": "Piano Sonata",
                    "movement": "I",
                    "ensemble": "Solo Piano",
                    "source": "Museopen",
                    "transcriber": "",
                    "catalog_name": "D1",
                    "seconds": "90",
                },
            ]
        )

    rows = extend_classical_with_musicnet(
        manifest,
        metadata,
        target_total=4,
        max_per_composer=1,
    )

    assert len(rows) == 4
    assert sum(row["source_dataset"] == "musicnet-zenodo-5120004" for row in rows) == 2
    assert all(row["source_track_id"] != "3" for row in rows)


def test_extract_selected_musicnet_wav(tmp_path: Path) -> None:
    wav_data = io.BytesIO()
    with wave.open(wav_data, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 8000)
    archive = tmp_path / "musicnet.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("musicnet/train_data/42.wav")
        info.size = len(wav_data.getvalue())
        handle.addfile(info, io.BytesIO(wav_data.getvalue()))

    manifest = tmp_path / "classical.csv"
    row = {column: "" for column in CANDIDATE_COLUMNS}
    row.update(
        {
            "track_id": "classical_musicnet_0042",
            "group": "classical",
            "source_dataset": "musicnet-zenodo-5120004",
            "source_track_id": "42",
            "relative_path": "classical_music/classical_musicnet_0042.wav",
            "download_status": "pending",
        }
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    counts = extract_musicnet_archive(
        archive,
        manifest,
        tmp_path / "data_raw",
        expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),  # noqa: S324
    )

    assert counts["verified"] == 1
    target = tmp_path / "data_raw/classical_music/classical_musicnet_0042.wav"
    assert target.is_file()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        updated = next(csv.DictReader(handle))
    assert updated["duration_seconds"] == "1.000"
