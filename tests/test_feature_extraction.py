from pathlib import Path

import numpy as np
import pytest

from features.contracts import FeatureExtractionConfig
from features.extractor import (
    FeatureExtractionError,
    LibrosaFeatureExtractor,
    _modulation_features,
)

librosa = pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")
scipy_fft = pytest.importorskip("scipy.fft")


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int = 22_050) -> None:
    soundfile.write(path, audio.astype(np.float32), sample_rate, subtype="FLOAT")


def test_sine_extracts_expected_pitch_and_shapes(tmp_path: Path) -> None:
    config = FeatureExtractionConfig()
    time = np.arange(4 * config.sample_rate) / config.sample_rate
    audio = 0.25 * np.sin(2.0 * np.pi * 440.0 * time)
    path = tmp_path / "a440.wav"
    _write_audio(path, audio)

    bundle = LibrosaFeatureExtractor().extract(path, track_id="tone", config=config)

    assert bundle.acoustic.vectors.shape[1] == config.acoustic_dimension
    assert np.all(np.diff(bundle.acoustic.times) > 0)
    assert np.count_nonzero(bundle.pitch.states == 9) >= bundle.pitch.states.size // 2
    assert bundle.modulation.spectrum.shape[1] == bundle.modulation.frequencies.size
    assert all(
        np.all(np.isfinite(values))
        for values in (
            bundle.acoustic.vectors,
            bundle.pitch.chroma,
            bundle.rhythm.vectors,
            bundle.modulation.spectrum,
        )
    )


def test_click_track_recovers_half_second_ioi(tmp_path: Path) -> None:
    config = FeatureExtractionConfig()
    audio = np.zeros(8 * config.sample_rate, dtype=np.float32)
    click_positions = (np.arange(0.5, 8.0, 0.5) * config.sample_rate).astype(int)
    for position in click_positions:
        audio[position : position + 64] = np.hanning(128)[:64]
    path = tmp_path / "clicks.wav"
    _write_audio(path, audio)

    bundle = LibrosaFeatureExtractor().extract(path, track_id="clicks", config=config)

    assert bundle.rhythm.inter_onset_intervals.size >= 10
    assert float(np.median(bundle.rhythm.inter_onset_intervals)) == pytest.approx(0.5, abs=0.06)
    assert np.all(np.diff(bundle.rhythm.beat_times) > 0)


@pytest.mark.parametrize(
    ("modulation_hz", "expected_key_band"),
    [(8.0, 0), (16.0, 1), (32.0, 2)],
)
def test_amplitude_modulation_peaks_in_key_band(
    modulation_hz: float, expected_key_band: int
) -> None:
    config = FeatureExtractionConfig(
        sample_rate=4_000,
        n_fft=512,
        hop_length=128,
        n_mels=32,
        tempo_bins=16,
        modulation_hop_length=32,
        acoustic_pca_components=16,
    )
    time = np.arange(8 * config.sample_rate) / config.sample_rate
    envelope = 1.0 + 0.9 * np.sin(2.0 * np.pi * modulation_hz * time)
    audio = (0.2 * envelope * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)

    features = _modulation_features(
        audio,
        duration=8.0,
        config=config,
        librosa_module=librosa,
        scipy_fft_module=scipy_fft,
    )

    median_energy = np.median(features.key_band_energies, axis=0)
    assert int(np.argmax(median_energy)) == expected_key_band
    assert np.all(features.valid)


def test_non_finite_audio_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nan.wav"
    audio = np.zeros(22_050, dtype=np.float32)
    audio[100] = np.nan
    _write_audio(path, audio)

    with pytest.raises(FeatureExtractionError, match="non-finite"):
        LibrosaFeatureExtractor().extract(
            path, track_id="invalid", config=FeatureExtractionConfig()
        )


def test_silence_produces_finite_features_and_uncertain_pitch(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    _write_audio(path, np.zeros(4 * 22_050, dtype=np.float32))

    bundle = LibrosaFeatureExtractor().extract(
        path, track_id="silence", config=FeatureExtractionConfig()
    )

    assert np.all(np.isfinite(bundle.acoustic.vectors))
    assert np.all(bundle.pitch.states == 12)
    assert not np.any(bundle.modulation.valid)
