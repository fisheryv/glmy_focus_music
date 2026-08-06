from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    AcousticFeatures,
    FeatureBundle,
    FeatureExtractionConfig,
    ModulationFeatures,
    PitchFeatures,
    RhythmFeatures,
)
from .states import dominant_pitch_states
from .structure import structural_features


class FeatureExtractionError(RuntimeError):
    pass


def _load_audio_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import librosa
        import soundfile
        from scipy import fft as scipy_fft
        from scipy.interpolate import interp1d
    except ImportError as exc:
        raise FeatureExtractionError(
            "audio dependencies are missing; install the project with the 'audio' extra"
        ) from exc
    return librosa, soundfile, scipy_fft, interp1d


def _regular_windows(
    duration: float, window_seconds: float, hop_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if duration <= window_seconds:
        starts = np.array([0.0], dtype=np.float64)
        ends = np.array([duration], dtype=np.float64)
    else:
        count = int(np.floor((duration - window_seconds) / hop_seconds)) + 1
        starts = np.arange(count, dtype=np.float64) * hop_seconds
        ends = starts + window_seconds
    return starts, ends, (starts + ends) / 2.0


def _pool_mean(
    values: np.ndarray,
    frame_times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool a feature-by-frame matrix into deterministic time windows."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (n_features, n_frames)")
    pooled = np.zeros((starts.size, matrix.shape[0]), dtype=np.float32)
    valid = np.zeros(starts.size, dtype=np.bool_)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        left = int(np.searchsorted(frame_times, start, side="left"))
        right = int(np.searchsorted(frame_times, end, side="left"))
        if right <= left:
            nearest = min(max(left, 0), max(matrix.shape[1] - 1, 0))
            right = min(nearest + 1, matrix.shape[1])
            left = nearest
        window = matrix[:, left:right]
        finite = np.isfinite(window)
        if window.size and finite.any():
            counts = finite.sum(axis=1)
            sums = np.where(finite, window, 0.0).sum(axis=1)
            pooled[index] = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums),
                where=counts > 0,
            ).astype(np.float32)
            valid[index] = bool(np.all(counts > 0))
    return pooled, valid


def _tempo_features(
    onset_envelope: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    tempo_axis: np.ndarray,
    librosa_module: Any,
    interp1d_function: Any,
) -> np.ndarray:
    win_length = max(32, int(round(8.0 * sample_rate / hop_length)))
    tempogram = librosa_module.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        norm=np.inf,
    )
    source_bpms = librosa_module.tempo_frequencies(
        tempogram.shape[0], sr=sample_rate, hop_length=hop_length
    )
    keep = np.isfinite(source_bpms) & (source_bpms > 0)
    source_bpms = source_bpms[keep][::-1]
    source_values = tempogram[keep][::-1]
    if source_bpms.size < 2:
        return np.zeros((tempo_axis.size, onset_envelope.size), dtype=np.float32)
    interpolate = interp1d_function(
        source_bpms,
        source_values,
        axis=0,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    return np.asarray(interpolate(tempo_axis), dtype=np.float32)


def _beat_synchronous_pitch(
    chroma: np.ndarray,
    beat_frames: np.ndarray,
    *,
    duration: float,
    sample_rate: int,
    hop_length: int,
    uncertainty_ratio: float,
) -> PitchFeatures:
    n_frames = chroma.shape[1]
    boundaries = np.unique(
        np.clip(
            np.concatenate(
                [np.array([0], dtype=int), beat_frames.astype(int), np.array([n_frames])]
            ),
            0,
            n_frames,
        )
    )
    if boundaries.size < 2:
        boundaries = np.array([0, max(n_frames, 1)], dtype=int)
    pooled: list[np.ndarray] = []
    times: list[float] = []
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        if right <= left:
            continue
        pooled.append(np.mean(chroma[:, left:right], axis=1))
        midpoint = (left + right) / 2.0 * hop_length / sample_rate
        times.append(min(midpoint, duration))
    if not pooled:
        pooled = [np.zeros(12, dtype=np.float32)]
        times = [duration / 2.0]
    beat_chroma = np.asarray(pooled, dtype=np.float32)
    states = np.asarray(
        dominant_pitch_states(beat_chroma, uncertainty_ratio=uncertainty_ratio),
        dtype=np.int16,
    )
    return PitchFeatures(
        times=np.asarray(times, dtype=np.float32),
        chroma=beat_chroma,
        states=states,
        valid=states != 12,
    )


def _rhythm_features(
    onset_envelope: np.ndarray,
    onset_times: np.ndarray,
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    centers: np.ndarray,
) -> RhythmFeatures:
    vectors = np.zeros((starts.size, 8), dtype=np.float32)
    masks = np.zeros_like(vectors, dtype=np.bool_)
    inter_onset_intervals = np.diff(onset_times).astype(np.float32)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        left = int(np.searchsorted(frame_times, start, side="left"))
        right = int(np.searchsorted(frame_times, end, side="left"))
        envelope = onset_envelope[left:right]
        if envelope.size:
            values = np.array(
                [np.mean(envelope), np.std(envelope), np.max(envelope)], dtype=np.float32
            )
            vectors[index, :3] = np.nan_to_num(values)
            masks[index, :3] = np.isfinite(values)

        local_onsets = onset_times[(onset_times >= start) & (onset_times < end)]
        span = max(end - start, np.finfo(float).eps)
        vectors[index, 3] = local_onsets.size / span
        masks[index, 3] = True
        local_ioi = np.diff(local_onsets)
        if local_ioi.size:
            vectors[index, 4] = float(np.mean(local_ioi))
            vectors[index, 5] = float(np.std(local_ioi))
            masks[index, 4:6] = True

        local_beats = beat_times[(beat_times >= start) & (beat_times < end)]
        beat_ioi = np.diff(local_beats)
        if beat_ioi.size and float(np.median(beat_ioi)) > 0:
            vectors[index, 6] = float(60.0 / np.median(beat_ioi))
            masks[index, 6] = True
        vectors[index, 7] = local_beats.size / span
        masks[index, 7] = True

    return RhythmFeatures(
        times=centers.astype(np.float32),
        vectors=vectors,
        valid=masks,
        onset_times=onset_times.astype(np.float32),
        inter_onset_intervals=inter_onset_intervals,
        beat_times=beat_times.astype(np.float32),
    )


def _streaming_mel_envelopes(
    audio: np.ndarray,
    *,
    config: FeatureExtractionConfig,
    librosa_module: Any,
    scipy_fft_module: Any,
    chunk_frames: int = 2_048,
) -> np.ndarray:
    pad = config.n_fft // 2
    mode = "reflect" if audio.size > 1 else "constant"
    padded = np.pad(audio.astype(np.float32, copy=False), (pad, pad), mode=mode)
    frames = librosa_module.util.frame(
        padded,
        frame_length=config.n_fft,
        hop_length=config.modulation_hop_length,
    )
    mel_basis = librosa_module.filters.mel(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        fmin=20.0,
        fmax=config.sample_rate / 2.0,
        dtype=np.float32,
    )
    window = np.hanning(config.n_fft).astype(np.float32)[:, None]
    envelopes = np.empty((config.n_mels, frames.shape[1]), dtype=np.float32)
    for start in range(0, frames.shape[1], chunk_frames):
        stop = min(start + chunk_frames, frames.shape[1])
        spectrum = scipy_fft_module.rfft(
            np.asarray(frames[:, start:stop], dtype=np.float32) * window,
            axis=0,
            workers=1,
        )
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        mel_power = mel_basis @ power
        envelopes[:, start:stop] = np.sqrt(np.maximum(mel_power, 0.0)).astype(np.float32)
    return envelopes


def _band_power(spectrum: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    return float(np.sum(spectrum[mask])) if np.any(mask) else 0.0


def _modulation_features(
    audio: np.ndarray,
    *,
    duration: float,
    config: FeatureExtractionConfig,
    librosa_module: Any,
    scipy_fft_module: Any,
) -> ModulationFeatures:
    envelopes = _streaming_mel_envelopes(
        audio,
        config=config,
        librosa_module=librosa_module,
        scipy_fft_module=scipy_fft_module,
    )
    envelope_rate = config.sample_rate / config.modulation_hop_length
    window_frames = max(2, int(round(config.modulation_window_seconds * envelope_rate)))
    hop_frames = max(1, int(round(config.modulation_hop_seconds * envelope_rate)))
    if envelopes.shape[1] <= window_frames:
        starts = np.array([0], dtype=int)
    else:
        count = (envelopes.shape[1] - window_frames) // hop_frames + 1
        starts = np.arange(count, dtype=int) * hop_frames

    all_frequencies = np.fft.rfftfreq(window_frames, d=1.0 / envelope_rate)
    frequency_mask = (all_frequencies >= config.modulation_min_hz) & (
        all_frequencies <= config.modulation_max_hz
    )
    frequencies = all_frequencies[frequency_mask].astype(np.float32)
    spectra = np.zeros((starts.size, frequencies.size), dtype=np.float32)
    band_energies = np.zeros((starts.size, 6), dtype=np.float32)
    key_band_energies = np.zeros((starts.size, 3), dtype=np.float32)
    valid = np.zeros(starts.size, dtype=np.bool_)
    window = np.hanning(window_frames).astype(np.float32)
    broad_bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 30.0), (30.0, 45.01))
    key_bands = ((7.0, 9.0), (12.0, 20.0), (30.0, 34.0))

    for index, start in enumerate(starts):
        stop = min(start + window_frames, envelopes.shape[1])
        segment = np.zeros((envelopes.shape[0], window_frames), dtype=np.float32)
        segment[:, : stop - start] = envelopes[:, start:stop]
        segment -= np.mean(segment, axis=1, keepdims=True)
        transformed = scipy_fft_module.rfft(segment * window, axis=1, workers=1)
        power = np.sum(
            transformed.real * transformed.real + transformed.imag * transformed.imag,
            axis=0,
        )
        selected = np.asarray(power[frequency_mask], dtype=np.float64)
        total = float(np.sum(selected))
        if np.isfinite(total) and total > np.finfo(float).eps:
            selected /= total
            spectra[index] = selected.astype(np.float32)
            valid[index] = True
        for band_index, (low, high) in enumerate(broad_bands):
            band_energies[index, band_index] = _band_power(selected, frequencies, low, high)
        for band_index, (low, high) in enumerate(key_bands):
            key_band_energies[index, band_index] = _band_power(selected, frequencies, low, high)

    times = np.minimum(
        (starts + window_frames / 2.0) / envelope_rate,
        duration,
    ).astype(np.float32)
    return ModulationFeatures(
        times=times,
        frequencies=frequencies,
        spectrum=spectra,
        band_energies=band_energies,
        key_band_energies=key_band_energies,
        valid=valid,
    )


class LibrosaFeatureExtractor:
    name = "librosa-multiview"
    version = "1.1.0"

    def extract(
        self,
        audio_path: Any,
        *,
        track_id: str,
        config: FeatureExtractionConfig,
    ) -> FeatureBundle:
        librosa, soundfile, scipy_fft, interp1d = _load_audio_dependencies()
        config.validate()
        audio, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=False)
        if sample_rate != config.sample_rate:
            raise FeatureExtractionError(
                f"expected {config.sample_rate} Hz audio, received {sample_rate} Hz"
            )
        if audio.ndim != 1:
            raise FeatureExtractionError("analysis audio must be mono")
        if audio.size == 0:
            raise FeatureExtractionError("analysis audio is empty")
        if not np.all(np.isfinite(audio)):
            raise FeatureExtractionError("analysis audio contains non-finite samples")

        duration = audio.size / sample_rate
        spectrum = librosa.stft(
            audio,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.n_fft,
            window="hann",
            center=True,
        )
        magnitude = np.abs(spectrum).astype(np.float32)
        if config.hpss:
            harmonic, percussive = librosa.decompose.hpss(magnitude)
        else:
            harmonic = percussive = magnitude
        power = magnitude * magnitude
        mel_power = librosa.feature.melspectrogram(
            S=power,
            sr=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            fmin=20.0,
            fmax=sample_rate / 2.0,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_power, ref=1.0, top_db=80.0)
        mfcc_base = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
        delta_width = min(
            9, mfcc_base.shape[1] if mfcc_base.shape[1] % 2 else mfcc_base.shape[1] - 1
        )
        if delta_width >= 3:
            mfcc_delta = librosa.feature.delta(
                mfcc_base, width=delta_width, order=1, mode="nearest"
            )
            mfcc_delta2 = librosa.feature.delta(
                mfcc_base, width=delta_width, order=2, mode="nearest"
            )
        else:
            mfcc_delta = np.zeros_like(mfcc_base)
            mfcc_delta2 = np.zeros_like(mfcc_base)
        mfcc = np.concatenate([mfcc_base, mfcc_delta, mfcc_delta2], axis=0)
        chroma = librosa.feature.chroma_stft(
            S=harmonic * harmonic,
            sr=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            tuning=0.0,
        )
        contrast = librosa.feature.spectral_contrast(
            S=magnitude,
            sr=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_bands=config.contrast_bands,
        )
        percussive_db = librosa.amplitude_to_db(percussive, ref=np.max, top_db=80.0)
        onset_envelope = librosa.onset.onset_strength(
            S=percussive_db,
            sr=sample_rate,
            hop_length=config.hop_length,
            center=False,
        )
        tempo_axis = np.geomspace(
            config.tempo_min_bpm, config.tempo_max_bpm, config.tempo_bins
        ).astype(np.float32)
        tempo_features = _tempo_features(
            onset_envelope,
            sample_rate=sample_rate,
            hop_length=config.hop_length,
            tempo_axis=tempo_axis,
            librosa_module=librosa,
            interp1d_function=interp1d,
        )

        n_frames = min(
            log_mel.shape[1],
            mfcc.shape[1],
            chroma.shape[1],
            contrast.shape[1],
            tempo_features.shape[1],
            onset_envelope.size,
        )
        log_mel = log_mel[:, :n_frames]
        mfcc = mfcc[:, :n_frames]
        chroma = chroma[:, :n_frames]
        contrast = contrast[:, :n_frames]
        tempo_features = tempo_features[:, :n_frames]
        onset_envelope = np.asarray(onset_envelope[:n_frames], dtype=np.float32)
        frame_times = librosa.frames_to_time(
            np.arange(n_frames), sr=sample_rate, hop_length=config.hop_length
        )
        starts, ends, centers = _regular_windows(
            duration, config.analysis_window_seconds, config.analysis_hop_seconds
        )
        pooled_log_mel, valid_log_mel = _pool_mean(log_mel, frame_times, starts, ends)
        pooled_mfcc, valid_mfcc = _pool_mean(mfcc, frame_times, starts, ends)
        pooled_chroma, valid_chroma = _pool_mean(chroma, frame_times, starts, ends)
        pooled_contrast, valid_contrast = _pool_mean(contrast, frame_times, starts, ends)
        pooled_tempo, valid_tempo = _pool_mean(tempo_features, frame_times, starts, ends)
        acoustic_vectors = np.concatenate(
            [
                pooled_log_mel,
                pooled_mfcc,
                pooled_chroma,
                pooled_contrast,
                pooled_tempo,
            ],
            axis=1,
        ).astype(np.float32)
        acoustic_valid = valid_log_mel & valid_mfcc & valid_chroma & valid_contrast & valid_tempo
        acoustic = AcousticFeatures(
            times=centers.astype(np.float32),
            log_mel=pooled_log_mel,
            mfcc=pooled_mfcc,
            chroma=pooled_chroma,
            spectral_contrast=pooled_contrast,
            tempogram=pooled_tempo,
            vectors=acoustic_vectors,
            valid=acoustic_valid,
        )

        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=config.hop_length,
            sparse=True,
        )
        beat_frames = np.asarray(beat_frames, dtype=int)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=config.hop_length,
            units="frames",
            backtrack=False,
        )
        onset_times = librosa.frames_to_time(
            onset_frames, sr=sample_rate, hop_length=config.hop_length
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sample_rate, hop_length=config.hop_length
        )
        pitch = _beat_synchronous_pitch(
            chroma,
            beat_frames,
            duration=duration,
            sample_rate=sample_rate,
            hop_length=config.hop_length,
            uncertainty_ratio=config.pitch_uncertainty_ratio,
        )
        rhythm = _rhythm_features(
            onset_envelope,
            np.asarray(onset_times, dtype=np.float64),
            np.asarray(beat_times, dtype=np.float64),
            frame_times,
            starts,
            ends,
            centers,
        )
        modulation = _modulation_features(
            audio,
            duration=duration,
            config=config,
            librosa_module=librosa,
            scipy_fft_module=scipy_fft,
        )
        structure = structural_features(
            acoustic.vectors,
            acoustic.times,
            acoustic.valid,
            duration=duration,
            kernel_seconds=config.structure_kernel_seconds,
            min_segment_seconds=config.structure_min_segment_seconds,
            max_segment_seconds=config.structure_max_segment_seconds,
            novelty_threshold=config.structure_novelty_threshold,
        )
        return FeatureBundle(
            track_id=track_id,
            duration_seconds=duration,
            acoustic=acoustic,
            pitch=pitch,
            rhythm=rhythm,
            modulation=modulation,
            structure=structure,
        )
