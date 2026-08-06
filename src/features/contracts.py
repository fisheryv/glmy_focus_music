from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int16]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    analysis_sample_rate: int = 22_050
    channels: int = 1
    segment_seconds: float = 180.0
    target_lufs: float = -15.0
    beat_grid: float = 1.0


@dataclass(frozen=True, slots=True)
class FeatureExtractionConfig:
    """Frozen MIR parameters shared by extraction and state modelling."""

    sample_rate: int = 22_050
    n_fft: int = 2_048
    hop_length: int = 512
    n_mels: int = 64
    n_mfcc: int = 13
    contrast_bands: int = 6
    tempo_bins: int = 32
    tempo_min_bpm: float = 30.0
    tempo_max_bpm: float = 300.0
    analysis_window_seconds: float = 1.0
    analysis_hop_seconds: float = 0.5
    hpss: bool = True
    modulation_hop_length: int = 128
    modulation_window_seconds: float = 4.0
    modulation_hop_seconds: float = 2.0
    modulation_min_hz: float = 0.5
    modulation_max_hz: float = 45.0
    pitch_uncertainty_ratio: float = 1.15
    rhythm_clusters: int = 10
    acoustic_pca_components: int = 32
    acoustic_clusters: int = 64
    structure_kernel_seconds: float = 8.0
    structure_min_segment_seconds: float = 8.0
    structure_max_segment_seconds: float = 45.0
    structure_novelty_threshold: float = 1.5
    structure_similarity_threshold: float = 0.75
    structure_clusters: int = 16
    max_fit_windows_per_group: int = 50_000
    random_seed: int = 20_260_716

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("n_fft must be a positive even integer")
        if self.hop_length <= 0 or self.modulation_hop_length <= 0:
            raise ValueError("hop lengths must be positive")
        if self.n_mels <= 0 or self.n_mfcc <= 0 or self.tempo_bins <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.contrast_bands != 6:
            raise ValueError("contrast_bands must be 6 to produce seven contrast features")
        if self.analysis_window_seconds <= 0 or self.analysis_hop_seconds <= 0:
            raise ValueError("analysis window and hop must be positive")
        if self.modulation_window_seconds <= 0 or self.modulation_hop_seconds <= 0:
            raise ValueError("modulation window and hop must be positive")
        modulation_rate = self.sample_rate / self.modulation_hop_length
        if self.modulation_max_hz >= modulation_rate / 2:
            raise ValueError("modulation_max_hz must be below the modulation Nyquist rate")
        if not 0 < self.modulation_min_hz < self.modulation_max_hz:
            raise ValueError("invalid modulation frequency range")
        if not 1.0 < self.pitch_uncertainty_ratio:
            raise ValueError("pitch_uncertainty_ratio must be greater than one")
        if self.rhythm_clusters < 2 or self.acoustic_clusters < 2:
            raise ValueError("state cluster counts must be at least two")
        if self.structure_clusters < 2:
            raise ValueError("structure_clusters must be at least two")
        if self.structure_kernel_seconds <= 0 or self.structure_min_segment_seconds <= 0:
            raise ValueError("structure kernel and minimum segment duration must be positive")
        if self.structure_max_segment_seconds < self.structure_min_segment_seconds:
            raise ValueError("structure maximum segment duration must not be shorter than minimum")
        if self.structure_novelty_threshold < 0:
            raise ValueError("structure_novelty_threshold must be non-negative")
        if not 0.0 <= self.structure_similarity_threshold <= 1.0:
            raise ValueError("structure_similarity_threshold must be in [0, 1]")
        if not 1 <= self.acoustic_pca_components <= self.acoustic_dimension:
            raise ValueError("invalid acoustic_pca_components")
        if self.max_fit_windows_per_group <= 0:
            raise ValueError("max_fit_windows_per_group must be positive")

    @property
    def acoustic_dimension(self) -> int:
        return self.n_mels + self.n_mfcc * 3 + 12 + self.contrast_bands + 1 + self.tempo_bins


@dataclass(frozen=True, slots=True)
class AcousticFeatures:
    times: FloatArray
    log_mel: FloatArray
    mfcc: FloatArray
    chroma: FloatArray
    spectral_contrast: FloatArray
    tempogram: FloatArray
    vectors: FloatArray
    valid: BoolArray


@dataclass(frozen=True, slots=True)
class PitchFeatures:
    times: FloatArray
    chroma: FloatArray
    states: IntArray
    valid: BoolArray


@dataclass(frozen=True, slots=True)
class RhythmFeatures:
    times: FloatArray
    vectors: FloatArray
    valid: BoolArray
    onset_times: FloatArray
    inter_onset_intervals: FloatArray
    beat_times: FloatArray


@dataclass(frozen=True, slots=True)
class ModulationFeatures:
    times: FloatArray
    frequencies: FloatArray
    spectrum: FloatArray
    band_energies: FloatArray
    key_band_energies: FloatArray
    valid: BoolArray


@dataclass(frozen=True, slots=True)
class StructureFeatures:
    """Macro sections obtained from an acoustic self-similarity matrix."""

    times: FloatArray
    boundary_times: FloatArray
    boundary_indices: NDArray[np.int32]
    self_similarity: FloatArray
    novelty: FloatArray
    block_vectors: FloatArray
    valid: BoolArray


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Numerical MIR outputs before discovery-only state transformations."""

    track_id: str
    duration_seconds: float
    acoustic: AcousticFeatures
    pitch: PitchFeatures
    rhythm: RhythmFeatures
    modulation: ModulationFeatures
    structure: StructureFeatures


class AudioFeatureExtractor(Protocol):
    name: str
    version: str

    def extract(
        self,
        audio_path: Path,
        *,
        track_id: str,
        config: FeatureExtractionConfig,
    ) -> FeatureBundle: ...
