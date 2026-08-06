"""Optional audio-file convenience API."""

from __future__ import annotations

import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .analysis import AnalysisConfig, TopologyAnalysis, analyze_states

if TYPE_CHECKING:
    from .features.contracts import FeatureExtractionConfig

AudioView = Literal["pitch", "modulation", "structure"]


def _audio_dependencies() -> tuple[type, type, object, object]:
    try:
        from .features.contracts import FeatureExtractionConfig
        from .features.extractor import LibrosaFeatureExtractor
        from .features.states import modulation_states, quantile_edges
    except ImportError as exc:
        raise ImportError(
            "audio analysis dependencies are missing; install with the 'audio' extra"
        ) from exc
    return (
        FeatureExtractionConfig,
        LibrosaFeatureExtractor,
        modulation_states,
        quantile_edges,
    )


def _extract_bundle(
    path: Path,
    *,
    track_id: str,
    config: FeatureExtractionConfig,
    extractor_type: type,
) -> object:
    """Load common audio formats and canonicalize sample rate/channels when needed."""

    try:
        import librosa
        import soundfile
    except ImportError as exc:
        raise ImportError(
            "audio analysis dependencies are missing; install with the 'audio' extra"
        ) from exc

    try:
        info = soundfile.info(str(path))
        is_canonical = (
            info.samplerate == config.sample_rate
            and info.channels == 1
        )
    except (OSError, RuntimeError):
        is_canonical = False

    if is_canonical:
        return extractor_type().extract(path, track_id=track_id, config=config)

    audio, _ = librosa.load(
        str(path),
        sr=config.sample_rate,
        mono=True,
        dtype=np.float32,
    )
    if audio.size == 0:
        raise ValueError("analysis audio is empty")
    with tempfile.TemporaryDirectory(prefix="focus_topology_") as temporary:
        canonical = Path(temporary) / "canonical.wav"
        soundfile.write(
            canonical,
            np.asarray(audio, dtype=np.float32),
            config.sample_rate,
            subtype="FLOAT",
        )
        return extractor_type().extract(
            canonical,
            track_id=track_id,
            config=config,
        )


def states_from_audio(
    audio_path: str | Path,
    *,
    view: AudioView = "pitch",
    track_id: str | None = None,
    feature_config: FeatureExtractionConfig | None = None,
) -> tuple[tuple[Hashable | None, ...], dict[str, object]]:
    """Extract a standalone pitch, modulation, or structural state sequence.

    ``pitch`` uses beat-synchronous dominant pitch classes. ``modulation`` uses
    per-track quantile states and is intended for standalone exploration. For
    comparisons across a dataset, fit and reuse one shared state model with the
    research batch pipeline.
    """

    (
        feature_config_type,
        extractor_type,
        modulation_states_function,
        quantile_edges_function,
    ) = _audio_dependencies()
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    if view not in ("pitch", "modulation", "structure"):
        raise ValueError("view must be 'pitch', 'modulation', or 'structure'")

    config = feature_config or feature_config_type()
    identifier = track_id or path.stem
    bundle = _extract_bundle(
        path,
        track_id=identifier,
        config=config,
        extractor_type=extractor_type,
    )

    if view == "pitch":
        states: tuple[Hashable | None, ...] = tuple(
            int(state) if bool(valid) else None
            for state, valid in zip(
                bundle.pitch.states,
                bundle.pitch.valid,
                strict=True,
            )
        )
        method = "beat-synchronous dominant pitch class"
    elif view == "modulation":
        energies = np.asarray(bundle.modulation.key_band_energies, dtype=float)
        valid = np.asarray(bundle.modulation.valid, dtype=bool)
        if not np.any(valid):
            raise ValueError("audio contains no valid modulation windows")
        edges_by_band = [
            quantile_edges_function(energies[valid, band], 3)
            for band in range(energies.shape[1])
        ]
        quantized = modulation_states_function(
            energies,
            levels=3,
            edges_by_band=edges_by_band,
        )
        states = tuple(
            tuple(int(value) for value in state) if is_valid else None
            for state, is_valid in zip(quantized, valid, strict=True)
        )
        method = "per-track three-level modulation-band quantiles"
    else:
        from .features.structure import abstract_structure_states

        states = tuple(
            abstract_structure_states(
                bundle.structure.block_vectors,
                bundle.structure.valid,
                similarity_threshold=config.structure_similarity_threshold,
            )
        )
        method = "SSM novelty boundaries with recurrent macro-section states"

    metadata: dict[str, object] = {
        "source": str(path),
        "track_id": identifier,
        "view": view,
        "duration_seconds": float(bundle.duration_seconds),
        "analysis_sample_rate": int(config.sample_rate),
        "state_extraction": method,
    }
    if view == "structure":
        metadata["boundary_seconds"] = [
            float(value) for value in bundle.structure.boundary_times
        ]
        metadata["structure_blocks"] = int(bundle.structure.times.size)
    return states, metadata


def analyze_audio(
    audio_path: str | Path,
    *,
    view: AudioView = "pitch",
    track_id: str | None = None,
    config: AnalysisConfig | None = None,
    feature_config: FeatureExtractionConfig | None = None,
) -> TopologyAnalysis:
    """Extract states and run directed persistent path homology on one audio file."""

    states, metadata = states_from_audio(
        audio_path,
        view=view,
        track_id=track_id,
        feature_config=feature_config,
    )
    return analyze_states(states, config=config, metadata=metadata)
