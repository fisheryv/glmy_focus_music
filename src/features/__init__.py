"""Audio features and feature-to-state transformations."""

from .contracts import (
    AcousticFeatures,
    AudioFeatureExtractor,
    FeatureBundle,
    FeatureExtractionConfig,
    ModulationFeatures,
    PitchFeatures,
    PreprocessConfig,
    RhythmFeatures,
    StructureFeatures,
)
from .extractor import FeatureExtractionError, LibrosaFeatureExtractor
from .states import dominant_pitch_states, modulation_states, quantize_1d
from .structure import (
    abstract_structure_states,
    checkerboard_novelty,
    self_similarity_matrix,
    structural_boundaries,
    structural_features,
)

__all__ = [
    "AcousticFeatures",
    "AudioFeatureExtractor",
    "FeatureBundle",
    "FeatureExtractionConfig",
    "FeatureExtractionError",
    "LibrosaFeatureExtractor",
    "ModulationFeatures",
    "PitchFeatures",
    "PreprocessConfig",
    "RhythmFeatures",
    "StructureFeatures",
    "abstract_structure_states",
    "checkerboard_novelty",
    "dominant_pitch_states",
    "modulation_states",
    "quantize_1d",
    "self_similarity_matrix",
    "structural_boundaries",
    "structural_features",
]
