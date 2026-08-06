"""Directed persistent path homology for musical state sequences.

The top-level package exposes the stable public API. Lower-level graph and
homology primitives remain available from :mod:`focus_topology.graphs` and
:mod:`focus_topology.homology`.
"""

from .analysis import (
    DEFAULT_THRESHOLDS,
    AnalysisConfig,
    TopologyAnalysis,
    TopologyAnalyzer,
    analyze_states,
)
from .audio import analyze_audio, states_from_audio
from .graphs import TransitionGraph, WeightedEdge, build_transition_graph
from .homology import (
    HomologyGroup,
    PersistenceInterval,
    PersistentPathResult,
    compute_path_homology,
    filtration_descriptors,
    persistent_path_homology,
)
from .pipeline import analyze_state_sequence

__all__ = [
    "DEFAULT_THRESHOLDS",
    "AnalysisConfig",
    "HomologyGroup",
    "PersistenceInterval",
    "PersistentPathResult",
    "TopologyAnalysis",
    "TopologyAnalyzer",
    "TransitionGraph",
    "WeightedEdge",
    "__version__",
    "analyze_audio",
    "analyze_state_sequence",
    "analyze_states",
    "build_transition_graph",
    "compute_path_homology",
    "filtration_descriptors",
    "persistent_path_homology",
    "states_from_audio",
]

__version__ = "0.3.0"
