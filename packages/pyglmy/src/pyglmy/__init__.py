"""Path homology, persistent path homology, and point-cloud TDA."""

from .graph import FiltrationDirection, WeightedDiGraph, WeightedEdge
from .path_homology import (
    HomologyGroup,
    PathComplex,
    PathHomology,
    PathHomologyResult,
    PersistenceInterval,
    PersistentPathResult,
    build_path_complex,
    compute_path_homology,
    enumerate_allowed_paths,
    filtration_descriptors,
    path_homology,
    persistent_path_homology,
)
from .tda import (
    RipsResult,
    TDAError,
    delay_embedding,
    diagram_descriptors,
    finite_lifetimes,
    finite_rows,
    normalize_distance_matrix,
    normalize_distance_scale,
    persistence_descriptors,
    uniform_sample,
    vietoris_rips,
)

__all__ = [
    "FiltrationDirection",
    "HomologyGroup",
    "PathComplex",
    "PathHomology",
    "PathHomologyResult",
    "PersistenceInterval",
    "PersistentPathResult",
    "RipsResult",
    "TDAError",
    "WeightedDiGraph",
    "WeightedEdge",
    "__version__",
    "build_path_complex",
    "compute_path_homology",
    "delay_embedding",
    "diagram_descriptors",
    "enumerate_allowed_paths",
    "filtration_descriptors",
    "finite_lifetimes",
    "finite_rows",
    "normalize_distance_matrix",
    "normalize_distance_scale",
    "path_homology",
    "persistence_descriptors",
    "persistent_path_homology",
    "uniform_sample",
    "vietoris_rips",
]

__version__ = "0.1.0"
