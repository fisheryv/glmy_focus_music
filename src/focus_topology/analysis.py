"""Stable, high-level API for directed music-topology analysis."""

from __future__ import annotations

import json
import math
import operator
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .graphs.transition import TransitionGraph, build_transition_graph
from .homology.glmy import PersistentPathResult, persistent_path_homology

DEFAULT_THRESHOLDS = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration shared by one or more topology analyses.

    Thresholds are outgoing transition-probability cutoffs. They are
    deduplicated and stored in descending order so the graph filtration only
    adds edges.
    """

    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    top_k: int | None = 6
    include_self_loops: bool = False
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        try:
            thresholds = tuple(
                sorted({float(value) for value in self.thresholds}, reverse=True)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("thresholds must be an iterable of numbers") from exc
        if not thresholds:
            raise ValueError("at least one threshold is required")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("thresholds must be finite values in [0, 1]")
        top_k = self.top_k
        if top_k is not None:
            if isinstance(top_k, bool):
                raise ValueError("top_k must be a positive integer or None")
            try:
                top_k = operator.index(top_k)
            except TypeError as exc:
                raise ValueError("top_k must be a positive integer or None") from exc
            if top_k < 1:
                raise ValueError("top_k must be a positive integer or None")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive finite number")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "top_k", top_k)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass(frozen=True, slots=True)
class TopologyAnalysis:
    """Complete result of analyzing one musical-state sequence."""

    states: tuple[Hashable | None, ...]
    graph: TransitionGraph
    persistence: PersistentPathResult
    metrics: Mapping[str, int | float]
    config: AnalysisConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def betti_curve(self, dimension: int) -> tuple[int, ...]:
        """Return the Betti-number curve for H0 or H1."""

        if dimension not in (0, 1):
            raise ValueError("persistent results are available for dimensions 0 and 1")
        return tuple(
            int(row[f"h{dimension}_betti"]) for row in self.persistence.descriptors
        )

    def to_dict(self, *, include_states: bool = True) -> dict[str, Any]:
        """Convert the result to JSON-compatible built-in Python values."""

        payload: dict[str, Any] = {
            "schema_version": 1,
            "config": asdict(self.config),
            "metadata": _json_safe(self.metadata),
            "metrics": _json_safe(self.metrics),
            "graph": {
                "vertices": [_json_safe(vertex) for vertex in self.graph.vertices],
                "edges": [
                    {
                        "source": _json_safe(edge.source),
                        "target": _json_safe(edge.target),
                        "weight": edge.weight,
                        "count": edge.count,
                    }
                    for edge in self.graph.edges
                ],
            },
            "persistence": {
                "thresholds": list(self.persistence.thresholds),
                "descriptors": [_json_safe(row) for row in self.persistence.descriptors],
                "intervals": [
                    _json_safe(asdict(interval)) for interval in self.persistence.intervals
                ],
                "h0_rank_invariant": self.persistence.h0_rank_invariant.tolist(),
                "h1_rank_invariant": self.persistence.h1_rank_invariant.tolist(),
            },
        }
        if include_states:
            payload["states"] = [_json_safe(state) for state in self.states]
        return payload

    def to_json(self, *, indent: int | None = 2, include_states: bool = True) -> str:
        """Serialize the result as UTF-8-compatible JSON text."""

        return json.dumps(
            self.to_dict(include_states=include_states),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        include_states: bool = True,
    ) -> Path:
        """Write a result JSON file and return its resolved path."""

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            self.to_json(indent=indent, include_states=include_states) + "\n",
            encoding="utf-8",
        )
        return output


def _validate_states(states: Iterable[Hashable | None]) -> tuple[Hashable | None, ...]:
    sequence = tuple(states)
    if not sequence:
        raise ValueError("states must contain at least one observation")
    for index, state in enumerate(sequence):
        if state is None:
            continue
        try:
            hash(state)
        except TypeError as exc:
            raise ValueError(f"state at index {index} is not hashable") from exc
    return sequence


def _sequence_metrics(
    states: tuple[Hashable | None, ...],
    graph: TransitionGraph,
) -> dict[str, int | float]:
    pairs = [
        (source, target)
        for source, target in zip(states, states[1:], strict=False)
        if source is not None and target is not None
    ]
    transition_counts = Counter(pairs)
    source_counts = Counter(source for source, _ in pairs)
    transition_total = len(pairs)
    probabilities = np.asarray(list(transition_counts.values()), dtype=float)
    if probabilities.size:
        probabilities /= probabilities.sum()
        transition_entropy = float(-np.sum(probabilities * np.log(probabilities)))
        if probabilities.size > 1:
            transition_entropy /= math.log(probabilities.size)
    else:
        transition_entropy = 0.0

    if transition_total:
        path_entropy = float(
            -sum(
                (count / transition_total) * math.log(count / source_counts[source])
                for (source, _), count in transition_counts.items()
            )
        )
        directed_recurrence = float(
            sum(count * count for count in transition_counts.values()) / transition_total**2
        )
    else:
        path_entropy = 0.0
        directed_recurrence = 0.0

    observed_state_count = len({state for state in states if state is not None})
    off_diagonal_edges = graph.edge_pairs
    off_diagonal_edge_set = set(off_diagonal_edges)
    reciprocal = sum(
        (target, source) in off_diagonal_edge_set for source, target in off_diagonal_edges
    )
    possible_edges = len(graph.vertices) * max(0, len(graph.vertices) - 1)
    return {
        "sequence_length": len(states),
        "valid_states": sum(state is not None for state in states),
        "valid_transitions": transition_total,
        "self_transitions": sum(source == target for source, target in pairs),
        "self_transition_ratio": (
            sum(source == target for source, target in pairs) / transition_total
            if transition_total
            else 0.0
        ),
        "vertex_count": len(graph.vertices),
        "edge_count": len(graph.edges),
        "edge_density": len(off_diagonal_edges) / possible_edges if possible_edges else 0.0,
        "reciprocity": reciprocal / len(off_diagonal_edges) if off_diagonal_edges else 0.0,
        "transition_entropy": transition_entropy,
        "path_entropy": path_entropy,
        "path_entropy_normalized": (
            path_entropy / math.log(observed_state_count) if observed_state_count > 1 else 0.0
        ),
        "directed_recurrence": directed_recurrence,
        "directed_recurrence_unbiased": (
            sum(count * (count - 1) for count in transition_counts.values())
            / (transition_total * (transition_total - 1))
            if transition_total > 1
            else 0.0
        ),
    }


def _topology_metrics(result: PersistentPathResult) -> dict[str, int | float]:
    thresholds = np.asarray(result.thresholds, dtype=float)
    widths = -np.diff(thresholds)
    output: dict[str, int | float] = {}
    for dimension in (0, 1):
        betti = np.asarray(
            [row[f"h{dimension}_betti"] for row in result.descriptors],
            dtype=float,
        )
        auc = (
            float(np.sum((betti[:-1] + betti[1:]) * 0.5 * widths))
            if betti.size > 1
            else 0.0
        )
        intervals = [
            interval for interval in result.intervals if interval.dimension == dimension
        ]
        output.update(
            {
                f"h{dimension}_betti_auc": auc,
                f"h{dimension}_betti_mean": float(np.mean(betti)),
                f"h{dimension}_betti_max": int(np.max(betti)),
                f"h{dimension}_interval_count": sum(
                    interval.multiplicity for interval in intervals
                ),
                f"h{dimension}_observed_persistence": float(
                    sum(
                        interval.lifetime * interval.multiplicity
                        for interval in intervals
                    )
                ),
                f"h{dimension}_censored_count": sum(
                    interval.multiplicity
                    for interval in intervals
                    if interval.censored
                ),
            }
        )
    return output


class TopologyAnalyzer:
    """Reusable analyzer with frozen graph and persistence settings."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        states: Iterable[Hashable | None],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TopologyAnalysis:
        """Analyze an iterable of discrete musical states."""

        sequence = _validate_states(states)
        graph = build_transition_graph(
            sequence,
            normalize=True,
            top_k=self.config.top_k,
            include_self_loops=self.config.include_self_loops,
        )
        persistence = persistent_path_homology(
            graph,
            self.config.thresholds,
            tolerance=self.config.tolerance,
        )
        metrics = {
            **_sequence_metrics(sequence, graph),
            **_topology_metrics(persistence),
        }
        return TopologyAnalysis(
            states=sequence,
            graph=graph,
            persistence=persistence,
            metrics=metrics,
            config=self.config,
            metadata=dict(metadata or {}),
        )


def analyze_states(
    states: Iterable[Hashable | None],
    *,
    config: AnalysisConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TopologyAnalysis:
    """One-call convenience API for a discrete musical-state sequence."""

    return TopologyAnalyzer(config).analyze(states, metadata=metadata)
