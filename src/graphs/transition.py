from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

State = TypeVar("State", bound=Hashable)


@dataclass(frozen=True, slots=True)
class WeightedEdge:
    source: Hashable
    target: Hashable
    weight: float
    count: int


@dataclass(frozen=True, slots=True)
class TransitionGraph:
    vertices: tuple[Hashable, ...]
    edges: tuple[WeightedEdge, ...]

    def threshold(self, minimum_weight: float) -> "TransitionGraph":
        if not 0.0 <= minimum_weight <= 1.0:
            raise ValueError("minimum_weight must be in [0, 1]")
        kept = tuple(edge for edge in self.edges if edge.weight >= minimum_weight)
        return TransitionGraph(vertices=self.vertices, edges=kept)

    @property
    def edge_pairs(self) -> tuple[tuple[Hashable, Hashable], ...]:
        return tuple((edge.source, edge.target) for edge in self.edges if edge.source != edge.target)


def build_transition_graph(
    states: Sequence[State | None] | Iterable[State | None],
    *,
    normalize: bool = True,
    top_k: int | None = None,
    include_self_loops: bool = False,
) -> TransitionGraph:
    """Build a deterministic weighted digraph from consecutive states.

    Weights are outgoing transition probabilities by default. Self-transitions
    can be retained for descriptive work, but regular GLMY paths ignore loops.
    """

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    sequence = list(states)
    counts: Counter[tuple[State, State]] = Counter()
    outgoing: Counter[State] = Counter()
    observed: set[State] = {state for state in sequence if state is not None}
    for source, target in zip(sequence, sequence[1:]):
        if source is None or target is None:
            continue
        if source == target and not include_self_loops:
            continue
        counts[(source, target)] += 1
        outgoing[source] += 1

    edges_by_source: dict[State, list[WeightedEdge]] = defaultdict(list)
    for (source, target), count in counts.items():
        weight = count / outgoing[source] if normalize else float(count)
        edges_by_source[source].append(WeightedEdge(source, target, weight, count))

    selected: list[WeightedEdge] = []
    for source in sorted(edges_by_source, key=repr):
        ranked = sorted(
            edges_by_source[source], key=lambda edge: (-edge.weight, repr(edge.target))
        )
        selected.extend(ranked if top_k is None else ranked[:top_k])

    vertices = tuple(sorted(observed, key=repr))
    edges = tuple(sorted(selected, key=lambda edge: (repr(edge.source), repr(edge.target))))
    return TransitionGraph(vertices=vertices, edges=edges)

