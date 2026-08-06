"""Small graph and point-cloud examples."""

import numpy as np

from pathhom_tda import (
    WeightedDiGraph,
    path_homology,
    persistent_path_homology,
    vietoris_rips,
)

cycle = [(0, 1), (1, 2), (2, 0)]
print(path_homology([0, 1, 2], cycle).betti_numbers)

weighted = WeightedDiGraph.from_edges(
    [(source, target, 0.9) for source, target in cycle]
)
print(
    persistent_path_homology(
        weighted,
        [0.95, 0.8],
    ).betti_curve(1)
)

theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
points = np.column_stack([np.cos(theta), np.sin(theta)])
print(vietoris_rips(points, normalize=True).diagram(1))
