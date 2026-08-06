# pathhom-tda

`pathhom-tda` is a domain-independent Python library for:

- real-coefficient GLMY path homology of directed graphs;
- finite persistent path homology of weighted digraph filtrations;
- Vietoris–Rips persistence of point clouds or distance matrices;
- deterministic point-cloud sampling, distance normalization, delay embedding,
  and persistence-diagram descriptors.

The core path-homology implementation depends only on NumPy. Ripser.py is an
optional backend installed by the `tda` extra.

## Install

From this monorepo:

```powershell
python -m pip install -e packages/pathhom_tda
python -m pip install -e "packages/pathhom_tda[tda]"
```

## Path homology

```python
from pathhom_tda import path_homology

result = path_homology(
    vertices=[0, 1, 2],
    edges=[(0, 1), (1, 2), (2, 0)],
    max_dimension=1,
)

print(result.betti_numbers)  # (1, 1)
print(result.complex.allowed_paths[1])
```

The returned `PathHomologyResult` exposes both group summaries and the computed
GLMY chain complex. `compute_path_homology(...)` is a smaller compatibility
function that returns only `HomologyGroup` objects.

## Persistent path homology

```python
from pathhom_tda import WeightedDiGraph, persistent_path_homology

graph = WeightedDiGraph.from_edges(
    [
        (0, 1, 0.9),
        (1, 2, 0.9),
        (2, 0, 0.9),
        (0, 2, 0.4),
    ]
)

result = persistent_path_homology(
    graph,
    thresholds=(0.95, 0.8, 0.3),
    max_dimension=2,
    direction="superlevel",
)

print(result.betti_curve(1))
print(result.intervals)
print(result.rank_invariant(1))
```

Use `direction="superlevel"` for similarities or probabilities, where larger
weights enter first. Use `direction="sublevel"` for distances or costs, where
smaller weights enter first.

The reference backend explicitly enumerates regular allowed paths and uses
dense SVD over the real numbers. It favors mathematical transparency and
cross-checking; high-dimensional dense graphs can grow exponentially.

## Vietoris–Rips TDA

```python
import numpy as np
from pathhom_tda import vietoris_rips

theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
circle = np.column_stack([np.cos(theta), np.sin(theta)])

result = vietoris_rips(
    circle,
    max_dimension=1,
    coefficient=2,
    normalize=True,
)

print(result.diagram(1))
print(result.descriptors(prominent_lifetime=0.1))
```

`vietoris_rips` follows the common Ripser.py vocabulary while returning a
typed `RipsResult`. It also supports `distance_matrix=True`, deterministic
`max_points`, representative cocycles, and landmark permutation through
`n_perm`.

## Command line

Graph JSON:

```json
{
  "vertices": [0, 1, 2],
  "edges": [[0, 1, 0.9], [1, 2, 0.9], [2, 0, 0.9]]
}
```

Commands:

```powershell
pathhom-tda path graph.json --max-dimension 2
pathhom-tda pph graph.json --levels 0.95,0.8,0.5
pathhom-tda rips points.csv --max-dimension 1 --normalize
```

## Scope and references

The API is inspired by the small graph-to-Betti workflow in
[PathHom](https://github.com/WeilabMSU/PathHom) and the structured
point-cloud workflow in
[Ripser.py](https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html).
The implementation in this repository is independently organized around the
existing GLMY and TDA code; no external source files are copied.

Before publishing this package publicly, the project owner must select an
explicit software license and a final package name.
