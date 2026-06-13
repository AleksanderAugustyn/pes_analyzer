# Usage: the pes_analyzer pipeline end-to-end

This walks the typical workflow from a raw long-form table to labelled basins and
a minimum-energy path. Every function here has a full contract in `API.md`; this
page shows how they compose. For algorithm details and the neighborhood-stencil
rationale, see `ALGORITHMS.md`.

## The pipeline

```
build_dense            sparse (coords, value) rows  ->  dense N-D grid
find_minima_grid       dense grid                   ->  local minima
find_watershed_segmentation  dense grid             ->  basin labels + merge tree
MergeTree              (labels, basins, merges)     ->  traversable basin tree
find_minimum_energy_path + analyze_path_profile     ->  barrier profile between two cells
```

## 1. Build the dense grid

`build_dense` pivots a long-form table into the dense `float64` array every other
function consumes. Dict insertion order fixes the axis order; missing cells become
`NaN`.

```python
import numpy as np
from pes_analyzer.grid import build_dense

coords = {
    "x": np.array([0.0, 0.0, 1.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
}
values = np.array([10.0, 11.0, 20.0, 21.0])

energies, axes = build_dense(coords, values)
# energies.shape == (2, 2); axes == {"x": [0., 1.], "y": [0., 1.]}
```

Map an N-D grid index back to physical coordinates with `axes`: cell `(i, j)` sits
at `(axes["x"][i], axes["y"][j])`.

## 2. Find the minima

```python
from pes_analyzer.extrema import find_minima_grid

minima = find_minima_grid(energies)            # [((i, ...), energy), ...] ascending
```

Extrema use the **Chebyshev king-move stencil** (`3**N - 1` neighbours by default).
Widen with `neighborhood_range=R`, or use the cheaper two-pass idiom
`find_minima_grid(energies, neighborhood_range=1, confirm_range=R)` for an
identical result. `find_maxima_grid` and `find_extrema_grid` are the dual and the
combined single-sweep variants.

## 3. Segment the whole surface

`find_watershed_segmentation` floods the entire grid and records every basin merge
as a saddle event — the full generalization of the two-point `find_iwf_grid`.

```python
from pes_analyzer.topology import find_watershed_segmentation

labels, basins, merges = find_watershed_segmentation(energies)
# labels: int32 array (shape == energies.shape); -1 marks NaN cells
# basins: [((min_index, ...), min_energy), ...] ascending; basins[0] is the global min
# merges: [((saddle_index, ...), saddle_energy, deeper_id, shallower_id), ...] ascending
```

## 4. Traverse the merge tree

`MergeTree` turns the `(labels, basins, merges)` triple into a rooted, traversable
tree. It is **physics-free**: it gives you neutral traversal, membership, and
geometry primitives, and you compose them with your own predicates to label ground
states, fission exits, or whatever your domain needs.

```python
from pes_analyzer.topology import MergeTree

tree = MergeTree(labels, basins, merges)
tree.root                       # basin id 0 (global minimum), or None if no basins
node = tree.node(0)             # BasinNode: minimum_index, minimum_energy, parent,
                                #            children, saddle_to_parent, persistence
tree.persistence(1)             # saddle_energy - basin_min_energy (root is +inf)
tree.path(1, 2)                 # tree path through the lowest common ancestor
tree.basin_of_point((0, 1))     # basin id at a grid cell (-1 for NaN)
tree.touches_edge(1, axis=0)    # does basin 1 reach an axis-0 boundary?
```

Filter noise with persistence: `compute_persistence(basins, merges)` gives the
per-basin value, and `prune_merge_tree(basins, merges, threshold)` drops basins
below a persistence floor.

## 5. Profile the barrier between two cells

`find_minimum_energy_path` returns the deep minimax path: it minimizes the highest
energy crossed (passing through exactly the watershed saddles) and descends to true
basin minima between barriers. Feed the energy profile to `analyze_path_profile` to
extract the alternating minima and saddles.

```python
from pes_analyzer.topology import find_minimum_energy_path, analyze_path_profile

result = find_minimum_energy_path(energies, start=(0, 0), end=(1, 1))
if result is not None:
    path_indices, path_energies = result        # (K, N) int64, (K,) float64
    profile = analyze_path_profile(path_energies, min_persistence=0.0)
    # profile.minima, profile.saddles: lists of (path_index, energy)
    # recover grid coords of path_index k via path_indices[k]
```
`find_minimum_energy_path` returns `None` when `start` and `end` lie in disjoint
non-`NaN` regions.

## Contracts you must get right

- **Neighborhood asymmetry.** Flood kernels (`find_iwf_grid`,
  `find_watershed_segmentation`, `find_minimum_energy_path`) default to axis-only
  von Neumann neighbours (`2N`), with opt-in `neighborhood="moore"` (`3**N - 1`).
  Extrema use the Chebyshev king-move stencil with `neighborhood_range`. This
  asymmetry is intentional — see `ALGORITHMS.md`.
- **Match `neighborhood` across the tree and the path.** Pass the *same*
  `neighborhood` to `find_watershed_segmentation` and `find_minimum_energy_path`,
  or their saddles will disagree. For one `neighborhood`, `max(path_energies)`
  equals the `find_iwf_grid` saddle energy between the same endpoints exactly.
- **`NaN` cells are impassable walls.** They are excluded from sorting and
  neighbour scans. A `NaN` at a required `start`/`end` is a usage error and raises
  `ValueError`.
- **Arrays must be C-contiguous `float64`** of ndim `N ∈ [2, 7]`. Pass a slice or
  transpose through `np.ascontiguousarray(arr)` first.

## Locating these docs at runtime

```python
import pes_analyzer
pes_analyzer.docs_path()        # -> Path to the bundled _docs/ directory
```
