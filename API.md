# API Reference

This is the canonical contract for the three public functions of `pes_analyzer`. Examples are runnable Python; copy/paste a block to verify behaviour.

## Conventions

- All energy grids are C-contiguous `numpy.ndarray[float64]`. Pass `np.ascontiguousarray(arr)` if you have a non-contiguous view.
- N-D indices are `tuple[int, ...]` in `numpy` axis order.
- `NaN` cells are treated as masked. They are impassable for saddle search and excluded from minimum search.
- Supported dimensionality: N ∈ [2, 7]. The Rust kernels enforce this at the boundary.

---

## `build_dense`

```python
from pes_analyzer.grid import build_dense

def build_dense(
    coords: dict[str, numpy.ndarray],
    values: numpy.ndarray,
) -> tuple[numpy.ndarray[float64], dict[str, numpy.ndarray]]:
    ...
```

### Parameters

- **`coords`** — Ordered mapping `{axis_name: coord_per_row_1d_array}`. The insertion order of the dict determines the axis order of the output ndarray. Each coordinate array must have length `len(values)`.
- **`values`** — 1-D array of scalars (energies or any per-row value).

### Returns

- **`dense`** — C-contiguous `float64` array of shape `(n_unique_axis_0, ..., n_unique_axis_{N-1})`. Missing cells are `np.nan`.
- **`axes`** — `{axis_name: sorted_unique_values_1d_array}`, same key order as `coords`.

### Raises

- `ValueError` if any coord array length disagrees with `len(values)` or if `coords` is empty.

### What it does

Pivots a long-form table of `(coord_0, ..., coord_{N-1}, value)` rows into the dense N-D grid that the other `pes_analyzer` functions consume. Axes are inferred from the unique values per coordinate column, sorted ascending.

### Example

```python
import numpy as np
from pes_analyzer.grid import build_dense

coords = {
    "x": np.array([0.0, 0.0, 1.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
}
values = np.array([10.0, 11.0, 20.0, 21.0])

dense, axes = build_dense(coords, values)
print(dense)
# [[10. 11.]
#  [20. 21.]]
print(axes)
# {'x': array([0., 1.]), 'y': array([0., 1.])}
```

### Notes and edge cases

- **Axis order** follows `coords` insertion order. Swap the dict to swap the axes.
- **Duplicate `(coords, ...)` rows**: last-write-wins.
- **Single-value axes are NOT squeezed.** If one axis has only one unique value, the output retains that length-1 axis. The caller is responsible for filtering active axes before calling.

---

## `find_iwf_grid`

```python
from pes_analyzer.saddle import find_iwf_grid

def find_iwf_grid(
    energies: numpy.ndarray[float64],
    start: tuple[int, ...],
    end: tuple[int, ...],
) -> tuple[tuple[int, ...], float] | None:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are treated as walls.
- **`start`** — N-tuple of `int` grid indices. Must reference a non-`NaN` cell.
- **`end`** — N-tuple of `int` grid indices. Must reference a non-`NaN` cell.

### Returns

- `tuple[tuple[int, ...], float]` — the saddle cell `(index, energy)`.
- `None` — if `start` and `end` lie in disjoint non-`NaN` regions (no path exists).

### Raises

- `ValueError` if `energies` is not C-contiguous, if `start`/`end` have the wrong length, if any index is out of bounds, or if either endpoint cell is `NaN`.

### What it does

Returns the saddle point between `start` and `end` using the imaginary water flow (watershed) algorithm: the lowest-energy cell on any axis-connected non-`NaN` path between them. See [`ALGORITHMS.md`](./ALGORITHMS.md#find_iwf_grid) for the underlying algorithm.

### Example

```python
import numpy as np
from pes_analyzer.saddle import find_iwf_grid

energies = np.full((3, 5), 10.0)
energies[1, 0] = 0.0   # start basin
energies[1, 4] = 0.0   # end basin
energies[1, 1:4] = [1.0, 2.0, 1.0]  # bridge with saddle at (1, 2)

print(find_iwf_grid(energies, start=(1, 0), end=(1, 4)))
# ((1, 2), 2.0)
```

### Notes and edge cases

- **`start == end`**: returns `(start, energies[start])` immediately.
- **`NaN` walls**: cells with `NaN` energy are excluded from the search. If `start` and `end` are not connected through the non-`NaN` region, the function returns `None`.
- **Neighbourhood**: axis-only (2N stencil). Diagonal moves are not allowed — see [`ALGORITHMS.md`](./ALGORITHMS.md#find_iwf_grid) for the rationale.

---

## `find_minima_grid`

```python
from pes_analyzer.extrema import find_minima_grid

def find_minima_grid(
    energies: numpy.ndarray[float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are skipped.
- **`neighborhood_range`** *(keyword-only, default `1`)* — Chebyshev half-width `r` of the neighbor stencil. A cell is compared against every in-bounds neighbor with `max_axis |Δi| ≤ r` (the `(2r+1)ᴺ − 1` stencil). Must satisfy `1 ≤ r ≤ 5`. The default `r = 1` is the classic 3ᴺ−1 king-move stencil.
- **`confirm_range`** *(keyword-only, default `None`)* — optional second-pass check. When `None`, the function returns the direct `neighborhood_range`-stencil minima. When set to an integer `R`, the function first finds candidates at `neighborhood_range` and then re-checks each candidate against the `R`-wide stencil. Must satisfy `1 ≤ R ≤ 5` and `R ≥ neighborhood_range`. The result for `neighborhood_range=1, confirm_range=R` is identical to `neighborhood_range=R` but typically much cheaper, since most cells are culled by the `r=1` pass.

### Returns

- `list[tuple[tuple[int, ...], float]]` — every cell that qualifies as a minimum as `(index, energy)`. Sorted ascending by energy; ties broken by `f64::total_cmp` for determinism.

### Raises

- `ValueError` if `energies` is not C-contiguous, if `energies.ndim` is outside `[2, 7]`, if `neighborhood_range` is outside `[1, 5]`, if `confirm_range` is outside `[1, 5]`, or if `confirm_range < neighborhood_range`.
- `TypeError` if `neighborhood_range` or `confirm_range` is passed positionally or is not an `int`/`None`.
- `OverflowError` if `neighborhood_range` or `confirm_range` is negative.

### What it does

Returns every cell whose energy is **not strictly greater than any non-`NaN` neighbour** in the Chebyshev box of half-width `neighborhood_range`. Equivalently: no neighbour with `max_axis |Δi| ≤ neighborhood_range` has strictly lower energy. Ties are allowed; a cell with one or more equal-energy neighbours still qualifies as long as none is lower. With the default `neighborhood_range = 1` this is the full 3ᴺ−1 (king-move) stencil. See [`ALGORITHMS.md`](./ALGORITHMS.md#find_minima_grid) for the underlying algorithm.

### Example

```python
import numpy as np
from pes_analyzer.extrema import find_minima_grid

energies = np.array([
    [5.0, 5.0, 5.0],
    [5.0, 0.0, 5.0],
    [5.0, 5.0, 5.0],
])

print(find_minima_grid(energies))
# [((1, 1), 0.0)]
```

A larger `neighborhood_range` disqualifies cells beaten by farther neighbours. The example uses `NaN` walls so only the intentional dips qualify:

```python
import numpy as np
from pes_analyzer.extrema import find_minima_grid

energies = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan,    5.0,    5.0,    5.0, np.nan],
    [np.nan,    5.0,    3.0,    5.0, np.nan],
    [np.nan,    5.0,    5.0,    5.0, np.nan],
    [   1.0, np.nan, np.nan, np.nan, np.nan],
], dtype=np.float64)

print(find_minima_grid(energies, neighborhood_range=1))
# [((4, 0), 1.0), ((2, 2), 3.0)]

print(find_minima_grid(energies, neighborhood_range=2))
# [((4, 0), 1.0)]
```

At `r = 1`, `(2, 2)` is a minimum because all eight king-move neighbours are `5.0`. At `r = 2`, the 5×5 stencil around `(2, 2)` reaches `(4, 0) = 1.0` — strictly lower — and `(2, 2)` is no longer reported.

The recommended way to compute the same result more cheaply is to find candidates at `r = 1` and confirm against the wider stencil:

```python
# Same result as neighborhood_range=2, but cheaper: only candidates that
# pass the r=1 check are re-tested against the wider stencil.
print(find_minima_grid(energies, neighborhood_range=1, confirm_range=2))
# [((4, 0), 1.0)]
```

### Notes and edge cases

- **Plateaus are reported.** Every cell on a flat plateau that has no king-move neighbour with strictly lower energy qualifies. In a uniformly-flat region, every cell whose stencil contains no lower neighbour will appear in the output — including corner and edge cells of the plateau where the stencil happens to be entirely within the plateau. If you only want strictly-isolated minima, filter the output yourself.
- **Boundary cells**: the stencil is clipped at array edges. A corner cell has fewer neighbours but is tested with the same rule.
- **`NaN` neighbours** are ignored — a cell is tested only against its non-`NaN` neighbours. A cell with at least one non-`NaN` neighbour and no strictly-lower one is reported; a cell whose entire stencil is `NaN` is not.

---

## `find_maxima_grid`

```python
from pes_analyzer.extrema import find_maxima_grid

def find_maxima_grid(
    energies: numpy.ndarray[float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are skipped.
- **`neighborhood_range`** *(keyword-only, default `1`)* — Chebyshev half-width `r` of the neighbor stencil. Must satisfy `1 ≤ r ≤ 5`.
- **`confirm_range`** *(keyword-only, default `None`)* — optional second-pass check. See [`find_minima_grid`](#find_minima_grid) for the semantics; the rule applies identically here with the comparator flipped.

### Returns

- `list[tuple[tuple[int, ...], float]]` — every cell that qualifies as a maximum as `(index, energy)`. Sorted descending by energy; ties broken by `f64::total_cmp` (reversed) for determinism.

### Raises

Same as `find_minima_grid`: `ValueError`, `TypeError`, `OverflowError` per the validation table.

### What it does

Strict dual of `find_minima_grid`: returns every cell whose energy is **not strictly less than any non-`NaN` neighbour** in the Chebyshev box of half-width `neighborhood_range`. Ties are allowed.

### Example

```python
import numpy as np
from pes_analyzer.extrema import find_maxima_grid

energies = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
])

print(find_maxima_grid(energies))
# [((1, 1), 1.0)]
```

---

## `find_extrema_grid`

```python
from pes_analyzer.extrema import find_extrema_grid

def find_extrema_grid(
    energies: numpy.ndarray[float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> tuple[
    list[tuple[tuple[int, ...], float]],  # minima, ascending
    list[tuple[tuple[int, ...], float]],  # maxima, descending
]:
    ...
```

### Parameters

Same as `find_minima_grid` / `find_maxima_grid`. The two polarities share a single `neighborhood_range` and a single `confirm_range`.

### Returns

A 2-tuple `(minima_list, maxima_list)`. Each list has the same shape as the corresponding single-polarity function would produce. Minima are sorted ascending by energy, maxima descending.

The combined result is byte-identical to `(find_minima_grid(arr, **k), find_maxima_grid(arr, **k))`. The optimisation is purely a constant-factor saving — one stencil walk per cell in the find stage instead of two.

### Raises

Same as `find_minima_grid`.

### Example

```python
import numpy as np
from pes_analyzer.extrema import find_extrema_grid

energies = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 0.0, 2.0],
    [1.0, 2.0, 1.0],
])

mins, maxs = find_extrema_grid(energies)
print(mins)  # [((1, 1), 0.0)]
print(maxs)  # plateau of 2.0s around the rim
```

---

## `find_watershed_segmentation`

```python
from pes_analyzer.topology import find_watershed_segmentation

def find_watershed_segmentation(
    energies: numpy.ndarray[float64],
) -> tuple[
    numpy.ndarray[int32],                              # labels
    list[tuple[tuple[int, ...], float]],               # basins
    list[tuple[tuple[int, ...], float, int, int]],     # merges
]:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are masked.

### Returns

A 3-tuple `(labels, basins, merges)`:

- **`labels`** — `int32` array with shape == `energies.shape`. `labels[cell] == -1` iff `energies[cell]` is NaN, otherwise the basin ID the cell first joined during the flood.
- **`basins`** — `list[tuple[tuple[int, ...], float]]`, one entry per basin. `(min_nd_index, min_energy)`, sorted ascending by energy. `basins[0]` always contains the global minimum cell.
- **`merges`** — `list[tuple[tuple[int, ...], float, int, int]]`, one entry per union that merged two distinct basins. `(saddle_nd_index, saddle_energy, deeper_basin_id, shallower_basin_id)`, sorted ascending by saddle energy. The convention `basins[deeper].min_e <= basins[shallower].min_e` always holds.

### Raises

- `ValueError` if `energies` is not C-contiguous or if `energies.ndim` is outside `[2, 7]`.

### What it does

Runs the imaginary-water-flow flood to completion (not just until two specified endpoints connect). Records every union that merges two previously-disconnected basins as a `(saddle, deeper, shallower)` event. This is the full segmentation that `find_iwf_grid` partially computes — `find_iwf_grid` is the two-point specialization that stops at the first basin-merge between its two endpoint cells.

### Example

```python
import numpy as np
from pes_analyzer.topology import find_watershed_segmentation

energies = np.full((5, 5), 10.0)
energies[0, 0] = 0.0
energies[4, 4] = 0.0
energies[2, :] = [3.0, 3.0, 4.0, 3.0, 3.0]   # bridge with saddle at (2, 2) = 4

labels, basins, merges = find_watershed_segmentation(energies)
print(basins)
# [((0, 0), 0.0), ((4, 4), 0.0)]
print(merges)
# [((2, 2), 4.0, 0, 1)]   # or (..., 1, 0) — both basins have min_e == 0.0
```

---

## Topology helpers

The pure-Python helpers in `pes_analyzer.topology` analyse the merge tree produced by `find_watershed_segmentation`.

### `compute_persistence(basins, merges) -> numpy.ndarray[float64]`

Per-basin topological persistence. The deepest basin (`basins[0]`) has persistence `+inf`. Every other basin has persistence `saddle_energy − basin_minimum_energy`, computed from the merge event in which the basin was absorbed.

### `prune_merge_tree(basins, merges, threshold) -> (list[int], list[merge])`

Drops basins whose persistence is strictly less than `threshold`. Returns `(surviving_basin_ids, kept_merges)`. `kept_merges` is the subset of input merges whose `shallower` basin survives; the `deeper` basin always survives too (proof: if a kept merge has `shallower.persistence >= threshold` and `deeper.min_e <= shallower.min_e`, then `deeper.persistence >= shallower.persistence >= threshold`).

### `identify_critical_points(basins, merges, threshold, *, gs_disqualifier=None) -> dict`

Walks the pruned merge tree outward from the ground state by Prim-style edge relaxation: at each step, the lowest-saddle-energy edge with one end visited and one end not is consumed. Returns a dict with keys `ground_state`, `secondary_minimum`, `inner_saddle`, `outer_saddle`, `fission_exit`. Each value is either a basin ID, a merge tuple, or `None`.

Three regimes:

| Pruned tree | Result |
|---|---|
| Just GS | only `ground_state` is set |
| GS + 1 outward basin (SHE-typical) | `ground_state`, `inner_saddle` (the one saddle), `fission_exit` set; SM and outer_saddle are `None` |
| GS + 2+ outward basins (actinide-typical) | all five fields populated |

The optional `gs_disqualifier` is a `Callable[[int], bool]`; returning `True` marks a basin as ineligible to be the ground state, and the picker falls back to the next-deepest surviving basin. Used by `MapMaker_FoS_SHE.py` to apply the `c <= 1.25` physical constraint.

### Example

```python
import numpy as np
from pes_analyzer.topology import (
    find_watershed_segmentation, identify_critical_points,
)

energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
_, basins, merges = find_watershed_segmentation(energies)
cp = identify_critical_points(basins, merges, threshold=2.0)
print(cp["secondary_minimum"])   # 1   (the SM basin id)
print(cp["fission_exit"])        # 2   (the FE basin id)
print(cp["inner_saddle"][1])     # 8.0 (inner saddle energy)
print(cp["outer_saddle"][1])     # 5.0 (outer saddle energy, lower than inner)
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: energies must be C-contiguous` | passed a slice or transposed array | `np.ascontiguousarray(arr)` before calling |
| `ValueError: ndim must be in [2, 7]` | wrong array shape | reshape or filter inactive axes |
| `ValueError: index ... out of bounds for shape ...` | `start`/`end` outside the grid | check tuple length and values |
| `ValueError: energy at \`start\` is NaN` | endpoint cell is masked | pick an endpoint inside the non-`NaN` region |
| `ValueError: neighborhood_range must be in [1, 5]` | passed `0` or `> 5` | choose `neighborhood_range ∈ {1, 2, 3, 4, 5}` |
| `ValueError: confirm_range must be in [1, 5]` | passed `0` or `> 5` | choose `confirm_range ∈ {1, 2, 3, 4, 5}` or `None` |
| `ValueError: confirm_range (c) must be >= neighborhood_range (n)` | `confirm_range < neighborhood_range` | raise `confirm_range` or lower `neighborhood_range` (most callers want `neighborhood_range=1, confirm_range=R`) |
