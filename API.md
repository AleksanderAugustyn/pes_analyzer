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
from pes_analyzer.minimum import find_minima_grid

def find_minima_grid(
    energies: numpy.ndarray[float64],
) -> list[tuple[tuple[int, ...], float]]:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are skipped.

### Returns

- `list[tuple[tuple[int, ...], float]]` — every cell that qualifies as a minimum as `(index, energy)`. Sorted ascending by energy; ties broken by `f64::total_cmp` for determinism.

### Raises

- `ValueError` if `energies` is not C-contiguous or if `energies.ndim` is outside `[2, 7]`.

### What it does

Returns every cell whose energy is **not strictly greater than any non-`NaN` neighbour** in the full 3ᴺ−1 (king-move) stencil. Equivalently: no neighbour has strictly lower energy. Ties are allowed; a cell with one or more equal-energy neighbours still qualifies as long as none is lower. See [`ALGORITHMS.md`](./ALGORITHMS.md#find_minima_grid) for the underlying algorithm.

### Example

```python
import numpy as np
from pes_analyzer.minimum import find_minima_grid

energies = np.array([
    [5.0, 5.0, 5.0],
    [5.0, 0.0, 5.0],
    [5.0, 5.0, 5.0],
])

print(find_minima_grid(energies))
# [((1, 1), 0.0)]
```

### Notes and edge cases

- **Plateaus are reported.** Every cell on a flat plateau that has no king-move neighbour with strictly lower energy qualifies. In a uniformly-flat region, every cell whose stencil contains no lower neighbour will appear in the output — including corner and edge cells of the plateau where the stencil happens to be entirely within the plateau. If you only want strictly-isolated minima, filter the output yourself.
- **Boundary cells**: the stencil is clipped at array edges. A corner cell has fewer neighbours but is tested with the same rule.
- **`NaN` neighbours** are ignored — a cell is tested only against its non-`NaN` neighbours. A cell with at least one non-`NaN` neighbour and no strictly-lower one is reported; a cell whose entire stencil is `NaN` is not.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: energies must be C-contiguous` | passed a slice or transposed array | `np.ascontiguousarray(arr)` before calling |
| `ValueError: ndim must be in [2, 7]` | wrong array shape | reshape or filter inactive axes |
| `ValueError: index ... out of bounds for shape ...` | `start`/`end` outside the grid | check tuple length and values |
| `ValueError: energy at \`start\` is NaN` | endpoint cell is masked | pick an endpoint inside the non-`NaN` region |
