# pes_analyzer

*Fast saddle-point and minimum analysis on dense N-D potential energy surface grids.*

Python ≥ 3.10 · Rust 2024 edition · N-D grids for N ∈ [2, 7] · MIT

## Why

Quantum-chemistry calculations produce potential energy surfaces (PES) as dense tables of energies on a multidimensional grid of geometric coordinates. Once that grid exists, the interesting analysis questions are topological: where are the minima, where are the saddle points, which basins are connected to which? `pes_analyzer` answers those questions on grids that may be too large for pure-Python approaches by pushing the inner loops into Rust.

## What it does

- **`pes_analyzer.saddle.find_iwf_grid`** — imaginary water flow (watershed) saddle search between two grid points.
- **`pes_analyzer.extrema.find_minima_grid`** — local minima on the Chebyshev king-move stencil (default 3ᴺ−1; widen via `confirm_range` for a fast two-pass check, or via `neighborhood_range` for a direct wider check).
- **`pes_analyzer.extrema.find_maxima_grid`** — strict dual of `find_minima_grid`. Same stencil and same `neighborhood_range` / `confirm_range` semantics; output sorted descending by energy.
- **`pes_analyzer.extrema.find_extrema_grid`** — combined single-sweep search. Returns `(minima, maxima)` byte-identical to calling the two single-polarity functions separately, at the cost of one extra-list allocation but one fewer stencil walk per cell.
- **`pes_analyzer.topology.find_watershed_segmentation`** — full watershed flood returning a `Watershed`: basin labels, the merge (saddle) events, and optionally each cell's flood parent (`parents=True`). Peak memory 8N + 4V bytes; `drop_labels()` releases the grid arrays.
- **`pes_analyzer.topology.find_minimum_energy_path`** — deep minimax path between two cells; with `tree=` it is reconstructed from the watershed's flood state instead of re-flooding.
- **`pes_analyzer.topology`** merge-tree helpers — pure-Python `compute_persistence`, `prune_merge_tree`, and the traversable `MergeTree(ws)` (whose nodes are `BasinNode`s) analyse that merge tree. `MergeTree` is physics-free: it exposes neutral traversal, membership, and geometry primitives that a consumer composes with its own predicates to label ground states, saddles, fission exits, etc.
- **`pes_analyzer.grid.build_dense`** — scatter helper that turns sparse `(coords, value)` rows into a dense `numpy` array indexed in axis order.

## Installation

```bash
pip install pes_analyzer
```

Wheels bundle the full reference docs under `pes_analyzer/_docs/` — see
[Documentation](#documentation). Building from source (PyO3 + Rust via
[maturin](https://www.maturin.rs/)) and contributor workflows are covered in
[`DEVELOPMENT.md`](./DEVELOPMENT.md).

## Quickstart

```python
import numpy as np

from pes_analyzer.saddle  import find_iwf_grid
from pes_analyzer.extrema import find_minima_grid

# A toy 2x5 PES: two basins at (0, 0) and (0, 4) along the top row,
# separated by a hump that peaks at (0, 2). The bottom row is a high
# wall, so any path between the basins must cross the hump.
energies = np.array([
    [0.0, 1.0, 2.0, 1.0, 0.0],
    [3.0, 3.0, 3.0, 3.0, 3.0],
])

print(find_minima_grid(energies))
# [((0, 0), 0.0), ((0, 4), 0.0)]

print(find_iwf_grid(energies, start=(0, 0), end=(0, 4)))
# ((0, 2), 2.0)
```

## API at a glance

| Function | Purpose | Reference |
|---|---|---|
| `grid.build_dense(coords, values)` | sparse rows → dense N-D array | [API.md](./python/pes_analyzer/_docs/API.md#build_dense) |
| `saddle.find_iwf_grid(energies, start, end)` | watershed saddle search | [API.md](./python/pes_analyzer/_docs/API.md#find_iwf_grid) |
| `extrema.find_minima_grid(energies, *, neighborhood_range=1, confirm_range=None)` | local minima (Chebyshev stencil) | [API.md](./python/pes_analyzer/_docs/API.md#find_minima_grid) |
| `extrema.find_maxima_grid(energies, *, neighborhood_range=1, confirm_range=None)` | local maxima (dual of `find_minima_grid`) | [API.md](./python/pes_analyzer/_docs/API.md#find_maxima_grid) |
| `extrema.find_extrema_grid(energies, *, neighborhood_range=1, confirm_range=None)` | combined single-sweep search | [API.md](./python/pes_analyzer/_docs/API.md#find_extrema_grid) |
| `topology.find_watershed_segmentation(energies, neighborhood="von_neumann", *, parents=False)` | full basin labelling + merge tree (`Watershed`) | [API.md](./python/pes_analyzer/_docs/API.md#find_watershed_segmentation) |
| `topology.find_minimum_energy_path(energies, start, end, tree=tree)` | deep minimax path from the watershed's flood state | [API.md](./python/pes_analyzer/_docs/API.md#find_minimum_energy_path) |
| `topology.compute_persistence(basins, merges)` | per-basin topological persistence | [API.md](./python/pes_analyzer/_docs/API.md#topology-helpers) |
| `topology.prune_merge_tree(basins, merges, threshold)` | drop low-persistence basins | [API.md](./python/pes_analyzer/_docs/API.md#topology-helpers) |
| `topology.MergeTree(ws)` | traversable basin merge tree (physics-free primitives; `drop_labels()`) | [API.md](./python/pes_analyzer/_docs/API.md#topology-helpers) |

## Documentation

The consumer reference docs are **bundled in the installed package** under
`pes_analyzer/_docs/`; locate them at runtime with `pes_analyzer.docs_path()`.

- [`USAGE.md`](./python/pes_analyzer/_docs/USAGE.md) — end-to-end pipeline cookbook (start here).
- [`API.md`](./python/pes_analyzer/_docs/API.md) — full API reference with examples.
- [`ALGORITHMS.md`](./python/pes_analyzer/_docs/ALGORITHMS.md) — how the watershed and minima algorithms work.

Developer-facing docs stay in the repo (not bundled):

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — repo layout, Python/Rust seam, GIL handling.
- [`DEVELOPMENT.md`](./DEVELOPMENT.md) — building, testing, common workflows.

## License

MIT — see `Cargo.toml`.
