# pes_analyzer

*Fast saddle-point and minimum analysis on dense N-D potential energy surface grids.*

Python ≥ 3.10 · Rust 2024 edition · N-D grids for N ∈ [2, 7] · MIT

## Why

Quantum-chemistry calculations produce potential energy surfaces (PES) as dense tables of energies on a multidimensional grid of geometric coordinates. Once that grid exists, the interesting analysis questions are topological: where are the minima, where are the saddle points, which basins are connected to which? `pes_analyzer` answers those questions on grids that may be too large for pure-Python approaches by pushing the inner loops into Rust.

## What it does

- **`pes_analyzer.saddle.find_iwf_grid`** — imaginary water flow (watershed) saddle search between two grid points.
- **`pes_analyzer.minimum.find_minima_grid`** — local minima on the Chebyshev king-move stencil (default 3ᴺ−1; widen with `neighborhood_range`).
- **`pes_analyzer.grid.build_dense`** — scatter helper that turns sparse `(coords, value)` rows into a dense `numpy` array indexed in axis order.

## Installation

The package builds from source via [maturin](https://www.maturin.rs/). From a checkout:

```bash
pip install maturin
maturin develop --release
```

For day-to-day development (editable installs, running tests, rebuilding after Rust changes) see [`DEVELOPMENT.md`](./DEVELOPMENT.md).

## Quickstart

```python
import numpy as np

from pes_analyzer.saddle  import find_iwf_grid
from pes_analyzer.minimum import find_minima_grid

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
| `grid.build_dense(coords, values)` | sparse rows → dense N-D array | [API.md](./API.md#build_dense) |
| `saddle.find_iwf_grid(energies, start, end)` | watershed saddle search | [API.md](./API.md#find_iwf_grid) |
| `minimum.find_minima_grid(energies, *, neighborhood_range=1)` | local minima (Chebyshev stencil) | [API.md](./API.md#find_minima_grid) |

## Documentation

- [`API.md`](./API.md) — full API reference with examples.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — repo layout, Python/Rust seam, GIL handling.
- [`ALGORITHMS.md`](./ALGORITHMS.md) — how the watershed and minima algorithms work.
- [`DEVELOPMENT.md`](./DEVELOPMENT.md) — building, testing, common workflows.

## License

MIT — see `Cargo.toml`.
