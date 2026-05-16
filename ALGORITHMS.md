# Algorithms

Concise descriptions of the two compute kernels in `pes_analyzer`. Enough to debug or extend; for deeper detail read the Rust sources.

## `find_iwf_grid`

**Imaginary water flow (watershed).** Imagine water rising over the energy surface from below. As the water level reaches each cell, the cell is unioned with already-flooded neighbours. The first level at which `start` and `end` lie in the same connected component is the saddle energy; the cell that triggers the merge is the saddle cell.

**Implementation** (see `src/saddle/iwf_grid.rs`):

1. Collect all non-`NaN` cells into a compact `Vec<(flat_index, energy)>` and sort ascending by energy (`total_cmp`).
2. Walk the sorted list. For each cell, mark it processed and union it with every already-processed axis-neighbour.
3. After each union, check whether `find(start)` equals `find(end)`. The first cell that produces equality is the saddle.

**Why axis-only (2N) neighbours, not king-move (3ᴺ−1)?** A physical reaction path on a PES grid follows the grid axes one step at a time. Diagonal moves would let the algorithm hop over a cell, potentially reporting a saddle that's lower than any path actually accessible at single-axis resolution.

**Complexity.** O(M log M) where M is the count of non-`NaN` cells. The sort dominates; the union-find with path compression and union-by-rank is effectively O(M α(M)).

**`NaN` handling.** `NaN` cells are excluded from the sorted set, so neighbour scans skip them and they act as impassable walls. `NaN` at `start` or `end` is treated as a usage error: the PyO3 wrapper raises `ValueError` before calling the kernel.

**Memory.** O(M) for the sorted vector, the linear→compact `remap` table, and the union-find arrays.

## `find_minima_grid`

**Local minima on the 3ᴺ−1 stencil.** The predicate is *no king-move neighbour has strictly lower energy*. Ties are allowed by the strict-less-than test — a cell with one or more equal-energy neighbours qualifies as long as none is lower. A cell is reported iff it is non-`NaN`, has at least one non-`NaN` neighbour, and no non-`NaN` neighbour has a strictly smaller energy value.

**Implementation** (see `src/minimum/local_minima.rs`):

1. Iterate every non-`NaN` cell in row-major order.
2. Walk its king-move neighbours via `common::nd::full_neighbors`.
3. If every non-`NaN` neighbour has energy ≥ the cell's own, emit `(nd_index, energy)`.
4. Sort the output ascending by energy with `f64::total_cmp` for determinism.

**Why king-move, not axis-only?** A cell that is below its axis neighbours but above a diagonal neighbour is not a minimum of the surface — water released there would slide off to the diagonally lower neighbour. The topological definition uses the full stencil.

**Plateaus.** Because ties are allowed, every cell on a flat plateau where no neighbour in the stencil is strictly lower will be reported. In a uniformly-flat region this produces one entry per cell whose stencil happens to lie entirely within the plateau. Callers that want strictly-isolated minima must filter the output.

**Boundary handling.** The stencil is clipped at the array edges (a corner cell in 2-D has 3 neighbours instead of 8). The predicate is unchanged.

**Complexity.** O(M · 3ᴺ) over non-`NaN` cells. For N = 7 the factor is 3⁷ − 1 = 2186 neighbour checks per cell — non-trivial but still linear in grid size.
