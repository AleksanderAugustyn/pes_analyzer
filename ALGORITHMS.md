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

**Local minima on the Chebyshev box of half-width `r` (default `r = 1`).** The predicate is *no neighbour within Chebyshev distance `r` has strictly lower energy*. At `r = 1` the stencil is the classic king-move 3ᴺ−1; at general `r` it is `(2r+1)ᴺ−1` cells. Ties are allowed by the strict-less-than test — a cell with one or more equal-energy neighbours qualifies as long as none is lower. A cell is reported iff it is non-`NaN`, has at least one non-`NaN` neighbour in the stencil, and no non-`NaN` neighbour has a strictly smaller energy value.

**Implementation** (see `src/minimum/local_minima.rs`):

1. Iterate every non-`NaN` cell in row-major order.
2. Walk its Chebyshev-box neighbours via `common::nd::full_neighbors`.
3. If every non-`NaN` neighbour has energy ≥ the cell's own, emit `(nd_index, energy)`.
4. Sort the output ascending by energy with `f64::total_cmp` for determinism.

**Why a king-move (Chebyshev) stencil, not axis-only?** A cell that is below its axis neighbours but above a diagonal neighbour is not a minimum of the surface — water released there would slide off to the diagonally lower neighbour. The topological definition uses the full Chebyshev box. The `neighborhood_range` parameter widens the box when the user wants to be robust against coarse grid sampling that places real minima two or more cells apart.

**Plateaus.** Because ties are allowed, every cell on a flat plateau where no neighbour in the stencil is strictly lower will be reported. In a uniformly-flat region this produces one entry per cell whose stencil happens to lie entirely within the plateau. Callers that want strictly-isolated minima must filter the output.

**Boundary handling.** The stencil is clipped at the array edges (a corner cell in 2-D has 3 neighbours instead of 8). The predicate is unchanged.

**Complexity.** O(M · (2r+1)ᴺ) over non-`NaN` cells. For `r = 1`, `N = 7` the factor is 3⁷ − 1 = 2186 neighbour checks per cell. The factor grows fast with `r`: at `N = 7`, `r = 2` is already 78124, and `r = 5` is ≈ 1.95 × 10⁷. The validation cap `r ≤ 5` is the safety net.

**Two-pass refinement via `confirm_range`.** Because the Chebyshev box of half-width 1 is a subset of every box with `r > 1`, every minimum at radius `R` is also a minimum at radius 1. Equivalently: any cell that fails the `r = 1` check cannot be a minimum at any wider radius. The `confirm_range` keyword exploits this: stage 1 runs the fast `r = 1` check on every cell to produce a usually-tiny candidate set, then stage 2 re-checks each candidate against the `confirm_range`-wide stencil. The result is identical to a direct check at `confirm_range`, but the cost drops from `O(M · (2R+1)ᴺ)` to `O(M · 3ᴺ + |candidates| · (2R+1)ᴺ)`. For smooth PES grids `|candidates|` is typically far below 1% of `M`.
