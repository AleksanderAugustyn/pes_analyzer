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

**Implementation** (see `src/extrema/local_minima.rs`):

1. Iterate every non-`NaN` cell in row-major order.
2. Walk its Chebyshev-box neighbours via `common::nd::full_neighbors`.
3. If every non-`NaN` neighbour has energy ≥ the cell's own, emit `(nd_index, energy)`.
4. Sort the output ascending by energy with `f64::total_cmp` for determinism.

**Why a king-move (Chebyshev) stencil, not axis-only?** A cell that is below its axis neighbours but above a diagonal neighbour is not a minimum of the surface — water released there would slide off to the diagonally lower neighbour. The topological definition uses the full Chebyshev box. The `neighborhood_range` parameter widens the box when the user wants to be robust against coarse grid sampling that places real minima two or more cells apart.

**Plateaus.** Because ties are allowed, every cell on a flat plateau where no neighbour in the stencil is strictly lower will be reported. In a uniformly-flat region this produces one entry per cell whose stencil happens to lie entirely within the plateau. Callers that want strictly-isolated minima must filter the output.

**Boundary handling.** The stencil is clipped at the array edges (a corner cell in 2-D has 3 neighbours instead of 8). The predicate is unchanged.

**Complexity.** O(M · (2r+1)ᴺ) over non-`NaN` cells. For `r = 1`, `N = 7` the factor is 3⁷ − 1 = 2186 neighbour checks per cell. The factor grows fast with `r`: at `N = 7`, `r = 2` is already 78124, and `r = 5` is ≈ 1.95 × 10⁷. The validation cap `r ≤ 5` is the safety net.

**Two-pass refinement via `confirm_range`.** Because the Chebyshev box of half-width 1 is a subset of every box with `r > 1`, every minimum at radius `R` is also a minimum at radius 1. Equivalently: any cell that fails the `r = 1` check cannot be a minimum at any wider radius. The `confirm_range` keyword exploits this: stage 1 runs the fast `r = 1` check on every cell to produce a usually-tiny candidate set, then stage 2 re-checks each candidate against the `confirm_range`-wide stencil. The result is identical to a direct check at `confirm_range`, but the cost drops from `O(M · (2R+1)ᴺ)` to `O(M · 3ᴺ + |candidates| · (2R+1)ᴺ)`. For smooth PES grids `|candidates|` is typically far below 1% of `M`.

## `find_maxima_grid`

Strict dual of `find_minima_grid` — same Chebyshev stencil, same two-pass `confirm_range` optimisation, same NaN handling. The only differences are:

- the comparator inside the inner loop is flipped (`ne > e` instead of `ne < e`),
- the output is sorted descending by energy via `f64::total_cmp` reversed.

Internally, both functions delegate to a single generic kernel `local_extreme_inner<C>` in `src/extrema/local_minima.rs`, parameterised by an `is_dominated(neighbour, center) -> bool` closure that Rust monomorphises at compile time.

## `find_extrema_grid`

Single-sweep combined search that produces both minima and maxima from one stencil walk per cell. For each non-NaN centre, the neighbour iteration maintains two flags (`beats_all_lower` for the minima check, `beats_all_higher` for the maxima check); the loop short-circuits only once both flags are settled. The two confirm-stage candidate lists are processed independently, since the surviving candidate sets are typically small and disjoint.

The function exists purely for callers who need both polarities (e.g. saddle sanity checks against hilltops). For callers who only need one, the single-polarity entry points are faster because they can short-circuit on the first disqualifier.

## `find_watershed_segmentation`

Same imaginary-water-flow flood as `find_iwf_grid`, run to completion instead of stopping at the first endpoint-merge. The added bookkeeping records every union that merges two previously-disconnected basins as a merge event, and labels every non-`NaN` cell with the basin it first joined.

**Implementation** (see `src/topology/watershed.rs`):

1. Collect all non-`NaN` cells into a sorted `Vec<(flat_index, energy)>` (ascending by `total_cmp`), as in `find_iwf_grid`.
2. Walk the sorted list. For each cell `c`:
   - Look at already-processed neighbors and find their current basins via `basin_of_root[dsu.find(nbr)]`.
   - If none are processed: `c` starts a new basin; assign it the next basin id; record `basins.push((c, energy))` and set `labels[c] = new_id`.
   - If all processed neighbors are in one basin `b`: union `c` into it; `labels[c] = b`.
   - If processed neighbors span two or more distinct basins: `c` is a saddle. Adopt the first basin as `labels[c]`. For each additional distinct basin, record a merge event `(c, energy, deeper, shallower)` with `deeper.min_e <= shallower.min_e`, then union and update the surviving basin id of the new root.
3. Expand the compact `labels: Vec<u32>` into the public `int32` ndarray; cells with no compact-index (NaN) become `-1`.

**Output convention.** `basins` is sorted ascending by minimum energy, so `basins[0]` is the deepest basin and contains the global minimum cell. `merges` is sorted ascending by saddle energy. For every merge, `basins[deeper].min_e <= basins[shallower].min_e`; the persistence of the shallower basin is `saddle_energy - basins[shallower].min_e`.

**Why axis-only neighbours.** Same reason as `find_iwf_grid`: a physical reaction path on a PES grid follows the grid axes one step at a time. The merge tree describes the topology of basins connected by axis-only paths, which is the topology of interest for fission-barrier analysis.

**Complexity.** `O(M log M)` where `M` = non-`NaN` cell count, identical to `find_iwf_grid`. The sort dominates; the union-find with path compression and union-by-rank is effectively `O(M α(M))`. Memory `O(M)` for the sorted vector, remap, DSU, basin-of-root, and labels.

**Relation to `find_iwf_grid`.** `find_iwf_grid` is the two-point specialization: same flood, but it terminates the moment the start and end cells share a DSU root and returns that single saddle cell. `find_watershed_segmentation` keeps the flood going to completion and records every merge along the way. Callers that only need the saddle between two specified cells should keep using `find_iwf_grid` — it has the early-exit and avoids building the merge tree.

## `find_minimum_energy_path`

**Deep minimax path.** Among all grid paths between two cells, the path that minimizes the highest energy crossed — so it passes through the exact IWF saddles — and that, between saddles, descends to the actual basin minimum cells. Its 1-D profile has true inter-basin saddles as local maxima and true basin minima as local minima; `analyze_path_profile` reads critical points directly off it.

**Implementation** (see `src/topology/mep.rs`):

1. Run the ascending-energy flood (same sort, remap, and DSU as `find_iwf_grid`), recording two extra structures:
   - **Flood parent** per cell: the first already-flooded neighbour the cell unions into. Because cells are processed in ascending order, parent chains descend monotonically (non-increasing energy) and terminate at the basin seed — the basin's minimum cell. This gives a guaranteed descent path with none of the stuck-cell ambiguity of greedy steepest descent.
   - **Kruskal forest** over components: leaves are basins; each merge event becomes an internal node storing the saddle cell, the already-flooded neighbour on the other component, and which child subtree holds the saddle-side component. Node ids increase with creation time, so a parent's id always exceeds its children's.
2. Early-exit the flood once `start` and `end` share a DSU root (the connecting event is the IWF saddle between them).
3. Reconstruct iteratively with a segment stack. `connect(a, b)`:
   - Same basin (descent chains end at the same seed): emit `a`'s chain down to the seed, then `b`'s chain reversed. The V through the seed is the deliberate deep dip to the basin minimum; if the two chains share a suffix, those cells are walked down and back up (the path is a walk, not necessarily simple).
   - Different basins: find the lowest common ancestor event in the forest (lift the smaller node id), orient the saddle crossing so `u` is on `a`'s side and `v` on `b`'s, and recurse on `(a, u)` and `(v, b)`. `u` and `v` are stencil neighbours, so the crossing is a valid move.

**Guarantees.** For the same `neighborhood`, the path's maximum energy equals the `find_iwf_grid` saddle energy bit-for-bit (identical sort order and union schedule). Every profile local minimum is a watershed basin seed.

**Complexity.** `O(M log M)` flood (sort-dominated) plus `O(K)` reconstruction for a path of K cells. Memory `O(M)`: parents, forest, node-of-root, remap, DSU.

**`NaN` handling.** As everywhere: NaN cells are excluded from the sort and act as walls. Disconnected endpoints return `None`; NaN at an endpoint is a `ValueError` raised by the wrapper.

## Neighborhood stencils

`find_iwf_grid`, `find_watershed_segmentation`, and `find_minimum_energy_path` accept `neighborhood="von_neumann"` (default; 2N axis neighbours) or `"moore"` (3ᴺ−1 Chebyshev neighbours at range 1; the range is fixed).

Von Neumann is the more physical choice for fission-barrier analysis: it cannot squeeze through two orthogonal barriers that meet at a corner via the unsampled diagonal. Moore matches the move set of Metropolis-style random walks on PES grids. The two bracket the continuum limit — von Neumann biases barriers slightly high (forbids diagonal moves the continuous surface allows), Moore slightly low (corner-cuts through cells it never samples) — so comparing both is a cheap grid-resolution diagnostic. Mixing stencils between the merge tree and the MEP makes their saddles disagree; pass the same value to both.

The extrema kernels (`find_minima_grid` etc.) are unaffected: they keep the king-move Chebyshev stencil with the separate `neighborhood_range` parameter, for the reasons given above.
