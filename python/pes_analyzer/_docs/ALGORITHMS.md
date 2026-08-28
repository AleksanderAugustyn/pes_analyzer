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

1. Split the flat grid into fixed chunks of 2¹⁶ cells and scan the chunks on the rayon pool. Each chunk yields its candidates in ascending linear index and the per-chunk lists are concatenated in chunk order, so the result is identical to a sequential row-major scan whatever the thread count.
2. For every non-`NaN` cell, walk its Chebyshev-box neighbours lazily via `common::nd::walk_box_neighbors` — the exact order `full_neighbors` would list them, but no neighbour list is materialised and the walk stops at the first strictly lower neighbour. On a smooth surface most cells are rejected after one or two neighbours, which is what makes the scan cheap (0.5 s for 1.6 × 10⁸ cells at N = 5 on 16 threads; 3.9 s on one).
3. If every non-`NaN` neighbour has energy ≥ the cell's own (and at least one exists), emit `(nd_index, energy)`.
4. Sort the output ascending by energy with `total_cmp` for determinism.

**Why a king-move (Chebyshev) stencil, not axis-only?** A cell that is below its axis neighbours but above a diagonal neighbour is not a minimum of the surface — water released there would slide off to the diagonally lower neighbour. The topological definition uses the full Chebyshev box. The `neighborhood_range` parameter widens the box when the user wants to be robust against coarse grid sampling that places real minima two or more cells apart.

**Plateaus.** Because ties are allowed, every cell on a flat plateau where no neighbour in the stencil is strictly lower will be reported. In a uniformly-flat region this produces one entry per cell whose stencil happens to lie entirely within the plateau. Callers that want strictly-isolated minima must filter the output.

**Boundary handling.** The stencil is clipped at the array edges (a corner cell in 2-D has 3 neighbours instead of 8). The predicate is unchanged.

**Complexity.** Worst case O(M · (2r+1)ᴺ) over non-`NaN` cells; the early exit makes the typical cost a handful of neighbours per cell. For `r = 1`, `N = 7` the worst-case factor is 3⁷ − 1 = 2186 neighbour checks per cell. The factor grows fast with `r`: at `N = 7`, `r = 2` is already 78124, and `r = 5` is ≈ 1.95 × 10⁷. The validation cap `r ≤ 5` is the safety net.

**Two-pass refinement via `confirm_range`.** Because the Chebyshev box of half-width 1 is a subset of every box with `r > 1`, every minimum at radius `R` is also a minimum at radius 1. Equivalently: any cell that fails the `r = 1` check cannot be a minimum at any wider radius. The `confirm_range` keyword exploits this: stage 1 runs the fast `r = 1` check on every cell to produce a usually-tiny candidate set, then stage 2 re-checks each candidate against the `confirm_range`-wide stencil. The result is identical to a direct check at `confirm_range`, but the cost drops from `O(M · (2R+1)ᴺ)` to `O(M · 3ᴺ + |candidates| · (2R+1)ᴺ)`. For smooth PES grids `|candidates|` is typically far below 1% of `M`.

## `find_maxima_grid`

Strict dual of `find_minima_grid` — same Chebyshev stencil, same two-pass `confirm_range` optimisation, same NaN handling. The only differences are:

- the comparator inside the inner loop is flipped (`ne > e` instead of `ne < e`),
- the output is sorted descending by energy via `f64::total_cmp` reversed.

Internally, both functions delegate to a single generic kernel `local_extreme_inner<C>` in `src/extrema/local_minima.rs`, parameterised by an `is_dominated(neighbour, center) -> bool` closure that Rust monomorphises at compile time.

## `find_extrema_grid`

Single-sweep combined search that produces both minima and maxima from one stencil walk per cell. For each non-NaN centre, the neighbour iteration maintains two flags (`beats_all_lower` for the minima check, `beats_all_higher` for the maxima check); the lazy walk stops as soon as both flags are settled. The find stage is chunked over the rayon pool exactly like the single-polarity kernels. The two confirm-stage candidate lists are processed independently, since the surviving candidate sets are typically small and disjoint.

The function exists purely for callers who need both polarities (e.g. saddle sanity checks against hilltops). For callers who only need one, the single-polarity entry points are faster because they can short-circuit on the first disqualifier.

## `find_watershed_segmentation`

Same imaginary-water-flow flood as `find_iwf_grid`, run to completion instead of stopping at the first endpoint-merge. The added bookkeeping records every union that merges two previously-disconnected basins as a merge event, and labels every non-`NaN` cell with the basin it first joined.

**Implementation** (see `src/topology/watershed.rs::flood`):

1. `sorted_order`: collect the non-`NaN` cells as `(energy, linear index)` pairs and sort them ascending by `(total_cmp(energy), index)` on the rayon pool (`par_sort_unstable_by`; the key is a total order, so the result is independent of the thread count). Only the `u32` index order is kept; the pairs are freed before the flood.
2. Flood in that order, indexed by linear cell id. Three grid-sized arrays: `parent: Vec<u32>` (the union-find), `labels: Vec<i32>` (the output, doubling as the processed flag — `labels[c] >= 0` iff `c` has been flooded) and, on request, `parents: Vec<u16>` (flood-parent direction codes). For each cell `c`, scanning its stencil neighbours in the fixed order of § Neighborhood stencils:
   - Skip neighbours with `labels[n] < 0` (`NaN` or not yet flooded).
   - The first flooded neighbour `n` gives `r = find(n)` and `b = labels[r]`: hang `c` under `r` (`parent[c] = r`), set `labels[c] = b` and, if recording, `parents[c] = code(c → n)`.
   - A later neighbour whose component's basin `b'` differs from the current one is a merge: the lower id is `deeper`, the other `shallower`; the shallower seed is hung under the deeper seed (`parent[seed_shallow] = seed_deep`) and the event `(saddle = c, other = n, energy, deeper, shallower, saddle_side)` is recorded. The current basin becomes `deeper`, so a third basin meeting the same cell merges with the survivor.
   - No flooded neighbour: `c` seeds a new basin with the next id, `labels[c] = id`, and `c` stays its own root.
3. `NaN` cells keep `-1`.

**Seed-rooted union-find invariant.** Every root is the seed of its component's deepest basin, so `labels[find(x)]` *is* the basin id of `x`'s component — no separate root-to-basin table. Ordinary cells hang under a root and never become one; a merge always links seed to seed with the deeper seed on top. Path halving keeps `find` near-constant.

**Tie order.** Cells are flooded in ascending `(energy, linear index)` and basin ids are assigned in that order. Hence at a merge the lower id is deeper (ties resolve to the lower seed index) and basin 0 is never `shallower` — it is always the root of the merge tree.

**Direction codes.** With `parents=True`, each cell's flood parent (its first flooded neighbour) is stored as a `u16` code: the odometer index of Δ ∈ {−1, 0, 1}ᴺ with the last axis fastest, `Σ (Δ_axis + 1)·3^(N−1−axis)`, so at most 3⁷ = 2187 codes; `65535` marks seeds, `NaN` cells and (in an early-stopped flood) never-flooded cells. Von Neumann codes are the subset with exactly one non-zero Δ. `common::nd::apply_code` decodes a step; `apply_code_checked` additionally rejects steps that leave the grid.

**Memory.** 8N + 4V bytes resident during the flood (`parent` 4N, `labels` 4N, order 4V), +2N with parents, plus the sort's 12V (`float32`) / 20V (`float64`) transient. `find_iwf_grid` keeps its own compact-remap kernel over `common::dsu`.

**Output convention.** `basins` is sorted ascending by minimum energy, so `basins[0]` is the deepest basin and contains the global minimum cell. `merges` is sorted ascending by saddle energy. For every merge, `basins[deeper].min_e <= basins[shallower].min_e`; the persistence of the shallower basin is `saddle_energy - basins[shallower].min_e`.

**Why axis-only neighbours.** Same reason as `find_iwf_grid`: a physical reaction path on a PES grid follows the grid axes one step at a time. The merge tree describes the topology of basins connected by axis-only paths, which is the topology of interest for fission-barrier analysis.

**Complexity.** `O(M log M)` where `M` = non-`NaN` cell count, identical to `find_iwf_grid`. The (parallel) sort dominates; the union-find with path halving is effectively `O(M α(M))`. Memory as above.

**Relation to `find_iwf_grid`.** `find_iwf_grid` is the two-point specialization: same flood, but it terminates the moment the start and end cells share a DSU root and returns that single saddle cell. `find_watershed_segmentation` keeps the flood going to completion and records every merge along the way. Callers that only need the saddle between two specified cells should keep using `find_iwf_grid` — it has the early-exit and avoids building the merge tree.

## `find_minimum_energy_path`

**Deep minimax path.** Among all grid paths between two cells, the path that minimizes the highest energy crossed — so it passes through the exact IWF saddles — and that, between saddles, descends to the actual basin minimum cells. Its 1-D profile has true inter-basin saddles as local maxima and true basin minima as local minima; `analyze_path_profile` reads critical points directly off it.

**Implementation** (see `src/topology/mep.rs`):

The path is *reconstructed* from a flood's state — `labels`, `parents` (direction codes, see `find_watershed_segmentation`) and the merge list — by `mep::reconstruct`:

1. **Kruskal forest from the merge list.** Leaves are basin ids `0..B`; merge `k` is node `B + k`, so a parent's id always exceeds its children's. Walking the merges in order, `cur_node[b]` tracks the forest node that currently contains basin `b`; each merge links `cur_node[deeper]` and `cur_node[shallower]` under the new node, remembers which child holds the saddle cell (`cur_node[saddle_side]`) and sets `cur_node[deeper]` to the new node.
2. **Leaf of a cell** = the basin id at the terminus of its flood-parent chain. Chains descend monotonically (non-increasing energy) and end at a seed, whose label is its own basin id. This gives a guaranteed descent with none of the stuck-cell ambiguity of greedy steepest descent.
3. **Segment stack.** `connect(a, b)`:
   - Same leaf: emit `a`'s chain down to the seed, then `b`'s chain reversed. The V through the seed is the deliberate deep dip to the basin minimum; if the two chains share a suffix, those cells are walked down and back up (the path is a walk, not necessarily simple).
   - Different leaves: the lowest common ancestor in the forest (lift the smaller node id) is the merge whose saddle is the minimax point between them. Orient `(saddle, other)` so `u` is on `a`'s side and `v` on `b`'s — `u` and `v` are stencil neighbours, so the crossing is a valid move — and push `(v, b)` then `(a, u)`.

All inputs may come from Python, so every step is bounds-checked; inconsistencies (labels ≥ B, invalid codes, chains that leave the grid or cycle, merges that contradict earlier ones) surface as `ValueError`, never as a panic. Endpoints with `labels < 0` (never flooded, in an early-stopped flood) or in different forest trees give `None`.

**Modes.** *Tree mode* (`tree=`) runs `reconstruct` on the `Watershed` arrays: O(K) memory for a K-cell path plus the forest (B + M nodes), no flood. *Standalone mode* runs `watershed::flood` with parents on and `stop_when_connected` at the endpoints — the connecting event is the IWF saddle between them — then reconstructs from that partial state.

**Guarantees.** For the same `neighborhood`, the path's maximum energy equals the `find_iwf_grid` saddle energy (the minimax value is unique; with tied energies the saddle cell may differ). Every profile local minimum is a watershed basin seed. Tree mode and standalone mode return the identical path.

**Complexity.** Standalone: the `O(M log M)` flood (sort-dominated) plus `O(K)` reconstruction, 10N + 4V bytes. Tree mode: one O(N) validation pass over `labels`, then `O(K)`.

**`NaN` handling.** As everywhere: NaN cells are excluded from the sort and act as walls. Disconnected endpoints return `None`; NaN at an endpoint is a `ValueError` raised by the wrapper.

## Neighborhood stencils

`find_iwf_grid`, `find_watershed_segmentation`, and `find_minimum_energy_path` accept `neighborhood="von_neumann"` (default; 2N axis neighbours) or `"moore"` (3ᴺ−1 Chebyshev neighbours at range 1; the range is fixed).

Von Neumann is the more physical choice for fission-barrier analysis: it cannot squeeze through two orthogonal barriers that meet at a corner via the unsampled diagonal. Moore matches the move set of Metropolis-style random walks on PES grids. The two bracket the continuum limit — von Neumann biases barriers slightly high (forbids diagonal moves the continuous surface allows), Moore slightly low (corner-cuts through cells it never samples) — so comparing both is a cheap grid-resolution diagnostic. Mixing stencils between the merge tree and the MEP makes their saddles disagree; pass the `MergeTree` (or `Watershed`) to `find_minimum_energy_path(tree=...)` so the path inherits the tree's stencil — an explicit `neighborhood` must then match the tree's.

The extrema kernels (`find_minima_grid` etc.) are unaffected: they keep the king-move Chebyshev stencil with the separate `neighborhood_range` parameter, for the reasons given above.
