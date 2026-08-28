//! Local-extremum search on a dense N-D grid using the Chebyshev box of
//! half-width `r` ((2r+1)ᴺ−1 stencil; the classic king-move 3ᴺ−1 at
//! `r = 1`). The single-polarity entry points `local_minima_inner` and
//! `local_maxima_inner` are thin wrappers around a shared generic kernel
//! `local_extreme_inner<C>` parameterised by an `is_dominated` closure.

use ndarray::ArrayViewD;
use rayon::prelude::*;

use crate::common::nd::{compute_strides, full_neighbors, linear_to_index, walk_box_neighbors};
use crate::common::scalar::Scalar;

/// Cells per rayon work item for the find stage.
const SCAN_CHUNK: usize = 1 << 16;

/// True iff `flat[lin]` has at least one non-NaN neighbour in the box of
/// half-width `r` and none of them dominates it. Stops at the first
/// dominating neighbour; no neighbour list is materialised.
#[inline]
fn is_extreme_at<T: Scalar, C: Fn(T, T) -> bool>(
    flat: &[T],
    lin: usize,
    e: T,
    shape: &[usize],
    strides: &[usize],
    r: usize,
    is_dominated: C,
) -> bool {
    let mut has_valid_neighbor = false;
    let mut beats_all = true;
    walk_box_neighbors(lin, shape, strides, r, |nbr| {
        let ne = flat[nbr];
        if ne.is_nan() {
            return true;
        }
        has_valid_neighbor = true;
        if is_dominated(ne, e) {
            beats_all = false;
            return false;
        }
        true
    });
    has_valid_neighbor && beats_all
}

/// Sort direction for the finalised output of a single-polarity search.
enum SortOrder {
    Ascending,
    Descending,
}

/// Convert flat `(lin, energy)` pairs into `(nd_index, energy)` and sort
/// them in the requested order. Ties broken by `Scalar::tcmp`
/// (total ordering, NaN-safe) for determinism.
fn finalize<T: Scalar>(
    raw: Vec<(usize, T)>,
    shape: &[usize],
    strides: &[usize],
    order: SortOrder,
) -> Vec<(Vec<usize>, T)> {
    let mut out: Vec<(Vec<usize>, T)> = raw
        .into_iter()
        .map(|(lin, e)| (linear_to_index(lin, shape, strides), e))
        .collect();
    match order {
        SortOrder::Ascending => out.sort_by(|a, b| a.1.tcmp(&b.1)),
        SortOrder::Descending => out.sort_by(|a, b| b.1.tcmp(&a.1)),
    }
    out
}

/// Generic single-polarity local-extremum search.
///
/// `is_dominated(neighbour_energy, center_energy)` returns `true` when
/// the neighbour disqualifies the centre. For minima pass `|ne, e| ne < e`;
/// for maxima pass `|ne, e| ne > e`. The closure is `Copy` so it can be
/// shared between the find stage and the confirm stage without re-borrowing;
/// Rust monomorphises one specialised copy per concrete closure type.
///
/// Returns flat `(lin, energy)` candidates; the public wrappers expand
/// `lin` to an nd-index and sort.
///
/// Preconditions (enforced by the PyO3 wrapper before calling):
/// - `energies` is a view over a C-contiguous buffer of element type `T`.
/// - `energies.ndim()` is in `[2, 7]`.
/// - `energies.len() <= u32::MAX`.
/// - `find_r >= 1` (validated upstream).
/// - `confirm_r`, if `Some(R)`, satisfies `R in [1, 5]`; values `R <= find_r` are accepted but silently skipped (no-op).
fn local_extreme_inner<T: Scalar, C>(
    energies: ArrayViewD<'_, T>,
    find_r: usize,
    confirm_r: Option<usize>,
    is_dominated: C,
) -> Vec<(usize, T)>
where
    C: Fn(T, T) -> bool + Copy + Sync,
{
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let flat: &[T] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    // Stage 1: parallel over fixed-size chunks; per-chunk vectors are
    // concatenated in chunk order, so candidates are in ascending linear
    // index exactly as a sequential scan would produce them.
    let per_chunk: Vec<Vec<(usize, T)>> = flat
        .par_chunks(SCAN_CHUNK)
        .enumerate()
        .map(|(ci, chunk)| {
            let base = ci * SCAN_CHUNK;
            let mut local: Vec<(usize, T)> = Vec::new();
            for (j, &e) in chunk.iter().enumerate() {
                if e.is_nan() {
                    continue;
                }
                let lin = base + j;
                if is_extreme_at(flat, lin, e, &shape, &strides, find_r, is_dominated) {
                    local.push((lin, e));
                }
            }
            local
        })
        .collect();
    // Move-extend: no element is cloned and each per-chunk vector is freed as
    // soon as it is consumed. Candidates are a tiny fraction of V on any real
    // PES (a fully flat grid is degenerate for a strict-minimum search).
    let total: usize = per_chunk.iter().map(Vec::len).sum();
    let mut candidates: Vec<(usize, T)> = Vec::with_capacity(total);
    for chunk in per_chunk {
        candidates.extend(chunk);
    }

    // Stage 2 unchanged: confirm against the wider stencil (few candidates)
    // if requested AND strictly wider than `find_r`; the equal case is a no-op.
    match confirm_r {
        Some(r) if r > find_r => {
            let stencil_max_r = (2 * r + 1).saturating_pow(shape.len() as u32);
            let mut nbrs_r: Vec<usize> = Vec::with_capacity(stencil_max_r);
            let mut kept: Vec<(usize, T)> = Vec::with_capacity(candidates.len());
            for (lin, e) in candidates {
                full_neighbors(lin, &shape, &strides, r, &mut nbrs_r);
                let mut has_valid_neighbor = false;
                let mut beats_all = true;
                for &nbr_lin in &nbrs_r {
                    let ne = flat[nbr_lin];
                    if ne.is_nan() {
                        continue;
                    }
                    has_valid_neighbor = true;
                    if is_dominated(ne, e) {
                        beats_all = false;
                        break;
                    }
                }
                if has_valid_neighbor && beats_all {
                    kept.push((lin, e));
                }
            }
            kept
        }
        _ => candidates,
    }
}

/// Find all strict local minima in `energies` using the Chebyshev box of
/// half-width `find_r`. See `local_extreme_inner` for parameter semantics.
/// Output sorted ascending by energy; ties broken by `Scalar::tcmp`.
pub fn local_minima_inner<T: Scalar>(
    energies: ArrayViewD<'_, T>,
    find_r: usize,
    confirm_r: Option<usize>,
) -> Vec<(Vec<usize>, T)> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let raw = local_extreme_inner(energies, find_r, confirm_r, |ne, e| ne < e);
    finalize(raw, &shape, &strides, SortOrder::Ascending)
}

/// Find all strict local maxima in `energies` using the Chebyshev box of
/// half-width `find_r`. Strict dual of `local_minima_inner`: a non-NaN
/// cell qualifies iff at least one in-bounds non-NaN neighbour exists in
/// the stencil and no such neighbour has strictly *higher* energy.
/// Output sorted descending by energy; ties broken by `Scalar::tcmp`
/// (reversed).
pub fn local_maxima_inner<T: Scalar>(
    energies: ArrayViewD<'_, T>,
    find_r: usize,
    confirm_r: Option<usize>,
) -> Vec<(Vec<usize>, T)> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let raw = local_extreme_inner(energies, find_r, confirm_r, |ne, e| ne > e);
    finalize(raw, &shape, &strides, SortOrder::Descending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD, IxDyn};

    fn to_dyn<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    #[test]
    fn single_bowl_2d_has_one_minimum_at_center() {
        // 3x3 bowl: center is 0, surroundings are 1.
        let e = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let arr = to_dyn([3, 3], e);
        let mins = local_minima_inner(arr.view(), 1, None);
        assert_eq!(mins.len(), 1);
        assert_eq!(mins[0].0, vec![1, 1]);
        assert_eq!(mins[0].1, 0.0);
    }

    #[test]
    fn two_basins_2d_returns_both() {
        // 5x5 with two clear minima at (0,0)=0 and (4,4)=0, ridge of 5 between.
        let mut e = vec![5.0; 25];
        e[0] = 0.0;
        e[1] = 1.0;
        e[5] = 1.0;
        e[6] = 2.0;
        e[24] = 0.0;
        e[23] = 1.0;
        e[19] = 1.0;
        e[18] = 2.0;
        let arr = to_dyn([5, 5], e);
        let mins = local_minima_inner(arr.view(), 1, None);
        let coords: Vec<Vec<usize>> = mins.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords.contains(&vec![0, 0]));
        assert!(coords.contains(&vec![4, 4]));
    }

    #[test]
    fn diagonal_neighbor_disqualifies_minimum() {
        // (1,1) = 5 with (0,0) = 0 — diagonal — must NOT be a local min.
        // Without diagonal connectivity, (1,1) would erroneously qualify.
        let e = vec![
            0.0, 10.0, 10.0, // row 0
            10.0, 5.0, 10.0, // row 1
            10.0, 10.0, 10.0, // row 2
        ];
        let arr = to_dyn([3, 3], e);
        let mins = local_minima_inner(arr.view(), 1, None);
        let coords: Vec<Vec<usize>> = mins.iter().map(|(c, _)| c.clone()).collect();
        assert!(!coords.contains(&vec![1, 1]));
        assert!(coords.contains(&vec![0, 0]));
    }

    #[test]
    fn ties_are_local_minima() {
        // Two adjacent cells with energy 0 surrounded by 1s; both qualify
        // because the predicate is strict `<`.
        let e = vec![
            1.0, 1.0, 1.0, 1.0, //
            1.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, //
        ];
        let arr = to_dyn([3, 4], e);
        let mins = local_minima_inner(arr.view(), 1, None);
        let coords: Vec<Vec<usize>> = mins.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords.contains(&vec![1, 1]));
        assert!(coords.contains(&vec![1, 2]));
    }

    #[test]
    fn nan_only_neighborhood_is_not_a_minimum() {
        // Single valid cell, all neighbors NaN → must NOT be a minimum
        // (no valid neighbor to compare against).
        let n = f64::NAN;
        let e = vec![n, n, n, n, 0.0, n, n, n, n];
        let arr = to_dyn([3, 3], e);
        let mins = local_minima_inner(arr.view(), 1, None);
        assert!(mins.is_empty());
    }

    #[test]
    fn output_is_sorted_ascending_by_energy() {
        // Gradient fill so no two cells tie; only the two explicitly
        // lowered corners (linear 0 and 24) qualify as local minima.
        let mut e: Vec<f64> = (0..25).map(|i| 100.0 + i as f64).collect();
        e[0] = 2.0;
        e[24] = 1.0;
        let arr = to_dyn([5, 5], e);
        let mins = local_minima_inner(arr.view(), 1, None);
        assert_eq!(mins.len(), 2);
        assert!(mins[0].1 <= mins[1].1);
        assert_eq!(mins[0].0, vec![4, 4]);
        assert_eq!(mins[0].1, 1.0);
    }

    #[test]
    fn dimensionality_sweep_2_to_7() {
        // For each ndim in 2..=7, build a tiny grid whose center cell is
        // 0.0 and all others are 1.0; assert the center is the only min.
        for ndim in 2..=7 {
            let shape = vec![3usize; ndim];
            let total: usize = shape.iter().product();
            let mut data = vec![1.0; total];

            // Build center index = (1, 1, ..., 1).
            let strides = compute_strides(&shape);
            let center_lin = index_to_linear_local(&vec![1; ndim], &strides);
            data[center_lin] = 0.0;

            let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let mins = local_minima_inner(arr.view(), 1, None);
            assert_eq!(mins.len(), 1, "ndim={ndim}");
            assert_eq!(mins[0].0, vec![1; ndim], "ndim={ndim}");
            assert_eq!(mins[0].1, 0.0, "ndim={ndim}");
        }
    }

    // Local helper to avoid a circular dep on common::nd public re-exports.
    fn index_to_linear_local(idx: &[usize], strides: &[usize]) -> usize {
        idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    }

    #[test]
    fn r2_excludes_minimum_with_far_diagonal_lower() {
        // 5x5 with (2,2) = 3 and (4,4) = 1; the rest = 5.
        //
        // At r=1: (2,2)'s eight neighbors are all 5 — it qualifies.
        //         (4,4)'s three neighbors are all 5 — it also qualifies.
        // At r=2: (2,2) now sees (4,4)=1 (Chebyshev distance 2) — disqualified.
        //         (4,4) still sees only values ≥ 1 in its r=2 stencil — qualifies.
        let mut e = vec![5.0; 25];
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);

        let mins_r1 = local_minima_inner(arr.view(), 1, None);
        let coords_r1: Vec<Vec<usize>> = mins_r1.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords_r1.contains(&vec![2, 2]));
        assert!(coords_r1.contains(&vec![4, 4]));

        let mins_r2 = local_minima_inner(arr.view(), 2, None);
        let coords_r2: Vec<Vec<usize>> = mins_r2.iter().map(|(c, _)| c.clone()).collect();
        assert!(!coords_r2.contains(&vec![2, 2]));
        assert!(coords_r2.contains(&vec![4, 4]));
    }

    #[test]
    fn r2_finds_minimum_when_far_neighbors_all_higher() {
        // 5x5 bowl: center (2,2) = 0, all others = 1. Center is a minimum
        // at r=1 AND r=2 (every cell in its 5x5 stencil is 1 ≥ 0).
        // A 5x5 grid ensures every non-center cell lies within Chebyshev
        // distance 2 of center, so all 1-valued cells see the 0 and are
        // disqualified, leaving center as the unique minimum.
        let mut e = vec![1.0; 25];
        e[2 * 5 + 2] = 0.0;
        let arr = to_dyn([5, 5], e);
        let mins = local_minima_inner(arr.view(), 2, None);
        assert_eq!(mins.len(), 1);
        assert_eq!(mins[0].0, vec![2, 2]);
        assert_eq!(mins[0].1, 0.0);
    }

    #[test]
    fn confirm_range_none_matches_direct_check() {
        // The r2_excludes_minimum_with_far_diagonal_lower fixture:
        // 5x5 with (2,2)=3, (4,4)=1, everything else 5.
        let mut e = vec![5.0; 25];
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);

        // None must reproduce the legacy direct-check behavior at both r=1 and r=2.
        let direct_r1 = local_minima_inner(arr.view(), 1, None);
        let direct_r2 = local_minima_inner(arr.view(), 2, None);

        let coords_r1: Vec<Vec<usize>> = direct_r1.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords_r1.contains(&vec![2, 2]));
        assert!(coords_r1.contains(&vec![4, 4]));

        let coords_r2: Vec<Vec<usize>> = direct_r2.iter().map(|(c, _)| c.clone()).collect();
        assert!(!coords_r2.contains(&vec![2, 2]));
        assert!(coords_r2.contains(&vec![4, 4]));
    }

    #[test]
    fn confirm_range_equals_neighborhood_is_noop() {
        let mut e = vec![5.0; 25];
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);

        let none = local_minima_inner(arr.view(), 2, None);
        let same = local_minima_inner(arr.view(), 2, Some(2));
        assert_eq!(none, same);
    }

    #[test]
    fn find_1_confirm_2_matches_direct_2() {
        // KEY CORRECTNESS EQUIVALENCE: find at r=1 + confirm at r=2 must
        // equal direct check at r=2.
        let mut e = vec![5.0; 25];
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);

        let two_pass = local_minima_inner(arr.view(), 1, Some(2));
        let direct = local_minima_inner(arr.view(), 2, None);
        assert_eq!(two_pass, direct);
    }

    #[test]
    fn find_1_confirm_r_equivalence_sweep() {
        // Deterministic 3D fixture: 4x4x4 grid with two planted minima
        // and a gradient backdrop to avoid ties.
        let shape = [4usize, 4, 4];
        let total: usize = shape.iter().product();
        let mut data: Vec<f64> = (0..total).map(|i| 10.0 + (i as f64) * 0.5).collect();
        // Plant low values at two corners so multiple R-stencils interact.
        data[0] = 1.0;
        data[total - 1] = 0.5;
        let arr = to_dyn(shape, data);

        for r in 2usize..=5 {
            let two_pass = local_minima_inner(arr.view(), 1, Some(r));
            let direct = local_minima_inner(arr.view(), r, None);
            assert_eq!(two_pass, direct, "mismatch at r={r}");
        }
    }

    #[test]
    fn confirm_culls_r1_candidates() {
        // Fixture: 5x5 with three planted dips at (0,0)=2, (2,2)=3, (4,4)=1,
        // everything else 5. At r=1 all three qualify (they each beat their
        // king-move neighbors). At confirm r=2 the (2,2)=3 cell sees (4,4)=1
        // and (0,0)=2 within its 5x5 stencil — both lower — so it is culled.
        // (0,0)=2 sees (2,2)=3 (higher, ok) but NOT (4,4)=1 (Chebyshev 4 > 2),
        // so it survives. (4,4)=1 sees (2,2)=3 (higher, ok), so it survives.
        let mut e = vec![5.0; 25];
        e[0] = 2.0;
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);

        let r1 = local_minima_inner(arr.view(), 1, None);
        let coords_r1: Vec<Vec<usize>> = r1.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords_r1.contains(&vec![0, 0]));
        assert!(coords_r1.contains(&vec![2, 2]));
        assert!(coords_r1.contains(&vec![4, 4]));

        let two_pass = local_minima_inner(arr.view(), 1, Some(2));
        let coords_2p: Vec<Vec<usize>> = two_pass.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords_2p.contains(&vec![0, 0]));
        assert!(!coords_2p.contains(&vec![2, 2]));
        assert!(coords_2p.contains(&vec![4, 4]));
        assert_eq!(coords_2p.len(), 2);
    }

    #[test]
    fn find_1_confirm_r_equivalence_with_nan_walls() {
        // 5x5 grid with a vertical NaN strip at column 2 and two planted
        // minima at (1,1)=3 and (1,3)=1. At r=1 both survive because their
        // non-NaN neighbours are all higher. At r >= 2 they see each other
        // across the NaN strip (NaN cells are skipped, not blocking), so
        // (1,1)=3 is culled by (1,3)=1. The two-pass invariant
        // local_minima_inner(arr, 1, Some(R)) == local_minima_inner(arr, R, None)
        // must hold even when NaN cells fall inside the stencil.
        let n = f64::NAN;
        #[rustfmt::skip]
        let e = vec![
            5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 3.0,   n, 1.0, 5.0,
            5.0, 5.0,   n, 5.0, 5.0,
            5.0, 5.0,   n, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0,
        ];
        let arr = to_dyn([5, 5], e);

        for r in 2usize..=4 {
            let two_pass = local_minima_inner(arr.view(), 1, Some(r));
            let direct = local_minima_inner(arr.view(), r, None);
            assert_eq!(two_pass, direct, "mismatch at r={r}");
        }
    }

    #[test]
    fn single_peak_2d_has_one_maximum_at_center() {
        // 3x3 mound: center is 1, surroundings are 0.
        let e = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let arr = to_dyn([3, 3], e);
        let maxes = local_maxima_inner(arr.view(), 1, None);
        assert_eq!(maxes.len(), 1);
        assert_eq!(maxes[0].0, vec![1, 1]);
        assert_eq!(maxes[0].1, 1.0);
    }

    #[test]
    fn two_peaks_2d_returns_both() {
        // 5x5 with two peaks at (0,0)=2 and (4,4)=2, valley of -3 between.
        let mut e = vec![-3.0; 25];
        e[0] = 2.0;
        e[1] = 1.0;
        e[5] = 1.0;
        e[6] = 0.0;
        e[24] = 2.0;
        e[23] = 1.0;
        e[19] = 1.0;
        e[18] = 0.0;
        let arr = to_dyn([5, 5], e);
        let maxes = local_maxima_inner(arr.view(), 1, None);
        let coords: Vec<Vec<usize>> = maxes.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords.contains(&vec![0, 0]));
        assert!(coords.contains(&vec![4, 4]));
    }

    #[test]
    fn diagonal_neighbor_disqualifies_maximum() {
        // (1,1) = 5 with (0,0) = 10 — diagonal — must NOT be a local max.
        let e = vec![
            10.0, 0.0, 0.0, //
            0.0, 5.0, 0.0,  //
            0.0, 0.0, 0.0,  //
        ];
        let arr = to_dyn([3, 3], e);
        let maxes = local_maxima_inner(arr.view(), 1, None);
        let coords: Vec<Vec<usize>> = maxes.iter().map(|(c, _)| c.clone()).collect();
        assert!(!coords.contains(&vec![1, 1]));
        assert!(coords.contains(&vec![0, 0]));
    }

    #[test]
    fn ties_are_local_maxima() {
        // Two adjacent cells with energy 10 surrounded by 0s; both qualify
        // because the predicate is strict `>`.
        let e = vec![
            0.0, 0.0, 0.0, 0.0,  //
            0.0, 10.0, 10.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,  //
        ];
        let arr = to_dyn([3, 4], e);
        let maxes = local_maxima_inner(arr.view(), 1, None);
        let coords: Vec<Vec<usize>> = maxes.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords.contains(&vec![1, 1]));
        assert!(coords.contains(&vec![1, 2]));
    }

    #[test]
    fn nan_only_neighborhood_is_not_a_maximum() {
        // Single valid cell, all neighbors NaN → must NOT be a maximum.
        let n = f64::NAN;
        let e = vec![n, n, n, n, 10.0, n, n, n, n];
        let arr = to_dyn([3, 3], e);
        let maxes = local_maxima_inner(arr.view(), 1, None);
        assert!(maxes.is_empty());
    }

    #[test]
    fn maxima_output_is_sorted_descending_by_energy() {
        // Gradient fill so no two cells tie; only the two explicitly
        // raised corners (linear 0 and 24) qualify as local maxima.
        let mut e: Vec<f64> = (0..25).map(|i| -100.0 - i as f64).collect();
        e[0] = -2.0;
        e[24] = -1.0;
        let arr = to_dyn([5, 5], e);
        let maxes = local_maxima_inner(arr.view(), 1, None);
        assert_eq!(maxes.len(), 2);
        assert!(maxes[0].1 >= maxes[1].1);
        assert_eq!(maxes[0].0, vec![4, 4]);
        assert_eq!(maxes[0].1, -1.0);
    }

    #[test]
    fn maxima_dimensionality_sweep_2_to_7() {
        // For each ndim in 2..=7, build a tiny grid whose center cell is
        // 1.0 and all others are 0.0; assert the center is the only max.
        for ndim in 2..=7 {
            let shape = vec![3usize; ndim];
            let total: usize = shape.iter().product();
            let mut data = vec![0.0; total];

            let strides = compute_strides(&shape);
            let center_lin: usize = vec![1usize; ndim]
                .iter()
                .zip(strides.iter())
                .map(|(i, s)| i * s)
                .sum();
            data[center_lin] = 1.0;

            let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let maxes = local_maxima_inner(arr.view(), 1, None);
            assert_eq!(maxes.len(), 1, "ndim={ndim}");
            assert_eq!(maxes[0].0, vec![1; ndim], "ndim={ndim}");
            assert_eq!(maxes[0].1, 1.0, "ndim={ndim}");
        }
    }

    #[test]
    fn maxima_find_1_confirm_2_matches_direct_2() {
        // Mirror of find_1_confirm_2_matches_direct_2 with values negated.
        let mut e = vec![-5.0; 25];
        e[2 * 5 + 2] = -3.0;
        e[4 * 5 + 4] = -1.0;
        let arr = to_dyn([5, 5], e);

        let two_pass = local_maxima_inner(arr.view(), 1, Some(2));
        let direct = local_maxima_inner(arr.view(), 2, None);
        assert_eq!(two_pass, direct);
    }

    #[test]
    fn maxima_find_1_confirm_r_equivalence_sweep() {
        // Same fixture style as find_1_confirm_r_equivalence_sweep, negated.
        let shape = [4usize, 4, 4];
        let total: usize = shape.iter().product();
        let mut data: Vec<f64> = (0..total).map(|i| -10.0 - (i as f64) * 0.5).collect();
        data[0] = -1.0;
        data[total - 1] = -0.5;
        let arr = to_dyn(shape, data);

        for r in 2usize..=5 {
            let two_pass = local_maxima_inner(arr.view(), 1, Some(r));
            let direct = local_maxima_inner(arr.view(), r, None);
            assert_eq!(two_pass, direct, "mismatch at r={r}");
        }
    }

    #[test]
    fn maxima_find_1_confirm_r_equivalence_with_nan_walls() {
        // Mirror of find_1_confirm_r_equivalence_with_nan_walls (negated).
        let n = f64::NAN;
        #[rustfmt::skip]
        let e = vec![
            -5.0, -5.0, -5.0, -5.0, -5.0,
            -5.0, -3.0,    n, -1.0, -5.0,
            -5.0, -5.0,    n, -5.0, -5.0,
            -5.0, -5.0,    n, -5.0, -5.0,
            -5.0, -5.0, -5.0, -5.0, -5.0,
        ];
        let arr = to_dyn([5, 5], e);

        for r in 2usize..=4 {
            let two_pass = local_maxima_inner(arr.view(), 1, Some(r));
            let direct = local_maxima_inner(arr.view(), r, None);
            assert_eq!(two_pass, direct, "mismatch at r={r}");
        }
    }

    #[test]
    fn local_minima_f32_matches_f64() {
        // 5x5 paraboloid bowl: single interior minimum at the center (2,2).
        let mut v64 = Vec::new();
        for i in 0..5i64 {
            for j in 0..5i64 {
                v64.push(((i - 2).pow(2) + (j - 2).pow(2)) as f64);
            }
        }
        let a64 = Array::from_shape_vec(IxDyn(&[5, 5]), v64.clone()).unwrap();
        let v32: Vec<f32> = v64.iter().map(|&x| x as f32).collect();
        let a32 = Array::from_shape_vec(IxDyn(&[5, 5]), v32).unwrap();

        let r64 = local_minima_inner(a64.view(), 1, Some(2));
        let r32 = local_minima_inner(a32.view(), 1, Some(2));

        let i64s: Vec<Vec<usize>> = r64.into_iter().map(|(idx, _)| idx).collect();
        let i32s: Vec<Vec<usize>> = r32.into_iter().map(|(idx, _)| idx).collect();
        assert_eq!(i64s, i32s);
        assert!(!i64s.is_empty(), "fixture should yield at least one minimum");
    }

    /// Sequential reference using the materialised stencil (the 0.9.0 loop).
    fn reference_minima(arr: &ArrayD<f64>, r: usize) -> Vec<(usize, f64)> {
        let shape = arr.shape().to_vec();
        let strides = compute_strides(&shape);
        let flat = arr.as_slice().unwrap();
        let mut nbrs = Vec::new();
        let mut out = Vec::new();
        for (lin, &e) in flat.iter().enumerate() {
            if e.is_nan() { continue; }
            crate::common::nd::full_neighbors(lin, &shape, &strides, r, &mut nbrs);
            let mut has_valid = false;
            let mut beats_all = true;
            for &n in &nbrs {
                let ne = flat[n];
                if ne.is_nan() { continue; }
                has_valid = true;
                if ne < e { beats_all = false; break; }
            }
            if has_valid && beats_all { out.push((lin, e)); }
        }
        out
    }

    #[test]
    fn parallel_lazy_scan_matches_sequential_reference_on_random_5d_with_nans() {
        let shape = [9usize, 8, 7, 6, 5];
        let total: usize = shape.iter().product();
        let mut x: u64 = 0x2545F4914F6CDD1D;
        let data: Vec<f64> = (0..total)
            .map(|i| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if i % 13 == 0 { f64::NAN } else { ((x >> 11) as f64) / ((1u64 << 53) as f64) }
            })
            .collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
        for r in 1..=2 {
            let expected = reference_minima(&arr, r);
            let got = local_extreme_inner(arr.view(), r, None, |ne, e| ne < e);
            assert_eq!(got, expected, "r={r}");
            // Candidate order is ascending linear index regardless of chunking.
            assert!(got.windows(2).all(|w| w[0].0 < w[1].0));
        }
    }
}
