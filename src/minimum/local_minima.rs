//! Local-minimum search on a dense N-D grid using the 3ᴺ−1 (king-move)
//! neighborhood. A cell is a local minimum iff its energy is non-NaN, at
//! least one in-bounds neighbor is non-NaN, and no non-NaN neighbor is
//! strictly less.

use ndarray::ArrayViewD;

use crate::common::nd::{compute_strides, full_neighbors, linear_to_index};

/// Find all strict local minima in `energies` using the Chebyshev box of
/// half-width `r` (i.e. the (2r+1)ᴺ−1 stencil; `r = 1` is the classic
/// king-move 3ᴺ−1 connectivity).
///
/// Returns a vector of `(nd_index, energy)` for every qualifying cell,
/// sorted ascending by energy (ties broken by `f64::total_cmp` for
/// determinism).
///
/// Preconditions (enforced by the PyO3 wrapper before calling):
/// - `energies` is a view over a C-contiguous f64 buffer.
/// - `energies.ndim()` is in `[2, 7]`.
/// - `energies.len() <= u32::MAX`.
/// - `r >= 1` (validated upstream).
pub fn local_minima_inner(energies: ArrayViewD<'_, f64>, r: usize) -> Vec<(Vec<usize>, f64)> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let flat: &[f64] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    let stencil_max = (2 * r + 1).saturating_pow(shape.len() as u32);
    let mut nbrs: Vec<usize> = Vec::with_capacity(stencil_max);
    let mut out: Vec<(Vec<usize>, f64)> = Vec::new();

    for (lin, &e) in flat.iter().enumerate() {
        if e.is_nan() {
            continue;
        }
        full_neighbors(lin, &shape, &strides, r, &mut nbrs);

        let mut has_valid_neighbor = false;
        let mut beats_all = true;
        for &nbr_lin in &nbrs {
            let ne = flat[nbr_lin];
            if ne.is_nan() {
                continue;
            }
            has_valid_neighbor = true;
            if ne < e {
                beats_all = false;
                break;
            }
        }

        if has_valid_neighbor && beats_all {
            out.push((linear_to_index(lin, &shape, &strides), e));
        }
    }

    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
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
        let mins = local_minima_inner(arr.view(), 1);
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
        let mins = local_minima_inner(arr.view(), 1);
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
        let mins = local_minima_inner(arr.view(), 1);
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
        let mins = local_minima_inner(arr.view(), 1);
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
        let mins = local_minima_inner(arr.view(), 1);
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
        let mins = local_minima_inner(arr.view(), 1);
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
            let mins = local_minima_inner(arr.view(), 1);
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

        let mins_r1 = local_minima_inner(arr.view(), 1);
        let coords_r1: Vec<Vec<usize>> = mins_r1.iter().map(|(c, _)| c.clone()).collect();
        assert!(coords_r1.contains(&vec![2, 2]));
        assert!(coords_r1.contains(&vec![4, 4]));

        let mins_r2 = local_minima_inner(arr.view(), 2);
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
        let mins = local_minima_inner(arr.view(), 2);
        assert_eq!(mins.len(), 1);
        assert_eq!(mins[0].0, vec![2, 2]);
        assert_eq!(mins[0].1, 0.0);
    }
}
