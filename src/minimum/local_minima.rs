//! Local-minimum search on a dense N-D grid using the 3ᴺ−1 (king-move)
//! neighborhood. A cell is a local minimum iff its energy is non-NaN, at
//! least one in-bounds neighbor is non-NaN, and no non-NaN neighbor is
//! strictly less.

use ndarray::ArrayViewD;

use crate::common::nd::{compute_strides, full_neighbors, linear_to_index};

/// Find all strict local minima in `energies` using 3ᴺ−1 connectivity.
///
/// Returns a vector of `(nd_index, energy)` for every qualifying cell,
/// sorted ascending by energy (ties broken by `f64::total_cmp` for
/// determinism).
///
/// Preconditions (enforced by the PyO3 wrapper before calling):
/// - `energies` is a view over a C-contiguous f64 buffer.
/// - `energies.ndim()` is in `[2, 7]`.
/// - `energies.len() <= u32::MAX`.
pub fn local_minima_inner(energies: ArrayViewD<'_, f64>) -> Vec<(Vec<usize>, f64)> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let flat: &[f64] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    let stencil_max = 3usize.saturating_pow(shape.len() as u32);
    let mut nbrs: Vec<usize> = Vec::with_capacity(stencil_max);
    let mut out: Vec<(Vec<usize>, f64)> = Vec::new();

    for (lin, &e) in flat.iter().enumerate() {
        if e.is_nan() {
            continue;
        }
        full_neighbors(lin, &shape, &strides, &mut nbrs);

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
        let mins = local_minima_inner(arr.view());
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
        let mins = local_minima_inner(arr.view());
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
        let mins = local_minima_inner(arr.view());
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
        let mins = local_minima_inner(arr.view());
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
        let mins = local_minima_inner(arr.view());
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
        let mins = local_minima_inner(arr.view());
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
            let mins = local_minima_inner(arr.view());
            assert_eq!(mins.len(), 1, "ndim={ndim}");
            assert_eq!(mins[0].0, vec![1; ndim], "ndim={ndim}");
            assert_eq!(mins[0].1, 0.0, "ndim={ndim}");
        }
    }

    // Local helper to avoid a circular dep on common::nd public re-exports.
    fn index_to_linear_local(idx: &[usize], strides: &[usize]) -> usize {
        idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
    }
}
