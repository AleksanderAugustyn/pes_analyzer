//! Imaginary water flow (watershed) saddle search on a dense N-D grid.

use ndarray::ArrayViewD;

use crate::common::dsu::DisjointSetUnion;
use crate::common::nd::{axis_neighbors, compute_strides, index_to_linear, linear_to_index};

/// Search for the saddle point between `start_idx` and `end_idx` on
/// `energies` using the watershed algorithm.
///
/// Returns `Some((nd_index, energy))` for the saddle cell, or `None` if the
/// two basins never connect within the non-NaN region.
///
/// Preconditions (all enforced by the PyO3 wrapper before calling this
/// function):
/// - `energies` is a view over a C-contiguous f64 buffer.
/// - `energies.ndim() == start_idx.len() == end_idx.len()`, in `[2, 7]`.
/// - Both `start_idx` and `end_idx` are in bounds and reference non-NaN cells.
/// - `energies.len() <= u32::MAX`.
pub fn iwf_grid_inner(
    energies: ArrayViewD<'_, f64>,
    start_idx: &[usize],
    end_idx: &[usize],
) -> Option<(Vec<usize>, f64)> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let n_total = energies.len();

    let flat: &[f64] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    let start_lin = index_to_linear(start_idx, &strides);
    let end_lin = index_to_linear(end_idx, &strides);

    // Fast path: start == end.
    if start_lin == end_lin {
        return Some((start_idx.to_vec(), flat[start_lin]));
    }

    // Sweep non-NaN cells, building `sorted` and the linear→compact remap.
    let mut remap: Vec<u32> = vec![u32::MAX; n_total];
    let mut sorted: Vec<(u32, f64)> = Vec::with_capacity(n_total);
    for i in 0..n_total {
        let e = flat[i];
        if !e.is_nan() {
            remap[i] = sorted.len() as u32;
            sorted.push((i as u32, e));
        }
    }

    let start_compact = remap[start_lin];
    let end_compact = remap[end_lin];
    // Upstream validation guarantees non-NaN at both endpoints, so neither
    // can be u32::MAX here.
    debug_assert_ne!(start_compact, u32::MAX);
    debug_assert_ne!(end_compact, u32::MAX);

    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let n_valid = sorted.len();
    let mut dsu = DisjointSetUnion::new(n_valid);
    let mut processed: Vec<bool> = vec![false; n_valid];

    // 2 * ndim is the maximum neighbor count.
    let mut nbrs: Vec<usize> = Vec::with_capacity(2 * shape.len());

    for &(lin_u32, energy) in &sorted {
        let lin = lin_u32 as usize;
        let cur_compact = remap[lin];
        processed[cur_compact as usize] = true;

        axis_neighbors(lin, &shape, &strides, &mut nbrs);
        for &nbr_lin in &nbrs {
            let nbr_compact = remap[nbr_lin];
            if nbr_compact == u32::MAX {
                continue;
            }
            if processed[nbr_compact as usize] {
                dsu.union(cur_compact, nbr_compact);
            }
        }

        if dsu.find(start_compact) == dsu.find(end_compact) {
            let nd_idx = linear_to_index(lin, &shape, &strides);
            return Some((nd_idx, energy));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD, IxDyn};

    fn to_dyn<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    #[test]
    fn start_equals_end_returns_start_immediately() {
        let arr = to_dyn([3, 3], vec![0.0; 9]);
        let result = iwf_grid_inner(arr.view(), &[1, 1], &[1, 1]).unwrap();
        assert_eq!(result.0, vec![1, 1]);
        assert_eq!(result.1, 0.0);
    }

    #[test]
    fn two_minima_separated_by_ridge_2d() {
        // 5x5 grid: two minima at (0,0)=0 and (4,4)=0.
        // A bridge row at row 2 connects them; saddle is (2,2)=4.0.
        // Basin 1 drains into the top-left, basin 2 into the bottom-right.
        // All other cells are 10 (impassable for the cheap path).
        let mut e = vec![10.0; 25];
        // Basin 1 around (0,0)
        e[0] = 0.0;
        e[1] = 1.0;
        e[5] = 1.0;
        e[6] = 2.0;
        // Bridge row: (2,0)–(2,4), saddle at (2,2)=4, shoulders at 3
        e[10] = 3.0;
        e[11] = 3.0;
        e[12] = 4.0; // saddle
        e[13] = 3.0;
        e[14] = 3.0;
        // Basin 2 around (4,4)
        e[4 * 5 + 4] = 0.0;
        e[3 * 5 + 4] = 1.0;
        e[4 * 5 + 3] = 1.0;
        e[3 * 5 + 3] = 2.0;
        let arr = to_dyn([5, 5], e);
        let (idx, energy) = iwf_grid_inner(arr.view(), &[0, 0], &[4, 4]).unwrap();
        assert_eq!(energy, 4.0);
        assert_eq!(idx, vec![2, 2]);
    }

    #[test]
    fn nan_wall_blocks_connection_returns_none() {
        // 3x5 grid. Middle column is NaN: an impassable wall.
        let nan = f64::NAN;
        let e = vec![
            0.0, 1.0, nan, 1.0, 0.0, 1.0, 2.0, nan, 2.0, 1.0, 0.0, 1.0, nan, 1.0, 0.0,
        ];
        let arr = to_dyn([3, 5], e);
        let result = iwf_grid_inner(arr.view(), &[1, 0], &[1, 4]);
        assert!(result.is_none());
    }

    #[test]
    fn adjacent_basins_saddle_is_max_of_endpoints() {
        let e = vec![0.0, 5.0, 0.0, 10.0, 10.0, 10.0];
        let arr = to_dyn([2, 3], e);
        // start=(0,0), end=(0,2). Bridge is (0,1)=5.0 on the cheap row.
        let (idx, energy) = iwf_grid_inner(arr.view(), &[0, 0], &[0, 2]).unwrap();
        assert_eq!(energy, 5.0);
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn dimensionality_sweep_2_to_7() {
        // For each ndim in 2..=7, build a tiny grid with a known single
        // saddle. Shape is (3, 1, 1, ..., 1): two minima at (0, 0, ..., 0)
        // and (2, 0, ..., 0), saddle at (1, 0, ..., 0) with energy 7.0.
        for ndim in 2..=7 {
            let mut shape = vec![1usize; ndim];
            shape[0] = 3;
            let total: usize = shape.iter().product();
            let mut data = vec![7.0; total];
            data[0] = 0.0;
            data[total - 1] = 0.0;
            let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let start = vec![0usize; ndim];
            let mut end = vec![0usize; ndim];
            end[0] = 2;
            let (_idx, energy) = iwf_grid_inner(arr.view(), &start, &end).unwrap();
            assert_eq!(energy, 7.0, "ndim={ndim}");
        }
    }
}
