//! Full watershed segmentation: flood the grid to completion, labeling
//! every non-NaN cell with its initial basin and recording every union
//! that merges two distinct basins as a `(saddle, deeper, shallower)`
//! tuple.

use ndarray::ArrayViewD;

use crate::common::dsu::DisjointSetUnion;
use crate::common::nd::{axis_neighbors, compute_strides, linear_to_index};

/// Result of a full watershed flood. See `API.md` for the semantics.
pub struct SegmentationResult {
    pub labels: Vec<i32>,
    pub basins: Vec<(Vec<usize>, f64)>,
    pub merges: Vec<(Vec<usize>, f64, u32, u32)>,
}

/// Full watershed flood with merge-tree recording.
///
/// Preconditions (enforced by the PyO3 wrapper):
/// - `energies` is a view over a C-contiguous f64 buffer.
/// - `energies.ndim() in [2, 7]`.
/// - `energies.len() <= u32::MAX`.
pub fn watershed_segmentation_inner(
    energies: ArrayViewD<'_, f64>,
) -> SegmentationResult {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let n_total = energies.len();

    let flat: &[f64] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

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
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let n_valid = sorted.len();
    let mut dsu = DisjointSetUnion::new(n_valid);
    let mut processed: Vec<bool> = vec![false; n_valid];
    // basin_of_root[r] is the basin ID owned by the DSU component rooted at r.
    // Only valid for cells currently acting as DSU roots; stale entries on
    // non-root cells are never read (`dsu.find` does not return them).
    let mut basin_of_root: Vec<u32> = vec![u32::MAX; n_valid];
    // One basin ID per compact cell; the basin the cell FIRST joined during
    // the flood (its "initial flood basin"), before any later merge events.
    let mut compact_labels: Vec<u32> = vec![u32::MAX; n_valid];

    let mut basins: Vec<(u32, f64)> = Vec::new();
    let mut merges: Vec<(u32, f64, u32, u32)> = Vec::new();

    let mut nbrs: Vec<usize> = Vec::with_capacity(2 * shape.len());

    // `sorted` is iterated by index so we can re-sort the compact remap
    // for nd-index lookup later. Indexing rather than `for &(...)` because
    // the borrow checker disallows mutating `remap`/`compact_labels` while
    // iterating `sorted` by reference.
    for &(lin_u32, energy) in &sorted {
        let lin = lin_u32 as usize;
        let cur_compact = remap[lin];
        processed[cur_compact as usize] = true;

        axis_neighbors(lin, &shape, &strides, &mut nbrs);

        // `my_basin` tracks the basin of the current cell's component as
        // we process neighbors. It is set on the first processed neighbor
        // and updated to the surviving basin after every merge event so
        // that a third basin meeting the same cell merges with the
        // already-survived basin, not the originally-adopted one.
        let mut my_basin: Option<u32> = None;

        for &nbr_lin in &nbrs {
            let nbr_compact = remap[nbr_lin];
            if nbr_compact == u32::MAX || !processed[nbr_compact as usize] {
                continue;
            }
            let nbr_root = dsu.find(nbr_compact);
            let nbr_basin = basin_of_root[nbr_root as usize];

            match my_basin {
                None => {
                    // First processed neighbor: adopt its basin and union.
                    my_basin = Some(nbr_basin);
                    compact_labels[cur_compact as usize] = nbr_basin;
                    dsu.union(cur_compact, nbr_root);
                    let new_root = dsu.find(cur_compact);
                    basin_of_root[new_root as usize] = nbr_basin;
                }
                Some(cur_basin) if cur_basin == nbr_basin => {
                    // Same component already; union is no-op or maintains
                    // structure. Refresh basin_of_root at the new root in
                    // case the DSU re-parented.
                    dsu.union(cur_compact, nbr_root);
                    let new_root = dsu.find(cur_compact);
                    basin_of_root[new_root as usize] = cur_basin;
                }
                Some(cur_basin) => {
                    // Distinct basins meet at this cell — merge event.
                    // Convention: deeper.min_e <= shallower.min_e.
                    let (deeper, shallower) =
                        if basins[cur_basin as usize].1 <= basins[nbr_basin as usize].1 {
                            (cur_basin, nbr_basin)
                        } else {
                            (nbr_basin, cur_basin)
                        };
                    merges.push((lin_u32, energy, deeper, shallower));
                    dsu.union(cur_compact, nbr_root);
                    let new_root = dsu.find(cur_compact);
                    basin_of_root[new_root as usize] = deeper;
                    my_basin = Some(deeper);
                }
            }
        }

        if my_basin.is_none() {
            // No processed neighbors → this cell starts a new basin.
            // Because cells are processed in ascending energy order, the
            // first cell of a component is also its minimum.
            let new_id = basins.len() as u32;
            basins.push((lin_u32, energy));
            basin_of_root[cur_compact as usize] = new_id;
            compact_labels[cur_compact as usize] = new_id;
        }
    }

    // Expand compact labels back onto the full buffer; NaN cells get -1.
    let mut labels: Vec<i32> = vec![-1; n_total];
    for i in 0..n_total {
        let c = remap[i];
        if c != u32::MAX {
            labels[i] = compact_labels[c as usize] as i32;
        }
    }

    // Convert basin/merge linear indices into nd-indices for the Python API.
    let basins_out: Vec<(Vec<usize>, f64)> = basins
        .iter()
        .map(|&(lin, e)| (linear_to_index(lin as usize, &shape, &strides), e))
        .collect();
    let merges_out: Vec<(Vec<usize>, f64, u32, u32)> = merges
        .iter()
        .map(|&(lin, e, d, s)| (linear_to_index(lin as usize, &shape, &strides), e, d, s))
        .collect();

    SegmentationResult {
        labels,
        basins: basins_out,
        merges: merges_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD, IxDyn};

    fn to_dyn<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    #[test]
    fn single_basin_returns_one_label_no_merges() {
        // 3x3 monotone bowl; minimum at (1,1)=0, increasing outward.
        let e = vec![5.0, 4.0, 5.0, 4.0, 0.0, 4.0, 5.0, 4.0, 5.0];
        let arr = to_dyn([3, 3], e);
        let r = watershed_segmentation_inner(arr.view());
        assert_eq!(r.basins.len(), 1);
        assert_eq!(r.basins[0], (vec![1, 1], 0.0));
        assert!(r.merges.is_empty());
        for &l in &r.labels {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn two_basins_separated_by_ridge_2d() {
        // Same layout as iwf_grid::two_minima_separated_by_ridge_2d, with
        // the second minimum offset slightly to break the basin-order tie.
        let mut e = vec![10.0; 25];
        e[0] = 0.0;
        e[1] = 1.0;
        e[5] = 1.0;
        e[6] = 2.0;
        e[10] = 3.0;
        e[11] = 3.0;
        e[12] = 4.0;
        e[13] = 3.0;
        e[14] = 3.0;
        e[4 * 5 + 4] = -0.1;          // slightly deeper than (0,0) basin
        e[3 * 5 + 4] = 1.0;
        e[4 * 5 + 3] = 1.0;
        e[3 * 5 + 3] = 2.0;
        let arr = to_dyn([5, 5], e);
        let r = watershed_segmentation_inner(arr.view());
        assert_eq!(r.basins.len(), 2);
        assert_eq!(r.basins[0], (vec![4, 4], -0.1));
        assert_eq!(r.basins[1], (vec![0, 0], 0.0));
        assert_eq!(r.merges.len(), 1);
        let m = &r.merges[0];
        assert_eq!(m.0, vec![2, 2]);
        assert_eq!(m.1, 4.0);
        assert_eq!(m.2, 0);             // deeper = basin (4,4) at E=-0.1
        assert_eq!(m.3, 1);             // shallower = basin (0,0) at E=0.0
        assert!(r.basins[m.2 as usize].1 <= r.basins[m.3 as usize].1);
    }

    #[test]
    fn merge_event_uses_deeper_shallower_convention() {
        // 1-D chain (shape [1, 9]) with three basins of distinct depth.
        //
        // positions: 0   1   2   3   4   5   6   7   8
        // energies : 0   3   5   4   1   3   6   4   2
        //          A.min     sad     B.min       sad   C.min
        //
        // Expect: basins ordered (A=0, B=1, C=2);
        //         merge (pos=2, E=5, deeper=A=0, shallower=B=1);
        //         merge (pos=6, E=6, deeper=A=0, shallower=C=2).
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e);
        let r = watershed_segmentation_inner(arr.view());
        assert_eq!(r.basins.len(), 3);
        assert_eq!(r.basins[0], (vec![0, 0], 0.0));
        assert_eq!(r.basins[1], (vec![0, 4], 1.0));
        assert_eq!(r.basins[2], (vec![0, 8], 2.0));
        assert_eq!(r.merges.len(), 2);
        assert_eq!(r.merges[0], (vec![0, 2], 5.0, 0, 1));
        assert_eq!(r.merges[1], (vec![0, 6], 6.0, 0, 2));
        for m in &r.merges {
            assert!(r.basins[m.2 as usize].1 <= r.basins[m.3 as usize].1);
        }
    }

    #[test]
    fn nan_wall_yields_disconnected_basins() {
        // 3x5 grid; middle column is NaN -> two disjoint non-NaN regions.
        // Each half has a single distinct minimum (no ties) so basin
        // count and ordering are deterministic.
        let nan = f64::NAN;
        let e = vec![
            5.0, 4.0, nan, 6.0, 7.0,
            4.0, 0.0, nan, 1.0, 6.0,    // (1,1)=0 and (1,3)=1 are the two minima
            5.0, 4.0, nan, 6.0, 7.0,
        ];
        let arr = to_dyn([3, 5], e);
        let r = watershed_segmentation_inner(arr.view());
        assert_eq!(r.basins.len(), 2);
        assert!(r.merges.is_empty());
        assert_eq!(r.basins[0], (vec![1, 1], 0.0));
        assert_eq!(r.basins[1], (vec![1, 3], 1.0));
        for (i, &val) in arr.iter().enumerate() {
            if val.is_nan() {
                assert_eq!(r.labels[i], -1, "expected -1 at NaN cell {}", i);
            } else {
                assert!(r.labels[i] >= 0, "expected basin label at non-NaN cell {}", i);
            }
        }
    }

    #[test]
    fn basin_order_is_min_energy_ascending() {
        // Hand-built grid with several basins of different depths.
        let e = vec![
            0.0, 5.0, 3.0, 5.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 5.0, 2.0, 5.0, 5.0,
        ];
        let arr = to_dyn([3, 5], e);
        let r = watershed_segmentation_inner(arr.view());
        for w in r.basins.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "basins must be sorted ascending by min energy: {:?}",
                r.basins
            );
        }
    }

    #[test]
    fn labels_match_basin_indices_at_minima() {
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e);
        let r = watershed_segmentation_inner(arr.view());
        // For each basin, the cell at its minimum index must carry that
        // basin's id in `labels` (the cell that started the basin is
        // initially labeled with the basin's own id).
        for (basin_id, (idx, _e)) in r.basins.iter().enumerate() {
            let strides = compute_strides(arr.shape());
            let lin: usize = idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
            assert_eq!(r.labels[lin], basin_id as i32);
        }
    }

    #[test]
    fn labels_minus_one_at_nan_cells() {
        let nan = f64::NAN;
        let e = vec![0.0, nan, 1.0, nan, 2.0, nan, 3.0, nan, 4.0];
        let arr = to_dyn([3, 3], e);
        let r = watershed_segmentation_inner(arr.view());
        for (i, &val) in arr.iter().enumerate() {
            if val.is_nan() {
                assert_eq!(r.labels[i], -1);
            }
        }
    }

    #[test]
    fn dimensionality_sweep_2_to_7() {
        // For each ndim in 2..=7, build a tiny grid with two known minima.
        // Shape (3, 1, 1, ..., 1): minima at (0,0,..,0)=0 and (2,0,..,0)=0;
        // saddle at (1,0,..,0)=7.
        for ndim in 2..=7 {
            let mut shape = vec![1usize; ndim];
            shape[0] = 3;
            let total: usize = shape.iter().product();
            let mut data = vec![7.0; total];
            data[0] = 0.0;
            data[total - 1] = 0.0;
            let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let r = watershed_segmentation_inner(arr.view());
            assert_eq!(r.basins.len(), 2, "ndim={ndim}");
            assert_eq!(r.merges.len(), 1, "ndim={ndim}");
            assert_eq!(r.merges[0].1, 7.0, "ndim={ndim}");
        }
    }
}
