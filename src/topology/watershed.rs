//! Full watershed flood: label every non-NaN cell with the basin it first
//! joins and record every union of two distinct basins as a merge event.
//! Indexed by linear cell id; the DSU root of every live component is the
//! seed of its current (deepest) basin, so `labels[root]` is the basin id
//! (see ALGORITHMS.md, "Flood kernel").

use ndarray::ArrayViewD;
use rayon::prelude::*;

use crate::common::nd::{compute_strides, Stencil, PARENT_NONE};
use crate::common::scalar::Scalar;

/// Options of one flood run.
#[derive(Clone, Copy, Debug)]
pub struct FloodOptions {
    pub stencil: Stencil,
    /// Record each cell's flood parent as a direction code (`parents`).
    pub record_parents: bool,
    /// Stop as soon as these two linear indices share a component
    /// (the standalone MEP's early exit). Cells never reached keep label -1.
    pub stop_when_connected: Option<(usize, usize)>,
}

/// One merge event: two distinct live basins met at `saddle` through its
/// already-flooded neighbour `other`. `saddle_side` is the basin id of the
/// saddle cell's own side at that moment (`deeper` or `shallower`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergeEvent<T> {
    pub saddle: u32,
    pub other: u32,
    pub energy: T,
    pub deeper: u32,
    pub shallower: u32,
    pub saddle_side: u32,
}

/// Result of a flood. `basins[i] = (seed linear index, seed energy)`.
pub struct FloodResult<T> {
    pub labels: Vec<i32>,
    pub parents: Option<Vec<u16>>,
    pub basins: Vec<(u32, T)>,
    pub merges: Vec<MergeEvent<T>>,
}

/// Representative of `x`'s component, with path halving.
#[inline]
fn find_root(parent: &mut [u32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize];
        if p == x {
            return x;
        }
        let gp = parent[p as usize];
        parent[x as usize] = gp;
        x = gp;
    }
}

/// Non-NaN cells in ascending `(energy, linear index)` order. The pair sort
/// runs on the rayon pool; the order is independent of the thread count.
/// Transient: size_of::<(T, u32)>() * V for the pairs (8V for f32, 16V for f64 —
/// the f64 pair is padded to 16 bytes) plus 4V for the output: 12V / 20V.
pub fn sorted_order<T: Scalar>(flat: &[T]) -> Vec<u32> {
    let n_valid = flat.par_iter().filter(|e| !e.is_nan()).count();
    let mut pairs: Vec<(T, u32)> = Vec::with_capacity(n_valid);
    pairs.extend(
        flat.iter()
            .enumerate()
            .filter(|(_, e)| !e.is_nan())
            .map(|(i, &e)| (e, i as u32)),
    );
    pairs.par_sort_unstable_by(|a, b| a.0.tcmp(&b.0).then(a.1.cmp(&b.1)));
    let mut order: Vec<u32> = Vec::with_capacity(n_valid);
    order.extend(pairs.iter().map(|p| p.1));
    order
}

/// Flood `energies` in ascending order. Preconditions (enforced by the
/// PyO3 wrapper): C-contiguous, ndim in [2, 7], len <= u32::MAX.
///
/// Peak resident memory: 8N + 4V bytes (+2N with `record_parents`) after the
/// sort's 12V (f32) / 20V (f64) transient has been released.
pub fn flood<T: Scalar>(energies: ArrayViewD<'_, T>, opts: FloodOptions) -> FloodResult<T> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let n_total = energies.len();
    let flat: &[T] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    let order = sorted_order(flat);

    let mut parent: Vec<u32> = (0..n_total as u32).collect();
    let mut labels: Vec<i32> = vec![-1; n_total];
    let mut parents: Option<Vec<u16>> = opts.record_parents.then(|| vec![PARENT_NONE; n_total]);
    let mut basins: Vec<(u32, T)> = Vec::new();
    let mut merges: Vec<MergeEvent<T>> = Vec::new();
    let mut nbrs: Vec<(usize, u16)> = Vec::new();

    for &lin_u32 in &order {
        let lin = lin_u32 as usize;
        let e = flat[lin];
        opts.stencil.neighbors_with_codes(lin, &shape, &strides, &mut nbrs);

        // Basin and DSU root of the current cell's component while its
        // neighbours are scanned; updated at every merge so a third basin
        // meeting the same cell merges with the survivor.
        let mut my_basin: Option<u32> = None;
        let mut my_root: u32 = lin_u32;

        for &(nbr, code) in &nbrs {
            if labels[nbr] < 0 {
                continue; // NaN or not yet flooded
            }
            let r = find_root(&mut parent, nbr as u32);
            let b = labels[r as usize] as u32;
            match my_basin {
                None => {
                    // First flooded neighbour: adopt its basin, hang under its root.
                    parent[lin] = r;
                    labels[lin] = b as i32;
                    if let Some(p) = parents.as_mut() {
                        p[lin] = code;
                    }
                    my_basin = Some(b);
                    my_root = r;
                }
                Some(mb) if mb == b => {
                    // One live component per basin id, so this is the same component.
                    debug_assert_eq!(r, my_root);
                }
                Some(mb) => {
                    // Two distinct basins meet here: merge, deeper survives.
                    // Ids are monotone in (seed energy, seed index), so the
                    // lower id is the deeper basin and ties resolve to the
                    // lower seed index; basin 0 can never be `shallower`.
                    let (deeper, shallower, deep_root, shallow_root) =
                        if mb < b { (mb, b, my_root, r) } else { (b, mb, r, my_root) };
                    parent[shallow_root as usize] = deep_root;
                    my_root = deep_root;
                    merges.push(MergeEvent {
                        saddle: lin_u32,
                        other: nbr as u32,
                        energy: e,
                        deeper,
                        shallower,
                        saddle_side: mb,
                    });
                    my_basin = Some(deeper);
                }
            }
        }

        if my_basin.is_none() {
            // No flooded neighbour: this cell seeds a new basin and is its own root.
            let id = basins.len() as u32;
            basins.push((lin_u32, e));
            labels[lin] = id as i32;
        }

        if let Some((a, b)) = opts.stop_when_connected {
            if labels[a] >= 0
                && labels[b] >= 0
                && find_root(&mut parent, a as u32) == find_root(&mut parent, b as u32)
            {
                break;
            }
        }
    }

    FloodResult { labels, parents, basins, merges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::nd::linear_to_index;
    use ndarray::{Array, ArrayD, IxDyn};

    fn to_dyn<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    fn opts(stencil: Stencil) -> FloodOptions {
        FloodOptions { stencil, record_parents: false, stop_when_connected: None }
    }

    fn ws(arr: &ArrayD<f64>, stencil: Stencil) -> FloodResult<f64> {
        flood(arr.view(), opts(stencil))
    }

    fn nd(arr: &ArrayD<f64>, lin: u32) -> Vec<usize> {
        let shape = arr.shape().to_vec();
        let strides = compute_strides(&shape);
        linear_to_index(lin as usize, &shape, &strides)
    }

    #[test]
    fn single_basin_returns_one_label_no_merges() {
        let e = vec![5.0, 4.0, 5.0, 4.0, 0.0, 4.0, 5.0, 4.0, 5.0];
        let arr = to_dyn([3, 3], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert_eq!(r.basins.len(), 1);
        assert_eq!(nd(&arr, r.basins[0].0), vec![1, 1]);
        assert_eq!(r.basins[0].1, 0.0);
        assert!(r.merges.is_empty());
        assert!(r.labels.iter().all(|&l| l == 0));
        assert!(r.parents.is_none());
    }

    #[test]
    fn two_basins_separated_by_ridge_2d() {
        let mut e = vec![10.0; 25];
        e[0] = 0.0; e[1] = 1.0; e[5] = 1.0; e[6] = 2.0;
        e[10] = 3.0; e[11] = 3.0; e[12] = 4.0; e[13] = 3.0; e[14] = 3.0;
        e[4 * 5 + 4] = -0.1; e[3 * 5 + 4] = 1.0; e[4 * 5 + 3] = 1.0; e[3 * 5 + 3] = 2.0;
        let arr = to_dyn([5, 5], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert_eq!(r.basins.len(), 2);
        assert_eq!(nd(&arr, r.basins[0].0), vec![4, 4]);
        assert_eq!(r.basins[0].1, -0.1);
        assert_eq!(nd(&arr, r.basins[1].0), vec![0, 0]);
        assert_eq!(r.merges.len(), 1);
        let m = &r.merges[0];
        assert_eq!(nd(&arr, m.saddle), vec![2, 2]);
        assert_eq!(m.energy, 4.0);
        assert_eq!(m.deeper, 0);
        assert_eq!(m.shallower, 1);
        assert!(m.saddle_side == 0 || m.saddle_side == 1);
        // `other` is a stencil neighbour of the saddle on the other component.
        let other_nd = nd(&arr, m.other);
        let d: usize = other_nd.iter().zip([2usize, 2]).map(|(a, b)| a.abs_diff(b)).sum();
        assert_eq!(d, 1);
        assert_ne!(r.labels[m.other as usize], m.saddle_side as i32);
    }

    #[test]
    fn merge_event_uses_deeper_shallower_convention() {
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert_eq!(r.basins.len(), 3);
        assert_eq!(r.basins.iter().map(|b| b.0).collect::<Vec<_>>(), vec![0, 4, 8]);
        assert_eq!(r.merges.len(), 2);
        assert_eq!((r.merges[0].saddle, r.merges[0].energy, r.merges[0].deeper, r.merges[0].shallower), (2, 5.0, 0, 1));
        assert_eq!((r.merges[1].saddle, r.merges[1].energy, r.merges[1].deeper, r.merges[1].shallower), (6, 6.0, 0, 2));
        for m in &r.merges {
            assert!(r.basins[m.deeper as usize].1 <= r.basins[m.shallower as usize].1);
            assert!(m.saddle_side == m.deeper || m.saddle_side == m.shallower);
        }
    }

    #[test]
    fn nan_wall_yields_disconnected_basins() {
        let nan = f64::NAN;
        let e = vec![5.0, 4.0, nan, 6.0, 7.0, 4.0, 0.0, nan, 1.0, 6.0, 5.0, 4.0, nan, 6.0, 7.0];
        let arr = to_dyn([3, 5], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert_eq!(r.basins.len(), 2);
        assert!(r.merges.is_empty());
        for (i, &val) in arr.iter().enumerate() {
            if val.is_nan() { assert_eq!(r.labels[i], -1); } else { assert!(r.labels[i] >= 0); }
        }
    }

    #[test]
    fn basin_order_is_min_energy_ascending() {
        let e = vec![0.0, 5.0, 3.0, 5.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 5.0, 2.0, 5.0, 5.0];
        let arr = to_dyn([3, 5], e);
        let r = ws(&arr, Stencil::VonNeumann);
        for w in r.basins.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn labels_at_seeds_equal_basin_ids() {
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e);
        let r = ws(&arr, Stencil::VonNeumann);
        for (bid, (lin, _)) in r.basins.iter().enumerate() {
            assert_eq!(r.labels[*lin as usize], bid as i32);
        }
    }

    #[test]
    fn moore_stencil_merges_through_diagonal() {
        let e = vec![0.0, 9.0, 9.0, 9.0, 1.0, 9.0, 9.0, 9.0, 0.0];
        let arr = to_dyn([3, 3], e);
        assert_eq!(ws(&arr, Stencil::VonNeumann).basins.len(), 3);
        let r_m = ws(&arr, Stencil::Moore);
        assert_eq!(r_m.basins.len(), 2);
        assert_eq!(r_m.merges.len(), 1);
        assert_eq!(r_m.merges[0].energy, 1.0);
    }

    #[test]
    fn tied_seeds_merge_with_the_lower_id_as_deeper_so_basin_zero_stays_root() {
        // Two equal minima (lin 2 and lin 6). Whatever side the saddle cell
        // adopts first, the merge must record deeper = 0, shallower = 1.
        let e = vec![5.0, 4.0, 1.0, 4.0, 5.0, 4.0, 1.0, 4.0, 5.0];
        let arr = to_dyn([1, 9], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert_eq!(r.merges.len(), 1);
        assert_eq!((r.merges[0].deeper, r.merges[0].shallower), (0, 1));
        // Mirror image: the saddle cell's first neighbour is on basin 1's side.
        let e = vec![5.0, 4.0, 1.0, 4.0, 4.5, 5.0, 1.0, 4.0, 5.0];
        let arr = to_dyn([1, 9], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert!(r.merges.iter().all(|m| m.shallower != 0));
    }

    #[test]
    fn ties_resolve_by_ascending_linear_index() {
        // Two equal minima at lin 2 and lin 6 on a 1x9 chain: basin 0 must be lin 2.
        let e = vec![5.0, 4.0, 1.0, 4.0, 5.0, 4.0, 1.0, 4.0, 5.0];
        let arr = to_dyn([1, 9], e);
        let r = ws(&arr, Stencil::VonNeumann);
        assert_eq!(r.basins[0].0, 2);
        assert_eq!(r.basins[1].0, 6);
        // Flat plateau: every cell equal -> one basin seeded at lin 0.
        let flat = to_dyn([2, 3], vec![1.0; 6]);
        let r = ws(&flat, Stencil::VonNeumann);
        assert_eq!(r.basins.len(), 1);
        assert_eq!(r.basins[0].0, 0);
    }

    #[test]
    fn sorted_order_is_energy_then_index_and_skips_nan() {
        let flat = [3.0f64, f64::NAN, 1.0, 3.0, 0.5];
        assert_eq!(sorted_order(&flat), vec![4, 2, 0, 3]);
    }

    #[test]
    fn parents_chains_descend_to_seeds_with_nonincreasing_energy() {
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e.clone());
        let strides = compute_strides(arr.shape());
        let r = flood(arr.view(), FloodOptions { stencil: Stencil::VonNeumann, record_parents: true, stop_when_connected: None });
        let parents = r.parents.as_ref().unwrap();
        let seeds: Vec<u32> = r.basins.iter().map(|b| b.0).collect();
        for lin in 0..e.len() {
            let mut c = lin;
            let mut steps = 0;
            while parents[c] != PARENT_NONE {
                let next = crate::common::nd::apply_code(c, parents[c], &strides);
                assert!(e[next] <= e[c], "chain must not ascend");
                c = next;
                steps += 1;
                assert!(steps <= e.len());
            }
            assert!(seeds.contains(&(c as u32)), "chain from {lin} ends at non-seed {c}");
        }
        // Seeds carry the sentinel.
        for &s in &seeds {
            assert_eq!(parents[s as usize], PARENT_NONE);
        }
    }

    #[test]
    fn stop_when_connected_leaves_higher_cells_unlabelled() {
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e);
        // Cols 0 and 4 connect at the saddle E=5 (col 2); E=6 at col 6 is never processed.
        let r = flood(arr.view(), FloodOptions { stencil: Stencil::VonNeumann, record_parents: true, stop_when_connected: Some((0, 4)) });
        assert_eq!(r.labels[6], -1);
        assert_eq!(r.merges.len(), 1);
        assert!(r.labels[0] >= 0 && r.labels[4] >= 0);
    }

    #[test]
    fn watershed_f32_matches_f64() {
        let e64 = vec![0.0f64, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let e32: Vec<f32> = e64.iter().map(|&x| x as f32).collect();
        let a64 = Array::from_shape_vec(IxDyn(&[1, 9]), e64).unwrap();
        let a32 = Array::from_shape_vec(IxDyn(&[1, 9]), e32).unwrap();
        let r64 = flood(a64.view(), opts(Stencil::VonNeumann));
        let r32 = flood(a32.view(), opts(Stencil::VonNeumann));
        assert_eq!(r64.labels, r32.labels);
        assert_eq!(r64.basins.iter().map(|b| b.0).collect::<Vec<_>>(), r32.basins.iter().map(|b| b.0).collect::<Vec<_>>());
        for (m64, m32) in r64.merges.iter().zip(&r32.merges) {
            assert_eq!((m64.saddle, m64.other, m64.deeper, m64.shallower, m64.saddle_side), (m32.saddle, m32.other, m32.deeper, m32.shallower, m32.saddle_side));
            assert!((m64.energy - m32.energy as f64).abs() < 1e-4);
        }
    }

    #[test]
    fn dimensionality_sweep_2_to_7() {
        for ndim in 2..=7 {
            let mut shape = vec![1usize; ndim];
            shape[0] = 3;
            let total: usize = shape.iter().product();
            let mut data = vec![7.0; total];
            data[0] = 0.0;
            data[total - 1] = 0.0;
            let arr = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
            let r = flood(arr.view(), opts(Stencil::VonNeumann));
            assert_eq!(r.basins.len(), 2, "ndim={ndim}");
            assert_eq!(r.merges.len(), 1, "ndim={ndim}");
            assert_eq!(r.merges[0].energy, 7.0, "ndim={ndim}");
        }
    }
}
