//! Deep minimax minimum-energy path between two grid cells, reconstructed
//! from a flood's state: the Kruskal forest is rebuilt from the merge list
//! (leaves = basins, events = merges) and descents follow the recorded
//! flood-parent direction codes. See `ALGORITHMS.md`.

use ndarray::ArrayViewD;

use crate::common::nd::{apply_code_checked, code_space, compute_strides, index_to_linear, Stencil, PARENT_NONE};
use crate::common::scalar::Scalar;
use crate::topology::watershed::{flood, FloodOptions, FloodResult};

/// The merge fields the reconstruction needs (energies are not needed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergeRecord {
    pub saddle: u32,
    pub other: u32,
    pub deeper: u32,
    pub shallower: u32,
    pub saddle_side: u32,
}

impl<T> From<&crate::topology::watershed::MergeEvent<T>> for MergeRecord {
    fn from(m: &crate::topology::watershed::MergeEvent<T>) -> Self {
        MergeRecord { saddle: m.saddle, other: m.other, deeper: m.deeper, shallower: m.shallower, saddle_side: m.saddle_side }
    }
}

const NONE: u32 = u32::MAX;

/// Deep minimax path from `start_lin` to `end_lin` over a completed (or
/// early-stopped) flood. `Ok(None)` when the endpoints are not connected or
/// were never flooded; `Err` when the inputs are inconsistent (all inputs
/// may come from Python, so nothing here may panic on bad data).
pub fn reconstruct(
    labels: &[i32],
    parents: &[u16],
    shape: &[usize],
    n_basins: usize,
    merges: &[MergeRecord],
    start_lin: usize,
    end_lin: usize,
) -> Result<Option<Vec<usize>>, String> {
    let n_total = labels.len();
    let strides = compute_strides(shape);
    let n_codes = code_space(shape.len());
    if parents.len() != n_total {
        return Err(format!("parents has {} cells, labels has {n_total}", parents.len()));
    }
    if start_lin >= n_total || end_lin >= n_total {
        return Err("endpoint outside the grid".into());
    }
    if let Some((i, &l)) = labels.iter().enumerate().find(|&(_, &l)| l >= n_basins as i32) {
        return Err(format!("label {l} at cell {i} is out of range for {n_basins} basins"));
    }
    for (k, m) in merges.iter().enumerate() {
        let nb = n_basins as u32;
        if m.deeper >= nb || m.shallower >= nb || m.saddle_side >= nb {
            return Err(format!("merge {k} refers to a basin id >= {n_basins}"));
        }
        if m.saddle_side != m.deeper && m.saddle_side != m.shallower {
            return Err(format!("merge {k}: saddle_side is neither deeper nor shallower"));
        }
        if m.saddle as usize >= n_total || m.other as usize >= n_total {
            return Err(format!("merge {k} refers to a cell outside the grid"));
        }
    }
    if labels[start_lin] < 0 || labels[end_lin] < 0 {
        return Ok(None); // never flooded (partial flood) — checked before the trivial path
    }
    if start_lin == end_lin {
        return Ok(Some(vec![start_lin]));
    }

    // ---- Kruskal forest from the merge list -------------------------------
    // Leaves are basin ids 0..n_basins; event k is node n_basins + k, so a
    // parent's id is always greater than its children's.
    let n_nodes = n_basins + merges.len();
    let mut tree_parent: Vec<u32> = vec![NONE; n_nodes];
    let mut cur_node: Vec<u32> = (0..n_basins as u32).collect();
    let mut child_saddle: Vec<u32> = vec![NONE; merges.len()];
    for (k, m) in merges.iter().enumerate() {
        let node = (n_basins + k) as u32;
        let d = cur_node[m.deeper as usize];
        let s = cur_node[m.shallower as usize];
        if d == s || tree_parent[d as usize] != NONE || tree_parent[s as usize] != NONE {
            return Err(format!("merge {k} is inconsistent with earlier merges"));
        }
        child_saddle[k] = cur_node[m.saddle_side as usize];
        tree_parent[d as usize] = node;
        tree_parent[s as usize] = node;
        cur_node[m.deeper as usize] = node;
    }

    // ---- Chain and forest walks (all bounds-checked) -----------------------
    let step = |c: usize| -> Result<Option<usize>, String> {
        let code = parents[c];
        if code == PARENT_NONE {
            return Ok(None);
        }
        if code >= n_codes {
            return Err(format!("invalid direction code {code} at cell {c}"));
        }
        apply_code_checked(c, code, shape, &strides)
            .map(Some)
            .ok_or_else(|| format!("direction code {code} at cell {c} leaves the grid"))
    };
    let terminus = |mut c: usize| -> Result<usize, String> {
        let mut steps = 0usize;
        while let Some(next) = step(c)? {
            c = next;
            steps += 1;
            if steps > n_total {
                return Err("flood-parent chain does not terminate".into());
            }
        }
        Ok(c)
    };
    let leaf_of = |c: usize| -> Result<u32, String> {
        let t = terminus(c)?;
        if labels[t] < 0 {
            return Err(format!("flood-parent chain ends at unlabelled cell {t}"));
        }
        Ok(labels[t] as u32)
    };
    let up = |n: u32| -> Result<u32, String> {
        let p = tree_parent[n as usize];
        if p == NONE { Err(format!("forest node {n} has no parent")) } else { Ok(p) }
    };
    let forest_root = |mut n: u32| -> u32 {
        while tree_parent[n as usize] != NONE {
            n = tree_parent[n as usize];
        }
        n
    };
    // Parents are created after children, so lifting the smaller id converges on the LCA.
    let lca = |mut a: u32, mut b: u32| -> Result<u32, String> {
        while a != b {
            if a < b { a = up(a)?; } else { b = up(b)?; }
        }
        Ok(a)
    };
    let child_under = |event: u32, mut node: u32| -> Result<u32, String> {
        while tree_parent[node as usize] != event {
            node = up(node)?;
        }
        Ok(node)
    };

    let la = leaf_of(start_lin)?;
    let lb = leaf_of(end_lin)?;
    if forest_root(la) != forest_root(lb) {
        return Ok(None);
    }

    // ---- Path assembly: LIFO of (a, b) segments, output left-to-right -----
    let mut path: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(start_lin, end_lin)];
    while let Some((a, b)) = stack.pop() {
        let la = leaf_of(a)?;
        let lb = leaf_of(b)?;
        if la == lb {
            if a == b {
                path.push(a);
                continue;
            }
            // Same basin: descend a to the seed (inclusive), ascend to b —
            // the deliberate V through the basin minimum.
            let mut x = a;
            loop {
                path.push(x);
                match step(x)? {
                    Some(next) => x = next,
                    None => break,
                }
            }
            let mut chain_b: Vec<usize> = Vec::new();
            let mut y = b;
            while let Some(next) = step(y)? {
                chain_b.push(y);
                y = next;
            }
            path.extend(chain_b.iter().rev());
        } else {
            let event = lca(la, lb)?;
            let k = event as usize - n_basins;
            let m = merges[k];
            // u on a's side, v on b's side; the saddle cell belongs to the
            // child_saddle component, and (saddle, other) are stencil neighbours.
            let (u, v) = if child_under(event, la)? == child_saddle[k] {
                (m.saddle as usize, m.other as usize)
            } else {
                (m.other as usize, m.saddle as usize)
            };
            stack.push((v, b));
            stack.push((a, u));
        }
    }
    Ok(Some(path))
}

/// Standalone deep minimax path: floods until the endpoints connect, then
/// reconstructs. Preconditions as for `flood`; endpoints in bounds and non-NaN.
pub fn mep_inner<T: Scalar>(
    energies: ArrayViewD<'_, T>,
    start_idx: &[usize],
    end_idx: &[usize],
    stencil: Stencil,
) -> Option<Vec<usize>> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let start_lin = index_to_linear(start_idx, &strides);
    let end_lin = index_to_linear(end_idx, &strides);
    if start_lin == end_lin {
        return Some(vec![start_lin]);
    }
    let r: FloodResult<T> = flood(
        energies,
        FloodOptions { stencil, record_parents: true, stop_when_connected: Some((start_lin, end_lin)) },
    );
    let records: Vec<MergeRecord> = r.merges.iter().map(MergeRecord::from).collect();
    reconstruct(&r.labels, r.parents.as_deref().expect("record_parents = true"), &shape, r.basins.len(), &records, start_lin, end_lin)
        .expect("a flood's own output is self-consistent")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::nd::linear_to_index;
    use ndarray::{Array, ArrayD, IxDyn};

    fn to_dyn<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    fn path_nd(arr: &ArrayD<f64>, path: &[usize]) -> Vec<Vec<usize>> {
        let shape: Vec<usize> = arr.shape().to_vec();
        let strides = compute_strides(&shape);
        path.iter()
            .map(|&l| linear_to_index(l, &shape, &strides))
            .collect()
    }

    #[test]
    fn start_equals_end_returns_single_cell() {
        let arr = to_dyn([3, 3], vec![0.0; 9]);
        let p = mep_inner(arr.view(), &[1, 1], &[1, 1], Stencil::VonNeumann).unwrap();
        assert_eq!(p, vec![4]);
    }

    #[test]
    fn chain_path_visits_every_cell_in_order() {
        // 1x9 chain with three basins (cols 0, 4, 8) and saddles at
        // cols 2 (E=5) and 6 (E=6); the deep path is the entire row.
        let e = vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0];
        let arr = to_dyn([1, 9], e.clone());
        let p = mep_inner(arr.view(), &[0, 0], &[0, 8], Stencil::VonNeumann).unwrap();
        assert_eq!(p, (0..9).collect::<Vec<usize>>());
        // Profile dips to the intermediate basin floor (E=1 at col 4).
        let max = p.iter().map(|&l| e[l]).fold(f64::MIN, f64::max);
        assert_eq!(max, 6.0);
    }

    #[test]
    fn path_crosses_ridge_at_iwf_saddle() {
        // Same grid as iwf_grid::two_minima_separated_by_ridge_2d.
        let mut e = vec![10.0; 25];
        e[0] = 0.0;
        e[1] = 1.0;
        e[5] = 1.0;
        e[6] = 2.0;
        e[10] = 3.0;
        e[11] = 3.0;
        e[12] = 4.0; // saddle (2,2)
        e[13] = 3.0;
        e[14] = 3.0;
        e[4 * 5 + 4] = 0.0;
        e[3 * 5 + 4] = 1.0;
        e[4 * 5 + 3] = 1.0;
        e[3 * 5 + 3] = 2.0;
        let arr = to_dyn([5, 5], e.clone());
        let p = mep_inner(arr.view(), &[0, 0], &[4, 4], Stencil::VonNeumann).unwrap();
        let nd = path_nd(&arr, &p);
        assert_eq!(nd.first().unwrap(), &vec![0, 0]);
        assert_eq!(nd.last().unwrap(), &vec![4, 4]);
        assert!(nd.contains(&vec![2, 2]), "path must cross the saddle cell");
        let max = p.iter().map(|&l| e[l]).fold(f64::MIN, f64::max);
        assert_eq!(max, 4.0, "highest path energy is the IWF saddle");
        // Von Neumann steps: exactly one axis changes by 1.
        for w in nd.windows(2) {
            let d: usize = w[0]
                .iter()
                .zip(w[1].iter())
                .map(|(&x, &y)| x.abs_diff(y))
                .sum();
            assert_eq!(d, 1, "step {:?} -> {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn moore_path_uses_diagonal_channel() {
        let e = vec![0.0, 9.0, 9.0, 9.0, 1.0, 9.0, 9.0, 9.0, 0.0];
        let arr = to_dyn([3, 3], e.clone());
        let p_vn = mep_inner(arr.view(), &[0, 0], &[2, 2], Stencil::VonNeumann).unwrap();
        let max_vn = p_vn.iter().map(|&l| e[l]).fold(f64::MIN, f64::max);
        assert_eq!(max_vn, 9.0);
        let p_m = mep_inner(arr.view(), &[0, 0], &[2, 2], Stencil::Moore).unwrap();
        assert_eq!(p_m, vec![0, 4, 8]); // corner → centre → corner
        // Moore steps: Chebyshev distance 1.
        let nd = path_nd(&arr, &p_m);
        for w in nd.windows(2) {
            let cheb = w[0]
                .iter()
                .zip(w[1].iter())
                .map(|(&x, &y)| x.abs_diff(y))
                .max()
                .unwrap();
            assert_eq!(cheb, 1);
        }
    }

    #[test]
    fn nan_wall_returns_none() {
        let nan = f64::NAN;
        let e = vec![
            0.0, 1.0, nan, 1.0, 0.0, 1.0, 2.0, nan, 2.0, 1.0, 0.0, 1.0, nan, 1.0, 0.0,
        ];
        let arr = to_dyn([3, 5], e);
        assert!(mep_inner(arr.view(), &[1, 0], &[1, 4], Stencil::VonNeumann).is_none());
    }

    #[test]
    fn mep_f32_matches_f64() {
        // Reuse the path_crosses_ridge_at_iwf_saddle fixture, as f64 and f32.
        let mut e64 = vec![10.0f64; 25];
        e64[0] = 0.0;
        e64[1] = 1.0;
        e64[5] = 1.0;
        e64[6] = 2.0;
        e64[10] = 3.0;
        e64[11] = 3.0;
        e64[12] = 4.0;
        e64[13] = 3.0;
        e64[14] = 3.0;
        e64[4 * 5 + 4] = 0.0;
        e64[3 * 5 + 4] = 1.0;
        e64[4 * 5 + 3] = 1.0;
        e64[3 * 5 + 3] = 2.0;
        let e32: Vec<f32> = e64.iter().map(|&x| x as f32).collect();
        let a64 = Array::from_shape_vec(IxDyn(&[5, 5]), e64).unwrap();
        let a32 = Array::from_shape_vec(IxDyn(&[5, 5]), e32).unwrap();

        let r64 = mep_inner(a64.view(), &[0, 0], &[4, 4], Stencil::VonNeumann);
        let r32 = mep_inner(a32.view(), &[0, 0], &[4, 4], Stencil::VonNeumann);
        assert_eq!(r64, r32);
    }

    #[test]
    fn dimensionality_sweep_2_to_7() {
        // Shape (3, 1, ..., 1): minima at both ends, saddle E=7 between.
        for ndim in 2..=7 {
            let mut shape = vec![1usize; ndim];
            shape[0] = 3;
            let total: usize = shape.iter().product();
            let mut data = vec![7.0; total];
            data[0] = 0.0;
            data[total - 1] = 0.0;
            let arr = Array::from_shape_vec(IxDyn(&shape), data.clone()).unwrap();
            let start = vec![0usize; ndim];
            let mut end = vec![0usize; ndim];
            end[0] = 2;
            let p = mep_inner(arr.view(), &start, &end, Stencil::VonNeumann).unwrap();
            assert_eq!(p.len(), 3, "ndim={ndim}");
            let max = p.iter().map(|&l| data[l]).fold(f64::MIN, f64::max);
            assert_eq!(max, 7.0, "ndim={ndim}");
        }
    }

    fn records<T>(r: &crate::topology::watershed::FloodResult<T>) -> Vec<MergeRecord> {
        r.merges
            .iter()
            .map(|m| MergeRecord { saddle: m.saddle, other: m.other, deeper: m.deeper, shallower: m.shallower, saddle_side: m.saddle_side })
            .collect()
    }

    fn via_tree(arr: &ArrayD<f64>, start: &[usize], end: &[usize], stencil: Stencil) -> Result<Option<Vec<usize>>, String> {
        use crate::topology::watershed::{flood, FloodOptions};
        let r = flood(arr.view(), FloodOptions { stencil, record_parents: true, stop_when_connected: None });
        let strides = compute_strides(arr.shape());
        reconstruct(&r.labels, r.parents.as_deref().unwrap(), arr.shape(), r.basins.len(), &records(&r),
                    index_to_linear(start, &strides), index_to_linear(end, &strides))
    }

    #[test]
    fn reconstruct_from_full_flood_matches_standalone_on_fixtures() {
        let chain = to_dyn([1, 9], vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]);
        let mut ridge = vec![10.0; 25];
        for (i, v) in [(0, 0.0), (1, 1.0), (5, 1.0), (6, 2.0), (10, 3.0), (11, 3.0), (12, 4.0), (13, 3.0), (14, 3.0), (24, 0.0), (19, 1.0), (23, 1.0), (18, 2.0)] {
            ridge[i] = v;
        }
        let ridge = to_dyn([5, 5], ridge);
        let diag = to_dyn([3, 3], vec![0.0, 9.0, 9.0, 9.0, 1.0, 9.0, 9.0, 9.0, 0.0]);
        let cases: Vec<(&ArrayD<f64>, Vec<usize>, Vec<usize>, Stencil)> = vec![
            (&chain, vec![0, 0], vec![0, 8], Stencil::VonNeumann),
            (&chain, vec![0, 8], vec![0, 0], Stencil::VonNeumann),
            (&chain, vec![0, 1], vec![0, 3], Stencil::VonNeumann),   // same basin: V through the seed
            (&ridge, vec![0, 0], vec![4, 4], Stencil::VonNeumann),
            (&diag, vec![0, 0], vec![2, 2], Stencil::VonNeumann),
            (&diag, vec![0, 0], vec![2, 2], Stencil::Moore),
        ];
        for (arr, s, e, st) in cases {
            let standalone = mep_inner(arr.view(), &s, &e, st);
            let tree = via_tree(arr, &s, &e, st).unwrap();
            assert_eq!(standalone, tree, "start={s:?} end={e:?} {st:?}");
        }
    }

    #[test]
    fn reconstruct_random_3d_matches_standalone() {
        // Deterministic LCG "random" grid; compare several endpoint pairs.
        let shape = [6usize, 7, 5];
        let total: usize = shape.iter().product();
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let data: Vec<f64> = (0..total).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((x >> 11) as f64) / ((1u64 << 53) as f64) }).collect();
        let arr = to_dyn(shape, data);
        for st in [Stencil::VonNeumann, Stencil::Moore] {
            for (s, e) in [([0, 0, 0], [5, 6, 4]), ([2, 3, 1], [3, 3, 1]), ([5, 0, 4], [0, 6, 0])] {
                assert_eq!(mep_inner(arr.view(), &s, &e, st), via_tree(&arr, &s, &e, st).unwrap(), "{s:?}->{e:?} {st:?}");
            }
        }
    }

    #[test]
    fn reconstruct_returns_none_across_nan_wall_and_for_unflooded_endpoint() {
        let nan = f64::NAN;
        let arr = to_dyn([3, 5], vec![0.0, 1.0, nan, 1.0, 0.0, 1.0, 2.0, nan, 2.0, 1.0, 0.0, 1.0, nan, 1.0, 0.0]);
        assert_eq!(via_tree(&arr, &[1, 0], &[1, 4], Stencil::VonNeumann), Ok(None));
        // Partial flood (stopped early): an endpoint above the stop energy is unlabelled -> None.
        use crate::topology::watershed::{flood, FloodOptions};
        let chain = to_dyn([1, 9], vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]);
        let r = flood(chain.view(), FloodOptions { stencil: Stencil::VonNeumann, record_parents: true, stop_when_connected: Some((0, 4)) });
        let out = reconstruct(&r.labels, r.parents.as_deref().unwrap(), chain.shape(), r.basins.len(), &records(&r), 0, 6);
        assert_eq!(out, Ok(None));
    }

    #[test]
    fn reconstruct_rejects_corrupt_inputs_without_panicking() {
        use crate::topology::watershed::{flood, FloodOptions};
        let chain = to_dyn([1, 9], vec![0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]);
        let r = flood(chain.view(), FloodOptions { stencil: Stencil::VonNeumann, record_parents: true, stop_when_connected: None });
        let parents = r.parents.as_deref().unwrap();
        let recs = records(&r);
        // label out of range
        let mut bad_labels = r.labels.clone();
        bad_labels[3] = 99;
        assert!(reconstruct(&bad_labels, parents, chain.shape(), r.basins.len(), &recs, 0, 8).is_err());
        // merge basin id out of range
        let mut bad_recs = recs.clone();
        bad_recs[0].deeper = 42;
        assert!(reconstruct(&r.labels, parents, chain.shape(), r.basins.len(), &bad_recs, 0, 8).is_err());
        // invalid direction code
        let mut bad_parents = parents.to_vec();
        bad_parents[1] = 500;
        assert!(reconstruct(&r.labels, &bad_parents, chain.shape(), r.basins.len(), &recs, 0, 8).is_err());
        // code stepping off the grid: Δ=(−1,0) at row 0 is digit 0 at axis 0 → code 1
        bad_parents[1] = 1;
        assert!(reconstruct(&r.labels, &bad_parents, chain.shape(), r.basins.len(), &recs, 0, 8).is_err());
        // parents length mismatch
        assert!(reconstruct(&r.labels, &parents[..5], chain.shape(), r.basins.len(), &recs, 0, 8).is_err());
        // cycle in the chain: cell 1 -> cell 0 (code 3 = Δ(0,−1)) and cell 0 -> cell 1 (code 5 = Δ(0,+1))
        let mut cyc = parents.to_vec();
        cyc[0] = 5;
        cyc[1] = 3;
        assert!(reconstruct(&r.labels, &cyc, chain.shape(), r.basins.len(), &recs, 1, 8).is_err());
    }
}
