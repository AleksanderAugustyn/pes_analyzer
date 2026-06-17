//! Deep minimax minimum-energy path between two grid cells.
//!
//! The path minimizes the highest energy crossed (it passes through the
//! exact IWF saddles) and between saddles descends to the actual basin
//! minima via flood-parent chains. See `ALGORITHMS.md`.

use ndarray::ArrayViewD;

use crate::common::dsu::DisjointSetUnion;
use crate::common::nd::{compute_strides, index_to_linear, Stencil};
use crate::common::scalar::Scalar;

/// Kruskal-forest node payload. Leaves are basins; events carry the saddle
/// cell, the already-flooded neighbour on the other component, and which
/// child subtree holds the saddle-side component.
#[derive(Clone, Copy)]
enum NodeKind {
    Leaf,
    Event {
        saddle: u32,       // compact id of the saddle cell
        other: u32,        // compact id of the other-side neighbour
        child_saddle: u32, // forest node id of the saddle-side component
    },
}

/// Deep minimax path from `start_idx` to `end_idx`, as a sequence of flat
/// linear indices (first = start, last = end). `None` if the endpoints
/// never connect within the non-NaN region.
///
/// Preconditions (enforced by the PyO3 wrapper): C-contiguous view of
/// element type `T`, ndim in [2, 7], endpoints in bounds and non-NaN,
/// len <= u32::MAX.
pub fn mep_inner<T: Scalar>(
    energies: ArrayViewD<'_, T>,
    start_idx: &[usize],
    end_idx: &[usize],
    stencil: Stencil,
) -> Option<Vec<usize>> {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let n_total = energies.len();

    let flat: &[T] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    let start_lin = index_to_linear(start_idx, &strides);
    let end_lin = index_to_linear(end_idx, &strides);
    if start_lin == end_lin {
        return Some(vec![start_lin]);
    }

    // Compact remap + ascending-energy schedule, as in watershed.rs.
    let mut remap: Vec<u32> = vec![u32::MAX; n_total];
    let mut lin_of: Vec<u32> = Vec::new();
    let mut sorted: Vec<(u32, T)> = Vec::with_capacity(n_total);
    for i in 0..n_total {
        let e = flat[i];
        if !e.is_nan() {
            remap[i] = sorted.len() as u32;
            lin_of.push(i as u32);
            sorted.push((i as u32, e));
        }
    }
    sorted.sort_unstable_by(|a, b| a.1.tcmp(&b.1));

    let start_compact = remap[start_lin];
    let end_compact = remap[end_lin];
    debug_assert_ne!(start_compact, u32::MAX);
    debug_assert_ne!(end_compact, u32::MAX);

    let n_valid = sorted.len();
    let mut dsu = DisjointSetUnion::new(n_valid);
    let mut processed = vec![false; n_valid];
    // Flood parent: first already-flooded neighbour each cell unions into.
    // Chains descend monotonically to the basin seed (u32::MAX there).
    let mut parent_cell: Vec<u32> = vec![u32::MAX; n_valid];
    // Kruskal forest. Node ids are allocated in creation order, so a
    // parent's id is always greater than its children's.
    let mut tree_parent: Vec<u32> = Vec::new();
    let mut node_kind: Vec<NodeKind> = Vec::new();
    // Leaf node id owned by each basin seed (read via descent terminus).
    let mut leaf_of_seed: Vec<u32> = vec![u32::MAX; n_valid];
    // Forest node currently owning each DSU root (same maintenance
    // pattern as basin_of_root in watershed.rs).
    let mut node_of_root: Vec<u32> = vec![u32::MAX; n_valid];

    let mut nbrs: Vec<usize> = Vec::new();
    let mut connected = false;

    for &(lin_u32, _energy) in &sorted {
        let lin = lin_u32 as usize;
        let cur = remap[lin];
        processed[cur as usize] = true;

        stencil.neighbors(lin, &shape, &strides, &mut nbrs);

        let mut my_node: Option<u32> = None;
        for &nbr_lin in &nbrs {
            let nbr = remap[nbr_lin];
            if nbr == u32::MAX || !processed[nbr as usize] {
                continue;
            }
            let nbr_root = dsu.find(nbr);
            let nbr_node = node_of_root[nbr_root as usize];
            match my_node {
                None => {
                    parent_cell[cur as usize] = nbr;
                    dsu.union(cur, nbr_root);
                    let r = dsu.find(cur);
                    node_of_root[r as usize] = nbr_node;
                    my_node = Some(nbr_node);
                }
                Some(cur_node) if cur_node == nbr_node => {
                    dsu.union(cur, nbr_root);
                    let r = dsu.find(cur);
                    node_of_root[r as usize] = cur_node;
                }
                Some(cur_node) => {
                    // Two components meet at this cell: merge event.
                    let new_node = tree_parent.len() as u32;
                    tree_parent.push(u32::MAX);
                    node_kind.push(NodeKind::Event {
                        saddle: cur,
                        other: nbr,
                        child_saddle: cur_node,
                    });
                    tree_parent[cur_node as usize] = new_node;
                    tree_parent[nbr_node as usize] = new_node;
                    dsu.union(cur, nbr_root);
                    let r = dsu.find(cur);
                    node_of_root[r as usize] = new_node;
                    my_node = Some(new_node);
                }
            }
        }

        if my_node.is_none() {
            // No processed neighbours: this cell seeds a new basin. It is
            // its own DSU root (never unioned yet).
            let leaf = tree_parent.len() as u32;
            tree_parent.push(u32::MAX);
            node_kind.push(NodeKind::Leaf);
            leaf_of_seed[cur as usize] = leaf;
            node_of_root[cur as usize] = leaf;
        }

        if dsu.find(start_compact) == dsu.find(end_compact) {
            connected = true;
            break;
        }
    }

    if !connected {
        return None;
    }

    // ---- Path reconstruction (iterative; no recursion depth limit) ----

    let leaf_of = |mut c: u32| -> u32 {
        while parent_cell[c as usize] != u32::MAX {
            c = parent_cell[c as usize];
        }
        leaf_of_seed[c as usize]
    };

    // Parents are created after children, so lifting the smaller id
    // converges on the lowest common ancestor.
    let lca = |mut a: u32, mut b: u32| -> u32 {
        while a != b {
            if a < b {
                a = tree_parent[a as usize];
            } else {
                b = tree_parent[b as usize];
            }
        }
        a
    };

    // The child of `event` whose subtree contains `node`.
    let child_under = |event: u32, mut node: u32| -> u32 {
        while tree_parent[node as usize] != event {
            node = tree_parent[node as usize];
        }
        node
    };

    let mut path: Vec<u32> = Vec::new();
    // Segments to connect, LIFO. Each (a, b) expands to a path from a to b
    // inclusive; pushed in reverse order so the output is left-to-right.
    let mut stack: Vec<(u32, u32)> = vec![(start_compact, end_compact)];

    while let Some((a, b)) = stack.pop() {
        let la = leaf_of(a);
        let lb = leaf_of(b);
        if la == lb {
            if a == b {
                path.push(a);
                continue;
            }
            // Same basin: descend a to the seed (inclusive), ascend to b.
            // This V through the basin minimum is the deliberate "deep"
            // detour; shared chain suffixes are walked down and back up.
            let mut x = a;
            loop {
                path.push(x);
                let p = parent_cell[x as usize];
                if p == u32::MAX {
                    break;
                }
                x = p;
            }
            let mut chain_b: Vec<u32> = Vec::new();
            let mut y = b;
            while parent_cell[y as usize] != u32::MAX {
                chain_b.push(y);
                y = parent_cell[y as usize];
            }
            path.extend(chain_b.iter().rev());
        } else {
            let event = lca(la, lb);
            let NodeKind::Event {
                saddle,
                other,
                child_saddle,
            } = node_kind[event as usize]
            else {
                unreachable!("LCA of two distinct leaves is a merge event");
            };
            // Orient the crossing: u on a's side, v on b's side. The
            // saddle cell belongs to the `child_saddle` component; u and v
            // are stencil neighbours, so the u→v step is a valid move.
            let (u, v) = if child_under(event, la) == child_saddle {
                (saddle, other)
            } else {
                (other, saddle)
            };
            stack.push((v, b));
            stack.push((a, u));
        }
    }

    Some(path.into_iter().map(|c| lin_of[c as usize] as usize).collect())
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
}
