//! N-dimensional indexing utilities for C-contiguous (row-major) arrays.

/// Compute C-contiguous strides for `shape`. The last axis has stride 1.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![0usize; n];
    if n == 0 {
        return strides;
    }
    strides[n - 1] = 1;
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a multi-index to a flat linear index.
pub fn index_to_linear(idx: &[usize], strides: &[usize]) -> usize {
    debug_assert_eq!(idx.len(), strides.len());
    idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Convert a flat linear index to a multi-index.
pub fn linear_to_index(mut linear: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    debug_assert_eq!(shape.len(), strides.len());
    let n = shape.len();
    let mut idx = vec![0usize; n];
    for i in 0..n {
        idx[i] = linear / strides[i];
        linear %= strides[i];
    }
    idx
}

/// Enumerate the 2N axis-aligned neighbors of the cell at `linear` and push
/// their linear indices into `out`. The buffer `out` is cleared first.
///
/// Boundary cells produce fewer than 2N neighbors.
pub fn axis_neighbors(linear: usize, shape: &[usize], strides: &[usize], out: &mut Vec<usize>) {
    out.clear();
    let ndim = shape.len();
    let mut remaining = linear;
    for axis in 0..ndim {
        let stride = strides[axis];
        let coord = remaining / stride;
        remaining %= stride;
        if coord > 0 {
            out.push(linear - stride);
        }
        if coord + 1 < shape[axis] {
            out.push(linear + stride);
        }
    }
}

// ndim is bounded at 7 by the public API (enforced in validate.rs::check_ndim).
const MAX_NDIM: usize = 7;

/// Enumerate the up-to-(2r+1)ᴺ−1 in-bounds neighbors of the cell at `linear`
/// inside the Chebyshev box of half-width `r` (king-move stencil at `r = 1`)
/// and push their linear indices into `out`. The buffer `out` is cleared first.
///
/// Walks `[-r, r]ᴺ` minus the all-zero offset. Boundary cells produce
/// fewer than (2r+1)ᴺ−1 neighbors. `r` must be ≥ 1; the PyO3 boundary
/// enforces this.
pub fn full_neighbors(
    linear: usize,
    shape: &[usize],
    strides: &[usize],
    r: usize,
    out: &mut Vec<usize>,
) {
    out.clear();
    let ndim = shape.len();
    let r_i = r as i64;

    // Decompose `linear` into per-axis coordinates once.
    // Fixed-size stack array avoids heap allocation; only [0..ndim] is used.
    let mut coords: [i64; MAX_NDIM] = [0; MAX_NDIM];
    let mut remaining = linear;
    for axis in 0..ndim {
        coords[axis] = (remaining / strides[axis]) as i64;
        remaining %= strides[axis];
    }

    // Walk all offsets in [-r, r]^N except the all-zero offset (self).
    // Fixed-size stack array avoids heap allocation; only [0..ndim] is used.
    let mut offset: [i64; MAX_NDIM] = [-r_i; MAX_NDIM];
    loop {
        // Skip the all-zero offset (self). Short-circuit on the first non-zero
        // entry (the common case), inspecting only the active [0..ndim] slice.
        let not_self = offset[..ndim].iter().any(|&o| o != 0);
        if not_self {
            // Build neighbor linear index, checking bounds per axis.
            let mut nbr_lin = 0usize;
            let mut in_bounds = true;
            for axis in 0..ndim {
                let nc = coords[axis] + offset[axis];
                if nc < 0 || nc >= shape[axis] as i64 {
                    in_bounds = false;
                    break;
                }
                nbr_lin += (nc as usize) * strides[axis];
            }
            if in_bounds {
                out.push(nbr_lin);
            }
        }

        // Increment the offset vector odometer-style: -r → ... → r → carry.
        let mut axis = ndim;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            if offset[axis] < r_i {
                offset[axis] += 1;
                break;
            }
            offset[axis] = -r_i;
        }
    }
}

/// Neighbor stencil selector shared by the flood-based kernels
/// (`find_iwf_grid`, `find_watershed_segmentation`, `find_minimum_energy_path`).
/// Range is fixed at 1 for both variants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stencil {
    /// 2N axis-aligned neighbors (von Neumann).
    VonNeumann,
    /// 3^N − 1 Chebyshev-box neighbors (Moore).
    Moore,
}

impl Stencil {
    /// Enumerate the in-bounds neighbors of `linear` into `out` (cleared first).
    pub fn neighbors(
        self,
        linear: usize,
        shape: &[usize],
        strides: &[usize],
        out: &mut Vec<usize>,
    ) {
        match self {
            Stencil::VonNeumann => axis_neighbors(linear, shape, strides, out),
            Stencil::Moore => full_neighbors(linear, shape, strides, 1, out),
        }
    }

    /// `neighbors` plus the direction code (see `apply_code`) of each neighbour.
    pub fn neighbors_with_codes(
        self,
        linear: usize,
        shape: &[usize],
        strides: &[usize],
        out: &mut Vec<(usize, u16)>,
    ) {
        match self {
            Stencil::VonNeumann => axis_neighbors_with_codes(linear, shape, strides, out),
            Stencil::Moore => box_neighbors_with_codes(linear, shape, strides, out),
        }
    }
}

/// Sentinel direction code: seed cell, NaN cell, or (partial floods) a cell
/// never flooded.
pub const PARENT_NONE: u16 = u16::MAX;

/// Number of direction codes for `ndim` axes: 3^ndim (2187 for ndim = 7).
pub fn code_space(ndim: usize) -> u16 {
    3u16.pow(ndim as u32)
}

/// Direction code of the all-zero offset: every digit is 1.
fn center_code(ndim: usize) -> u16 {
    (code_space(ndim) - 1) / 2
}

/// Decode `code` (odometer index of Δ ∈ {−1,0,1}ᴺ, last axis fastest:
/// `Σ (Δ_axis + 1)·3^(ndim−1−axis)`) and step from `linear`. Unchecked:
/// the caller guarantees the target is in bounds (codes are only ever
/// recorded for real neighbours).
pub fn apply_code(linear: usize, code: u16, strides: &[usize]) -> usize {
    let mut rem = code as usize;
    let mut out = linear as i64;
    for axis in (0..strides.len()).rev() {
        let delta = (rem % 3) as i64 - 1;
        rem /= 3;
        out += delta * strides[axis] as i64;
    }
    out as usize
}

/// Bounds-checked variant of `apply_code`: `None` if the step leaves the grid
/// on any axis. Used when walking codes supplied from Python.
pub fn apply_code_checked(
    linear: usize,
    code: u16,
    shape: &[usize],
    strides: &[usize],
) -> Option<usize> {
    let ndim = shape.len();
    let mut digits: [i64; MAX_NDIM] = [0; MAX_NDIM];
    let mut rem = code as usize;
    for axis in (0..ndim).rev() {
        digits[axis] = (rem % 3) as i64 - 1;
        rem /= 3;
    }
    let mut remaining = linear;
    let mut out = 0usize;
    for axis in 0..ndim {
        let coord = (remaining / strides[axis]) as i64 + digits[axis];
        remaining %= strides[axis];
        if coord < 0 || coord >= shape[axis] as i64 {
            return None;
        }
        out += coord as usize * strides[axis];
    }
    Some(out)
}

/// `axis_neighbors` with the direction code of each neighbour.
pub fn axis_neighbors_with_codes(
    linear: usize,
    shape: &[usize],
    strides: &[usize],
    out: &mut Vec<(usize, u16)>,
) {
    out.clear();
    let ndim = shape.len();
    let center = center_code(ndim);
    let mut remaining = linear;
    for axis in 0..ndim {
        let stride = strides[axis];
        let coord = remaining / stride;
        remaining %= stride;
        let digit_weight = 3u16.pow((ndim - 1 - axis) as u32);
        if coord > 0 {
            out.push((linear - stride, center - digit_weight));
        }
        if coord + 1 < shape[axis] {
            out.push((linear + stride, center + digit_weight));
        }
    }
}

/// `full_neighbors` at r = 1 with the direction code of each neighbour. The
/// odometer index IS the code, so the enumeration order matches
/// `full_neighbors` exactly.
pub fn box_neighbors_with_codes(
    linear: usize,
    shape: &[usize],
    strides: &[usize],
    out: &mut Vec<(usize, u16)>,
) {
    out.clear();
    let ndim = shape.len();
    let mut coords: [i64; MAX_NDIM] = [0; MAX_NDIM];
    let mut remaining = linear;
    for axis in 0..ndim {
        coords[axis] = (remaining / strides[axis]) as i64;
        remaining %= strides[axis];
    }
    let mut offset: [i64; MAX_NDIM] = [-1; MAX_NDIM];
    let mut code: u16 = 0;
    loop {
        if offset[..ndim].iter().any(|&o| o != 0) {
            let mut nbr_lin = 0usize;
            let mut in_bounds = true;
            for axis in 0..ndim {
                let nc = coords[axis] + offset[axis];
                if nc < 0 || nc >= shape[axis] as i64 {
                    in_bounds = false;
                    break;
                }
                nbr_lin += (nc as usize) * strides[axis];
            }
            if in_bounds {
                out.push((nbr_lin, code));
            }
        }
        code += 1;
        let mut axis = ndim;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            if offset[axis] < 1 {
                offset[axis] += 1;
                break;
            }
            offset[axis] = -1;
        }
    }
}

/// Visit the in-bounds neighbours of `linear` inside the Chebyshev box of
/// half-width `r`, in exactly the order `full_neighbors` would list them,
/// without materialising the list. `visit` returns `false` to stop early.
/// Returns `false` iff the walk was stopped early.
pub fn walk_box_neighbors(
    linear: usize,
    shape: &[usize],
    strides: &[usize],
    r: usize,
    mut visit: impl FnMut(usize) -> bool,
) -> bool {
    let ndim = shape.len();
    let r_i = r as i64;
    let mut coords: [i64; MAX_NDIM] = [0; MAX_NDIM];
    let mut remaining = linear;
    for axis in 0..ndim {
        coords[axis] = (remaining / strides[axis]) as i64;
        remaining %= strides[axis];
    }
    let mut offset: [i64; MAX_NDIM] = [-r_i; MAX_NDIM];
    loop {
        if offset[..ndim].iter().any(|&o| o != 0) {
            let mut nbr_lin = 0usize;
            let mut in_bounds = true;
            for axis in 0..ndim {
                let nc = coords[axis] + offset[axis];
                if nc < 0 || nc >= shape[axis] as i64 {
                    in_bounds = false;
                    break;
                }
                nbr_lin += (nc as usize) * strides[axis];
            }
            if in_bounds && !visit(nbr_lin) {
                return false;
            }
        }
        let mut axis = ndim;
        loop {
            if axis == 0 {
                return true;
            }
            axis -= 1;
            if offset[axis] < r_i {
                offset[axis] += 1;
                break;
            }
            offset[axis] = -r_i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_c_contiguous() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(
            compute_strides(&[2, 2, 2, 2, 2, 2, 2]),
            vec![64, 32, 16, 8, 4, 2, 1]
        );
    }

    #[test]
    fn linear_index_round_trip_2d() {
        let shape = [3, 4];
        let strides = compute_strides(&shape);
        for i in 0..3 {
            for j in 0..4 {
                let lin = index_to_linear(&[i, j], &strides);
                assert_eq!(linear_to_index(lin, &shape, &strides), vec![i, j]);
            }
        }
    }

    #[test]
    fn linear_index_round_trip_7d() {
        let shape = [2, 3, 2, 3, 2, 3, 2];
        let strides = compute_strides(&shape);
        let total: usize = shape.iter().product();
        for lin in 0..total {
            let idx = linear_to_index(lin, &shape, &strides);
            assert_eq!(index_to_linear(&idx, &strides), lin);
        }
    }

    #[test]
    fn neighbors_interior_2d_returns_four() {
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        // Interior cell (2, 2) → linear 12
        axis_neighbors(12, &shape, &strides, &mut buf);
        let mut got: Vec<usize> = buf.to_vec();
        got.sort();
        // Neighbors: (1,2)=7, (3,2)=17, (2,1)=11, (2,3)=13
        let mut expected = vec![7, 11, 13, 17];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn neighbors_corner_2d_returns_two() {
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        // Corner (0, 0) → linear 0
        axis_neighbors(0, &shape, &strides, &mut buf);
        buf.sort();
        assert_eq!(buf, vec![1, 5]);
    }

    #[test]
    fn neighbors_count_at_5d_interior_is_ten() {
        let shape = [4, 4, 4, 4, 4];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        // Interior cell (2,2,2,2,2)
        let lin = index_to_linear(&[2, 2, 2, 2, 2], &strides);
        axis_neighbors(lin, &shape, &strides, &mut buf);
        assert_eq!(buf.len(), 10); // 2 * 5
    }

    #[test]
    fn neighbors_count_at_7d_interior_is_fourteen() {
        let shape = [3, 3, 3, 3, 3, 3, 3];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[1, 1, 1, 1, 1, 1, 1], &strides);
        axis_neighbors(lin, &shape, &strides, &mut buf);
        assert_eq!(buf.len(), 14); // 2 * 7
    }

    #[test]
    fn axis_neighbors_clears_existing_buffer() {
        let shape = [3, 3];
        let strides = compute_strides(&shape);
        let mut buf = vec![99, 99, 99];
        axis_neighbors(0, &shape, &strides, &mut buf);
        // Should have been cleared then filled with corner neighbors only.
        buf.sort();
        assert_eq!(buf, vec![1, 3]);
    }

    #[test]
    fn full_neighbors_interior_2d_returns_eight() {
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        // Interior cell (2, 2) → linear 12
        full_neighbors(12, &shape, &strides, 1, &mut buf);
        assert_eq!(buf.len(), 8); // 3^2 - 1
    }

    #[test]
    fn full_neighbors_corner_2d_returns_three() {
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        // Corner (0, 0) → linear 0; neighbors are (0,1)=1, (1,0)=5, (1,1)=6
        full_neighbors(0, &shape, &strides, 1, &mut buf);
        buf.sort();
        assert_eq!(buf, vec![1, 5, 6]);
    }

    #[test]
    fn full_neighbors_clears_existing_buffer() {
        let shape = [3, 3];
        let strides = compute_strides(&shape);
        let mut buf = vec![99, 99, 99];
        full_neighbors(0, &shape, &strides, 1, &mut buf);
        buf.sort();
        // Corner (0,0): (0,1)=1, (1,0)=3, (1,1)=4
        assert_eq!(buf, vec![1, 3, 4]);
    }

    #[test]
    fn full_neighbors_count_at_5d_interior_is_242() {
        let shape = [4, 4, 4, 4, 4];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[2, 2, 2, 2, 2], &strides);
        full_neighbors(lin, &shape, &strides, 1, &mut buf);
        assert_eq!(buf.len(), 3usize.pow(5) - 1); // 242
    }

    #[test]
    fn full_neighbors_count_at_7d_interior_is_2186() {
        let shape = [3, 3, 3, 3, 3, 3, 3];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[1, 1, 1, 1, 1, 1, 1], &strides);
        full_neighbors(lin, &shape, &strides, 1, &mut buf);
        assert_eq!(buf.len(), 3usize.pow(7) - 1); // 2186
    }

    #[test]
    fn full_neighbors_excludes_self() {
        // The function must not include the cell itself among its neighbors.
        let shape = [3, 3, 3];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[1, 1, 1], &strides);
        full_neighbors(lin, &shape, &strides, 1, &mut buf);
        assert!(!buf.contains(&lin));
        assert_eq!(buf.len(), 26); // 3^3 - 1
    }

    #[test]
    fn full_neighbors_r2_interior_2d_returns_24() {
        // 7x7 grid, interior cell (3, 3); r=2 stencil has 5*5 - 1 = 24 neighbors.
        let shape = [7, 7];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[3, 3], &strides);
        full_neighbors(lin, &shape, &strides, 2, &mut buf);
        assert_eq!(buf.len(), 24);
    }

    #[test]
    fn full_neighbors_r2_interior_3d_returns_124() {
        // 7x7x7 grid, interior cell (3, 3, 3); r=2 stencil has 5^3 - 1 = 124.
        let shape = [7, 7, 7];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[3, 3, 3], &strides);
        full_neighbors(lin, &shape, &strides, 2, &mut buf);
        assert_eq!(buf.len(), 124);
    }

    #[test]
    fn full_neighbors_r2_corner_clips_to_eight() {
        // Corner (0, 0) in a 5x5 grid at r=2: in-bounds cells are
        // [0..3, 0..3] minus self = 3*3 - 1 = 8.
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        full_neighbors(0, &shape, &strides, 2, &mut buf);
        assert_eq!(buf.len(), 8);
    }

    #[test]
    fn stencil_von_neumann_matches_axis_neighbors() {
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        Stencil::VonNeumann.neighbors(12, &shape, &strides, &mut a);
        axis_neighbors(12, &shape, &strides, &mut b);
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn stencil_moore_matches_full_neighbors_r1() {
        let shape = [5, 5];
        let strides = compute_strides(&shape);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        Stencil::Moore.neighbors(12, &shape, &strides, &mut a);
        full_neighbors(12, &shape, &strides, 1, &mut b);
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn full_neighbors_r2_excludes_self() {
        // Even with the offset domain including 0, self must not appear.
        let shape = [5, 5, 5];
        let strides = compute_strides(&shape);
        let mut buf = Vec::new();
        let lin = index_to_linear(&[2, 2, 2], &strides);
        full_neighbors(lin, &shape, &strides, 2, &mut buf);
        assert!(!buf.contains(&lin));
        assert_eq!(buf.len(), 124); // 5^3 - 1
    }

    #[test]
    fn code_space_is_three_to_the_ndim() {
        assert_eq!(code_space(2), 9);
        assert_eq!(code_space(5), 243);
        assert_eq!(code_space(7), 2187);
        assert!(code_space(7) < PARENT_NONE);
    }

    #[test]
    fn neighbors_with_codes_round_trip_both_stencils_2_to_7() {
        for ndim in 2..=7 {
            let shape: Vec<usize> = (0..ndim).map(|a| 3 + (a % 2)).collect();
            let strides = compute_strides(&shape);
            let total: usize = shape.iter().product();
            let mut plain = Vec::new();
            let mut coded = Vec::new();
            for stencil in [Stencil::VonNeumann, Stencil::Moore] {
                for lin in 0..total {
                    stencil.neighbors(lin, &shape, &strides, &mut plain);
                    stencil.neighbors_with_codes(lin, &shape, &strides, &mut coded);
                    // Same neighbours in the same order as the plain enumeration.
                    let got: Vec<usize> = coded.iter().map(|&(n, _)| n).collect();
                    assert_eq!(got, plain, "ndim={ndim} lin={lin} {stencil:?}");
                    // Every code decodes back to its neighbour, checked and unchecked.
                    for &(nbr, code) in &coded {
                        assert!(code < code_space(ndim));
                        assert_ne!(code, PARENT_NONE);
                        assert_eq!(apply_code(lin, code, &strides), nbr);
                        assert_eq!(apply_code_checked(lin, code, &shape, &strides), Some(nbr));
                    }
                    // Codes are distinct within one cell.
                    let mut codes: Vec<u16> = coded.iter().map(|&(_, c)| c).collect();
                    codes.sort();
                    codes.dedup();
                    assert_eq!(codes.len(), coded.len());
                }
            }
        }
    }

    #[test]
    fn von_neumann_codes_are_a_subset_of_moore_codes() {
        let shape = [4, 5, 3];
        let strides = compute_strides(&shape);
        let lin = index_to_linear(&[2, 2, 1], &strides);
        let mut vn = Vec::new();
        let mut moore = Vec::new();
        Stencil::VonNeumann.neighbors_with_codes(lin, &shape, &strides, &mut vn);
        Stencil::Moore.neighbors_with_codes(lin, &shape, &strides, &mut moore);
        for pair in &vn {
            assert!(moore.contains(pair), "{pair:?} missing from Moore");
        }
    }

    #[test]
    fn apply_code_checked_rejects_out_of_bounds_step() {
        let shape = [3, 3];
        let strides = compute_strides(&shape);
        // Cell (0,0): the code for Δ = (-1, 0) is digit 0 at axis 0 and 1 at axis 1 → 0*3 + 1 = 1.
        assert_eq!(apply_code_checked(0, 1, &shape, &strides), None);
        // Δ = (+1, 0) → digit 2 at axis 0, 1 at axis 1 → 2*3 + 1 = 7 → cell (1,0) = 3.
        assert_eq!(apply_code_checked(0, 7, &shape, &strides), Some(3));
    }

    #[test]
    fn walk_box_neighbors_matches_full_neighbors_and_stops_early() {
        for ndim in 2..=5 {
            let shape = vec![4usize; ndim];
            let strides = compute_strides(&shape);
            let total: usize = shape.iter().product();
            for r in 1..=2 {
                for lin in [0, total / 2, total - 1] {
                    let mut expected = Vec::new();
                    full_neighbors(lin, &shape, &strides, r, &mut expected);
                    let mut got = Vec::new();
                    let finished = walk_box_neighbors(lin, &shape, &strides, r, |n| {
                        got.push(n);
                        true
                    });
                    assert!(finished);
                    assert_eq!(got, expected, "ndim={ndim} r={r} lin={lin}");
                    // Early stop after two visits.
                    let mut seen = 0;
                    let finished = walk_box_neighbors(lin, &shape, &strides, r, |_| {
                        seen += 1;
                        seen < 2
                    });
                    assert!(!finished);
                    assert_eq!(seen, 2.min(expected.len()));
                }
            }
        }
    }
}
