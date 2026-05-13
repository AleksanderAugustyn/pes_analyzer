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
pub fn axis_neighbors(
    linear: usize,
    shape: &[usize],
    strides: &[usize],
    out: &mut Vec<usize>,
) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_c_contiguous() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[2, 2, 2, 2, 2, 2, 2]), vec![64, 32, 16, 8, 4, 2, 1]);
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
        let mut got: Vec<usize> = buf.iter().copied().collect();
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
}
