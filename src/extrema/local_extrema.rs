//! Combined single-sweep local-minima + local-maxima search on a dense
//! N-D grid. One stencil walk per cell produces both polarities; the
//! confirm stage runs independently per polarity because the surviving
//! candidate sets are typically small.

use ndarray::ArrayViewD;

use crate::common::nd::{compute_strides, full_neighbors, linear_to_index};
use crate::common::scalar::Scalar;

/// Find all strict local minima AND local maxima in `energies` in one
/// sweep. Each non-NaN cell's neighbour walk updates both a
/// `beats_all_lower` flag (for the minima check) and a `beats_all_higher`
/// flag (for the maxima check); neither flag can short-circuit until both
/// are decided, so the inner loop runs to completion.
///
/// Returns `(minima, maxima)` with the same shapes as the single-polarity
/// kernels. Minima are sorted ascending by energy; maxima descending.
///
/// Preconditions (enforced by the PyO3 wrapper before calling):
/// - `energies` is a view over a C-contiguous buffer of element type `T`.
/// - `energies.ndim()` is in `[2, 7]`.
/// - `energies.len() <= u32::MAX`.
/// - `find_r >= 1`.
/// - `confirm_r`, if `Some(R)`, satisfies `R in [1, 5]`; `R <= find_r` is a no-op.
pub fn local_extrema_inner<T: Scalar>(
    energies: ArrayViewD<'_, T>,
    find_r: usize,
    confirm_r: Option<usize>,
) -> (Vec<(Vec<usize>, T)>, Vec<(Vec<usize>, T)>) {
    let shape: Vec<usize> = energies.shape().to_vec();
    let strides = compute_strides(&shape);
    let flat: &[T] = energies
        .as_slice()
        .expect("energies must be C-contiguous (enforced upstream)");

    // Stage 1: single sweep, two flags per cell.
    let stencil_max = (2 * find_r + 1).saturating_pow(shape.len() as u32);
    let mut nbrs: Vec<usize> = Vec::with_capacity(stencil_max);
    let mut min_candidates: Vec<(usize, T)> = Vec::new();
    let mut max_candidates: Vec<(usize, T)> = Vec::new();

    for (lin, &e) in flat.iter().enumerate() {
        if e.is_nan() {
            continue;
        }
        full_neighbors(lin, &shape, &strides, find_r, &mut nbrs);

        let mut has_valid_neighbor = false;
        let mut beats_all_lower = true;   // tentatively a minimum
        let mut beats_all_higher = true;  // tentatively a maximum
        for &nbr_lin in &nbrs {
            let ne = flat[nbr_lin];
            if ne.is_nan() {
                continue;
            }
            has_valid_neighbor = true;
            if ne < e {
                beats_all_lower = false;
            }
            if ne > e {
                beats_all_higher = false;
            }
            if !beats_all_lower && !beats_all_higher {
                break;
            }
        }

        if has_valid_neighbor {
            if beats_all_lower {
                min_candidates.push((lin, e));
            }
            if beats_all_higher {
                max_candidates.push((lin, e));
            }
        }
    }

    // Stage 2: confirm each polarity independently against the wider
    // stencil, if requested.
    let (min_kept, max_kept) = match confirm_r {
        Some(r) if r > find_r => {
            let stencil_max_r = (2 * r + 1).saturating_pow(shape.len() as u32);
            let mut nbrs_r: Vec<usize> = Vec::with_capacity(stencil_max_r);
            let min_kept = confirm(&min_candidates, flat, &shape, &strides, r, &mut nbrs_r, |ne, e| ne < e);
            let max_kept = confirm(&max_candidates, flat, &shape, &strides, r, &mut nbrs_r, |ne, e| ne > e);
            (min_kept, max_kept)
        }
        _ => (min_candidates, max_candidates),
    };

    let mut mins: Vec<(Vec<usize>, T)> = min_kept
        .into_iter()
        .map(|(lin, e)| (linear_to_index(lin, &shape, &strides), e))
        .collect();
    mins.sort_by(|a, b| a.1.tcmp(&b.1));

    let mut maxs: Vec<(Vec<usize>, T)> = max_kept
        .into_iter()
        .map(|(lin, e)| (linear_to_index(lin, &shape, &strides), e))
        .collect();
    maxs.sort_by(|a, b| b.1.tcmp(&a.1));

    (mins, maxs)
}

fn confirm<T: Scalar, C>(
    candidates: &[(usize, T)],
    flat: &[T],
    shape: &[usize],
    strides: &[usize],
    r: usize,
    nbrs: &mut Vec<usize>,
    is_dominated: C,
) -> Vec<(usize, T)>
where
    C: Fn(T, T) -> bool,
{
    let mut kept: Vec<(usize, T)> = Vec::with_capacity(candidates.len());
    for &(lin, e) in candidates {
        full_neighbors(lin, shape, strides, r, nbrs);
        let mut has_valid_neighbor = false;
        let mut beats_all = true;
        for &nbr_lin in nbrs.iter() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extrema::local_minima::{local_maxima_inner, local_minima_inner};
    use ndarray::{Array, ArrayD, IxDyn};

    fn to_dyn<const N: usize>(shape: [usize; N], data: Vec<f64>) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&shape), data).unwrap()
    }

    #[test]
    fn equivalence_with_separate_calls_2d_r1_none() {
        let e = vec![
            1.0, 2.0, 1.0,
            2.0, 0.0, 2.0,
            1.0, 2.0, 1.0,
        ];
        let arr = to_dyn([3, 3], e);
        let combined = local_extrema_inner(arr.view(), 1, None);
        let separate = (
            local_minima_inner(arr.view(), 1, None),
            local_maxima_inner(arr.view(), 1, None),
        );
        assert_eq!(combined, separate);
    }

    #[test]
    fn equivalence_with_separate_calls_2d_r1_confirm_2() {
        let mut e = vec![5.0; 25];
        e[0] = 2.0;
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);
        let combined = local_extrema_inner(arr.view(), 1, Some(2));
        let separate = (
            local_minima_inner(arr.view(), 1, Some(2)),
            local_maxima_inner(arr.view(), 1, Some(2)),
        );
        assert_eq!(combined, separate);
    }

    #[test]
    fn equivalence_with_separate_calls_2d_r2_none() {
        let mut e = vec![5.0; 25];
        e[0] = 2.0;
        e[2 * 5 + 2] = 3.0;
        e[4 * 5 + 4] = 1.0;
        let arr = to_dyn([5, 5], e);
        let combined = local_extrema_inner(arr.view(), 2, None);
        let separate = (
            local_minima_inner(arr.view(), 2, None),
            local_maxima_inner(arr.view(), 2, None),
        );
        assert_eq!(combined, separate);
    }

    #[test]
    fn equivalence_with_nan_walls() {
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
        for r in 1usize..=3 {
            for c in [None, Some(2), Some(3), Some(4)] {
                if let Some(ci) = c {
                    if ci < r {
                        continue;
                    }
                }
                let combined = local_extrema_inner(arr.view(), r, c);
                let separate = (
                    local_minima_inner(arr.view(), r, c),
                    local_maxima_inner(arr.view(), r, c),
                );
                assert_eq!(combined, separate, "mismatch at r={r}, c={c:?}");
            }
        }
    }

    #[test]
    fn equivalence_5d_smoke() {
        let shape = [3usize, 3, 3, 3, 3];
        let total: usize = shape.iter().product();
        let mut data: Vec<f64> = (0..total).map(|i| (i as f64) * 0.5).collect();
        // Plant one clear minimum and one clear maximum in the interior.
        let center: Vec<usize> = vec![1; 5];
        let center_lin: usize = {
            let strides = compute_strides(&shape);
            center.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
        };
        data[center_lin] = -100.0;
        data[0] = 1_000.0;
        let arr = to_dyn(shape, data);

        let combined = local_extrema_inner(arr.view(), 1, None);
        let separate = (
            local_minima_inner(arr.view(), 1, None),
            local_maxima_inner(arr.view(), 1, None),
        );
        assert_eq!(combined, separate);
    }

    #[test]
    fn all_nan_stencil_disqualifies_both() {
        let n = f64::NAN;
        let e = vec![n, n, n, n, 0.0, n, n, n, n];
        let arr = to_dyn([3, 3], e);
        let (mins, maxes) = local_extrema_inner(arr.view(), 1, None);
        assert!(mins.is_empty());
        assert!(maxes.is_empty());
    }

    #[test]
    fn tuple_shape_smoke() {
        let arr = to_dyn([2, 2], vec![0.0, 1.0, 1.0, 0.0]);
        let (mins, maxes) = local_extrema_inner(arr.view(), 1, None);
        // Just checks the destructure works and lists are populated as expected.
        assert!(!mins.is_empty() || !maxes.is_empty());
    }
}
