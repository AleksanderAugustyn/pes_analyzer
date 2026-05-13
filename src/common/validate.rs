//! Input validation. The functions here return `PyResult` so that PyO3
//! converts the errors to the appropriate Python exception types
//! (`ValueError`, `IndexError`). The pure-Rust shape checks are exposed so
//! they can be unit-tested without a Python interpreter.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

pub fn check_ndim(ndim: usize) -> PyResult<()> {
    if (2..=7).contains(&ndim) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "ndim must be in [2, 7], got {ndim}"
        )))
    }
}

pub fn check_index_length(shape: &[usize], idx_len: usize) -> PyResult<()> {
    if idx_len != shape.len() {
        return Err(PyValueError::new_err(format!(
            "index length {idx_len} does not match ndim {}",
            shape.len()
        )));
    }
    Ok(())
}

pub fn check_index_in_bounds(shape: &[usize], idx: &[usize]) -> PyResult<()> {
    for (axis, (&i, &dim)) in idx.iter().zip(shape.iter()).enumerate() {
        if i >= dim {
            return Err(PyIndexError::new_err(format!(
                "index {i} out of bounds for axis {axis} with size {dim}"
            )));
        }
    }
    Ok(())
}

pub fn check_total_cells_fit_u32(total: usize) -> PyResult<()> {
    if total > u32::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "grid has {total} cells; this build is capped at u32::MAX = {}",
            u32::MAX
        )));
    }
    Ok(())
}

/// Convert a signed Python integer index tuple to `usize`, returning
/// `IndexError` on negatives.
pub fn coerce_signed_indices(raw: &[i64]) -> PyResult<Vec<usize>> {
    raw.iter()
        .map(|&v| {
            if v < 0 {
                Err(PyIndexError::new_err(format!(
                    "negative index {v} is not allowed"
                )))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_ndim_accepts_2_through_7() {
        for n in 2..=7 {
            assert!(check_ndim(n).is_ok());
        }
    }

    #[test]
    fn check_ndim_rejects_1_and_8() {
        assert!(check_ndim(1).is_err());
        assert!(check_ndim(8).is_err());
        assert!(check_ndim(0).is_err());
    }

    #[test]
    fn check_index_tuples_accepts_in_bounds() {
        let shape = [3, 4, 5];
        assert!(check_index_in_bounds(&shape, &[0, 0, 0]).is_ok());
        assert!(check_index_in_bounds(&shape, &[2, 3, 4]).is_ok());
    }

    #[test]
    fn check_index_tuples_rejects_out_of_bounds() {
        let shape = [3, 4, 5];
        assert!(check_index_in_bounds(&shape, &[3, 0, 0]).is_err());
        assert!(check_index_in_bounds(&shape, &[0, 4, 0]).is_err());
        assert!(check_index_in_bounds(&shape, &[0, 0, 5]).is_err());
    }

    #[test]
    fn check_index_length_rejects_mismatch() {
        let shape = [3, 4];
        assert!(check_index_length(&shape, 2).is_ok());
        assert!(check_index_length(&shape, 3).is_err());
        assert!(check_index_length(&shape, 1).is_err());
    }

    #[test]
    fn check_total_under_u32_max() {
        assert!(check_total_cells_fit_u32(1000).is_ok());
        assert!(check_total_cells_fit_u32(u32::MAX as usize).is_ok());
        assert!(check_total_cells_fit_u32(u32::MAX as usize + 1).is_ok() == false);
    }
}
