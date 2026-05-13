//! `pes_analyzer.saddle` submodule: saddle-point search algorithms.

pub mod iwf_grid;

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use crate::common::validate::{
    check_index_in_bounds, check_index_length, check_ndim, check_total_cells_fit_u32,
    coerce_signed_indices,
};

#[pyfunction]
#[pyo3(name = "find_iwf_grid")]
fn py_find_iwf_grid<'py>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, f64>,
    start: Vec<i64>,
    end: Vec<i64>,
) -> PyResult<Option<(Py<PyTuple>, f64)>> {
    if !energies.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "energies must be C-contiguous; call np.ascontiguousarray(energies) if you intend a copy",
        ));
    }

    let arr = energies.as_array();
    let ndim = arr.ndim();
    check_ndim(ndim)?;
    check_index_length(arr.shape(), start.len())?;
    check_index_length(arr.shape(), end.len())?;
    check_total_cells_fit_u32(arr.len())?;

    let start_idx = coerce_signed_indices(&start)?;
    let end_idx = coerce_signed_indices(&end)?;
    check_index_in_bounds(arr.shape(), &start_idx)?;
    check_index_in_bounds(arr.shape(), &end_idx)?;

    // NaN-at-endpoint check (needs the array data, so done here not in
    // common::validate which has no array dependency).
    let start_e = arr[start_idx.as_slice()];
    let end_e = arr[end_idx.as_slice()];
    if start_e.is_nan() {
        return Err(PyValueError::new_err("energy at `start` is NaN"));
    }
    if end_e.is_nan() {
        return Err(PyValueError::new_err("energy at `end` is NaN"));
    }

    // Compute under released GIL. `arr` is a view over the NumPy buffer;
    // `PyReadonlyArrayDyn` keeps the underlying array alive and read-locked
    // for the duration of this function, so the pointer remains valid.
    let result = py.allow_threads(|| iwf_grid::iwf_grid_inner(arr, &start_idx, &end_idx));

    // Spec §3.2 requires the returned index to be a `tuple[int, ...]`, not
    // a list. PyO3's default `Vec<usize>` conversion produces a Python list,
    // so we build the tuple explicitly.
    match result {
        None => Ok(None),
        Some((idx, energy)) => {
            let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py))).unbind();
            Ok(Some((idx_tuple, energy)))
        }
    }
}

/// Register the `saddle` submodule on `parent`. Also inserts the submodule
/// into `sys.modules` under the dotted name `pes_analyzer.saddle` so that
/// `from pes_analyzer.saddle import find_iwf_grid` works at runtime.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "saddle")?;
    m.add_function(wrap_pyfunction!(py_find_iwf_grid, &m)?)?;
    parent.add_submodule(&m)?;

    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pes_analyzer.saddle", &m)?;
    Ok(())
}
