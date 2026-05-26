//! `pes_analyzer.extrema` submodule: local-extremum search algorithms.

pub mod local_extrema;
pub mod local_minima;

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyTuple};

use crate::common::validate::{
    check_confirm_range, check_ndim, check_neighborhood_range, check_total_cells_fit_u32,
};

#[pyfunction]
#[pyo3(
    name = "find_minima_grid",
    signature = (energies, *, neighborhood_range = 1, confirm_range = None)
)]
fn py_find_minima_grid<'py>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, f64>,
    neighborhood_range: usize,
    confirm_range: Option<usize>,
) -> PyResult<Py<PyList>> {
    if !energies.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "energies must be C-contiguous; call np.ascontiguousarray(energies) if you intend a copy",
        ));
    }

    let arr = energies.as_array();
    check_ndim(arr.ndim())?;
    check_total_cells_fit_u32(arr.len())?;
    check_neighborhood_range(neighborhood_range)?;
    check_confirm_range(confirm_range, neighborhood_range)?;

    // Compute under released GIL.
    let result = py.allow_threads(|| {
        local_minima::local_minima_inner(arr, neighborhood_range, confirm_range)
    });

    // Build Python list of (tuple[int, ...], float).
    let list = PyList::empty_bound(py);
    for (idx, energy) in result {
        let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py)));
        let pair = PyTuple::new_bound(py, [idx_tuple.into_py(py), energy.into_py(py)]);
        list.append(pair)?;
    }
    Ok(list.unbind())
}

#[pyfunction]
#[pyo3(
    name = "find_maxima_grid",
    signature = (energies, *, neighborhood_range = 1, confirm_range = None)
)]
fn py_find_maxima_grid<'py>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, f64>,
    neighborhood_range: usize,
    confirm_range: Option<usize>,
) -> PyResult<Py<PyList>> {
    if !energies.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "energies must be C-contiguous; call np.ascontiguousarray(energies) if you intend a copy",
        ));
    }

    let arr = energies.as_array();
    check_ndim(arr.ndim())?;
    check_total_cells_fit_u32(arr.len())?;
    check_neighborhood_range(neighborhood_range)?;
    check_confirm_range(confirm_range, neighborhood_range)?;

    let result = py.allow_threads(|| {
        local_minima::local_maxima_inner(arr, neighborhood_range, confirm_range)
    });

    let list = PyList::empty_bound(py);
    for (idx, energy) in result {
        let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py)));
        let pair = PyTuple::new_bound(py, [idx_tuple.into_py(py), energy.into_py(py)]);
        list.append(pair)?;
    }
    Ok(list.unbind())
}

#[pyfunction]
#[pyo3(
    name = "find_extrema_grid",
    signature = (energies, *, neighborhood_range = 1, confirm_range = None)
)]
fn py_find_extrema_grid<'py>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, f64>,
    neighborhood_range: usize,
    confirm_range: Option<usize>,
) -> PyResult<Py<PyTuple>> {
    if !energies.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "energies must be C-contiguous; call np.ascontiguousarray(energies) if you intend a copy",
        ));
    }

    let arr = energies.as_array();
    check_ndim(arr.ndim())?;
    check_total_cells_fit_u32(arr.len())?;
    check_neighborhood_range(neighborhood_range)?;
    check_confirm_range(confirm_range, neighborhood_range)?;

    let (mins, maxes) = py.allow_threads(|| {
        local_extrema::local_extrema_inner(arr, neighborhood_range, confirm_range)
    });

    let mins_list = PyList::empty_bound(py);
    for (idx, energy) in mins {
        let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py)));
        let pair = PyTuple::new_bound(py, [idx_tuple.into_py(py), energy.into_py(py)]);
        mins_list.append(pair)?;
    }

    let maxes_list = PyList::empty_bound(py);
    for (idx, energy) in maxes {
        let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py)));
        let pair = PyTuple::new_bound(py, [idx_tuple.into_py(py), energy.into_py(py)]);
        maxes_list.append(pair)?;
    }

    let pair = PyTuple::new_bound(py, [mins_list.into_py(py), maxes_list.into_py(py)]);
    Ok(pair.unbind())
}

/// Register the `extrema` submodule on `parent`. Also inserts the submodule
/// into `sys.modules` under the dotted name `pes_analyzer.extrema` so that
/// `from pes_analyzer.extrema import find_minima_grid` works at runtime.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "extrema")?;
    m.add_function(wrap_pyfunction!(py_find_minima_grid, &m)?)?;
    m.add_function(wrap_pyfunction!(py_find_maxima_grid, &m)?)?;
    m.add_function(wrap_pyfunction!(py_find_extrema_grid, &m)?)?;
    parent.add_submodule(&m)?;

    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pes_analyzer.extrema", &m)?;
    Ok(())
}
