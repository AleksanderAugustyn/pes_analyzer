//! `pes_analyzer.topology` submodule: full watershed segmentation and
//! merge-tree construction.

pub mod mep;
pub mod watershed;

use ndarray::{Array, Array1, Array2, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyTuple};

use crate::common::nd::{compute_strides, linear_to_index};
use crate::common::scalar::Scalar;
use crate::common::validate::{
    check_index_in_bounds, check_index_length, check_ndim, check_total_cells_fit_u32,
    coerce_signed_indices, parse_neighborhood,
};

#[pyfunction]
#[pyo3(name = "find_watershed_segmentation", signature = (energies, neighborhood = "von_neumann", parents = false))]
fn py_find_watershed_segmentation<'py>(
    py: Python<'py>,
    energies: &Bound<'py, PyAny>,
    neighborhood: &str,
    parents: bool,
) -> PyResult<Py<PyTuple>> {
    if let Ok(a) = energies.extract::<PyReadonlyArrayDyn<f32>>() {
        run_find_watershed_segmentation(py, a, neighborhood, parents)
    } else if let Ok(a) = energies.extract::<PyReadonlyArrayDyn<f64>>() {
        run_find_watershed_segmentation(py, a, neighborhood, parents)
    } else {
        Err(PyValueError::new_err("energies dtype must be float32 or float64"))
    }
}

/// Returns `(labels, parents | None, basins, merges, merge_table)`; see
/// `python/pes_analyzer/topology/_flood.py` for the Python-facing object.
fn run_find_watershed_segmentation<'py, T: Scalar>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, T>,
    neighborhood: &str,
    parents: bool,
) -> PyResult<Py<PyTuple>> {
    if !energies.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "energies must be C-contiguous; call np.ascontiguousarray(energies) if you intend a copy",
        ));
    }
    let arr = energies.as_array();
    check_ndim(arr.ndim())?;
    check_total_cells_fit_u32(arr.len())?;
    let stencil = parse_neighborhood(neighborhood)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let strides = compute_strides(&shape);

    let opts = watershed::FloodOptions { stencil, record_parents: parents, stop_when_connected: None };
    let result = py.allow_threads(|| watershed::flood(arr, opts));

    let labels_py = Array::from_shape_vec(IxDyn(&shape), result.labels)
        .map_err(|e| PyValueError::new_err(format!("labels reshape failed: {e}")))?
        .into_pyarray_bound(py)
        .into_any();
    let parents_py = match result.parents {
        Some(p) => Array::from_shape_vec(IxDyn(&shape), p)
            .map_err(|e| PyValueError::new_err(format!("parents reshape failed: {e}")))?
            .into_pyarray_bound(py)
            .into_any(),
        None => py.None().into_bound(py),
    };
    let nd_tuple = |lin: u32| {
        PyTuple::new_bound(py, linear_to_index(lin as usize, &shape, &strides).iter().map(|&i| i.into_py(py)))
    };
    let basins_py = PyList::new_bound(
        py,
        result.basins.iter().map(|&(lin, e)| {
            PyTuple::new_bound(py, [nd_tuple(lin).into_py(py), e.to_f64().into_py(py)])
        }),
    );
    let merges_py = PyList::new_bound(
        py,
        result.merges.iter().map(|m| {
            PyTuple::new_bound(
                py,
                [nd_tuple(m.saddle).into_py(py), m.energy.to_f64().into_py(py), m.deeper.into_py(py), m.shallower.into_py(py)],
            )
        }),
    );
    let n_merges = result.merges.len();
    let mut table: Vec<u32> = Vec::with_capacity(n_merges * 5);
    for m in &result.merges {
        table.extend_from_slice(&[m.saddle, m.other, m.deeper, m.shallower, m.saddle_side]);
    }
    let table_py = Array2::from_shape_vec((n_merges, 5), table)
        .map_err(|e| PyValueError::new_err(format!("merge_table reshape failed: {e}")))?
        .into_pyarray_bound(py)
        .into_any();

    Ok(PyTuple::new_bound(
        py,
        [labels_py, parents_py, basins_py.into_any(), merges_py.into_any(), table_py],
    )
    .unbind())
}

#[pyfunction]
#[pyo3(name = "find_minimum_energy_path", signature = (energies, start, end, neighborhood = "von_neumann"))]
fn py_find_minimum_energy_path<'py>(
    py: Python<'py>,
    energies: &Bound<'py, PyAny>,
    start: Vec<i64>,
    end: Vec<i64>,
    neighborhood: &str,
) -> PyResult<Option<(Py<PyArray2<i64>>, Py<PyArray1<f64>>)>> {
    if let Ok(a) = energies.extract::<PyReadonlyArrayDyn<f32>>() {
        run_find_minimum_energy_path(py, a, start, end, neighborhood)
    } else if let Ok(a) = energies.extract::<PyReadonlyArrayDyn<f64>>() {
        run_find_minimum_energy_path(py, a, start, end, neighborhood)
    } else {
        Err(PyValueError::new_err(
            "energies dtype must be float32 or float64",
        ))
    }
}

fn run_find_minimum_energy_path<'py, T: Scalar>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, T>,
    start: Vec<i64>,
    end: Vec<i64>,
    neighborhood: &str,
) -> PyResult<Option<(Py<PyArray2<i64>>, Py<PyArray1<f64>>)>> {
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
    let stencil = parse_neighborhood(neighborhood)?;

    let start_idx = coerce_signed_indices(&start)?;
    let end_idx = coerce_signed_indices(&end)?;
    check_index_in_bounds(arr.shape(), &start_idx)?;
    check_index_in_bounds(arr.shape(), &end_idx)?;
    if arr[start_idx.as_slice()].is_nan() {
        return Err(PyValueError::new_err("energy at `start` is NaN"));
    }
    if arr[end_idx.as_slice()].is_nan() {
        return Err(PyValueError::new_err("energy at `end` is NaN"));
    }

    let result = py.allow_threads(|| mep::mep_inner(arr.view(), &start_idx, &end_idx, stencil));

    match result {
        None => Ok(None),
        Some(path_lin) => {
            let shape: Vec<usize> = arr.shape().to_vec();
            let strides = compute_strides(&shape);
            let flat = arr
                .as_slice()
                .expect("energies must be C-contiguous (checked above)");
            let k = path_lin.len();
            let mut idx_flat: Vec<i64> = Vec::with_capacity(k * ndim);
            let mut path_e: Vec<f64> = Vec::with_capacity(k);
            for &lin in &path_lin {
                let nd_idx = linear_to_index(lin, &shape, &strides);
                idx_flat.extend(nd_idx.iter().map(|&i| i as i64));
                path_e.push(flat[lin].to_f64());
            }
            let indices = Array2::from_shape_vec((k, ndim), idx_flat)
                .map_err(|e| PyValueError::new_err(format!("path reshape failed: {e}")))?;
            Ok(Some((
                indices.into_pyarray_bound(py).unbind(),
                Array1::from_vec(path_e).into_pyarray_bound(py).unbind(),
            )))
        }
    }
}

/// Register the `topology` submodule on `parent`. Unlike `saddle` and
/// `extrema`, we do NOT patch `sys.modules` here: the real on-disk
/// `python/pes_analyzer/topology/__init__.py` is what makes the dotted
/// import path resolve at runtime.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "topology")?;
    m.add_function(wrap_pyfunction!(py_find_watershed_segmentation, &m)?)?;
    m.add_function(wrap_pyfunction!(py_find_minimum_energy_path, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
