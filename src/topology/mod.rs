//! `pes_analyzer.topology` submodule: full watershed segmentation and
//! merge-tree construction.

pub mod mep;
pub mod watershed;

use ndarray::{Array, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyTuple};

use crate::common::validate::{check_ndim, check_total_cells_fit_u32, parse_neighborhood};

#[pyfunction]
#[pyo3(name = "find_watershed_segmentation", signature = (energies, neighborhood = "von_neumann"))]
fn py_find_watershed_segmentation<'py>(
    py: Python<'py>,
    energies: PyReadonlyArrayDyn<'py, f64>,
    neighborhood: &str,
) -> PyResult<(Py<PyArrayDyn<i32>>, Py<PyList>, Py<PyList>)> {
    if !energies.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "energies must be C-contiguous; call np.ascontiguousarray(energies) if you intend a copy",
        ));
    }
    let arr = energies.as_array();
    let ndim = arr.ndim();
    check_ndim(ndim)?;
    check_total_cells_fit_u32(arr.len())?;
    let stencil = parse_neighborhood(neighborhood)?;

    let shape: Vec<usize> = arr.shape().to_vec();
    let result = py.allow_threads(|| watershed::watershed_segmentation_inner(arr, stencil));

    // Reshape Vec<i32> labels into an ndarray matching `energies.shape`.
    let labels_nd = Array::from_shape_vec(IxDyn(&shape), result.labels)
        .map_err(|e| PyValueError::new_err(format!("labels reshape failed: {}", e)))?;
    let labels_py = labels_nd.into_pyarray_bound(py).unbind();

    let basins_py = PyList::new_bound(
        py,
        result.basins.iter().map(|(idx, e)| {
            let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py)));
            PyTuple::new_bound(py, [idx_tuple.into_py(py), (*e).into_py(py)])
        }),
    )
    .unbind();

    let merges_py = PyList::new_bound(
        py,
        result.merges.iter().map(|(idx, e, d, s)| {
            let idx_tuple = PyTuple::new_bound(py, idx.iter().map(|&i| i.into_py(py)));
            PyTuple::new_bound(
                py,
                [
                    idx_tuple.into_py(py),
                    (*e).into_py(py),
                    (*d).into_py(py),
                    (*s).into_py(py),
                ],
            )
        }),
    )
    .unbind();

    Ok((labels_py, basins_py, merges_py))
}

/// Register the `topology` submodule on `parent`. Unlike `saddle` and
/// `extrema`, we do NOT patch `sys.modules` here: the real on-disk
/// `python/pes_analyzer/topology/__init__.py` is what makes the dotted
/// import path resolve at runtime.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "topology")?;
    m.add_function(wrap_pyfunction!(py_find_watershed_segmentation, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
