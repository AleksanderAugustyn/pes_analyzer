//! pes_analyzer: Rust-backed Python package for fast PES analysis.
//!
//! Public Python API: `pes_analyzer.saddle.find_iwf_grid`.

use pyo3::prelude::*;

mod common;
mod saddle;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    saddle::register(m)?;
    Ok(())
}
