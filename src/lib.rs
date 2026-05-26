//! pes_analyzer: Rust-backed Python package for fast PES analysis.
//!
//! Public Python API: `pes_analyzer.saddle.find_iwf_grid`.

// PyO3 0.22 + Rust edition 2024: the #[pyfunction]/#[pymodule] macros emit
// unsafe calls without wrapping them in `unsafe {}` blocks (edition 2024
// requires explicit `unsafe {}`), and also trigger spurious useless_conversion
// lints in generated wrapper code. Allow both crate-wide until PyO3 ships an
// edition-2024-clean release.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod common;
mod extrema;
mod saddle;
mod topology;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    extrema::register(m)?;
    saddle::register(m)?;
    topology::register(m)?;
    Ok(())
}
