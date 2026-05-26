# Architecture

A tour of `pes_analyzer` for contributors. Read this before modifying the Rust kernels or the Python wrapper.

## Repo layout

```
pes_analyzer/
├── Cargo.toml            — Rust crate manifest
├── pyproject.toml        — maturin build config
├── src/                  — Rust sources
│   ├── lib.rs            — PyO3 module entry point, registers submodules
│   ├── common/           — shared internals (no PyO3)
│   │   ├── nd.rs         — N-D indexing helpers
│   │   ├── dsu.rs        — union-find used by IWF
│   │   └── validate.rs   — pre-flight input checks for PyO3 wrappers
│   ├── extrema/
│   │   ├── mod.rs              — PyO3 wrappers for find_minima_grid, find_maxima_grid, find_extrema_grid
│   │   ├── local_minima.rs     — generic local_extreme_inner<C> kernel + minima/maxima wrappers
│   │   └── local_extrema.rs    — combined single-sweep local_extrema_inner kernel
│   └── saddle/
│       ├── mod.rs        — PyO3 wrapper for find_iwf_grid
│       └── iwf_grid.rs   — pure-Rust kernel
├── python/pes_analyzer/  — Python-side package
│   ├── __init__.py       — re-exports submodules
│   ├── grid.py           — pure-Python build_dense helper
│   ├── _native.abi3.so   — compiled extension (built by maturin)
│   ├── saddle/__init__.pyi — type stub (see below)
│   └── extrema/__init__.pyi — type stubs for all three extrema functions
└── tests/                — pytest integration tests
```

## The Python / Rust seam

`maturin` builds a single compiled extension installed at `python/pes_analyzer/_native.abi3.so`. The Python package `pes_analyzer` re-exports `_native.saddle` and `_native.extrema` so users can write `from pes_analyzer.saddle import find_iwf_grid`.

### The `.pyi` stub trick

The runtime submodules `pes_analyzer.saddle` and `pes_analyzer.extrema` are registered dynamically from Rust by patching `sys.modules`:

```rust
// src/saddle/mod.rs:78-80
let sys = py.import_bound("sys")?;
let modules = sys.getattr("modules")?;
modules.set_item("pes_analyzer.saddle", &m)?;
```

Static type checkers cannot see that registration, so the on-disk stubs `python/pes_analyzer/saddle/__init__.pyi` and `python/pes_analyzer/extrema/__init__.pyi` exist purely to satisfy them. **Any new submodule needs both halves**: the `sys.modules` patch in Rust *and* a matching `__init__.pyi` for the type checker.

## GIL handling

All compute happens inside `py.allow_threads(|| ...)`:

```rust
// src/saddle/mod.rs:55
let result = py.allow_threads(|| iwf_grid::iwf_grid_inner(arr, &start_idx, &end_idx));
```

Inputs cross the boundary as `PyReadonlyArrayDyn`, which holds a read-lock on the NumPy buffer for the duration of the call. The raw pointer obtained via `.as_array()` remains valid without the GIL, and concurrent Python code cannot mutate the buffer underneath.

## Shared internals: `src/common/`

- **`nd.rs`** — N-D indexing helpers used by both algorithms.
  - `compute_strides(shape)` — row-major (C order) strides.
  - `index_to_linear(idx, strides)` / `linear_to_index(lin, shape, strides)` — N-D ↔ flat index conversion.
  - `axis_neighbors(lin, shape, strides, &mut out)` — fills the 2N axis-only neighbour list. Used by `find_iwf_grid`.
  - `full_neighbors(lin, shape, strides, &mut out)` — fills the 3ᴺ−1 king-move neighbour list. Used by `find_minima_grid`.
- **`dsu.rs`** — `DisjointSetUnion`, the union-find structure that drives the watershed.
- **`validate.rs`** — pre-flight input checks (`check_ndim`, `check_total_cells_fit_u32`, `check_index_length`, `check_index_in_bounds`, `coerce_signed_indices`). PyO3 wrappers call these before doing any work.

## Adding a new algorithm

1. Implement the pure-Rust inner function in `src/<area>/<algo>.rs`. Unit-test it directly with `ndarray::ArrayD` views.
2. Add a PyO3 wrapper in `src/<area>/mod.rs` that:
   - validates inputs through `common::validate::*`,
   - rejects non-contiguous arrays explicitly,
   - calls the inner function inside `py.allow_threads(|| ...)`,
   - converts results back to Python types (use `PyTuple::new_bound` for `tuple[int, ...]` returns — PyO3's default `Vec<usize>` conversion produces a `list`).
3. If introducing a new submodule (not just a new function in an existing one):
   - Register it in `src/lib.rs` (`<area>::register(m)?`).
   - Patch `sys.modules` for the dotted name in `<area>/mod.rs::register`.
   - Create `python/pes_analyzer/<area>/__init__.pyi` matching the runtime API.
3a. **If the submodule needs Python helpers alongside the Rust kernel**, make `python/pes_analyzer/<area>/` a real Python package with an `__init__.py` that imports from `pes_analyzer._native.<area>` and re-exports the helpers. In this case, skip the `sys.modules` patch in `<area>/mod.rs::register` — the real on-disk package handles the dotted-name resolution. `pes_analyzer.topology` is the reference example.
4. Add a Python integration test in `tests/test_<area>.py`. Tests run against the compiled `.so`, so `maturin develop` must be current.

## Build profile

`Cargo.toml` sets `lto = "fat"` and `codegen-units = 1` for release. Linking is slow; inner loops are tight.
