# Development

How to build, test, and iterate on `pes_analyzer`.

## Prerequisites

- **Python ≥ 3.10.** Verify with `python --version`.
- **Rust toolchain** (stable, edition 2024 support — Rust ≥ 1.85). Install via [rustup](https://rustup.rs/): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
- **maturin ≥ 1.7.** `pip install maturin`.

A virtualenv is recommended:
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install maturin
```

## Editable install

```bash
maturin develop --release
```

This rebuilds `python/pes_analyzer/_native.abi3.so` and installs `pes_analyzer` into the active Python environment.

**Always use `--release` unless you are actively debugging Rust.** The IWF and minima kernels are dramatically slower in debug builds — debug runs of the test suite can take minutes instead of seconds.

## Running tests

**Python integration tests** (run against the compiled extension; require `maturin develop` to be current):

```bash
pytest tests/
```

**Rust unit tests** (kernels and helpers; no Python rebuild needed):

```bash
cargo test
```

Both should be green before any commit that touches `src/` or `python/pes_analyzer/`.

## Common workflows

- **Edit Rust kernel** → `maturin develop --release` → `pytest tests/`.
- **Edit Python wrapper only** (e.g. `python/pes_analyzer/grid.py`) → no rebuild needed → `pytest tests/`.
- **Edit a Rust internal in `src/common/`** → `cargo test` first to exercise the unit tests, then `maturin develop --release` and `pytest tests/`.

## Release build profile

`Cargo.toml` sets:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
```

Expect link times of tens of seconds in exchange for tight inner loops. If you need a fast turnaround during development, you can build without LTO ad-hoc by passing `--profile dev` to `maturin develop`, but this also disables many compile-time optimisations and is rarely worth it.

## Versioning

The version number is duplicated between `Cargo.toml` (`[package].version`) and `pyproject.toml` (`[project].version`). **`Cargo.toml` is the source of truth.** When bumping, update both manually; there is no automated check yet.
