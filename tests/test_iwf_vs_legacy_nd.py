"""Side-by-side equivalence: pes_analyzer.saddle.find_iwf_grid vs an
in-test pure-Python watershed oracle, on a real PES parquet.

This is gated on the presence of pes_Z120_N180.parquet. The N-D dense
grid is built via pes_analyzer.grid.build_dense (the same code path
MapMaker uses).
"""

from pathlib import Path

import numpy as np
import pytest

# polars is only needed by this data-gated test, not by the package; skip the
# whole module when it is absent (e.g. the clean-venv checks in CI).
pl = pytest.importorskip("polars")

from pes_analyzer.grid   import build_dense
from pes_analyzer.saddle import find_iwf_grid

PARQUET_PATH = Path(__file__).resolve().parent.parent / 'pes_Z120_N180.parquet'

pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason=f'requires {PARQUET_PATH.name} in repo root',
)


def _build_grid_from_parquet():
    df = pl.read_parquet(PARQUET_PATH).filter(pl.col('is_valid'))
    active = ('c', 'a3', 'a4', 'a5', 'a6')
    coords = {n: df[n].to_numpy() for n in active}
    energies, axes = build_dense(coords, df['total_energy'].to_numpy())
    return energies, axes


def _legacy_watershed(energies, start, end):
    """Reference watershed for cross-checking. O(N log N + N · 2D) in
    pure Python — slow but obviously correct.
    """
    shape = energies.shape
    flat = energies.ravel()
    n_total = flat.size

    sorted_idx = sorted(
        (i for i in range(n_total) if not np.isnan(flat[i])),
        key=lambda i: flat[i],
    )
    parent = list(range(n_total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def to_lin(ix):
        lin = 0
        stride = 1
        for v, s in zip(reversed(ix), reversed(shape)):
            lin += v * stride
            stride *= s
        return lin

    def axis_neighbors(lin):
        rem = lin
        idx = [0] * len(shape)
        for axis in range(len(shape)):
            s = 1
            for d in shape[axis + 1:]:
                s *= d
            idx[axis] = rem // s
            rem %= s
        out = []
        for axis in range(len(shape)):
            for d in (-1, 1):
                nidx = list(idx)
                nidx[axis] += d
                if 0 <= nidx[axis] < shape[axis]:
                    out.append(to_lin(nidx))
        return out

    start_lin = to_lin(start)
    end_lin   = to_lin(end)
    processed = np.zeros(n_total, dtype=bool)

    for i in sorted_idx:
        processed[i] = True
        for n in axis_neighbors(i):
            if processed[n]:
                union(i, n)
        if find(start_lin) == find(end_lin):
            rem = i
            nd_idx = [0] * len(shape)
            for axis in range(len(shape)):
                s = 1
                for d in shape[axis + 1:]:
                    s *= d
                nd_idx[axis] = rem // s
                rem %= s
            return tuple(nd_idx), float(flat[i])
    return None


def _basin_minimum(energies, axes, c_lo, c_hi):
    """Return the (idx_tuple, energy) of the lowest-energy non-NaN cell
    with c_lo <= c < c_hi. Used to derive start/end without depending on
    the basin-finder pipeline.
    """
    c_idx_lo = int(np.searchsorted(axes['c'], c_lo, side='left'))
    c_idx_hi = int(np.searchsorted(axes['c'], c_hi, side='left'))
    slicer = [slice(None)] * energies.ndim
    slicer[0] = slice(c_idx_lo, c_idx_hi)
    sub = energies[tuple(slicer)]
    flat_argmin = int(np.nanargmin(sub))
    sub_idx = np.unravel_index(flat_argmin, sub.shape)
    full_idx = list(sub_idx)
    full_idx[0] += c_idx_lo
    return tuple(int(i) for i in full_idx), float(energies[tuple(full_idx)])


def test_inner_saddle_matches_legacy_oracle():
    energies, axes = _build_grid_from_parquet()
    start, _ = _basin_minimum(energies, axes, 0.0, 1.3)
    end,   _ = _basin_minimum(energies, axes, 1.3, 1.7)

    legacy = _legacy_watershed(energies, start, end)
    rust   = find_iwf_grid(energies, start, end)
    assert rust is not None
    assert legacy is not None
    assert rust[0] == legacy[0]
    assert abs(rust[1] - legacy[1]) < 1e-12


def test_outer_saddle_matches_legacy_oracle():
    energies, axes = _build_grid_from_parquet()
    start, _ = _basin_minimum(energies, axes, 1.3, 1.7)
    end,   _ = _basin_minimum(energies, axes, 1.7, 100.0)

    legacy = _legacy_watershed(energies, start, end)
    rust   = find_iwf_grid(energies, start, end)
    assert rust is not None
    assert legacy is not None
    assert rust[0] == legacy[0]
    assert abs(rust[1] - legacy[1]) < 1e-12
