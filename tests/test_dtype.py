"""float32/float64 dtype-dispatch parity at the PyO3 boundary.

Each kernel accepts both float32 and float64 energies. The f32 path must
agree with the f64 path on integer indices (exactly) and on energies (to
float32 precision). Non-float dtypes raise ``ValueError``.
"""

import numpy as np
import pytest

from pes_analyzer.extrema import find_minima_grid
from pes_analyzer.topology import (
    find_watershed_segmentation,
    find_minimum_energy_path,
)
from pes_analyzer.saddle import find_iwf_grid


def _bowl(dtype):
    g = np.empty((5, 5), dtype=dtype)
    for i in range(5):
        for j in range(5):
            g[i, j] = (i - 2) ** 2 + (j - 2) ** 2
    return np.ascontiguousarray(g)


def test_find_minima_grid_f32_matches_f64():
    g32, g64 = _bowl(np.float32), _bowl(np.float64)
    m32 = find_minima_grid(g32, neighborhood_range=1, confirm_range=2)
    m64 = find_minima_grid(g64, neighborhood_range=1, confirm_range=2)
    assert [i for i, _ in m32] == [i for i, _ in m64]
    for (_, e32), (_, e64) in zip(m32, m64):
        assert abs(e32 - e64) < 1e-4


def test_watershed_f32_matches_f64():
    g32, g64 = _bowl(np.float32), _bowl(np.float64)
    l32, b32, _ = find_watershed_segmentation(g32, neighborhood="von_neumann")
    l64, b64, _ = find_watershed_segmentation(g64, neighborhood="von_neumann")
    assert list(np.asarray(l32).ravel()) == list(np.asarray(l64).ravel())


def test_mep_f32_matches_f64():
    g32, g64 = _bowl(np.float32), _bowl(np.float64)
    r32 = find_minimum_energy_path(g32, (0, 0), (4, 4), neighborhood="von_neumann")
    r64 = find_minimum_energy_path(g64, (0, 0), (4, 4), neighborhood="von_neumann")
    assert (r32 is None) == (r64 is None)
    if r32 is not None:
        idx32, e32 = r32
        idx64, e64 = r64
        assert np.array_equal(np.asarray(idx32), np.asarray(idx64))
        assert np.allclose(np.asarray(e32), np.asarray(e64), atol=1e-4)


def test_iwf_f32_matches_f64():
    g32, g64 = _bowl(np.float32), _bowl(np.float64)
    r32 = find_iwf_grid(g32, (0, 0), (4, 4), neighborhood="von_neumann")
    r64 = find_iwf_grid(g64, (0, 0), (4, 4), neighborhood="von_neumann")
    assert (r32 is None) == (r64 is None)
    if r32 is not None:
        idx32, e32 = r32
        idx64, e64 = r64
        assert idx32 == idx64
        assert abs(e32 - e64) < 1e-4


def test_int_dtype_rejected():
    g = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        find_minima_grid(g, neighborhood_range=1, confirm_range=2)
