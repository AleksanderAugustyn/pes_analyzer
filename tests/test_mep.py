"""Integration tests for find_minimum_energy_path (compiled extension).

Run against the compiled extension; require ``maturin develop --release``
to be current.
"""

from __future__ import annotations

import numpy as np
import pytest

from pes_analyzer.saddle import find_iwf_grid
from pes_analyzer.topology import find_minimum_energy_path, find_watershed_segmentation


def _chain_grid() -> np.ndarray:
    # 1x9 chain: basins at cols 0, 4, 8; saddles at cols 2 (E=5), 6 (E=6).
    return np.array([[0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]])


def _diagonal_grid() -> np.ndarray:
    return np.array([[0.0, 9.0, 9.0], [9.0, 1.0, 9.0], [9.0, 9.0, 0.0]])


# -------- output contract ---------------------------------------------------


def test_returns_index_matrix_and_profile():
    idx, prof = find_minimum_energy_path(_chain_grid(), (0, 0), (0, 8))
    assert idx.dtype == np.int64 and idx.shape == (9, 2)
    assert prof.dtype == np.float64 and prof.shape == (9,)
    np.testing.assert_array_equal(idx[:, 1], np.arange(9))
    np.testing.assert_array_equal(prof, _chain_grid()[0])


def test_endpoints_are_first_and_last_rows():
    idx, _prof = find_minimum_energy_path(_chain_grid(), (0, 0), (0, 8))
    assert tuple(idx[0]) == (0, 0)
    assert tuple(idx[-1]) == (0, 8)


def test_start_equals_end_is_single_row():
    idx, prof = find_minimum_energy_path(_chain_grid(), (0, 4), (0, 4))
    assert idx.shape == (1, 2) and tuple(idx[0]) == (0, 4)
    assert prof[0] == 1.0


def test_path_dips_to_intermediate_basin_minimum():
    _idx, prof = find_minimum_energy_path(_chain_grid(), (0, 0), (0, 8))
    assert 1.0 in prof  # basin floor at col 4 is visited


# -------- consistency with find_iwf_grid ------------------------------------


@pytest.mark.parametrize("neighborhood", ["von_neumann", "moore"])
def test_highest_profile_energy_equals_iwf_saddle(neighborhood):
    rng = np.random.default_rng(42)
    e = rng.random((6, 7, 5))
    start, end = (0, 0, 0), (5, 6, 4)
    _idx, prof = find_minimum_energy_path(e, start, end, neighborhood=neighborhood)
    saddle = find_iwf_grid(e, start, end, neighborhood=neighborhood)
    assert saddle is not None
    assert prof.max() == saddle[1]


# -------- step adjacency per stencil -----------------------------------------


def test_von_neumann_steps_change_one_axis_by_one():
    rng = np.random.default_rng(7)
    e = rng.random((6, 7, 5))
    idx, _prof = find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4))
    steps = np.abs(np.diff(idx, axis=0))
    assert (steps.sum(axis=1) == 1).all()


def test_moore_steps_are_chebyshev_one():
    rng = np.random.default_rng(7)
    e = rng.random((6, 7, 5))
    idx, _prof = find_minimum_energy_path(
        e, (0, 0, 0), (5, 6, 4), neighborhood="moore"
    )
    steps = np.abs(np.diff(idx, axis=0))
    assert (steps.max(axis=1) == 1).all()
    assert (steps.sum(axis=1) >= 1).all()


def test_moore_crosses_diagonal_channel_von_neumann_does_not():
    e = _diagonal_grid()
    _i, prof_vn = find_minimum_energy_path(e, (0, 0), (2, 2))
    assert prof_vn.max() == 9.0
    _i, prof_m = find_minimum_energy_path(e, (0, 0), (2, 2), neighborhood="moore")
    assert prof_m.max() == 1.0


# -------- neighborhood kwarg on the other flood kernels ----------------------


def test_iwf_grid_accepts_neighborhood():
    e = _diagonal_grid()
    assert find_iwf_grid(e, (0, 0), (2, 2))[1] == 9.0
    assert find_iwf_grid(e, (0, 0), (2, 2), neighborhood="moore")[1] == 1.0


def test_watershed_accepts_neighborhood():
    e = _diagonal_grid()
    _l, basins_vn, _m = find_watershed_segmentation(e)
    _l, basins_m, merges_m = find_watershed_segmentation(e, neighborhood="moore")
    assert len(basins_vn) == 3
    assert len(basins_m) == 2
    assert merges_m[0][1] == 1.0


# -------- error handling ------------------------------------------------------


def test_disconnected_returns_none():
    nan = float("nan")
    e = np.array([[0.0, nan, 1.0], [0.5, nan, 0.5], [1.0, nan, 0.0]])
    assert find_minimum_energy_path(e, (0, 0), (2, 2)) is None


def test_nan_endpoint_raises():
    e = np.array([[np.nan, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="NaN"):
        find_minimum_energy_path(e, (0, 0), (1, 1))


def test_unknown_neighborhood_raises():
    with pytest.raises(ValueError, match="neighborhood"):
        find_minimum_energy_path(_chain_grid(), (0, 0), (0, 8), neighborhood="king")
    with pytest.raises(ValueError, match="neighborhood"):
        find_iwf_grid(_chain_grid(), (0, 0), (0, 8), neighborhood="king")
    with pytest.raises(ValueError, match="neighborhood"):
        find_watershed_segmentation(_chain_grid(), neighborhood="king")


def test_non_contiguous_raises():
    e = np.asarray(np.random.default_rng(0).random((4, 5)).T)
    assert not e.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        find_minimum_energy_path(e, (0, 0), (4, 3))


# -------- analyze_path_profile (pure Python) ---------------------------------

from pes_analyzer.topology import PathProfile, analyze_path_profile  # noqa: E402


def test_profile_alternating_extrema():
    prof = analyze_path_profile(np.array([0.0, 2.0, 1.0, 3.0, -1.0]))
    assert prof.minima == [(0, 0.0), (2, 1.0), (4, -1.0)]
    assert prof.saddles == [(1, 2.0), (3, 3.0)]


def test_profile_persistence_prunes_shallow_pair():
    prof = analyze_path_profile(
        np.array([0.0, 2.0, 1.0, 3.0, -1.0]), min_persistence=1.5
    )
    assert prof.minima == [(0, 0.0), (4, -1.0)]
    assert prof.saddles == [(3, 3.0)]


def test_profile_flat_is_single_minimum():
    prof = analyze_path_profile(np.array([1.0, 1.0, 1.0]))
    assert prof.minima == [(0, 1.0)]
    assert prof.saddles == []


def test_profile_single_point():
    prof = analyze_path_profile(np.array([5.0]))
    assert prof.minima == [(0, 5.0)]
    assert prof.saddles == []


def test_profile_plateau_extremum_uses_last_plateau_cell():
    prof = analyze_path_profile(np.array([0.0, 2.0, 2.0, 1.0, 3.0]))
    assert prof.saddles[0] == (2, 2.0)


def test_profile_global_minimum_survives_aggressive_pruning():
    prof = analyze_path_profile(
        np.array([0.0, 0.2, -5.0, 0.1, 0.05]), min_persistence=10.0
    )
    assert prof.minima == [(2, -5.0)]


def test_profile_monotone_rise_keeps_start_minimum():
    prof = analyze_path_profile(np.array([0.0, 1.0, 2.0, 3.0]))
    assert prof.minima == [(0, 0.0)]
    assert prof.saddles == [(3, 3.0)]


def test_profile_rejects_nan_and_empty():
    with pytest.raises(ValueError):
        analyze_path_profile(np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        analyze_path_profile(np.array([]))
