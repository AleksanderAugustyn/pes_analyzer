"""Integration tests for pes_analyzer.minimum.find_minima_grid.

The cargo unit tests already cover the core algorithm; these tests
verify the Python-facing contract (dtype, contiguity, ndim, error
shapes) and a couple of multi-D shape sanity checks against the Rust
implementation as a black box.
"""

import numpy as np
import pytest

from pes_analyzer.minimum import find_minima_grid


def test_single_bowl_2d():
    e = np.array(
        [[1.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    result = find_minima_grid(e)
    assert result == [((1, 1), 0.0)]


def test_two_basins_separated_by_nan_wall():
    nan = np.nan
    e = np.array(
        [[0.0, 1.0, nan, 1.0, 0.0],
         [1.0, 2.0, nan, 2.0, 1.0],
         [1.0, 1.0, nan, 1.0, 1.0]],
        dtype=np.float64,
    )
    result = find_minima_grid(e)
    coords = [idx for idx, _ in result]
    # Two known minima at (0, 0) and (0, 4).
    assert (0, 0) in coords
    assert (0, 4) in coords


def test_output_is_sorted_ascending_by_energy():
    e = np.full((5, 5), 10.0, dtype=np.float64)
    e[0, 0] = 2.0
    e[4, 4] = 1.0
    result = find_minima_grid(e)
    energies = [en for _, en in result]
    assert energies == sorted(energies)


def test_result_indices_are_int_tuples_not_numpy_scalars():
    e = np.array(
        [[1.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    [(idx, _)] = find_minima_grid(e)
    assert isinstance(idx, tuple)
    for component in idx:
        assert isinstance(component, int)
        assert not isinstance(component, np.integer)


def test_5d_minimum_at_center():
    shape = (3, 3, 3, 3, 3)
    e = np.full(shape, 1.0, dtype=np.float64)
    e[1, 1, 1, 1, 1] = 0.0
    result = find_minima_grid(e)
    assert result == [((1, 1, 1, 1, 1), 0.0)]


def test_rejects_ndim_1():
    e = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"ndim must be in \[2, 7\]"):
        find_minima_grid(e)


def test_rejects_ndim_8():
    e = np.zeros((2,) * 8, dtype=np.float64)
    with pytest.raises(ValueError, match=r"ndim must be in \[2, 7\]"):
        find_minima_grid(e)


def test_rejects_non_contiguous():
    e = np.zeros((5, 5), dtype=np.float64)
    sliced = e[::2, :]  # not C-contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        find_minima_grid(sliced)


def test_rejects_non_float64():
    e = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        find_minima_grid(e)


def test_diagonal_neighbor_disqualifies_minimum():
    """3^N - 1 connectivity: a cell with a strictly-lower diagonal
    neighbor is NOT a local minimum, even if it beats its axis neighbors.
    """
    e = np.array(
        [[0.0, 10.0, 10.0],
         [10.0, 5.0, 10.0],
         [10.0, 10.0, 10.0]],
        dtype=np.float64,
    )
    result = find_minima_grid(e)
    coords = [idx for idx, _ in result]
    assert (1, 1) not in coords
    assert (0, 0) in coords
