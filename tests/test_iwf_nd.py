"""Multi-dimensional + edge-case integration tests for find_iwf_grid."""

import numpy as np
import pytest

from pes_analyzer.saddle import find_iwf_grid


def test_start_equals_end_returns_start_immediately():
    e = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = find_iwf_grid(e, (1, 1), (1, 1))
    assert result == ((1, 1), 4.0)


def test_adjacent_cells_2d():
    # Cheapest bridge between (0,0) and (0,2) on a 2x3 grid is (0,1).
    e = np.array(
        [[0.0, 5.0, 0.0],
         [10.0, 10.0, 10.0]],
        dtype=np.float64,
    )
    result = find_iwf_grid(e, (0, 0), (0, 2))
    assert result is not None
    idx, energy = result
    assert energy == 5.0
    assert idx == (0, 1)


def test_nan_wall_blocks_connection_returns_none():
    # 3x5 grid; middle column is NaN.
    e = np.array(
        [[0.0, 1.0, np.nan, 1.0, 0.0],
         [1.0, 2.0, np.nan, 2.0, 1.0],
         [0.0, 1.0, np.nan, 1.0, 0.0]],
        dtype=np.float64,
    )
    result = find_iwf_grid(e, (1, 0), (1, 4))
    assert result is None


def test_three_minima_3d_two_distinct_saddles():
    # 3-D grid with three minima M1, M2, M3 arranged along axis 0.
    n = 9
    e = np.full((n, 3, 3), 10.0, dtype=np.float64)
    e[0, 1, 1] = 0.0  # M1
    e[4, 1, 1] = 0.0  # M2
    e[8, 1, 1] = 0.0  # M3
    # Bridge M1-M2 has its max at i=2 with energy 3.0
    for i in range(1, 4):
        e[i, 1, 1] = 3.0
    # Bridge M2-M3 has its max at i=6 with energy 7.0
    for i in range(5, 8):
        e[i, 1, 1] = 7.0

    r12 = find_iwf_grid(e, (0, 1, 1), (4, 1, 1))
    assert r12 is not None
    assert r12[1] == 3.0

    r23 = find_iwf_grid(e, (4, 1, 1), (8, 1, 1))
    assert r23 is not None
    assert r23[1] == 7.0


@pytest.mark.parametrize("ndim", [2, 3, 4, 5, 6, 7])
def test_dimensionality_sweep(ndim: int):
    """For each supported ndim, build a grid with shape (3, 1, 1, ..., 1)
    where the saddle along axis 0 has a known energy.
    """
    shape = [1] * ndim
    shape[0] = 3
    total = int(np.prod(shape))
    data = np.full(total, 7.0, dtype=np.float64)
    data[0] = 0.0
    data[-1] = 0.0
    e = data.reshape(shape)

    start = tuple([0] * ndim)
    end = tuple([2] + [0] * (ndim - 1))
    result = find_iwf_grid(e, start, end)
    assert result is not None, f"ndim={ndim}"
    _, energy = result
    assert energy == 7.0, f"ndim={ndim}, energy={energy}"
