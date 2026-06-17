"""Tests for the input-validation error contract of the public PyO3 wrappers."""

import numpy as np
import pytest

from pes_analyzer.saddle import find_iwf_grid


def _ok_grid_2d() -> np.ndarray:
    return np.zeros((4, 4), dtype=np.float64)


def test_float32_is_accepted():
    # float32 is now a first-class energy dtype (see test_dtype.py).
    e = np.zeros((4, 4), dtype=np.float32)
    assert find_iwf_grid(e, (0, 0), (3, 3)) is not None


def test_non_float_dtype_raises_valueerror():
    e = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        find_iwf_grid(e, (0, 0), (3, 3))


def test_ndim_too_low_raises_valueerror():
    e = np.zeros(10, dtype=np.float64)  # 1-D
    with pytest.raises(ValueError, match="ndim"):
        find_iwf_grid(e, (0,), (9,))


def test_ndim_too_high_raises_valueerror():
    e = np.zeros((2,) * 8, dtype=np.float64)  # 8-D
    with pytest.raises(ValueError, match="ndim"):
        find_iwf_grid(e, (0,) * 8, (1,) * 8)


def test_non_c_contiguous_raises_valueerror():
    base = np.zeros((4, 4), dtype=np.float64)
    e = base.T  # F-contiguous, not C-contiguous
    assert not e.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        find_iwf_grid(e, (0, 0), (3, 3))


def test_index_length_mismatch_raises_valueerror():
    e = _ok_grid_2d()
    with pytest.raises(ValueError, match="length"):
        find_iwf_grid(e, (0,), (3, 3))
    with pytest.raises(ValueError, match="length"):
        find_iwf_grid(e, (0, 0), (3,))


def test_index_out_of_bounds_raises_indexerror():
    e = _ok_grid_2d()
    with pytest.raises(IndexError):
        find_iwf_grid(e, (4, 0), (3, 3))
    with pytest.raises(IndexError):
        find_iwf_grid(e, (0, 0), (0, 4))


def test_negative_index_raises_indexerror():
    e = _ok_grid_2d()
    with pytest.raises(IndexError):
        find_iwf_grid(e, (-1, 0), (3, 3))


def test_nan_at_start_raises_valueerror():
    e = _ok_grid_2d()
    e[0, 0] = np.nan
    with pytest.raises(ValueError, match="start"):
        find_iwf_grid(e, (0, 0), (3, 3))


def test_nan_at_end_raises_valueerror():
    e = _ok_grid_2d()
    e[3, 3] = np.nan
    with pytest.raises(ValueError, match="end"):
        find_iwf_grid(e, (0, 0), (3, 3))
