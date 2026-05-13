"""Regression test against a pinned real-data grid.

When ``tests/data/regression_5d.npz`` is present, this test loads the grid
plus the expected saddle (index + energy) from the same file and asserts
that ``find_iwf_grid`` reproduces them. Until the data is provided, the
test is skipped.

The .npz file must contain:
    grid:           float64 N-D array, C-contiguous
    start:          int64 1-D array of length ndim
    end:            int64 1-D array of length ndim
    expected_index: int64 1-D array of length ndim
    expected_energy: float64 scalar
"""

from pathlib import Path

import numpy as np
import pytest

from pes_analyzer.saddle import find_iwf_grid

DATA_PATH = Path(__file__).parent / "data" / "regression_5d.npz"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="regression data not present")
def test_regression_5d():
    npz = np.load(DATA_PATH)
    grid = np.ascontiguousarray(npz["grid"], dtype=np.float64)
    start = tuple(int(x) for x in npz["start"])
    end = tuple(int(x) for x in npz["end"])
    expected_idx = tuple(int(x) for x in npz["expected_index"])
    expected_energy = float(npz["expected_energy"])

    result = find_iwf_grid(grid, start, end)
    assert result is not None, "regression saddle search returned None"
    idx, energy = result
    assert idx == expected_idx, f"saddle index drifted: {idx} != {expected_idx}"
    assert energy == pytest.approx(expected_energy, rel=0, abs=1e-12), (
        f"saddle energy drifted: {energy} != {expected_energy}"
    )
