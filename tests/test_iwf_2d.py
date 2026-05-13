"""2D saddle-search tests.

Uses synthetic surfaces with hard high-energy walls forcing the watershed
through a known channel. This makes the expected saddle index and energy
exact, not approximate.
"""

import numpy as np
import pytest

from pes_analyzer.saddle import find_iwf_grid


def test_two_wells_separated_by_a_single_pass_2d():
    """5x5 grid. Two wells at (1,1) and (3,3) connected by exactly one
    channel; the highest cell on that channel has energy 4.0.
    """
    wall = 100.0
    e = np.full((5, 5), wall, dtype=np.float64)
    # Two basins
    e[1, 1] = 0.0
    e[3, 3] = 0.0
    # Channel: (1,1) → (1,2)=2 → (2,2)=4 → (3,2)=3 → (3,3)
    e[1, 2] = 2.0
    e[2, 2] = 4.0
    e[3, 2] = 3.0
    result = find_iwf_grid(e, (1, 1), (3, 3))
    assert result is not None
    idx, energy = result
    assert energy == 4.0
    assert idx == (2, 2)


def test_two_channels_picks_lower_pass_2d():
    """7x5 grid with two distinct connecting channels; the algorithm must
    pick the lower-energy one.
    """
    wall = 100.0
    e = np.full((7, 5), wall, dtype=np.float64)
    e[0, 0] = 0.0
    e[6, 0] = 0.0
    # Channel A along column 0: max cell energy = 5.0
    e[1, 0] = 3.0
    e[2, 0] = 5.0
    e[3, 0] = 4.0
    e[4, 0] = 3.0
    e[5, 0] = 2.0
    # Channel B along column 4: max cell energy = 9.0 (worse)
    e[0, 4] = 8.0
    e[1, 4] = 8.0
    e[2, 4] = 9.0
    e[3, 4] = 8.0
    e[4, 4] = 8.0
    e[5, 4] = 8.0
    e[6, 4] = 8.0
    # Add cheap bridges (energy 6) on rows 0 and 6 so channel B is reachable
    # but worse than channel A.
    for j in range(5):
        e[0, j] = min(e[0, j], 6.0)
        e[6, j] = min(e[6, j], 6.0)
    e[0, 0] = 0.0  # restore start
    e[6, 0] = 0.0  # restore end
    # Cheapest path is start → column 0 channel (max=5.0) → end.
    result = find_iwf_grid(e, (0, 0), (6, 0))
    assert result is not None
    _, energy = result
    assert energy == 5.0


def test_returns_tuple_of_python_ints():
    e = np.zeros((3, 3), dtype=np.float64)
    e[1, 1] = 1.0
    result = find_iwf_grid(e, (0, 0), (2, 2))
    assert result is not None
    idx, energy = result
    assert isinstance(idx, tuple), f"index should be a tuple, got {type(idx)}"
    assert all(isinstance(i, int) for i in idx)
    assert isinstance(energy, float)
