"""Integration tests for pes_analyzer.extrema.find_minima_grid.

The cargo unit tests already cover the core algorithm; these tests
verify the Python-facing contract (dtype, contiguity, ndim, error
shapes) and a couple of multi-D shape sanity checks against the Rust
implementation as a black box.
"""

import numpy as np
import pytest

from pes_analyzer.extrema import find_minima_grid, find_maxima_grid, find_extrema_grid


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


def test_accepts_float32():
    # float32 is now a first-class energy dtype (see test_dtype.py).
    e = np.zeros((3, 3), dtype=np.float32)
    find_minima_grid(e)  # must not raise


def test_rejects_non_float_dtype():
    e = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
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


def test_default_neighborhood_range_matches_r1_explicit():
    """The default argument value (r=1) must be byte-identical to passing it explicitly."""
    e = np.array(
        [[5.0, 5.0, 5.0, 5.0, 5.0],
         [5.0, 5.0, 5.0, 5.0, 5.0],
         [5.0, 5.0, 3.0, 5.0, 5.0],
         [5.0, 5.0, 5.0, 5.0, 5.0],
         [5.0, 5.0, 5.0, 5.0, 1.0]],
        dtype=np.float64,
    )
    assert find_minima_grid(e) == find_minima_grid(e, neighborhood_range=1)


def test_neighborhood_range_2_disqualifies_far_diagonal_minimum():
    """At r=2, a cell beaten by a Chebyshev-distance-2 diagonal is no longer a minimum.

    The 5x5 grid has (2,2)=3 and (4,4)=1, everything else 5. At r=1 both
    (2,2) and (4,4) are local minima; at r=2 only (4,4) survives because
    (2,2) sees (4,4)=1 in its r=2 stencil.
    """
    e = np.full((5, 5), 5.0, dtype=np.float64)
    e[2, 2] = 3.0
    e[4, 4] = 1.0

    coords_r1 = [idx for idx, _ in find_minima_grid(e, neighborhood_range=1)]
    assert (2, 2) in coords_r1
    assert (4, 4) in coords_r1

    coords_r2 = [idx for idx, _ in find_minima_grid(e, neighborhood_range=2)]
    assert (2, 2) not in coords_r2
    assert (4, 4) in coords_r2


def test_rejects_neighborhood_range_0():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"neighborhood_range must be in \[1, 5\]"):
        find_minima_grid(e, neighborhood_range=0)


def test_rejects_neighborhood_range_6():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"neighborhood_range must be in \[1, 5\]"):
        find_minima_grid(e, neighborhood_range=6)


def test_neighborhood_range_must_be_keyword_only():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        find_minima_grid(e, 2)  # type: ignore[misc]


def test_neighborhood_range_negative_raises():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(OverflowError):
        find_minima_grid(e, neighborhood_range=-1)


def test_confirm_range_default_unchanged():
    """Omitting confirm_range must produce the same result as before the parameter existed."""
    e = np.full((5, 5), 5.0, dtype=np.float64)
    e[2, 2] = 3.0
    e[4, 4] = 1.0
    baseline = find_minima_grid(e, neighborhood_range=1)
    assert find_minima_grid(e) == baseline
    assert find_minima_grid(e, neighborhood_range=1, confirm_range=None) == baseline


def test_confirm_range_equivalent_to_wider_neighborhood():
    """find at r=1 + confirm at r=2 must equal direct check at r=2."""
    e = np.full((5, 5), 5.0, dtype=np.float64)
    e[2, 2] = 3.0
    e[4, 4] = 1.0
    two_pass = find_minima_grid(e, neighborhood_range=1, confirm_range=2)
    direct = find_minima_grid(e, neighborhood_range=2)
    assert two_pass == direct


def test_confirm_range_validation_errors():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"confirm_range must be in \[1, 5\]"):
        find_minima_grid(e, confirm_range=0)
    with pytest.raises(ValueError, match=r"confirm_range must be in \[1, 5\]"):
        find_minima_grid(e, confirm_range=6)
    with pytest.raises(
        ValueError,
        match=r"confirm_range \(1\) must be >= neighborhood_range \(2\)",
    ):
        find_minima_grid(e, neighborhood_range=2, confirm_range=1)


def test_confirm_range_must_be_keyword_only():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        find_minima_grid(e, 1, 2)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# find_maxima_grid
# ---------------------------------------------------------------------------


def test_maxima_single_peak_2d():
    e = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = find_maxima_grid(e)
    assert result == [((1, 1), 1.0)]


def test_maxima_two_peaks_separated_by_nan_wall():
    nan = np.nan
    e = np.array(
        [[5.0, 4.0, nan, 4.0, 5.0],
         [4.0, 3.0, nan, 3.0, 4.0],
         [4.0, 4.0, nan, 4.0, 4.0]],
        dtype=np.float64,
    )
    result = find_maxima_grid(e)
    coords = [idx for idx, _ in result]
    assert (0, 0) in coords
    assert (0, 4) in coords


def test_maxima_output_is_sorted_descending_by_energy():
    e = np.full((5, 5), -10.0, dtype=np.float64)
    e[0, 0] = -2.0
    e[4, 4] = -1.0
    result = find_maxima_grid(e)
    energies = [en for _, en in result]
    assert energies == sorted(energies, reverse=True)


def test_maxima_result_indices_are_int_tuples_not_numpy_scalars():
    e = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    [(idx, _)] = find_maxima_grid(e)
    assert isinstance(idx, tuple)
    for component in idx:
        assert isinstance(component, int)
        assert not isinstance(component, np.integer)


def test_maxima_5d_at_center():
    shape = (3, 3, 3, 3, 3)
    e = np.full(shape, 0.0, dtype=np.float64)
    e[1, 1, 1, 1, 1] = 1.0
    result = find_maxima_grid(e)
    assert result == [((1, 1, 1, 1, 1), 1.0)]


def test_maxima_rejects_ndim_1():
    e = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"ndim must be in \[2, 7\]"):
        find_maxima_grid(e)


def test_maxima_rejects_ndim_8():
    e = np.zeros((2,) * 8, dtype=np.float64)
    with pytest.raises(ValueError, match=r"ndim must be in \[2, 7\]"):
        find_maxima_grid(e)


def test_maxima_rejects_non_contiguous():
    e = np.zeros((5, 5), dtype=np.float64)
    sliced = e[::2, :]
    with pytest.raises(ValueError, match="C-contiguous"):
        find_maxima_grid(sliced)


def test_maxima_accepts_float32():
    # float32 is now a first-class energy dtype (see test_dtype.py).
    e = np.zeros((3, 3), dtype=np.float32)
    find_maxima_grid(e)  # must not raise


def test_maxima_rejects_non_float_dtype():
    e = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        find_maxima_grid(e)


def test_maxima_diagonal_neighbor_disqualifies_maximum():
    e = np.array(
        [[10.0, 0.0, 0.0],
         [0.0, 5.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = find_maxima_grid(e)
    coords = [idx for idx, _ in result]
    assert (1, 1) not in coords
    assert (0, 0) in coords


def test_maxima_neighborhood_range_2_disqualifies_far_diagonal_maximum():
    e = np.full((5, 5), -5.0, dtype=np.float64)
    e[2, 2] = -3.0
    e[4, 4] = -1.0

    coords_r1 = [idx for idx, _ in find_maxima_grid(e, neighborhood_range=1)]
    assert (2, 2) in coords_r1
    assert (4, 4) in coords_r1

    coords_r2 = [idx for idx, _ in find_maxima_grid(e, neighborhood_range=2)]
    assert (2, 2) not in coords_r2
    assert (4, 4) in coords_r2


def test_maxima_rejects_neighborhood_range_0():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"neighborhood_range must be in \[1, 5\]"):
        find_maxima_grid(e, neighborhood_range=0)


def test_maxima_rejects_neighborhood_range_6():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"neighborhood_range must be in \[1, 5\]"):
        find_maxima_grid(e, neighborhood_range=6)


def test_maxima_neighborhood_range_must_be_keyword_only():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        find_maxima_grid(e, 2)  # type: ignore[misc]


def test_maxima_confirm_range_equivalent_to_wider_neighborhood():
    e = np.full((5, 5), -5.0, dtype=np.float64)
    e[2, 2] = -3.0
    e[4, 4] = -1.0
    two_pass = find_maxima_grid(e, neighborhood_range=1, confirm_range=2)
    direct = find_maxima_grid(e, neighborhood_range=2)
    assert two_pass == direct


def test_maxima_confirm_range_validation_errors():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"confirm_range must be in \[1, 5\]"):
        find_maxima_grid(e, confirm_range=0)
    with pytest.raises(ValueError, match=r"confirm_range must be in \[1, 5\]"):
        find_maxima_grid(e, confirm_range=6)
    with pytest.raises(
        ValueError,
        match=r"confirm_range \(1\) must be >= neighborhood_range \(2\)",
    ):
        find_maxima_grid(e, neighborhood_range=2, confirm_range=1)


# ---------------------------------------------------------------------------
# find_extrema_grid
# ---------------------------------------------------------------------------


def test_extrema_returns_tuple_of_two_lists():
    e = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, 0.5, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    result = find_extrema_grid(e)
    assert isinstance(result, tuple)
    assert len(result) == 2
    mins, maxs = result
    assert isinstance(mins, list)
    assert isinstance(maxs, list)


def test_extrema_equivalent_to_separate_calls_simple():
    e = np.array(
        [[1.0, 2.0, 1.0],
         [2.0, 0.0, 2.0],
         [1.0, 2.0, 1.0]],
        dtype=np.float64,
    )
    combined = find_extrema_grid(e)
    separate = (find_minima_grid(e), find_maxima_grid(e))
    assert combined == separate


def test_extrema_equivalent_to_separate_calls_with_nan_walls():
    nan = np.nan
    e = np.array(
        [[5.0, 5.0, 5.0, 5.0, 5.0],
         [5.0, 3.0, nan, 1.0, 5.0],
         [5.0, 5.0, nan, 5.0, 5.0],
         [5.0, 5.0, nan, 5.0, 5.0],
         [5.0, 5.0, 5.0, 5.0, 5.0]],
        dtype=np.float64,
    )
    for r in (1, 2, 3):
        for c in (None, 2, 3, 4):
            if c is not None and c < r:
                continue
            combined = find_extrema_grid(e, neighborhood_range=r, confirm_range=c)
            separate = (
                find_minima_grid(e, neighborhood_range=r, confirm_range=c),
                find_maxima_grid(e, neighborhood_range=r, confirm_range=c),
            )
            assert combined == separate, f"mismatch at r={r}, c={c}"


def test_extrema_minima_ascending_maxima_descending():
    e = np.array(
        [[1.0, 5.0, 1.0, 6.0, 1.0],
         [5.0, 0.0, 4.0, 0.5, 5.0],
         [1.0, 5.0, 1.0, 5.0, 1.0]],
        dtype=np.float64,
    )
    mins, maxs = find_extrema_grid(e)
    min_energies = [en for _, en in mins]
    max_energies = [en for _, en in maxs]
    assert min_energies == sorted(min_energies)
    assert max_energies == sorted(max_energies, reverse=True)


def test_extrema_all_nan_stencil_disqualifies_both():
    nan = np.nan
    e = np.array(
        [[nan, nan, nan],
         [nan, 0.0, nan],
         [nan, nan, nan]],
        dtype=np.float64,
    )
    mins, maxs = find_extrema_grid(e)
    assert mins == []
    assert maxs == []


def test_extrema_rejects_non_contiguous():
    e = np.zeros((5, 5), dtype=np.float64)
    sliced = e[::2, :]
    with pytest.raises(ValueError, match="C-contiguous"):
        find_extrema_grid(sliced)


def test_extrema_rejects_ndim_1():
    e = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"ndim must be in \[2, 7\]"):
        find_extrema_grid(e)


def test_extrema_confirm_range_validation_errors():
    e = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"confirm_range must be in \[1, 5\]"):
        find_extrema_grid(e, confirm_range=0)
    with pytest.raises(
        ValueError,
        match=r"confirm_range \(1\) must be >= neighborhood_range \(2\)",
    ):
        find_extrema_grid(e, neighborhood_range=2, confirm_range=1)
