"""Flood state: parents, merge_table, ownership, fingerprint (compiled extension)."""
from __future__ import annotations

import numpy as np
import pytest

from pes_analyzer.topology import Watershed, energy_fingerprint, find_watershed_segmentation


def _chain_grid() -> np.ndarray:
    return np.array([[0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]])


def test_parents_recorded_on_request():
    e = _chain_grid()
    ws = find_watershed_segmentation(e, parents=True)
    assert ws.parents is not None
    assert ws.parents.dtype == np.uint16 and ws.parents.shape == e.shape
    assert not ws.parents.flags.writeable
    seeds = [idx for idx, _ in ws.basins]
    for s in seeds:
        assert ws.parents[s] == np.iinfo(np.uint16).max
    # Non-seed cells carry a real code (< 3^ndim).
    non_seed = [(0, c) for c in range(9) if (0, c) not in seeds]
    assert all(ws.parents[p] < 9 for p in non_seed)


def test_merge_table_matches_merges():
    e = _chain_grid()
    ws = find_watershed_segmentation(e)
    strides = np.array([9, 1])
    for (idx, _e, deeper, shallower), row in zip(ws.merges, ws.merge_table):
        saddle_lin, other_lin, d, s, side = (int(x) for x in row)
        assert saddle_lin == int(np.dot(idx, strides))
        assert (d, s) == (deeper, shallower)
        assert side in (d, s)
        assert abs(other_lin - saddle_lin) == 1        # von Neumann neighbour on a 1x9 chain
        assert ws.labels.ravel()[other_lin] != side     # other side of the saddle


def test_drop_labels_frees_all_grid_arrays():
    ws = find_watershed_segmentation(_chain_grid(), parents=True)
    assert ws.has_labels
    ws.drop_labels()
    assert ws.labels is None and ws.parents is None and ws.merge_table is None
    assert not ws.has_labels
    assert len(ws.basins) == 3 and len(ws.merges) == 2   # tree data survives


def test_fingerprint_depends_on_values_shape_and_dtype():
    e = _chain_grid()
    f = energy_fingerprint(e)
    assert f == energy_fingerprint(e.copy())
    assert f != energy_fingerprint(e + 1.0)
    assert f != energy_fingerprint(e.astype(np.float32))
    assert f != energy_fingerprint(e.reshape(3, 3))
    assert find_watershed_segmentation(e).fingerprint == f


def test_fingerprint_subsamples_large_arrays_consistently():
    big = np.random.default_rng(0).random((64, 64, 64, 8))   # 2^21 cells -> k = 2
    assert energy_fingerprint(big) == energy_fingerprint(np.ascontiguousarray(big))
    changed = big.copy()
    changed.reshape(-1)[0] += 1.0                             # sampled cell
    assert energy_fingerprint(changed) != energy_fingerprint(big)


# -------- find_minimum_energy_path(tree=) -----------------------------------

from pes_analyzer.topology import MergeTree, find_minimum_energy_path


def _random_grid(seed=3, shape=(6, 7, 5)):
    return np.random.default_rng(seed).random(shape)


@pytest.mark.parametrize("neighborhood", ["von_neumann", "moore"])
def test_mep_with_tree_matches_standalone(neighborhood):
    e = _random_grid()
    tree = MergeTree(find_watershed_segmentation(e, neighborhood=neighborhood, parents=True))
    for start, end in [((0, 0, 0), (5, 6, 4)), ((2, 3, 1), (3, 3, 1)), ((5, 0, 4), (0, 6, 0)), ((1, 1, 1), (1, 1, 1))]:
        ref = find_minimum_energy_path(e, start, end, neighborhood=neighborhood)
        got = find_minimum_energy_path(e, start, end, tree=tree)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])


def test_mep_with_tree_accepts_watershed_and_explicit_matching_neighborhood():
    e = _random_grid()
    ws = find_watershed_segmentation(e, neighborhood="moore", parents=True)
    a = find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=ws)
    b = find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), neighborhood="moore", tree=MergeTree(ws))
    np.testing.assert_array_equal(a[0], b[0])


def test_mep_with_tree_disconnected_returns_none_and_nan_endpoint_raises():
    nan = float("nan")
    e = np.array([[0.0, nan, 1.0], [0.5, nan, 0.5], [1.0, nan, 0.0]])
    tree = MergeTree(find_watershed_segmentation(e, parents=True))
    assert find_minimum_energy_path(e, (0, 0), (2, 2), tree=tree) is None
    with pytest.raises(ValueError, match="NaN"):
        find_minimum_energy_path(e, (0, 1), (2, 2), tree=tree)


def test_mep_with_tree_rejects_mismatches():
    e = _random_grid()
    no_parents = MergeTree(find_watershed_segmentation(e))
    with pytest.raises(ValueError, match="parents=True"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=no_parents)

    tree = MergeTree(find_watershed_segmentation(e, parents=True))
    with pytest.raises(ValueError, match="neighborhood"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), neighborhood="moore", tree=tree)
    with pytest.raises(ValueError, match="different energy grid"):
        find_minimum_energy_path(e + 1.0, (0, 0, 0), (5, 6, 4), tree=tree)
    with pytest.raises(ValueError, match="different energy grid"):
        find_minimum_energy_path(e.astype(np.float32), (0, 0, 0), (5, 6, 4), tree=tree)
    with pytest.raises(ValueError, match="shape"):
        find_minimum_energy_path(e.reshape(7, 6, 5), (0, 0, 0), (5, 5, 4), tree=tree)

    tree.drop_labels()
    with pytest.raises(ValueError, match="dropped"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=tree)


def test_mep_with_tree_rejects_corrupt_arrays_without_crashing():
    e = _random_grid()
    ws = find_watershed_segmentation(e, parents=True)
    bad = ws.merge_table.copy()
    bad[:, 2] = 10**6
    ws.merge_table = bad
    with pytest.raises(ValueError):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=ws)
    ws = find_watershed_segmentation(e, parents=True)
    ws.parents = ws.parents[:, :, :4].copy()
    with pytest.raises(ValueError, match="shape"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=ws)
    ws = find_watershed_segmentation(e, parents=True)
    ws.labels = ws.labels.astype(np.int64)                 # wrong dtype -> ValueError, not TypeError
    with pytest.raises(ValueError, match="int32"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=ws)
    ws = find_watershed_segmentation(e, parents=True)
    ws.merge_table = ws.merge_table[:, :4].copy()           # wrong width -> ValueError
    with pytest.raises(ValueError, match=r"\(M, 5\)"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), tree=ws)


def test_mep_rejects_empty_neighborhood_in_both_modes():
    e = _random_grid()
    with pytest.raises(ValueError, match="neighborhood"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), neighborhood="")
    tree = MergeTree(find_watershed_segmentation(e, parents=True))
    with pytest.raises(ValueError, match="neighborhood"):
        find_minimum_energy_path(e, (0, 0, 0), (5, 6, 4), neighborhood="", tree=tree)


def test_mep_endpoint_errors_match_standalone_mode():
    # Endpoint validation goes through common::validate in both modes:
    # IndexError for out-of-range / negative, ValueError for a wrong length.
    e = _random_grid()
    tree = MergeTree(find_watershed_segmentation(e, parents=True))
    for bad, exc in [((9, 0, 0), IndexError), ((-1, 0, 0), IndexError), ((0, 0), ValueError)]:
        with pytest.raises(exc):
            find_minimum_energy_path(e, bad, (5, 6, 4))
        with pytest.raises(exc):
            find_minimum_energy_path(e, bad, (5, 6, 4), tree=tree)
