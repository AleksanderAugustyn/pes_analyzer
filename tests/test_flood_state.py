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
