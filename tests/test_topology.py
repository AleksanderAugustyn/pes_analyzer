"""Integration tests for ``pes_analyzer.topology``.

Run against the compiled extension; require ``maturin develop`` to be
current.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pes_analyzer.topology import (
    Watershed,
    compute_persistence,
    find_watershed_segmentation,
    prune_merge_tree,
)


# -------- find_watershed_segmentation --------------------------------------


def test_segmentation_returns_watershed_object():
    energies = np.array([[5.0, 4.0, 5.0], [4.0, 0.0, 4.0], [5.0, 4.0, 5.0]])
    ws = find_watershed_segmentation(energies)
    assert isinstance(ws, Watershed)
    assert ws.labels.dtype == np.int32 and ws.labels.shape == energies.shape
    assert ws.parents is None                      # off by default
    assert ws.neighborhood == "von_neumann"
    assert ws.dtype == np.dtype("float64")
    assert isinstance(ws.fingerprint, bytes) and len(ws.fingerprint) == 16
    assert all(isinstance(b, tuple) and len(b) == 2 for b in ws.basins)
    assert all(isinstance(b[0], tuple) and isinstance(b[1], float) for b in ws.basins)
    assert all(isinstance(m, tuple) and len(m) == 4 for m in ws.merges)
    assert ws.merge_table.dtype == np.uint32 and ws.merge_table.shape == (len(ws.merges), 5)
    assert not ws.labels.flags.writeable
    assert not ws.merge_table.flags.writeable


def test_segmentation_known_2d_grid():
    # 5x5 grid: two minima at (0,0)=0 and (4,4)=0; bridge row 2 connects them;
    # saddle at (2,2)=4. Mirrors the Rust two_basins_separated_by_ridge_2d test.
    e = np.full((5, 5), 10.0)
    e[0, 0], e[0, 1] = 0.0, 1.0
    e[1, 0], e[1, 1] = 1.0, 2.0
    e[2, 0], e[2, 1], e[2, 2], e[2, 3], e[2, 4] = 3.0, 3.0, 4.0, 3.0, 3.0
    e[4, 4], e[3, 4], e[4, 3], e[3, 3] = 0.0, 1.0, 1.0, 2.0
    ws = find_watershed_segmentation(e)
    labels, basins, merges = ws.labels, ws.basins, ws.merges
    assert len(basins) == 2
    assert len(merges) == 1
    saddle_idx, saddle_e, deeper, shallower = merges[0]
    assert saddle_idx == (2, 2)
    assert saddle_e == 4.0
    assert {deeper, shallower} == {0, 1}


# -------- compute_persistence ----------------------------------------------


def test_compute_persistence_deepest_is_inf():
    energies = np.array([[0.0, 5.0, 1.0]])  # two basins, one merge
    ws = find_watershed_segmentation(energies)
    _labels, basins, merges = ws.labels, ws.basins, ws.merges
    persistence = compute_persistence(basins, merges)
    assert math.isinf(persistence[0])


def test_compute_persistence_non_root_is_saddle_minus_min():
    energies = np.array([[0.0, 5.0, 1.0]])
    ws = find_watershed_segmentation(energies)
    _labels, basins, merges = ws.labels, ws.basins, ws.merges
    persistence = compute_persistence(basins, merges)
    saddle_e = merges[0][1]
    shallower = merges[0][3]
    assert persistence[shallower] == pytest.approx(saddle_e - basins[shallower][1])


# -------- prune_merge_tree -------------------------------------------------


def test_prune_drops_low_persistence_basins():
    # 3 basins (1-D chain treated as 1x9) with explicit persistences:
    # basin 0 (E=0) is global; basin 1 (E=1) dies at E=5 -> persistence 4;
    # basin 2 (E=2) dies at E=6 -> persistence 4.
    energies = np.array([[0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]])
    ws = find_watershed_segmentation(energies)
    _labels, basins, merges = ws.labels, ws.basins, ws.merges
    assert len(basins) == 3
    # Threshold above both persistences -> only basin 0 survives.
    surviving, kept = prune_merge_tree(basins, merges, threshold=5.0)
    assert surviving == [0]
    assert kept == []
    # Threshold below both -> all survive.
    surviving, kept = prune_merge_tree(basins, merges, threshold=1.0)
    assert surviving == [0, 1, 2]
    assert len(kept) == 2


# -------- validation -------------------------------------------------------


def test_validation_errors():
    # 1-D input rejected.
    with pytest.raises(ValueError, match=r"ndim must be in \[2, 7\]"):
        find_watershed_segmentation(np.zeros(10, dtype=np.float64))

    # Non-contiguous rejected.
    arr = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        find_watershed_segmentation(arr.T)

    # All-NaN is not an error; returns sentinel result.
    arr = np.full((3, 3), np.nan, dtype=np.float64)
    ws = find_watershed_segmentation(arr)
    labels, basins, merges = ws.labels, ws.basins, ws.merges
    assert basins == []
    assert merges == []
    assert (labels == -1).all()
