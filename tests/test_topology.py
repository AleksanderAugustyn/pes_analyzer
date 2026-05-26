"""Integration tests for ``pes_analyzer.topology``.

Run against the compiled extension; require ``maturin develop`` to be
current.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pes_analyzer.topology import (
    compute_persistence,
    find_watershed_segmentation,
    identify_critical_points,
    prune_merge_tree,
)


# -------- find_watershed_segmentation --------------------------------------


def test_segmentation_returns_expected_types():
    energies = np.array([[5.0, 4.0, 5.0], [4.0, 0.0, 4.0], [5.0, 4.0, 5.0]])
    labels, basins, merges = find_watershed_segmentation(energies)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == energies.shape
    assert isinstance(basins, list)
    assert all(isinstance(b, tuple) and len(b) == 2 for b in basins)
    assert all(isinstance(b[0], tuple) and isinstance(b[1], float) for b in basins)
    assert isinstance(merges, list)
    assert all(isinstance(m, tuple) and len(m) == 4 for m in merges)


def test_segmentation_known_2d_grid():
    # 5x5 grid: two minima at (0,0)=0 and (4,4)=0; bridge row 2 connects them;
    # saddle at (2,2)=4. Mirrors the Rust two_basins_separated_by_ridge_2d test.
    e = np.full((5, 5), 10.0)
    e[0, 0], e[0, 1] = 0.0, 1.0
    e[1, 0], e[1, 1] = 1.0, 2.0
    e[2, 0], e[2, 1], e[2, 2], e[2, 3], e[2, 4] = 3.0, 3.0, 4.0, 3.0, 3.0
    e[4, 4], e[3, 4], e[4, 3], e[3, 3] = 0.0, 1.0, 1.0, 2.0
    labels, basins, merges = find_watershed_segmentation(e)
    assert len(basins) == 2
    assert len(merges) == 1
    saddle_idx, saddle_e, deeper, shallower = merges[0]
    assert saddle_idx == (2, 2)
    assert saddle_e == 4.0
    assert {deeper, shallower} == {0, 1}


# -------- compute_persistence ----------------------------------------------


def test_compute_persistence_deepest_is_inf():
    energies = np.array([[0.0, 5.0, 1.0]])  # two basins, one merge
    _labels, basins, merges = find_watershed_segmentation(energies)
    persistence = compute_persistence(basins, merges)
    assert math.isinf(persistence[0])


def test_compute_persistence_non_root_is_saddle_minus_min():
    energies = np.array([[0.0, 5.0, 1.0]])
    _labels, basins, merges = find_watershed_segmentation(energies)
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
    _labels, basins, merges = find_watershed_segmentation(energies)
    assert len(basins) == 3
    # Threshold above both persistences -> only basin 0 survives.
    surviving, kept = prune_merge_tree(basins, merges, threshold=5.0)
    assert surviving == [0]
    assert kept == []
    # Threshold below both -> all survive.
    surviving, kept = prune_merge_tree(basins, merges, threshold=1.0)
    assert surviving == [0, 1, 2]
    assert len(kept) == 2


# -------- identify_critical_points -----------------------------------------


def test_identify_critical_points_actinide_like():
    # 4 basins on a 1-D chain (shape 1x15):
    #   basin A (E=0) at pos 0  — GS
    #   basin B (E=1) at pos 6  — SM
    #   basin C (E=3) at pos 12 — FE
    #   basin D (E=4) at pos 14 — noise (small persistence)
    # Saddles chosen so B and C have persistence > threshold but D does not.
    energies = np.array(
        [[0.0, 3.0, 6.0, 5.0, 4.0, 3.0, 1.0, 3.0, 5.0, 7.0, 6.0, 5.0, 3.0, 4.5, 4.0]]
    )
    _labels, basins, merges = find_watershed_segmentation(energies)
    cp = identify_critical_points(basins, merges, threshold=2.0)
    assert cp["ground_state"] is not None
    assert cp["secondary_minimum"] is not None
    assert cp["inner_saddle"] is not None
    assert cp["outer_saddle"] is not None
    assert cp["fission_exit"] is not None
    # GS is the deepest basin.
    assert basins[cp["ground_state"]][1] == 0.0


def test_identify_critical_points_outer_lower_than_inner():
    # Linear chain where the outer barrier has LOWER energy than the inner.
    # positions 0..8 with energies:
    #   0:0 (GS) 1:3 2:8 (inner saddle) 3:4 4:1 (SM) 5:3 6:5 (outer saddle) 7:4 8:2 (FE)
    energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
    _labels, basins, merges = find_watershed_segmentation(energies)
    cp = identify_critical_points(basins, merges, threshold=2.0)
    assert cp["ground_state"] == 0           # GS is basin A (E=0)
    assert cp["secondary_minimum"] == 1      # SM is basin B (E=1), not C (E=2)
    assert cp["fission_exit"] == 2           # FE is basin C (E=2)
    assert cp["inner_saddle"][1] == 8.0      # inner saddle energy is 8
    assert cp["outer_saddle"][1] == 5.0      # outer saddle energy is 5 (LOWER)


def test_identify_critical_points_she_like():
    # Two basins only; no fission isomer. Expect single saddle in
    # inner_saddle slot and fission_exit pointing at the outward basin.
    energies = np.array([[0.0, 3.0, 5.0, 3.0, 2.0]])
    _labels, basins, merges = find_watershed_segmentation(energies)
    cp = identify_critical_points(basins, merges, threshold=1.0)
    assert cp["ground_state"] == 0
    assert cp["secondary_minimum"] is None
    assert cp["inner_saddle"] is not None
    assert cp["outer_saddle"] is None
    assert cp["fission_exit"] is not None


def test_gs_disqualifier_falls_back_to_next_basin():
    # Two basins; pass a disqualifier that rejects basin 0.
    energies = np.array([[0.0, 5.0, 1.0]])
    _labels, basins, merges = find_watershed_segmentation(energies)
    cp = identify_critical_points(
        basins, merges, threshold=0.0, gs_disqualifier=lambda bid: bid == 0
    )
    assert cp["ground_state"] == 1


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
    labels, basins, merges = find_watershed_segmentation(arr)
    assert basins == []
    assert merges == []
    assert (labels == -1).all()
