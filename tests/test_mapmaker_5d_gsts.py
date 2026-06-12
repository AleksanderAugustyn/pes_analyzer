"""Unit tests for basin-led MEP classification in MapMaker_FoS_SHE_5D_GSTS.py.

The MapMaker is a standalone script, not a package, so we load it by path
(same pattern as test_mapmaker_fos.py). Python-only: no maturin rebuild needed.
"""
from __future__ import annotations

import importlib.util
import io
import pathlib

import numpy as np
import polars as pl
import pytest

from pes_analyzer.topology import (
    MergeTree,
    PathProfile,
    find_watershed_segmentation,
)

_MM_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mapmaker_test"
    / "5D_GSTS"
    / "MapMaker_FoS_SHE_5D_GSTS.py"
)


@pytest.fixture(scope="module")
def mm():
    spec = importlib.util.spec_from_file_location(
        "mapmaker_fos_she_5d_gsts", _MM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMepRejectReason:
    """Spec section 5.3: checks (a)-(c) are path physics, (d) is basin persistence."""

    LAST = 100  # half-path boundary at k=50

    def test_passes_all_checks(self, mm):
        assert mm._mep_reject_reason(10, 2.0, 0.0, 0.0, self.LAST, 1.0, 2.5) is None

    def test_below_gs_energy(self, mm):
        reason = mm._mep_reject_reason(10, -1.0, 0.0, 0.0, self.LAST, 1.0, 9.0)
        assert "below GS energy" in reason

    def test_below_previous_minimum(self, mm):
        reason = mm._mep_reject_reason(10, 1.5, 2.0, 0.0, self.LAST, 1.0, 9.0)
        assert "below previous" in reason

    def test_beyond_half_path(self, mm):
        reason = mm._mep_reject_reason(51, 2.0, 0.0, 0.0, self.LAST, 1.0, 9.0)
        assert "beyond half-path" in reason

    def test_persistence_below_floor(self, mm):
        reason = mm._mep_reject_reason(10, 2.0, 0.0, 0.0, self.LAST, 1.0, 0.4)
        assert "basin persistence" in reason
        assert "0.40" in reason

    def test_persistence_tie_passes(self, mm):
        # spec section 7: reject on <, exact equality with the floor passes
        assert mm._mep_reject_reason(10, 2.0, 0.0, 0.0, self.LAST, 1.0, 1.0) is None


def _line_tree():
    """Hand-built 1-D map, 6 cells, 3 basins. Root (id 0) is the FE basin.

    cells:   0    1    2    3    4    5
    labels:  1    1    2    2    0    0
    basin 1: GS,  seed cell 0, E=0.0, dies at 5.0 -> persistence 5.0
    basin 2: mid, seed cell 3, E=1.0, dies at 3.0 -> persistence 2.0
    basin 0: FE,  seed cell 5, E=-2.0, root -> persistence inf
    """
    labels = np.array([1, 1, 2, 2, 0, 0], dtype=np.int32)
    basins = [((5,), -2.0), ((0,), 0.0), ((3,), 1.0)]
    merges = [((4,), 3.0, 0, 2), ((1,), 5.0, 0, 1)]
    return MergeTree(labels, basins, merges)


def test_collect_candidates_dedup_and_role_exclusion(mm):
    """Spec section 5.2: first-visit kept; GS re-entry, FE wells, and basin
    revisits (P2) are flagged with a skip reason, never classified."""
    tree = _line_tree()
    # A walk (P2): GS -> mid -> back into GS -> mid again -> FE
    path_cells = [0, 1, 3, 1, 0, 1, 3, 4, 5]
    path_idx = np.array([[c] for c in path_cells], dtype=np.int64)
    profile = PathProfile(
        minima=[(0, 0.0), (2, 1.0), (4, 0.0), (6, 1.0), (8, -2.0)],
        saddles=[(1, 5.0), (3, 5.0), (5, 5.0), (7, 3.0)],
    )
    interior = mm._collect_mep_candidates(
        profile, path_idx, tree, gs_basin=1, fe_basin=0)
    assert [(k, bid, skip is None) for k, _e, bid, skip in interior] == [
        (2, 2, True),    # first visit of the mid basin -> candidate
        (4, 1, False),   # GS well re-entry -> skipped
        (6, 2, False),   # revisit of basin 2 (P2 dedup) -> skipped
    ]
    assert "GS basin" in interior[1][3]
    assert "revisit of basin #2" in interior[2][3]


def test_collect_candidates_excludes_fission_exit_basin(mm):
    """A profile minimum inside the FE basin must not become a TM candidate
    (deep-descent topologies where the half-path rule would not fire)."""
    tree = _line_tree()
    path_cells = [0, 1, 3, 2, 4, 5]
    path_idx = np.array([[c] for c in path_cells], dtype=np.int64)
    profile = PathProfile(
        minima=[(0, 0.0), (2, 1.0), (4, -1.0), (5, -2.0)],
        saddles=[(1, 5.0), (3, 3.0)],
    )
    interior = mm._collect_mep_candidates(
        profile, path_idx, tree, gs_basin=1, fe_basin=0)
    by_k = {k: skip for k, _e, _bid, skip in interior}
    assert by_k[2] is None                       # mid basin: candidate
    assert "fission-exit basin" in by_k[4]       # cell 4 -> basin 0
    assert 5 not in by_k                         # endpoint excluded entirely
