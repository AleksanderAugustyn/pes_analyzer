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


def _axes_for(shape):
    """Synthetic physical axes: c along axis 0 (0.8 + 0.1*i, so row 0 is
    inside the GS region c < 1.4), a4 along axis 1."""
    return {
        "c": np.linspace(0.8, 0.8 + 0.1 * (shape[0] - 1), shape[0]),
        "a4": np.linspace(0.0, 0.05 * (shape[1] - 1), shape[1]),
    }


def _run_selection(mm, energies, minima_gs_sm):
    axes = _axes_for(energies.shape)
    labels, basins, merges = find_watershed_segmentation(energies)
    tree = MergeTree(labels, basins, merges)
    log = io.StringIO()
    sel, path_idx, path_e, profile = mm.select_mep_critical_points(
        energies, minima_gs_sm, axes, tree, out=log)
    return sel, tree, log.getvalue()


def _rg_like_grid():
    """Corridor PES (column 1; 9-MeV walls either side), 276Rg-like:

    row:  0    1    2    3     4     5    6    7     8     9     10    11
    E:    0.0  6.0  2.0  2.45  2.05  4.0  1.0  -0.5  -1.0  -1.5  -2.0  -3.0
          GS   S1   SM   dimple-saddle/min  S2  ---- descent to FE ----

    The shallow dimple at row 4 survives MEP-profile pruning (pair gap
    0.40 >= MEP_PERSISTENCE = 0.4: analyze_path_profile cancels only on
    strict <) and sits inside the half-path window (k=4 <= 11//2), but its
    basin persistence is 2.45 - 2.05 = 0.40 MeV: rejected topologically.
    The deep SM basin dies at the outer saddle: persistence 4.0 - 2.0 = 2.0.
    The GS basin dies at the inner saddle: persistence 6.0 (= barrier height).
    """
    e = np.full((12, 3), 9.0)
    e[:, 1] = [0.0, 6.0, 2.0, 2.45, 2.05, 4.0,
               1.0, -0.5, -1.0, -1.5, -2.0, -3.0]
    return e


def test_rg_like_dimple_rejected_on_basin_persistence(mm):
    energies = _rg_like_grid()
    sel, tree, log = _run_selection(
        mm, energies, [((0, 1), 0.0), ((2, 1), 2.0)])

    assert sel["ground_state"] == tree.basin_of_point((0, 1))
    assert sel["secondary_minimum"] == tree.basin_of_point((2, 1))
    assert sel["third_minimum"] is None
    assert sel["fission_exit"] == tree.basin_of_point((11, 1))

    # GS basin persistence = minimax fission-barrier height (spec 6.4);
    # also the spec-9 residual check that MergeTree.node().persistence is
    # the raw saddle_e - min_e with no pruning applied.
    assert sel["gs_persistence_bf"] == pytest.approx(6.0)

    # segment-max saddles, exact by P3, with segment names (spec 5.6)
    assert sel["inner_saddle"][1] == pytest.approx(6.0)
    assert sel["outer_saddle"][1] == pytest.approx(4.0)
    assert sel["third_saddle"] is None
    assert sel["saddle_segments"] == {
        "inner_saddle": "gs-sm", "outer_saddle": "sm-fe"}

    # the dimple was rejected on topological grounds (as TM candidate,
    # floor THIRD_MIN_PERSISTENCE = 0.5), not by a geometry proxy
    assert "basin persistence 0.40 MeV < 0.5 MeV" in log

    # diagnostics: both measures and the basin id are logged (spec 5.5)
    assert "persistence=" in log and "profile-depth=" in log and "basin #" in log


def _sideways_leak_grid():
    """SM candidate whose lowest escape is OFF the GS->FE path (spec 8.1.ii).

    Path corridor is column 2 (rows 0..11); column 0 holds a dead-end
    pocket deeper than the SM candidate, joined through column 1:

      (2,0) = -0.6 pocket minimum (below GS energy, so even if the deep
                   minimax walker detours into it, check (a) rejects it)
      (2,1) =  2.5 leak saddle pocket <-> SM
      (2,2) =  2.0 SM candidate

    On-path well depth of the SM candidate: min(6.0, 4.0) - 2.0 = 2.0 MeV
    (old rule would ACCEPT). Basin persistence: 2.5 - 2.0 = 0.5 MeV
    (new rule REJECTS: 0.5 < SM_PERSISTENCE = 1.0). Spec 6.1: switching
    the floor to persistence can only reject more, never accept new.
    """
    e = np.full((12, 4), 9.0)
    e[:, 2] = [0.0, 6.0, 2.0, 4.0, 1.5, 1.0,
               0.0, -0.5, -1.0, -1.5, -2.0, -3.0]
    e[2, 0] = -0.6
    e[2, 1] = 2.5
    return e


def test_sideways_leak_rejected_by_persistence_not_profile_depth(mm):
    energies = _sideways_leak_grid()
    sel, tree, log = _run_selection(
        mm, energies, [((0, 2), 0.0), ((2, 2), 2.0)])

    assert sel["secondary_minimum"] is None
    assert sel["third_minimum"] is None
    assert "basin persistence 0.50 MeV < 1.0 MeV" in log

    # all candidates rejected -> single gs-fe segment, lone barrier kept
    # as the inner saddle (spec section 7)
    assert sel["inner_saddle"][1] == pytest.approx(6.0)
    assert sel["outer_saddle"] is None
    assert sel["saddle_segments"]["inner_saddle"] == "gs-fe"
