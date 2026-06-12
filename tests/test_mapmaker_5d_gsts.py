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


def _fragmented_tree():
    """1-D map, 8 cells: a sub-noise satellite fragment of a deeper well.

    cells:   0    1    2    3    4    5    6    7
    labels:  1    1    3    3    2    2    0    0
    basin 1: GS,        seed cell 0, E=0.0,  dies at 5.0  -> pers 5.0
    basin 3: satellite, seed cell 2, E=1.05, dies at 1.15 -> pers 0.10
    basin 2: well,      seed cell 4, E=1.0,  dies at 4.0  -> pers 3.0
    basin 0: FE (root), seed cell 7, E=-2.0
    """
    labels = np.array([1, 1, 3, 3, 2, 2, 0, 0], dtype=np.int32)
    basins = [((7,), -2.0), ((0,), 0.0), ((4,), 1.0), ((2,), 1.05)]
    merges = [((3,), 1.15, 2, 3), ((5,), 4.0, 0, 2), ((1,), 5.0, 0, 1)]
    return MergeTree(labels, basins, merges)


def test_collect_candidates_reanchors_subnoise_satellite(mm):
    """A basin dying through a merge shallower than the profile noise floor
    is a satellite fragment of a larger well: identity and energy re-anchor
    to the merge parent (the 232Th case: MEP bottoms in a 0.09 MeV ripple
    two steps from the true SM seed)."""
    tree = _fragmented_tree()
    path_cells = [0, 1, 2, 5, 6, 7]   # bottoms in the satellite (cell 2)
    path_idx = np.array([[c] for c in path_cells], dtype=np.int64)
    profile = PathProfile(
        minima=[(0, 0.0), (2, 1.05), (5, -2.0)],
        saddles=[(1, 5.0), (3, 4.0)],
    )
    interior = mm._collect_mep_candidates(
        profile, path_idx, tree, gs_basin=1, fe_basin=0, reanchor_floor=0.4)
    assert len(interior) == 1
    k, e, bid, skip = interior[0]
    assert (k, bid, skip) == (2, 2, None)   # satellite #3 -> well #2
    assert e == pytest.approx(1.0)          # representative's minimum energy

    # without a floor the satellite keeps its own id (and its own energy)
    interior0 = mm._collect_mep_candidates(
        profile, path_idx, tree, gs_basin=1, fe_basin=0)
    assert interior0[0][2] == 3
    assert interior0[0][1] == pytest.approx(1.05)


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


def _fragmented_well_grid():
    """232Th-like sub-noise fragmentation (spec 5.2 re-anchoring).

    Path corridor is column 1. The MEP bottoms in a satellite basin at
    (2,1)=2.05 whose only escape below the path barriers is a 2.15 MeV
    ripple cell (2,2) into the true well seed (2,3)=2.0 one step off-path:

      satellite persistence 2.15 - 2.05 = 0.10 < MEP_PERSISTENCE (0.4)
      well       persistence 4.0  - 2.0  = 2.0  >= SM_PERSISTENCE (1.0)

    Without re-anchoring the SM is rejected (0.10 < 1.0) and the deep well
    is lost; with re-anchoring the SM is the off-path well seed.
    """
    e = np.full((12, 4), 9.0)
    e[:, 1] = [0.0, 6.0, 2.05, 4.0, 1.0, -0.5,
               -1.0, -1.5, -2.0, -2.5, -2.8, -3.0]
    e[2, 2] = 2.15
    e[2, 3] = 2.0
    return e


def test_fragmented_well_reanchored_to_true_seed(mm):
    energies = _fragmented_well_grid()
    sel, tree, log = _run_selection(
        mm, energies, [((0, 1), 0.0), ((2, 3), 2.0)])

    # SM identity is the re-anchored well, seeded one step OFF the path
    assert sel["secondary_minimum"] == tree.basin_of_point((2, 3))
    assert sel["secondary_minimum"] != tree.basin_of_point((2, 1))
    assert tree.node(sel["secondary_minimum"]).minimum_index == (2, 3)
    assert "re-anchored" in log

    assert sel["inner_saddle"][1] == pytest.approx(6.0)
    assert sel["outer_saddle"][1] == pytest.approx(4.0)
    assert sel["saddle_segments"] == {
        "inner_saddle": "gs-sm", "outer_saddle": "sm-fe"}


def test_csv_gains_basin_columns(mm, tmp_path):
    """Spec 5.6: minima rows carry basin_id / basin_persistence /
    profile_well_depth; saddle rows carry segment; gs_persistence_bf is a
    per-nucleus column. Empty persistence = undying root basin."""
    nucleus = mm.NucleusInfo(Z=111, N=165, A=276, symbol="Rg",
                             isotope_label="276Rg")

    gs = mm.CriticalPoint(mm.CriticalPointType.GROUND_STATE, "Ground State")
    gs.point = mm.GridPoint(c=1.05, total_energy=-5.93, valid=True)
    gs.found = True
    gs.basin_id = 17
    gs.basin_persistence = 5.70
    gs.profile_well_depth = 5.95

    fe = mm.CriticalPoint(mm.CriticalPointType.FISSION_EXIT, "Fission Exit")
    fe.point = mm.GridPoint(c=2.3, total_energy=-9.1, valid=True)
    fe.found = True
    fe.basin_id = 0
    fe.basin_persistence = float('nan')   # root basin: never dies -> empty cell

    sad = mm.CriticalPoint(mm.CriticalPointType.FIRST_SADDLE, "Inner Saddle")
    sad.point = mm.GridPoint(c=1.3, total_energy=-0.23, valid=True)
    sad.found = True
    sad.segment = "gs-sm"

    cps = {cp.point_type: cp for cp in (gs, fe, sad)}
    out_file = mm.save_critical_points_csv(
        cps, nucleus, output_dir=str(tmp_path),
        gs_persistence_bf=5.70, out=io.StringIO())

    df = pl.read_csv(out_file)
    for colname in ("basin_id", "basin_persistence", "profile_well_depth",
                    "segment", "gs_persistence_bf"):
        assert colname in df.columns, colname

    gs_row = df.filter(pl.col("point_type") == "Ground State")
    assert gs_row["basin_id"][0] == 17
    assert gs_row["basin_persistence"][0] == pytest.approx(5.70)
    assert gs_row["profile_well_depth"][0] == pytest.approx(5.95)
    assert gs_row["gs_persistence_bf"][0] == pytest.approx(5.70)

    fe_row = df.filter(pl.col("point_type") == "Fission Exit")
    assert fe_row["basin_persistence"][0] is None

    sad_row = df.filter(pl.col("point_type") == "Inner Saddle")
    assert sad_row["segment"][0] == "gs-sm"
    assert sad_row["basin_id"][0] is None
