"""Tests for the merge-tree plotting helpers in MapMaker_FoS_SHE."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import MapMaker_FoS_SHE as mm  # noqa: E402
from pes_analyzer.topology import MergeTree, prune_merge_tree  # noqa: E402


def _synthetic_tree():
    """Four basins along c (axis 0): GS(0), SM(1), noise(2, persist 0.05), FE(3).

    Tree shape: 1 and 3 merge into root 0; noise 2 merges into 1.
      basin: (minimum_index, minimum_energy)
      merge: (saddle_index, saddle_energy, deeper, shallower)
    Persistences: 1 -> 2.0, 3 -> 2.0, 2 -> 0.05 (below MARKER_MIN).
    """
    basins = [
        ((1, 1), -10.0),   # 0  GS   (c index 1)
        ((3, 1), -8.0),    # 1  SM   (c index 3)
        ((5, 1), -7.95),   # 2  noise(c index 5)
        ((7, 1), -7.0),    # 3  FE   (c index 7)
    ]
    merges = [
        ((4, 1), -7.90, 1, 2),  # noise 2 dies into 1  (persist 0.05)
        ((2, 1), -6.00, 0, 1),  # SM 1 dies into 0     (persist 2.0)
        ((6, 1), -5.00, 0, 3),  # FE 3 dies into 0     (persist 2.0)
    ]
    labels = np.zeros((9, 3), dtype=np.int32)
    tree = MergeTree(labels, basins, merges)
    axes = {"c": np.linspace(0.0, 0.8, 9), "a4": np.linspace(-0.1, 0.1, 3)}
    return tree, axes, basins, merges


def test_segments_order_x_by_c_and_root_runs_to_top():
    tree, axes, _, _ = _synthetic_tree()
    displayed = {0, 1, 2, 3}
    top = 0.0
    x_of, vsegs, hsegs = mm._merge_tree_segments(tree, displayed, c_pos=0, top_energy=top)

    # x ordered by c index: 0(c1) < 1(c3) < 2(c5) < 3(c7)
    assert x_of == {0: 0, 1: 1, 2: 2, 3: 3}
    # one vertical segment per displayed basin
    assert set(vsegs.keys()) == {0, 1, 2, 3}
    # root (0) has no parent -> runs up to top_energy and is not a child in any hseg
    assert vsegs[0][2] == top
    assert all(child != 0 for (_xc, _xa, _e, child) in hsegs)
    # basin 1 connects to ancestor 0 at its saddle energy -6.0
    h1 = [h for h in hsegs if h[3] == 1][0]
    assert h1[1] == x_of[0] and h1[2] == -6.0


def test_segments_skip_pruned_ancestor():
    # Display only {0, 2}: 2's real parent (1) is pruned, so 2 must connect to 0.
    tree, axes, _, _ = _synthetic_tree()
    x_of, vsegs, hsegs = mm._merge_tree_segments(tree, {0, 2}, c_pos=0, top_energy=0.0)
    h2 = [h for h in hsegs if h[3] == 2][0]
    assert h2[1] == x_of[0]   # connects to nearest *displayed* ancestor (root)


def test_plot_merge_tree_writes_two_panel_png(tmp_path):
    tree, axes, _, _ = _synthetic_tree()
    roles = {
        "ground_state": 0,
        "secondary_minimum": 1,
        "third_minimum": None,
        "fission_exit": 3,
        "inner_saddle": ((2, 1), -6.0),
        "outer_saddle": ((6, 1), -5.0),
        "third_saddle": None,
    }
    nucleus = mm.NucleusInfo()
    nucleus.isotope_label = "Test"
    nucleus.A, nucleus.symbol = 999, "Ts"

    out = tmp_path / "merge_tree.png"
    import io
    mm.plot_merge_tree(tree, roles, axes, nucleus, str(out), out=io.StringIO())

    assert out.exists() and out.stat().st_size > 0


def test_prune_reduces_displayed_count():
    # Sanity: at the default prune threshold the noise basin (persist 0.05) drops.
    tree, _, basins, merges = _synthetic_tree()
    surviving, _kept = prune_merge_tree(basins, merges, mm.MERGE_TREE_PRUNE)
    assert set(surviving) == {0, 1, 3}        # noise basin 2 removed
    assert len(surviving) < len(basins)


def test_no_isomer_fallback_reports_gs_barrier_and_exit():
    """No basin clears the SM floor: select must still find GS, fission saddle, exit.

    Two basins: a compact GS (interior) and a shallow elongated exit whose
    minimum sits on the a4-max wall. The exit's persistence (0.2 MeV) is below
    SM_PERSISTENCE, so it cannot be a secondary minimum -- but it is a valid
    fission exit. The fallback searches outward from the GS and stores the lone
    barrier as the inner saddle.
    """
    # labels: 2D (c=9, a4=3); a4-max wall is index 2.
    basins = [
        ((1, 1), -10.0),   # 0  GS   compact (c idx 1), interior a4
        ((7, 2), -7.0),    # 1  exit  elongated (c idx 7), on the a4-max wall
    ]
    merges = [
        ((6, 2), -6.8, 0, 1),   # exit 1 dies into GS 0 -> persistence 0.2 (< 0.4)
    ]
    labels = np.zeros((9, 3), dtype=np.int32)
    tree = MergeTree(labels, basins, merges)
    axes = {"c": np.linspace(0.0, 0.8, 9), "a4": np.linspace(-0.1, 0.1, 3)}

    def has_min(_b):
        return True

    def c_of(b):
        return float(axes["c"][tree.node(b).minimum_index[0]])

    sel = mm.select_fos_critical_points(tree, has_min, c_of, c_axis=0, a4_axis=1)

    assert sel["ground_state"] == 0
    assert sel["secondary_minimum"] is None     # nothing clears the SM floor
    assert sel["fission_exit"] == 1             # shallow exit still found
    assert sel["inner_saddle"] is not None      # the lone fission barrier
    assert sel["inner_saddle"][1] == -6.8       # stored at the barrier energy
    assert sel["outer_saddle"] is None
    assert sel["third_minimum"] is None
    assert sel["third_saddle"] is None
