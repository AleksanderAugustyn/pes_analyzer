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
