"""Unit tests for pes_analyzer.topology.MergeTree (pure Python; no rebuild)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pes_analyzer.topology import MergeTree, compute_persistence


def _actinide_chain():
    """Hand-built (labels, basins, merges) for a 3-basin chain on a 5x5 grid.

    basin 0: min (0,0) E=0.0  (deepest, root)
    basin 1: min (2,2) E=1.0  -> merges into 0 at saddle (1,1) E=5.0
    basin 2: min (4,4) E=2.0  -> merges into 1 at saddle (3,3) E=4.0
    labels: col 0..1 -> 0, cols 2..3 / rows -> 1, column 4 -> 2 (touches axis-1 max edge).
    """
    labels = np.array(
        [
            [0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2],
        ],
        dtype=np.int32,
    )
    basins = [((0, 0), 0.0), ((2, 2), 1.0), ((4, 4), 2.0)]
    merges = [((1, 1), 5.0, 0, 1), ((3, 3), 4.0, 1, 2)]
    return labels, basins, merges


def test_construction_links_and_persistence():
    labels, basins, merges = _actinide_chain()
    tree = MergeTree(labels, basins, merges)

    assert tree.root == 0
    root = tree.node(0)
    assert root.parent is None
    assert math.isinf(root.persistence)
    assert sorted(root.children) == [1]

    n1 = tree.node(1)
    assert n1.parent == 0
    assert n1.saddle_to_parent == ((1, 1), 5.0)
    assert n1.children == [2]
    assert n1.persistence == pytest.approx(5.0 - 1.0)
    assert n1.minimum_index == (2, 2)
    assert n1.minimum_energy == 1.0

    n2 = tree.node(2)
    assert n2.parent == 1
    assert n2.saddle_to_parent == ((3, 3), 4.0)
    assert n2.children == []
    assert n2.persistence == pytest.approx(4.0 - 2.0)

    # persistence agrees with the free function
    p = compute_persistence(basins, merges)
    assert tree.persistence(2) == pytest.approx(p[2])


def test_empty_tree_is_graceful():
    labels = np.full((3, 3), -1, dtype=np.int32)
    tree = MergeTree(labels, [], [])
    assert tree.root is None
    assert dict(tree.nodes) == {}


def test_neighbors():
    labels, basins, merges = _actinide_chain()
    tree = MergeTree(labels, basins, merges)
    assert sorted(tree.neighbors(0)) == [1]        # root: child only
    assert sorted(tree.neighbors(1)) == [0, 2]     # parent + child
    assert sorted(tree.neighbors(2)) == [1]        # leaf: parent only


def test_path():
    labels, basins, merges = _actinide_chain()
    tree = MergeTree(labels, basins, merges)
    assert tree.path(0, 0) == [0]
    assert tree.path(0, 2) == [0, 1, 2]
    assert tree.path(2, 0) == [2, 1, 0]
    assert tree.path(1, 2) == [1, 2]


def test_bfs_hop_order_and_advance():
    labels, basins, merges = _actinide_chain()
    tree = MergeTree(labels, basins, merges)
    # full bfs from leaf reaches every node, increasing depth
    order = list(tree.bfs(2))
    assert order[0] == (2, 0)
    assert {bid for bid, _ in order} == {0, 1, 2}
    assert dict(order)[0] == 2  # basin 0 is two hops from leaf 2

    # advance gate: only walk toward strictly-larger basin id
    seen = [bid for bid, _ in tree.bfs(0, advance=lambda a, b: b > a)]
    assert seen == [0, 1, 2]
    # advance gate blocking everything yields only the start
    assert [bid for bid, _ in tree.bfs(1, advance=lambda a, b: False)] == [1]
