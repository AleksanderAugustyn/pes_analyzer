"""Unit tests for pes_analyzer.topology.MergeTree (pure Python; no rebuild)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pes_analyzer.topology import MergeTree, Watershed, compute_persistence


def _actinide_chain():
    """Hand-built Watershed for a 3-basin chain on a 5x5 grid.

    basin 0: min (0,0) E=0.0  (deepest, root)
    basin 1: min (2,2) E=1.0  -> merges into 0 at saddle (1,1) E=5.0
    basin 2: min (4,4) E=2.0  -> merges into 1 at saddle (3,3) E=4.0
    labels: col 0..1 -> 0, cols 2..3 / rows -> 1, column 4 -> 2 (touches axis-1 max edge).
    """
    labels = np.array(
        [[0, 0, 1, 1, 2]] * 5,
        dtype=np.int32,
    )
    basins = [((0, 0), 0.0), ((2, 2), 1.0), ((4, 4), 2.0)]
    merges = [((1, 1), 5.0, 0, 1), ((3, 3), 4.0, 1, 2)]
    return Watershed(labels=labels, basins=basins, merges=merges, neighborhood="von_neumann",
                     parents=None, merge_table=np.zeros((2, 5), np.uint32),
                     dtype=np.dtype("float64"), fingerprint=b"\x00" * 16)


def test_construction_links_and_persistence():
    ws = _actinide_chain()
    tree = MergeTree(ws)

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
    p = compute_persistence(ws.basins, ws.merges)
    assert tree.persistence(2) == pytest.approx(p[2])


def test_empty_tree_is_graceful():
    ws = Watershed(labels=np.full((3, 3), -1, np.int32), basins=[], merges=[], neighborhood="von_neumann",
                   parents=None, merge_table=np.zeros((0, 5), np.uint32), dtype=np.dtype("float64"), fingerprint=b"")
    tree = MergeTree(ws)
    assert tree.root is None
    assert dict(tree.nodes) == {}


def test_neighbors():
    tree = MergeTree(_actinide_chain())
    assert sorted(tree.neighbors(0)) == [1]        # root: child only
    assert sorted(tree.neighbors(1)) == [0, 2]     # parent + child
    assert sorted(tree.neighbors(2)) == [1]        # leaf: parent only


def test_path():
    tree = MergeTree(_actinide_chain())
    assert tree.path(0, 0) == [0]
    assert tree.path(0, 2) == [0, 1, 2]
    assert tree.path(2, 0) == [2, 1, 0]
    assert tree.path(1, 2) == [1, 2]


def test_bfs_hop_order_and_advance():
    tree = MergeTree(_actinide_chain())
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


def test_basin_of_point():
    tree = MergeTree(_actinide_chain())
    assert tree.basin_of_point((0, 0)) == 0
    assert tree.basin_of_point((2, 2)) == 1
    assert tree.basin_of_point((4, 4)) == 2


def test_basins_containing_groups_points_and_skips_nan():
    base = _actinide_chain()
    modified = base.labels.copy()
    modified[0, 4] = -1  # a NaN cell
    ws = Watershed(labels=modified, basins=base.basins, merges=base.merges, neighborhood=base.neighborhood,
                   parents=base.parents, merge_table=base.merge_table, dtype=base.dtype, fingerprint=base.fingerprint)
    tree = MergeTree(ws)
    grouped = tree.basins_containing([(0, 0), (1, 0), (2, 2), (4, 4), (0, 4)])
    assert sorted(grouped.keys()) == [0, 1, 2]
    assert (0, 0) in grouped[0] and (1, 0) in grouped[0]
    assert grouped[1] == [(2, 2)]
    assert grouped[2] == [(4, 4)]  # (0,4) had label -1 -> skipped


def test_touches_edge():
    tree = MergeTree(_actinide_chain())
    # basin 2 occupies column 4 == shape[1]-1 (axis-1 max edge)
    assert tree.touches_edge(2, axis=1, side="max") is True
    assert tree.touches_edge(2, axis=1, side="min") is False
    # basin 0 occupies column 0 (axis-1 min edge), not the max
    assert tree.touches_edge(0, axis=1, side="min") is True
    assert tree.touches_edge(0, axis=1, side="max") is False
    # basin 1 (columns 2..3) touches neither axis-1 edge
    assert tree.touches_edge(1, axis=1, side="both") is False
    with pytest.raises(ValueError, match="side must be"):
        tree.touches_edge(0, axis=0, side="bogus")


def test_tree_delegates_to_watershed_arrays():
    ws = _actinide_chain()
    tree = MergeTree(ws)
    assert tree.ws is ws
    assert tree.labels is ws.labels
    assert tree.parents is None and tree.merge_table is ws.merge_table
    assert tree.neighborhood == "von_neumann" and tree.dtype == np.dtype("float64")
    assert tree.fingerprint == ws.fingerprint
    assert tree.has_labels


def test_basin_mask():
    tree = MergeTree(_actinide_chain())
    mask = tree.basin_mask(1)
    assert mask.dtype == bool and mask.shape == (5, 5)
    assert mask.sum() == 10 and mask[:, 2:4].all()


def test_drop_labels_releases_for_every_holder_and_guards_queries():
    ws = _actinide_chain()
    tree = MergeTree(ws)
    tree.drop_labels()
    assert not tree.has_labels and ws.labels is None and tree.labels is None
    for call in (
        lambda: tree.basin_of_point((0, 0)),
        lambda: tree.basins_containing([(0, 0)]),
        lambda: tree.touches_edge(0, axis=1),
        lambda: tree.basin_mask(0),
    ):
        with pytest.raises(RuntimeError, match="labels were dropped"):
            call()
    # Tree queries keep working.
    assert tree.path(0, 2) == [0, 1, 2]
    assert tree.node(2).persistence == pytest.approx(2.0)


def test_touches_edge_uses_face_slices_only(monkeypatch):
    # Guard against regressions to the full-grid `labels == bid` temporary:
    # np.take must be called on the labels (face) before any comparison.
    ws = _actinide_chain()
    tree = MergeTree(ws)
    calls = []
    real_take = np.take

    def spy(a, *args, **kwargs):
        calls.append((a.shape, a.dtype, a is ws.labels))
        return real_take(a, *args, **kwargs)

    monkeypatch.setattr(np, "take", spy)
    assert tree.touches_edge(2, axis=1, side="max") is True
    # Exactly one take, on the int32 label grid object itself — a `labels == bid`
    # temporary would show up here as a bool array that is not ws.labels.
    assert calls == [((5, 5), np.dtype(np.int32), True)]
