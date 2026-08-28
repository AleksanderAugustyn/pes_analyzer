"""Traversable merge tree over watershed basins.

Pure Python on top of a :class:`Watershed` from ``find_watershed_segmentation``.
One node per basin, rooted at the deepest basin (id 0). The tree is
physics-free: it knows nothing about ground states or fission.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

from ._flood import Watershed
from ._tree import compute_persistence


@dataclass
class BasinNode:
    basin_id: int
    minimum_index: tuple[int, ...]
    minimum_energy: float
    parent: Optional[int]
    saddle_to_parent: Optional[tuple[tuple[int, ...], float]]
    persistence: float
    children: list[int] = field(default_factory=list)


class MergeTree:
    """Rooted tree of watershed basins over a :class:`Watershed`.

    The tree never copies the grid arrays: ``labels``/``parents``/``merge_table``
    are properties of the owning ``Watershed``; :meth:`drop_labels` releases
    them for every holder.

    Parameters
    ----------
    ws : Watershed
        Output of ``find_watershed_segmentation``. Its ``basins`` and
        ``merges`` define the nodes and edges; its grid arrays back the
        membership queries (``basin_of_point``, ``basins_containing``,
        ``basin_mask``, ``touches_edge``).
    """

    def __init__(self, ws: Watershed) -> None:
        self.ws = ws
        self._basins = list(ws.basins)
        self._merges = list(ws.merges)
        basins, merges = self._basins, self._merges

        persistence = compute_persistence(basins, merges) if basins else np.array([])

        nodes: dict[int, BasinNode] = {}
        for bid, (min_idx, min_e) in enumerate(basins):
            nodes[bid] = BasinNode(
                basin_id=bid,
                minimum_index=tuple(int(i) for i in min_idx),
                minimum_energy=float(min_e),
                parent=None,
                children=[],
                saddle_to_parent=None,
                persistence=float(persistence[bid]),
            )
        for saddle_idx, saddle_e, deeper, shallower in merges:
            child = nodes[shallower]
            child.parent = deeper
            child.saddle_to_parent = (tuple(int(i) for i in saddle_idx), float(saddle_e))
            nodes[deeper].children.append(shallower)

        self.nodes = nodes
        self.root: Optional[int] = 0 if basins else None

    # -- grid arrays (owned by the Watershed) --------------------------------

    @property
    def labels(self) -> Optional[np.ndarray]:
        return self.ws.labels

    @property
    def parents(self) -> Optional[np.ndarray]:
        return self.ws.parents

    @property
    def merge_table(self) -> Optional[np.ndarray]:
        return self.ws.merge_table

    @property
    def neighborhood(self) -> str:
        return self.ws.neighborhood

    @property
    def dtype(self) -> np.dtype:
        return self.ws.dtype

    @property
    def fingerprint(self) -> bytes:
        return self.ws.fingerprint

    @property
    def has_labels(self) -> bool:
        return self.ws.labels is not None

    def drop_labels(self) -> None:
        """Release labels, parents and merge_table (4N + 2N bytes) for every holder."""
        self.ws.drop_labels()

    def _labels(self) -> np.ndarray:
        if self.ws.labels is None:
            raise RuntimeError("MergeTree labels were dropped; membership queries are unavailable")
        return self.ws.labels

    # -- node access --------------------------------------------------------

    def node(self, bid: int) -> BasinNode:
        """Return the BasinNode for basin ``bid``."""
        return self.nodes[bid]

    def persistence(self, bid: int) -> float:
        """Return the topological persistence of basin ``bid``."""
        return self.nodes[bid].persistence

    # -- traversal ----------------------------------------------------------

    def neighbors(self, bid: int) -> list[int]:
        n = self.nodes[bid]
        out = list(n.children)
        if n.parent is not None:
            out.append(n.parent)
        return out

    def path(self, a: int, b: int) -> list[int]:
        """Inclusive tree path from ``a`` to ``b``."""
        chain_a: list[int] = []
        x: Optional[int] = a
        while x is not None:
            chain_a.append(x)
            x = self.nodes[x].parent
        index_in_a = {node: i for i, node in enumerate(chain_a)}

        up_from_b: list[int] = []
        x = b
        while x not in index_in_a:
            up_from_b.append(x)
            x = self.nodes[x].parent
        lca = x
        return chain_a[: index_in_a[lca] + 1] + list(reversed(up_from_b))

    def bfs(
        self,
        start: int,
        *,
        advance: Optional[Callable[[int, int], bool]] = None,
    ) -> Iterator[tuple[int, int]]:
        """Breadth-first walk from ``start`` yielding ``(basin_id, depth)``.

        ``advance(from_bid, to_bid) -> bool`` gates edge traversal; when it
        returns ``False`` the edge (and the subtree beyond it) is skipped.
        """
        seen = {start}
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        while queue:
            bid, depth = queue.popleft()
            yield bid, depth
            for nb in self.neighbors(bid):
                if nb in seen:
                    continue
                if advance is not None and not advance(bid, nb):
                    continue
                seen.add(nb)
                queue.append((nb, depth + 1))

    # -- membership & geometry ---------------------------------------------

    def basin_of_point(self, index: tuple[int, ...]) -> int:
        """Return the basin ID at grid cell ``index`` (-1 for a NaN cell)."""
        return int(self._labels()[tuple(index)])

    def basins_containing(
        self, points: list[tuple[int, ...]]
    ) -> dict[int, list[tuple[int, ...]]]:
        """Group index-tuples by the basin each lands in. NaN cells (-1) skipped."""
        labels = self._labels()
        out: dict[int, list[tuple[int, ...]]] = {}
        for p in points:
            bid = int(labels[tuple(p)])
            if bid < 0:
                continue
            out.setdefault(bid, []).append(p)
        return out

    def basin_mask(self, bid: int) -> np.ndarray:
        """Boolean grid, True on the cells of basin ``bid``."""
        return self._labels() == bid

    def touches_edge(self, bid: int, axis: int, side: str = "max") -> bool:
        """True iff any cell of basin ``bid`` lies on the ``axis`` boundary.

        ``side`` is ``'min'`` (index 0), ``'max'`` (index shape[axis]-1), or
        ``'both'``. Only the boundary face is compared (no full-grid temporary).
        """
        labels = self._labels()
        last = labels.shape[axis] - 1
        if side == "min":
            return bool((np.take(labels, 0, axis=axis) == bid).any())
        if side == "max":
            return bool((np.take(labels, last, axis=axis) == bid).any())
        if side == "both":
            return bool(
                (np.take(labels, 0, axis=axis) == bid).any()
                or (np.take(labels, last, axis=axis) == bid).any()
            )
        raise ValueError("side must be 'min', 'max', or 'both'")
