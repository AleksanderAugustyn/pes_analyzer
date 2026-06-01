"""Type stubs for ``pes_analyzer.topology``."""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np
import numpy.typing as npt

def find_watershed_segmentation(
    energies: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int32],
    list[tuple[tuple[int, ...], float]],
    list[tuple[tuple[int, ...], float, int, int]],
]: ...

def compute_persistence(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
) -> npt.NDArray[np.float64]: ...

def prune_merge_tree(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
    threshold: float,
) -> tuple[list[int], list[tuple[tuple[int, ...], float, int, int]]]: ...

class BasinNode:
    basin_id: int
    minimum_index: tuple[int, ...]
    minimum_energy: float
    parent: Optional[int]
    children: list[int]
    saddle_to_parent: Optional[tuple[tuple[int, ...], float]]
    persistence: float

class MergeTree:
    labels: npt.NDArray[np.int32]
    nodes: dict[int, BasinNode]
    root: Optional[int]
    def __init__(
        self,
        labels: npt.NDArray[np.int32],
        basins: list[tuple[tuple[int, ...], float]],
        merges: list[tuple[tuple[int, ...], float, int, int]],
    ) -> None: ...
    def node(self, bid: int) -> BasinNode: ...
    def persistence(self, bid: int) -> float: ...
    def neighbors(self, bid: int) -> list[int]: ...
    def path(self, a: int, b: int) -> list[int]: ...
    def bfs(
        self,
        start: int,
        *,
        advance: Optional[Callable[[int, int], bool]] = ...,
    ) -> Iterator[tuple[int, int]]: ...
    def basin_of_point(self, index: tuple[int, ...]) -> int: ...
    def basins_containing(
        self, points: list[tuple[int, ...]]
    ) -> dict[int, list[tuple[int, ...]]]: ...
    def touches_edge(self, bid: int, axis: int, side: str = ...) -> bool: ...
