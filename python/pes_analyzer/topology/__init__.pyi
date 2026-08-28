"""Type stubs for ``pes_analyzer.topology``."""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np
import numpy.typing as npt

class Watershed:
    labels: npt.NDArray[np.int32] | None
    basins: list[tuple[tuple[int, ...], float]]
    merges: list[tuple[tuple[int, ...], float, int, int]]
    neighborhood: str
    parents: npt.NDArray[np.uint16] | None
    merge_table: npt.NDArray[np.uint32] | None
    dtype: np.dtype
    fingerprint: bytes
    @property
    def has_labels(self) -> bool: ...
    def drop_labels(self) -> None: ...

def energy_fingerprint(energies: npt.NDArray[np.floating]) -> bytes: ...

def find_watershed_segmentation(
    energies: npt.NDArray[np.floating],
    neighborhood: str = ...,
    *,
    parents: bool = ...,
) -> Watershed: ...

def find_minimum_energy_path(
    energies: npt.NDArray[np.floating],
    start: tuple[int, ...],
    end: tuple[int, ...],
    neighborhood: str | None = ...,
    *,
    tree: MergeTree | Watershed | None = ...,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] | None: ...

class PathProfile:
    minima: list[tuple[int, float]]
    saddles: list[tuple[int, float]]

def analyze_path_profile(
    path_energies: npt.NDArray[np.float64],
    min_persistence: float = ...,
) -> PathProfile: ...

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
    ws: Watershed
    nodes: dict[int, BasinNode]
    root: Optional[int]
    def __init__(self, ws: Watershed) -> None: ...
    @property
    def labels(self) -> npt.NDArray[np.int32] | None: ...
    @property
    def parents(self) -> npt.NDArray[np.uint16] | None: ...
    @property
    def merge_table(self) -> npt.NDArray[np.uint32] | None: ...
    @property
    def neighborhood(self) -> str: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def fingerprint(self) -> bytes: ...
    @property
    def has_labels(self) -> bool: ...
    def drop_labels(self) -> None: ...
    def basin_mask(self, bid: int) -> npt.NDArray[np.bool_]: ...
    def node(self, bid: int) -> BasinNode: ...
    def persistence(self, bid: int) -> float: ...
    def neighbors(self, bid: int) -> list[int]: ...
    def path(self, a: int, b: int) -> list[int]: ...
    def bfs(self, start: int, *, advance: Optional[Callable[[int, int], bool]] = ...) -> Iterator[tuple[int, int]]: ...
    def basin_of_point(self, index: tuple[int, ...]) -> int: ...
    def basins_containing(self, points: list[tuple[int, ...]]) -> dict[int, list[tuple[int, ...]]]: ...
    def touches_edge(self, bid: int, axis: int, side: str = ...) -> bool: ...
