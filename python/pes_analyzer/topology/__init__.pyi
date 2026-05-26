"""Type stubs for ``pes_analyzer.topology``."""

from __future__ import annotations

from typing import Callable, Optional

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

def identify_critical_points(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
    threshold: float,
    *,
    gs_disqualifier: Optional[Callable[[int], bool]] = None,
) -> dict[str, object]: ...
