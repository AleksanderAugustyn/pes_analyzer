"""Persistence-pruned merge-tree analysis for the watershed segmentation.

These helpers are pure Python; the heavy lifting (the flood) is in the
Rust kernel ``find_watershed_segmentation``. See ``API.md`` for the full
contract.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "compute_persistence",
    "prune_merge_tree",
]


def compute_persistence(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
) -> npt.NDArray[np.float64]:
    """Per-basin topological persistence.

    The deepest basin (``basins[0]``) never dies; its persistence is
    ``+inf``. Every other basin appears as the ``shallower`` end of
    exactly one merge event, and its persistence is the energy gap
    between that saddle and the basin's own minimum.

    Parameters
    ----------
    basins
        Output of :func:`find_watershed_segmentation`.
    merges
        Output of :func:`find_watershed_segmentation`.

    Returns
    -------
    persistence
        ``float64`` array of length ``len(basins)``; ``persistence[i]``
        is the persistence of basin ``i``.
    """
    persistence = np.full(len(basins), np.inf, dtype=np.float64)
    for _saddle_idx, saddle_e, _deeper, shallower in merges:
        persistence[shallower] = saddle_e - basins[shallower][1]
    return persistence


def prune_merge_tree(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
    threshold: float,
) -> tuple[list[int], list[tuple[tuple[int, ...], float, int, int]]]:
    """Drop basins with persistence below ``threshold`` and their death events.

    Parameters
    ----------
    basins, merges
        Outputs of :func:`find_watershed_segmentation`.
    threshold
        Minimum persistence (same units as energy) for a basin to survive.

    Returns
    -------
    surviving
        Basin IDs (indices into ``basins``) that pass the threshold, in
        ascending order. Always includes basin 0 (its persistence is
        ``+inf``).
    kept
        The subset of ``merges`` where the ``shallower`` basin survives.
        Input order (= ascending saddle energy) is preserved.
    """
    persistence = compute_persistence(basins, merges)
    surviving = [i for i, p in enumerate(persistence) if p >= threshold]
    survivor_set = set(surviving)
    kept = [m for m in merges if m[3] in survivor_set]
    return surviving, kept


