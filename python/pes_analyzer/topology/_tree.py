"""Persistence-pruned merge-tree analysis for the watershed segmentation.

These helpers are pure Python; the heavy lifting (the flood) is in the
Rust kernel ``find_watershed_segmentation``. See ``API.md`` for the full
contract.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "compute_persistence",
    "prune_merge_tree",
    "identify_critical_points",
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


def identify_critical_points(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
    threshold: float,
    *,
    gs_disqualifier: Optional[Callable[[int], bool]] = None,
) -> dict[str, object]:
    """Walk the pruned merge tree outward from the ground state.

    Uses a Prim-style edge relaxation on the (already-a-)tree: at each
    step, the lowest-saddle-energy edge connecting a visited basin to an
    unvisited one is consumed. This is **not** equivalent to a linear
    scan of the pruned merges in saddle-energy order, because the outer
    barrier may be lower than the inner barrier (which is physically
    common for actinides).

    Parameters
    ----------
    basins, merges
        Outputs of :func:`find_watershed_segmentation`.
    threshold
        Persistence threshold passed to :func:`prune_merge_tree`.
    gs_disqualifier
        Optional callable ``basin_id -> bool``. Returning ``True`` marks
        the basin as ineligible to be the ground state, and the walk
        falls back to the next-deepest surviving basin. Used by callers
        that need to apply a physical constraint such as "GS must lie
        within the normal-shape region."

    Returns
    -------
    A dict with keys ``ground_state``, ``secondary_minimum``,
    ``inner_saddle``, ``outer_saddle``, ``fission_exit``. Each value is
    either a basin ID (``int``) / merge tuple, or ``None`` if the
    pruned topology does not contain that critical point.
    """
    surviving, kept = prune_merge_tree(basins, merges, threshold)

    gs: Optional[int] = None
    for bid in surviving:
        if gs_disqualifier is None or not gs_disqualifier(bid):
            gs = bid
            break

    result: dict[str, object] = {
        "ground_state": gs,
        "secondary_minimum": None,
        "inner_saddle": None,
        "outer_saddle": None,
        "fission_exit": None,
    }
    if gs is None:
        return result

    visited = {gs}
    remaining = list(kept)
    steps: list[tuple[tuple[tuple[int, ...], float, int, int], int]] = []
    while len(steps) < 2:
        active = [
            (i, m)
            for i, m in enumerate(remaining)
            if (m[2] in visited) ^ (m[3] in visited)
        ]
        if not active:
            break
        i, m = min(active, key=lambda im: im[1][1])
        new = m[3] if m[2] in visited else m[2]
        steps.append((m, new))
        visited.add(new)
        remaining.pop(i)

    if len(steps) == 1:
        # SHE-like: single outward basin; report the saddle in the
        # inner_saddle slot for downstream-schema compatibility.
        m, new = steps[0]
        result["inner_saddle"] = m
        result["fission_exit"] = new
    elif len(steps) == 2:
        (m1, new1), (m2, new2) = steps
        result["inner_saddle"] = m1
        result["secondary_minimum"] = new1
        result["outer_saddle"] = m2
        result["fission_exit"] = new2

    return result
