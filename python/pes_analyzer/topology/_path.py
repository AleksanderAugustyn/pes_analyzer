"""1-D critical-point extraction along a minimum-energy-path profile.

Pure Python; the path itself comes from the Rust kernel
``find_minimum_energy_path``. See ``_docs/API.md`` for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["PathProfile", "analyze_path_profile"]


@dataclass(frozen=True)
class PathProfile:
    """Alternating critical points of a 1-D energy profile.

    ``minima`` and ``saddles`` hold ``(path_index, energy)`` pairs in path
    order. Along the path they alternate: minimum, saddle, minimum, ...
    Path endpoints participate like any extremum (a rising start / falling
    end is a minimum).
    """

    minima: list[tuple[int, float]]
    saddles: list[tuple[int, float]]


def _alternating_extrema(e: npt.NDArray[np.float64]) -> list[tuple[int, bool]]:
    """Indices of alternating local extrema of ``e``; bool means minimum.

    Plateaus contribute their last cell (the index where the direction
    change is observed). A fully flat profile is a single minimum at 0.
    """
    k = len(e)
    if k == 1:
        return [(0, True)]
    diffs = np.sign(np.diff(e))
    nonzero = np.flatnonzero(diffs)
    if len(nonzero) == 0:
        return [(0, True)]
    ext: list[tuple[int, bool]] = []
    cur_dir = diffs[nonzero[0]]
    ext.append((0, cur_dir > 0))  # rising start => start is a minimum
    for j in nonzero[1:]:
        if diffs[j] != cur_dir:
            ext.append((int(j), diffs[j] > 0))
            cur_dir = diffs[j]
    ext.append((k - 1, cur_dir < 0))  # falling end => end is a minimum
    return ext


def analyze_path_profile(
    path_energies: npt.NDArray[np.float64],
    min_persistence: float = 0.0,
) -> PathProfile:
    """Extract minima and saddles of a path energy profile.

    Builds the alternating extrema sequence of the 1-D profile, then
    repeatedly cancels the adjacent (minimum, saddle) pair with the
    smallest energy gap until every surviving pair clears
    ``min_persistence``. The profile's global minimum is never cancelled.

    Parameters
    ----------
    path_energies
        1-D float array, e.g. the second output of
        ``find_minimum_energy_path``. Must be NaN-free and non-empty.
    min_persistence
        Energy gap (same units as the profile) below which an adjacent
        minimum/saddle pair counts as noise and is removed. ``0.0`` keeps
        every strict extremum.

    Returns
    -------
    PathProfile
        Surviving minima and saddles, alternating in path order.
    """
    e = np.asarray(path_energies, dtype=np.float64)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("path_energies must be a non-empty 1-D array")
    if np.isnan(e).any():
        raise ValueError("path_energies must not contain NaN")

    ext = _alternating_extrema(e)
    while len(ext) > 2:
        energies_of = [float(e[i]) for i, _ in ext]
        gmin_pos = min(
            (p for p, (_, is_min) in enumerate(ext) if is_min),
            key=lambda p: energies_of[p],
        )
        best_pos, best_gap = None, np.inf
        for p in range(len(ext) - 1):
            if p == gmin_pos or p + 1 == gmin_pos:
                continue  # the global minimum is never cancelled
            gap = abs(energies_of[p + 1] - energies_of[p])
            if gap < best_gap:
                best_pos, best_gap = p, gap
        if best_pos is None or best_gap >= min_persistence:
            break
        del ext[best_pos : best_pos + 2]

    minima = [(i, float(e[i])) for i, is_min in ext if is_min]
    saddles = [(i, float(e[i])) for i, is_min in ext if not is_min]
    return PathProfile(minima=minima, saddles=saddles)
