"""Flood state returned by ``find_watershed_segmentation``.

The ``Watershed`` object is the single owner of the grid-sized arrays
(``labels``, ``parents``, ``merge_table``); ``MergeTree`` only refers to it.
``drop_labels()`` releases them for every holder at once.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pes_analyzer._native import topology as _native_topology

__all__ = ["Watershed", "energy_fingerprint", "find_watershed_segmentation"]

_FINGERPRINT_SAMPLES = 1 << 20


def energy_fingerprint(energies: np.ndarray) -> bytes:
    """16-byte blake2b over shape, dtype and every k-th cell, k = max(1, size // 2**20).

    Cheap enough for 10^9 cells (~1M samples) and sensitive to a different
    grid of the same shape; not a cryptographic identity of the full array.
    """
    energies = np.asarray(energies)
    flat = np.ascontiguousarray(energies).reshape(-1)
    k = max(1, flat.size // _FINGERPRINT_SAMPLES)
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(tuple(energies.shape)).encode())
    h.update(energies.dtype.str.encode())
    h.update(np.ascontiguousarray(flat[::k]).tobytes())
    return h.digest()


@dataclass
class Watershed:
    """Output of :func:`find_watershed_segmentation`.

    ``labels``/``parents``/``merge_table`` are ``None`` after :meth:`drop_labels`.
    ``basins`` and ``merges`` keep the 0.9.0 tuple formats.
    """

    labels: np.ndarray | None
    basins: list[tuple[tuple[int, ...], float]]
    merges: list[tuple[tuple[int, ...], float, int, int]]
    neighborhood: str
    parents: np.ndarray | None
    merge_table: np.ndarray | None
    dtype: np.dtype
    fingerprint: bytes

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    def drop_labels(self) -> None:
        """Release the grid-sized arrays (4N labels, 2N parents) and the merge table."""
        self.labels = None
        self.parents = None
        self.merge_table = None


def find_watershed_segmentation(
    energies: np.ndarray,
    neighborhood: str = "von_neumann",
    *,
    parents: bool = False,
) -> Watershed:
    """Full watershed flood of a dense N-D grid. See ``_docs/API.md``."""
    energies = np.asarray(energies)
    labels, parents_arr, basins, merges, merge_table = _native_topology.find_watershed_segmentation(
        energies, neighborhood, parents
    )
    labels.flags.writeable = False
    merge_table.flags.writeable = False
    if parents_arr is not None:
        parents_arr.flags.writeable = False
    return Watershed(
        labels=labels,
        basins=basins,
        merges=merges,
        neighborhood=neighborhood,
        parents=parents_arr,
        merge_table=merge_table,
        dtype=energies.dtype,
        fingerprint=energy_fingerprint(energies),
    )
