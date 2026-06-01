"""pes_analyzer.topology: watershed segmentation and merge-tree analysis.

The Rust-backed kernel ``find_watershed_segmentation`` is re-exported from
``pes_analyzer._native.topology``; the pure-Python analysis layer provides
``compute_persistence`` / ``prune_merge_tree`` (in ``_tree.py``) and the
traversable ``MergeTree`` (in ``merge_tree.py``).
"""

from __future__ import annotations

# Attribute access on the compiled `_native` extension. `_native` is a
# single-file extension (not a package), so `from pes_analyzer._native.topology
# import ...` does not work without `sys.modules` patching; PyO3 has already
# registered `topology` as an attribute of `_native` via `add_submodule`.
from pes_analyzer._native import topology as _native_topology

from ._tree import (
    compute_persistence,
    prune_merge_tree,
)
from .merge_tree import BasinNode, MergeTree

find_watershed_segmentation = _native_topology.find_watershed_segmentation

__all__ = [
    "find_watershed_segmentation",
    "compute_persistence",
    "prune_merge_tree",
    "MergeTree",
    "BasinNode",
]
