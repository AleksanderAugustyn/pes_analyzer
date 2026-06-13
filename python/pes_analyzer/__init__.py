"""pes_analyzer: Rust-backed PES analysis.

The compiled extension lives at ``pes_analyzer._native``. This module
re-exports the user-facing submodules so callers can use the spec-defined
import paths::

    from pes_analyzer.saddle   import find_iwf_grid
    from pes_analyzer.extrema  import find_minima_grid, find_maxima_grid, find_extrema_grid
    from pes_analyzer.grid     import build_dense
    from pes_analyzer.topology import find_watershed_segmentation
"""

from importlib.resources import files
from pathlib import Path

from pes_analyzer._native import extrema, saddle  # noqa: F401
from pes_analyzer import grid  # noqa: F401
from pes_analyzer import topology  # noqa: F401


def docs_path() -> Path:
    """Return the path to the bundled ``_docs/`` directory.

    Uses ``importlib.resources`` so it resolves from an installed wheel as well
    as from a source checkout. Lets a consumer (or an agent) locate the bundled
    reference docs (``API.md``, ``ALGORITHMS.md``, ``USAGE.md``) programmatically.

    Returns
    -------
    pathlib.Path
        Filesystem path to ``pes_analyzer/_docs``.
    """
    return Path(str(files(__name__) / "_docs"))


__all__ = ["extrema", "saddle", "grid", "topology", "docs_path"]
