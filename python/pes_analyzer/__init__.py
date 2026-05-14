"""pes_analyzer: Rust-backed PES analysis.

The compiled extension lives at ``pes_analyzer._native``. This module
re-exports the user-facing submodules so callers can use the spec-defined
import paths::

    from pes_analyzer.saddle  import find_iwf_grid
    from pes_analyzer.minimum import find_minima_grid
    from pes_analyzer.grid    import build_dense
"""

from pes_analyzer._native import minimum, saddle  # noqa: F401
from pes_analyzer import grid  # noqa: F401

__all__ = ["minimum", "saddle", "grid"]
