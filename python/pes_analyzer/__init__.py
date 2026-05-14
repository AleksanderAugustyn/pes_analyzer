"""pes_analyzer: Rust-backed PES analysis.

The compiled extension lives at ``pes_analyzer._native``. This module
re-exports the user-facing submodules so callers can use the spec-defined
import paths::

    from pes_analyzer.saddle  import find_iwf_grid
    from pes_analyzer.minimum import find_minima_grid
    from pes_analyzer.grid    import build_dense
"""

from pes_analyzer._native import minimum, saddle  # noqa: F401

# `pes_analyzer.grid` is a pure-Python submodule added in Task 6 of the
# v0.2 plan. The re-export line `from pes_analyzer import grid` is added
# at that point so that this docstring's import path resolves.

__all__ = ["minimum", "saddle"]
