"""pes_analyzer: Rust-backed PES analysis.

The compiled extension lives at ``pes_analyzer._native``. This module
re-exports the user-facing submodules so callers can use the spec-defined
import paths::

    from pes_analyzer.saddle import find_iwf_grid
"""

from pes_analyzer._native import saddle  # noqa: F401

__all__ = ["saddle"]
