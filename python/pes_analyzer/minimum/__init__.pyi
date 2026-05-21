"""Type stubs for ``pes_analyzer.minimum``.

The submodule itself is registered at runtime by the Rust extension via
``sys.modules`` patching; this stub exists purely so static type checkers
can resolve ``from pes_analyzer.minimum import find_minima_grid``.
"""

import numpy as np
import numpy.typing as npt

def find_minima_grid(
    energies: npt.NDArray[np.float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    """Find all local minima on a dense N-D energy grid using the Chebyshev
    box of half-width ``neighborhood_range`` (the ``(2r+1)^N − 1`` stencil).
    The default ``neighborhood_range = 1`` is the classic 3^N − 1 king-move
    stencil. A cell qualifies when no neighbor in the stencil has strictly
    lower energy (ties are allowed).

    ``neighborhood_range`` must be in ``[1, 5]``; values outside raise
    ``ValueError``. The argument is keyword-only.

    ``confirm_range`` (keyword-only, default ``None``) optionally runs a
    second pass that re-checks each candidate against the ``confirm_range``-
    wide stencil. When set, it must satisfy ``confirm_range ∈ [1, 5]`` and
    ``confirm_range >= neighborhood_range``. The recommended idiom for
    wide-stencil minima is ``neighborhood_range=1, confirm_range=R``: the
    fast ``r = 1`` pass culls most cells before the expensive wide check.
    The result is identical to ``neighborhood_range=R``.

    See ``API.md`` (``find_minima_grid``) for the full contract.
    """
    ...
