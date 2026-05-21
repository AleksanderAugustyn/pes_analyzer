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
) -> list[tuple[tuple[int, ...], float]]:
    """Find all local minima on a dense N-D energy grid using the Chebyshev
    box of half-width ``neighborhood_range`` (the ``(2r+1)^N − 1`` stencil).
    The default ``neighborhood_range = 1`` is the classic 3^N − 1 king-move
    stencil. A cell qualifies when no neighbor in the stencil has strictly
    lower energy (ties are allowed).

    ``neighborhood_range`` must be in ``[1, 5]``; values outside raise
    ``ValueError``. The argument is keyword-only.

    See ``API.md`` (``find_minima_grid``) for the full contract.
    """
    ...
