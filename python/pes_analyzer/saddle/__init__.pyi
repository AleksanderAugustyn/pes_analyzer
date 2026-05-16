"""Type stubs for ``pes_analyzer.saddle``.

The submodule itself is registered at runtime by the Rust extension via
``sys.modules`` patching; this stub exists purely so static type checkers
can resolve ``from pes_analyzer.saddle import find_iwf_grid``.
"""

from typing import overload

import numpy as np
import numpy.typing as npt

@overload
def find_iwf_grid(
    energies: npt.NDArray[np.float64],
    start: tuple[int, ...],
    end: tuple[int, ...],
) -> tuple[tuple[int, ...], float] | None:
    """Find the saddle point between ``start`` and ``end`` on a pre-computed
    energy grid using the imaginary water flow (watershed) algorithm.

    See ``API.md`` (``find_iwf_grid``) for the full contract.
    """
    ...
