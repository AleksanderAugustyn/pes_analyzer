"""Type stubs for ``pes_analyzer.minimum``.

The submodule itself is registered at runtime by the Rust extension via
``sys.modules`` patching; this stub exists purely so static type checkers
can resolve ``from pes_analyzer.minimum import find_minima_grid``.
"""

import numpy as np
import numpy.typing as npt

def find_minima_grid(
    energies: npt.NDArray[np.float64],
) -> list[tuple[tuple[int, ...], float]]:
    """Find all strict local minima on a dense N-D energy grid using the
    full 3ᴺ−1 (king-move) neighborhood.

    See ``docs/SPEC.md`` §3.6 for the full contract.
    """
    ...
