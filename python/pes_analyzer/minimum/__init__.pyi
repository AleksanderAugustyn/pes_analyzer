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
    """Find all local minima on a dense N-D energy grid using the full
    3ᴺ−1 (king-move) neighborhood. A cell qualifies when no king-move
    neighbor has strictly lower energy (ties are allowed).

    See ``API.md`` (``find_minima_grid``) for the full contract.
    """
    ...
