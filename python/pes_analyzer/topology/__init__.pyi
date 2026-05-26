"""Type stubs for ``pes_analyzer.topology``."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def find_watershed_segmentation(
    energies: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int32],
    list[tuple[tuple[int, ...], float]],
    list[tuple[tuple[int, ...], float, int, int]],
]:
    """Full watershed segmentation on a dense N-D energy grid.

    See ``API.md`` (``find_watershed_segmentation``) for the full contract.
    """
    ...
