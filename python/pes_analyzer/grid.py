"""pes_analyzer.grid: dense N-D ndarray construction from sparse coords.

This is the canonical scatter-to-dense helper paired with the Rust
kernels in ``pes_analyzer.saddle`` and ``pes_analyzer.minimum``. Pure
NumPy; no polars dependency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["build_dense"]


def build_dense(
    coords: dict[str, npt.NDArray],
    values: npt.NDArray,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
    """Scatter (coord_per_row, value_per_row) data into a dense N-D ndarray.

    Parameters
    ----------
    coords
        Ordered mapping ``{axis_name: coord_per_row_1d_array}``. The
        insertion order of the dict determines the axis order of the
        output ndarray (Python 3.7+ dicts preserve insertion order). Each
        coordinate array must have length equal to ``len(values)``.
    values
        1-D array of scalars (energies or any per-row value).

    Returns
    -------
    dense
        C-contiguous ``float64`` array of shape
        ``(n_unique_axis_0, ..., n_unique_axis_{N-1})``. Missing cells
        are ``np.nan``.
    axes
        ``{axis_name: sorted_unique_values_1d_array}``, same key order
        as ``coords``.

    Notes
    -----
    - Axes with only one unique value are NOT squeezed; the caller is
      responsible for filtering active axes before calling.
    - Duplicate ``(coords, ...)`` rows: last-write-wins.

    Raises
    ------
    ValueError
        If any coord array length disagrees with ``len(values)`` or if
        ``coords`` is empty.
    """
    if not coords:
        raise ValueError("coords must contain at least one axis")

    n_rows = len(values)
    for name, arr in coords.items():
        if len(arr) != n_rows:
            raise ValueError(
                f"coord '{name}' has length {len(arr)}, "
                f"but values has length {n_rows}"
            )

    axis_names = list(coords)
    uniques: dict[str, npt.NDArray] = {
        name: np.unique(coords[name]) for name in axis_names
    }
    shape = tuple(len(uniques[name]) for name in axis_names)

    dense = np.full(shape, np.nan, dtype=np.float64)
    idx_per_axis = tuple(
        np.searchsorted(uniques[name], coords[name]) for name in axis_names
    )
    dense[idx_per_axis] = np.asarray(values, dtype=np.float64)
    return dense, uniques
