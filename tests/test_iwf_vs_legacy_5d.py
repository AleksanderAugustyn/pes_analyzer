"""Side-by-side equivalence test: legacy Python IWF loop vs pes_analyzer.saddle.find_iwf_grid.

Skipped automatically when ``pes_Z120_N180.parquet`` is absent (matches the
file-presence skip convention from ``test_iwf_regression.py``).

The legacy algorithm is preserved here as ``_legacy_find_saddle_5d`` — it is a
verbatim copy (minus print/CriticalPoint plumbing) of the body that lived in
``MapMaker_FoS_SHE.find_saddle_point_5d`` before the substitution.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pes_analyzer.saddle import find_iwf_grid

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "pes_Z120_N180.parquet"


class _LegacyDSU:
    """Verbatim copy of DisjointSetUnion from MapMaker_FoS_SHE.py:671."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1


def _legacy_neighbors_5d(key, shape, energy_map):
    """Verbatim copy of Grid5D.get_neighbors_5d from MapMaker_FoS_SHE.py:165."""
    ic, ia3, ia4, ia5, ia6 = key
    nc, na3, na4, na5, na6 = shape
    neighbors = []
    deltas = [
        (-1, 0, 0, 0, 0), (1, 0, 0, 0, 0),
        (0, -1, 0, 0, 0), (0, 1, 0, 0, 0),
        (0, 0, -1, 0, 0), (0, 0, 1, 0, 0),
        (0, 0, 0, -1, 0), (0, 0, 0, 1, 0),
        (0, 0, 0, 0, -1), (0, 0, 0, 0, 1),
    ]
    for dc, da3, da4, da5, da6 in deltas:
        n = (ic + dc, ia3 + da3, ia4 + da4, ia5 + da5, ia6 + da6)
        if (0 <= n[0] < nc and 0 <= n[1] < na3 and
                0 <= n[2] < na4 and 0 <= n[3] < na5 and
                0 <= n[4] < na6 and n in energy_map):
            neighbors.append(n)
    return neighbors


def _legacy_find_saddle_5d(energy_map, shape, start_key, end_key):
    """Verbatim copy of the IWF watershed loop from the pre-substitution
    ``find_saddle_point_5d`` (MapMaker_FoS_SHE.py:1316-1357), stripped of
    print/CriticalPoint plumbing and reformulated to take/return raw tuples.

    Returns ``(saddle_key, saddle_energy)`` or ``None``.
    """
    valid_items = [(k, e) for k, e in energy_map.items() if not np.isnan(e)]
    points = sorted(valid_items, key=lambda x: x[1])

    key_to_id = {key: idx for idx, (key, _) in enumerate(points)}
    n_points = len(points)

    dsu = _LegacyDSU(n_points)
    processed = set()

    start_id = key_to_id.get(start_key)
    end_id = key_to_id.get(end_key)
    if start_id is None or end_id is None:
        return None

    for key, energy in points:
        current_id = key_to_id[key]
        processed.add(key)
        for nb in _legacy_neighbors_5d(key, shape, energy_map):
            if nb in processed:
                dsu.union(current_id, key_to_id[nb])
        if dsu.find(start_id) == dsu.find(end_id):
            return key, float(energy)

    return None


def _basin_min_keys(grid):
    """Pick two reproducible (start, end) pairs by c-range basin minima.

    Returns
    -------
    inner_pair : (tuple, tuple)
        (lowest-E cell with c < 1.3, lowest-E cell with 1.3 <= c < 1.7)
    outer_pair : (tuple, tuple)
        (lowest-E cell with 1.3 <= c < 1.7, lowest-E cell with c >= 1.7)
    """
    keys = list(grid.energy_map.keys())
    c_axis = grid.unique_c

    low_c   = [k for k in keys if c_axis[k[0]] < 1.3]
    mid_c   = [k for k in keys if 1.3 <= c_axis[k[0]] < 1.7]
    high_c  = [k for k in keys if c_axis[k[0]] >= 1.7]
    if not (low_c and mid_c and high_c):
        pytest.skip("Parquet grid does not span the expected c ranges")

    low_min  = min(low_c,  key=lambda k: grid.energy_map[k])
    mid_min  = min(mid_c,  key=lambda k: grid.energy_map[k])
    high_min = min(high_c, key=lambda k: grid.energy_map[k])
    return (low_min, mid_min), (mid_min, high_min)


@pytest.fixture(scope="module")
def real_grid():
    if not PARQUET_PATH.exists():
        pytest.skip(f"{PARQUET_PATH.name} not present")
    from MapMaker_FoS_SHE import build_5d_energy_grid
    df = pl.read_parquet(PARQUET_PATH)
    return build_5d_energy_grid(df)


def _assert_legacy_matches_rust(grid, start_key, end_key):
    shape = (grid.nc, grid.na3, grid.na4, grid.na5, grid.na6)
    legacy = _legacy_find_saddle_5d(grid.energy_map, shape, start_key, end_key)
    new = find_iwf_grid(grid.energies, start_key, end_key)

    assert legacy is not None, "legacy IWF returned None on a connected pair"
    assert new is not None, "find_iwf_grid returned None on a connected pair"

    legacy_idx, legacy_E = legacy
    new_idx, new_E = new

    assert new_idx == legacy_idx, (
        f"saddle index drifted: rust={new_idx}, legacy={legacy_idx}"
    )
    assert abs(new_E - legacy_E) < 1e-12, (
        f"saddle energy drifted: rust={new_E!r}, legacy={legacy_E!r}"
    )


def test_iwf_matches_legacy_inner_pair(real_grid):
    (start_key, end_key), _ = _basin_min_keys(real_grid)
    _assert_legacy_matches_rust(real_grid, start_key, end_key)


def test_iwf_matches_legacy_outer_pair(real_grid):
    _, (start_key, end_key) = _basin_min_keys(real_grid)
    _assert_legacy_matches_rust(real_grid, start_key, end_key)
