"""Unit tests for the FoS physics-selection helper in MapMaker_FoS_SHE.py.

MapMaker is a standalone script, not a package, so we load it by path.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from pes_analyzer.topology import MergeTree

_MM_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mapmaker_test"
    / "MapMaker_FoS_SHE.py"
)


def _load_mapmaker():
    spec = importlib.util.spec_from_file_location("mapmaker_fos_she", _MM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chain_tree():
    """4-basin chain. axis 0 == c (row), axis 1 == a4 (col, max edge = col 4).

    basin 0: min (0,0) E=0.0  c small        -> GS
    basin 1: min (1,2) E=2.0  c mid, NO conf min (pass-through B)
    basin 2: min (2,2) E=1.0  c larger       -> SM (has conf min)
    basin 3: min (4,4) E=1.5  c largest, touches col-4 max edge -> FE
    """
    labels = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 2, 2, 3],
            [0, 0, 2, 2, 3],
            [0, 0, 2, 2, 3],
        ],
        dtype=np.int32,
    )
    basins = [((0, 0), 0.0), ((1, 2), 2.0), ((2, 2), 1.0), ((4, 4), 1.5)]
    merges = [
        ((0, 2), 6.0, 0, 1),  # basin 1 -> 0, saddle E=6 (inner)
        ((2, 1), 5.0, 1, 2),  # basin 2 -> 1, saddle E=5
        ((3, 3), 4.0, 2, 3),  # basin 3 -> 2, saddle E=4 (outer)
    ]
    return MergeTree(labels, basins, merges)


def test_select_identifies_gs_sm_fe_and_saddles():
    mm = _load_mapmaker()
    tree = _chain_tree()
    c_vals = {0: 1.0, 1: 1.2, 2: 1.3, 3: 1.8}      # all valid GS-region except by depth
    has = {0: True, 1: False, 2: True, 3: True}    # basin 1 is a pass-through
    sel = mm.select_fos_critical_points(
        tree, has_min=lambda b: has[b], c_of=lambda b: c_vals[b], a4_axis=1
    )
    assert sel["ground_state"] == 0
    assert sel["secondary_minimum"] == 2          # basin 1 skipped (no conf min)
    assert sel["fission_exit"] == 3               # touches a4 max edge
    assert sel["inner_saddle"][1] == 6.0          # highest saddle on path GS..SM
    assert sel["outer_saddle"][1] == 4.0          # highest saddle on path SM..FE


def test_select_gs_respects_c_threshold():
    mm = _load_mapmaker()
    tree = _chain_tree()
    # basin 0 is the deepest but sits at c >= 1.4 -> disqualified as GS.
    # basin 2 (c=1.0, has a confirmed min) is the next-eligible deepest -> GS.
    c_vals = {0: 1.6, 1: 1.2, 2: 1.0, 3: 1.8}
    has = {0: True, 1: False, 2: True, 3: True}
    sel = mm.select_fos_critical_points(
        tree, has_min=lambda b: has[b], c_of=lambda b: c_vals[b], a4_axis=1
    )
    assert sel["ground_state"] == 2  # fell back past the disqualified deepest basin


def test_select_returns_none_when_no_edge_basin():
    mm = _load_mapmaker()
    tree = _chain_tree()
    c_vals = {0: 1.0, 1: 1.2, 2: 1.3, 3: 1.8}
    has = {0: True, 1: False, 2: True, 3: True}
    sel = mm.select_fos_critical_points(
        tree, has_min=lambda b: has[b], c_of=lambda b: c_vals[b], a4_axis=None
    )
    assert sel["secondary_minimum"] == 2
    assert sel["fission_exit"] is None            # no a4 axis -> nothing touches the edge
