"""Unit tests for basin-led MEP classification in MapMaker_FoS_SHE_5D_GSTS.py.

The MapMaker is a standalone script, not a package, so we load it by path
(same pattern as test_mapmaker_fos.py). Python-only: no maturin rebuild needed.
"""
from __future__ import annotations

import importlib.util
import io
import pathlib

import numpy as np
import polars as pl
import pytest

from pes_analyzer.topology import (
    MergeTree,
    PathProfile,
    find_watershed_segmentation,
)

_MM_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mapmaker_test"
    / "5D_GSTS"
    / "MapMaker_FoS_SHE_5D_GSTS.py"
)


@pytest.fixture(scope="module")
def mm():
    spec = importlib.util.spec_from_file_location(
        "mapmaker_fos_she_5d_gsts", _MM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMepRejectReason:
    """Spec section 5.3: checks (a)-(c) are path physics, (d) is basin persistence."""

    LAST = 100  # half-path boundary at k=50

    def test_passes_all_checks(self, mm):
        assert mm._mep_reject_reason(10, 2.0, 0.0, 0.0, self.LAST, 1.0, 2.5) is None

    def test_below_gs_energy(self, mm):
        reason = mm._mep_reject_reason(10, -1.0, 0.0, 0.0, self.LAST, 1.0, 9.0)
        assert "below GS energy" in reason

    def test_below_previous_minimum(self, mm):
        reason = mm._mep_reject_reason(10, 1.5, 2.0, 0.0, self.LAST, 1.0, 9.0)
        assert "below previous" in reason

    def test_beyond_half_path(self, mm):
        reason = mm._mep_reject_reason(51, 2.0, 0.0, 0.0, self.LAST, 1.0, 9.0)
        assert "beyond half-path" in reason

    def test_persistence_below_floor(self, mm):
        reason = mm._mep_reject_reason(10, 2.0, 0.0, 0.0, self.LAST, 1.0, 0.4)
        assert "basin persistence" in reason
        assert "0.40" in reason

    def test_persistence_tie_passes(self, mm):
        # spec section 7: reject on <, exact equality with the floor passes
        assert mm._mep_reject_reason(10, 2.0, 0.0, 0.0, self.LAST, 1.0, 1.0) is None
