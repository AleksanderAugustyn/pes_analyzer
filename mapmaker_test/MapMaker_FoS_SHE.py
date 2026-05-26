#!/usr/bin/env python3
"""PES Map Maker with N-D Critical Point Analysis for FoS Parameterization.

Reads FoS (Fourier-over-Spheroid) parameterization data from Parquet
files with schema:
    c, a3, a4, a5, a6, a7, a8, is_valid, total_energy, mass_excess,
    macro_energy, micro_energy, surface_energy, coulomb_energy,
    proton_k, neutron_k, proton_gap, neutron_gap

Active deformation axes are auto-detected from the input: any column
with more than one unique value becomes an active axis. The pipeline
runs uniformly on the resulting dense N-D grid (2 <= N <= 7), delegating
local-minima search and saddle-point search to the Rust-backed
pes_analyzer package.

Features
--------
    1. Auto-discovers pes_ZXX_NYYY.parquet files if no specific file is given.
    2. Critical-point analysis: ground state, secondary minimum, fission
       exit, inner and outer saddles.
    3. Saves critical points to CSV.
    4. Plots c-vs-a3 and c-vs-a4 contour maps with critical points marked
       (whichever of a3/a4 is active).

Usage
-----
    python MapMaker_FoS_SHE.py                         # Process all found files
    python MapMaker_FoS_SHE.py pes_Z120_N180.parquet   # Process one file
    python MapMaker_FoS_SHE.py --help                  # Show help
"""
import argparse
import io
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TextIO

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; safe to call from worker threads.
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import numpy as np
import polars as pl
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from pes_analyzer.saddle  import find_iwf_grid
from pes_analyzer.extrema import find_minima_grid
from pes_analyzer.grid    import build_dense

# =============================================================================
# CONFIGURATION
# =============================================================================

# General settings
ASPECT_RATIO = 'equal'
DPI = 300
USE_FLOAT32 = True

# Axes that may be active. The actual active set is auto-detected per file
# from the parquet contents (any column with >1 unique value).
CANDIDATE_AXES = ('c', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8')

# =============================================================================
# SHAPE PARAMETER LIMITS
# =============================================================================
C_LIMITS = (1.0, 2.0)
A3_LIMITS = (0.0, 0.25)
A4_LIMITS = (-0.15, 0.36)
A5_LIMITS = (-0.05, 0.05)
A6_LIMITS = (-0.05, 0.05)
A7_LIMITS = (None, None)
A8_LIMITS = (None, None)

PARAMETER_LIMITS = {
    'c': C_LIMITS,
    'a3': A3_LIMITS,
    'a4': A4_LIMITS,
    'a5': A5_LIMITS,
    'a6': A6_LIMITS,
    'a7': A7_LIMITS,
    'a8': A8_LIMITS,
}

# =============================================================================
# ANALYSIS THRESHOLDS
# =============================================================================
GROUND_STATE_C_THRESHOLD = 1.25
SECONDARY_MINIMUM_C_MAX_THRESHOLD = 1.45
SECONDARY_MINIMUM_A3_MAX_THRESHOLD = 0.10
SECONDARY_MINIMUM_A4_MAX_THRESHOLD = 0.20

# When True (default), run the full GS / SM / Inner Saddle / Outer Saddle / FE
# pipeline. When False, skip the secondary minimum and outer saddle searches
# entirely and report a single saddle between GS and FE. Use False for SHE
# where there is no genuine fission isomer and the SM search picks up noise.
SECOND_MINIMUM_SEARCH = False

# Lower-c offset used by the Fission Exit search when SECOND_MINIMUM_SEARCH=False.
# In that mode there is no secondary minimum to anchor to, so FE is the global
# minimum over cells with c > gs.c + FISSION_EXIT_GS_C_OFFSET.
FISSION_EXIT_GS_C_OFFSET = 0.2


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CriticalPointType(Enum):
    """Types of critical points on a potential energy surface."""
    GROUND_STATE = auto()
    SECONDARY_MINIMUM = auto()
    FIRST_SADDLE = auto()
    SECOND_SADDLE = auto()
    FISSION_EXIT = auto()


@dataclass
class GridPoint:
    """Represents a point on the deformation grid with all coordinates and energy.

    The per-axis fields (c, a3, ..., a8) are a flat convenience view for
    CSV writers and plotting code. The N-D `index` and the `coords` dict
    are the authoritative sources of truth.
    """
    c:  float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    a6: float = 0.0
    a7: float = 0.0
    a8: float = 0.0
    total_energy: float = np.inf
    mass_excess:    float = np.nan
    macro_energy:   float = np.nan
    micro_energy:   float = np.nan
    surface_energy: float = np.nan
    coulomb_energy: float = np.nan
    valid: bool = False
    index:  tuple[int, ...] = field(default_factory=tuple)
    coords: dict[str, float] = field(default_factory=dict)


@dataclass
class CriticalPoint:
    """Represents a critical point found during analysis."""
    point_type: CriticalPointType
    name: str
    point: GridPoint = field(default_factory=GridPoint)
    found: bool = False


@dataclass
class NucleusInfo:
    """Information about the nucleus being analyzed."""
    Z: int = 0
    N: int = 0
    A: int = 0
    symbol: str = ""
    isotope_label: str = ""


@dataclass
class FissionBarriers:
    """Fission barrier heights relative to ground state."""
    inner_barrier: float = np.nan  # First saddle - ground state
    outer_barrier: float = np.nan  # Second saddle - ground state
    nucleus: NucleusInfo = field(default_factory=NucleusInfo)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

ELEMENT_SYMBOLS = {
    90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds',
    111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og', 119: 'Uue', 120: 'Ubn',
}


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def extract_nucleus_info(filename: str | Path) -> NucleusInfo:
    """Extract nucleus information (Z, N, A) from filename."""
    stem = Path(filename).stem
    info = NucleusInfo()

    z_match = re.search(r'Z(\d+)', stem, re.IGNORECASE)
    n_match = re.search(r'N(\d+)', stem, re.IGNORECASE)

    if z_match and n_match:
        info.Z = int(z_match.group(1))
        info.N = int(n_match.group(1))
        info.A = info.Z + info.N
        info.symbol = ELEMENT_SYMBOLS.get(info.Z, f'Z{info.Z}')
        info.isotope_label = f'{info.A}{info.symbol}'

    return info


def find_pes_files() -> list[Path]:
    """Find all PES parquet files matching pes_ZXX_NYYY.parquet pattern."""
    parquet_files = list(Path('.').glob('pes_Z*_N*.parquet'))

    if not parquet_files:
        # Also try alternate patterns
        parquet_files = list(Path('.').glob('*Z*_N*.parquet'))

    # Sort by Z and N
    def extract_z_n(filepath: Path) -> tuple[int, int]:
        stem = filepath.stem
        try:
            z_match = re.search(r'Z(\d+)', stem, re.IGNORECASE)
            n_match = re.search(r'N(\d+)', stem, re.IGNORECASE)
            z = int(z_match.group(1)) if z_match else 0
            n = int(n_match.group(1)) if n_match else 0
            return z, n
        except (AttributeError, ValueError):
            return 0, 0

    return sorted(parquet_files, key=extract_z_n)


def apply_parameter_limits(df: pl.DataFrame, out: TextIO = sys.stdout) -> pl.DataFrame:
    """Apply parameter limits to filter the DataFrame."""
    initial_count = len(df)
    active_limits = []
    filters = []

    for param, (pmin, pmax) in PARAMETER_LIMITS.items():
        if param not in df.columns:
            continue
        if pmin is not None:
            filters.append(pl.col(param) >= pmin)
            active_limits.append(f"{param} >= {pmin}")
        if pmax is not None:
            filters.append(pl.col(param) <= pmax)
            active_limits.append(f"{param} <= {pmax}")

    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter & f
        df = df.filter(combined_filter)

    filtered_count = len(df)
    if active_limits:
        print(f"    Applied parameter limits:", file=out)
        for limit in active_limits:
            print(f"      - {limit}", file=out)
        print(f"    Filtered: {initial_count:,} -> {filtered_count:,} points "
              f"({initial_count - filtered_count:,} removed)", file=out)

    return df


def detect_active_axes(df: pl.DataFrame) -> tuple[str, ...]:
    """Return the tuple of axis names with more than one unique value.

    Axis order follows CANDIDATE_AXES so `c` is always first. Axes
    present in the file but constant (e.g. a7=a8=0.0 in current data)
    are excluded.
    """
    return tuple(
        name for name in CANDIDATE_AXES
        if name in df.columns and df[name].n_unique() > 1
    )


def build_grids(
    df: pl.DataFrame, active_axes: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Dense N-D energy grid + component grids + inactive-axis constants.

    Parameters
    ----------
    df
        DataFrame containing at least the active axes and 'total_energy'.
    active_axes
        Output of detect_active_axes(df); insertion order is the axis
        order of the returned ndarrays.

    Returns
    -------
    energies
        (n_a0, n_a1, ..., n_aN) float64, NaN at missing cells.
    axes
        {axis_name: sorted unique values along that axis}.
    inactive_axes
        {axis_name: constant_value} for every CANDIDATE_AXES entry that
        is in the file but not active. Used to fill CSV columns.
    components
        {'mass_excess': ndarray, 'macro_energy': ndarray, ...}, one per
        present energy-component column. Same shape as `energies`.
    """
    coords = {name: df[name].to_numpy() for name in active_axes}
    values = df['total_energy'].to_numpy()
    energies, axes = build_dense(coords, values)

    components: dict[str, np.ndarray] = {}
    for cname in ('mass_excess', 'macro_energy', 'micro_energy',
                  'surface_energy', 'coulomb_energy'):
        if cname in df.columns:
            cvals = df[cname].to_numpy()
            comp_dense, _ = build_dense(coords, cvals)
            components[cname] = comp_dense

    inactive_axes: dict[str, float] = {}
    for name in CANDIDATE_AXES:
        if name not in active_axes and name in df.columns:
            inactive_axes[name] = float(df[name].unique().to_numpy()[0])

    return energies, axes, inactive_axes, components


# =============================================================================
# FILE I/O
# =============================================================================

def read_parquet_file(filename: str | Path, out: TextIO = sys.stdout) -> pl.DataFrame:
    """Read the Parquet file and extract relevant columns for analysis."""
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if filepath.suffix != '.parquet':
        raise ValueError(f"Expected a Parquet file, got: {filepath.suffix}")

    needed_columns = ['c', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'is_valid',
                      'total_energy', 'mass_excess', 'macro_energy', 'micro_energy',
                      'surface_energy', 'coulomb_energy']

    print(f"  Reading Parquet file: {filepath.name}", file=out)
    print(f"    File size: {get_file_size_mb(filepath):.1f} MB", file=out)
    start_time = time.time()

    try:
        # Get schema to check available columns
        schema = pl.read_parquet_schema(filepath)
        available_columns = list(schema.keys())
        print(f"    Parquet file contains {len(available_columns)} columns", file=out)

        # Use available columns
        columns_to_read = [col for col in needed_columns if col in available_columns]
        missing = [col for col in needed_columns if col not in available_columns]
        if missing:
            print(f"    Note: Missing columns (will be NaN): {missing}", file=out)

        df = pl.read_parquet(filepath, columns=columns_to_read)
        read_time = time.time() - start_time
        print(f"    Read {len(df):,} rows in {read_time:.1f} seconds", file=out)

        # Filter valid points
        initial_count = len(df)
        if 'is_valid' in df.columns:
            df = df.filter(pl.col('is_valid') == True)
            filtered_count = len(df)
            print(f"    Filtered to {filtered_count:,} valid points "
                  f"({initial_count - filtered_count:,} invalid removed)", file=out)
            df = df.drop('is_valid')

        # Filter out rows with extremely low total_energy (likely invalid)
        if 'total_energy' in df.columns:
            before_energy_filter = len(df)
            df = df.filter(pl.col('total_energy') >= -300)
            after_energy_filter = len(df)
            if before_energy_filter != after_energy_filter:
                print(f"    Filtered out {before_energy_filter - after_energy_filter:,} points "
                      f"with total_energy < -300 MeV", file=out)

        # Apply parameter limits
        df = apply_parameter_limits(df, out=out)

        # Convert to float32 if configured
        if USE_FLOAT32:
            float_cols = ['c', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8',
                          'total_energy', 'mass_excess', 'macro_energy', 'micro_energy',
                          'surface_energy', 'coulomb_energy']
            cast_exprs = [
                pl.col(col).cast(pl.Float32)
                for col in float_cols
                if col in df.columns
            ]
            if cast_exprs:
                df = df.with_columns(cast_exprs)

        # Fill missing columns with NaN
        for col in needed_columns:
            if col not in df.columns and col != 'is_valid':
                df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

        return df

    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {str(e)}")


# =============================================================================
# MINIMIZATION AND GRID OPERATIONS
# =============================================================================

def minimize_to_2d(
    energies: np.ndarray,
    axes: dict[str, np.ndarray],
    x_axis: str,
    y_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project N-D energies down to a 2D (x_axis, y_axis) surface by
    taking the per-(x, y) minimum over all remaining axes.

    Returns
    -------
    xv, yv : 1-D arrays of x_axis / y_axis coordinate values.
    energy_2d : (nx, ny) float64 array of minimized energies; NaN where
                the entire (x, y) column is all-NaN.
    argmin_flat : (nx, ny) int array - index into the flattened
                  "remaining axes" stack identifying the minimizing
                  cell. Apply the same transpose+reshape to any parallel
                  N-D ndarray and gather with np.take_along_axis(...,
                  argmin_flat[..., None], -1).squeeze(-1). Values are
                  undefined where energy_2d is NaN; caller must mask.

    For the 2-D edge case (only x_axis and y_axis active), the
    "remaining axes" stack has length 1 and argmin_flat is uniformly 0.
    """
    axes_idx = {n: i for i, n in enumerate(axes)}
    order = [axes_idx[x_axis], axes_idx[y_axis]] + [
        i for i, n in enumerate(axes) if n not in (x_axis, y_axis)
    ]
    moved = energies.transpose(order)
    nx, ny = moved.shape[0], moved.shape[1]
    flat = moved.reshape(nx, ny, -1)
    all_nan = np.all(np.isnan(flat), axis=-1)
    flat_safe = np.where(np.isnan(flat), np.inf, flat)
    argmin_flat = np.argmin(flat_safe, axis=-1)
    energy_2d = np.take_along_axis(flat, argmin_flat[..., None], -1).squeeze(-1)
    energy_2d[all_nan] = np.nan
    return axes[x_axis], axes[y_axis], energy_2d, argmin_flat


def save_minimized_data_2d(
    xv: np.ndarray,
    yv: np.ndarray,
    e2d: np.ndarray,
    argmin_flat: np.ndarray,
    axes: dict[str, np.ndarray],
    inactive_axes: dict[str, float],
    x_axis: str,
    y_axis: str,
    output_name: str,
    output_dir: str = 'minimized_data_FoS',
    out: TextIO = sys.stdout,
) -> Path:
    """Write the minimized (x_axis, y_axis) CSV.

    Column order matches the legacy save_minimized_data:
        c, <y_axis>, <ax_min for each ax in (a3,a4,a5,a6,a7,a8) - {y_axis}>,
        total_energy_min

    Inactive axes contribute their constant value from `inactive_axes`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    rest_names = [n for n in axes if n not in (x_axis, y_axis)]
    rest_shape = tuple(len(axes[n]) for n in rest_names)

    if rest_shape:
        unraveled = np.unravel_index(argmin_flat, rest_shape)
        rest_value_at_min = {
            n: axes[n][unraveled[i]] for i, n in enumerate(rest_names)
        }
    else:
        rest_value_at_min = {}

    minimized_axes = [ax for ax in ('a3', 'a4', 'a5', 'a6', 'a7', 'a8')
                      if ax != y_axis]

    mask = ~np.isnan(e2d)
    x_grid, y_grid = np.meshgrid(xv, yv, indexing='ij')

    columns: dict[str, np.ndarray] = {
        x_axis: x_grid[mask].astype(np.float64),
        y_axis: y_grid[mask].astype(np.float64),
    }
    for ax in minimized_axes:
        col_name = f'{ax}_min'
        if ax in rest_value_at_min:
            columns[col_name] = rest_value_at_min[ax][mask].astype(np.float64)
        elif ax in inactive_axes:
            columns[col_name] = np.full(int(mask.sum()), inactive_axes[ax],
                                        dtype=np.float64)
        else:
            columns[col_name] = np.zeros(int(mask.sum()), dtype=np.float64)
    columns['total_energy_min'] = e2d[mask].astype(np.float64)

    out_df = pl.DataFrame(columns)
    out_df = out_df.sort([x_axis, y_axis])
    suffix = '_minimized.csv' if y_axis == 'a4' else f'_minimized_c{y_axis}.csv'
    output_file = output_path / f'{output_name}{suffix}'
    out_df.write_csv(output_file, float_precision=6)
    print(f"    Saved minimized data: {output_file}", file=out)
    return output_file


# =============================================================================
# CRITICAL POINT ANALYSIS
# =============================================================================

def _format_coords(idx: tuple[int, ...], axes: dict[str, np.ndarray]) -> str:
    """Format an N-D index as a human-readable physical coordinate string.

    Example: '(c=1.2000, a3=0.0500, a4=0.0900)'.
    """
    parts = [f"{name}={float(axes[name][idx[i]]):.4f}"
             for i, name in enumerate(axes)]
    return "(" + ", ".join(parts) + ")"


def index_to_gridpoint(
    idx: tuple[int, ...],
    axes: dict[str, np.ndarray],
    inactive_axes: dict[str, float],
    components: dict[str, np.ndarray],
    energy: float,
) -> GridPoint:
    """Build a GridPoint from an N-D index into the dense grid.

    Per-axis fields (c, a3, ..., a8) are populated from `axes` for
    active axes and from `inactive_axes` for present-but-constant axes;
    any axis not in either dict stays at its dataclass default (0.0).

    Energy components are looked up at the same N-D index in each
    components ndarray.
    """
    coords = {name: float(axes[name][i]) for name, i in zip(axes, idx)}
    flat = {**inactive_axes, **coords}

    def comp(name: str) -> float:
        if name in components:
            return float(components[name][idx])
        return float('nan')

    return GridPoint(
        c =flat.get('c',  0.0),
        a3=flat.get('a3', 0.0),
        a4=flat.get('a4', 0.0),
        a5=flat.get('a5', 0.0),
        a6=flat.get('a6', 0.0),
        a7=flat.get('a7', 0.0),
        a8=flat.get('a8', 0.0),
        total_energy=float(energy),
        mass_excess   = comp('mass_excess'),
        macro_energy  = comp('macro_energy'),
        micro_energy  = comp('micro_energy'),
        surface_energy= comp('surface_energy'),
        coulomb_energy= comp('coulomb_energy'),
        valid=True,
        index=tuple(int(i) for i in idx),
        coords=coords,
    )


def find_ground_state_nd(
    all_minima: list[tuple[tuple[int, ...], float]],
    axes: dict[str, np.ndarray],
    energies: np.ndarray,
    c_threshold: float,
) -> tuple[tuple[int, ...] | None, float]:
    """Pick the ground-state cell from sorted local minima.

    Tries the global energy minimum across all cells (not just minima)
    first; if it satisfies c <= c_threshold, returns it. Otherwise falls
    back to the lowest local minimum with c <= c_threshold.

    Matches the 5D pipeline semantics (which match the C++ reference).

    Returns
    -------
    (idx, energy) - idx is None if no candidate satisfies the threshold.
    """
    if np.all(np.isnan(energies)):
        return None, float('nan')
    flat_argmin = int(np.nanargmin(energies))
    global_idx = np.unravel_index(flat_argmin, energies.shape)
    global_c = float(axes['c'][global_idx[list(axes).index('c')]])
    global_e = float(energies[global_idx])
    if global_c <= c_threshold:
        return tuple(int(i) for i in global_idx), global_e

    c_axis_pos = list(axes).index('c')
    for idx, energy in all_minima:
        c_val = float(axes['c'][idx[c_axis_pos]])
        if c_val <= c_threshold:
            return tuple(int(i) for i in idx), energy
    return None, float('nan')


def find_secondary_minimum_nd(
    all_minima: list[tuple[tuple[int, ...], float]],
    axes: dict[str, np.ndarray],
    gs_idx: tuple[int, ...] | None,
    *,
    c_max: float,
    a3_max: float,
    a4_max: float,
) -> tuple[tuple[int, ...] | None, float]:
    """Pick the secondary minimum from sorted local minima.

    Conditions (any condition referencing an inactive axis is silently
    skipped):
      - c > gs.c (always)
      - c > gs.c + 0.1 OR a4 > gs.a4 + 0.1
      - c <= c_max
      - a3 <= a3_max
      - a4 <= a4_max

    Returns the first match (i.e. the lowest-energy match since
    `all_minima` is sorted ascending), or (None, NaN).
    """
    if gs_idx is None:
        return None, float('nan')

    axes_list = list(axes)

    def coord(idx: tuple[int, ...], name: str) -> float | None:
        if name not in axes:
            return None
        return float(axes[name][idx[axes_list.index(name)]])

    gs_c  = coord(gs_idx, 'c')
    gs_a4 = coord(gs_idx, 'a4')

    for idx, energy in all_minima:
        c  = coord(idx, 'c')
        a3 = coord(idx, 'a3')
        a4 = coord(idx, 'a4')

        if c is None or c <= gs_c:
            continue
        c_far  = c > gs_c + 0.1
        a4_far = (a4 is not None and gs_a4 is not None and a4 > gs_a4 + 0.1)
        if not (c_far or a4_far):
            continue
        if c > c_max:
            continue
        if a3 is not None and a3 > a3_max:
            continue
        if a4 is not None and a4 > a4_max:
            continue
        return tuple(int(i) for i in idx), energy
    return None, float('nan')


def find_fission_exit_nd(
    energies: np.ndarray,
    axes: dict[str, np.ndarray],
    c_threshold: float,
) -> tuple[tuple[int, ...] | None, float]:
    """Global minimum (not necessarily local) of `energies` over cells
    with c > c_threshold. Caller supplies the threshold (typically
    sm.c + 0.05 when a secondary minimum is found, or gs.c + offset
    when running without the SM search).
    """
    if not np.isfinite(c_threshold):
        return None, float('nan')

    c_pos = list(axes).index('c')
    c_threshold_idx = int(np.searchsorted(axes['c'], c_threshold, side='right'))

    if c_threshold_idx >= energies.shape[c_pos]:
        return None, float('nan')

    slicer = [slice(None)] * energies.ndim
    slicer[c_pos] = slice(c_threshold_idx, None)
    sub = energies[tuple(slicer)]
    if np.all(np.isnan(sub)):
        return None, float('nan')
    flat_argmin = int(np.nanargmin(sub))
    sub_idx = np.unravel_index(flat_argmin, sub.shape)
    full_idx = list(sub_idx)
    full_idx[c_pos] += c_threshold_idx
    return tuple(int(i) for i in full_idx), float(energies[tuple(full_idx)])


def run_critical_point_analysis_nd(
    df: pl.DataFrame,
    out: TextIO = sys.stdout,
) -> tuple[
    dict[CriticalPointType, CriticalPoint],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray],
]:
    """Single dimension-agnostic critical-point pipeline (replaces the
    paired run_critical_point_analysis / run_critical_point_analysis_5d).

    Returns
    -------
    critical_points : dict[CriticalPointType, CriticalPoint]
    energies        : dense N-D grid (used by the plot path).
    axes            : sorted unique values per active axis.
    inactive_axes   : {axis_name: constant_value} for present-but-constant axes.
    components      : per-component dense N-D grids.
    """
    print("\n  --- Critical Point Analysis (N-D) ---", file=out)
    t_total = time.perf_counter()

    active_axes = detect_active_axes(df)
    print(f"  Active axes: {active_axes}  (ndim = {len(active_axes)})", file=out)

    print("\n  Step 0: Building dense grid", file=out)
    t0 = time.perf_counter()
    energies, axes, inactive_axes, components = build_grids(df, active_axes)
    print(f"    Grid shape: {energies.shape}; "
          f"non-NaN cells: {int(np.count_nonzero(~np.isnan(energies))):,}", file=out)
    print(f"  [time] Step 0 (build grid): {time.perf_counter() - t0:.2f} s", file=out)

    print("\n  Step 1: Finding all local minima", file=out)
    print("    Criterion: a cell is a strict local minimum iff its energy is non-NaN,", file=out)
    print("               at least one in-bounds neighbor (3^N - 1 king-move neighborhood)", file=out)
    print("               is non-NaN, and no non-NaN neighbor is strictly smaller.", file=out)
    t0 = time.perf_counter()
    all_minima = find_minima_grid(energies, confirm_range=2)
    print(f"    Found {len(all_minima)} local minima (sorted ascending by energy):", file=out)
    width = max(2, len(str(len(all_minima))))
    for i, (idx, energy) in enumerate(all_minima, start=1):
        idx_t = tuple(int(j) for j in idx)
        print(f"      [{i:>{width}}] idx={idx_t} coords={_format_coords(idx_t, axes)}"
              f"  E = {float(energy):.4f} MeV", file=out)
    print(f"  [time] Step 1 (local minima): {time.perf_counter() - t0:.2f} s", file=out)

    print("\n  Step 2: Selecting Ground State", file=out)
    print(f"    Criterion: global energy minimum across all cells, accepted if c <= {GROUND_STATE_C_THRESHOLD}.", file=out)
    print(f"               Otherwise, the lowest-energy local minimum with c <= {GROUND_STATE_C_THRESHOLD}.", file=out)
    gs_idx, gs_e = find_ground_state_nd(
        all_minima, axes, energies,
        c_threshold=GROUND_STATE_C_THRESHOLD,
    )
    if gs_idx is not None:
        print(f"    ✓ Ground State at idx={gs_idx} {_format_coords(gs_idx, axes)},"
              f" E = {gs_e:.4f} MeV", file=out)
    else:
        print("    ✗ Ground State not found.", file=out)

    def _saddle(start_idx, end_idx, name):
        if start_idx is None or end_idx is None:
            return None, float('nan')
        if np.isnan(energies[start_idx]) or np.isnan(energies[end_idx]):
            return None, float('nan')
        result = find_iwf_grid(energies, start_idx, end_idx)
        if result is None:
            return None, float('nan')
        s_idx, s_e = result
        s_idx = tuple(int(i) for i in s_idx)
        print(f"    ✓ {name} at idx={s_idx} {_format_coords(s_idx, axes)},"
              f" E = {s_e:.4f} MeV", file=out)
        return s_idx, float(s_e)

    c_axis_pos = list(axes).index('c')

    if SECOND_MINIMUM_SEARCH:
        print("\n  Step 3: Selecting Secondary Minimum", file=out)
        print("    Criterion: lowest-energy local minimum with:", file=out)
        print("      - c  > ground_state.c  (always)", file=out)
        print("      - c  > ground_state.c + 0.1  OR  a4 > ground_state.a4 + 0.1", file=out)
        print(f"      - c  <= {SECONDARY_MINIMUM_C_MAX_THRESHOLD}", file=out)
        print(f"      - a3 <= {SECONDARY_MINIMUM_A3_MAX_THRESHOLD}  (skipped if a3 inactive)", file=out)
        print(f"      - a4 <= {SECONDARY_MINIMUM_A4_MAX_THRESHOLD}  (skipped if a4 inactive)", file=out)
        sm_idx, sm_e = find_secondary_minimum_nd(
            all_minima, axes, gs_idx,
            c_max =SECONDARY_MINIMUM_C_MAX_THRESHOLD,
            a3_max=SECONDARY_MINIMUM_A3_MAX_THRESHOLD,
            a4_max=SECONDARY_MINIMUM_A4_MAX_THRESHOLD,
        )
        if sm_idx is not None:
            print(f"    ✓ Secondary Minimum at idx={sm_idx} {_format_coords(sm_idx, axes)},"
                  f" E = {sm_e:.4f} MeV", file=out)
        else:
            print("    ✗ Secondary Minimum not found.", file=out)

        print("\n  Step 4: Selecting Fission Exit", file=out)
        print("    Criterion: global energy minimum over all cells with c > secondary_minimum.c + 0.05.", file=out)
        if sm_idx is not None:
            fe_threshold = float(axes['c'][sm_idx[c_axis_pos]]) + 0.05
        else:
            fe_threshold = float('nan')
        fe_idx, fe_e = find_fission_exit_nd(energies, axes, fe_threshold)
        if fe_idx is not None:
            print(f"    ✓ Fission Exit at idx={fe_idx} {_format_coords(fe_idx, axes)},"
                  f" E = {fe_e:.4f} MeV", file=out)
        else:
            print("    ✗ Fission Exit not found.", file=out)

        print("\n  Step 5: Inner Saddle", file=out)
        print("    Criterion: lowest barrier on the optimal path from Ground State to Secondary", file=out)
        print("               Minimum, found by the Iterative Watershed/Flooding algorithm", file=out)
        print("               (pes_analyzer.saddle.find_iwf_grid).", file=out)
        t0 = time.perf_counter()
        inner_idx, inner_e = _saddle(gs_idx, sm_idx, "Inner Saddle")
        print(f"  [time] Step 5 (inner saddle): {time.perf_counter() - t0:.2f} s", file=out)

        print("\n  Step 6: Outer Saddle", file=out)
        print("    Criterion: lowest barrier on the optimal path from Secondary Minimum to", file=out)
        print("               Fission Exit, found by the same IWF algorithm.", file=out)
        t0 = time.perf_counter()
        outer_idx, outer_e = _saddle(sm_idx, fe_idx, "Outer Saddle")
        print(f"  [time] Step 6 (outer saddle): {time.perf_counter() - t0:.2f} s", file=out)
    else:
        print("\n  Step 3: Secondary Minimum search DISABLED by SECOND_MINIMUM_SEARCH=False.", file=out)
        print("    Running the simplified GS -> Saddle -> FE topology suitable for SHE", file=out)
        print("    where no genuine fission isomer exists.", file=out)
        sm_idx, sm_e = None, float('nan')

        print("\n  Step 4: Selecting Fission Exit", file=out)
        print(f"    Criterion: global energy minimum over all cells with c > ground_state.c + {FISSION_EXIT_GS_C_OFFSET}.", file=out)
        if gs_idx is not None:
            fe_threshold = float(axes['c'][gs_idx[c_axis_pos]]) + FISSION_EXIT_GS_C_OFFSET
        else:
            fe_threshold = float('nan')
        fe_idx, fe_e = find_fission_exit_nd(energies, axes, fe_threshold)
        if fe_idx is not None:
            print(f"    ✓ Fission Exit at idx={fe_idx} {_format_coords(fe_idx, axes)},"
                  f" E = {fe_e:.4f} MeV", file=out)
        else:
            print("    ✗ Fission Exit not found.", file=out)

        print("\n  Step 5: Fission Saddle (single saddle between Ground State and Fission Exit)", file=out)
        print("    Criterion: lowest barrier on the optimal path from Ground State to Fission", file=out)
        print("               Exit, found by the Iterative Watershed/Flooding algorithm", file=out)
        print("               (pes_analyzer.saddle.find_iwf_grid). Reported as 'Inner Saddle'", file=out)
        print("               in CSVs/plots for downstream-schema compatibility.", file=out)
        t0 = time.perf_counter()
        inner_idx, inner_e = _saddle(gs_idx, fe_idx, "Inner Saddle")
        print(f"  [time] Step 5 (fission saddle): {time.perf_counter() - t0:.2f} s", file=out)

        print("\n  Step 6: Outer Saddle search SKIPPED (no secondary minimum).", file=out)
        outer_idx, outer_e = None, float('nan')

    def _to_cp(cp_type, name, idx, energy):
        cp = CriticalPoint(cp_type, name)
        if idx is None:
            return cp
        cp.point = index_to_gridpoint(idx, axes, inactive_axes, components, energy)
        cp.found = True
        return cp

    critical_points = {
        CriticalPointType.GROUND_STATE:      _to_cp(CriticalPointType.GROUND_STATE,      "Ground State",      gs_idx,    gs_e),
        CriticalPointType.SECONDARY_MINIMUM: _to_cp(CriticalPointType.SECONDARY_MINIMUM, "Secondary Minimum", sm_idx,    sm_e),
        CriticalPointType.FISSION_EXIT:      _to_cp(CriticalPointType.FISSION_EXIT,      "Fission Exit",      fe_idx,    fe_e),
        CriticalPointType.FIRST_SADDLE:      _to_cp(CriticalPointType.FIRST_SADDLE,      "Inner Saddle",      inner_idx, inner_e),
        CriticalPointType.SECOND_SADDLE:     _to_cp(CriticalPointType.SECOND_SADDLE,     "Outer Saddle",      outer_idx, outer_e),
    }

    print(f"\n  [time] Total critical point analysis: "
          f"{time.perf_counter() - t_total:.2f} s", file=out)
    return critical_points, energies, axes, inactive_axes, components


# =============================================================================
# 5D CRITICAL POINT ANALYSIS
# =============================================================================

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_critical_points_csv(critical_points: dict[CriticalPointType, CriticalPoint],
                             nucleus: NucleusInfo, output_dir: str = 'critical_points',
                             out: TextIO = sys.stdout) -> Path:
    """Save critical points to a CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    rows = []
    for cp_type, cp in critical_points.items():
        if cp.found:
            rows.append({
                'isotope': nucleus.isotope_label,
                'Z': nucleus.Z,
                'N': nucleus.N,
                'A': nucleus.A,
                'point_type': cp.name,
                'c': cp.point.c,
                'a3': cp.point.a3,
                'a4': cp.point.a4,
                'a5': cp.point.a5,
                'a6': cp.point.a6,
                'a7': cp.point.a7,
                'a8': cp.point.a8,
                'total_energy': cp.point.total_energy,
                'mass_excess': cp.point.mass_excess,
                'macro_energy': cp.point.macro_energy,
                'micro_energy': cp.point.micro_energy,
                'surface_energy': cp.point.surface_energy,
                'coulomb_energy': cp.point.coulomb_energy,
            })

    if not rows:
        print("    No critical points to save", file=out)
        return None

    df = pl.DataFrame(rows)
    output_file = output_path / f'{nucleus.isotope_label}_critical_points.csv'
    df.write_csv(output_file, float_precision=6)
    print(f"    Saved critical points to: {output_file}", file=out)

    return output_file


def print_analysis_summary(critical_points: dict[CriticalPointType, CriticalPoint],
                           nucleus: NucleusInfo,
                           out: TextIO = sys.stdout):
    """Print a formatted summary of the analysis results."""
    print("\n" + "=" * 90, file=out)
    print(f"  CRITICAL POINT ANALYSIS RESULTS - {nucleus.isotope_label}", file=out)
    print("=" * 90, file=out)
    print(f"  {'Point Type':<20} {'c':>8} {'a3':>8} {'a4':>8} {'a5':>8} {'a6':>8} "
          f"{'a7':>8} {'a8':>8} {'E (MeV)':>12}", file=out)
    print("-" * 90, file=out)

    gs = critical_points.get(CriticalPointType.GROUND_STATE)
    gs_energy = gs.point.total_energy if gs and gs.found else 0.0

    for cp_type, cp in critical_points.items():
        if cp.found:
            rel_e = cp.point.total_energy - gs_energy if gs and gs.found else 0.0
            energy_str = f"{cp.point.total_energy:.4f} (+{rel_e:.4f})" if cp_type != CriticalPointType.GROUND_STATE else f"{cp.point.total_energy:.4f}"
            print(f"  {cp.name:<20} {cp.point.c:>8.4f} {cp.point.a3:>8.4f} "
                  f"{cp.point.a4:>8.4f} {cp.point.a5:>8.4f} {cp.point.a6:>8.4f} "
                  f"{cp.point.a7:>8.4f} {cp.point.a8:>8.4f} {energy_str:>16}", file=out)
        else:
            print(f"  {cp.name:<20} Not found", file=out)

    print("=" * 90, file=out)


# =============================================================================
# PLOTTING
# =============================================================================

# Matplotlib (including the mathtext parser used for '$a_3$' labels) is not
# thread-safe. Serialize all matplotlib operations across worker threads.
_PLOT_LOCK = threading.Lock()


def create_discrete_colormap(vmin: float, vmax: float):
    """Create a discrete colormap with 1 MeV intervals."""
    start = np.floor(vmin)
    boundaries = np.arange(start, 21, 1.0)
    n_colors = len(boundaries) - 1
    rainbow_cmap = plt.get_cmap('rainbow')
    colors = [rainbow_cmap(i / (n_colors - 1)) for i in range(n_colors)]
    colors.append('white')
    # Overflow bin must start above both the last regular boundary and vmax
    final_boundary = max(boundaries[-1], vmax) + 1
    boundaries = np.append(boundaries, final_boundary)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, boundaries


def create_contour_plot(c: np.ndarray, y: np.ndarray, energy: np.ndarray,
                        nucleus: NucleusInfo, output_filename: str,
                        critical_points: Optional[dict] = None,
                        vmin: float = None, vmax: float = None,
                        y_param: str = 'a4',
                        out: TextIO = sys.stdout):
    """Create a c vs y contour plot with critical points marked.

    Args:
        y_param: 'a4' or 'a3' — controls y-axis label, tick spacing,
                 and which coordinate to read from critical points.
    """
    if vmin is None:
        vmin = energy.min()
    if vmax is None:
        vmax = energy.max()

    # Grid setup (numpy/scipy — heavy compute, GIL released; safe to overlap
    # across worker threads, so kept outside the plot lock).
    unique_c = np.unique(c)
    unique_y = np.unique(y)
    step_c = np.mean(np.diff(unique_c)) if len(unique_c) > 1 else 0.01
    step_y = np.mean(np.diff(unique_y)) if len(unique_y) > 1 else 0.01

    num_points_c = int((c.max() - c.min()) / (step_c / 2)) + 1 if step_c > 1e-9 else 2
    num_points_y = int((y.max() - y.min()) / (step_y / 2)) + 1 if step_y > 1e-9 else 2

    ci = np.linspace(c.min(), c.max(), num_points_c)
    yi = np.linspace(y.min(), y.max(), num_points_y)
    ci, yi = np.meshgrid(ci, yi)

    zi = griddata((c, y), energy, (ci, yi), method='cubic')
    zi = gaussian_filter(zi, sigma=2.0)

    cmap, norm, boundaries = create_discrete_colormap(vmin, vmax)
    contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1.0)

    # Matplotlib (mathtext parser in particular) is not thread-safe.
    # Serialize all figure/render operations across worker threads.
    with _PLOT_LOCK:
        # OO API (no pyplot global state) — combined with the lock this is
        # the only correct way to drive matplotlib from threads.
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        cf = ax.contourf(ci, yi, zi, levels=boundaries[:-1], cmap=cmap, norm=norm, extend='max')
        cs = ax.contour(ci, yi, zi, levels=contour_levels, colors='black', linewidths=1.5, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=12, fmt='%0.0f')

        # Plot critical points
        if critical_points:
            marker_styles = {
                CriticalPointType.GROUND_STATE: ('o', 'lime', 10, 'Ground State'),
                CriticalPointType.FIRST_SADDLE: ('s', 'lime', 8, 'Inner Saddle'),
                CriticalPointType.SECONDARY_MINIMUM: ('o', 'red', 10, 'Secondary Minimum'),
                CriticalPointType.SECOND_SADDLE: ('s', 'red', 8, 'Outer Saddle'),
                CriticalPointType.FISSION_EXIT: ('o', 'white', 10, 'Fission Exit'),
            }

            for cp_type, cp in critical_points.items():
                if cp.found:
                    marker, color, size, label = marker_styles.get(
                        cp_type, ('o', 'gray', 8, cp.name))
                    cp_y = cp.point.a3 if y_param == 'a3' else cp.point.a4
                    ax.plot(cp.point.c, cp_y, marker, color=color,
                            markersize=size, markeredgecolor='black',
                            markeredgewidth=1.5, zorder=10, label=label)

            # Add legend only on c vs a4 map (shown in tandem with c vs a3)
            if y_param != 'a3':
                ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

        # Labels and formatting
        y_subscript = '3' if y_param == 'a3' else '4'
        ax.set_xlabel('$c$', fontsize=18, labelpad=-5.0)
        ax.set_ylabel(f'$a_{{{y_subscript}}}$', fontsize=18, labelpad=-5.0)

        # Isotope label
        if nucleus.isotope_label:
            formatted_label = f'$^{{{nucleus.A}}}${nucleus.symbol}'
        else:
            formatted_label = Path(output_filename).stem
        element_box = {"boxstyle": 'round,pad=0.2', "facecolor": 'white', "alpha": 0.95}
        ax.text(0.02, 0.97, formatted_label, transform=ax.transAxes, fontsize=28,
                horizontalalignment='left', verticalalignment='top',
                bbox=element_box, fontweight='bold', zorder=20)

        ax.tick_params(axis='both', labelsize=14)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))

        x_tick = 0.20
        padded_c_min = np.floor(c.min() / x_tick) * x_tick - 0.0001
        padded_c_max = np.ceil(c.max() / x_tick) * x_tick + 0.0001
        ax.set_xlim(padded_c_min, padded_c_max)

        y_tick = 0.05
        padded_y_min = np.floor(y.min() / y_tick) * y_tick - 0.0001
        padded_y_max = np.ceil(y.max() / y_tick) * y_tick + 0.0001
        ax.set_ylim(padded_y_min, padded_y_max)

        ax.set_aspect(ASPECT_RATIO)

        fig.tight_layout()
        fig.savefig(output_filename, dpi=DPI, bbox_inches='tight', pad_inches=0.025)

    print(f"  Saved plot: {output_filename}", file=out)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_single_file(parquet_file: Path, output_plot: str = None,
                        save_minimized: bool = True,
                        out: TextIO = sys.stdout) -> dict:
    """Process a single parquet file: dimension-agnostic critical-point
    analysis + plot the (c, a3) and (c, a4) projections where those
    axes are active.
    """
    print(f"\n{'=' * 70}", file=out)
    print(f"Processing: {parquet_file.name}", file=out)
    print('=' * 70, file=out)

    nucleus = extract_nucleus_info(parquet_file)
    print(f"  Nucleus: {nucleus.isotope_label} (Z={nucleus.Z}, N={nucleus.N})", file=out)

    df = read_parquet_file(parquet_file, out=out)

    critical_points, energies, axes, inactive_axes, components = \
        run_critical_point_analysis_nd(df, out=out)

    print_analysis_summary(critical_points, nucleus, out=out)
    save_critical_points_csv(critical_points, nucleus, out=out)

    output_name = parquet_file.stem
    active_axes = tuple(axes.keys())

    for y_axis in ('a3', 'a4'):
        if y_axis not in active_axes:
            print(f"  Skipping c-vs-{y_axis} plot ({y_axis} not active)", file=out)
            continue

        xv, yv, e2d, argmin_flat = minimize_to_2d(energies, axes, 'c', y_axis)
        if save_minimized:
            save_minimized_data_2d(xv, yv, e2d, argmin_flat,
                                   axes, inactive_axes,
                                   'c', y_axis, output_name, out=out)

        if y_axis == 'a4' and output_plot:
            output_filename = output_plot
        else:
            output_filename = f'{output_name}_c_vs_{y_axis}.png'

        c_grid, y_grid = np.meshgrid(xv, yv, indexing='ij')
        mask = ~np.isnan(e2d)
        c_flat = c_grid[mask]
        y_flat = y_grid[mask]
        e_flat = e2d[mask]
        create_contour_plot(c_flat, y_flat, e_flat, nucleus, output_filename,
                            critical_points, y_param=y_axis, out=out)

    barriers = calculate_fission_barriers(critical_points, nucleus)

    return {
        'file': parquet_file,
        'nucleus': nucleus,
        'critical_points': critical_points,
        'barriers': barriers,
    }


def calculate_fission_barriers(critical_points: dict[CriticalPointType, CriticalPoint],
                               nucleus: NucleusInfo) -> FissionBarriers:
    """Calculate fission barrier heights relative to ground state."""
    barriers = FissionBarriers(nucleus=nucleus)

    gs = critical_points.get(CriticalPointType.GROUND_STATE)
    if not gs or not gs.found:
        return barriers

    gs_energy = gs.point.total_energy

    # Inner barrier (first saddle)
    s1 = critical_points.get(CriticalPointType.FIRST_SADDLE)
    if s1 and s1.found:
        barriers.inner_barrier = s1.point.total_energy - gs_energy

    # Outer barrier (second saddle)
    s2 = critical_points.get(CriticalPointType.SECOND_SADDLE)
    if s2 and s2.found:
        barriers.outer_barrier = s2.point.total_energy - gs_energy

    return barriers


def print_fission_barriers_table(all_barriers: list[FissionBarriers]):
    """Print a summary table of fission barrier heights for all nuclei."""
    print("\n" + "=" * 70)
    print("  FISSION BARRIER HEIGHTS SUMMARY")
    print("=" * 70)
    print(f"  {'Isotope':<12} {'Z':>4} {'N':>4} {'A':>4} {'Inner (MeV)':>14} {'Outer (MeV)':>14}")
    print("-" * 70)

    for b in sorted(all_barriers, key=lambda x: (x.nucleus.Z, x.nucleus.N)):
        inner_str = f"{b.inner_barrier:>14.4f}" if not np.isnan(b.inner_barrier) else "         N/A  "
        outer_str = f"{b.outer_barrier:>14.4f}" if not np.isnan(b.outer_barrier) else "         N/A  "
        print(f"  {b.nucleus.isotope_label:<12} {b.nucleus.Z:>4} {b.nucleus.N:>4} "
              f"{b.nucleus.A:>4} {inner_str} {outer_str}")

    print("=" * 70)


def save_fission_barriers_csv(all_barriers: list[FissionBarriers],
                              output_file: str = 'fission_barriers.csv') -> Path:
    """Save fission barrier heights to a CSV file."""
    rows = []
    for b in sorted(all_barriers, key=lambda x: (x.nucleus.Z, x.nucleus.N)):
        rows.append({
            'isotope': b.nucleus.isotope_label,
            'Z': b.nucleus.Z,
            'N': b.nucleus.N,
            'A': b.nucleus.A,
            'inner_barrier_MeV': b.inner_barrier if not np.isnan(b.inner_barrier) else None,
            'outer_barrier_MeV': b.outer_barrier if not np.isnan(b.outer_barrier) else None,
        })

    if rows:
        df = pl.DataFrame(rows)
        df.write_csv(output_file, float_precision=6)
        print(f"\nSaved fission barriers: {output_file}")
        return Path(output_file)
    return None


def save_combined_critical_points(all_results: list[dict], output_file: str = 'all_critical_points.csv'):
    """Save critical points from all processed files to a single CSV."""
    rows = []
    for result in all_results:
        nucleus = result['nucleus']
        for cp_type, cp in result['critical_points'].items():
            if cp.found:
                gs = result['critical_points'].get(CriticalPointType.GROUND_STATE)
                gs_energy = gs.point.total_energy if gs and gs.found else 0.0
                rel_e = cp.point.total_energy - gs_energy

                rows.append({
                    'isotope': nucleus.isotope_label,
                    'Z': nucleus.Z,
                    'N': nucleus.N,
                    'A': nucleus.A,
                    'point_type': cp.name,
                    'c': cp.point.c,
                    'a3': cp.point.a3,
                    'a4': cp.point.a4,
                    'a5': cp.point.a5,
                    'a6': cp.point.a6,
                    'a7': cp.point.a7,
                    'a8': cp.point.a8,
                    'total_energy': cp.point.total_energy,
                    'relative_energy': rel_e,
                    'mass_excess': cp.point.mass_excess,
                    'macro_energy': cp.point.macro_energy,
                    'micro_energy': cp.point.micro_energy,
                    'surface_energy': cp.point.surface_energy,
                    'coulomb_energy': cp.point.coulomb_energy,
                })

    if rows:
        df = pl.DataFrame(rows)
        df = df.sort(['Z', 'N', 'point_type'])
        df.write_csv(output_file, float_precision=6)
        print(f"\nSaved combined critical points: {output_file}")


def _process_single_file_captured(parquet_file: Path, save_minimized: bool) -> tuple[dict | None, str]:
    """Run process_single_file with its output routed to a per-call StringIO.

    Thread-safe: no global state mutation. Each worker writes to its own
    buffer, returned alongside the result so the caller can print it in
    a controlled order.

    On success returns (result_dict, captured_output).
    On failure returns (None, captured_output_including_error).
    """
    buffer = io.StringIO()
    try:
        result = process_single_file(parquet_file, output_plot=None,
                                     save_minimized=save_minimized,
                                     out=buffer)
    except Exception as e:
        import traceback
        print(f"ERROR processing {parquet_file}: {e}", file=buffer)
        traceback.print_exc(file=buffer)
        result = None
    return result, buffer.getvalue()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Enhanced PES Map Maker with Critical Point Analysis for FoS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python MapMaker_FoS_enhanced.py                          # Process all pes_Z*_N*.parquet files
    python MapMaker_FoS_enhanced.py pes_Z92_N144.parquet     # Process specific file
    python MapMaker_FoS_enhanced.py -o output.png file.parquet  # Specify output name

The program will:
  1. Read Parquet file(s) with FoS parameterization data
  2. Find critical points (ground state, secondary minimum, fission exit, saddles)
  3. Minimize over a3, a5, a6, a7, a8 to create c vs a4 maps
  4. Minimize over a4, a5, a6, a7, a8 to create c vs a3 maps
  5. Save critical points and minimized data to CSV files
  6. Create contour plots with critical points marked (*_c_vs_a4.png and *_c_vs_a3.png)
        """
    )
    parser.add_argument('parquet_file', type=str, nargs='?', default=None,
                        help='Path to specific Parquet file (if not given, process all found files)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename for the plot (default: auto-generated)')
    parser.add_argument('--no-save-minimized', action='store_true',
                        help='Do not save the minimized data CSV')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Max parallel workers for batch processing (default: 4)')

    args = parser.parse_args()

    program_start_time = time.time()

    print("=" * 70)
    print("FoS PES Map Maker with Critical Point Analysis")
    print("=" * 70)
    print("Configuration:")
    print(f"  Use float32 precision: {USE_FLOAT32}")
    print(f"  Critical point analysis: N-D (auto-detected axes)")
    print(f"  Output DPI: {DPI}")
    print(f"  Ground state c threshold: {GROUND_STATE_C_THRESHOLD}")
    print(f"  Secondary minimum search: {SECOND_MINIMUM_SEARCH}")
    if SECOND_MINIMUM_SEARCH:
        print(f"  Secondary minimum c max: {SECONDARY_MINIMUM_C_MAX_THRESHOLD}")
    else:
        print(f"  Fission exit c offset from GS: {FISSION_EXIT_GS_C_OFFSET}")
    print("=" * 70)

    # Determine files to process
    if args.parquet_file:
        files_to_process = [Path(args.parquet_file)]
        if not files_to_process[0].exists():
            print(f"ERROR: File not found: {args.parquet_file}")
            sys.exit(1)
    else:
        files_to_process = find_pes_files()
        if not files_to_process:
            print("No PES files found matching pattern 'pes_Z*_N*.parquet'")
            print("Specify a file explicitly or ensure files are in the current directory")
            sys.exit(1)
        print(f"\nFound {len(files_to_process)} PES files to process")

    # Process files
    all_results = []
    if len(files_to_process) == 1:
        # Single file: run directly (no overhead, preserves --output flag, live output)
        try:
            result = process_single_file(
                files_to_process[0],
                output_plot=args.output,
                save_minimized=not args.no_save_minimized,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {files_to_process[0]}: {e}")
    else:
        # Multiple files: run in parallel threads, print in submission order
        # so the output mirrors the sorted (Z, N) file list rather than
        # completion order. Compute still runs in parallel because the
        # heavy work (find_minima_grid, find_iwf_grid) releases the GIL.
        n_workers = min(args.workers, len(files_to_process))
        print(f"\nProcessing {len(files_to_process)} files with {n_workers} thread workers")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _process_single_file_captured, fp,
                    save_minimized=not args.no_save_minimized,
                )
                for fp in files_to_process
            ]
            for fp, future in zip(files_to_process, futures):
                try:
                    result, captured_output = future.result()
                    print(captured_output, end='')
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    print(f"ERROR processing {fp}: {e}")

    # Collect and display fission barriers
    all_barriers = [result['barriers'] for result in all_results]
    if all_barriers:
        print_fission_barriers_table(all_barriers)
        save_fission_barriers_csv(all_barriers)

    # Save combined critical points if multiple files
    if len(all_results) > 1:
        save_combined_critical_points(all_results)

    # Final summary
    program_end_time = time.time()
    total_time = program_end_time - program_start_time

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Files processed: {len(all_results)}/{len(files_to_process)}")
    print(f"  Total execution time: {total_time:.1f} seconds")
    print("  Output files:")
    print("    - critical_points/       (CSV files with critical points)")
    print("    - minimized_data_FoS/    (minimized energy surfaces)")
    print("    - fission_barriers.csv   (inner and outer barrier heights)")
    print("    - *.png                  (contour plots)")
    print("=" * 70)


if __name__ == "__main__":
    main()
