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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import polars as pl
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from pes_analyzer.extrema  import find_minima_grid
from pes_analyzer.grid     import build_dense
from pes_analyzer.topology import (
    MergeTree,
    PathProfile,
    analyze_path_profile,
    compute_persistence,
    find_minimum_energy_path,
    find_watershed_segmentation,
    prune_merge_tree,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# General settings
ASPECT_RATIO = 'equal'
DPI = 300
USE_FLOAT32 = True

# Restrict the analysis to the (c, a3, a4) subspace: keep only rows with every
# parameter beyond a4 at zero (a5 = a6 = a7 = a8 = 0). Comparing runs with this
# on and off isolates the impact of higher-order deformations on both the
# energy surface and the scission region.
MASK_3D_ONLY = False

# Scission-overlay semantics. True: shade a pixel when ANY shape in its
# minimized-out column is scissioning. False: shade only when the displayed
# (min-energy) configuration itself is scissioning.
SCISSION_FULL = True

# Axes that may be active. The actual active set is auto-detected per file
# from the parquet contents (any column with >1 unique value).
CANDIDATE_AXES = ('c', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8')

# =============================================================================
# SHAPE PARAMETER LIMITS
# =============================================================================
C_LIMITS = (None, None)
A3_LIMITS = (None, None)
A4_LIMITS = (None, None)
A5_LIMITS = (None, None)
A6_LIMITS = (None, None)
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
# Elongation gate. Every parameterization has an elongation parameter (Q20,
# B20, FoS c). On a GS-to-scission map the barrier complex lives at the start
# of its range: GS and SM in the first quarter, the (hyperdeformed) third
# minimum may extend into the first third (232Th TM sits at fraction 0.26).
# Window limits are inclusive; replaces the old absolute c < 1.4 threshold
# and the half-path step rule.
GS_SM_C_FRAC = 0.25           # GS + SM within the first quarter of the c-range
TM_C_FRAC = 1.0 / 3.0         # TM within the first third of the c-range
GS_SM_CONFIRM_RANGE = 2       # range-2 confirmation for ground-state candidate selection
MEP_PERSISTENCE = 0.4         # noise floor (MeV) for pruning the raw MEP profile
SM_PERSISTENCE = 1.0          # min N-D BASIN persistence (MeV) for the secondary minimum
THIRD_MIN_PERSISTENCE = 0.5   # min N-D BASIN persistence (MeV) for the third minimum
MERGE_TREE_PRUNE = 0.4        # presentation-only (pruned-tree panel); NOT used in classification
MERGE_TREE_MARKER_MIN = 0.1   # presentation-only: marker/label floor on the tree panels

# P5 (stencil consistency): the watershed/merge tree and the MEP must use the
# same neighbour stencil, or basin ids looked up from path cells (P1) and
# segment-max saddle energies (P3) stop being exact. Pass this constant to
# BOTH find_watershed_segmentation and find_minimum_energy_path.
FLOOD_NEIGHBORHOOD = "von_neumann"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CriticalPointType(Enum):
    """Types of critical points on a potential energy surface."""
    GROUND_STATE = auto()
    SECONDARY_MINIMUM = auto()
    THIRD_MINIMUM = auto()
    FIRST_SADDLE = auto()
    SECOND_SADDLE = auto()
    THIRD_SADDLE = auto()
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
    basin_id: Optional[int] = None      # minima only: watershed basin id
    basin_persistence: float = np.nan   # MeV; NaN for saddles and undying root basin
    profile_well_depth: float = np.nan  # MeV; 1-D path diagnostic (spec 5.5)
    segment: str = ""                   # saddles only: "gs-sm", "sm-tm", "tm-fe", ...


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
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float],
           dict[str, np.ndarray], Optional[np.ndarray]]:
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
    scission_grid
        0/1/NaN grid from `is_scissioning` (same shape as `energies`),
        or None when the parquet predates the neck columns.
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

    scission_grid: Optional[np.ndarray] = None
    if 'is_scissioning' in df.columns:
        sc_vals = df['is_scissioning'].cast(pl.Float64).to_numpy()
        scission_grid, _ = build_dense(coords, sc_vals)

    inactive_axes: dict[str, float] = {}
    for name in CANDIDATE_AXES:
        if name not in active_axes and name in df.columns:
            inactive_axes[name] = float(df[name].unique().to_numpy()[0])

    return energies, axes, inactive_axes, components, scission_grid


# =============================================================================
# FILE I/O
# =============================================================================

NECK_COLUMNS = ('has_neck', 'neck_radius', 'is_scissioning')


def verify_scission_band(df: pl.DataFrame, out: TextIO = sys.stdout) -> None:
    """Check the scission-band invariant on the valid-filtered frame.

    Per docs/2026-06-11-parquet-neck-columns.md: every valid scissioning row
    must have 1.2 < neck_radius < 1.5 fm, and on necked rows the scission
    label must match the 1.5 fm threshold. Violations warn instead of raising
    so one bad map cannot kill a batch run.
    """
    sciss = df.filter(pl.col('is_scissioning'))
    n_sciss = len(sciss)
    if n_sciss == 0:
        print("    Scission check: no scissioning rows", file=out)
        return
    radius = sciss['neck_radius']
    out_of_band = int(((radius <= 1.2) | (radius >= 1.5)).sum())
    necked = df.filter(pl.col('has_neck'))
    mislabeled = int((necked['is_scissioning'] != (necked['neck_radius'] < 1.5)).sum())
    if out_of_band == 0 and mislabeled == 0:
        print(f"    Scission check PASS: {n_sciss:,} scissioning rows, "
              f"neck_radius in ({radius.min():.4f}, {radius.max():.4f}) fm", file=out)
    else:
        print(f"    Scission check WARNING: {out_of_band:,} of {n_sciss:,} scissioning "
              f"rows outside (1.2, 1.5) fm; {mislabeled:,} necked rows with "
              f"label/threshold mismatch", file=out)


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

        # Neck/scission columns exist only in maps generated from 2026-06-11 on.
        has_neck_cols = set(NECK_COLUMNS) <= set(available_columns)
        if has_neck_cols:
            columns_to_read += list(NECK_COLUMNS)
        else:
            print("    Note: no neck/scission columns (pre-2026-06-11 map); "
                  "scission overlay disabled", file=out)

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

        if has_neck_cols:
            verify_scission_band(df, out=out)

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

        if MASK_3D_ONLY:
            higher = [ax for ax in ('a5', 'a6', 'a7', 'a8') if ax in df.columns]
            before = len(df)
            df = df.filter(pl.all_horizontal(
                [pl.col(ax).abs() < 1e-12 for ax in higher]))
            print(f"    3D-only mask ({' = '.join(higher)} = 0): "
                  f"{before:,} -> {len(df):,} points", file=out)

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


def gather_at_argmin_2d(
    grid: np.ndarray,
    axes: dict[str, np.ndarray],
    x_axis: str,
    y_axis: str,
    argmin_flat: np.ndarray,
) -> np.ndarray:
    """Sample a parallel N-D grid at the cells selected by minimize_to_2d.

    Applies the same transpose+reshape as minimize_to_2d and gathers with its
    argmin_flat, so the result at (x, y) is the grid's value at the 5D point
    that provides the displayed minimum energy. Values are undefined where
    the projected energy is NaN; caller must mask.
    """
    axes_idx = {n: i for i, n in enumerate(axes)}
    order = [axes_idx[x_axis], axes_idx[y_axis]] + [
        i for i, n in enumerate(axes) if n not in (x_axis, y_axis)
    ]
    moved = grid.transpose(order)
    flat = moved.reshape(moved.shape[0], moved.shape[1], -1)
    return np.take_along_axis(flat, argmin_flat[..., None], -1).squeeze(-1)


def project_any_2d(
    grid: np.ndarray,
    axes: dict[str, np.ndarray],
    x_axis: str,
    y_axis: str,
) -> np.ndarray:
    """Project an N-D 0/1/NaN mask to 2D: True at (x, y) iff ANY cell in the
    column over the remaining axes is set. NaN counts as unset.

    Companion to gather_at_argmin_2d with SCISSION_FULL semantics: shows
    where scissioning shapes exist at all, not just where the min-energy
    configuration is scissioning. Same axis ordering as minimize_to_2d.
    """
    axes_idx = {n: i for i, n in enumerate(axes)}
    order = [axes_idx[x_axis], axes_idx[y_axis]] + [
        i for i, n in enumerate(axes) if n not in (x_axis, y_axis)
    ]
    moved = grid.transpose(order)
    flat = moved.reshape(moved.shape[0], moved.shape[1], -1)
    return (np.where(np.isnan(flat), 0.0, flat) > 0.5).any(axis=-1)


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


def _print_minima_list(label: str,
                       minima: list[tuple[tuple[int, ...], float]],
                       axes: dict[str, np.ndarray],
                       shape: tuple[int, ...],
                       out: TextIO = sys.stdout) -> None:
    """List every local minimum: coords, energy, and c/a4 max-edge flags.

    Minima arrive sorted ascending by energy from ``find_minima_grid``. The
    ``c@max`` / ``a4@max`` flags mark cells sitting on the elongation / a4
    grid wall — i.e. the fission-exit and scission candidates.
    """
    axis_names = list(axes)
    c_pos = axis_names.index('c') if 'c' in axis_names else None
    a4_pos = axis_names.index('a4') if 'a4' in axis_names else None
    print(f"    {label} (n={len(minima)}):", file=out)
    for idx, energy in minima:
        flags = []
        if c_pos is not None and idx[c_pos] == shape[c_pos] - 1:
            flags.append("c@max")
        if a4_pos is not None and idx[a4_pos] == shape[a4_pos] - 1:
            flags.append("a4@max")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"      {_format_coords(idx, axes)}  E={energy:>9.4f}{flag_str}", file=out)


def _mep_reject_reason(e: float, prev_e: float, gs_e: float,
                       c: float, c_limit: float,
                       depth_floor: float,
                       persistence: float) -> Optional[str]:
    """Why an interior MEP profile minimum cannot be labeled; None if it can.

    Checks (a)-(b) are energy ordering from the profile: a labeled fission
    well sits above the previous labeled minimum (GS < SM < TM; 232Th is
    the template). Check (c) is the elongation window: the barrier complex
    lives in the first GS_SM_C_FRAC / TM_C_FRAC of the c-range on a
    GS-to-scission map (replaces the old half-path step rule -- no path or
    grid dependence). Check (d) is the N-D basin persistence -- the climb
    to the lowest escape saddle in ANY direction, not just along the path
    (basin-led classification spec, section 5.3). Boundaries are inclusive:
    reject on < (floor) / > (window), so exact ties stay accepted.
    """
    if e <= gs_e:
        return "below GS energy (scission slope)"
    if e <= prev_e:
        return "below previous labeled minimum (valley on the descent)"
    if c > c_limit:
        return f"c={c:.3f} beyond elongation window (c <= {c_limit:.3f})"
    if persistence < depth_floor:
        return f"basin persistence {persistence:.2f} MeV < {depth_floor} MeV"
    return None


def _collect_mep_candidates(profile, path_idx, tree, gs_basin, fe_basin,
                            reanchor_floor=0.0):
    """Map interior profile minima to basins; flag GS/FE re-entries and revisits.

    Returns one ``(k, e, basin_id, skip_reason)`` tuple per interior profile
    minimum, in path order; ``e`` is the basin minimum energy. ``skip_reason``
    None marks a classification candidate: the first on-path visit of a basin
    that is neither the GS nor the fission-exit well. By P1 every
    profile-minimum cell is its basin's seed, so basin_of_point is exact and
    no proximity confirmation is needed; by P2 the path is a walk that may
    revisit basins, hence the first-visit deduplication (basin-led
    classification spec, section 5.2).

    Re-anchoring: a basin that dies through a merge shallower than
    ``reanchor_floor`` (the MEP profile noise floor) is a sub-noise satellite
    fragment of a larger well, not a well of its own -- identity, energy and
    seed move to the merge parent, repeatedly (232Th: the MEP bottoms in a
    0.09 MeV a6-ripple satellite two grid steps from the true SM seed).
    """
    last = len(path_idx) - 1
    seen = {gs_basin, fe_basin}
    out = []
    for k, _e_path in profile.minima:
        if k in (0, last):
            continue
        bid = tree.basin_of_point(tuple(int(v) for v in path_idx[k]))
        while (tree.node(bid).persistence < reanchor_floor
               and tree.node(bid).parent is not None):
            bid = tree.node(bid).parent
        e = float(tree.node(bid).minimum_energy)
        if bid == gs_basin:
            skip = "GS basin, skipped"
        elif bid == fe_basin:
            skip = "fission-exit basin, skipped"
        elif bid in seen:
            skip = f"revisit of basin #{bid} at step {k}"
        else:
            skip = None
            seen.add(bid)
        out.append((k, float(e), bid, skip))
    return out


def select_mep_critical_points(energies, minima_gs_sm, axes, tree, out=sys.stdout):
    """GS / SM / 3rd-min / FE / saddles: MEP ordering, merge-tree identity.

    The ground state is the deepest confirmed minimum within the first
    GS_SM_C_FRAC of the c-range. The fission exit is the global-minimum
    cell of the grid. The deep-minimax MEP between them orders the wells
    and assigns saddles to segments; the merge tree is authoritative for
    well identity (basin id, P1/P2 dedup) and significance (N-D basin
    persistence vs SM_PERSISTENCE / THIRD_MIN_PERSISTENCE). Classification
    runs on the UNPRUNED tree; MERGE_TREE_PRUNE is presentation-only.

    Returns (sel, path_indices, path_energies, profile). ``sel`` keys:
    basin ids for ground_state / secondary_minimum / third_minimum /
    fission_exit; (index, energy) tuples for inner/outer/third saddles;
    plus ``gs_persistence_bf`` (GS basin persistence = minimax barrier
    height, None while unknown or for an undying GS root basin),
    ``saddle_segments`` (saddle key -> "gs-sm" style segment name),
    ``well_depths`` (role -> 1-D profile escape depth, diagnostic only),
    and ``mep_steps`` (role -> path step of the marker on the MEP profile).
    """
    sel = {
        "ground_state": None, "secondary_minimum": None, "third_minimum": None,
        "fission_exit": None, "inner_saddle": None, "outer_saddle": None,
        "third_saddle": None,
        "gs_persistence_bf": None, "saddle_segments": {}, "well_depths": {},
        "mep_steps": {},
    }
    c_pos = list(axes).index('c')
    c_lo, c_hi = float(axes['c'][0]), float(axes['c'][-1])
    c_sm_max = c_lo + GS_SM_C_FRAC * (c_hi - c_lo)
    c_tm_max = c_lo + TM_C_FRAC * (c_hi - c_lo)
    print(f"    Elongation windows: GS/SM c <= {c_sm_max:.3f} (first quarter), "
          f"TM c <= {c_tm_max:.3f} (first third) of c in [{c_lo:g}, {c_hi:g}]",
          file=out)
    # GS = deepest confirmed minimum in the window whose basin traps at
    # least as well as anything we label (persistence >= SM_PERSISTENCE).
    # Without the floor, a deep low-persistence pocket on the scission
    # slope admitted by the elongation window steals the GS (279Mt:
    # E=-10.23 at c=1.525, persistence 0.48, vs the true GS at c=1.050).
    gs_candidates = []
    for idx, e in minima_gs_sm:
        if float(axes['c'][idx[c_pos]]) > c_sm_max:
            continue
        pers = tree.node(tree.basin_of_point(tuple(idx))).persistence
        if pers < SM_PERSISTENCE:
            print(f"      GS candidate {_format_coords(idx, axes)} E={e:.4f} "
                  f"skipped: basin persistence {pers:.2f} MeV "
                  f"< {SM_PERSISTENCE} MeV", file=out)
            continue
        gs_candidates.append((idx, e))
    if not gs_candidates:
        return sel, None, None, None
    gs_idx, _gs_e = min(gs_candidates, key=lambda t: t[1])
    sel["ground_state"] = tree.basin_of_point(tuple(gs_idx))

    fe_idx = tuple(int(i) for i in
                   np.unravel_index(int(np.nanargmin(energies)), energies.shape))
    if fe_idx == tuple(gs_idx):
        return sel, None, None, None      # degenerate map: GS is the deepest cell

    mep = find_minimum_energy_path(energies, tuple(gs_idx), fe_idx,
                                   neighborhood=FLOOD_NEIGHBORHOOD)
    if mep is None:
        return sel, None, None, None      # NaN wall between GS and exit
    path_idx, path_e = mep
    sel["fission_exit"] = tree.basin_of_point(fe_idx)

    profile = analyze_path_profile(path_e, min_persistence=MEP_PERSISTENCE)
    print(f"    MEP: {len(path_e)} steps; profile after {MEP_PERSISTENCE} MeV pruning: "
          f"{len(profile.minima)} minima, {len(profile.saddles)} saddles", file=out)

    def grid_pt(k):
        return tuple(int(v) for v in path_idx[k]), float(path_e[k])

    last = len(path_e) - 1
    gs_e = float(path_e[0])
    sel["mep_steps"]["ground_state"] = 0
    sel["mep_steps"]["fission_exit"] = last

    # 1-D escape depth along the path: DIAGNOSTIC only. Persistence is the
    # authority for significance; profile depth >= persistence always, and
    # the gap measures how often the lowest escape bypasses the path
    # (basin-led classification spec, sections 4 and 6.1).
    all_ext = sorted(
        [(k, e, True) for k, e in profile.minima]
        + [(k, e, False) for k, e in profile.saddles]
    )

    def well_depth(k0):
        i = next((j for j, (k, _e, _m) in enumerate(all_ext) if k == k0), None)
        if i is None:
            return float('nan')
        left = [e for _k, e, m in all_ext[:i] if not m]
        right = [e for _k, e, m in all_ext[i + 1:] if not m]
        barrier = min(left[-1] if left else np.inf,
                      right[0] if right else np.inf)
        return barrier - all_ext[i][1]

    interior = _collect_mep_candidates(
        profile, path_idx, tree, sel["ground_state"], sel["fission_exit"],
        reanchor_floor=MEP_PERSISTENCE)

    # Walk interior minima in path order: the first candidate passing the
    # SM conditions is the secondary minimum; the first one after it
    # passing the TM conditions is the third minimum. No SM means no TM
    # search. Identity = basin id, significance = basin persistence; the
    # profile only sets the ordering.
    sm = tm = None                        # (k, e, basin_id)
    for k, e, bid, skip in interior:
        node = tree.node(bid)
        pers = node.persistence
        c_cand = float(axes['c'][node.minimum_index[c_pos]])
        if skip is not None:
            label = None
            verdict = f"-> ({skip})"
        elif sm is None:
            reason = _mep_reject_reason(e, gs_e, gs_e, c_cand, c_sm_max,
                                        SM_PERSISTENCE, pers)
            label = "SM" if reason is None else None
            verdict = f"-> {label}" if label else f"[{reason}]"
        elif tm is None:
            reason = _mep_reject_reason(e, sm[1], gs_e, c_cand, c_tm_max,
                                        THIRD_MIN_PERSISTENCE, pers)
            label = "TM" if reason is None else None
            verdict = f"-> {label}" if label else f"[{reason}]"
        else:
            label = None
            verdict = "[SM and TM already assigned]"
        print(f"      interior minimum: step {k} "
              f"{_format_coords(grid_pt(k)[0], axes)} E={e:.4f}", file=out)
        on_path_bid = tree.basin_of_point(grid_pt(k)[0])
        anchor_note = ("" if on_path_bid == bid else
                       f"  (re-anchored from satellite basin #{on_path_bid}, "
                       f"pers {tree.node(on_path_bid).persistence:.2f} "
                       f"< {MEP_PERSISTENCE})")
        print(f"          basin #{bid}  persistence={pers:.2f} MeV  "
              f"profile-depth={well_depth(k):.2f} MeV  {verdict}{anchor_note}",
              file=out)
        if label == "SM":
            sm = (k, e, bid)
        elif label == "TM":
            tm = (k, e, bid)

    if sm is not None:
        sel["secondary_minimum"] = sm[2]
        sel["well_depths"]["secondary_minimum"] = well_depth(sm[0])
        sel["mep_steps"]["secondary_minimum"] = sm[0]
    if tm is not None:
        sel["third_minimum"] = tm[2]
        sel["well_depths"]["third_minimum"] = well_depth(tm[0])
        sel["mep_steps"]["third_minimum"] = tm[0]
    sel["well_depths"]["ground_state"] = well_depth(0)

    # GS basin persistence = minimax fission-barrier height from the GS
    # (spec 6.4): finite whenever the surface descends past scission. It
    # must match the highest path saddle minus the GS energy unless the
    # lowest escape bypasses the MEP -- log that case (spec 8.5).
    gs_node = tree.node(sel["ground_state"])
    gs_pers = float(gs_node.persistence)
    if np.isfinite(gs_pers):
        sel["gs_persistence_bf"] = gs_pers
        if profile.saddles:
            bf_path = max(e for _k, e in profile.saddles) - gs_node.minimum_energy
            if gs_pers < bf_path - 1e-6:
                print(f"    NOTE: GS persistence {gs_pers:.4f} MeV < path barrier "
                      f"{bf_path:.4f} MeV -- lowest escape bypasses the MEP", file=out)

    # One barrier per pair of consecutive labeled minima (GS, SM, TM, FE):
    # the highest profile saddle inside that segment, exact by P3. Dropping
    # an unlabeled pocket merges its two segments, so the max is the
    # controlling barrier; segment names go to the CSV (spec 5.6).
    anchors = [(0, "gs")] \
        + [(m[0], name) for m, name in ((sm, "sm"), (tm, "tm")) if m is not None] \
        + [(last, "fe")]
    saddle_keys = ["inner_saddle", "outer_saddle", "third_saddle"]
    for key, ((a, name_a), (b, name_b)) in zip(saddle_keys, zip(anchors, anchors[1:])):
        segment = [(k, e) for k, e in profile.saddles if a < k < b]
        if segment:
            k_max, _e_max = max(segment, key=lambda t: t[1])
            sel[key] = grid_pt(k_max)
            sel["saddle_segments"][key] = f"{name_a}-{name_b}"
            sel["mep_steps"][key] = k_max
    return sel, path_idx, path_e, profile


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


def run_critical_point_analysis_nd(
    df: pl.DataFrame,
    out: TextIO = sys.stdout,
) -> tuple[
    dict[CriticalPointType, CriticalPoint],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray],
    Optional[np.ndarray],
    "MergeTree",
    dict,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[PathProfile],
]:
    """Dimension-agnostic PES topology analysis + FoS physics layer.

    The library does the physics-free part (extrema, watershed, merge tree,
    MEP); the GS / SM / FE identification reads the MEP profile here.

    Returns critical_points, energies, axes, inactive_axes, components,
    scission_grid, tree, sel, mep_idx, mep_e, mep_profile.
    """
    print("\n  --- Critical Point Analysis (N-D) ---", file=out)
    t_total = time.perf_counter()

    active_axes = detect_active_axes(df)
    print(f"  Active axes: {active_axes}  (ndim = {len(active_axes)})", file=out)

    print("\n  Step 0: Building dense grid", file=out)
    t0 = time.perf_counter()
    energies, axes, inactive_axes, components, scission_grid = build_grids(df, active_axes)
    print(f"    Grid shape: {energies.shape}; "
          f"non-NaN cells: {int(np.count_nonzero(~np.isnan(energies))):,}", file=out)
    print(f"  [time] Step 0 (build grid): {time.perf_counter() - t0:.2f} s", file=out)

    print("\n  Step 1: Confirmed local minima (range-2)", file=out)
    t0 = time.perf_counter()
    minima_gs_sm = find_minima_grid(energies, neighborhood_range=1, confirm_range=GS_SM_CONFIRM_RANGE)
    print(f"    Confirmed minima: r{GS_SM_CONFIRM_RANGE}={len(minima_gs_sm)} (GS/SM)", file=out)
    print(f"  [time] Step 1 (minima): {time.perf_counter() - t0:.2f} s", file=out)

    _print_minima_list(f"Local minima r{GS_SM_CONFIRM_RANGE} (GS/SM)",
                       minima_gs_sm, axes, energies.shape, out=out)

    print("\n  Step 2: Watershed segmentation + merge tree (no pruning)", file=out)
    t0 = time.perf_counter()
    labels, basins, merges = find_watershed_segmentation(
        energies, neighborhood=FLOOD_NEIGHBORHOOD)
    tree = MergeTree(labels, basins, merges)
    print(f"    Basins: {len(basins)}; merge events: {len(merges)}", file=out)
    print(f"  [time] Step 2 (watershed+tree): {time.perf_counter() - t0:.2f} s", file=out)

    print("\n  Step 3: Critical-point identification (MEP profile)", file=out)
    t0 = time.perf_counter()

    sel, mep_idx, mep_e, mep_profile = select_mep_critical_points(
        energies, minima_gs_sm, axes, tree, out=out)
    print(f"  [time] Step 3 (identification): {time.perf_counter() - t0:.2f} s", file=out)

    def _basin_gp(bid):
        if bid is None:
            return None, float('nan')
        node = tree.node(bid)
        return node.minimum_index, node.minimum_energy

    def _saddle_gp(s):
        if s is None:
            return None, float('nan')
        return s[0], s[1]

    gs_idx,    gs_e    = _basin_gp (sel["ground_state"])
    sm_idx,    sm_e    = _basin_gp (sel["secondary_minimum"])
    tm_idx,    tm_e    = _basin_gp (sel["third_minimum"])
    fe_idx,    fe_e    = _basin_gp (sel["fission_exit"])
    inner_idx, inner_e = _saddle_gp(sel["inner_saddle"])
    outer_idx, outer_e = _saddle_gp(sel["outer_saddle"])
    third_idx, third_e = _saddle_gp(sel["third_saddle"])

    if gs_idx is not None:
        print(f"    ✓ Ground State      at idx={gs_idx} {_format_coords(gs_idx, axes)}, E = {gs_e:.4f} MeV", file=out)
    else:
        print("    ✗ Ground State not found.", file=out)
    if sm_idx is not None:
        print(f"    ✓ Secondary Minimum at idx={sm_idx} {_format_coords(sm_idx, axes)}, E = {sm_e:.4f} MeV", file=out)
    else:
        print("    - Secondary Minimum: none found.", file=out)
    if tm_idx is not None:
        print(f"    ✓ Third Minimum     at idx={tm_idx} {_format_coords(tm_idx, axes)}, E = {tm_e:.4f} MeV", file=out)
    else:
        print("    - Third Minimum: none found.", file=out)
    if inner_idx is not None:
        print(f"    ✓ Inner Saddle      at idx={inner_idx} {_format_coords(inner_idx, axes)}, E = {inner_e:.4f} MeV", file=out)
    if outer_idx is not None:
        print(f"    ✓ Outer Saddle      at idx={outer_idx} {_format_coords(outer_idx, axes)}, E = {outer_e:.4f} MeV", file=out)
    if third_idx is not None:
        print(f"    ✓ Third Saddle      at idx={third_idx} {_format_coords(third_idx, axes)}, E = {third_e:.4f} MeV", file=out)
    if fe_idx is not None:
        print(f"    ✓ Fission Exit      at idx={fe_idx} {_format_coords(fe_idx, axes)}, E = {fe_e:.4f} MeV", file=out)
    else:
        print("    ✗ Fission Exit not found.", file=out)

    def _to_cp(cp_type, name, idx, energy):
        cp_obj = CriticalPoint(cp_type, name)
        if idx is None:
            return cp_obj
        cp_obj.point = index_to_gridpoint(idx, axes, inactive_axes, components, energy)
        cp_obj.found = True
        return cp_obj

    critical_points = {
        CriticalPointType.GROUND_STATE:      _to_cp(CriticalPointType.GROUND_STATE,      "Ground State",      gs_idx,    gs_e),
        CriticalPointType.SECONDARY_MINIMUM: _to_cp(CriticalPointType.SECONDARY_MINIMUM, "Secondary Minimum", sm_idx,    sm_e),
        CriticalPointType.THIRD_MINIMUM:     _to_cp(CriticalPointType.THIRD_MINIMUM,     "Third Minimum",     tm_idx,    tm_e),
        CriticalPointType.FISSION_EXIT:      _to_cp(CriticalPointType.FISSION_EXIT,      "Fission Exit",      fe_idx,    fe_e),
        CriticalPointType.FIRST_SADDLE:      _to_cp(CriticalPointType.FIRST_SADDLE,      "Inner Saddle",      inner_idx, inner_e),
        CriticalPointType.SECOND_SADDLE:     _to_cp(CriticalPointType.SECOND_SADDLE,     "Outer Saddle",      outer_idx, outer_e),
        CriticalPointType.THIRD_SADDLE:      _to_cp(CriticalPointType.THIRD_SADDLE,      "Third Saddle",      third_idx, third_e),
    }

    # Basin-level facts for the CSV (spec 5.6). Persistence comes from the
    # UNPRUNED tree; an undying root basin (inf) becomes NaN -> empty cell.
    min_roles = {
        CriticalPointType.GROUND_STATE:      "ground_state",
        CriticalPointType.SECONDARY_MINIMUM: "secondary_minimum",
        CriticalPointType.THIRD_MINIMUM:     "third_minimum",
        CriticalPointType.FISSION_EXIT:      "fission_exit",
    }
    for cp_type, sel_key in min_roles.items():
        bid = sel[sel_key]
        if bid is None:
            continue
        cp = critical_points[cp_type]
        cp.basin_id = bid
        pers = float(tree.node(bid).persistence)
        cp.basin_persistence = pers if np.isfinite(pers) else float('nan')
        cp.profile_well_depth = float(sel["well_depths"].get(sel_key, float('nan')))
    saddle_roles = {
        CriticalPointType.FIRST_SADDLE:  "inner_saddle",
        CriticalPointType.SECOND_SADDLE: "outer_saddle",
        CriticalPointType.THIRD_SADDLE:  "third_saddle",
    }
    for cp_type, sel_key in saddle_roles.items():
        critical_points[cp_type].segment = sel["saddle_segments"].get(sel_key, "")

    print(f"\n  [time] Total critical point analysis: "
          f"{time.perf_counter() - t_total:.2f} s", file=out)
    return (critical_points, energies, axes, inactive_axes, components,
            scission_grid, tree, sel, mep_idx, mep_e, mep_profile)


# =============================================================================
# 5D CRITICAL POINT ANALYSIS
# =============================================================================

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_critical_points_csv(critical_points: dict[CriticalPointType, CriticalPoint],
                             nucleus: NucleusInfo, output_dir: str = 'critical_points',
                             gs_persistence_bf: Optional[float] = None,
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
                'basin_id': cp.basin_id,
                'basin_persistence': None if np.isnan(cp.basin_persistence)
                                     else float(cp.basin_persistence),
                'profile_well_depth': None if np.isnan(cp.profile_well_depth)
                                      else float(cp.profile_well_depth),
                'segment': cp.segment or None,
                'gs_persistence_bf': gs_persistence_bf,
            })

    if not rows:
        print("    No critical points to save", file=out)
        return None

    df = pl.DataFrame(rows)
    output_file = output_path / f'{nucleus.isotope_label}_critical_points.csv'
    df.write_csv(output_file, float_precision=6)
    print(f"    Saved critical points to: {output_file}", file=out)

    return output_file


def save_basins_csv(tree: "MergeTree", axes: dict[str, np.ndarray],
                    nucleus: NucleusInfo, output_dir: str = 'basins',
                    out: TextIO = sys.stdout) -> Optional[Path]:
    """Write every basin's minimum coordinates, energy, and persistence."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    axis_names = list(axes)
    rows = []
    for bid, node in sorted(tree.nodes.items()):
        row = {'basin_id': bid}
        for i, name in enumerate(axis_names):
            row[name] = float(axes[name][node.minimum_index[i]])
        row['minimum_energy'] = node.minimum_energy
        row['persistence'] = node.persistence
        rows.append(row)

    if not rows:
        print("    No basins to save", file=out)
        return None

    df = pl.DataFrame(rows)
    output_file = output_path / f'{nucleus.isotope_label}_basins.csv'
    df.write_csv(output_file, float_precision=6)
    print(f"    Saved {len(rows)} basins to: {output_file}", file=out)
    return output_file


def plot_persistence_histogram(tree: "MergeTree", nucleus: NucleusInfo,
                               output_filename: str,
                               out: TextIO = sys.stdout) -> None:
    """Histogram of finite basin persistences (the root's +inf is excluded)."""
    persistences = [
        node.persistence for node in tree.nodes.values()
        if np.isfinite(node.persistence)
    ]
    if not persistences:
        print("    No finite persistences to plot", file=out)
        return

    with _PLOT_LOCK:
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(persistences, bins=min(50, max(10, len(persistences) // 5)),
                color='steelblue', edgecolor='black')
        ax.set_xlabel('Persistence (MeV)', fontsize=14)
        ax.set_ylabel('Basin count', fontsize=14)
        title = nucleus.isotope_label or Path(output_filename).stem
        ax.set_title(f'Basin persistence — {title}  (n={len(persistences)})', fontsize=14)
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(output_filename, dpi=DPI, bbox_inches='tight')

    print(f"  Saved persistence histogram: {output_filename}", file=out)


def _merge_tree_segments(tree, displayed_ids, c_pos, top_energy):
    """Draw-ready dendrogram segments for a subset of merge-tree basins.

    Pure layout, no matplotlib. X positions are the rank of each displayed
    basin in ascending-``c`` order (even spacing). Each basin gets a vertical
    segment from its minimum up to the energy at which it merges into the
    nearest *displayed* ancestor; a horizontal connector links it to that
    ancestor at the saddle energy. Basins whose nearest displayed ancestor is
    the root (or themselves the root) run up to ``top_energy``.

    Parameters
    ----------
    tree : MergeTree
    displayed_ids : iterable of int
        Basin ids to draw (all basins, or the pruned survivors).
    c_pos : int
        Index of the ``c`` axis in each basin's ``minimum_index``.
    top_energy : float
        Energy at which root / parentless branches terminate.

    Returns
    -------
    x_of : dict[int, int]
        Basin id -> x position (rank in c order).
    vsegs : dict[int, tuple[int, float, float]]
        Basin id -> (x, e_bottom, e_top).
    hsegs : list[tuple[int, int, float, int]]
        (x_child, x_ancestor, e_saddle, child_id) connectors.
    """
    disp = set(displayed_ids)
    ids = sorted(disp, key=lambda b: tree.node(b).minimum_index[c_pos])
    x_of = {b: i for i, b in enumerate(ids)}

    vsegs = {}
    hsegs = []
    for b in ids:
        node = tree.node(b)
        e_bottom = node.minimum_energy

        anc = node.parent
        while anc is not None and anc not in disp:
            anc = tree.node(anc).parent

        if node.saddle_to_parent is None or anc is None:
            e_top = top_energy
        else:
            e_top = node.saddle_to_parent[1]
            hsegs.append((x_of[b], x_of[anc], e_top, b))

        vsegs[b] = (x_of[b], e_bottom, e_top)

    return x_of, vsegs, hsegs


def _merge_tree_label(idx, energy, axes, components):
    """One-field-per-line label: every active axis, then total/micro/macro E.

    Parameters
    ----------
    idx : sequence of int
        Grid cell — a basin's ``minimum_index`` or a saddle index.
    energy : float
        Total energy at the cell (shown as ``E``).
    axes : dict[str, np.ndarray]
        Active-axis name -> sorted values, in grid-axis order.
    components : dict[str, np.ndarray] | None
        Energy-component grids. ``micro_energy`` / ``macro_energy`` are read at
        ``idx`` when present; a missing grid or a NaN value drops that line so a
        release-build ``nan`` never reaches the figure.
    """
    cell = tuple(int(i) for i in idx)
    lines = [f"{name}={float(values[cell[pos]]):.3f}"
             for pos, (name, values) in enumerate(axes.items())]
    lines.append(f"E={energy:.3f}")
    for key, tag in (("micro_energy", "Emic"), ("macro_energy", "Emac")):
        grid = (components or {}).get(key)
        if grid is None:
            continue
        val = float(grid[cell])
        if not np.isnan(val):
            lines.append(f"{tag}={val:.3f}")
    return "\n".join(lines)


def _resolve_label_positions(anchors, sizes, base_dx=12.0, gap=2.0):
    """Greedy non-overlapping placement of right-side labels, in pixel space.

    Each label is placed to the right of its anchor and, when it would overlap
    an already-placed label, nudged upward until clear. Input order sets
    priority: earlier labels are placed first and stay lower.

    Parameters
    ----------
    anchors : list[tuple[float, float]]
        (x, y) anchor pixel coords (the marker each label belongs to).
    sizes : list[tuple[float, float]]
        (width, height) of each label box in pixels.
    base_dx : float
        Horizontal gap (px) from the anchor to the label's left edge.
    gap : float
        Minimum gap (px) kept between stacked label boxes.

    Returns
    -------
    list[tuple[float, float]]
        (x0, y0) bottom-left pixel position per label, in input order.
    """
    placed = []          # (x0, y0, x1, y1) of already-positioned boxes
    out = []
    for (ax_pix, ay_pix), (w, h) in zip(anchors, sizes):
        x0 = ax_pix + base_dx
        x1 = x0 + w
        y0 = ay_pix - h / 2.0                     # centered on the anchor
        moved = True
        while moved:                              # only ever moves upward
            moved = False
            for px0, py0, px1, py1 in placed:
                if x0 < px1 and x1 > px0 and y0 < py1 and (y0 + h) > py0:
                    y0 = py1 + gap                # sit just above the collision
                    moved = True
        placed.append((x0, y0, x1, y0 + h))
        out.append((x0, y0))
    return out


# Role -> (matplotlib marker, face color, label) for the named critical points.
# Mirrors create_contour_plot's marker_styles so the two figures agree.
_MERGE_TREE_ROLE_STYLE = {
    "ground_state":      ("o", "lime",   "Ground State"),
    "secondary_minimum": ("o", "red",    "Secondary Minimum"),
    "third_minimum":     ("o", "orange", "Third Minimum"),
    "fission_exit":      ("o", "white",  "Fission Exit"),
}
_MERGE_TREE_SADDLE_STYLE = {
    "inner_saddle": ("lime",   "Inner Saddle"),
    "outer_saddle": ("red",    "Outer Saddle"),
    "third_saddle": ("orange", "Third Saddle"),
}


def _draw_merge_tree_panel(ax, tree, displayed_ids, roles, axes, c_pos,
                           components, detailed_labels):
    """Draw one dendrogram panel onto ``ax``.

    When ``detailed_labels`` is True (the pruned panel), every drawn basin
    marker and named saddle gets a stacked label (all active axes + total/
    micro/macro E); the created annotation artists are returned as
    ``[(annotation, x, energy), ...]`` so the caller can de-overlap them once
    the figure has a renderer. When False (the unpruned panel) no text is drawn
    and an empty list is returned.
    """
    disp = set(displayed_ids)

    # Top of the panel: a margin above the highest minimum / saddle on display.
    energies = [tree.node(b).minimum_energy for b in disp]
    for b in disp:
        s = tree.node(b).saddle_to_parent
        if s is not None:
            energies.append(s[1])
    e_lo, e_hi = min(energies), max(energies)
    top_energy = e_hi + 0.05 * (e_hi - e_lo + 1e-9)

    x_of, vsegs, hsegs = _merge_tree_segments(tree, disp, c_pos, top_energy)

    # Branches (vertical) and saddle connectors (horizontal).
    for _b, (x, e_bot, e_top) in vsegs.items():
        ax.plot([x, x], [e_bot, e_top], color="0.4", lw=1.2, zorder=1)
    for x_child, x_anc, e_saddle, _child in hsegs:
        ax.plot([x_child, x_anc], [e_saddle, e_saddle], color="0.4", lw=1.2, zorder=1)

    # Named-basin lookup: basin id -> role key.
    basin_role = {
        roles[k]: k for k in _MERGE_TREE_ROLE_STYLE
        if roles.get(k) is not None and roles[k] in disp
    }
    # Saddle-index -> role color, for coloring connectors.
    saddle_role_color = {}
    for k, (color, _lbl) in _MERGE_TREE_SADDLE_STYLE.items():
        s = roles.get(k)
        if s is not None:
            saddle_role_color[tuple(int(i) for i in s[0])] = color

    annotations = []        # (annotation artist, anchor_x, anchor_energy)

    def _label(idx, x, energy):
        # clip_on=False so the label survives outside the axes; bbox_inches=
        # "tight" then grows the saved figure to include it. Final position is
        # set later by _resolve_label_positions once extents are measurable.
        ann = ax.annotate(
            _merge_tree_label(idx, energy, axes, components),
            xy=(x, energy), xytext=(12, 0), textcoords="offset pixels",
            va="center", ha="left", fontsize=7, zorder=6, clip_on=False)
        annotations.append((ann, x, energy))

    # Saddle squares + labels on the matching connectors. Saddles are named
    # critical points, so they are always labeled when the panel is detailed.
    for x_child, x_anc, e_saddle, child in hsegs:
        s = tree.node(child).saddle_to_parent
        if s is None:
            continue
        color = saddle_role_color.get(tuple(int(i) for i in s[0]))
        if color is not None:
            xs = (x_child + x_anc) / 2.0
            ax.plot(xs, e_saddle, "s", color=color, markersize=9,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=4)
            if detailed_labels:
                _label(s[0], xs, e_saddle)

    # Basin markers + labels. A basin is marked iff it is named OR clears the
    # persistence floor.
    n_markers = 0
    for b in sorted(disp):      # sorted -> deterministic, reproducible placement
        node = tree.node(b)
        named = b in basin_role
        if not named and not (node.persistence > MERGE_TREE_MARKER_MIN):
            continue
        if named:
            marker, color, _lbl = _MERGE_TREE_ROLE_STYLE[basin_role[b]]
            size = 10
        else:
            marker, color, size = "o", "lightgray", 6
        x = x_of[b]
        ax.plot(x, node.minimum_energy, marker, color=color, markersize=size,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        n_markers += 1
        if detailed_labels:
            _label(node.minimum_index, x, node.minimum_energy)

    ax.set_xlim(-1, max(x_of.values()) + 1 if x_of else 1)
    ax.set_xticks([])
    return annotations


def plot_mep_profile(mep_idx, mep_e, profile, axes, sel, tree, nucleus,
                     output_filename, out=sys.stdout):
    """1-D energy profile along the MEP with critical points marked.

    X axis is the path step. Named critical points (from ``sel``) use the
    same marker styles and legend labels as the contour maps; unnamed
    profile extrema keep generic green/red triangles. Every extremum is
    annotated with its elongation c and energy. A named minimum whose
    basin minimum lies off the path (sub-noise re-anchoring) additionally
    gets a shaded box with the basin minimum energy and persistence.
    """
    if mep_e is None:
        print("    No MEP to plot", file=out)
        return
    c_pos = list(axes).index('c')
    c_vals = axes['c'][mep_idx[:, c_pos]]
    steps = sel.get("mep_steps", {})
    min_role_at = {steps[r]: r for r in _MERGE_TREE_ROLE_STYLE if r in steps}
    saddle_role_at = {steps[r]: r for r in _MERGE_TREE_SADDLE_STYLE if r in steps}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(len(mep_e)), mep_e, lw=1.2, color='tab:blue')
    for k, e in profile.minima:
        role = min_role_at.get(k)
        if role is not None:
            marker, color, label = _MERGE_TREE_ROLE_STYLE[role]
            ax.plot(k, e, marker, color=color, ms=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5, label=label)
        else:
            ax.plot(k, e, 'v', color='tab:green', ms=9)
        ax.annotate(f"c={c_vals[k]:.2f}\n{e:.2f}", (k, e), textcoords="offset points",
                    xytext=(0, -28), ha='center', fontsize=8)
        # The path's local minimum is only the basin's seed on the MEP; after
        # sub-noise re-anchoring the true critical point sits off the path,
        # so show where the basin actually bottoms out and how deep it traps.
        if role is not None:
            node = tree.node(sel[role])
            if tuple(int(v) for v in node.minimum_index) != \
                    tuple(int(v) for v in mep_idx[k]):
                pers = (f"{node.persistence:.2f}"
                        if np.isfinite(node.persistence) else "∞")
                ax.annotate(
                    f"basin min: {node.minimum_energy:.2f}\nP = {pers} MeV",
                    (k, e), textcoords="offset points", xytext=(0, -64),
                    ha='center', fontsize=8,
                    bbox={"boxstyle": 'round,pad=0.3', "facecolor": '0.9',
                          "alpha": 0.85})
    for k, e in profile.saddles:
        role = saddle_role_at.get(k)
        if role is not None:
            color, label = _MERGE_TREE_SADDLE_STYLE[role]
            ax.plot(k, e, 's', color=color, ms=9, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5, label=label)
        else:
            ax.plot(k, e, '^', color='tab:red', ms=9)
        ax.annotate(f"c={c_vals[k]:.2f}\n{e:.2f}", (k, e), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel("MEP step")
    ax.set_ylabel("Energy (MeV)")
    ax.set_title(f"{nucleus.isotope_label}: minimum-energy path, GS → fission exit")
    ax.grid(alpha=0.3)
    # Upper right is empty by construction: the path ends in the scission
    # descent, so the lower right holds the FE marker and its labels.
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"  Saved MEP profile plot: {output_filename}", file=out)


def plot_merge_tree(tree, roles, axes, nucleus, output_filename,
                    components=None, out: TextIO = sys.stdout) -> None:
    """Two-panel merge-tree dendrogram: unpruned (all basins) vs pruned.

    ``roles`` is the ``sel`` dict from select_mep_critical_points: keys
    ``ground_state`` / ``secondary_minimum`` / ``third_minimum`` /
    ``fission_exit`` (basin ids or None) and ``inner_saddle`` / ``outer_saddle``
    / ``third_saddle`` ((index, energy) tuples or None). Energy is the vertical
    axis; x is c-ordering only. See the design spec for the full contract.
    """
    if not tree.nodes:
        print("    No basins to plot in merge tree", file=out)
        return

    axis_names = list(axes)
    c_pos = axis_names.index("c") if "c" in axis_names else 0

    all_ids = list(tree.nodes.keys())
    basins = [(tree.node(b).minimum_index, tree.node(b).minimum_energy)
              for b in sorted(tree.nodes)]
    surviving, _kept = prune_merge_tree(basins, tree._merges, MERGE_TREE_PRUNE)

    # Never prune away a named critical point: a shallow SM / third min / FE
    # basin would otherwise vanish from the pruned panel. Protect the minima
    # basins directly, and the child basin carrying each named saddle so its
    # connector + square still render (saddles are edges, not basins).
    protected = {roles[k] for k in _MERGE_TREE_ROLE_STYLE if roles.get(k) is not None}
    saddle_idx = {tuple(int(i) for i in roles[k][0])
                  for k in _MERGE_TREE_SADDLE_STYLE if roles.get(k) is not None}
    if saddle_idx:
        for b in tree.nodes:
            s = tree.node(b).saddle_to_parent
            if s is not None and tuple(int(i) for i in s[0]) in saddle_idx:
                protected.add(b)
    surviving = sorted(set(surviving) | protected)

    panels = [
        ("Unpruned", all_ids),
        (f"Pruned ≤{MERGE_TREE_PRUNE:g} MeV", surviving),
    ]

    with _PLOT_LOCK:
        fig = Figure(figsize=(14, 8))
        title = nucleus.isotope_label or Path(output_filename).stem

        right_ax = None
        right_annotations = []
        for col, (panel_label, ids) in enumerate(panels):
            ax = fig.add_subplot(1, 2, col + 1)
            detailed = (col == 1)                 # right panel = pruned
            anns = _draw_merge_tree_panel(ax, tree, ids, roles, axes, c_pos,
                                          components, detailed)
            if detailed:
                right_ax, right_annotations = ax, anns
            ax.set_title(f"{panel_label}  (n={len(ids)})", fontsize=13)
            ax.set_ylabel("E (MeV)", fontsize=13)

        # De-overlap the right-panel labels. Measuring text extents needs a
        # renderer, so attach an Agg canvas and draw once before repositioning.
        # Process bottom-to-top so stacking (which only moves labels up) is
        # deterministic.
        if right_annotations:
            right_annotations.sort(key=lambda t: t[2])
            FigureCanvasAgg(fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            anchors, sizes, anns = [], [], []
            for ann, x, energy in right_annotations:
                bbox = ann.get_window_extent(renderer)
                anchors.append(tuple(right_ax.transData.transform((x, energy))))
                sizes.append((bbox.width, bbox.height))
                anns.append(ann)
            placements = _resolve_label_positions(anchors, sizes)
            for ann, (ax_pix, ay_pix), (x0, y0) in zip(anns, anchors, placements):
                # set_position is in the annotation's textcoords (offset pixels),
                # i.e. relative to its anchor point.
                ann.set_position((x0 - ax_pix, y0 - ay_pix))

        # One shared legend (named criticals) on the LEFT (text-free) panel.
        handles = []
        for k, (marker, color, lbl) in _MERGE_TREE_ROLE_STYLE.items():
            if roles.get(k) is not None:
                handles.append(Line2D([], [], marker=marker, color="none",
                                      markerfacecolor=color, markeredgecolor="black",
                                      markersize=9, label=lbl))
        for k, (color, lbl) in _MERGE_TREE_SADDLE_STYLE.items():
            if roles.get(k) is not None:
                handles.append(Line2D([], [], marker="s", color="none",
                                      markerfacecolor=color, markeredgecolor="black",
                                      markersize=9, label=lbl))
        if handles:
            # Default to upper right, but drop to lower left when the deepest
            # basin sits below -20 MeV — there the legend would overlap the tree.
            lowest = min((e for _, e in basins), default=float("inf"))
            loc = "lower left" if lowest < -20.0 else "upper right"
            fig.axes[0].legend(handles=handles, loc=loc, fontsize=9,
                               framealpha=0.95)

        fig.suptitle(f"Merge tree — {title}", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_filename, dpi=DPI, bbox_inches="tight")

    print(f"  Saved merge-tree plot: {output_filename}", file=out)


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
                        scission: Optional[np.ndarray] = None,
                        mep: Optional[tuple[np.ndarray, np.ndarray]] = None,
                        out: TextIO = sys.stdout):
    """Create a c vs y contour plot with critical points marked.

    Args:
        y_param: 'a4' or 'a3' — controls y-axis label, tick spacing,
                 and which coordinate to read from critical points.
        scission: optional bool array aligned with c/y/energy; True points
                  are shaded black with alpha (the scission region).
        mep: optional (c, y) coordinate arrays of the minimum-energy path,
             projected onto this map's two axes.
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
    # gaussian_filter propagates NaN to every pixel its kernel touches, which
    # would blanket a wide margin around invalid (e.g. non-star-convex) data.
    # Fill holes with nearest-neighbour values before smoothing, then restore
    # them so the invalid region stays white without bleeding into valid map.
    hole_mask = np.isnan(zi)
    if hole_mask.any():
        zi_near = griddata((c, y), energy, (ci, yi), method='nearest')
        zi[hole_mask] = zi_near[hole_mask]
    zi = gaussian_filter(zi, sigma=2.0)
    zi[hole_mask] = np.nan

    # Scission mask on the same fine grid. Nearest-neighbour keeps the 0/1
    # field crisp (cubic would ring at the boundary); holes stay unshaded.
    sc_mask = None
    if scission is not None and scission.any():
        sc_fine = griddata((c, y), scission.astype(np.float64), (ci, yi),
                           method='nearest')
        sc_mask = (sc_fine > 0.5) & ~hole_mask

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

        if sc_mask is not None:
            ax.contourf(ci, yi, sc_mask.astype(float), levels=[0.5, 1.5],
                        colors='black', alpha=0.35, zorder=3)

        # MEP projected onto this map's axes. The path lives on the full N-D
        # grid while the map is minimized over the other axes, so it can
        # stray off the displayed valley floor — that is real information,
        # not a bug. Red stays visible: high-energy red regions are where
        # the path never goes.
        if mep is not None:
            mep_c, mep_y = mep
            ax.plot(mep_c, mep_y, color='red', lw=1.8, ls=(0, (3, 1.5)),
                    zorder=8, label='MEP')

        # Plot critical points
        if critical_points:
            marker_styles = {
                CriticalPointType.GROUND_STATE: ('o', 'lime', 10, 'Ground State'),
                CriticalPointType.FIRST_SADDLE: ('s', 'lime', 8, 'Inner Saddle'),
                CriticalPointType.SECONDARY_MINIMUM: ('o', 'red', 10, 'Secondary Minimum'),
                CriticalPointType.SECOND_SADDLE: ('s', 'red', 8, 'Outer Saddle'),
                CriticalPointType.THIRD_MINIMUM: ('o', 'orange', 10, 'Third Minimum'),
                CriticalPointType.THIRD_SADDLE: ('s', 'orange', 8, 'Third Saddle'),
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
                # contourf is not auto-collected into the legend; add a proxy.
                handles, _ = ax.get_legend_handles_labels()
                if sc_mask is not None:
                    handles.append(Patch(facecolor='black', alpha=0.35,
                                         label='Scission ($r_{neck}$ < 1.5 fm)'))
                ax.legend(handles=handles, loc='lower right', fontsize=5,
                          markerscale=0.5, framealpha=0.95)

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

    (critical_points, energies, axes, inactive_axes, components, scission_grid,
     tree, sel, mep_idx, mep_e, mep_profile) = \
        run_critical_point_analysis_nd(df, out=out)

    print_analysis_summary(critical_points, nucleus, out=out)
    save_critical_points_csv(critical_points, nucleus,
                             gs_persistence_bf=sel["gs_persistence_bf"], out=out)
    save_basins_csv(tree, axes, nucleus, out=out)
    plot_persistence_histogram(
        tree, nucleus,
        f'{parquet_file.stem}_persistence_hist.png', out=out,
    )
    plot_merge_tree(
        tree, sel, axes, nucleus,
        f'{parquet_file.stem}_merge_tree.png', components=components, out=out,
    )
    plot_mep_profile(
        mep_idx, mep_e, mep_profile, axes, sel, tree, nucleus,
        f'{parquet_file.stem}_mep_profile.png', out=out,
    )

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
        mep_proj = None
        if mep_idx is not None:
            axis_names = list(axes)
            mep_proj = (axes['c'][mep_idx[:, axis_names.index('c')]],
                        axes[y_axis][mep_idx[:, axis_names.index(y_axis)]])
        scission_flat = None
        if scission_grid is not None:
            if SCISSION_FULL:
                scission_flat = project_any_2d(scission_grid, axes, 'c', y_axis)[mask]
            else:
                sc2d = gather_at_argmin_2d(scission_grid, axes, 'c', y_axis, argmin_flat)
                scission_flat = (np.where(np.isnan(sc2d), 0.0, sc2d) > 0.5)[mask]
        create_contour_plot(c_flat, y_flat, e_flat, nucleus, output_filename,
                            critical_points, y_param=y_axis,
                            scission=scission_flat, mep=mep_proj, out=out)

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
    print(f"  Elongation windows: GS/SM first {GS_SM_C_FRAC:.0%} of c-range, "
          f"TM first {TM_C_FRAC:.0%}")
    print(f"  3D-only mask (a5=a6=a7=a8=0): {MASK_3D_ONLY}")
    print(f"  Scission overlay: {'any shape in column' if SCISSION_FULL else 'min-energy configuration'}")
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
        # heavy work (find_watershed_segmentation) releases the GIL.
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
