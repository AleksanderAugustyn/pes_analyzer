#!/usr/bin/env python3
"""
Enhanced PES Map Maker with Critical Point Analysis for FoS Parameterization.

Reads FoS (Fourier-over-Spheroid) parameterization data from Parquet files with schema:
    c, a3, a4, a5, a6, a7, a8, is_valid, mass_excess, total_energy,
    macro_energy, micro_energy, surface_energy, coulomb_energy, proton_k, neutron_k, proton_gap, neutron_gap

Features:
    1. Auto-discovers pes_ZXX_NYYY.parquet files if no specific file is given
    2. Performs critical point analysis (ground state, secondary minimum,
       fission exit, inner saddle, outer saddle)
    3. Saves critical points to CSV for later use
    4. Plots contour maps with critical points marked

Usage:
    python MapMaker_FoS_enhanced.py                          # Process all found files
    python MapMaker_FoS_enhanced.py pes_Z92_N144.parquet     # Process specific file
    python MapMaker_FoS_enhanced.py --help                   # Show help

Based on:
    - MapMaker_FoS.py (plotting)
    - MapMaker_a5a6.py (file discovery)
    - PESAnalyzer_a5a6_parquet_main.cpp (critical point analysis)
"""
import argparse
import io
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from pes_analyzer.saddle  import find_iwf_grid
from pes_analyzer.minimum import find_minima_grid
from pes_analyzer.grid    import build_dense

# =============================================================================
# CONFIGURATION
# =============================================================================

# General settings
ASPECT_RATIO = 'equal'
DPI = 300
USE_FLOAT32 = True

# 5D analysis: when True, run saddle point detection on full 5D (c, a3, a4, a5, a6) space
# instead of the minimized 2D (c, a4) space
USE_5D_ANALYSIS = True

# Columns to minimize over (all deformation parameters except c and a4)
MINIMIZE_OVER = ['a3', 'a5', 'a6', 'a7', 'a8']

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
SECONDARY_MINIMUM_C_MAX_THRESHOLD = 1.5
SECONDARY_MINIMUM_A3_MAX_THRESHOLD = 0.10
SECONDARY_MINIMUM_A4_MAX_THRESHOLD = 0.20


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
    """Represents a point on the deformation grid with all coordinates and energy."""
    c: float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    a6: float = 0.0
    a7: float = 0.0
    a8: float = 0.0
    c_idx: int = 0
    a4_idx: int = 0
    total_energy: float = np.inf
    mass_excess: float = np.nan
    macro_energy: float = np.nan
    micro_energy: float = np.nan
    surface_energy: float = np.nan
    coulomb_energy: float = np.nan
    valid: bool = False


@dataclass
class CriticalPoint:
    """Represents a critical point found during analysis."""
    point_type: CriticalPointType
    name: str
    point: GridPoint = field(default_factory=GridPoint)
    found: bool = False


@dataclass
class Grid5D:
    """Sparse 5D energy grid for saddle point detection.

    Uses sparse dictionary storage since dense array would require ~30GB+ for typical grids.
    Ignores a7 and a8 dimensions (always 0.0 in current data).
    """
    unique_c: np.ndarray
    unique_a3: np.ndarray
    unique_a4: np.ndarray
    unique_a5: np.ndarray
    unique_a6: np.ndarray
    nc: int
    na3: int
    na4: int
    na5: int
    na6: int
    energy_map: dict  # {(ic, ia3, ia4, ia5, ia6): energy}
    energies: np.ndarray  # dense (nc, na3, na4, na5, na6) float64, NaN = missing
    deform_map: dict  # {(ic, ia3, ia4, ia5, ia6): (a7, a8)} - store other deformation values
    # Energy components map: {key: (mass_excess, macro_energy, micro_energy, surface_energy, coulomb_energy)}
    energy_components_map: dict
    c_to_idx: dict
    a3_to_idx: dict
    a4_to_idx: dict
    a5_to_idx: dict
    a6_to_idx: dict

    def to_linear_index(self, ic: int, ia3: int, ia4: int, ia5: int, ia6: int) -> int:
        """Convert 5D indices to a linear index for Union-Find."""
        return (((ic * self.na3 + ia3) * self.na4 + ia4) * self.na5 + ia5) * self.na6 + ia6

    def get_neighbors_5d(self, ic: int, ia3: int, ia4: int, ia5: int, ia6: int) -> list:
        """Return 10-connected face-adjacent neighbors in 5D.

        In 5D, each point has up to 10 face-adjacent neighbors (2 per dimension).
        """
        neighbors = []
        deltas = [
            (-1, 0, 0, 0, 0), (1, 0, 0, 0, 0),  # c direction
            (0, -1, 0, 0, 0), (0, 1, 0, 0, 0),  # a3 direction
            (0, 0, -1, 0, 0), (0, 0, 1, 0, 0),  # a4 direction
            (0, 0, 0, -1, 0), (0, 0, 0, 1, 0),  # a5 direction
            (0, 0, 0, 0, -1), (0, 0, 0, 0, 1),  # a6 direction
        ]
        for dc, da3, da4, da5, da6 in deltas:
            n = (ic + dc, ia3 + da3, ia4 + da4, ia5 + da5, ia6 + da6)
            if (0 <= n[0] < self.nc and 0 <= n[1] < self.na3 and
                    0 <= n[2] < self.na4 and 0 <= n[3] < self.na5 and
                    0 <= n[4] < self.na6 and n in self.energy_map):
                neighbors.append(n)
        return neighbors


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


def apply_parameter_limits(df: pl.DataFrame) -> pl.DataFrame:
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
        print(f"    Applied parameter limits:")
        for limit in active_limits:
            print(f"      - {limit}")
        print(f"    Filtered: {initial_count:,} -> {filtered_count:,} points "
              f"({initial_count - filtered_count:,} removed)")

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

def read_parquet_file(filename: str | Path) -> pl.DataFrame:
    """Read the Parquet file and extract relevant columns for analysis."""
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if filepath.suffix != '.parquet':
        raise ValueError(f"Expected a Parquet file, got: {filepath.suffix}")

    needed_columns = ['c', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'is_valid',
                      'total_energy', 'mass_excess', 'macro_energy', 'micro_energy',
                      'surface_energy', 'coulomb_energy']

    print(f"  Reading Parquet file: {filepath.name}")
    print(f"    File size: {get_file_size_mb(filepath):.1f} MB")
    start_time = time.time()

    try:
        # Get schema to check available columns
        schema = pl.read_parquet_schema(filepath)
        available_columns = list(schema.keys())
        print(f"    Parquet file contains {len(available_columns)} columns")

        # Use available columns
        columns_to_read = [col for col in needed_columns if col in available_columns]
        missing = [col for col in needed_columns if col not in available_columns]
        if missing:
            print(f"    Note: Missing columns (will be NaN): {missing}")

        df = pl.read_parquet(filepath, columns=columns_to_read)
        read_time = time.time() - start_time
        print(f"    Read {len(df):,} rows in {read_time:.1f} seconds")

        # Filter valid points
        initial_count = len(df)
        if 'is_valid' in df.columns:
            df = df.filter(pl.col('is_valid') == True)
            filtered_count = len(df)
            print(f"    Filtered to {filtered_count:,} valid points "
                  f"({initial_count - filtered_count:,} invalid removed)")
            df = df.drop('is_valid')

        # Filter out rows with extremely low total_energy (likely invalid)
        if 'total_energy' in df.columns:
            before_energy_filter = len(df)
            df = df.filter(pl.col('total_energy') >= -300)
            after_energy_filter = len(df)
            if before_energy_filter != after_energy_filter:
                print(f"    Filtered out {before_energy_filter - after_energy_filter:,} points "
                      f"with total_energy < -300 MeV")

        # Apply parameter limits
        df = apply_parameter_limits(df)

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

def minimize_over_deformations(df: pl.DataFrame) -> tuple:
    """
    For each (c, a4), find the minimum total_energy over a3, a5, a6, a7, a8.

    Returns:
        Tuple of arrays: (c, a4, energy, a3_min, a5_min, a6_min, a7_min, a8_min)
    """
    print("    Minimizing over a3, a5, a6, a7, a8...")
    start_time = time.time()

    # Group by (c, a4) and select the row with minimum total_energy
    min_rows = (
        df.sort('total_energy')
        .group_by(['c', 'a4'])
        .first()
    )

    min_time = time.time() - start_time
    print(f"    Minimization completed in {min_time:.1f} seconds")
    print(f"    Reduced to {len(min_rows):,} grid points")

    def get_column_or_nan(col_name: str) -> np.ndarray:
        if col_name in min_rows.columns:
            return min_rows[col_name].to_numpy()
        return np.full(len(min_rows), np.nan)

    return (
        min_rows['c'].to_numpy(),
        min_rows['a4'].to_numpy(),
        min_rows['total_energy'].to_numpy(),
        min_rows['a3'].to_numpy(),
        min_rows['a5'].to_numpy(),
        min_rows['a6'].to_numpy(),
        min_rows['a7'].to_numpy(),
        min_rows['a8'].to_numpy(),
        get_column_or_nan('mass_excess'),
        get_column_or_nan('macro_energy'),
        get_column_or_nan('micro_energy'),
        get_column_or_nan('surface_energy'),
        get_column_or_nan('coulomb_energy'),
    )


def minimize_over_deformations_ca3(df: pl.DataFrame) -> tuple:
    """
    For each (c, a3), find the minimum total_energy over a4, a5, a6, a7, a8.

    Returns:
        Tuple of arrays: (c, a3, energy, a4_min, a5_min, a6_min, a7_min, a8_min,
                          mass_excess, macro_energy, micro_energy, surface_energy, coulomb_energy)
    """
    print("    Minimizing over a4, a5, a6, a7, a8 (c vs a3 surface)...")
    start_time = time.time()

    min_rows = (
        df.sort('total_energy')
        .group_by(['c', 'a3'])
        .first()
    )

    min_time = time.time() - start_time
    print(f"    Minimization completed in {min_time:.1f} seconds")
    print(f"    Reduced to {len(min_rows):,} grid points")

    def get_column_or_nan(col_name: str) -> np.ndarray:
        if col_name in min_rows.columns:
            return min_rows[col_name].to_numpy()
        return np.full(len(min_rows), np.nan)

    return (
        min_rows['c'].to_numpy(),
        min_rows['a3'].to_numpy(),
        min_rows['total_energy'].to_numpy(),
        min_rows['a4'].to_numpy(),
        min_rows['a5'].to_numpy(),
        min_rows['a6'].to_numpy(),
        min_rows['a7'].to_numpy(),
        min_rows['a8'].to_numpy(),
        get_column_or_nan('mass_excess'),
        get_column_or_nan('macro_energy'),
        get_column_or_nan('micro_energy'),
        get_column_or_nan('surface_energy'),
        get_column_or_nan('coulomb_energy'),
    )


def build_2d_energy_grid(c: np.ndarray, a4: np.ndarray, energy: np.ndarray,
                         a3: np.ndarray, a5: np.ndarray, a6: np.ndarray,
                         a7: np.ndarray, a8: np.ndarray,
                         mass_excess: np.ndarray = None, macro_energy: np.ndarray = None,
                         micro_energy: np.ndarray = None, surface_energy: np.ndarray = None,
                         coulomb_energy: np.ndarray = None) -> dict:
    """
    Build a 2D grid data structure for (c, a4) analysis.

    Returns a dictionary with:
        - unique_c, unique_a4: sorted unique values
        - energy_grid: 2D array [c_idx][a4_idx]
        - deform_grids: dict of 2D arrays for a3, a5, a6, a7, a8
        - energy_component_grids: dict of 2D arrays for mass_excess, macro_energy, etc.
        - coord_to_idx: mapping from (c, a4) to linear index
        - c_to_idx, a4_to_idx: value to index mappings
    """
    unique_c = np.sort(np.unique(c))
    unique_a4 = np.sort(np.unique(a4))

    nc, na4 = len(unique_c), len(unique_a4)
    c_to_idx = {val: idx for idx, val in enumerate(unique_c)}
    a4_to_idx = {val: idx for idx, val in enumerate(unique_a4)}

    energy_grid = np.full((nc, na4), np.nan)
    a3_grid = np.full((nc, na4), np.nan)
    a5_grid = np.full((nc, na4), np.nan)
    a6_grid = np.full((nc, na4), np.nan)
    a7_grid = np.full((nc, na4), np.nan)
    a8_grid = np.full((nc, na4), np.nan)

    # Energy component grids
    mass_excess_grid = np.full((nc, na4), np.nan)
    macro_energy_grid = np.full((nc, na4), np.nan)
    micro_energy_grid = np.full((nc, na4), np.nan)
    surface_energy_grid = np.full((nc, na4), np.nan)
    coulomb_energy_grid = np.full((nc, na4), np.nan)

    # Default to NaN arrays if not provided
    if mass_excess is None:
        mass_excess = np.full(len(c), np.nan)
    if macro_energy is None:
        macro_energy = np.full(len(c), np.nan)
    if micro_energy is None:
        micro_energy = np.full(len(c), np.nan)
    if surface_energy is None:
        surface_energy = np.full(len(c), np.nan)
    if coulomb_energy is None:
        coulomb_energy = np.full(len(c), np.nan)

    coord_to_idx = {}
    for i, (ci, a4i, ei, a3i, a5i, a6i, a7i, a8i, me, macro, micro, surf, coul) in enumerate(
            zip(c, a4, energy, a3, a5, a6, a7, a8,
                mass_excess, macro_energy, micro_energy, surface_energy, coulomb_energy)):
        ic = c_to_idx[ci]
        ia4 = a4_to_idx[a4i]
        energy_grid[ic, ia4] = ei
        a3_grid[ic, ia4] = a3i
        a5_grid[ic, ia4] = a5i
        a6_grid[ic, ia4] = a6i
        a7_grid[ic, ia4] = a7i
        a8_grid[ic, ia4] = a8i
        mass_excess_grid[ic, ia4] = me
        macro_energy_grid[ic, ia4] = macro
        micro_energy_grid[ic, ia4] = micro
        surface_energy_grid[ic, ia4] = surf
        coulomb_energy_grid[ic, ia4] = coul
        coord_to_idx[(ci, a4i)] = i

    return {
        'unique_c': unique_c,
        'unique_a4': unique_a4,
        'energy_grid': energy_grid,
        'a3_grid': a3_grid,
        'a5_grid': a5_grid,
        'a6_grid': a6_grid,
        'a7_grid': a7_grid,
        'a8_grid': a8_grid,
        'mass_excess_grid': mass_excess_grid,
        'macro_energy_grid': macro_energy_grid,
        'micro_energy_grid': micro_energy_grid,
        'surface_energy_grid': surface_energy_grid,
        'coulomb_energy_grid': coulomb_energy_grid,
        'c_to_idx': c_to_idx,
        'a4_to_idx': a4_to_idx,
        'coord_to_idx': coord_to_idx,
        'nc': nc,
        'na4': na4,
    }


def build_5d_energy_grid(df: pl.DataFrame) -> Grid5D:
    """Build sparse 5D grid from DataFrame (ignoring a7, a8).

    Creates a sparse representation of the 5D energy landscape over (c, a3, a4, a5, a6).
    Dense storage would require ~30GB+ for typical grids, so we use a dictionary.

    Args:
        df: Polars DataFrame with columns c, a3, a4, a5, a6, a7, a8, total_energy

    Returns:
        Grid5D object with sparse energy storage
    """
    print("    Building 5D energy grid...")
    start_time = time.time()

    # Extract unique sorted values for each dimension
    unique_c = np.sort(df['c'].unique().to_numpy())
    unique_a3 = np.sort(df['a3'].unique().to_numpy())
    unique_a4 = np.sort(df['a4'].unique().to_numpy())
    unique_a5 = np.sort(df['a5'].unique().to_numpy())
    unique_a6 = np.sort(df['a6'].unique().to_numpy())

    nc, na3, na4, na5, na6 = len(unique_c), len(unique_a3), len(unique_a4), len(unique_a5), len(unique_a6)

    print(f"      Grid dimensions: c={nc}, a3={na3}, a4={na4}, a5={na5}, a6={na6}")
    print(f"      Total possible points: {nc * na3 * na4 * na5 * na6:,}")

    # Build index mappings
    c_to_idx = {val: idx for idx, val in enumerate(unique_c)}
    a3_to_idx = {val: idx for idx, val in enumerate(unique_a3)}
    a4_to_idx = {val: idx for idx, val in enumerate(unique_a4)}
    a5_to_idx = {val: idx for idx, val in enumerate(unique_a5)}
    a6_to_idx = {val: idx for idx, val in enumerate(unique_a6)}

    # Populate sparse energy map
    energy_map = {}
    deform_map = {}
    energy_components_map = {}
    # Dense shadow of energy_map for pes_analyzer.saddle.find_iwf_grid; NaN = missing.
    energies = np.full((nc, na3, na4, na5, na6), np.nan, dtype=np.float64)

    # Extract arrays for fast iteration
    c_arr = df['c'].to_numpy()
    a3_arr = df['a3'].to_numpy()
    a4_arr = df['a4'].to_numpy()
    a5_arr = df['a5'].to_numpy()
    a6_arr = df['a6'].to_numpy()
    a7_arr = df['a7'].to_numpy() if 'a7' in df.columns else np.zeros(len(df))
    a8_arr = df['a8'].to_numpy() if 'a8' in df.columns else np.zeros(len(df))
    energy_arr = df['total_energy'].to_numpy()

    # Energy components
    mass_excess_arr = df['mass_excess'].to_numpy() if 'mass_excess' in df.columns else np.full(len(df), np.nan)
    macro_energy_arr = df['macro_energy'].to_numpy() if 'macro_energy' in df.columns else np.full(len(df), np.nan)
    micro_energy_arr = df['micro_energy'].to_numpy() if 'micro_energy' in df.columns else np.full(len(df), np.nan)
    surface_energy_arr = df['surface_energy'].to_numpy() if 'surface_energy' in df.columns else np.full(len(df), np.nan)
    coulomb_energy_arr = df['coulomb_energy'].to_numpy() if 'coulomb_energy' in df.columns else np.full(len(df), np.nan)

    for i in range(len(df)):
        ic = c_to_idx[c_arr[i]]
        ia3 = a3_to_idx[a3_arr[i]]
        ia4 = a4_to_idx[a4_arr[i]]
        ia5 = a5_to_idx[a5_arr[i]]
        ia6 = a6_to_idx[a6_arr[i]]

        key = (ic, ia3, ia4, ia5, ia6)
        energy_map[key] = energy_arr[i]
        energies[ic, ia3, ia4, ia5, ia6] = energy_arr[i]
        deform_map[key] = (a7_arr[i], a8_arr[i])
        energy_components_map[key] = (
            mass_excess_arr[i], macro_energy_arr[i], micro_energy_arr[i],
            surface_energy_arr[i], coulomb_energy_arr[i]
        )

    build_time = time.time() - start_time
    print(f"      Actual points stored: {len(energy_map):,}")
    print(f"      Memory estimate: ~{len(energy_map) * 100 / 1024 / 1024:.1f} MB")
    print(f"      Build time: {build_time:.1f} seconds")

    if __debug__:
        # Both structures use last-write-wins for duplicate keys. The dense array
        # has n_non_nan valid cells; energy_map includes NaN-valued entries for
        # rows where total_energy is NaN. Compare unique cell counts correctly.
        n_non_nan = int(np.count_nonzero(~np.isnan(energies)))
        n_nan_in_map = sum(1 for v in energy_map.values() if np.isnan(v))
        assert n_non_nan == len(energy_map) - n_nan_in_map, (
            f"dense/sparse size mismatch: {n_non_nan} non-NaN cells vs "
            f"{len(energy_map) - n_nan_in_map} non-NaN dict entries"
        )

    return Grid5D(
        unique_c=unique_c,
        unique_a3=unique_a3,
        unique_a4=unique_a4,
        unique_a5=unique_a5,
        unique_a6=unique_a6,
        nc=nc, na3=na3, na4=na4, na5=na5, na6=na6,
        energy_map=energy_map,
        energies=energies,
        deform_map=deform_map,
        energy_components_map=energy_components_map,
        c_to_idx=c_to_idx,
        a3_to_idx=a3_to_idx,
        a4_to_idx=a4_to_idx,
        a5_to_idx=a5_to_idx,
        a6_to_idx=a6_to_idx,
    )


# =============================================================================
# CRITICAL POINT ANALYSIS
# =============================================================================

class DisjointSetUnion:
    """Union-Find data structure for saddle point detection."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1


def is_local_minimum_2d(grid: dict, ic: int, ia4: int) -> bool:
    """Check if a point is a local minimum in the 2D energy grid."""
    energy_grid = grid['energy_grid']
    nc, na4 = grid['nc'], grid['na4']

    if np.isnan(energy_grid[ic, ia4]):
        return False

    center_energy = energy_grid[ic, ia4]
    has_valid_neighbor = False

    # 8-connected neighbors
    for dc in [-1, 0, 1]:
        for da4 in [-1, 0, 1]:
            if dc == 0 and da4 == 0:
                continue
            nic, nia4 = ic + dc, ia4 + da4
            if 0 <= nic < nc and 0 <= nia4 < na4:
                neighbor_energy = energy_grid[nic, nia4]
                if not np.isnan(neighbor_energy):
                    has_valid_neighbor = True
                    if neighbor_energy < center_energy:
                        return False

    return has_valid_neighbor


def find_ground_state(grid: dict, c: np.ndarray, a4: np.ndarray,
                      energy: np.ndarray, a3: np.ndarray, a5: np.ndarray,
                      a6: np.ndarray, a7: np.ndarray, a8: np.ndarray,
                      mass_excess: np.ndarray = None, macro_energy: np.ndarray = None,
                      micro_energy: np.ndarray = None, surface_energy: np.ndarray = None,
                      coulomb_energy: np.ndarray = None) -> CriticalPoint:
    """
    Find the ground state: lowest energy point below c threshold.
    If global minimum is above threshold, find lowest energy below threshold.
    """
    gs = CriticalPoint(CriticalPointType.GROUND_STATE, "Ground State")

    # Default to NaN arrays if not provided
    n = len(c)
    if mass_excess is None:
        mass_excess = np.full(n, np.nan)
    if macro_energy is None:
        macro_energy = np.full(n, np.nan)
    if micro_energy is None:
        micro_energy = np.full(n, np.nan)
    if surface_energy is None:
        surface_energy = np.full(n, np.nan)
    if coulomb_energy is None:
        coulomb_energy = np.full(n, np.nan)

    # Find global minimum first
    valid_mask = ~np.isnan(energy)
    if not np.any(valid_mask):
        print("    Warning: No valid points found!")
        return gs

    global_min_idx = np.argmin(np.where(valid_mask, energy, np.inf))
    global_min_c = c[global_min_idx]
    global_min_energy = energy[global_min_idx]

    print(f"    Global minimum: c={global_min_c:.4f}, E={global_min_energy:.4f} MeV")

    # Check if global minimum satisfies threshold
    if global_min_c <= GROUND_STATE_C_THRESHOLD:
        print(f"    Global minimum satisfies c <= {GROUND_STATE_C_THRESHOLD}, using as ground state")
        idx = global_min_idx
    else:
        print(f"    Global minimum has c > {GROUND_STATE_C_THRESHOLD}, "
              f"searching for lowest energy with c <= {GROUND_STATE_C_THRESHOLD}")
        mask = (c <= GROUND_STATE_C_THRESHOLD) & valid_mask
        if not np.any(mask):
            print("    Warning: No valid points below threshold!")
            return gs
        idx = np.argmin(np.where(mask, energy, np.inf))

    gs.point = GridPoint(
        c=c[idx], a3=a3[idx], a4=a4[idx], a5=a5[idx], a6=a6[idx],
        a7=a7[idx], a8=a8[idx], total_energy=energy[idx],
        mass_excess=mass_excess[idx], macro_energy=macro_energy[idx],
        micro_energy=micro_energy[idx], surface_energy=surface_energy[idx],
        coulomb_energy=coulomb_energy[idx], valid=True
    )
    gs.found = True
    print(f"    ✓ Ground State: c={gs.point.c:.4f}, a3={gs.point.a3:.4f}, "
          f"a4={gs.point.a4:.4f}, E={gs.point.total_energy:.4f} MeV")

    return gs


def find_secondary_minimum(grid: dict, gs: CriticalPoint,
                           c: np.ndarray, a4: np.ndarray, energy: np.ndarray,
                           a3: np.ndarray, a5: np.ndarray, a6: np.ndarray,
                           a7: np.ndarray, a8: np.ndarray) -> CriticalPoint:
    """
    Find the secondary minimum: a local minimum beyond the ground state,
    with c and a4 thresholds.
    """
    sm = CriticalPoint(CriticalPointType.SECONDARY_MINIMUM, "Secondary Minimum")

    if not gs.found:
        print("    Cannot find secondary minimum without ground state")
        return sm

    c_threshold = gs.point.c + 0.1
    a4_threshold = gs.point.a4 + 0.1
    print(f"    Searching for secondary minimum with (c > {c_threshold:.4f} or a4 > {a4_threshold:.4f}) and c > {gs.point.c:.4f}")

    unique_c = grid['unique_c']
    unique_a4 = grid['unique_a4']
    energy_grid = grid['energy_grid']
    nc, na4 = grid['nc'], grid['na4']

    candidates = []

    # Find all local minima in 2D grid
    for ic in range(nc):
        for ia4 in range(na4):
            if is_local_minimum_2d(grid, ic, ia4):
                c_val = unique_c[ic]
                a4_val = unique_a4[ia4]

                # Always require c > gs.c (physical requirement for fission isomer)
                if c_val <= gs.point.c:
                    continue
                # Also skip if too close to ground state in both c and a4
                if c_val <= c_threshold and a4_val <= a4_threshold:
                    continue

                # Apply additional thresholds
                a3_val = grid['a3_grid'][ic, ia4]
                if (c_val > SECONDARY_MINIMUM_C_MAX_THRESHOLD or
                        a3_val > SECONDARY_MINIMUM_A3_MAX_THRESHOLD or a4_val > SECONDARY_MINIMUM_A4_MAX_THRESHOLD):
                    continue

                candidates.append({
                    'c': c_val, 'a4': a4_val,
                    'a3': a3_val,
                    'a5': grid['a5_grid'][ic, ia4],
                    'a6': grid['a6_grid'][ic, ia4],
                    'a7': grid['a7_grid'][ic, ia4],
                    'a8': grid['a8_grid'][ic, ia4],
                    'energy': energy_grid[ic, ia4],
                    'mass_excess': grid['mass_excess_grid'][ic, ia4],
                    'macro_energy': grid['macro_energy_grid'][ic, ia4],
                    'micro_energy': grid['micro_energy_grid'][ic, ia4],
                    'surface_energy': grid['surface_energy_grid'][ic, ia4],
                    'coulomb_energy': grid['coulomb_energy_grid'][ic, ia4],
                })

    if candidates:
        print(f"    Found {len(candidates)} candidate(s):")
        for cand in sorted(candidates, key=lambda x: x['energy']):
            print(f"      c={cand['c']:.4f}, a3={cand['a3']:.4f}, a4={cand['a4']:.4f}, E={cand['energy']:.4f} MeV")

    if not candidates:
        print("    ✗ Secondary Minimum: Not found")
        return sm

    # Select lowest energy candidate
    best = min(candidates, key=lambda x: x['energy'])
    sm.point = GridPoint(
        c=best['c'], a3=best['a3'], a4=best['a4'], a5=best['a5'],
        a6=best['a6'], a7=best['a7'], a8=best['a8'],
        total_energy=best['energy'],
        mass_excess=best['mass_excess'], macro_energy=best['macro_energy'],
        micro_energy=best['micro_energy'], surface_energy=best['surface_energy'],
        coulomb_energy=best['coulomb_energy'], valid=True
    )
    sm.found = True
    print(f"    ✓ Secondary Minimum: c={sm.point.c:.4f}, a3={sm.point.a3:.4f}, "
          f"a4={sm.point.a4:.4f}, E={sm.point.total_energy:.4f} MeV")

    return sm


def find_fission_exit(grid: dict, sm: CriticalPoint) -> CriticalPoint:
    """
    Find the fission exit: global minimum of total_energy beyond the secondary
    minimum. For a ground-state-to-scission grid, this is the lowest-energy
    point on the descent past the outer barrier — the natural endpoint between
    which the outer saddle is sought.
    """
    fe = CriticalPoint(CriticalPointType.FISSION_EXIT, "Fission Exit")

    if not sm.found:
        print("    Cannot find fission exit without secondary minimum")
        return fe

    unique_c = grid['unique_c']
    energy_grid = grid['energy_grid']

    c_threshold = sm.point.c + 0.05
    print(f"    Searching for fission exit as global minimum with c > {c_threshold:.4f}")

    c_mask = unique_c > c_threshold
    valid_mask = c_mask[:, None] & ~np.isnan(energy_grid)

    if not np.any(valid_mask):
        print("    ✗ Fission Exit: Not found (no valid points past secondary minimum)")
        return fe

    masked_energy = np.where(valid_mask, energy_grid, np.inf)
    flat_idx = np.argmin(masked_energy)
    ic, ia4 = np.unravel_index(flat_idx, energy_grid.shape)

    fe.point = GridPoint(
        c=unique_c[ic],
        a3=grid['a3_grid'][ic, ia4],
        a4=grid['unique_a4'][ia4],
        a5=grid['a5_grid'][ic, ia4],
        a6=grid['a6_grid'][ic, ia4],
        a7=grid['a7_grid'][ic, ia4],
        a8=grid['a8_grid'][ic, ia4],
        total_energy=energy_grid[ic, ia4],
        mass_excess=grid['mass_excess_grid'][ic, ia4],
        macro_energy=grid['macro_energy_grid'][ic, ia4],
        micro_energy=grid['micro_energy_grid'][ic, ia4],
        surface_energy=grid['surface_energy_grid'][ic, ia4],
        coulomb_energy=grid['coulomb_energy_grid'][ic, ia4],
        valid=True,
    )
    fe.found = True
    print(f"    ✓ Fission Exit (global min): c={fe.point.c:.4f}, "
          f"a4={fe.point.a4:.4f}, E={fe.point.total_energy:.4f} MeV")
    return fe


def find_saddle_point(grid: dict, start: CriticalPoint, end: CriticalPoint,
                      saddle_type: CriticalPointType, name: str) -> CriticalPoint:
    """
    Find saddle point between two minima using flooding algorithm (Union-Find).

    The algorithm:
    1. Sort all grid points by energy
    2. Process points in ascending energy order, connecting them to neighbors
    3. The point at which start and end basins connect is the saddle
    """
    saddle = CriticalPoint(saddle_type, name)

    if not start.found or not end.found:
        print(f"    ✗ {name}: Cannot find (invalid start/end points)")
        return saddle

    print(f"    Searching for {name} between c={start.point.c:.4f} and c={end.point.c:.4f}")

    unique_c = grid['unique_c']
    unique_a4 = grid['unique_a4']
    energy_grid = grid['energy_grid']
    c_to_idx = grid['c_to_idx']
    a4_to_idx = grid['a4_to_idx']
    nc, na4 = grid['nc'], grid['na4']

    # Get indices of start and end points
    start_ic = np.argmin(np.abs(unique_c - start.point.c))
    start_ia4 = np.argmin(np.abs(unique_a4 - start.point.a4))
    end_ic = np.argmin(np.abs(unique_c - end.point.c))
    end_ia4 = np.argmin(np.abs(unique_a4 - end.point.a4))

    # Create list of all valid points sorted by energy
    points = []
    for ic in range(nc):
        for ia4 in range(na4):
            e = energy_grid[ic, ia4]
            if not np.isnan(e):
                points.append((e, ic, ia4))
    points.sort(key=lambda x: x[0])

    # Initialize Union-Find
    def to_1d(ic, ia4):
        return ic * na4 + ia4

    dsu = DisjointSetUnion(nc * na4)
    processed = np.zeros((nc, na4), dtype=bool)

    start_id = to_1d(start_ic, start_ia4)
    end_id = to_1d(end_ic, end_ia4)

    # Flood fill
    for e, ic, ia4 in points:
        current_id = to_1d(ic, ia4)
        processed[ic, ia4] = True

        # Connect to already-processed neighbors (4-connected)
        for dic, dia4 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nic, nia4 = ic + dic, ia4 + dia4
            if 0 <= nic < nc and 0 <= nia4 < na4 and processed[nic, nia4]:
                neighbor_id = to_1d(nic, nia4)
                dsu.union(current_id, neighbor_id)

        # Check if start and end are now connected
        if dsu.find(start_id) == dsu.find(end_id):
            c_val = unique_c[ic]
            a4_val = unique_a4[ia4]
            saddle.point = GridPoint(
                c=c_val, a4=a4_val,
                a3=grid['a3_grid'][ic, ia4],
                a5=grid['a5_grid'][ic, ia4],
                a6=grid['a6_grid'][ic, ia4],
                a7=grid['a7_grid'][ic, ia4],
                a8=grid['a8_grid'][ic, ia4],
                total_energy=e,
                mass_excess=grid['mass_excess_grid'][ic, ia4],
                macro_energy=grid['macro_energy_grid'][ic, ia4],
                micro_energy=grid['micro_energy_grid'][ic, ia4],
                surface_energy=grid['surface_energy_grid'][ic, ia4],
                coulomb_energy=grid['coulomb_energy_grid'][ic, ia4],
                valid=True
            )
            saddle.found = True
            print(f"    ✓ {name}: c={c_val:.4f}, a3={saddle.point.a3:.4f}, "
                  f"a4={a4_val:.4f}, E={e:.4f} MeV")
            return saddle

    print(f"    ✗ {name}: Not found (basins did not connect)")
    return saddle


def run_critical_point_analysis(c: np.ndarray, a4: np.ndarray, energy: np.ndarray,
                                a3: np.ndarray, a5: np.ndarray, a6: np.ndarray,
                                a7: np.ndarray, a8: np.ndarray,
                                mass_excess: np.ndarray = None, macro_energy: np.ndarray = None,
                                micro_energy: np.ndarray = None, surface_energy: np.ndarray = None,
                                coulomb_energy: np.ndarray = None) -> dict[CriticalPointType, CriticalPoint]:
    """
    Run the full critical point analysis pipeline.

    Returns dictionary mapping CriticalPointType to CriticalPoint.
    """
    print("\n  --- Critical Point Analysis ---")

    # Build 2D grid with energy components
    grid = build_2d_energy_grid(c, a4, energy, a3, a5, a6, a7, a8,
                                mass_excess, macro_energy, micro_energy,
                                surface_energy, coulomb_energy)

    # Step 1: Ground State
    print("\n  Step 1: Finding Ground State")
    gs = find_ground_state(grid, c, a4, energy, a3, a5, a6, a7, a8,
                           mass_excess, macro_energy, micro_energy,
                           surface_energy, coulomb_energy)

    # Step 2: Secondary Minimum
    print("\n  Step 2: Finding Secondary Minimum")
    sm = find_secondary_minimum(grid, gs, c, a4, energy, a3, a5, a6, a7, a8)

    # Step 3: Fission Exit
    print("\n  Step 3: Finding Fission Exit")
    fe = find_fission_exit(grid, sm)

    # Step 4: First (Inner) Saddle
    print("\n  Step 4: Finding Inner Saddle")
    s1 = find_saddle_point(grid, gs, sm, CriticalPointType.FIRST_SADDLE, "Inner Saddle")

    # Step 5: Second (Outer) Saddle
    print("\n  Step 5: Finding Outer Saddle")
    s2 = find_saddle_point(grid, sm, fe, CriticalPointType.SECOND_SADDLE, "Outer Saddle")

    return {
        CriticalPointType.GROUND_STATE: gs,
        CriticalPointType.SECONDARY_MINIMUM: sm,
        CriticalPointType.FISSION_EXIT: fe,
        CriticalPointType.FIRST_SADDLE: s1,
        CriticalPointType.SECOND_SADDLE: s2,
    }


# =============================================================================
# 5D CRITICAL POINT ANALYSIS
# =============================================================================

def is_local_minimum_5d(grid: Grid5D, key: tuple) -> bool:
    if key not in grid.energy_map:
        return False

    center_energy = grid.energy_map[key]

    # nan energy points are NOT local minima
    if np.isnan(center_energy):
        return False

    neighbors = grid.get_neighbors_5d(*key)

    if not neighbors:
        return False

    has_valid_neighbor = False
    for neighbor_key in neighbors:
        neighbor_energy = grid.energy_map[neighbor_key]
        if np.isnan(neighbor_energy):
            continue  # Skip nan neighbors
        has_valid_neighbor = True
        if neighbor_energy < center_energy:
            return False

    return has_valid_neighbor


def _key_to_gridpoint(grid: Grid5D, key: tuple) -> GridPoint:
    """Convert a 5D grid key to a GridPoint object."""
    ic, ia3, ia4, ia5, ia6 = key
    a7, a8 = grid.deform_map.get(key, (0.0, 0.0))
    mass_excess, macro_energy, micro_energy, surface_energy, coulomb_energy = grid.energy_components_map.get(
        key, (np.nan, np.nan, np.nan, np.nan, np.nan)
    )
    return GridPoint(
        c=grid.unique_c[ic],
        a3=grid.unique_a3[ia3],
        a4=grid.unique_a4[ia4],
        a5=grid.unique_a5[ia5],
        a6=grid.unique_a6[ia6],
        a7=a7, a8=a8,
        c_idx=ic, a4_idx=ia4,
        total_energy=grid.energy_map[key],
        mass_excess=mass_excess,
        macro_energy=macro_energy,
        micro_energy=micro_energy,
        surface_energy=surface_energy,
        coulomb_energy=coulomb_energy,
        valid=True
    )


def find_ground_state_5d(grid: Grid5D) -> CriticalPoint:
    """Find the ground state in 5D: lowest energy point below c threshold.

    Args:
        grid: Grid5D object

    Returns:
        CriticalPoint for ground state
    """
    gs = CriticalPoint(CriticalPointType.GROUND_STATE, "Ground State")

    if not grid.energy_map:
        print("    Warning: No valid points in 5D grid!")
        return gs

    # Find global minimum
    min_key = min(grid.energy_map.keys(), key=lambda k: grid.energy_map[k])
    global_min_energy = grid.energy_map[min_key]
    global_min_c = grid.unique_c[min_key[0]]

    print(f"    Global minimum: c={global_min_c:.4f}, E={global_min_energy:.4f} MeV")

    # Check if global minimum satisfies threshold
    if global_min_c <= GROUND_STATE_C_THRESHOLD:
        print(f"    Global minimum satisfies c <= {GROUND_STATE_C_THRESHOLD}, using as ground state")
        key = min_key
    else:
        print(f"    Global minimum has c > {GROUND_STATE_C_THRESHOLD}, "
              f"searching for lowest energy with c <= {GROUND_STATE_C_THRESHOLD}")
        # Find lowest energy with c <= threshold
        c_threshold_idx = np.searchsorted(grid.unique_c, GROUND_STATE_C_THRESHOLD, side='right')
        valid_keys = [k for k in grid.energy_map.keys() if k[0] < c_threshold_idx]

        if not valid_keys:
            print("    Warning: No valid points below c threshold!")
            return gs

        key = min(valid_keys, key=lambda k: grid.energy_map[k])

    gs.point = _key_to_gridpoint(grid, key)
    gs.found = True
    print(f"    ✓ Ground State: c={gs.point.c:.4f}, a3={gs.point.a3:.4f}, "
          f"a4={gs.point.a4:.4f}, a5={gs.point.a5:.4f}, a6={gs.point.a6:.4f}, "
          f"E={gs.point.total_energy:.4f} MeV")

    return gs


def find_secondary_minimum_5d(grid: Grid5D, gs: CriticalPoint) -> CriticalPoint:
    """Find the secondary minimum in 5D: a local minimum beyond ground state.

    Args:
        grid: Grid5D object
        gs: Ground state CriticalPoint

    Returns:
        CriticalPoint for secondary minimum
    """
    sm = CriticalPoint(CriticalPointType.SECONDARY_MINIMUM, "Secondary Minimum")

    if not gs.found:
        print("    Cannot find secondary minimum without ground state")
        return sm

    c_threshold = gs.point.c + 0.1
    a4_threshold = gs.point.a4 + 0.1

    # Find indices for thresholds
    gs_c_idx = np.searchsorted(grid.unique_c, gs.point.c, side='right') - 1
    c_threshold_idx = np.searchsorted(grid.unique_c, c_threshold, side='right')
    a4_threshold_idx = np.searchsorted(grid.unique_a4, a4_threshold, side='right')

    print(f"    Searching for secondary minimum with (c > {c_threshold:.4f} or a4 > {a4_threshold:.4f}) and c > {gs.point.c:.4f}")

    candidates = []

    for key in grid.energy_map.keys():
        ic, ia3, ia4, ia5, ia6 = key

        # Always require c > gs.c (physical requirement for fission isomer)
        if ic <= gs_c_idx:
            continue
        # Also skip if too close to ground state in both c and a4
        if ic < c_threshold_idx and ia4 < a4_threshold_idx:
            continue

        # Check if it's a local minimum in 5D
        if not is_local_minimum_5d(grid, key):
            continue

        c_val = grid.unique_c[ic]
        a3_val = grid.unique_a3[ia3]
        a4_val = grid.unique_a4[ia4]

        # Apply additional thresholds (same as 2D version)
        if (c_val > SECONDARY_MINIMUM_C_MAX_THRESHOLD or
                a3_val > SECONDARY_MINIMUM_A3_MAX_THRESHOLD or a4_val > SECONDARY_MINIMUM_A4_MAX_THRESHOLD):
            continue

        candidates.append(key)

    if candidates:
        print(f"    Found {len(candidates)} candidate(s):")
        for key in sorted(candidates, key=lambda k: grid.energy_map[k]):
            ic, ia3, ia4, ia5, ia6 = key
            print(f"      c={grid.unique_c[ic]:.4f}, a3={grid.unique_a3[ia3]:.4f}, "
                  f"a4={grid.unique_a4[ia4]:.4f}, a5={grid.unique_a5[ia5]:.4f}, "
                  f"a6={grid.unique_a6[ia6]:.4f}, E={grid.energy_map[key]:.4f} MeV")

    if not candidates:
        print("    ✗ Secondary Minimum: Not found")
        return sm

    # Select lowest energy candidate
    best_key = min(candidates, key=lambda k: grid.energy_map[k])
    sm.point = _key_to_gridpoint(grid, best_key)
    sm.found = True
    print(f"    ✓ Secondary Minimum: c={sm.point.c:.4f}, a3={sm.point.a3:.4f}, "
          f"a4={sm.point.a4:.4f}, a5={sm.point.a5:.4f}, a6={sm.point.a6:.4f}, "
          f"E={sm.point.total_energy:.4f} MeV")

    return sm


def find_fission_exit_5d(grid: Grid5D, sm: CriticalPoint) -> CriticalPoint:
    """Find the fission exit in 5D: global minimum of total_energy beyond the
    secondary minimum.

    For a ground-state-to-scission grid, the lowest-energy point past the
    secondary minimum lies on the descent after the outer barrier and serves
    as the endpoint between which the outer saddle is sought.

    Args:
        grid: Grid5D object
        sm: Secondary minimum CriticalPoint

    Returns:
        CriticalPoint for fission exit
    """
    fe = CriticalPoint(CriticalPointType.FISSION_EXIT, "Fission Exit")

    if not sm.found:
        print("    Cannot find fission exit without secondary minimum")
        return fe

    c_threshold = sm.point.c + 0.05
    c_threshold_idx = np.searchsorted(grid.unique_c, c_threshold, side='right')

    print(f"    Searching for fission exit as global minimum with c > {c_threshold:.4f}")

    best_key = None
    best_energy = np.inf
    for key, energy in grid.energy_map.items():
        if key[0] < c_threshold_idx:
            continue
        if np.isnan(energy):
            continue
        if energy < best_energy:
            best_energy = energy
            best_key = key

    if best_key is None:
        print("    ✗ Fission Exit: Not found (no valid points past secondary minimum)")
        return fe

    fe.point = _key_to_gridpoint(grid, best_key)
    fe.found = True
    print(f"    ✓ Fission Exit (global min): c={fe.point.c:.4f}, "
          f"a4={fe.point.a4:.4f}, E={fe.point.total_energy:.4f} MeV")
    return fe


def find_saddle_point_5d(grid: Grid5D, start: CriticalPoint, end: CriticalPoint,
                         saddle_type: CriticalPointType, name: str) -> CriticalPoint:
    """Find saddle point between two minima in 5D via the Rust-backed
    imaginary-water-flow watershed.

    Delegates to ``pes_analyzer.saddle.find_iwf_grid`` operating on
    ``grid.energies`` (dense float64 ndarray). See ``docs/SPEC.md`` for the
    algorithm and ``docs/superpowers/specs/2026-05-13-iwf-substitution-design.md``
    for the substitution rationale.

    Args:
        grid: Grid5D object (must have ``energies`` populated).
        start: Starting minimum (e.g., ground state).
        end: Ending minimum (e.g., secondary minimum).
        saddle_type: Type of saddle point.
        name: Name for logging.

    Returns:
        CriticalPoint for the saddle.
    """
    saddle = CriticalPoint(saddle_type, name)

    if not start.found or not end.found:
        print(f"    ✗ {name}: Cannot find (invalid start/end points)")
        return saddle

    print(f"    Searching for {name} between c={start.point.c:.4f} and "
          f"c={end.point.c:.4f} (5D)")

    start_key = (
        int(np.argmin(np.abs(grid.unique_c  - start.point.c))),
        int(np.argmin(np.abs(grid.unique_a3 - start.point.a3))),
        int(np.argmin(np.abs(grid.unique_a4 - start.point.a4))),
        int(np.argmin(np.abs(grid.unique_a5 - start.point.a5))),
        int(np.argmin(np.abs(grid.unique_a6 - start.point.a6))),
    )
    end_key = (
        int(np.argmin(np.abs(grid.unique_c  - end.point.c))),
        int(np.argmin(np.abs(grid.unique_a3 - end.point.a3))),
        int(np.argmin(np.abs(grid.unique_a4 - end.point.a4))),
        int(np.argmin(np.abs(grid.unique_a5 - end.point.a5))),
        int(np.argmin(np.abs(grid.unique_a6 - end.point.a6))),
    )

    if np.isnan(grid.energies[start_key]) or np.isnan(grid.energies[end_key]):
        print(f"    ✗ {name}: Start or end snapped to a NaN cell")
        return saddle

    result = find_iwf_grid(grid.energies, start_key, end_key)
    if result is None:
        print(f"    ✗ {name}: Not found (basins did not connect)")
        return saddle

    saddle_idx, saddle_energy = result
    saddle.point = _key_to_gridpoint(grid, saddle_idx)
    saddle.found = True
    print(f"    ✓ {name}: c={saddle.point.c:.4f}, a3={saddle.point.a3:.4f}, "
          f"a4={saddle.point.a4:.4f}, a5={saddle.point.a5:.4f}, "
          f"a6={saddle.point.a6:.4f}, E={saddle_energy:.4f} MeV")
    return saddle


def run_critical_point_analysis_5d(df: pl.DataFrame) -> dict[CriticalPointType, CriticalPoint]:
    """Run the full critical point analysis pipeline on 5D grid.

    This performs saddle point detection on the full 5D (c, a3, a4, a5, a6) space
    instead of the minimized 2D (c, a4) space, finding true transition states.

    Args:
        df: Polars DataFrame with all deformation parameters and energy

    Returns:
        Dictionary mapping CriticalPointType to CriticalPoint
    """
    print("\n  --- Critical Point Analysis (5D) ---")
    t_total_start = time.perf_counter()

    # Step 0: Build 5D grid
    print("\n  Step 0: Building 5D energy grid")
    t0 = time.perf_counter()
    grid = build_5d_energy_grid(df)
    print(f"  [time] Step 0 (build grid): {time.perf_counter() - t0:.2f} s")

    # Step 1: Ground State
    print("\n  Step 1: Finding Ground State (5D)")
    t0 = time.perf_counter()
    gs = find_ground_state_5d(grid)
    print(f"  [time] Step 1 (ground state): {time.perf_counter() - t0:.2f} s")

    # Step 2: Secondary Minimum
    print("\n  Step 2: Finding Secondary Minimum (5D)")
    t0 = time.perf_counter()
    sm = find_secondary_minimum_5d(grid, gs)
    print(f"  [time] Step 2 (secondary minimum): {time.perf_counter() - t0:.2f} s")

    # Step 3: Fission Exit
    print("\n  Step 3: Finding Fission Exit (5D)")
    t0 = time.perf_counter()
    fe = find_fission_exit_5d(grid, sm)
    print(f"  [time] Step 3 (fission exit): {time.perf_counter() - t0:.2f} s")

    # Step 4: First (Inner) Saddle
    print("\n  Step 4: Finding Inner Saddle (5D)")
    t0 = time.perf_counter()
    s1 = find_saddle_point_5d(grid, gs, sm, CriticalPointType.FIRST_SADDLE, "Inner Saddle")
    print(f"  [time] Step 4 (inner saddle): {time.perf_counter() - t0:.2f} s")

    # Step 5: Second (Outer) Saddle
    print("\n  Step 5: Finding Outer Saddle (5D)")
    t0 = time.perf_counter()
    s2 = find_saddle_point_5d(grid, sm, fe, CriticalPointType.SECOND_SADDLE, "Outer Saddle")
    print(f"  [time] Step 5 (outer saddle): {time.perf_counter() - t0:.2f} s")

    print(f"\n  [time] Total critical point analysis (5D): {time.perf_counter() - t_total_start:.2f} s")

    return {
        CriticalPointType.GROUND_STATE: gs,
        CriticalPointType.SECONDARY_MINIMUM: sm,
        CriticalPointType.FISSION_EXIT: fe,
        CriticalPointType.FIRST_SADDLE: s1,
        CriticalPointType.SECOND_SADDLE: s2,
    }


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_critical_points_csv(critical_points: dict[CriticalPointType, CriticalPoint],
                             nucleus: NucleusInfo, output_dir: str = 'critical_points') -> Path:
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
        print("    No critical points to save")
        return None

    df = pl.DataFrame(rows)
    output_file = output_path / f'{nucleus.isotope_label}_critical_points.csv'
    df.write_csv(output_file, float_precision=6)
    print(f"    Saved critical points to: {output_file}")

    return output_file


def save_minimized_data(c, a4, energy, a3, a5, a6, a7, a8,
                        output_name: str, output_dir: str = 'minimized_data_FoS',
                        y_param: str = 'a4') -> Path:
    """Save the minimized data to a CSV file.

    Args:
        y_param: 'a4' for c-vs-a4 surface (minimized over a3,a5,a6,a7,a8),
                 'a3' for c-vs-a3 surface (minimized over a4,a5,a6,a7,a8).
                 When 'a3', the a4/a3 arguments are swapped: a4 holds the
                 minimized-over parameter values and a3 holds the y-axis values.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if y_param == 'a4':
        df = pl.DataFrame({
            'c': c, 'a4': a4, 'a3_min': a3, 'a5_min': a5,
            'a6_min': a6, 'a7_min': a7, 'a8_min': a8, 'total_energy_min': energy
        })
        df = df.sort(['c', 'a4'])
        suffix = '_minimized.csv'
    else:
        # a3 surface: c and a3 are axes; a4 is minimized over
        df = pl.DataFrame({
            'c': c, 'a3': a3, 'a4_min': a4, 'a5_min': a5,
            'a6_min': a6, 'a7_min': a7, 'a8_min': a8, 'total_energy_min': energy
        })
        df = df.sort(['c', 'a3'])
        suffix = '_minimized_ca3.csv'

    output_file = output_path / f'{output_name}{suffix}'
    df.write_csv(output_file, float_precision=6)
    print(f"    Saved minimized data: {output_file}")

    return output_file


def print_analysis_summary(critical_points: dict[CriticalPointType, CriticalPoint],
                           nucleus: NucleusInfo):
    """Print a formatted summary of the analysis results."""
    print("\n" + "=" * 90)
    print(f"  CRITICAL POINT ANALYSIS RESULTS - {nucleus.isotope_label}")
    print("=" * 90)
    print(f"  {'Point Type':<20} {'c':>8} {'a3':>8} {'a4':>8} {'a5':>8} {'a6':>8} "
          f"{'a7':>8} {'a8':>8} {'E (MeV)':>12}")
    print("-" * 90)

    gs = critical_points.get(CriticalPointType.GROUND_STATE)
    gs_energy = gs.point.total_energy if gs and gs.found else 0.0

    for cp_type, cp in critical_points.items():
        if cp.found:
            rel_e = cp.point.total_energy - gs_energy if gs and gs.found else 0.0
            energy_str = f"{cp.point.total_energy:.4f} (+{rel_e:.4f})" if cp_type != CriticalPointType.GROUND_STATE else f"{cp.point.total_energy:.4f}"
            print(f"  {cp.name:<20} {cp.point.c:>8.4f} {cp.point.a3:>8.4f} "
                  f"{cp.point.a4:>8.4f} {cp.point.a5:>8.4f} {cp.point.a6:>8.4f} "
                  f"{cp.point.a7:>8.4f} {cp.point.a8:>8.4f} {energy_str:>16}")
        else:
            print(f"  {cp.name:<20} Not found")

    print("=" * 90)


# =============================================================================
# PLOTTING
# =============================================================================

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
                        y_param: str = 'a4'):
    """Create a c vs y contour plot with critical points marked.

    Args:
        y_param: 'a4' or 'a3' — controls y-axis label, tick spacing,
                 and which coordinate to read from critical points.
    """
    if vmin is None:
        vmin = energy.min()
    if vmax is None:
        vmax = energy.max()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Grid setup
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
    cf = ax.contourf(ci, yi, zi, levels=boundaries[:-1], cmap=cmap, norm=norm, extend='max')
    contour_levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1.0)
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
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.20))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

    x_tick = 0.20
    padded_c_min = np.floor(c.min() / x_tick) * x_tick - 0.0001
    padded_c_max = np.ceil(c.max() / x_tick) * x_tick + 0.0001
    ax.set_xlim(padded_c_min, padded_c_max)

    y_tick = 0.05
    padded_y_min = np.floor(y.min() / y_tick) * y_tick - 0.0001
    padded_y_max = np.ceil(y.max() / y_tick) * y_tick + 0.0001
    ax.set_ylim(padded_y_min, padded_y_max)

    ax.set_aspect(ASPECT_RATIO)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=DPI, bbox_inches='tight', pad_inches=0.025)
    plt.close()

    print(f"  Saved plot: {output_filename}")


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_single_file(parquet_file: Path, output_plot: str = None,
                        save_minimized: bool = True) -> dict:
    """
    Process a single parquet file: analyze critical points and create plot.

    When USE_5D_ANALYSIS is True, critical point detection runs on the full 5D
    (c, a3, a4, a5, a6) space. The 2D minimized surface is still used for plotting.

    Returns dictionary with results for potential batch processing.
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {parquet_file.name}")
    print('=' * 70)

    # Extract nucleus info
    nucleus = extract_nucleus_info(parquet_file)
    print(f"  Nucleus: {nucleus.isotope_label} (Z={nucleus.Z}, N={nucleus.N})")

    # Read data
    df = read_parquet_file(parquet_file)

    # Run critical point analysis (5D or 2D based on configuration)
    if USE_5D_ANALYSIS:
        print(f"\n  Using 5D critical point analysis")
        critical_points = run_critical_point_analysis_5d(df)
    else:
        print(f"\n  Using 2D critical point analysis")
        # For 2D analysis, minimize first then run analysis
        result = minimize_over_deformations(df)
        c, a4, energy, a3, a5, a6, a7, a8, mass_excess, macro_e, micro_e, surf_e, coul_e = result
        critical_points = run_critical_point_analysis(c, a4, energy, a3, a5, a6, a7, a8,
                                                      mass_excess, macro_e, micro_e, surf_e, coul_e)

    # Print summary
    print_analysis_summary(critical_points, nucleus)

    # Save critical points
    save_critical_points_csv(critical_points, nucleus)

    # Minimize over deformations for plotting (always needed for 2D visualization)
    result_ca4 = minimize_over_deformations(df)
    c, a4, energy, a3, a5, a6, a7, a8, mass_excess, macro_e, micro_e, surf_e, coul_e = result_ca4

    # Save minimized data (c vs a4)
    output_name = parquet_file.stem
    if save_minimized:
        save_minimized_data(c, a4, energy, a3, a5, a6, a7, a8, output_name)

    # Create c vs a4 plot (2D contour with critical points projected onto it)
    if output_plot:
        output_filename = output_plot
    else:
        output_filename = f'{output_name}_c_vs_a4.png'

    create_contour_plot(c, a4, energy, nucleus, output_filename, critical_points, y_param='a4')

    # --- c vs a3 surface ---
    result_ca3 = minimize_over_deformations_ca3(df)
    c3, a3_y, energy3, a4_min3, a5_min3, a6_min3, a7_min3, a8_min3, *_ = result_ca3

    if save_minimized:
        save_minimized_data(c3, a4_min3, energy3, a3_y, a5_min3, a6_min3, a7_min3, a8_min3,
                            output_name, y_param='a3')

    output_filename_ca3 = f'{output_name}_c_vs_a3.png'
    create_contour_plot(c3, a3_y, energy3, nucleus, output_filename_ca3,
                        critical_points, y_param='a3')

    # Calculate fission barriers
    barriers = calculate_fission_barriers(critical_points, nucleus)

    return {
        'file': parquet_file,
        'nucleus': nucleus,
        'critical_points': critical_points,
        'barriers': barriers,
        'c': c, 'a4': a4, 'energy': energy,
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
    """Wrapper that captures all stdout from process_single_file into a string.

    On success returns (result_dict, captured_output).
    On failure returns (None, captured_output_including_error).
    """
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        result = process_single_file(parquet_file, output_plot=None,
                                     save_minimized=save_minimized)
    except Exception as e:
        import traceback
        print(f"ERROR processing {parquet_file}: {e}")
        traceback.print_exc()
        result = None
    finally:
        sys.stdout = old_stdout
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
    print(f"  Minimize over: {', '.join(MINIMIZE_OVER)}")
    print(f"  Critical point analysis: {'5D (c, a3, a4, a5, a6)' if USE_5D_ANALYSIS else '2D (c, a4)'}")
    print(f"  Output DPI: {DPI}")
    print(f"  Ground state c threshold: {GROUND_STATE_C_THRESHOLD}")
    print(f"  Secondary minimum c max: {SECONDARY_MINIMUM_C_MAX_THRESHOLD}")
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
        # Multiple files: run in parallel with captured output
        n_workers = min(args.workers, len(files_to_process))
        print(f"\nProcessing {len(files_to_process)} files with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_file = {
                executor.submit(
                    _process_single_file_captured, fp,
                    save_minimized=not args.no_save_minimized,
                ): fp
                for fp in files_to_process
            }
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
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
