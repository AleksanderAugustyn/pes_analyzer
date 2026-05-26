//! Full watershed segmentation: flood the grid to completion, labeling
//! every non-NaN cell with its initial basin and recording every union
//! that merges two distinct basins as a `(saddle, deeper, shallower)`
//! tuple.

use ndarray::ArrayViewD;

/// Result of a full watershed flood. See `API.md` for the semantics.
pub struct SegmentationResult {
    /// One i32 per cell of the input buffer (row-major); -1 at NaN cells,
    /// otherwise the basin ID the cell first joined during the flood.
    pub labels: Vec<i32>,
    /// One entry per basin, sorted ascending by minimum energy.
    /// `(min_nd_index, min_energy)`.
    pub basins: Vec<(Vec<usize>, f64)>,
    /// One entry per merge event, sorted ascending by saddle energy.
    /// `(saddle_nd_index, saddle_energy, deeper_basin_id, shallower_basin_id)`.
    pub merges: Vec<(Vec<usize>, f64, u32, u32)>,
}

/// Full watershed flood with merge-tree recording. Empty stub for now.
pub fn watershed_segmentation_inner(
    energies: ArrayViewD<'_, f64>,
) -> SegmentationResult {
    let n_total = energies.len();
    SegmentationResult {
        labels: vec![-1; n_total],
        basins: Vec::new(),
        merges: Vec::new(),
    }
}
