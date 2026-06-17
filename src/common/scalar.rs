//! Float element types accepted at the numpy boundary. Bundles the numeric
//! traits the kernels need plus NaN-safe total ordering (preserved for f32).
use std::cmp::Ordering;

pub trait Scalar:
    numpy::Element + ndarray::NdFloat + Send + Sync + Copy + 'static
{
    fn nan() -> Self;
    fn neg_inf() -> Self;
    fn tcmp(&self, other: &Self) -> Ordering;
    fn to_f64(self) -> f64;
}

impl Scalar for f64 {
    #[inline] fn nan() -> Self { f64::NAN }
    #[inline] fn neg_inf() -> Self { f64::NEG_INFINITY }
    #[inline] fn tcmp(&self, o: &Self) -> Ordering { self.total_cmp(o) }
    #[inline] fn to_f64(self) -> f64 { self }
}
impl Scalar for f32 {
    #[inline] fn nan() -> Self { f32::NAN }
    #[inline] fn neg_inf() -> Self { f32::NEG_INFINITY }
    #[inline] fn tcmp(&self, o: &Self) -> Ordering { self.total_cmp(o) }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
}
