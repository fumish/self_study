//! This is a crate to calculate kernel function
use crate::util;
use candle_core::{Result, Tensor};
use std::fmt;

/// This is a trait to calculate a value of kernel function
pub trait Kernel {
    /// calculate kernel function
    fn kernel(&self, x: &Tensor, y: &Tensor) -> Result<Tensor>;
}

/// This enum represents the type of band width, in several kernels
/// Fix: fix the band width
/// Adaptive: adapt the band width based on original paper of SVGD
#[derive(Debug)]
pub enum BandType {
    Fix(f64),
    Adaptive,
}

#[derive(Debug)]
pub struct RBF {
    band: BandType,
}

impl RBF {
    pub fn new(band: BandType) -> Result<Self> {
        Ok(Self { band })
    }
}

impl Kernel for RBF {
    /// assume x and y are (n, d) tensor
    fn kernel(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        // calculate pairwise distance
        let dist = util::xypair_dist(x, y)?;

        let band_width = match self.band {
            BandType::Fix(band) => band,
            BandType::Adaptive => {
                let dist_pair = util::triu::<f64>(&dist)?;
                let median = util::tensor_median::<f64>(&dist_pair)?;
                median * median / (dist_pair.dim(0)? as f64).ln()
            }
        };

        (dist / (-2. * band_width))?.exp()
    }
}

impl fmt::Display for RBF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RBF kernel, band = {:?}", self.band)
    }
}
