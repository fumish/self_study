//! This is a crate to express probability distribution
//! In SVGD, the logarithm of the probability distribution is necessary
use candle_core::shape::Dim;
use candle_core::{Result, Tensor};
use std::f64::consts::PI;
use std::fmt;

/// This is a struct to represent a sample
#[derive(Debug)]
pub struct Sample {
    x: Option<Tensor>,
    y: Option<Tensor>,
}

impl Sample {
    pub fn new(x: Option<Tensor>, y: Option<Tensor>) -> Result<Self> {
        Ok(Self { x, y })
    }

    /// get the dimension of x at dim_ind
    /// Error is returned when x is None
    pub fn x_dim<D: Dim>(&self, dim: D) -> Result<usize> {
        match &self.x {
            Some(x) => Ok(x.dim(dim)?),
            None => Err(candle_core::Error::EmptyTensor {
                op: ("field x has no value"),
            }),
        }
    }

    /// calculate the order th moment of x
    pub fn moment_x(&self, order: f64) -> Result<Tensor> {
        match &self.x {
            Some(x) => x.powf(order)?.mean((0,)),
            None => Err(candle_core::Error::EmptyTensor {
                op: ("field x has no value"),
            }),
        }
    }

    pub fn moment_y(&self, order: f64) -> Result<Tensor> {
        match &self.y {
            Some(y) => y.powf(order)?.mean((0,)),
            None => Err(candle_core::Error::EmptyTensor {
                op: ("field x has no value"),
            }),
        }
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x = {:?}, y = {:?}", self.x, self.y)
    }
}

/// This is a trait to express probability distribution for SVGD
/// log_p: calculate log probability for each n sample, therefore, the output should be (n,) tensor
/// sample: sample from the distribution
/// In this idea, the struct that implements this trait does not manage the parameters,
/// therefore, the struct has no learnable parameters and its fields are only hyperparameters.
pub trait ProbDist {
    /// calculate log probability
    /// params are assumed to learnable parameters, so the impl struct does not manage the parameters
    fn log_p(&self, sample: &Sample, params: &Tensor) -> Result<Tensor>;

    /// sample from the distribution
    /// params are assumed to learnable parameters, so the impl struct does not manage the parameters
    fn sample(&self, n: usize, params: &Tensor) -> Result<Tensor>;
}

/// multivariate normal distribution, with diagonal covariance matrix
/// std is just a hyperparameter of this struct.
/// and the mean is given by the upper layer when log_p is called.
#[derive(Debug, Clone)]
pub struct NormalMean {
    /// standard deviation of multivariate normal distribution
    /// d dimensional vector
    std: Tensor,
}
impl NormalMean {
    pub fn new(std: Tensor) -> Result<Self> {
        Ok(Self { std })
    }

    pub fn get_dim<D: Dim>(&self, dim: D) -> Result<usize> {
        self.std.dim(dim)
    }
}
impl ProbDist for NormalMean {
    /// x is assumed to be (n, d) tensor
    fn log_p(&self, sample: &Sample, params: &Tensor) -> Result<Tensor> {
        // sample.x is assumed to be (n, l, d) tensor, and l should be kept
        let x = match &sample.x {
            Some(x) => x,
            None => &params
                .zeros_like()?
                .broadcast_as((1, params.dim(0)?, params.dim(1)?))?,
        };

        let diff = x.broadcast_sub(params)?.broadcast_div(&self.std)?.sqr()?;
        let val = (-0.5 * (2. * PI * self.std.sqr()?)?.log()?)?;
        (-0.5 * diff)?.broadcast_add(&val)?.sum((0, 2))
    }

    fn sample(&self, n: usize, params: &Tensor) -> Result<Tensor> {
        let d = params.dim(0)?;
        let noise = Tensor::randn(0., 1., (n, d), params.device())?;
        noise.broadcast_mul(&self.std)?.broadcast_add(&params)
        //&self.mean + &self.std * noise
    }
}

impl fmt::Display for NormalMean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "std = {}", self.std,)
    }
}
