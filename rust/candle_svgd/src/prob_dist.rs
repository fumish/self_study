//! This is a crate to express probability distribution
//! In SVGD, the logarithm of the probability distribution is necessary
use crate::util::sigmoid;
use candle_core::shape::Dim;
use candle_core::{DType, Result, Tensor};
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

    pub fn clone_x(&self) -> Result<Tensor> {
        match &self.x {
            None => Err(candle_core::Error::EmptyTensor {
                op: ("field x has no value"),
            }),
            Some(x) => Ok(x.clone()),
        }
    }

    pub fn clone_y(&self) -> Result<Tensor> {
        match &self.y {
            None => Err(candle_core::Error::EmptyTensor {
                op: ("field x has no value"),
            }),
            Some(y) => Ok(y.clone()),
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
    /// x is assumed to be (n,l,d) tensor
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

/// distribution of logistic regression
/// p(y=1|x, w) = 1/(1 + exp(-x * w)), where x in R^d, w in R^d
pub struct LogisticRegression {
    /// input x
    /// x is needed to conduct sample function
    /// Therefore, it is unnecessary when only log_p is called, and that case x is None
    x: Option<Tensor>,
}
impl LogisticRegression {
    pub fn new(x: Option<Tensor>) -> Self {
        Self { x }
    }
}
impl ProbDist for LogisticRegression {
    /// log p(y|x,w) = y * log(1/(1 + exp(-x * w))) + (1 - y) * log(1 - 1/(1 + exp(-x * w)))
    /// After check none of x and y, calculate the log probability
    /// Note that x and y should have (n,l,d) tensor, and l should be kept
    /// params is assumed to be (l,d) tensor
    fn log_p(&self, sample: &Sample, params: &Tensor) -> Result<Tensor> {
        match (&sample.x, &sample.y) {
            (Some(x), Some(y)) => {
                //println!("x = {:?}, params = {:?}", x, params);
                let p_x = sigmoid(&x.broadcast_mul(params)?.sum(2)?)?;
                //let p_x = sigmoid(&x.broadcast_matmul(params)?)?;
                let cross_entropy = ((y * p_x.log()?)? + ((1. - y)? * (1. - p_x)?.log()?)?)?.sum(0);
                cross_entropy
            }
            _ => Err(candle_core::Error::EmptyTensor {
                op: ("field x and y should have a value to calculate log probability"),
            }),
        }
    }
    fn sample(&self, n: usize, params: &Tensor) -> Result<Tensor> {
        match &self.x {
            None => {
                return Err(candle_core::Error::EmptyTensor {
                    op: ("field x should have a value to sample"),
                });
            }
            Some(x) => {
                let p_x = sigmoid(&x.broadcast_matmul(params)?)?;
                p_x.ge(&p_x.rand_like(0., 1.)?)?.to_dtype(DType::F64)
            }
        }
    }
}

/// distribution of singular toy model
/// p(y|x,w) = N(y | a tanh(b * x), 1), a \in R, b, x \in R^d
pub struct SingularToyModel {
    /// input x
    /// x is needed to conduct sample function
    /// Therefore, it is unnecessary when only log_p is called, and that case x is None
    x: Option<Tensor>,
}

impl SingularToyModel {
    pub fn new(x: Option<Tensor>) -> Self {
        Self { x }
    }
}
impl ProbDist for SingularToyModel {
    /// log p(y|x,w) = -0.5 * (y - a tanh(x * b))^2
    /// After check none of x and y, calculate the log probability
    /// Note that x and y should have (n,l) tensor, and l should be kept
    /// params is assumed to be (l,2) tensor
    fn log_p(&self, sample: &Sample, params: &Tensor) -> Result<Tensor> {
        match (&sample.x, &sample.y) {
            (Some(x), Some(y)) => {
                // params are assumed to (l,2) tensor
                let params_a = params.t()?.get(0)?;
                let params_b = params.t()?.get(1)?;
                let fx = x
                    .broadcast_mul(&params_a)?
                    .tanh()?
                    .broadcast_mul(&params_b)?;
                ((y - &fx)?.sqr()? * (-0.5))?.sum(0)
            }
            _ => Err(candle_core::Error::EmptyTensor {
                op: ("field x and y should have a value to calculate log probability"),
            }),
        }
    }
    fn sample(&self, n: usize, params: &Tensor) -> Result<Tensor> {
        match &self.x {
            None => {
                return Err(candle_core::Error::EmptyTensor {
                    op: ("field x should have a value to sample"),
                });
            }
            Some(x) => {
                // params is (2) tensor and is picked a first column
                let params_a = params.get(0)?.to_scalar::<f64>()?;
                let params_b = params.get(1)?.to_scalar::<f64>()?;
                let fx = ((x * params_a)?.tanh()? * params_b)?;
                fx + Tensor::randn(0., 1., (n,), params.device())
            }
        }
    }
}
