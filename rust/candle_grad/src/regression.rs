use candle_core::{Device, Result, Tensor};
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::{Linear, VarBuilder};
use std::fmt;

/// Trait for regression model
/// forward: calculate output from input
/// create_data: create random data for training
/// log_p_param: calculate value of log prior
pub trait ReressionModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn create_date(&self, n: usize, device: &Device) -> Result<(Tensor, Tensor)>;
    fn log_p_param(&self) -> Result<Tensor>;
}

/// Linear regression model
/// candle_nn::Linear has coef of input and intercept, so we utilize it for linear regression.
#[derive(Debug)]
pub struct LinearRegParams {
    params: Linear,
    lambda: Option<f64>,
}

impl LinearRegParams {
    pub fn new(vs: &VarBuilder, input_dim: usize, lambda: Option<f64>) -> Result<Self> {
        let params = candle_nn::linear(input_dim, 1, vs.pp("params"))?;
        Ok(Self { params, lambda })
    }

    pub fn from_params(params: Linear, lambda: Option<f64>) -> Result<Self> {
        Ok(Self { params, lambda })
    }
}

impl ReressionModel for LinearRegParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let wx = x.matmul(&self.params.weight().t()?)?;
        match self.params.bias() {
            Some(b) => wx.broadcast_add(b),
            None => Ok(wx),
        }
    }

    fn create_date(&self, n: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let feature_dim = self.params.weight().dim(0)?;

        let x = Tensor::randn(0., 1., (n, feature_dim), device)?;
        let wx = x.matmul(self.params.weight())?;
        let y = match self.params.bias() {
            Some(b) => wx.broadcast_add(b),
            None => Ok(wx),
        };
        let y = (y + Tensor::randn(0., 0.5, (n, 1), device)?)?;
        Ok((x, y))
    }

    /// calculate value of log prior
    fn log_p_param(&self) -> Result<Tensor> {
        let reg = self.params.weight().sqr()?.sum_all()?;
        let reg = match self.params.bias() {
            Some(b) => (reg + b.sqr()?.sum_all()?)?,
            _ => reg,
        };
        -self.lambda.unwrap_or(0.) * reg
    }
}

impl fmt::Display for LinearRegParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "weight = {}, bias = {}, lambda = {:?}",
            self.params.weight(),
            self.params.bias().unwrap(),
            self.lambda,
        )
    }
}

/// Parameter of Reduced Rank Regression
/// y = BA x + bias, where y in R^{n, N}, x in R^{n, M}, A in R^{H, M}, B in R^{N, H}, bias in R^{N}
/// equivalent with three layer perceptron with linear activation function.
#[derive(Debug)]
pub struct ReducedRankRegParams {
    weight_a: Tensor,
    weight_b: Tensor,
    bias: Option<Tensor>,
    lambda: Option<f64>,
}

impl ReducedRankRegParams {
    pub fn new(
        vs: &VarBuilder,
        dims: (usize, usize, usize),
        biased: bool,
        lambda: Option<f64>,
    ) -> Result<Self> {
        let (output_dim, hidden_dim, input_dim) = dims;
        let weight_a =
            vs.get_with_hints((hidden_dim, input_dim), "weight_a", DEFAULT_KAIMING_NORMAL)?;
        let weight_b =
            vs.get_with_hints((output_dim, hidden_dim), "weight_b", DEFAULT_KAIMING_NORMAL)?;
        let bias = if biased {
            Some(vs.get_with_hints((output_dim), "bias", DEFAULT_KAIMING_NORMAL)?)
        } else {
            None
        };
        Ok(Self {
            weight_a,
            weight_b,
            bias,
            lambda,
        })
    }

    pub fn from_params(
        weight_a: Tensor,
        weight_b: Tensor,
        bias: Option<Tensor>,
        lambda: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            weight_a,
            weight_b,
            bias,
            lambda,
        })
    }
}

impl ReressionModel for ReducedRankRegParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight_a.t()?)?;
        let x = x.matmul(&self.weight_b.t()?)?;
        match self.bias.as_ref() {
            Some(b) => x.broadcast_add(b),
            _ => Ok(x),
        }
    }

    fn create_date(&self, n: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let input_dim = self.weight_a.dim(1)?;
        let out_dim = self.weight_b.dim(0)?;

        let x = Tensor::randn(0., 1., (n, input_dim), device)?;
        let y = self.forward(&x)?;
        let y = (y + Tensor::randn(0., 0.5, (n, out_dim), device)?)?;
        Ok((x, y))
    }

    fn log_p_param(&self) -> Result<Tensor> {
        let reg = (self.weight_a.sqr()?.sum_all()? + self.weight_b.sqr()?.sum_all()?)?;
        let reg = match self.bias.as_ref() {
            Some(b) => (reg + b.sqr()?.sum_all()?)?,
            _ => reg,
        };
        -self.lambda.unwrap_or(0.) * reg
    }
}

impl fmt::Display for ReducedRankRegParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "A = {}, B = {}, bias = {:?}, lambda = {:?}, BA = {}",
            self.weight_a,
            self.weight_b,
            self.bias,
            self.lambda,
            self.weight_b.matmul(&self.weight_a).unwrap(),
        )
    }
}
