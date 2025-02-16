use std::fmt::Display;

use crate::prob_dist::ProbDist;
/// This crate is utility functions for candle_svgd
use candle_core::{DType, Error, IndexOp, Result, Tensor, WithDType};
use candle_nn::{Optimizer, VarMap};
use candle_optimisers as opt;

/// calculate distance between mat_x and mat_y
/// mat_x and mat_y are 2d tensor
/// mat_x: (n, d)
/// mat_y: (m, d)
pub fn xypair_dist(mat_x: &Tensor, mat_y: &Tensor) -> Result<Tensor> {
    let dims = mat_x.dims();
    if mat_x.dims().len() != 2 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 2,
            got: dims.len(),
            shape: mat_x.shape().clone(),
        });
    }

    let dims = mat_y.dims();
    if mat_y.dims().len() != 2 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 2,
            got: dims.len(),
            shape: mat_y.shape().clone(),
        });
    }

    // In order to keep gradient information of mat_2d, pair of mat_2d is obtained by broadcast_sub
    // debug is necessary

    let n_x = mat_x.dim(0)?;

    let broadcast_y = mat_y.detach().repeat((n_x, 1, 1))?; // dim of tensor is (n_x, n_y, d)
    let pair_mat = broadcast_y.transpose(0, 1)?; // dim of tensor is (n_y, n_x, d)
    pair_mat.broadcast_sub(&mat_x)?.sqr()?.sum((2,)) // dim of mat_x is broadcasted to (n_y, n_x, d)
}

/// calculate pairwise distance for each row of 2d tensor
pub fn pairwise_dist(mat_2d: &Tensor) -> Result<Tensor> {
    let dims = mat_2d.dims();
    if dims.len() != 2 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 2,
            got: dims.len(),
            shape: mat_2d.shape().clone(),
        });
    }

    // In order to keep gradient information of mat_2d, pair of mat_2d is obtained by broadcast_sub
    // debug is necessary
    let n_sample = dims[0];
    let broadcasted_mat = mat_2d.detach().repeat((n_sample, 1, 1))?; // x[i,j,:] = x[k,j,:] for all i,k
    let pair_mat = broadcasted_mat.transpose(0, 1)?; // x[i,j,:] goes to x[j,i,:]

    pair_mat.broadcast_sub(&mat_2d)?.sqr()?.sum((2,))
}

/// calculate median of all value of mat
pub fn tensor_median<T>(mat: &Tensor) -> Result<T>
where
    T: WithDType,
{
    let mat_1d = mat.flatten_all()?;

    let n_elems = mat_1d.dim(0)? as usize;
    let mut vec_1d = mat_1d.to_vec1::<T>()?;
    vec_1d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(vec_1d[n_elems / 2])
}

/// extract upper triangular part of 2d tensor by 1d tensor
pub fn triu<T>(mat_2d: &Tensor) -> Result<Tensor>
where
    T: WithDType,
{
    let dims = mat_2d.dims();
    let device = mat_2d.device();
    if dims.len() != 2 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 2,
            got: dims.len(),
            shape: mat_2d.shape().clone(),
        });
    }

    let n_sample = dims[0];
    let mut elems = vec![];
    for row in 0..n_sample {
        for col in (row + 1)..n_sample {
            elems.push(mat_2d.i((row, col))?.to_vec0::<T>()?);
        }
    }

    let n_elems = elems.len();
    Tensor::from_vec(elems, (n_elems,), device)
}

/// This is an enum to represent the interval of loop
/// NO_DISP: no display
/// DISP: display interval by the variants
#[derive(Debug)]
pub enum DispInterval {
    NO_DISP,
    DISP(usize),
}

/// This is a struct to manage parameters for ordinal learning loop
#[derive(Debug)]
pub struct LoopParam {
    /// maximum number of iteration
    pub max_iter: usize,

    /// difference of loss between two iteration
    /// None means tol is not used
    pub tol: Option<f64>,

    /// interval of iteration to print loss, None means no print
    pub disp_interval: DispInterval,
}

impl Display for LoopParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_inter = {}, tol = {}, loop_interval = {}",
            self.max_iter,
            self.tol.unwrap_or(-1.),
            match &self.disp_interval {
                DispInterval::NO_DISP => "NO_DISP".to_string(),
                DispInterval::DISP(interval) => interval.to_string(),
            }
        )
    }
}

impl Default for LoopParam {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: None,
            disp_interval: DispInterval::NO_DISP,
        }
    }
}

pub fn create_default_rmsprop_optimizer(var_map: VarMap) -> Result<opt::rmsprop::RMSprop> {
    let rms_params = opt::rmsprop::ParamsRMSprop {
        lr: 0.01,
        ..Default::default()
    };
    opt::rmsprop::RMSprop::new(var_map.all_vars(), rms_params)
}

pub fn create_default_adam_optimizer(var_map: VarMap) -> Result<opt::adam::Adam> {
    let adam_params = opt::adam::ParamsAdam {
        lr: 0.01,
        ..Default::default()
    };
    opt::adam::Adam::new(var_map.all_vars(), adam_params)
}

pub fn create_default_adamw_optimizer(var_map: VarMap) -> Result<candle_nn::AdamW> {
    let adamw_params = candle_nn::ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    candle_nn::AdamW::new(var_map.all_vars(), adamw_params)
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    1. / (1. + (-1. * x)?.exp()?)?
}

pub fn create_prob_samples<T>(
    true_prob: T,
    true_param: Tensor,
    n_sample: usize,
    n_params: usize,
) -> Result<Tensor>
where
    T: ProbDist,
{
    // data generation
    let gen_sample = true_prob.sample(n_sample, &true_param)?;

    // repeat n_params times
    let mut repeat_size = vec![n_params];
    repeat_size.extend(vec![1; gen_sample.dims().len()]);

    gen_sample.repeat(repeat_size)?.transpose(0, 1)
}
