//! This is a crate to implement Stein Variational Gradient Descent
//! By using probability distribution and kernel function, we can optimize the distribution

use crate::kernel::Kernel;
use crate::prob_dist::{ProbDist, Sample};
use crate::util::{DispInterval, LoopParam};
use candle_core::{Result, Tensor};
use candle_nn::optim::Optimizer;

#[derive(Debug)]
pub struct SVGD<S, T>
where
    S: ProbDist,
    T: Kernel,
{
    /// probability distribution of p(x|w), where w should be learnable parameter.
    pxw: S,

    /// probability distribution of p(w), where w should be learnable parameter.
    pw: S,

    /// kernel function
    kernel: T,
}

impl<S, T> SVGD<S, T>
where
    S: ProbDist,
    T: Kernel,
{
    /// create a new SVGD model
    /// post_param: posterior samples, thus this tensor is differentiated
    /// pxw: probability distribution of p(x|w), where w should be learnable parameter.
    /// pw: probability distribution of p(w), where w should be learnable parameter.
    /// kernel: kernel function, where the first argument should be learnable parameter.
    pub fn new(pxw: S, pw: S, kernel: T) -> Result<Self> {
        Ok(Self { pxw, pw, kernel })
    }

    /// calculate derivative of KL(q_e||p) with e at e=0
    /// where q_e(z) is pushforwarded probability density function
    pub fn forward(&self, sample: &Sample, post_param: &Tensor) -> Result<Tensor> {
        let detach_post_param = post_param.detach();
        let null_sample = Sample::new(None, None)?;

        let kernel_post_detach = self.kernel.kernel(&detach_post_param, &detach_post_param)?;
        let kernel_post = self.kernel.kernel(post_param, &detach_post_param)?;

        let logpxw = self.pxw.log_p(sample, post_param)?;
        let logpw = self.pw.log_p(&null_sample, post_param)?;

        // calculate loss before derivative
        let phiw = kernel_post_detach
            .t()?
            .matmul(&(logpxw + &logpw)?.broadcast_as((1, logpw.dim(0)?))?.t()?)?
            .squeeze(1)?
            + kernel_post.sum((1,))?;
        phiw
    }

    /// Optimize parameter distribution by SVGD
    pub fn optimize(
        &self,
        sample: &Sample,
        post_param: &Tensor,
        optimizer: &mut impl Optimizer,
        //optimizer: &mut T,
        loop_param: LoopParam,
    ) -> Result<Tensor> {
        for ite in 0..loop_param.max_iter {
            let loss = (-1. * self.forward(&sample, &post_param)?)?;
            let _ = optimizer.backward_step(&loss)?;
            if let DispInterval::DISP(disp_interval) = loop_param.disp_interval {
                if ite % disp_interval == 0 {
                    println!("ite = {}, loss = {}", ite, &loss);
                }
            }
        }
        Ok(post_param.clone())
    }
}
