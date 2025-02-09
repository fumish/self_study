mod kernel;
mod prob_dist;
mod svgd;
mod util;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{init::DEFAULT_KAIMING_NORMAL, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_optimisers::rmsprop::{ParamsRMSprop, RMSprop};
use kernel::{BandType, Kernel, RBF};
use prob_dist::{NormalMean, ProbDist, Sample};
use svgd::SVGD;
use util::{DispInterval, LoopParam};

fn test_dist() -> Result<()> {
    println!("Hello, world!");
    let device = Device::Cpu;

    //let test1 = Tensor::randn(0., 1., (2, 3, 4), &device)?;
    //let test2 = Tensor::randn(0., 1., (2, 4, 5), &device)?;
    const M: usize = 3;
    const N: usize = 4;

    // we assume that test1 can be differential, so we do not change the structure.
    let test1 = Tensor::randn(0., 1., (N, M), &device)?;

    let broadcasted_mat = test1.detach().repeat((N, 1, 1))?; // x[i,j,:] = x[k,j,:] for all i,k
    let pair_mat = broadcasted_mat.transpose(0, 1)?; // x[i,j,:] goes to x[j,i,:]
                                                     // pairwise distance
    let pairwise_dist = pair_mat.broadcast_sub(&test1)?.sqr()?.sum((2,))?;
    //let pairwise_dist = (broadcasted_mat - pair_mat)?.sqr()?.sum((2,))?;

    //let res = test1.matmul(&test2);
    println!("test = {test1}");

    let debug_val = (test1.i((0,))? - test1.i((1,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);

    let debug_val = (test1.i((0,))? - test1.i((2,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);

    let debug_val = (test1.i((1,))? - test1.i((2,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);

    println!("res = {}", pairwise_dist);

    let res_triu = util::triu::<f64>(&pairwise_dist)?;
    println!("res_triu = {res_triu}");

    let res_median = util::tensor_median::<f64>(&res_triu)?;
    println!("res_median = {res_median}");

    let res_median = util::tensor_median::<f64>(&pairwise_dist)?;
    println!("res_median = {res_median}");

    let test1 = Tensor::randn(0., 1., (2, M), &device)?;
    let test2 = Tensor::randn(0., 1., (3, M), &device)?;
    println!("test1 = {test1}, test2 = {test2}");
    let res = util::xypair_dist(&test1, &test2)?;
    println!("res = {res}");

    let debug_val = (test1.i((0,))? - test2.i((0,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);
    let debug_val = (test1.i((1,))? - test2.i((0,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);

    let debug_val = (test1.i((0,))? - test2.i((1,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);
    let debug_val = (test1.i((1,))? - test2.i((1,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);

    let debug_val = (test1.i((0,))? - test2.i((2,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);
    let debug_val = (test1.i((1,))? - test2.i((2,))?)?.sqr()?.sum((0,))?;
    println!("debug = {}", debug_val);

    Ok(())
}

fn kernel_test() -> Result<()> {
    const M: usize = 3;
    let device = Device::Cpu;
    let test_x = Tensor::randn(0., 1., (3, M), &device)?;
    println!("test_x = {test_x}");

    let dist = util::pairwise_dist(&test_x)?;
    let exp_val = (dist / (-2.))?.exp()?;
    println!("exp_val = {exp_val}");

    let rbf = kernel::RBF::new(kernel::BandType::Adaptive)?;
    let res = rbf.kernel(&test_x, &test_x.detach())?;
    println!("res = {res}");

    let rbf = kernel::RBF::new(kernel::BandType::Fix(1.))?;
    let res = rbf.kernel(&test_x, &test_x.detach())?;
    println!("res = {res}");
    Ok(())
}

fn prob_test() -> Result<()> {
    let device = Device::Cpu;
    let mean = Tensor::new(vec![1., 1.], &device)?;
    let std = Tensor::new(vec![2., 2.], &device)?;
    let normal_dist = NormalMean::new(std)?;

    let n_sample = 100;
    let sample_x = normal_dist.sample(n_sample, &mean)?;
    let sample = Sample::new(Some(sample_x), None)?;
    //println!("sample = {sample}");
    //println!("sample_sq = {}", sample.sqr()?.mean((0,))?);

    let init_mean = Tensor::new(vec![0., 0.], &device)?;
    let init_std = Tensor::new(vec![1., 1.], &device)?;
    let init_normal = NormalMean::new(init_std)?;
    let init_logp = init_normal.log_p(&sample, &init_mean)?;
    println!("init_logp = {init_logp}");

    let sample_mean = sample.moment_x(1.)?;
    let sample_std = (sample.moment_x(2.)? - &sample_mean.powf(2.)?)?.sqrt()?;
    println!("sample_mean = {sample_mean}, sample_std = {sample_std}");
    let est_normal = NormalMean::new(sample_std)?;
    let est_logp = est_normal.log_p(&sample, &sample_mean)?;
    println!("est_logp = {est_logp}");

    println!("true_logp = {}", normal_dist.log_p(&sample, &mean)?);
    Ok(())
}

fn create_normal_samples(
    true_normal: NormalMean,
    true_mean: Tensor,
    n_sample: usize,
    n_params: usize,
) -> Result<Sample> {
    // data generation
    let sample_x = true_normal
        .sample(n_sample, &true_mean)?
        .repeat((n_params, 1, 1))?
        .transpose(0, 1)?;

    Sample::new(Some(sample_x), None)
}

fn svgd_normal(device: &Device, sample: &Sample, n_params: usize) -> Result<Tensor> {
    // settings
    let dim = sample.x_dim(candle_core::D::Minus1)?;
    let pri_beta = 1.;
    let band_type = BandType::Adaptive;

    // create a new SVGD model
    let post_std = Tensor::new(vec![1.; dim], &device)?;
    let pxw = NormalMean::new(post_std)?;
    let pri_std = (pri_beta * Tensor::ones((dim,), DType::F64, &device)?)?;
    let pw = NormalMean::new(pri_std)?;
    let kernel = RBF::new(band_type)?;
    let svgd_model = SVGD::new(pxw, pw, kernel)?;

    // create vars
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F64, &device);
    let post_param = vs.get_with_hints(
        (n_params, dim),
        "post_param_of_mean",
        DEFAULT_KAIMING_NORMAL,
    )?;
    let mut optimizer = util::create_default_rmsprop_optimizer(var_map)?;

    // loop setting
    let loop_param = LoopParam {
        max_iter: 1000,
        disp_interval: DispInterval::DISP(200),
        ..Default::default()
    };

    // learning
    svgd_model.optimize(&sample, &post_param, &mut optimizer, loop_param)?;
    //println!("post_param = {}", &post_param);

    Ok(post_param)
}

fn svgd_test() -> Result<()> {
    // settings
    let device = Device::Cpu;
    let n_params = 10;
    let dim = 5;
    let pri_beta = 1.;
    let band_type = BandType::Adaptive;
    let n_sample = 50;

    // data generation
    let true_mean = Tensor::new(vec![2., -2., 2., -2., 2.], &device)?;
    let true_std = Tensor::new(vec![1., 1., 1., 1., 1.], &device)?;
    let normal_dist = NormalMean::new(true_std)?;
    let sample_x = normal_dist
        .sample(n_sample, &true_mean)?
        .repeat((n_params, 1, 1))?
        .transpose(0, 1)?;
    let sample = Sample::new(Some(sample_x), None)?;
    println!("sample: {}", sample);

    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F64, &device);
    let post_param = vs.get_with_hints(
        (n_params, dim),
        "post_param_of_mean",
        DEFAULT_KAIMING_NORMAL,
    )?;

    // optimizer
    let rms_params = ParamsRMSprop {
        lr: 0.01,
        ..Default::default()
    };
    let mut optimizer = RMSprop::new(var_map.all_vars(), rms_params)?;
    //let adamw_params = ParamsAdamW {
    //    lr: 0.05,
    //    ..Default::default()
    //};
    //let mut optimizer = AdamW::new(var_map.all_vars(), adamw_params)?;

    // model settings
    let post_std = Tensor::new(vec![1., 1., 1., 1., 1.], &device)?;
    let pxw = NormalMean::new(post_std)?;
    let pri_std = (pri_beta * Tensor::ones((dim,), DType::F64, &device)?)?;
    let pw = NormalMean::new(pri_std)?;
    let kernel = RBF::new(band_type)?;
    let svgd_model = SVGD::new(pxw, pw, kernel)?;

    // loop setting
    let loop_param = LoopParam {
        max_iter: 1000,
        disp_interval: DispInterval::DISP(500),
        ..Default::default()
    };

    // learning
    svgd_model.optimize(&sample, &post_param, &mut optimizer, loop_param)?;
    println!("post_param = {}", &post_param);

    Ok(())
}

fn main() -> Result<()> {
    //println!("kernel test");
    //let res = kernel_test();
    //println!("res = {:?}", res);

    //println!("prob test");
    //let res = prob_test();
    //println!("res = {:?}", res);

    //println!("svgd test");
    //let res = svgd_test();
    //println!("res = {:?}", res);
    //Ok(())

    let device = Device::Cpu;

    // data setting
    let n_sample = 50;
    let n_params = 10;
    let true_mean = Tensor::new(vec![2., -2., 2., -2., 2.], &device)?;
    let true_std = Tensor::new(vec![1., 1., 1., 1., 1.], &device)?;
    let normal_dist = NormalMean::new(true_std)?;

    let sample = create_normal_samples(normal_dist, true_mean, n_sample, n_params)?;

    let post_params = svgd_normal(&device, &sample, n_params)?;
    println!("post_params = {}", post_params);

    Ok(())
}
