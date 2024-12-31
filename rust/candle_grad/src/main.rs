//! This crate is experimental for auto grad use for candle.
//! Mainly focus on whether we can use candle as optimization.

mod regression;

use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_nn::{AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use regression::{LinearRegParams, ReducedRankRegParams, ReressionModel};

fn loglik_linear_reg<T>(y: &Tensor, x: &Tensor, param: &T) -> Result<Tensor>
where
    T: ReressionModel,
{
    let logp = (y - param.forward(x)?)?.sqr()?.sum_all()?;
    logp + param.log_p_param()?
}

fn linear_regression_test() -> Result<()> {
    let device = Device::Cpu;
    let n = 100;
    let dim = 5;

    // data generation
    let true_param = Linear::new(
        Tensor::new(&[[0.5, -0.5, 1.5, -2.5, 4.5]], &device)?.t()?,
        Some(Tensor::new(&[1.], &device)?),
    );
    let true_model = LinearRegParams::from_params(true_param, None)?;
    let (train_x, train_y) = true_model.create_date(n, &device)?;

    println!("x = {:?}, y = {:?}", train_x, train_y);

    // model
    let var_map = VarMap::new();
    let var_builder_args = VarBuilder::from_varmap(&var_map, DType::F64, &device);
    let model = LinearRegParams::new(&var_builder_args, dim, Some(1.))?;

    // optimizer
    let adamw_params = ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(var_map.all_vars(), adamw_params)?;

    println!("start learning");
    for epoch in 1..200 {
        let mut sum_loss = 0f64;
        let mut samples = 0f64;

        // train
        let loss = loglik_linear_reg(&train_y, &train_x, &model)?;
        let _ = optimizer.backward_step(&loss);
        sum_loss += loss.to_vec0::<f64>()?;
        samples += train_x.dim(0)? as f64;
        println!("Epoch: {}, loss: {}", epoch, sum_loss / samples);
    }
    println!("{}", model);

    println!("all_vars = {:?}", var_map.data());
    Ok(())
}

fn reduced_rank_regression_test() -> Result<()> {
    let device = Device::Cpu;
    let n = 100;
    let input_dim = 5;
    let out_dim = 2;
    let true_hidden_dim = 3;
    let train_hidden_dim = 5;

    // true model
    let true_model = ReducedRankRegParams::from_params(
        Tensor::randn(0., 1., (true_hidden_dim, input_dim), &device)?,
        Tensor::randn(0., 1., (out_dim, true_hidden_dim), &device)?,
        None,
        None,
    )?;
    let (train_x, train_y) = true_model.create_date(n, &device)?;
    let (test_x, test_y) = true_model.create_date(n, &device)?;

    println!("x = {:?}, y = {:?}", train_x, train_y);

    // model
    let var_map = VarMap::new();
    let var_builder_args = VarBuilder::from_varmap(&var_map, DType::F64, &device);
    let model = ReducedRankRegParams::new(
        &var_builder_args,
        (out_dim, train_hidden_dim, input_dim),
        false,
        Some(1.),
    )?;
    println!("all_vars = {:?}", var_map.data());

    // optimizer
    let adamw_params = ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(var_map.all_vars(), adamw_params)?;

    println!("start learning");
    for epoch in 1..200 {
        let mut sum_loss = 0f64;
        let mut samples = 0f64;

        // train
        let loss = loglik_linear_reg(&train_y, &train_x, &model)?;
        let _ = optimizer.backward_step(&loss);
        sum_loss += loss.to_vec0::<f64>()?;
        samples += train_x.dim(0)? as f64;
        //println!("Epoch: {}, loss: {}", epoch, sum_loss / samples);

        let test_loss = (&test_y - model.forward(&test_x)?)?
            .sqr()?
            .sum_all()?
            .to_vec0::<f64>()?
            / n as f64;
        println!("Epoch: {}, test loss: {}", epoch, test_loss);
    }
    println!("true_model = {}", true_model);
    println!("learned_model = {}", model);

    //println!("all_vars = {:?}", var_map.data());
    Ok(())
}

fn main() -> Result<()> {
    //let res = linear_regression_test();
    let res = reduced_rank_regression_test();
    res
}
