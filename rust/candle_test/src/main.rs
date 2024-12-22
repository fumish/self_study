mod eval;
mod loader;
mod net;
mod vae;

use candle_core::{DType, Device, Result, Tensor};
use candle_datasets::vision::mnist;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use net::{BasicNN, Model};
use rand::prelude::*;
use vae::VanilaVae;

fn image_classification(
    train_images: &Tensor,
    train_labels: &Tensor,
    test_images: &Tensor,
    test_labels: &Tensor,
    device: &Device,
    batch_size: usize,
) -> Result<()> {
    // model
    let var_map = VarMap::new();
    let var_builder_args = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let model = BasicNN::new(var_builder_args.clone())?;

    // Optimizer
    let adamw_params = ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(var_map.all_vars(), adamw_params)?;

    //let batches = train_images.dim(0)?; // / BSIZE;
    let batches = train_images.dim(0)? / batch_size;
    let mut batch_indices = (0..batches).collect::<Vec<usize>>();

    println!("start learning");
    for epoch in 1..20 {
        let mut sum_loss = 0f32;

        //optimizer.backward_step(&loss)?;
        //sum_loss += loss.to_vec0::<f32>()?;
        //let avg_loss = sum_loss / batches as f32;

        // train
        batch_indices.shuffle(&mut thread_rng());
        for batch_index in batch_indices.iter() {
            let train_images = train_images.narrow(0, batch_index * batch_size, batch_size)?;
            let train_labels = train_labels.narrow(0, batch_index * batch_size, batch_size)?;
            let loss = net::cross_entropy_loss(&model, &train_images, &train_labels)?;
            optimizer.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / batches as f32;

        // test
        let pred_labels = model.predict(&test_images)?;
        let test_accuracy = eval::calc_accuracy(&pred_labels, &test_labels)?;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }
    Ok(())
}


fn image_reconstruction(train_images: &Tensor, device: &Device, batch_size: usize) -> Result<()> {
    let input_size = train_images.dim(1)?;

    // model
    let var_map = VarMap::new();
    let var_builder_args = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let model = VanilaVae::new(var_builder_args.clone(), input_size)?;

    // Optimizer
    let adamw_params = ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(var_map.all_vars(), adamw_params)?;

    let batches = train_images.dim(0)? / batch_size;
    let mut batch_indices = (0..batches).collect::<Vec<usize>>();

    println!("start learning");
    for epoch in 1..20 {
        let mut sum_loss = 0f32;
        let mut samples = 0f32;

        // train
        batch_indices.shuffle(&mut thread_rng());
        for batch_index in batch_indices.iter() {
            let train_images = train_images.narrow(0, batch_index * batch_size, batch_size)?;
            let loss = vae::mse_loss(&model, &train_images)?;
            optimizer.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
            samples += train_images.dim(0)? as f32;
            //println!("sum_loss = {:?}, samples = {:?}", &sum_loss, &samples);
        }
        println!("Epoch: {}, loss: {}", epoch, sum_loss / samples);
    }

    Ok(())
}

fn main() -> Result<()> {
    const BSIZE: usize = 128;

    let device = Device::cuda_if_available(0)?;
    println!("device = {:?}", &device);

    // dataset
    let dataset = mnist::load_dir("./data").unwrap();
    let (train_images, train_labels, test_images, test_labels) =
        loader::create_train_test_dataset(dataset, &device)?;

    println!("train-images: {:?}", train_images.shape());
    println!("train-labels: {:?}", train_labels.shape());
    println!("test-images: {:?}", test_images.shape());
    println!("test-labels: {:?}", test_labels.shape());

    let res = image_reconstruction(&train_images, &device, BSIZE);
    println!("res = {:?}", res);
    Ok(())

    //image_classification(
    //    &train_images,
    //    &train_labels,
    //    &test_images,
    //    &test_labels,
    //    &device,
    //    BSIZE,
    //)

    //// model
    //let var_map = VarMap::new();
    //let var_builder_args = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    //let model = BasicNN::new(var_builder_args.clone())?;

    //// Optimizer
    //let adamw_params = candle_nn::ParamsAdamW {
    //    lr: 0.05,
    //    ..Default::default()
    //};
    //let mut optimizer = candle_nn::AdamW::new(var_map.all_vars(), adamw_params)?;

    ////let batches = train_images.dim(0)?; // / BSIZE;
    //let batches = train_images.dim(0)? / BSIZE;
    //let mut batch_indices = (0..batches).collect::<Vec<usize>>();

    //println!("start learning");
    //for epoch in 1..20 {
    //    let mut sum_loss = 0f32;

    //    //optimizer.backward_step(&loss)?;
    //    //sum_loss += loss.to_vec0::<f32>()?;
    //    //let avg_loss = sum_loss / batches as f32;

    //    // train
    //    batch_indices.shuffle(&mut thread_rng());
    //    for batch_index in batch_indices.iter() {
    //        let train_images = train_images.narrow(0, batch_index * BSIZE, BSIZE)?;
    //        let train_labels = train_labels.narrow(0, batch_index * BSIZE, BSIZE)?;
    //        let loss = net::cross_entropy_loss(&model, &train_images, &train_labels)?;
    //        optimizer.backward_step(&loss)?;
    //        sum_loss += loss.to_vec0::<f32>()?;
    //    }
    //    let avg_loss = sum_loss / batches as f32;

    //    // test
    //    let pred_labels = model.predict(&test_images)?;
    //    let test_accuracy = eval::calc_accuracy(&pred_labels, &test_labels)?;
    //    println!(
    //        "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
    //        avg_loss,
    //        100. * test_accuracy
    //    );
    //}

    //Ok(())
}
