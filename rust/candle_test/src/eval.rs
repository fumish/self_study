use candle_core::{DType, Result, Tensor};

/// calc accuracy between pred and test
pub fn calc_accuracy(pred_labels: &Tensor, test_labels: &Tensor) -> Result<f32> {
    let sum_ok = pred_labels
        .eq(test_labels)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let test_accuracy = sum_ok / test_labels.dims1()? as f32;
    Ok(test_accuracy)
}
