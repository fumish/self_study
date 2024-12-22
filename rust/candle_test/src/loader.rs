use candle_core::{DType, Device, Result, Tensor};
use candle_datasets::vision::Dataset;

pub fn create_train_test_dataset(
    dataset: Dataset,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let test_images = dataset.test_images.to_device(device)?;
    let test_labels = dataset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(device)?;
    let train_images = dataset.train_images.to_device(device)?;
    let train_labels = dataset
        .train_labels
        .to_dtype(DType::U32)?
        .to_device(device)?;
    Ok((train_images, train_labels, test_images, test_labels))
}
