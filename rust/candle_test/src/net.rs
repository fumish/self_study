use candle_core::{Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, VarBuilder};

pub struct BasicNN {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

pub trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
    fn predict(&self, xs: &Tensor) -> Result<Tensor>;
}

impl Model for BasicNN {
    fn new(vs: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(784, 128, vs.pp("layer1"))?;
        let layer2 = candle_nn::linear(128, 64, vs.pp("layer2"))?;
        let layer3 = candle_nn::linear(64, 10, vs.pp("layer3"))?;
        Ok(Self {
            layer1,
            layer2,
            layer3,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, image_dimension) = xs.dims2()?;
        let xs = xs.reshape((batch_size, image_dimension))?;
        let xs = self.layer1.forward(&xs)?;
        let xs = &xs.silu()?;
        let xs = self.layer2.forward(&xs)?;
        let xs = &xs.relu()?;
        self.layer3.forward(&xs)
    }

    fn predict(&self, xs: &Tensor) -> Result<Tensor> {
        let pred_logits = self.forward(xs)?;
        pred_logits.argmax(D::Minus1)
    }
}

/// calc cross entropy loss from output probability of model via images and labels
pub fn cross_entropy_loss(model: &impl Model, images: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let logits = model.forward(images)?;
    let log_softmax = ops::log_softmax(&logits, D::Minus1)?;
    loss::nll(&log_softmax, labels)
}
