use candle_core::{Result, Tensor};
use candle_nn::{loss, ops, Linear, Module, VarBuilder};

pub struct VanilaVae {
    enc1: Linear,
    enc21: Linear,
    enc22: Linear,
    dec1: Linear,
    dec2: Linear,
}

pub trait EncoderDecoderModel: Sized {
    fn encode(&self, xs: &Tensor) -> Result<(Tensor, Tensor)>;
    fn decode(&self, zs: &Tensor) -> Result<Tensor>;
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (mu, logvar) = self.encode(xs)?;
        let std = (&logvar * 0.5)?.exp()?;
        let eps = std.randn_like(0., 1.);
        let zs = (&mu + eps * std)?;
        let recon_x = self.decode(&zs)?;

        Ok((recon_x, mu, logvar))
    }
}

impl EncoderDecoderModel for VanilaVae {
    fn encode(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let h1 = xs.apply(&self.enc1)?.relu()?;
        let mu = self.enc21.forward(&h1)?;
        let logvar = self.enc22.forward(&h1)?;

        Ok((mu, logvar))
    }

    fn decode(&self, zs: &Tensor) -> Result<Tensor> {
        let x = zs.apply(&self.dec1)?.relu()?.apply(&self.dec2)?;
        // Note: sigmoid function is not implemented in cuda version yet.
        // simple 1/(1+exp(-x)) leads to NaN, so we use tanh instead.
        ((0.5*x)?.tanh()? + 1.)? * 0.5
    }
}

impl VanilaVae {
    pub fn new(vs: VarBuilder, input_dim: usize) -> Result<Self> {
        let enc1 = candle_nn::linear(input_dim, 400, vs.pp("enc1"))?;
        let enc21 = candle_nn::linear(400, 20, vs.pp("enc21"))?;
        let enc22 = candle_nn::linear(400, 20, vs.pp("enc22"))?;
        let dec1 = candle_nn::linear(20, 400, vs.pp("dec1"))?;
        let dec2 = candle_nn::linear(400, input_dim, vs.pp("dec2"))?;
        Ok(Self {
            enc1,
            enc21,
            enc22,
            dec1,
            dec2,
        })
    }
}

pub fn mse_loss(model: &VanilaVae, x: &Tensor) -> Result<Tensor> {
    let (recon_x, mu, logvar) = model.forward(x)?;
    let logp = (&recon_x - x)?.sqr()?.sum_all()?;
    let kl = (-0.5 * ((1f64 + &logvar - &mu.powf(2.)? - &logvar.exp()?)?.sum_all()?))?;
    kl + logp
}

/// write target_image by filename
/// target_image: Tensor with shape(n, H*W)
/// image_size: (H, W)
pub fn write_image(target_image: &Tensor, image_size: (u32, u32), filename: &str) -> Result<()> {
    let mut img = image::GrayImage::new(image_size.0, image_size.1);
    let target_image = target_image.to_vec2::<u8>()?;



    target_images.to_vec2::<u8>()?
    image::GrayImage::from
    //let image = image.to_device(Device::Cpu)?;
    //let image = image.to_dtype(DType::U8)?;
    //candle_datasets::vision:
    //let image = image.mul_scalar(255f32)
}