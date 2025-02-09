//! This crate defines a sample, which is a set of data points

pub enum Sample {
    Unsupervised(Tensor),
    Supervised(Tensor, Tensor),
}
