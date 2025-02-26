mod kernel;
mod prob_dist;
mod snippets;
mod svgd;
mod util;

use candle_core::{Device, Result};

fn main() -> Result<()> {
    let n_sample = 100;
    let n_params = 50;
    let device = Device::Cpu;
    //let res = svgd_normal_test(&device, n_sample, n_params);
    //let res = snippets::svgd_lr_test(&device, n_sample, n_params);
    let res = snippets::toy_singular_test(&device, n_sample, n_params);
    //let res = snippets::dkernel_test();
    println!("res = {:?}", res);

    Ok(())
}
