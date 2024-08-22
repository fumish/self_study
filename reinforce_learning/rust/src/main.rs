//! # reinforce_learning
//! 強化学習を行うライブラリ
//! 

extern crate intel_mkl_src;
extern crate openblas_src;
extern crate blas_src;
extern crate ndarray;

use ndarray::{array, Array1};
use rand::Rng;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use reinforce_learning::params::AlgorithmParameters; 

fn main() {
    let n_state = 2;
    let n_action = 3;
    let p_t = array![0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0]
    .into_shape((n_state, n_state, n_action)).unwrap();
        // 0.8, 0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    let g_t = array![5.0, 10.0, 2.0, 1.0, 0.0, 0.0].into_shape((n_state, n_action)).unwrap();

    // usize個の乱数を生成
    let mut rng = ChaCha8Rng::seed_from_u64(20240607);
    let init_state = Array1::<usize>::from_shape_fn(n_state, |_| rng.gen_range(0..n_action));

    let learning_params = AlgorithmParameters {
        gamma: 0.95,
        max_loop: 100,
        eps: 0.1,
        convergence_method: "inf_norm",
        init_state: Some(init_state),
        weight: None,
    };
    println!("value_iteration_method start");
    let result = reinforce_learning::value_iteration_method(&p_t, &g_t, &learning_params);
    match result {
        Ok(v) => {
            println!(
                "value_vec={:?}, policy={:?}, loop_count={:?}, convergence_value={:?}, debug_historys={:?}", 
                v.value_vec, v.policy, v.loop_count, v.convergence_value, v.debug_historys,
            );
        },
        Err(e) => {
            println!("{:?}", e);
        }
    };
    println!("value_iteration_method end");
    println!("--------------------------");

    println!("policy_iteration_method start");
    let result = reinforce_learning::policy_iteration_method(
        &p_t, &g_t, &learning_params
    );
    match result {
        Ok(v) => {
            println!(
                "value_vec={:?}, policy={:?}, loop_count={:?}, convergence_value={:?}, debug_historys={:?}", 
                v.value_vec, v.policy, v.loop_count, v.convergence_value, v.debug_historys,
            );
        },
        Err(e) => {
            println!("{:?}", e);
        }
    };
    println!("policy_iteration_method end");
    println!("--------------------------");

    println!("linear_programming_method start");
    let result = reinforce_learning::linear_programming_method(
        &p_t, &g_t, &learning_params
    );
    match result {
        Ok(v) => println!("{:?}", v),
        Err(e) => println!("{:?}", e),
    };
    println!("linear_programming_method end");
    println!("--------------------------");
}
