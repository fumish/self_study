extern crate intel_mkl_src;
extern crate openblas_src;
extern crate blas_src;
extern crate ndarray;

use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_stats::QuantileExt;
use ndarray_linalg::Solve;
use std::collections::HashMap;
use good_lp::{constraint, default_solver, variable, variables, Expression, SolverModel, Solution};

pub mod params;
use params::{AlgorithmParameters, MyError, AlgOutput, History, LpOutput};

/// 価値反復法を用いて最適価値関数と最適方策を求める
/// # パラメータ:
/// * `p_t` - 遷移確率行列
/// * `g_t` - 報酬行列
/// * `learning_params` - 学習パラメータ
/// 
/// # 戻り値:
/// * `AlgOutput` - アルゴリズムの出力
/// * `MyError` - エラー型
/// 
/// # Example
/// ```
/// use ndarray::array;
/// use reinforce_learning::value_iteration_method;
/// use reinforce_learning::params::AlgorithmParameters;
/// 
/// let n_state = 2;
/// let n_action = 3;
/// let p_t = array![0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0]
/// .into_shape((n_state, n_state, n_action)).unwrap();
/// let g_t = array![5.0, 10.0, 2.0, 1.0, 0.0, 0.0].into_shape((n_state, n_action)).unwrap();
/// let learning_params = AlgorithmParameters::default();
/// let result = value_iteration_method(&p_t, &g_t, &learning_params);
/// 
/// ```
/// 
pub fn value_iteration_method(
    p_t: &Array3<f64>, 
    g_t: &Array2<f64>,
    learning_params: &AlgorithmParameters,
) -> Result<AlgOutput, MyError> {
    let n_state = p_t.shape()[0];

    let max_loop = learning_params.max_loop;
    let eps = learning_params.eps;
    let gamma = learning_params.gamma;
    let convergence_method = learning_params.convergence_method;

    let mut curr_value_vec = Array1::<f64>::zeros(n_state);
    let mut value_vec_history = vec![];
    let mut convergence_value_history = vec![];

    for ite in 0..max_loop {
        // 最適価値関数の更新
        let next_value_vec = (
            g_t + gamma * (
                p_t * curr_value_vec.clone().into_shape((n_state, 1, 1)).unwrap()
            ).sum_axis(Axis(0))
        ).map_axis(
            Axis(1),
            |view| *view.max().unwrap()
        );
        
        // 収束判定の値計算
        let diff_value_vec = &next_value_vec - &curr_value_vec;
        let convergence_value = match convergence_method {
            "inf_norm" => {
                *diff_value_vec.mapv_into(
                    |x| x.abs()
                ).max().unwrap()
            },
            "min_max" => {
                diff_value_vec.max().unwrap() - diff_value_vec.min().unwrap()
            },
            _ => {
                return Err(MyError::ParseError("convergence_method is either inf_norm or min_max".to_string()))
            },
        };

        // 履歴の保存
        convergence_value_history.push(convergence_value);
        value_vec_history.push(next_value_vec.clone());

        // 収束判定
        if convergence_value < eps {
            // 最適方策の計算
            let est_pi_s = (
                g_t + gamma * (
                    p_t * next_value_vec.clone().into_shape((n_state, 1, 1)).unwrap()
                ).sum_axis(Axis(0))
            ).map_axis(
                Axis(1),
                |view| view.argmax().unwrap()
            );

            // デバッグ用の履歴をまとめる
            let mut debug_historys = HashMap::new();
            debug_historys.insert("value_vec".to_string(), History::ValueVec(value_vec_history));
            debug_historys.insert("convergence_value".to_string(), History::ConvergenceValue(convergence_value_history));

            return Ok(
                AlgOutput {
                    value_vec: next_value_vec,
                    policy: est_pi_s,
                    loop_count: ite, 
                    convergence_value: convergence_value,
                    debug_historys: debug_historys,
                }
            )
        } else {
            // 収束しなかった場合は次のループに進む
            value_vec_history.push(next_value_vec.clone());
            curr_value_vec.assign(&next_value_vec);
        }
    }
    // 収束しなかった場合はエラーを返す
    Err(MyError::ConvergenceError("Failed to converge value".to_string()))
}


/// 方策反復法を用いて最適価値関数と最適方策を求める
/// # パラメータ:
/// * `p_t` - 遷移確率行列
/// * `g_t` - 報酬行列
/// * `learning_params` - 学習パラメータ
/// 
/// # Example
/// ```
/// use ndarray::{array, Array1};
/// use reinforce_learning::policy_iteration_method;
/// use reinforce_learning::params::AlgorithmParameters;
/// 
/// let n_state = 2;
/// let n_action = 3;
/// let p_t = array![0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0]
/// .into_shape((n_state, n_state, n_action)).unwrap();
/// let g_t = array![5.0, 10.0, 2.0, 1.0, 0.0, 0.0].into_shape((n_state, n_action)).unwrap();
/// 
/// let learning_params = AlgorithmParameters {
///   gamma: 0.9,
///   max_loop: 100,
///   eps: 0.1,
///   convergence_method: "inf_norm",
///   init_state: None,
/// };
/// let result = policy_iteration_method(&p_t, &g_t, &learning_params);
/// ```
/// 
pub fn policy_iteration_method(
    p_t: &Array3<f64>,
    g_t: &Array2<f64>,
    learning_params: &AlgorithmParameters,
) -> Result<AlgOutput, MyError> {
    println!("{:?}", learning_params);
    let n_state = p_t.shape()[0];
    let mut curr_pi_s = match &learning_params.init_state {
        Some(init_state) => init_state.clone(),
        None => Array1::<usize>::zeros(n_state),
    };

    let max_loop = learning_params.max_loop;
    let gamma = learning_params.gamma;
    let eps = learning_params.eps;

    let mut value_vec_history = vec![];
    let mut convergence_value_history = vec![];

    for ite in 0..max_loop{
        let mut state_mat = Array2::<f64>::zeros((n_state, n_state));
        let mut reward_state = Array1::<f64>::zeros(n_state);
        for i in 0..n_state {
            state_mat.row_mut(i).assign(&p_t.slice(s![..,i,curr_pi_s[i]]));
            reward_state[i] = g_t[[i, curr_pi_s[i]]];
        }

        let est_v_pi = (Array2::eye(n_state) - gamma*&state_mat)
        .solve_into(reward_state).unwrap();
        // 最適方策の計算
        let next_pi_s = (
            g_t + gamma * (
                p_t * est_v_pi.clone().into_shape((n_state, 1, 1)).unwrap()
            ).sum_axis(Axis(0))
        ).map_axis(
            Axis(1),
            |view| view.argmax().unwrap()
        );

        let convergence_value = *(
            &next_pi_s.mapv(|x| x as f64) - &curr_pi_s.mapv(|x| x as f64)
        ).mapv(|x| x.abs()).max().unwrap();
        convergence_value_history.push(convergence_value);
        value_vec_history.push(est_v_pi.clone());
        //        next_pi_s.mapv(|x| x as f64).abs_diff_eq(&curr_pi_s.mapv(|x| x as f64), eps)
        if convergence_value < eps {
            // デバッグ用の履歴をまとめる
            let mut debug_historys = HashMap::new();
            debug_historys.insert("value_vec".to_string(), History::ValueVec(value_vec_history));
            debug_historys.insert("convergence_value".to_string(), History::ConvergenceValue(convergence_value_history));

            return Ok(
                AlgOutput {
                    value_vec: est_v_pi,
                    policy: next_pi_s,
                    loop_count: ite, 
                    convergence_value: convergence_value, 
                    debug_historys: debug_historys,
                }
            )
        } else {
            curr_pi_s.assign(&next_pi_s);
        }    
    }

    Err(MyError::ConvergenceError("Failed to converge value".to_string()))
}


/// 線形計画法を用いて最適価値関数と最適方策を求める
/// # パラメータ:
/// * `p_t` - 遷移確率行列
/// * `g_t` - 報酬行列
/// * `learning_params` - 学習パラメータ
/// 
pub fn linear_programming_method(
    p_t: &Array3<f64>,
    g_t: &Array2<f64>,
    learning_params: &AlgorithmParameters,
) -> Result<LpOutput, MyError> {
    let n_state = p_t.shape()[0];
    let n_action = p_t.shape()[2];
    let gamma = learning_params.gamma;
    let weight = match &learning_params.weight {
        Some(weight) => weight.clone(),
        None => Array1::<f64>::ones(n_state) / n_state as f64,
    };
    
    // 目的変数の設定
    let mut variables = variables!();
    let mut vs = HashMap::new();
    for i in 0..n_state {
        vs.insert(
            i,
            variables.add(variable())
        );
    }

    // 目的関数の設定
    let objective = vs
        .iter()
        .map(|(&i, &var)| weight[i] * var)
        .sum::<Expression>();

    // 問題の設定
    let mut problem = variables.minimise(objective).using(default_solver);

    // 不等式制約の設定
    for s1 in 0..n_state {
        for a in 0..n_action {
            // 遷移確率の和の部分の計算
            let transition_sum = (0..n_state)
                .map(|s2| gamma * p_t[[s2, s1, a]] * vs[&s2])
                .sum::<Expression>();
            
            // transition_sumを使って不等式制約を記述
            problem = problem.with(
                constraint!(
                    vs[&s1] - g_t[[s1, a]] - transition_sum >= 0
                )
            );
        }
    }
    let solution = problem.solve();

    match solution {
        Ok(sol) => {
            let value_vec = Array1::from(
                vs
                .iter()
                .map(|(_, &var)| sol.value(var))
                .collect::<Vec<f64>>()
            );
            let policy = (
                g_t + gamma * (
                    p_t * value_vec.clone().into_shape((n_state, 1, 1)).unwrap()
                ).sum_axis(Axis(0))
            ).map_axis(
                Axis(1),
                |view| view.argmax().unwrap()
            );
            Ok(
                LpOutput{
                    value_vec: value_vec,
                    policy: policy
                }
            )
        },
        Err(e) => Err(MyError::LpError(e.to_string())),
    }

}