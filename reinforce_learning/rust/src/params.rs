//! パラメータやエラー型を定義するモジュール
//! このモジュールでは、アルゴリズムのパラメータやエラー型を定義している。
//! パラメータは、アルゴリズムのハイパーパラメータをまとめたものであり、
//! エラー型は、このプログラムで用いるエラーを定義している。
//! また、アルゴリズムの出力をまとめるための構造体も定義している。
//! このモジュールは、`src/`ディレクトリに配置されている。
//! 

use ndarray::Array1;
use thiserror::Error;
use std::collections::HashMap;

/// このプログラムで用いるエラー型
#[derive(Error, Debug)]
pub enum MyError {
    /// 入力のパースに失敗した場合に返すエラー
    #[error("Failed to parse input: {0}")]
    ParseError(String),

    /// 収束しなかった場合に返すエラー
    #[error("Failed to converge value: {0}")]
    ConvergenceError(String),

    #[error("Failed to solve linear programming: {0}")]
    LpError(String),
}

/// 各アルゴリズムで用いるハイパーパラメータ
#[derive(Debug)]
pub struct AlgorithmParameters<'a> {
    /// 割引率
    pub gamma: f64,

    /// 最大ループ回数
    pub max_loop: i32,

    /// 収束判定の閾値
    pub eps: f64,

    /// 収束判定の方法
    pub convergence_method: &'a str,

    /// 初期状態
    pub init_state: Option<Array1<usize>>,

    /// 重み関数
    pub weight: Option<Array1<f64>>,
}
impl Default for AlgorithmParameters<'_> {
    fn default() -> Self {
        AlgorithmParameters {
            gamma: 0.9,
            max_loop: 100,
            eps: 0.1,
            convergence_method: "inf_norm",
            init_state: None,
            weight: None,
        }
    }
}

/// アルゴリズムの出力
#[derive(Debug)]
pub struct AlgOutput {
    /// 推定された最適価値関数
    pub value_vec: Array1<f64>,

    /// 推定された最適方策
    pub policy: Array1<usize>,

    /// 収束までのループ回数
    pub loop_count: i32,

    /// 収束した際の収束値
    pub convergence_value: f64,

    /// デバッグ用の履歴
    pub debug_historys: HashMap<String, History>,
}

/// デバッグで用いる変数をまとめるための列挙型
#[derive(Debug)]
pub enum History {
    /// 推定された最適価値関数の履歴を保持するための列挙型
    ValueVec(Vec<Array1<f64>>),

    /// 収束値の履歴を保持するための列挙型
    ConvergenceValue(Vec<f64>),
}

/// 線形計画法の出力
/// 
/// 線形計画法の出力は、最適価値関数と最適方策をまとめたものである。
/// この構造体は、`src/linear_programming_method`関数で用いられる。
/// 
#[derive(Debug)]
pub struct LpOutput {
    /// 最適価値関数
    pub value_vec: Array1<f64>,

    /// 最適方策
    pub policy: Array1<usize>,
}