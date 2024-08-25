extern crate nalgebra as na;

use anyhow::Result;
use linfa::traits::Transformer;
use linfa_clustering::{Dbscan, DbscanParams};
use linfa_datasets::generate;
use ndarray::Array;
use pcd_rs::{PcdDeserialize, Reader};

#[derive(PcdDeserialize, Debug)]
pub struct Point {
    x: f32,
    y: f32,
    z: f32,
}

impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Point { x, y, z }
    }
    pub fn to_tuple(&self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.x, self.y, self.z]
    }
}

fn read_pcd_file(filename: &str) -> Result<Vec<Point>> {
    let reader = Reader::open(filename);

    match reader {
        Err(e) => {
            println!("Error: {:?}", e);
            Err(e)
        }
        Ok(reader) => reader.collect(),
    }
}

fn main() {
    let read_res = read_pcd_file("./data/bunny.pcd");
    let points = match read_res {
        Err(e) => {
            println!("Error: {:?}", e);
            return;
        }
        Ok(points) => points,
    };
    // println!("points is {:?}", points);

    let point_vec = points
        .into_iter()
        .map(|point| point.to_vec())
        .flatten()
        .collect::<Vec<f32>>();

    println!("point_vec size is {:?}", point_vec.len());

    // 2d vec to na array
    //let point_arr = na::DMatrix::from_vec(point_vec.len() / 3, 3, point_vec);
    let point_arr = Array::from_shape_vec((point_vec.len() / 3, 3), point_vec).unwrap();

    println!("point_arr is {:?}", point_arr);
    println!("point_arr shape is {:?}", point_arr.shape());
    // println!("point_arr dim is {:?}", point_arr.ndim());
    //
    let min_samples = 5;
    let clusters = Dbscan::params(min_samples)
        .tolerance(1e-2)
        .transform(&point_arr);
    println!("res = {:?}", clusters);
}
