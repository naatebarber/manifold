use crate::Substrate;
use ndarray::{Array1, Array2, Array3};

pub trait Layer {
    fn forward(&mut self, x: Array3<f64>) -> Array3<f64>;
    fn backward(&mut self, grad_output: Array3<f64>) -> Array3<f64>;
}

pub trait Attached {
    fn gather(&mut self, substrate: &Substrate) -> &mut Self;
    fn shift_weights(&mut self, shift: &Array2<usize>) -> &mut Self;
    fn shift_bias(&mut self, shift: &Array1<usize>) -> &mut Self;
    fn assign_grad_w(&mut self, grad: Array2<f64>) -> &mut Self;
    fn assign_grad_b(&mut self, grad: Array1<f64>) -> &mut Self;
}
