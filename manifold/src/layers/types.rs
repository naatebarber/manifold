use super::Dense;
use crate::{Activations, Substrate};
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

pub trait IsolatedLayer {
    fn forward(&mut self, x: Array3<f64>) -> Array3<f64>;
    fn backward(&mut self, grad_output: Array3<f64>) -> Array3<f64>;
    fn gradients(&self) -> (Array2<f64>, Array1<f64>);
}

pub trait Layer {
    fn forward(&mut self, x: Array3<f64>) -> Array3<f64>;
    fn backward(&mut self, grad_output: Array3<f64>) -> Array3<f64>;
    fn gradients(&self) -> (Array2<f64>, Array1<f64>);
    fn gather(&mut self, substrate: &Substrate);
    fn shift_weights(&mut self, shift: &Array2<usize>);
    fn assign_bi(&mut self, shift: &Array1<usize>);
    fn assign_grad_w(&mut self, grad: Array2<f64>);
    fn assign_grad_b(&mut self, grad: Array1<f64>);
    fn gradient_bindings(&self) -> (Array2<usize>, Array1<usize>);
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Layers {
    Dense,
}

impl Layers {
    pub fn wake(
        layer: Layers,
        pool_size: usize,
        x_shape: (usize, usize, usize),
        w_shape: (usize, usize),
        b_shape: usize,
        activation: Activations,
    ) -> Box<dyn Layer> {
        match layer {
            Layers::Dense => Box::new(Dense::new(pool_size, x_shape, w_shape, b_shape, activation)),
        }
    }
}
