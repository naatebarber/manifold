use core::fmt::Debug;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::substrate::Substrate;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Layer {
    pub x: Array2<f64>,
    pub wi: Array2<usize>,
    pub bi: Array1<usize>,
    pub w: Array2<f64>,
    pub b: Array1<f64>,
    pub d_z: Array2<f64>,
    pub grad_w: Array2<f64>,
    pub grad_b: Array1<f64>,
    pub activation: Activations,
}

impl Layer {
    pub fn new(
        pool_size: usize,
        x_shape: (usize, usize),
        w_shape: (usize, usize),
        b_shape: usize,
        activation: Activations,
    ) -> Layer {
        Layer {
            x: Array2::zeros(x_shape),
            wi: Array2::random(w_shape, Uniform::new(0, pool_size)),
            bi: Array::random(b_shape, Uniform::new(0, pool_size)),
            w: Array2::zeros(w_shape),
            b: Array::zeros(b_shape),
            d_z: Array2::zeros(w_shape),
            grad_w: Array2::zeros(w_shape),
            grad_b: Array::zeros(b_shape),
            activation,
        }
    }

    pub fn gather(&mut self, substrate: &Substrate) -> &mut Self {
        self.w = self.wi.map(|ix| substrate.get(*ix));
        return self;
    }

    pub fn shift_weights(&mut self, shift: &Array2<usize>) -> &mut Self {
        self.wi += shift;
        return self;
    }

    pub fn shift_bias(&mut self, shift: &Array1<usize>) -> &mut Self {
        self.bi += shift;
        return self;
    }

    pub fn assign_grad_w(&mut self, grad: Array2<f64>) -> &mut Self {
        self.grad_w = grad;
        self
    }

    pub fn assign_grad_b(&mut self, grad: Array1<f64>) -> &mut Self {
        self.grad_b = grad;
        self
    }

    pub fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.x = x.clone();
        let z = x.dot(&self.w) + &self.b;
        let a_z = self.activation.wake().a(z.clone());
        let d_z = self.activation.wake().d(z.clone());
        self.d_z = d_z;
        a_z
    }

    pub fn backward(&mut self, grad_output: Array2<f64>) -> Array2<f64> {
        let grad_z = grad_output * &self.d_z;
        let grad_input = grad_z.dot(&self.w.t());
        let grad_w = self.x.t().dot(&grad_z);
        let grad_b = grad_z.sum_axis(Axis(0));

        self.grad_w -= &(grad_w);
        self.grad_b -= &(grad_b);

        grad_input
    }
}
