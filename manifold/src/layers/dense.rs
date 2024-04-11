use core::fmt::Debug;

use ndarray::{Array, Array1, Array2, Array3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::substrate::Substrate;

use super::types::Layer;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Dense {
    pub x: Array3<f64>,
    pub wi: Array2<usize>,
    pub bi: Array1<usize>,
    pub w: Array2<f64>,
    pub b: Array1<f64>,
    pub d_z: Array3<f64>,
    pub grad_w: Array2<f64>,
    pub grad_b: Array1<f64>,
    pub activation: Activations,
}

impl Dense {
    pub fn new(
        pool_size: usize,
        x_shape: (usize, usize, usize),
        w_shape: (usize, usize),
        b_shape: usize,
        activation: Activations,
    ) -> Dense {
        Dense {
            x: Array3::zeros(x_shape),
            wi: Array2::random(w_shape, Uniform::new(0, pool_size)),
            bi: Array::random(b_shape, Uniform::new(0, pool_size)),
            w: Array2::zeros(w_shape),
            b: Array::zeros(b_shape),
            d_z: Array3::zeros(x_shape),
            grad_w: Array2::zeros(w_shape),
            grad_b: Array::zeros(b_shape),
            activation,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: Array3<f64>) -> Array3<f64> {
        let batch_size = x.shape()[0];
        let sequence_length = x.shape()[1];
        let features = x.shape()[2];

        let x_reshaped = x
            .clone()
            .into_shape((batch_size * sequence_length, features))
            .unwrap();
        let z_batch = x_reshaped.dot(&self.w) + &self.b;
        let activ = self.activation.wake();

        let a_z_batch = activ.a(z_batch.clone());
        let d_z_batch = activ.d(z_batch);

        let new_features = a_z_batch.shape()[1];

        // Reshape a_z and d_z back into 3d
        let a_z = a_z_batch
            .into_shape((batch_size, sequence_length, new_features))
            .unwrap();
        let d_z = d_z_batch
            .into_shape((batch_size, sequence_length, new_features))
            .unwrap();

        self.d_z = d_z;
        self.x = x;
        a_z
    }

    fn backward(&mut self, grad_output: Array3<f64>) -> Array3<f64> {
        let dz_batch_size = self.d_z.shape()[0];
        let dz_sequence_length = self.d_z.shape()[1];
        let dz_features = self.d_z.shape()[2];

        let d_z_batch = &self
            .d_z
            .clone()
            .into_shape((dz_batch_size * dz_sequence_length, dz_features))
            .unwrap();

        let x_batch_size = self.x.shape()[0];
        let x_sequence_length = self.x.shape()[1];
        let x_features = self.x.shape()[2];

        let x_batch = &self
            .x
            .clone()
            .into_shape((x_batch_size * x_sequence_length, x_features))
            .unwrap();

        let grad_batch_size = grad_output.shape()[0];
        let grad_features = grad_output.shape()[2];
        let grad_sequence_length = grad_output.shape()[1];

        let grad_output_batch = grad_output
            .into_shape((grad_batch_size * grad_sequence_length, grad_features))
            .unwrap();
        let grad_z = grad_output_batch * d_z_batch;

        let wt = self.w.t();
        let grad_input = grad_z.dot(&wt);

        let grad_w = x_batch.t().dot(&grad_z);
        let grad_b = grad_z.sum_axis(Axis(0));

        // Mean gradients instead of accumulating
        let avg_grad_w = grad_w.mapv(|x| x / x_batch_size as f64);
        let avg_grad_b = grad_b.mapv(|x| x / x_batch_size as f64);

        self.grad_w -= &(avg_grad_w);
        self.grad_b -= &(avg_grad_b);

        grad_input
            .into_shape((x_batch_size, x_sequence_length, x_features))
            .unwrap()
    }
    
    fn gradients(&self) -> (Array2<f64>, Array1<f64>) {
        (self.grad_w.clone(), self.grad_b.clone())
    }

    fn gather(&mut self, substrate: &Substrate) {
        self.w = self.wi.map(|ix| substrate.get(*ix));
    }

    fn shift_weights(&mut self, shift: &Array2<usize>) {
        self.wi += shift;
    }

    fn shift_bias(&mut self, shift: &Array1<usize>) {
        self.bi += shift;
    }

    fn assign_grad_w(&mut self, grad: Array2<f64>) {
        self.grad_w = grad;
    }

    fn assign_grad_b(&mut self, grad: Array1<f64>) {
        self.grad_b = grad;
    }

    fn gradient_bindings(&self) -> (Array2<usize>, Array1<usize>) {
        (self.wi.clone(), self.bi.clone())
    }
}