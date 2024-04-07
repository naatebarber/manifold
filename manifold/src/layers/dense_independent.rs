use core::fmt::Debug;

use ndarray::{Array, Array1, Array2, Array3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Layer {
    pub x: Array3<f64>,
    pub w: Array2<f64>,
    pub b: Array1<f64>,
    pub d_z: Array3<f64>,
    pub grad_w: Array2<f64>,
    pub grad_b: Array1<f64>,
    pub activation: Activations,
}

impl Layer {
    pub fn new(
        x_shape: (usize, usize, usize),
        w_shape: (usize, usize),
        b_shape: usize,
        activation: Activations,
    ) -> Layer {
        Layer {
            x: Array3::zeros(x_shape),
            w: Array2::random(w_shape, Uniform::new(0., 1.)),
            b: Array::random(b_shape, Uniform::new(0., 1.)),
            d_z: Array3::zeros(x_shape),
            grad_w: Array2::zeros(w_shape),
            grad_b: Array::zeros(b_shape),
            activation,
        }
    }

    pub fn forward(&mut self, x: Array3<f64>) -> Array3<f64> {
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

    pub fn backward(&mut self, grad_output: Array3<f64>, learning_rate: f64) -> Array3<f64> {
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
        let avg_grad_w = grad_w.mapv(|x| learning_rate * (x / x_batch_size as f64));
        let avg_grad_b = grad_b.mapv(|x| learning_rate * (x / x_batch_size as f64));

        self.w -= &(avg_grad_w);
        self.b -= &(avg_grad_b);

        grad_input
            .into_shape((x_batch_size, x_sequence_length, x_features))
            .unwrap()
    }
}
