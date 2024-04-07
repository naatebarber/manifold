use core::fmt::Debug;
use std::error::Error;
use std::ops::Range;
use std::rc::Rc;

use ndarray::{Array, Array1, Array2, Array3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::{rand_distr, RandomExt};
use rand::{thread_rng, Rng};

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::loss3::{Loss, Losses};

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

#[derive(Serialize, Deserialize, Clone)]
pub enum GradientRetention {
    Roll,
    Zero,
}

pub type LayerSchema = Vec<usize>;
pub type Web = Vec<Layer>;

#[derive(Serialize, Deserialize, Clone)]
pub struct Manifold {
    #[serde(skip)]
    d_in: usize,
    d_out: usize,
    web: Web,
    hidden_activation: Activations,
    verbose: bool,
    gradient_retention: GradientRetention,
    pub layers: LayerSchema,
    pub loss: Losses,
}

impl Manifold {
    pub fn new(d_in: usize, d_out: usize, layers: Vec<usize>) -> Manifold {
        Manifold {
            d_in,
            d_out,
            layers,
            web: Web::new(),
            hidden_activation: Activations::Relu,
            verbose: false,
            loss: Losses::MeanSquaredError,
            gradient_retention: GradientRetention::Zero,
        }
    }

    pub fn dynamic(
        d_in: usize,
        d_out: usize,
        breadth: Range<usize>,
        depth: Range<usize>,
    ) -> Manifold {
        let mut rng = thread_rng();
        let depth = rng.gen_range(depth);
        let layers = (0..depth)
            .map(|_| rng.gen_range(breadth.clone()))
            .collect::<Vec<usize>>();

        Manifold::new(d_in, d_out, layers)
    }

    pub fn set_hidden_activation(&mut self, activation: Activations) -> &mut Self {
        self.hidden_activation = activation;
        self
    }

    pub fn set_loss(&mut self, loss: Losses) -> &mut Self {
        self.loss = loss;
        self
    }

    pub fn set_gradient_retention(&mut self, method: GradientRetention) -> &mut Self {
        self.gradient_retention = method;
        self
    }

    pub fn weave(&mut self) -> &mut Self {
        let mut x_shape = (1, 1, self.d_in);
        let mut w_shape: (usize, usize);
        let mut b_shape: usize;
        let mut p_dim = self.d_in;

        for layer_size in self.layers.iter() {
            w_shape = (p_dim, *layer_size);
            b_shape = w_shape.1;

            self.web.push(Layer::new(
                x_shape,
                w_shape,
                b_shape,
                self.hidden_activation,
            ));
            p_dim = *layer_size;
            x_shape = (1, 1, w_shape.1);
        }

        let w_shape = (p_dim, self.d_out);
        let b_shape = w_shape.1;

        self.web
            .push(Layer::new(x_shape, w_shape, b_shape, Activations::Identity));
        self
    }

    pub fn forward(&mut self, mut x: Array3<f64>) -> Array3<f64> {
        for layer in self.web.iter_mut() {
            x = layer.forward(x);
        }
        x
    }

    pub fn backwards(
        &mut self,
        y_pred: Array2<f64>,
        y: Array2<f64>,
        loss: Rc<dyn Loss>,
        learning_rate: f64,
    ) {
        // TODO Check loss next. Batch is not training.
        let grad_output_i = loss.d(y_pred, y);
        let mut grad_output = grad_output_i.insert_axis(Axis(1));

        for layer in self.web.iter_mut().rev() {
            grad_output = layer.backward(grad_output, learning_rate);
        }
    }

    pub fn dump(&mut self) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn load(serialized: &Vec<u8>) -> Result<Manifold, Box<dyn Error>> {
        Ok(bincode::deserialize(serialized)?)
    }
}
