use core::fmt::Debug;
use std::error::Error;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use ndarray::{Array, Array1, Array2, Array3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::{seq, thread_rng, Rng};

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::loss3::{Loss, Losses};
use crate::substrate::Substrate;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Layer {
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

impl Layer {
    pub fn new(
        pool_size: usize,
        x_shape: (usize, usize, usize),
        w_shape: (usize, usize),
        b_shape: usize,
        activation: Activations,
    ) -> Layer {
        Layer {
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

    pub fn forward(&mut self, mut x: Array3<f64>) -> Array3<f64> {
        let batch_size = x.shape()[0];
        let features = x.shape()[1];
        let sequence_length = x.shape()[2];

        let x_reshaped = x.into_shape((batch_size * sequence_length, features)).unwrap();
        let z_batch = x_reshaped.dot(&self.w) + &self.b;

        let activ = self.activation.wake();
        let a_z_batch = activ.a(z_batch.clone());
        let d_z_batch = activ.d(z_batch);

        // Reshape a_z and d_z back into 3d
        let a_z = a_z_batch.into_shape((batch_size, features, sequence_length)).unwrap();
        let d_z = d_z_batch.into_shape((batch_size, features, sequence_length)).unwrap();

        self.d_z = d_z;
        a_z
    }

    pub fn backward(&mut self, grad_output: Array3<f64>) -> Array3<f64> {
        let batch_size = self.d_z.shape()[0];
        let features = self.d_z.shape()[1];
        let sequence_length = self.d_z.shape()[2];

        let d_z_batch = &self.d_z.clone().into_shape((batch_size * sequence_length, features)).unwrap();
        let x_batch = &self.x.clone().into_shape((batch_size * sequence_length, features)).unwrap();

        let grad_batch_size = grad_output.shape()[0];
        let grad_features = grad_output.shape()[1];
        let grad_sequence_length = grad_output.shape()[2];

        let grad_output_batch = grad_output.into_shape((grad_batch_size * grad_sequence_length, grad_features)).unwrap();

        let grad_z = grad_output_batch * d_z_batch;
        let grad_input = grad_z.dot(&self.w.t());

        // TODO Average the mf gradients.
        let grad_w = x_batch.t().dot(&grad_z);
        let grad_b = grad_z.sum_axis(Axis(0));

        self.grad_w -= &(grad_w);
        self.grad_b -= &(grad_b);

        grad_input.into_shape((batch_size, features, sequence_length)).unwrap()
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
    substrate: Arc<Substrate>,
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
    pub fn new(
        substrate: Arc<Substrate>,
        d_in: usize,
        d_out: usize,
        layers: Vec<usize>,
    ) -> Manifold {
        Manifold {
            substrate,
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

        Manifold::new(Arc::new(Substrate::blank()), d_in, d_out, layers)
    }

    pub fn set_substrate(&mut self, substrate: Arc<Substrate>) -> &mut Self {
        self.substrate = substrate;
        self
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
                self.substrate.size,
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

        self.web.push(Layer::new(
            self.substrate.size,
            x_shape,
            w_shape,
            b_shape,
            Activations::Identity,
        ));
        self
    }

    pub fn gather(&mut self) -> &mut Self {
        for layer in self.web.iter_mut() {
            layer.gather(&self.substrate);
        }
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
        let grad_output_i = loss.d(y_pred, y);
        let mut grad_output = grad_output_i.insert_axis(Axis(1));

        for layer in self.web.iter_mut().rev() {

            grad_output = layer.backward(grad_output);

            let grad_b_dim = layer.grad_b.raw_dim();
            let grad_w_dim = layer.grad_w.raw_dim();

            let mut b_grad_reshaped = layer.grad_b.to_owned().insert_axis(Axis(1));
            let mut b_link_reshaped = layer.bi.to_owned().insert_axis(Axis(1));

            self.substrate
                .highspeed(&mut layer.grad_w, &mut layer.wi, learning_rate);
            self.substrate
                .highspeed(&mut b_grad_reshaped, &mut b_link_reshaped, learning_rate);

            layer
                .shift_bias(&b_link_reshaped.remove_axis(Axis(1)))
                .assign_grad_b(b_grad_reshaped.remove_axis(Axis(1)))
                .gather(&self.substrate);

            match self.gradient_retention {
                GradientRetention::Zero => {
                    layer
                        .assign_grad_b(Array1::zeros(grad_b_dim))
                        .assign_grad_w(Array2::zeros(grad_w_dim));
                }
                GradientRetention::Roll => (),
            }
        }
    }

    pub fn dump(&mut self) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn load(serialized: &Vec<u8>) -> Result<Manifold, Box<dyn Error>> {
        Ok(bincode::deserialize(serialized)?)
    }
}
