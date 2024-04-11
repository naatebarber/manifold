use std::error::Error;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Axis};
use rand::{thread_rng, Rng};

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::layers::Dense;
use crate::layers::types::Layer;
use crate::loss::{Loss, Losses};
use crate::substrate::Substrate;

use super::types::{GradientRetention, Manifold};

pub type LayerSchema = Vec<usize>;
pub type Web = Vec<Dense>;

#[derive(Serialize, Deserialize, Clone)]
pub struct DNN {
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

impl DNN {
    pub fn new(substrate: Arc<Substrate>, d_in: usize, d_out: usize, layers: Vec<usize>) -> DNN {
        DNN {
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

    pub fn dynamic(d_in: usize, d_out: usize, breadth: Range<usize>, depth: Range<usize>) -> DNN {
        let mut rng = thread_rng();
        let depth = rng.gen_range(depth);
        let layers = (0..depth)
            .map(|_| rng.gen_range(breadth.clone()))
            .collect::<Vec<usize>>();

        DNN::new(Arc::new(Substrate::blank()), d_in, d_out, layers)
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

    pub fn gather(&mut self) -> &mut Self {
        for layer in self.web.iter_mut() {
            layer.gather(&self.substrate);
        }
        self
    }

    pub fn dump(&mut self) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn load(serialized: &Vec<u8>) -> Result<DNN, Box<dyn Error>> {
        Ok(bincode::deserialize(serialized)?)
    }
}

impl Manifold for DNN {
    fn weave(&mut self) -> &mut Self {
        let mut x_shape = (1, 1, self.d_in);
        let mut w_shape: (usize, usize);
        let mut b_shape: usize;
        let mut p_dim = self.d_in;

        for layer_size in self.layers.iter() {
            w_shape = (p_dim, *layer_size);
            b_shape = w_shape.1;

            self.web.push(Dense::new(
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

        self.web.push(Dense::new(
            self.substrate.size,
            x_shape,
            w_shape,
            b_shape,
            Activations::Identity,
        ));
        self
    }

    fn forward(&mut self, mut x: Array3<f64>) -> Array3<f64> {
        for layer in self.web.iter_mut() {
            x = layer.forward(x);
        }
        x
    }

    fn backwards(
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

            layer.shift_bias(&b_link_reshaped.remove_axis(Axis(1)));
            layer.assign_grad_b(b_grad_reshaped.remove_axis(Axis(1)));
            layer.gather(&self.substrate);

            match self.gradient_retention {
                GradientRetention::Zero => {
                    layer.assign_grad_b(Array1::zeros(grad_b_dim));
                    layer.assign_grad_w(Array2::zeros(grad_w_dim));
                }
                GradientRetention::Roll => (),
            }
        }
    }

    fn get_loss_fn(&mut self) -> Rc<dyn Loss> {
        self.loss.wake()
    }
}
