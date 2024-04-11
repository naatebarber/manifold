use std::error::Error;
use std::rc::Rc;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Axis};

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::layers::types::{Layer, Layers};
use crate::layers::Dense;
use crate::loss::{Loss, Losses};
use crate::substrate::Substrate;

use super::types::{GradientRetention, Manifold};


pub type LayerDefinition = (usize, Activations, Layers);
pub type Web = Vec<Box<dyn Layer>>;

#[derive(Serialize, Deserialize)]
pub struct Composable {
    #[serde(skip)]
    substrate: Arc<Substrate>,
    d_in: usize,
    d_out: usize,
    #[serde(skip)]
    web: Web,
    layers: Vec<LayerDefinition>,
    hidden_activation: Activations,
    verbose: bool,
    gradient_retention: GradientRetention,
    pub loss: Losses,
}

impl Composable {
    pub fn new(
        substrate: Arc<Substrate>,
        d_in: usize,
        d_out: usize,
    ) -> Composable {
        Composable {
            substrate,
            d_in,
            d_out,
            layers: Vec::new(),
            web: Web::new(),
            hidden_activation: Activations::Relu,
            verbose: false,
            loss: Losses::MeanSquaredError,
            gradient_retention: GradientRetention::Zero,
        }
    }

    pub fn layer(&mut self, size: usize, activation: Activations, layer: Layers) -> &mut Self {
        let ld: LayerDefinition = (size, activation, layer);
        self.layers.push(ld);
        self
    }

    pub fn set_substrate(&mut self, substrate: Arc<Substrate>) -> &mut Self {
        self.substrate = substrate;
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

    pub fn load(serialized: &Vec<u8>) -> Result<Composable, Box<dyn Error>> {
        Ok(bincode::deserialize::<Composable>(serialized)?)
    }
}

impl Manifold for Composable {
    fn weave(&mut self) -> &mut Self {
        let mut x_shape = (1, 1, self.d_in);
        let mut w_shape: (usize, usize);
        let mut b_shape: usize;
        let mut p_dim = self.d_in;

        for layer_definition in self.layers.iter() {
            let (size, activation, layer) = layer_definition;
            w_shape = (p_dim, *size);
            b_shape = w_shape.1;

            self.web.push(Layers::wake(*layer, self.substrate.size, x_shape, w_shape, b_shape, *activation));

            p_dim = *size;
            x_shape = (1, 1, w_shape.1);
        }

        let w_shape = (p_dim, self.d_out);
        let b_shape = w_shape.1;

        self.web.push(Box::new(Dense::new(
            self.substrate.size,
            x_shape,
            w_shape,
            b_shape,
            Activations::Identity,
        )));

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

            let (mut grad_w, grad_b) = layer.gradients();
            let (mut wi, bi) = layer.gradient_bindings();

            let grad_b_dim = grad_b.raw_dim();
            let grad_w_dim = grad_w.raw_dim();

            let mut b_grad_reshaped = grad_b.insert_axis(Axis(1));
            let mut b_link_reshaped = bi.insert_axis(Axis(1));

            self.substrate
                .highspeed(&mut grad_w, &mut wi, learning_rate);
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
