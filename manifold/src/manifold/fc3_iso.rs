use std::error::Error;
use std::ops::Range;
use std::rc::Rc;

use ndarray::{Array2, Array3, Axis};
use rand::{thread_rng, Rng};

use serde::{self, Deserialize, Serialize};

use crate::activation::Activations;
use crate::layers::DenseIndependent;
use crate::loss3::{Loss, Losses};

#[derive(Serialize, Deserialize, Clone)]
pub enum GradientRetention {
    Roll,
    Zero,
}

pub type LayerSchema = Vec<usize>;
pub type Web = Vec<DenseIndependent>;

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

            self.web.push(DenseIndependent::new(
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

        self.web.push(DenseIndependent::new(
            x_shape,
            w_shape,
            b_shape,
            Activations::Identity,
        ));
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
