use core::fmt::Debug;
use std::collections::VecDeque;
use std::ops::Range;
use std::rc::Rc;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotly::{Bar, Plot};
use rand::{prelude::*, thread_rng, Rng};

use crate::activation::{Activation, Transparent};
use crate::loss::{Loss, MSE};
use crate::substrate::Substrate;

#[derive(Debug)]
pub struct Layer {
    pub x: Array2<f64>,
    pub wi: Array2<usize>,
    pub bi: Array1<usize>,
    pub w: Array2<f64>,
    pub b: Array1<f64>,
    pub d_z: Array2<f64>,
    pub activation: Rc<dyn Activation>,
}

impl Layer {
    pub fn new(
        pool_size: usize,
        x_shape: (usize, usize),
        w_shape: (usize, usize),
        b_shape: usize,
        activation: Rc<dyn Activation>,
    ) -> Layer {
        Layer {
            x: Array2::zeros(x_shape),
            wi: Array2::random(w_shape, Uniform::new(0, pool_size)),
            bi: Array::random(b_shape, Uniform::new(0, pool_size)),
            w: Array2::zeros(w_shape),
            b: Array::zeros(b_shape),
            d_z: Array2::zeros(w_shape),
            activation,
        }
    }

    pub fn gather(&mut self, substrate: &Substrate) {
        self.w = self.wi.map(|ix| substrate[*ix].get())
    }

    pub fn shift_weight(&mut self, m: usize, n: usize, new_wi: usize, substrate: &Substrate) {
        self.wi[[m, n]] = new_wi;
        self.w[[m, n]] = substrate[new_wi].get()
    }

    pub fn shift_bias(&mut self, m: usize, new_bi: usize, substrate: &Substrate) {
        self.bi[m] = new_bi;
        self.b[m] = substrate[new_bi].get()
    }

    pub fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.x = x.clone();
        let z = x.dot(&self.w) + &self.b;
        let a_z = z.map(|x| self.activation.a(*x));
        let d_z = z.map(|x| self.activation.d(*x));
        self.d_z = d_z;
        a_z
    }

    pub fn backward(&mut self, grad_output: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let grad_z = grad_output * &self.d_z;
        let grad_input = grad_z.dot(&self.w.t());
        let grad_w = self.x.t().dot(&grad_z);
        let grad_b = grad_z.sum_axis(Axis(0));

        self.w -= &(grad_w * learning_rate);
        self.b -= &(grad_b * learning_rate);

        grad_input
    }
}

pub type LayerSchema = Vec<usize>;
pub type Web = Vec<Layer>;

pub struct Manifold {
    pool_size: usize,
    d_in: usize,
    d_out: usize,
    layers: LayerSchema,
    web: Web,
    verbose: bool,
    loss: Rc<dyn Loss>,
    epochs: usize,
    learning_rate: f64,
    sample_size: usize,
    losses: Vec<f64>,
}

impl Manifold {
    pub fn new(pool_size: usize, d_in: usize, d_out: usize, layers: Vec<usize>) -> Manifold {
        Manifold {
            pool_size,
            d_in,
            d_out,
            layers,
            web: Web::new(),
            verbose: false,
            loss: MSE::new(),
            epochs: 1000,
            learning_rate: 0.05,
            sample_size: 10,
            losses: vec![],
        }
    }

    pub fn dynamic(
        pool_size: usize,
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

        Manifold {
            pool_size,
            d_in,
            d_out,
            web: Web::new(),
            layers,
            verbose: false,
            loss: MSE::new(),
            epochs: 1000,
            learning_rate: 0.05,
            sample_size: 1,
            losses: vec![],
        }
    }

    pub fn verbose(&mut self) -> &mut Self {
        self.verbose = true;
        self
    }

    pub fn set_loss(&mut self, loss: Rc<dyn Loss>) -> &mut Self {
        self.loss = loss;
        self
    }

    pub fn set_epochs(&mut self, epochs: usize) -> &mut Self {
        self.epochs = epochs;
        self
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn set_sample_size(&mut self, sample_size: usize) -> &mut Self {
        self.sample_size = sample_size;
        self
    }

    pub fn weave(&mut self, activation: Rc<dyn Activation>) -> &mut Self {
        let mut x_shape = (1, self.d_in);
        let mut w_shape: (usize, usize);
        let mut b_shape: usize;
        let mut p_dim = self.d_in;

        for layer_size in self.layers.iter() {
            w_shape = (p_dim, *layer_size);
            b_shape = w_shape.1;

            self.web.push(Layer::new(
                self.pool_size,
                x_shape,
                w_shape,
                b_shape,
                Rc::clone(&activation),
            ));
            p_dim = *layer_size;
            x_shape = (1, w_shape.1);
        }

        let transparent_out = Transparent::new();
        let w_shape = (p_dim, self.d_out);
        let b_shape = w_shape.1;

        self.web.push(Layer::new(
            self.pool_size,
            x_shape,
            w_shape,
            b_shape,
            transparent_out,
        ));
        self
    }

    pub fn gather(&mut self, substrate: &Substrate) -> &mut Self {
        for layer in self.web.iter_mut() {
            layer.gather(&substrate)
        }
        self
    }

    fn prepare(&self, x: Vec<f64>) -> Array2<f64> {
        let l = x.len();
        let mut xvd: VecDeque<f64> = VecDeque::from(x);
        Array2::zeros((1, l)).mapv_into(|_| xvd.pop_front().unwrap())
    }

    pub fn forward(&mut self, xv: Vec<f64>) -> Array1<f64> {
        let mut x = self.prepare(xv);
        for layer in self.web.iter_mut() {
            x = layer.forward(x);
        }
        let shape = x.len();
        x.into_shape(shape).unwrap()
    }

    pub fn backwards(
        &mut self,
        y_pred: Array1<f64>,
        y: Vec<f64>,
        loss: Rc<dyn Loss>,
        learning_rate: f64,
    ) {
        let y_target = Array1::from(y);
        let grad_output_i = loss.d(y_pred, y_target);

        let grad_output_shape = (1, grad_output_i.len());
        let mut grad_output = grad_output_i.into_shape(grad_output_shape).unwrap();

        for layer in self.web.iter_mut().rev() {
            grad_output = layer.backward(grad_output, learning_rate);
        }
    }

    pub fn train(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> &mut Self {
        let xy = x
            .into_iter()
            .zip(y.into_iter())
            .collect::<Vec<(Vec<f64>, Vec<f64>)>>();
        let mut rng = thread_rng();

        for epoch in 0..self.epochs {
            let sample = xy
                .choose_multiple(&mut rng, self.sample_size)
                .collect::<Vec<&(Vec<f64>, Vec<f64>)>>();
            let mut total_loss: Vec<f64> = vec![];

            for &xy in sample.iter() {
                let (x, y) = xy.clone();

                let y_pred = self.forward(x);
                total_loss.push(self.loss.a(y_pred.clone(), Array1::from(y.clone())));
                self.backwards(y_pred, y, Rc::clone(&self.loss), self.learning_rate);
            }

            let ct = total_loss.len() as f64;
            let avg_loss = total_loss.into_iter().fold(0., |a, v| a + v) / ct;

            self.losses.push(avg_loss);

            println!("({}/{}) Loss = {}", epoch, self.epochs, avg_loss);
        }

        self
    }

    pub fn loss_graph(&mut self) -> &mut Self {
        let mut plot = Plot::new();

        let x = (0..self.losses.len()).collect();

        let trace = Bar::new(x, self.losses.clone());
        plot.add_trace(trace);
        plot.write_html("loss.html");
        plot.show();

        self
    }
}
