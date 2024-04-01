use core::fmt::Debug;
use std::collections::VecDeque;
use std::ops::Range;
use std::rc::Rc;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::{prelude::*, thread_rng, Rng};

use crate::activation::{Activation, Transparent};
use crate::loss::{Loss, MSE};
use crate::substrate::Substrate;

#[derive(Debug)]
pub struct Layer {
    pub x: Array2<f64>,
    pub wi: Array2<usize>,
    pub bi: Array2<usize>,
    pub w: Array2<f64>,
    pub b: Array2<f64>,
    pub d_x: Array2<f64>,
    pub activation: Rc<dyn Activation>,
}

impl Layer {
    pub fn new(
        pool_size: usize,
        x_shape: (usize, usize),
        w_shape: (usize, usize),
        b_shape: (usize, usize),
        activation: Rc<dyn Activation>,
    ) -> Layer {
        Layer {
            x: Array2::zeros(x_shape),
            wi: Array2::random(w_shape, Uniform::new(0, pool_size)),
            bi: Array2::random(b_shape, Uniform::new(0, pool_size)),
            w: Array2::zeros(w_shape),
            b: Array2::zeros(b_shape),
            d_x: Array2::zeros(w_shape),
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

    pub fn shift_bias(&mut self, m: usize, n: usize, new_bi: usize, substrate: &Substrate) {
        self.bi[[m, n]] = new_bi;
        self.b[[m, n]] = substrate[new_bi].get()
    }

    pub fn forward(&mut self, x: Array2<f64>) -> Array2<f64> {
        self.x = x.clone();
        let z = x.dot(&self.w) + &self.b;
        let a_z = z.map(|x| self.activation.a(*x));
        let d_z = z.map(|x| self.activation.d(*x));
        let d_x = d_z.dot(&self.w.t());
        self.d_x = d_x;
        a_z
    }

    pub fn backward(&mut self, grad_output: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        println!("grad output {:?} d_x {:?}", grad_output.shape(), self.d_x.shape());
        let grad_z = grad_output * &self.d_x;

        println!("grad z {:?} w {:?}", grad_z.shape(), self.w.shape());
        let grad_input = grad_z.dot(&self.w);

        println!("x {:?} grad_z {:?}", self.x.shape(), grad_z.t().shape());
        let grad_w = self.x.dot(&grad_z.t());

        let grad_b = grad_z.sum_axis(Axis(0)).insert_axis(Axis(0));
        println!("{:?}", grad_b.shape());

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

    pub fn weave(&mut self, activation: Rc<dyn Activation>) -> &mut Self {
        let mut x_shape = (1, self.d_in);
        let mut w_shape: (usize, usize);
        let mut b_shape: (usize, usize);
        let mut p_dim = self.d_in;

        for layer_size in self.layers.iter() {
            w_shape = (p_dim, *layer_size);
            b_shape = (x_shape.0, w_shape.1);

            self.web.push(Layer::new(
                self.pool_size,
                x_shape,
                w_shape,
                b_shape,
                Rc::clone(&activation),
            ));
            p_dim = *layer_size;
            x_shape = b_shape;
        }

        let transparent_out = Transparent::new();
        let w_shape = (p_dim, self.d_out);
        let b_shape = (x_shape.0, w_shape.1);

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
            println!("{:?}", x.shape());
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

        println!("Loss grad done.");

        let grad_output_shape = (grad_output_i.len(), 1);
        let mut grad_output = grad_output_i.into_shape(grad_output_shape).unwrap();

        println!("Loss grad reshape done.");

        for (i, layer) in self.web.iter_mut().rev().enumerate() {
            grad_output = layer.backward(grad_output, learning_rate);
            println!("{} Layer grad done.", i);
        }
    }

    pub fn train(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) {
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
                println!("Forward done.");
                total_loss.push(self.loss.a(y_pred.clone(), Array1::from(y.clone())));
                println!("Loss done.");
                self.backwards(y_pred, y, Rc::clone(&self.loss), self.learning_rate);
                println!("Backward done.");
            }

            let ct = total_loss.len() as f64;
            let avg_loss = total_loss.into_iter().fold(0., |a, v| a + v) / ct;

            println!("({}/{}) Loss = {}", epoch, self.epochs, avg_loss);
        }
    }
}
