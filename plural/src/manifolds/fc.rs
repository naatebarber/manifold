use core::fmt::Debug;
use std::collections::VecDeque;
use std::ops::Range;
use std::rc::Rc;

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::{thread_rng, Rng};

use crate::activation::{Activation, Transparent};
use crate::substrate::Substrate;

#[derive(Debug)]
pub struct Layer {
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
        w_shape: (usize, usize),
        b_shape: (usize, usize),
        activation: Rc<dyn Activation>,
    ) -> Layer {
        Layer {
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
        let z = x.dot(&self.w) + &self.b;
        let a_z = z.map(|x| self.activation.a(*x));
        let d_z = z.map(|x| self.activation.d(*x));
        let d_x = d_z.dot(&self.w.t());
        self.d_x = d_x;
        a_z
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
    learning_rate: f64,
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
            learning_rate: 0.,
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
            learning_rate: 0.,
        }
    }

    pub fn verbose(&mut self) -> &mut Self {
        self.verbose = true;
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

    pub fn forward(&mut self, xv: Vec<f64>) -> Array2<f64> {
        let mut x = self.prepare(xv);
        for layer in self.web.iter_mut() {
            println!("{:?}", x.shape());
            x = layer.forward(x);
        }
        x
    }
}
