use std::fmt::Debug;
use std::rc::Rc;

use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

pub trait Activation {
    fn a(&self, x: Array2<f64>) -> Array2<f64>;
    fn d(&self, x: Array2<f64>) -> Array2<f64>;
}

impl Debug for dyn Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ActivationFn")
    }
}

pub struct Relu;

impl Relu {
    pub fn new() -> Rc<Relu> {
        Rc::new(Relu)
    }
}

impl Activation for Relu {
    fn a(&self, x: Array2<f64>) -> Array2<f64> {
        x.map(|v| {
            if *v < 0. {
                return 0.;
            }
            *v
        })
    }

    fn d(&self, x: Array2<f64>) -> Array2<f64> {
        x.map(|v| {
            if *v < 0. {
                return 0.;
            }
            return *v;
        })
    }
}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Rc<Softmax> {
        Rc::new(Softmax)
    }
}

impl Activation for Softmax {
    fn a(&self, x: Array2<f64>) -> Array2<f64> {
        let max_vals = x.map_axis(Axis(1), |row| row.max().unwrap().to_owned());
        let exps = (&x - &max_vals.insert_axis(Axis(1))).map(|v| v.exp());
        let sum_exps = exps.sum_axis(Axis(1)).insert_axis(Axis(1));
        let f = exps / &sum_exps;

        f
    }

    fn d(&self, s: Array2<f64>) -> Array2<f64> {
        let row = s.row(0);
        let outer_product = &row.view().into_shape((row.len(), 1)).unwrap() * &row.t();
        let mut mask = Array2::zeros(outer_product.raw_dim());
        mask.diag_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = row[i] * (1. - row[i]));
        let jacobian = mask - outer_product;
        jacobian
    }
}

pub struct Identity;

impl Identity {
    pub fn new() -> Rc<Identity> {
        Rc::new(Identity)
    }
}

impl Activation for Identity {
    fn a(&self, x: Array2<f64>) -> Array2<f64> {
        x
    }

    fn d(&self, x: Array2<f64>) -> Array2<f64> {
        x
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Activations {
    Relu,
    Identity,
}

impl Activations {
    pub fn wake(&self) -> Rc<dyn Activation> {
        match self {
            Activations::Identity => Identity::new(),
            Activations::Relu => Relu::new(),
        }
    }
}
