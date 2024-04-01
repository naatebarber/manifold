use std::fmt::Debug;
use std::rc::Rc;

use ndarray::Array1;

pub trait Loss {
    fn a(&self, pred: Array1<f64>, target: Array1<f64>) -> f64;
    fn d(&self, pred: Array1<f64>, target: Array1<f64>) -> Array1<f64>;
}

impl Debug for dyn Loss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LossFn")
    }
}

pub struct MSE;

impl MSE {
    pub fn new() -> Rc<MSE> {
        Rc::new(MSE)
    }
}

impl Loss for MSE {
    fn a(&self, pred: Array1<f64>, target: Array1<f64>) -> f64 {
        pred.iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .fold(0., |a, v| a + v)
            / pred.len() as f64
    }

    fn d(&self, pred: Array1<f64>, target: Array1<f64>) -> Array1<f64> {
        pred.iter()
            .zip(target.iter())
            .map(|(p, t)| ((p - t) * 2.) / pred.len() as f64)
            .collect::<Array1<f64>>()
    }
}

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Rc<BinaryCrossEntropy> {
        Rc::new(BinaryCrossEntropy)
    }
}

impl Loss for BinaryCrossEntropy {
    fn a(&self, pred: Array1<f64>, target: Array1<f64>) -> f64 {
        pred.iter()
            .zip(target.iter())
            .map(|(p, t)| {
                // Avoid log(0) which is undefined
                let p_clipped = p.clamp(1e-9, 1.0 - 1e-9);
                let v = t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln();
                println!("{}", v);

                v
            })
            .sum::<f64>()
            / (pred.len() as f64)
    }

    fn d(&self, pred: Array1<f64>, target: Array1<f64>) -> Array1<f64> {
        pred.iter()
            .zip(target.iter())
            .map(|(p, t)| {
                // Avoid division by 0
                let p_clipped = p.clamp(1e-9, 1.0 - 1e-9);
                (-t / p_clipped + (1.0 - t) / (1.0 - p_clipped)) / pred.len() as f64
            })
            .collect::<Array1<f64>>()
    }
}
