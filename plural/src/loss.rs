use std::fmt::Debug;
use std::rc::Rc;

use ndarray::Array1;
use ndarray_stats::QuantileExt;

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
        let len = pred.len();
        let diff = pred - target;
        let exp = diff.mapv_into(|x| x.powi(2));
        let sum = exp.sum();
        sum / len as f64
    }

    fn d(&self, pred: Array1<f64>, target: Array1<f64>) -> Array1<f64> {
        let len = pred.len();
        let diff = pred - target;
        let dx = diff.mapv_into(|x| (x * 2.) / len as f64);
        dx
    }
}

pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Rc<CategoricalCrossEntropy> {
        Rc::new(CategoricalCrossEntropy)
    }
}

impl Loss for CategoricalCrossEntropy {
    fn a(&self, pred: Array1<f64>, target: Array1<f64>) -> f64 {
        let log_pred = pred.map(|x| x.ln());
        let prod = log_pred * target;
        -prod.sum()
    }

    fn d(&self, pred: Array1<f64>, target: Array1<f64>) -> Array1<f64> {
        pred - target
    }
}

pub struct SoftmaxCrossEntropy;

impl SoftmaxCrossEntropy {
    pub fn new() -> Rc<SoftmaxCrossEntropy> {
        Rc::new(SoftmaxCrossEntropy)
    }
}

impl Loss for SoftmaxCrossEntropy {
    fn a(&self, pred: Array1<f64>, target: Array1<f64>) -> f64 {
        let max = pred.max().unwrap();
        let exps = pred.map(|v| (v - max).exp());
        let sum_exps = exps.sum();
        let softmax_pred = exps.map(|v| v / sum_exps);

        let log_pred = softmax_pred.map(|x| x.ln());
        let prod = log_pred * target;
        -prod.sum()
    }

    fn d(&self, pred: Array1<f64>, target: Array1<f64>) -> Array1<f64> {
        pred - target
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
