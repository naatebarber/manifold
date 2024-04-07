use std::fmt::Debug;
use std::rc::Rc;

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

pub trait Loss {
    fn a(&self, pred: Array2<f64>, target: Array2<f64>) -> Array1<f64>;
    fn d(&self, pred: Array2<f64>, target: Array2<f64>) -> Array2<f64>;
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
    fn a(&self, pred: Array2<f64>, target: Array2<f64>) -> Array1<f64> {
        assert_eq!(
            pred.shape(),
            target.shape(),
            "Predictions and targets must have the same shape."
        );

        let features = pred.shape()[1];

        let diff = pred - target;
        let exp = diff.mapv_into(|x| x.powi(2));
        let sum = exp.sum_axis(Axis(1));
        let mse = sum.mapv_into(|x| x / features as f64);
        mse
    }

    fn d(&self, pred: Array2<f64>, target: Array2<f64>) -> Array2<f64> {
        let features = pred.shape()[1];

        let diff = pred - target;
        let dx = diff.mapv_into(|x| (x * 2.) / features as f64);
        dx
    }
}

pub struct SoftmaxCrossEntropy;

impl SoftmaxCrossEntropy {
    pub fn new() -> Rc<SoftmaxCrossEntropy> {
        Rc::new(SoftmaxCrossEntropy)
    }
}

impl SoftmaxCrossEntropy {
    pub fn softmax(&self, pred: Array2<f64>) -> Array2<f64> {
        let batch_size = pred.shape()[0];

        let max_mask = pred.map_axis(Axis(1), |axis| *axis.max().unwrap());
        let mask_2d = max_mask.into_shape((batch_size, 1)).unwrap();
        let broadcast_mask = mask_2d.broadcast(pred.raw_dim()).unwrap();

        let applied_max_mask = &pred - &broadcast_mask;

        let exps = applied_max_mask.mapv_into(|x| x.exp());
        let sum_exps = exps.sum_axis(Axis(1));
        let sum_exps_2d = sum_exps.into_shape((pred.nrows(), 1)).unwrap();

        let softmax_pred = exps / &sum_exps_2d;

        softmax_pred
    }
}

impl Loss for SoftmaxCrossEntropy {
    fn a(&self, pred: Array2<f64>, target: Array2<f64>) -> Array1<f64> {
        let softmax_pred = self.softmax(pred);

        let log_pred = softmax_pred.map(|x| x.ln());
        let prod = log_pred * target;
        prod.map_axis(Axis(1), |axis| -axis.sum())
    }

    fn d(&self, pred: Array2<f64>, target: Array2<f64>) -> Array2<f64> {
        let softmax_pred = self.softmax(pred);
        softmax_pred - target
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Losses {
    MeanSquaredError,
    SoftmaxCrossEntropy,
}

impl Losses {
    pub fn wake(&self) -> Rc<dyn Loss> {
        match self {
            Losses::MeanSquaredError => MSE::new(),
            Losses::SoftmaxCrossEntropy => SoftmaxCrossEntropy::new(),
        }
    }
}
