use std::rc::Rc;

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::loss::Loss;

#[derive(Serialize, Deserialize, Clone)]
pub enum GradientRetention {
    Roll,
    Zero,
}

pub trait Manifold {
    fn weave(&mut self) -> &mut Self;
    fn forward(&mut self, x: Array3<f64>) -> Array3<f64>;
    fn backwards(
        &mut self,
        pred: Array2<f64>,
        target: Array2<f64>,
        loss: Rc<dyn Loss>,
        learning_rate: f64,
    );
    fn get_loss_fn(&mut self) -> Rc<dyn Loss>;
}
