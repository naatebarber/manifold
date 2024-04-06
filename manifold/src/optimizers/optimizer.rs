use ndarray::Array2;

#[derive(Clone)]
pub struct Hyper {
    pub epochs: usize,
    pub sample_size: usize,
    pub learning_rate: f64,
    pub decay: f64,
    pub patience: usize,
    pub min_delta: f64,
    pub early_stopping: bool,
}

impl Hyper {
    pub fn new() -> Hyper {
        Hyper {
            learning_rate: 0.001,
            decay: 1.,
            epochs: 1000,
            sample_size: 10,
            patience: 0,
            min_delta: 0.,
            early_stopping: false,
        }
    }
}

pub trait Optimizer {
    fn override_hyper(&mut self, hyper: Hyper) -> &mut Self;
    fn step(&mut self, x: Array2<f64>, y: Array2<f64>) -> &mut Self;
}
