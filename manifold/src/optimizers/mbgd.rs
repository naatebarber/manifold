use ndarray::Array1;
use plotly::{Bar, Plot};
use rand::{prelude::*, thread_rng};

use super::Hyper;
use crate::nn::fc::Manifold;

pub struct MiniBatchGradientDescent<'a> {
    manifold: &'a mut Manifold,
    hyper: Hyper,
    early_terminate: Box<dyn Fn(&Vec<f64>) -> bool>,
    verbose: bool,
    pub losses: Vec<f64>,
}

impl MiniBatchGradientDescent<'_> {
    pub fn new(manifold: &mut Manifold) -> MiniBatchGradientDescent {
        MiniBatchGradientDescent {
            manifold,
            hyper: Hyper::new(),
            early_terminate: Box::new(|_| false),
            losses: vec![],
            verbose: false,
        }
    }

    pub fn override_hyper(&mut self, hyper: Hyper) -> &mut Self {
        self.hyper = hyper;
        self
    }

    pub fn verbose(&mut self) -> &mut Self {
        self.verbose = true;
        self
    }

    pub fn set_learning_rate(&mut self, rate: f64) -> &mut Self {
        self.hyper.learning_rate = rate;
        self
    }

    pub fn set_decay(&mut self, decay: f64) -> &mut Self {
        self.hyper.decay = decay;
        self
    }

    pub fn set_patience(&mut self, patience: usize) -> &mut Self {
        self.hyper.patience = patience;
        self
    }

    pub fn set_min_delta(&mut self, min_delta: f64) -> &mut Self {
        self.hyper.min_delta = min_delta;
        self
    }

    pub fn until(&mut self) -> &mut Self {
        let patience = self.hyper.patience.clone();
        let min_delta = self.hyper.min_delta.clone();

        let early_terminate = move |losses: &Vec<f64>| {
            let mut deltas: Vec<f64> = vec![];
            let len = losses.len();

            if patience + 2 > len {
                return false;
            }

            for i in ((len - patience)..len).rev() {
                let c = losses[i];
                let c2 = losses[i - 1];

                let delta = c2 - c;
                deltas.push(delta);
            }

            let avg_delta = deltas.iter().fold(0., |a, v| a + *v) / deltas.len() as f64;

            println!("avg delta {}", avg_delta);

            if avg_delta < min_delta {
                return true;
            }

            return false;
        };

        self.early_terminate = Box::new(early_terminate);
        self
    }

    pub fn until_some(
        &mut self,
        early_terminate: impl Fn(&Vec<f64>) -> bool + 'static,
    ) -> &mut Self {
        self.early_terminate = Box::new(early_terminate);
        self
    }

    pub fn set_epochs(&mut self, epochs: usize) -> &mut Self {
        self.hyper.epochs = epochs;
        self
    }

    pub fn set_sample_size(&mut self, sample_size: usize) -> &mut Self {
        self.hyper.sample_size = sample_size;
        self
    }

    pub fn step(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> &mut Self {
        let xy = x
            .into_iter()
            .zip(y.into_iter())
            .collect::<Vec<(Vec<f64>, Vec<f64>)>>();
        let mut rng = thread_rng();

        for epoch in 0..self.hyper.epochs {
            let sample = xy
                .choose_multiple(&mut rng, self.hyper.sample_size)
                .collect::<Vec<&(Vec<f64>, Vec<f64>)>>();
            let mut total_loss: Vec<f64> = vec![];

            for &xy in sample.iter() {
                let (x, y) = xy.clone();

                let y_pred = self.manifold.forward(x);
                total_loss.push(
                    self.manifold
                        .loss
                        .wake()
                        .a(y_pred.clone(), Array1::from(y.clone())),
                );
                self.manifold.backwards(
                    y_pred,
                    y,
                    self.manifold.loss.wake(),
                    self.hyper.learning_rate,
                );
            }

            self.hyper.learning_rate *= self.hyper.decay;

            let ct = total_loss.len() as f64;
            let avg_loss = total_loss.into_iter().fold(0., |a, v| a + v) / ct;
            self.losses.push(avg_loss);

            if (self.early_terminate)(&self.losses) {
                if self.verbose {
                    println!("Early termination condition met.");
                }

                break;
            }

            if self.verbose {
                println!("({}/{}) Loss = {}", epoch, self.hyper.epochs, avg_loss);
            }
        }

        self
    }

    pub fn loss_graph(&mut self) -> &mut Self {
        let mut plot = Plot::new();

        let x = (0..self.losses.len()).collect();

        let trace = Bar::new(x, self.losses.clone());
        plot.add_trace(trace);
        plot.show();

        self
    }
}
