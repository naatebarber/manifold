use ndarray::Array1;
use plotly::{Bar, Plot};
use rand::{prelude::*, thread_rng};

use super::fc::Manifold;

#[derive(Clone)]
pub struct Hyper {
    epochs: usize,
    sample_size: usize,
    learning_rate: f64,
    decay: f64,
}

impl Hyper {
    pub fn new() -> Hyper {
        Hyper {
            learning_rate: 0.001,
            decay: 1.,
            epochs: 1000,
            sample_size: 10,
        }
    }
}

pub struct Trainer<'a> {
    manifold: &'a mut Manifold,
    hyper: Hyper,
    losses: Vec<f64>,
    early_terminate: Box<dyn Fn(&Vec<f64>) -> bool>,
    verbose: bool,
}

impl Trainer<'_> {
    pub fn new(manifold: &mut Manifold) -> Trainer {
        Trainer {
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

    pub fn until(&mut self, patience: usize, min_delta: f64) -> &mut Self {
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

    pub fn train(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> &mut Self {
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
        plot.write_html("loss.html");
        plot.show();

        self
    }
}
