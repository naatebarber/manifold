use ndarray::{stack, Array1, Array2, Array3, Axis};
use plotly::{Bar, Plot};
use rand::{prelude::*, thread_rng};

use super::Hyper;
use crate::nn::fc3_iso::Manifold;

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

    pub fn prepare(x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> (Array3<f64>, Array3<f64>) {
        let x_a2 = x
            .into_iter()
            .map(|xv| Array1::from(xv).insert_axis(Axis(0)))
            .collect::<Vec<Array2<f64>>>();
        let y_a2 = y
            .into_iter()
            .map(|yv| Array1::from(yv).insert_axis(Axis(0)))
            .collect::<Vec<Array2<f64>>>();

        let x_3 = stack(
            Axis(0),
            &x_a2.iter().map(|ar| ar.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let y_3 = stack(
            Axis(0),
            &y_a2.iter().map(|ar| ar.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        (x_3, y_3)
    }

    pub fn train(&mut self, x: &Array3<f64>, y: &Array3<f64>) -> &mut Self {
        assert_eq!(
            x.shape(),
            y.shape(),
            "X and Y must be of the same shape for training."
        );

        for epoch in 0..self.hyper.epochs {
            let mut indices: Vec<usize> = Vec::with_capacity(self.hyper.sample_size);
            let mut rng = thread_rng();
            for _ in 0..self.hyper.sample_size {
                indices.push(rng.gen_range(0..x.shape()[0]));
            }

            let batch_x_vec = indices
                .iter()
                .map(|ix| x.index_axis(Axis(0), *ix))
                .collect::<Vec<_>>();
            let batch_y_vec = indices
                .iter()
                .map(|ix| y.index_axis(Axis(0), *ix))
                .collect::<Vec<_>>();

            let batch_x: Array3<f64> = stack(Axis(0), &batch_x_vec).unwrap();
            let batch_y: Array3<f64> = stack(Axis(0), &batch_y_vec).unwrap();

            let y_pred = self.manifold.forward(batch_x);
            let y_pred_reshaped = y_pred.remove_axis(Axis(1));

            let y_reshaped = batch_y.remove_axis(Axis(1));

            let loss = self.manifold.loss.wake();
            let a_loss = loss.a(y_pred_reshaped.clone(), y_reshaped.clone());
            let sum_batch_loss = a_loss.sum() / a_loss.len() as f64;

            self.manifold
                .backwards(y_pred_reshaped, y_reshaped, loss, self.hyper.learning_rate);

            self.losses.push(sum_batch_loss);
            self.hyper.learning_rate *= self.hyper.decay;

            if (&self.early_terminate)(&self.losses) {
                println!("Early termination condition met, stopping.");
                break;
            }

            if self.verbose {
                println!(
                    "({}/{}) Loss = {}",
                    epoch, self.hyper.epochs, sum_batch_loss
                );
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
