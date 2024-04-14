use manifold::nn::types::{GradientRetention, Manifold};
use manifold::nn::DNN;
use manifold::optimizers::MiniBatchGradientDescent;
use manifold::util::as_tensor;
use manifold::Losses;

use manifold::Activations;
use manifold::Substrate;

use rand::{prelude::*, thread_rng};

fn gen_training_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();

    let mutator = |x: f64| x.exp() * x.powi(3);

    let num = rng.gen_range(0.0..1.0);

    (vec![num], vec![mutator(num)])
}

fn main() {
    let (mut x, mut y) = (vec![], vec![]);
    for _ in 0..5000 {
        let (_x, _y) = gen_training_data();
        x.push(_x);
        y.push(_y);
    }

    let (mut tx, mut ty) = (vec![], vec![]);
    for _ in 0..50 {
        let (_x, _y) = gen_training_data();
        tx.push(_x);
        ty.push(_y);
    }

    let substrate = Substrate::new(10000, 0.0..1.0).share();

    let mut nn = DNN::new(substrate, 1, 1, vec![8, 12]);
    nn.set_hidden_activation(Activations::Relu)
        .set_loss(Losses::MeanSquaredError)
        .set_gradient_retention(GradientRetention::Roll)
        .weave()
        .gather();

    let x = as_tensor(x);
    let y = as_tensor(y);

    let mut trainer = MiniBatchGradientDescent::new(&mut nn);
    trainer
        .set_learning_rate(0.1)
        .set_decay(0.999)
        .set_epochs(1000)
        .set_sample_size(100)
        .verbose()
        .train(&x, &y)
        .loss_graph();
}
