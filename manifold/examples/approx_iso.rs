use manifold::nn::fc3_iso::Manifold;
use manifold::optimizers::mbgd3_iso::MiniBatchGradientDescent;
use rand::{prelude::*, thread_rng};

type Dataset = (Vec<Vec<f64>>, Vec<Vec<f64>>);

fn gen_xy(size: usize) -> Dataset {
    let mut rng = thread_rng();

    let mutator = |x: f64| x.exp() - x.powi(3);

    let (mut x, mut y): Dataset = (vec![], vec![]);

    for _ in 0..size {
        let xv: f64 = rng.gen_range(0.0..1.0);
        x.push(vec![xv.clone()]);
        y.push(vec![mutator(xv)]);
    }

    (x, y)
}

fn main() {
    let (x, y) = gen_xy(10000);

    let mut nn = Manifold::new(1, 1, vec![4, 4]);
    nn.weave();

    let three = MiniBatchGradientDescent::prepare(x, y);

    let mut trainer = MiniBatchGradientDescent::new(&mut nn);
    trainer
        .set_learning_rate(0.0001)
        .set_decay(0.999)
        .set_epochs(5000)
        .set_sample_size(100)
        .set_patience(100)
        .set_min_delta(0.02)
        .verbose()
        .train(&three.0, &three.1)
        .loss_graph();
}
