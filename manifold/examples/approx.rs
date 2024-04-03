use manifold::nn::fc::Manifold;
use manifold::Substrate;
use rand::{prelude::*, thread_rng};

type Dataset = (Vec<Vec<f64>>, Vec<Vec<f64>>);

fn gen_xy(size: usize) -> Dataset {
    let mut rng = thread_rng();

    let mutator = |x: f64| x.powi(2);

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

    let substrate = Substrate::new(100000, -0.0..1.0).share();

    let mut nn = Manifold::new(substrate, 1, 1, vec![4, 4]);
    nn.weave()
        .gather()
        .set_gradient_retention(manifold::GradientRetention::Zero)
        .get_trainer()
        .set_learning_rate(0.0001)
        .set_decay(0.999)
        .set_epochs(500)
        .set_sample_size(100)
        .until(100, 0.02)
        .train(x, y)
        .loss_graph();

    let (x, y) = gen_xy(10);
    let testxy = x.into_iter().zip(y.into_iter());

    for (x, y) in testxy.into_iter() {
        let y_pred = nn.forward(x.clone());
        let v = y_pred.to_vec();

        println!("{:?} =?= {:?}", v, y);
    }
}
