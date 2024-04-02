use plural::f;
use plural::loss::MSE;
use plural::manifolds::fc::Manifold;
use plural::Mote;
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

    let (substrate, pool_size) = Mote::substrate(10000, -0.0..1.0);
    let mut nn = Manifold::new(pool_size, 1, 1, vec![4, 4]);
    nn.weave()
        .gather(&substrate)
        .set_learning_rate(0.00001)
        .set_epochs(10000)
        .set_sample_size(10)
        .train(x, y)
        .loss_graph();

    let (x, y) = gen_xy(1000);
    let testxy = x.into_iter().zip(y.into_iter());

    for (x, y) in testxy.into_iter() {
        let y_pred = nn.forward(x.clone());
        let v = y_pred.to_vec();

        println!("{:?} =?= {:?}", v, y);
    }
}
