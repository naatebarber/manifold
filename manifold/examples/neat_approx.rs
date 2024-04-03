use std::collections::VecDeque;
use std::error::Error;

use manifold::nn::trainer::Hyper;
use manifold::Neat;
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

fn main() -> Result<(), Box<dyn Error>> {
    let (_x, _y) = gen_xy(1000000);
    let x = VecDeque::from(_x);
    let y = VecDeque::from(_y);

    let substrate = Substrate::new(100000, -0.0..1.0).share();

    let mut neat = Neat::new(1, 1)?;

    neat.set_breadth(5..30)
        .set_depth(2..20)
        .set_arc_substrate(substrate)
        .set_arch_epochs(1000)
        .set_retain(1)
        .set_sample_window(10000)
        .set_chunk_size(1000)
        .set_chunks_per_generation(5)
        .set_hyper(Hyper {
            epochs: 1000,
            sample_size: 10,
            learning_rate: 0.0001,
            decay: 0.999,
        });

    let mut nn = neat.sift(x, y)?.pop().unwrap();

    let (x, y) = gen_xy(10);
    let testxy = x.into_iter().zip(y.into_iter());

    for (x, y) in testxy.into_iter() {
        let y_pred = nn.forward(x.clone());
        let v = y_pred.to_vec();

        println!("{:?} =?= {:?}", v, y);
    }

    Ok(())
}
