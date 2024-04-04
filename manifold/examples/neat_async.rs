use std::error::Error;

use manifold::async_neat::Neat;
use manifold::nn::trainer::Hyper;
use manifold::Substrate;
use rand::{prelude::*, thread_rng};

fn generator() -> (Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();
    let mutator = |x: f64| x.powi(2);

    let xv: f64 = rng.gen_range(0.0..1.0);

    (vec![xv], vec![mutator(xv)])
}

fn main() -> Result<(), Box<dyn Error>> {
    let substrate = Substrate::new(100000, -0.0..1.0).share();

    let mut neat = Neat::new(1, 1)?;

    neat.set_breadth(5..30)
        .set_depth(2..20)
        .set_arc_substrate(substrate)
        .set_arch_epochs(100)
        .set_retain(3)
        .set_chunk_window(100)
        .set_chunk_size(1000)
        .set_chunks_per_generation(10)
        .set_live_dataset(generator)
        .set_hyper(Hyper {
            epochs: 1000,
            sample_size: 10,
            learning_rate: 0.0001,
            decay: 0.999,
            patience: 300,
            min_delta: 0.001,
            early_stopping: false,
        });

    let manifolds = neat.sift()?;

    println!("\nNEAT evolved architecture:");
    for manifold in manifolds.iter() {
        println!("Layers {:?}", manifold.layers);
    }

    Ok(())
}
