use plural::{activation::Relu, loss::BinaryCrossEntropy};
use plural::manifolds::fc::Manifold;
use plural::Mote;
use rand::{prelude::*, thread_rng};

fn gen_training_data(size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut x: Vec<Vec<f64>> = vec![];
    let mut y: Vec<Vec<f64>> = vec![];
    let mut rng = thread_rng();

    let classes: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![0., 1.], vec![1., 0.]),
        (vec![1., 1.], vec![0., 1.]),
        (vec![1., 0.], vec![1., 0.]),
        (vec![0., 0.], vec![0., 1.]),
    ];

    for _ in 0..size {
        let data = classes.choose(&mut rng).unwrap();
        x.push(data.0.clone());
        y.push(data.1.clone());
    }

    (x, y)
}

fn main() {
    let (train_x, train_y) = gen_training_data(5000);
    let (test_x, test_y) = gen_training_data(100);

    let (substrate, pool_size) = Mote::substrate(10000, -1.0..1.0);
    let mut nn = Manifold::new(pool_size, 2, 2, vec![10, 20]);
    let y_pred = nn
        .weave(Relu::new())
        .gather(&substrate)
        .set_learning_rate(0.05)
        .set_epochs(1000)
        .set_loss(BinaryCrossEntropy::new())
        .train(train_x, train_y);

    println!("{:?}", y_pred);
}
