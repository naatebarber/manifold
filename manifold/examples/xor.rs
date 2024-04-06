use manifold::f;
use manifold::nn::fc::Manifold;
use manifold::Activations;
use manifold::Losses;
use manifold::Substrate;
use rand::{prelude::*, thread_rng};

fn gen_training_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = thread_rng();

    let classes: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![0., 1.], vec![1., 0.]),
        (vec![1., 1.], vec![0., 1.]),
        (vec![1., 0.], vec![1., 0.]),
        (vec![0., 0.], vec![0., 1.]),
    ];

    let data = classes.choose(&mut rng).unwrap();
    (data.0.clone(), data.1.clone())
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

    let mut nn = Manifold::new(substrate, 2, 2, vec![4]);
    nn.set_hidden_activation(Activations::Relu)
        .set_loss(Losses::SoftmaxCrossEntropy)
        .set_gradient_retention(manifold::GradientRetention::Zero)
        .weave()
        .gather()
        .get_trainer()
        .set_learning_rate(0.001)
        .set_decay(0.999)
        .set_epochs(4000)
        .set_sample_size(1)
        .verbose()
        .train(x, y);

    let txy = tx
        .iter()
        .zip(ty.iter())
        .collect::<Vec<(&Vec<f64>, &Vec<f64>)>>();

    let mut correct = 0;
    let mut total = 0;

    for (x, y) in txy.into_iter() {
        let y_pred = nn.forward(x.clone());
        let v = y_pred.to_vec();
        let choice_pred = f::argmax(&v);
        let actual = f::argmax(y);

        total += 1;

        if choice_pred == actual {
            correct += 1;
        }
    }

    println!("Accuracy: {}%", (correct as f64 / total as f64) * 100.);
}
