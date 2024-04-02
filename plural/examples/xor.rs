use plural::activation::Relu;
use plural::activation::Softmax;
use plural::f;
use plural::loss::CategoricalCrossEntropy;
use plural::loss::SoftmaxCrossEntropy;
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
    let (train_x, train_y) = gen_training_data(500000);
    let (test_x, test_y) = gen_training_data(100);

    let (substrate, pool_size) = Mote::substrate(10000, -0.0..1.0);
    let mut nn = Manifold::new(pool_size, 2, 2, vec![16]);
    nn.set_hidden_activation(Relu::new())
        .weave()
        .gather(&substrate)
        .set_learning_rate(0.00001)
        .set_epochs(1000)
        .set_loss(SoftmaxCrossEntropy::new())
        .set_sample_size(10)
        .train(train_x, train_y);

    let testxy = test_x
        .iter()
        .zip(test_y.iter())
        .collect::<Vec<(&Vec<f64>, &Vec<f64>)>>();

    let mut correct = 0;
    let mut total = 0;

    for (x, y) in testxy.into_iter() {
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
