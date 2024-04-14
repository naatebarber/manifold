use manifold::nn::types::Manifold;
use manifold::nn::Composable;
use manifold::util;
use manifold::Activations;
use manifold::Layers;
use manifold::Substrate;

fn main() {
    let sust = Substrate::new(1000, 0.0..1.0).share();
    let mut nn = Composable::new(sust, 2, 2);

    nn.layer(4, Activations::Relu, Layers::Dense)
        .layer(5, Activations::Relu, Layers::Dense)
        .weave();

    let x = util::as_tensor(vec![vec![3., 2.]]);
    let datay = util::as_tensor(vec![vec![4., 5.]]);

    let val = nn.forward(x);
    println!("Val {:?}", val);
}
