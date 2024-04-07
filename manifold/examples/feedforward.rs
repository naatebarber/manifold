use manifold::nn::fc_single::Manifold;
use manifold::Substrate;

fn main() {
    let substrate = Substrate::new(10000, -1.0..1.0).share();
    let mut nn = Manifold::new(substrate, 4, 2, vec![10, 20]);
    let y_pred = nn.weave().gather().forward(vec![0.8, 0.2, 0.3, 0.1]);

    println!("{:?}", y_pred);
}
