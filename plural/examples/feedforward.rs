use plural::activation::Relu;
use plural::manifolds::fc::Manifold;
use plural::Mote;

fn main() {
    let (substrate, pool_size) = Mote::substrate(10000, -1.0..1.0);
    let mut nn = Manifold::new(pool_size, 4, 2, vec![10, 20]);
    let y_pred = nn
        .weave(Relu::new())
        .gather(&substrate)
        .forward(vec![0.8, 0.2, 0.3, 0.1]);

    println!("{:?}", y_pred);
}
