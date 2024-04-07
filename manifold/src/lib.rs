mod activation;
pub mod f;
pub mod layers;
mod loss;
pub mod manifold;
pub mod neat;
pub mod optimizers;
pub mod substrate;
pub mod util;

pub use activation::Activations;
pub use loss::Losses;
pub use manifold as nn;
pub use neat::Neat;
pub use substrate::Substrate;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(true, true);
    }
}
