mod activation;
pub mod f;
pub mod layers;
mod loss;
mod loss3;
pub mod manifold;
pub mod neat;
pub mod optimizers;
pub mod substrate;
mod util;

pub use activation::Activations;
pub use loss::Losses;
pub use loss3::Losses as Losses3;
pub use manifold as nn;
pub use manifold::fc_single::GradientRetention;
pub use neat::{async_neat, sync_neat};
pub use substrate::Substrate;

// pub type Population = VecDeque<Arc<Mutex<Manifold>>>;
pub type Dataset = (Vec<Vec<f64>>, Vec<Vec<f64>>);
pub type DatasetReference<'a> = (Vec<&'a Vec<f64>>, Vec<&'a Vec<f64>>);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(true, true);
    }
}
