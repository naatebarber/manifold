mod activation;
pub mod optimizers;
pub mod f;
mod loss;
mod loss3;
pub mod manifold;
pub mod neat;
pub mod substrate;
mod util;

pub use activation::Activations;
pub use loss::Losses;
pub use manifold as nn;
pub use manifold::fc::GradientRetention;
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
