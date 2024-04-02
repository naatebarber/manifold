pub mod activation;
pub mod f;
pub mod loss;
pub mod manifold;
pub mod neat;
pub mod substrate;

pub use manifold::fc::GradientRetention;
pub use neat::Neat;
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
