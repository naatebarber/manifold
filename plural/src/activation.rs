use serde::{Deserialize, Serialize};

pub struct Activation;

impl Activation {
    pub fn relu(x: f64) -> f64 {
        if x < 0. {
            return 0.;
        }
        x
    }

    pub fn leaky_relu(x: f64) -> f64 {
        if x < 0. {
            return 0.1 * x;
        }
        x
    }

    pub fn elu(x: f64) -> f64 {
        if x < 0. {
            return x.exp() - 1.;
        }
        x
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ActivationType {
    Relu,
    LeakyRelu,
    Elu,
    None,
}
