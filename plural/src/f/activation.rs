use std::f64::consts::E;

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

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + E.powf(-x))
}
