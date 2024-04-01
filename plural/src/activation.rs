use std::fmt::Debug;
use std::rc::Rc;

pub trait Activation {
    fn a(&self, x: f64) -> f64;
    fn d(&self, x: f64) -> f64;
}

impl Debug for dyn Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ActivationFn")
    }
}

pub struct Relu;

impl Relu {
    pub fn new() -> Rc<Relu> {
        Rc::new(Relu)
    }
}

impl Activation for Relu {
    fn a(&self, x: f64) -> f64 {
        if x < 0. {
            return 0.;
        }
        x
    }

    fn d(&self, x: f64) -> f64 {
        if x < 0. {
            return 0.;
        }
        1.
    }
}

pub struct Transparent;

impl Transparent {
    pub fn new() -> Rc<Transparent> {
        Rc::new(Transparent)
    }
}

impl Activation for Transparent {
    fn a(&self, x: f64) -> f64 {
        x
    }

    fn d(&self, x: f64) -> f64 {
        x
    }
}
