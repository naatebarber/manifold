use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use crate::activation::{Activation, ActivationType};

use rand::distributions::Uniform;
use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

pub type Substrate = Arc<VecDeque<Neuron>>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    w: f64,
    b: f64,
    a: ActivationType,
}

impl Neuron {
    pub fn random_normal(
        activation: ActivationType,
        distribution: Uniform<f64>,
        rng: &mut ThreadRng,
    ) -> Neuron {
        Neuron {
            w: distribution.sample(rng),
            b: distribution.sample(rng),
            a: activation,
        }
    }

    pub fn substrate(
        size: usize,
        range: Range<f64>,
        activation: ActivationType,
    ) -> (Substrate, usize) {
        let mut neurons: VecDeque<Neuron> = VecDeque::new();

        let distribution = Uniform::new(range.start, range.end);
        let mut rng = thread_rng();

        for _ in 0..=size {
            neurons.push_back(Neuron::random_normal(
                activation.clone(),
                distribution,
                &mut rng,
            ))
        }

        neurons.make_contiguous().sort_unstable_by(|a, b| {
            let v = a.w * 2. + a.b;
            let w = b.w * 2. + b.b;
            match v > w {
                true => std::cmp::Ordering::Greater,
                false => std::cmp::Ordering::Less,
            }
        });

        (Arc::new(neurons), size - 1)
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self.a {
            ActivationType::Relu => Activation::relu(x),
            ActivationType::LeakyRelu => Activation::leaky_relu(x),
            ActivationType::Elu => Activation::elu(x),
            ActivationType::None => x,
        }
    }

    pub fn forward(&self, inputs: Vec<f64>, discount: f64) -> f64 {
        let mut affected = inputs
            .into_iter()
            .map(|mut i| {
                let after = i * self.w;
                let mut diff = after - i;
                diff *= discount;
                i += diff;
                i
            })
            .fold(0., |a, v| a + v);

        affected += self.b;

        self.activation(affected)
    }

    pub fn dump_substrate(neuros: Substrate, tag: &str) -> Result<(), Box<dyn Error>> {
        let substrate_str = serde_json::to_string(&neuros)?;
        let cwd = env::current_dir()?;
        fs::write(
            cwd.join(PathBuf::from_str(
                format!(".models/{}.substrate.json", tag).as_str(),
            )?),
            substrate_str,
        )?;

        Ok(())
    }

    pub fn load_substrate(tag: &str) -> Result<(Substrate, usize), Box<dyn Error>> {
        let cwd = env::current_dir()?;

        let serial = fs::read_to_string(cwd.join(PathBuf::from_str(
            format!(".models/{}.substrate.json", tag).as_str(),
        )?))?;

        let substrate: Substrate = Arc::new(serde_json::from_str(&serial)?);
        let substrate_len = substrate.len() - 1;

        Ok((substrate, substrate_len))
    }

    pub fn load_substrate_or_create(
        tag: &str,
        size: usize,
        range: Range<f64>,
        activation: ActivationType,
    ) -> (Substrate, usize) {
        let (substrate, len) = match Neuron::load_substrate(tag) {
            Ok(s) => s,
            _ => {
                let (neuros, mesh_len) = Neuron::substrate(size, range, activation);
                Neuron::dump_substrate(neuros.clone(), tag)
                    .unwrap_or_else(|_| println!("Failed to save substrate."));
                (neuros, mesh_len)
            }
        };

        (substrate, len)
    }
}
