use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use ndarray::Array2;

use rand::distributions::Uniform;
use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Substrate {
    weights: VecDeque<f64>,
    pub size: usize,
}

impl Substrate {
    pub fn new(size: usize, range: Range<f64>) -> Substrate {
        let mut weights: VecDeque<f64> = VecDeque::new();

        let distribution = Uniform::new(range.start, range.end);
        let mut rng = thread_rng();

        for _ in 0..=size {
            weights.push_back(distribution.sample(&mut rng))
        }

        weights
            .make_contiguous()
            .sort_unstable_by(|a, b| match a > b {
                true => std::cmp::Ordering::Greater,
                false => std::cmp::Ordering::Less,
            });

        Substrate {
            weights,
            size: size - 1,
        }
    }

    pub fn get(&self, i: usize) -> f64 {
        let w = self.weights.get(i).expect(
            format!(
                "Tried to access Substrate[{}] from Substrate[{}..{}]",
                i, 0, self.size
            )
            .as_str(),
        );
        *w
    }

    pub fn share(self) -> Arc<Self> {
        Arc::new(self)
    }

    pub fn highspeed<'a>(
        &'a self,
        gradient: &'a mut Array2<f64>,
        link: &'a mut Array2<usize>,
        learning_rate: f64,
    ) {
        assert_eq!(gradient.raw_dim(), link.raw_dim());

        // Keep these things in mind.
        // Relation between gradient values and integer updates
        // Size of the substrate, and how much impact each integer update has.
        // Maybe make it more difficult to approach the edge of the substrate than navigate the middle.

        // Assume a step is 1/1000 of the substrate
        // Define everything else in terms of step
        let step = self.size as f64 / 1000.;

        // Combine highspeed rate with step for gradient element-wise influence on link.
        let mut gradient_steps = gradient.map(|x| step * learning_rate * x);
        let actionable_steps = gradient_steps.map(|x| x.floor());

        // Use actionable steps as a mask on gradient_steps
        gradient_steps -= &actionable_steps;

        // Reverse the scaled gradient steps
        *gradient = gradient_steps.mapv_into(|x| x / (step * learning_rate));

        // Extract values with amplitude > 1 out of gradient and leave the
        // 0<abs(x)<1 float values to be fed back into layer
        let delta_signed = link.map(|x| *x as i64) + actionable_steps.map(|x| *x as i64);

        let mut leftedge = 0;
        let mut rightedge = 0;

        let delta = delta_signed.map(|x| {
            if *x < 0 {
                leftedge += 1;
                return 0 as usize;
            }

            if *x > self.size as i64 {
                rightedge += 1;
                return self.size as usize;
            }

            return *x as usize;
        });

        *link = delta;
    }

    pub fn dump(&self, tag: &str) -> Result<(), Box<dyn Error>> {
        let substrate_str = serde_json::to_string(&self)?;
        let cwd = env::current_dir()?;
        fs::write(
            cwd.join(PathBuf::from_str(
                format!(".models/{}.substrate.json", tag).as_str(),
            )?),
            substrate_str,
        )?;

        Ok(())
    }

    pub fn load(tag: &str) -> Result<Substrate, Box<dyn Error>> {
        let cwd = env::current_dir()?;

        let serial = fs::read_to_string(cwd.join(PathBuf::from_str(
            format!(".models/{}.substrate.json", tag).as_str(),
        )?))?;

        let substrate: Substrate = serde_json::from_str(&serial)?;

        Ok(substrate)
    }

    pub fn load_substrate_or_create(tag: &str, size: usize, range: Range<f64>) -> Substrate {
        let s = match Substrate::load(tag) {
            Ok(s) => s,
            _ => {
                let s = Substrate::new(size, range);
                s.dump(tag)
                    .unwrap_or_else(|_| println!("Failed to save substrate."));
                s
            }
        };

        s
    }
}
