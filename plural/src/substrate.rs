use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use rand::distributions::Uniform;
use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

pub type Substrate = Arc<VecDeque<Mote>>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Mote {
    v: f64,
}

impl Mote {
    pub fn random_normal(distribution: Uniform<f64>, rng: &mut ThreadRng) -> Mote {
        Mote {
            v: distribution.sample(rng),
        }
    }

    pub fn get(&self) -> f64 {
        self.v.clone()
    }

    pub fn substrate(size: usize, range: Range<f64>) -> (Substrate, usize) {
        let mut motes: VecDeque<Mote> = VecDeque::new();

        let distribution = Uniform::new(range.start, range.end);
        let mut rng = thread_rng();

        for _ in 0..=size {
            motes.push_back(Mote::random_normal(distribution, &mut rng))
        }

        motes
            .make_contiguous()
            .sort_unstable_by(|a, b| match a.v > b.v {
                true => std::cmp::Ordering::Greater,
                false => std::cmp::Ordering::Less,
            });

        (Arc::new(motes), size - 1)
    }

    pub fn dump_substrate(substrate: Substrate, tag: &str) -> Result<(), Box<dyn Error>> {
        let substrate_str = serde_json::to_string(&substrate)?;
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
    ) -> (Substrate, usize) {
        let (substrate, len) = match Mote::load_substrate(tag) {
            Ok(s) => s,
            _ => {
                let (neuros, mesh_len) = Mote::substrate(size, range);
                Mote::dump_substrate(neuros.clone(), tag)
                    .unwrap_or_else(|_| println!("Failed to save substrate."));
                (neuros, mesh_len)
            }
        };

        (substrate, len)
    }
}
