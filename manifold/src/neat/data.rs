use std::error::Error;

use bincode;
use serde::{self, Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TrainChunk(pub Vec<Vec<f64>>, pub Vec<Vec<f64>>);

impl TrainChunk {
    pub fn new() -> TrainChunk {
        TrainChunk(Vec::new(), Vec::new())
    }

    pub fn insert(&mut self, x: Vec<f64>, y: Vec<f64>) -> &mut Self {
        self.0.push(x);
        self.1.push(y);
        self
    }

    pub fn bulk(&mut self, mut x: Vec<Vec<f64>>, mut y: Vec<Vec<f64>>) -> &mut Self {
        if x.len() != y.len() {
            eprintln!("Mismatched X Y lengths, skipping insert.");
            return self;
        }

        self.0.append(&mut x);
        self.1.append(&mut y);

        self
    }

    pub fn dump(&mut self) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn load(serialized: &Vec<u8>) -> Result<TrainChunk, Box<dyn Error>> {
        Ok(bincode::deserialize(&serialized)?)
    }
}
