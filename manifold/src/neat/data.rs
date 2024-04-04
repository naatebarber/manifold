use std::{collections::VecDeque, error::Error};

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

pub trait NeatDataset {
    fn sample_chunk(&mut self, chunk_size: usize) -> TrainChunk;
}

pub struct RollingDataset {
    x: VecDeque<Vec<f64>>,
    y: VecDeque<Vec<f64>>,
}

impl RollingDataset {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> RollingDataset {
        RollingDataset {
            x: VecDeque::from(x),
            y: VecDeque::from(y),
        }
    }
}

impl NeatDataset for RollingDataset {
    fn sample_chunk(&mut self, chunk_size: usize) -> TrainChunk {
        let mut tc = TrainChunk::new();

        for _ in 0..chunk_size {
            let (cx, cy) = match (self.x.pop_front(), self.y.pop_front()) {
                (Some(x), Some(y)) => (x, y),
                _ => {
                    return tc;
                }
            };

            tc.insert(cx.clone(), cy.clone());

            self.x.push_back(cx);
            self.y.push_back(cy);
        }

        tc
    }
}

pub type Xy = (Vec<f64>, Vec<f64>);
pub type Generator = Box<dyn Fn() -> Xy>;
pub struct LiveDataset {
    generator: Generator,
}

impl LiveDataset {
    pub fn new(generator: impl Fn() -> Xy + 'static) -> LiveDataset {
        LiveDataset {
            generator: Box::new(generator),
        }
    }
}

impl NeatDataset for LiveDataset {
    fn sample_chunk(&mut self, chunk_size: usize) -> TrainChunk {
        let mut tc = TrainChunk::new();

        for _ in 0..chunk_size {
            let (x, y) = (self.generator)();
            tc.insert(x, y);
        }

        tc
    }
}
