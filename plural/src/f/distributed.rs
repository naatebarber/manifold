use std::thread::JoinHandle;
use std::thread::{self, available_parallelism};

pub fn distributed<T: Send + 'static>(tasks: Vec<Box<dyn (FnOnce() -> T) + Send>>) -> Vec<T> {
    let cores: usize = available_parallelism().unwrap().into();
    let mut batches: Vec<Vec<Box<dyn (FnOnce() -> T) + Send>>> = Vec::with_capacity(cores);

    for _ in 0..cores {
        batches.push(vec![]);
    }

    for (i, task) in tasks.into_iter().enumerate() {
        let batch = i % cores;
        batches[batch].push(task);
    }

    let handles: Vec<JoinHandle<Vec<T>>> = batches
        .into_iter()
        .map(|mut batch| {
            thread::spawn(move || {
                let mut results: Vec<T> = vec![];
                for task in batch.drain(..) {
                    let r: T = task();
                    results.push(r)
                }
                results
            })
        })
        .collect();

    let mut results: Vec<T> = vec![];
    for handle in handles.into_iter() {
        if let Some(mut v) = handle.join().ok() {
            results.append(&mut v);
        }
    }

    results
}
