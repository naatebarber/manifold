use super::sigmoid;
use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;

pub fn argmax(d: &[f64]) -> usize {
    if d.len() < 1 {
        return 0;
    }

    let mut max_ix = 0;
    let mut max = &d[0];

    for (i, v) in d.iter().enumerate() {
        if v > max {
            max = v;
            max_ix = i;
        }
    }

    max_ix
}

pub fn argmin(d: &[f64]) -> usize {
    if d.len() < 1 {
        return 0;
    }

    let mut max_ix = 0;
    let mut max = &d[0];

    for (i, v) in d.iter().enumerate() {
        if v < max {
            max = v;
            max_ix = i;
        }
    }

    max_ix
}

pub fn max(x: Array2<f64>) -> f64 {
    x.fold(f64::MIN, |a, v| {
        if *v > a {
            return *v;
        }
        return a;
    })
}

pub fn min(x: Array2<f64>) -> f64 {
    x.fold(f64::MAX, |a, v| {
        if *v < a {
            return *v;
        }
        return a;
    })
}

pub fn onehot(i: u64, size: u64) -> Vec<f64> {
    let mut oh = vec![0.; size as usize];
    if i < size {
        oh[i as usize] = 1.;
    }
    oh
}

pub fn shape_sigmoid(scores: &[f64]) -> Vec<f64> {
    scores.iter().map(|x| sigmoid(*x)).collect::<Vec<f64>>()
}

pub fn softmax(x: Array2<f64>) -> Array2<f64> {
    let max_vals = x.map_axis(Axis(1), |row| row.max().unwrap().to_owned());
    let exps = (&x - &max_vals.insert_axis(Axis(1))).map(|v| v.exp());
    let sum_exps = exps.sum_axis(Axis(1)).insert_axis(Axis(1));
    let f = exps / &sum_exps;

    f
}
