use super::sigmoid;

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

pub fn onehot(i: u8, m: u8) -> Vec<f64> {
    let mut oh = vec![0.; m.into()];
    if i < m {
        oh[i as usize] = 1.;
    }
    oh
}

pub fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores
        .iter()
        .map(|&x| ((x - max_score) as f64).exp())
        .collect();
    let sum_exp_scores: f64 = exp_scores.iter().sum();
    exp_scores.iter().map(|&x| x / sum_exp_scores).collect()
}

pub fn shape_sigmoid(scores: &[f64]) -> Vec<f64> {
    scores.iter().map(|x| sigmoid(*x)).collect::<Vec<f64>>()
}
