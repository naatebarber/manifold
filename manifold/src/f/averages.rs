pub fn weighted_average(x: f64, wx: f64, y: f64, wy: f64) -> f64 {
    ((x * wx) + (y * wy)) / (wx + wy)
}
