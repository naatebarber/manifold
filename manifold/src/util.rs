use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};

use ndarray::{stack, Array1, Array2, Array3, Axis};

pub fn timestamp() -> Result<u64, Box<dyn Error>> {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    let in_ms =
        since_the_epoch.as_secs() * 1000 + since_the_epoch.subsec_nanos() as u64 / 1_000_000;

    Ok(in_ms)
}

pub fn as_tensor(x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> (Array3<f64>, Array3<f64>) {
    let x_a2 = x
        .into_iter()
        .map(|xv| Array1::from(xv).insert_axis(Axis(0)))
        .collect::<Vec<Array2<f64>>>();
    let y_a2 = y
        .into_iter()
        .map(|yv| Array1::from(yv).insert_axis(Axis(0)))
        .collect::<Vec<Array2<f64>>>();

    let x_3 = stack(
        Axis(0),
        &x_a2.iter().map(|ar| ar.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let y_3 = stack(
        Axis(0),
        &y_a2.iter().map(|ar| ar.view()).collect::<Vec<_>>(),
    )
    .unwrap();

    (x_3, y_3)
}
