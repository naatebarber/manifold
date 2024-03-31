pub fn accuracy<T: Copy + PartialEq>(prediction: &[T], actual: &[T]) -> f64 {
    let mut tracks: Vec<u64> = vec![];

    prediction.iter().enumerate().for_each(|(i, v)| {
        let actual_v = actual[i];
        if *v == actual_v {
            tracks.push(1);
            return;
        }

        tracks.push(0);
    });

    (tracks.iter().fold(0, |a, v| a + v) as f64 / tracks.len() as f64) * 100.
}
