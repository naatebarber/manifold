use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn timestamp() -> Result<u64, Box<dyn Error>> {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    let in_ms =
        since_the_epoch.as_secs() * 1000 + since_the_epoch.subsec_nanos() as u64 / 1_000_000;

    Ok(in_ms)
}
