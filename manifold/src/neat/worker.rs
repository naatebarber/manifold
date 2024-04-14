use bincode;
use std::collections::VecDeque;
use std::error::Error;
use std::sync::Arc;
use zmq::{poll, Context, PULL, PUSH, SUB};

use super::data::TrainChunk;
use crate::nn::DNN;
use crate::optimizers::{Hyper, MiniBatchGradientDescent};
use crate::util::as_tensor;
use crate::Substrate;

pub fn worker(
    name: String,
    substrate: Arc<Substrate>,
    hyper: Arc<Hyper>,
    chunk_window: usize,
    chunks_per_generation: usize,
) -> Result<(), Box<dyn Error>> {
    let context = Context::new();

    let worker_sock = context.socket(PULL)?;
    worker_sock.connect("tcp://127.0.0.1:12021")?;

    let result_sock = context.socket(PUSH)?;
    result_sock.connect("tcp://127.0.0.1:12023")?;

    let data_sock = context.socket(SUB)?;
    data_sock.set_subscribe("xy".as_bytes())?;
    data_sock.connect("tcp://127.0.0.1:12024")?;

    let cmd_sock = context.socket(SUB)?;
    cmd_sock.set_subscribe("cmd".as_bytes())?;
    cmd_sock.connect("tcp://127.0.0.1:12024")?;

    let send_state = |state: &str| {
        let _ = result_sock.send_multipart(["state".as_bytes(), state.as_bytes()], 0);
    };

    let mut state = "idle";
    send_state(state);

    let mut worker_losses: Vec<f64> = vec![];
    let mut consumed_chunks: usize = 0;
    let mut chunks: VecDeque<TrainChunk> = VecDeque::new();

    loop {
        let worker_msgb = worker_sock.recv_multipart(0)?;
        let command = &worker_msgb[0];

        if command == "kill".as_bytes() {
            break;
        }

        if command != "arch".as_bytes() {
            continue;
        }

        if worker_msgb.len() < 1 {
            eprintln!("[{}] Bad command received.", name);
            continue;
        }

        let mut manifold = match DNN::load(&worker_msgb[1]) {
            Ok(manifold) => manifold,
            Err(_) => {
                eprintln!("[{}] Received malformed arch binary.", name);
                continue;
            }
        };

        println!("[🔨 {}] Received an architecture.", name);

        state = "awaiting_data";
        send_state(state);

        let mut sockets = [data_sock.as_poll_item(zmq::POLLIN)];

        loop {
            poll(&mut sockets, 10)?;

            if sockets[0].is_readable() {
                let msgb = data_sock.recv_multipart(0)?;

                let bad_msg = || println!("[{}] received bad training data message.", name);

                if msgb.len() < 2 {
                    bad_msg();
                    continue;
                }

                match TrainChunk::load(&msgb[1]) {
                    Ok(tc) => {
                        chunks.push_back(tc);

                        while chunks.len() > chunk_window {
                            chunks.pop_front();
                        }

                        if state != "training" {
                            state = "training";
                            send_state(state);
                        }
                    }
                    Err(_) => {
                        eprintln!("[{}] Received malformed train chunk.", name);
                    }
                }
            }

            // Train step here.

            while let Some(chunk) = chunks.pop_front() {
                let x_data = chunk.0;
                let y_data = chunk.1;
                let x = as_tensor(x_data);
                let y = as_tensor(y_data);

                let nn = manifold.set_substrate(substrate.clone());
                let mut trainer = MiniBatchGradientDescent::new(nn);

                trainer.override_hyper((*hyper).clone()).train(&x, &y);

                worker_losses.extend(trainer.losses.drain(..));
                consumed_chunks += 1;
                if consumed_chunks >= chunks_per_generation {
                    break;
                }
            }

            if consumed_chunks >= chunks_per_generation {
                let bin_nn = manifold.dump()?;
                let result = [
                    "dump".as_bytes().to_vec(),
                    bin_nn,
                    bincode::serialize(&worker_losses)?,
                ];
                result_sock.send_multipart(result, 0)?;
                consumed_chunks = 0;

                state = "idle";
                send_state(state);

                println!("[🔨 {}] Finished evaluating an architecture.", name);

                break;
            } else {
                state = "awaiting_data";
                send_state(state);
            }
        }
    }

    Ok(())
}
