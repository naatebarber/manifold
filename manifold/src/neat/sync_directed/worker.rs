use bincode;
use std::collections::VecDeque;
use std::error::Error;
use std::sync::Arc;
use zmq::{poll, Context, PULL, PUSH, SUB};

use super::super::data::TrainChunk;
use crate::nn::fc_single::Manifold;
use crate::nn::trainer::Hyper;
use crate::Substrate;

pub fn worker(
    name: String,
    substrate: Arc<Substrate>,
    hyper: Arc<Hyper>,
    sample_window: usize,
    chunk_size: usize,
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

    let mut sockets = [
        worker_sock.as_poll_item(zmq::POLLIN),
        data_sock.as_poll_item(zmq::POLLIN),
        cmd_sock.as_poll_item(zmq::POLLIN),
    ];

    let mut state = "idle";
    let mut manifold: Option<Manifold> = None;
    let mut x_q: VecDeque<Vec<f64>> = VecDeque::new();
    let mut y_q: VecDeque<Vec<f64>> = VecDeque::new();

    let mut worker_losses: Vec<f64> = vec![];

    let send_state = |state: &str| {
        let _ = result_sock.send_multipart(["state".as_bytes(), state.as_bytes()], 0);
    };

    send_state(state);

    loop {
        poll(&mut sockets, 10)?;

        if sockets[0].is_readable() {
            let msgb = worker_sock.recv_multipart(0)?;
            if msgb.len() < 1 {
                println!("[{}] received bad architecture message.", name);
            }

            match Manifold::load(&msgb[0]) {
                Ok(m) => {
                    manifold = Some(m);
                    state = "awaiting_data";
                    worker_losses = vec![];
                    send_state(state);
                }
                Err(_) => {
                    eprintln!("[{}] Received malformed binary architecture.", name);
                    state = "idle";
                    manifold = None
                }
            }
        }

        if sockets[1].is_readable() {
            let msgb = data_sock.recv_multipart(0)?;

            let bad_msg = || println!("[{}] received bad training data message.", name);

            if msgb.len() < 2 {
                bad_msg();
                continue;
            }

            match TrainChunk::load(&msgb[1]) {
                Ok(tc) => {
                    x_q.append(&mut VecDeque::from(tc.0));
                    y_q.append(&mut VecDeque::from(tc.1));

                    while x_q.len() > sample_window && y_q.len() > sample_window {
                        x_q.pop_front();
                        y_q.pop_front();
                    }

                    state = "trainable";
                    send_state(state);
                }
                Err(_) => {
                    eprintln!("[{}] Received malformed train chunk.", name);
                }
            }
        }

        if sockets[2].is_readable() {
            let msgb = cmd_sock.recv_multipart(0)?;
            let cmd = String::from_utf8(msgb[1].clone())?;
            match cmd.as_str() {
                "dump" => {
                    if let Some(nn) = &mut manifold {
                        let bin_nn = nn.dump()?;
                        let result = [
                            "arch".as_bytes().to_vec(),
                            bin_nn,
                            bincode::serialize(&worker_losses)?,
                            name.as_bytes().to_vec(),
                        ];
                        result_sock.send_multipart(result, 0)?;
                    }
                }
                "state" => send_state(state),
                "kill" => {
                    state = "terminating";
                    send_state(state);
                    return Ok(());
                }
                _ => println!("[{}] Unknown command {}", name, cmd),
            }
        }

        if x_q.len() < chunk_size || y_q.len() < chunk_size {
            state = "awaiting_data";
            continue;
        }

        if let Some(nn) = &mut manifold {
            let mut trainer = nn.set_substrate(substrate.clone()).get_trainer();

            trainer.override_hyper((*hyper).clone()).train(
                x_q.drain(0..chunk_size).collect(),
                y_q.drain(0..chunk_size).collect(),
            );

            worker_losses.extend(trainer.losses.drain(..));
        }
    }
}
