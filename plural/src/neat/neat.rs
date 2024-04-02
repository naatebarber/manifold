use std::{
    error::Error,
    sync::Arc,
    thread::{self, available_parallelism, JoinHandle},
};
use zmq::{
    poll, Context, Socket,
    SocketType::{PUB, SUB},
};

use crate::{manifold::fc::Manifold, Substrate};

pub struct Neat {
    context: Context,
    pub_sock: Socket,
    substrate: Arc<Substrate>,
}

impl Neat {
    pub fn new() -> Result<Neat, Box<dyn Error>> {
        let context = Context::new();
        let pub_sock = context.socket(PUB)?;

        Ok(Neat {
            context,
            pub_sock,
            substrate: Substrate::blank().share(),
        })
    }

    pub fn with_arc_substrate(&mut self, substrate: Arc<Substrate>) -> &mut Self {
        self.substrate = substrate;
        self
    }

    pub fn worker(name: String, substrate: Arc<Substrate>) -> Result<(), Box<dyn Error>> {
        let context = Context::new();

        let arch_sock = context.socket(SUB)?;
        arch_sock.connect("tcp://127.0.0.1:12021")?;
        arch_sock.set_subscribe("arch".as_bytes())?;

        let xy_sock = context.socket(SUB)?;
        xy_sock.connect("tcp://127.0.0.1:12021")?;
        xy_sock.set_subscribe("xy".as_bytes())?;

        let mut sockets = [
            arch_sock.as_poll_item(zmq::POLLIN),
            xy_sock.as_poll_item(zmq::POLLIN),
        ];

        let mut current_architecture: Manifold;
        let mut recv_arch = false;
        let mut aggregated_xy: Vec<(Vec<f64>, Vec<f64>)> = vec![];

        loop {
            poll(&mut sockets, 10)?;

            if sockets[0].is_readable() {
                let msgb = arch_sock.recv_multipart(0)?;
                if msgb.len() < 2 {
                    println!("[{}] received bad architecture message.", name);
                }
                let configuration = &msgb[1];
            }

            if sockets[1].is_readable() {
                let msgb = xy_sock.recv_multipart(0)?;
                if msgb.len() < 2 {
                    println!("[{}] received bad training data message.", name);
                }
                let xy = &msgb[1];
            }

            if recv_arch {
                // Train the model and evaluate performance.
            }
        }
    }

    pub fn spawn_workers(&mut self) -> Vec<JoinHandle<()>> {
        let cores: usize = available_parallelism().unwrap().into();
        let mut workers = vec![];

        for core in 0..cores {
            let worker_name = format!("plural-neat-worker-{}", core);
            let id = worker_name.clone();
            let substrate = self.substrate.clone();

            let worker_handle =
                match thread::Builder::new()
                    .name(worker_name.clone())
                    .spawn(move || match Neat::worker(id.clone(), substrate) {
                        Err(e) => {
                            eprint!("{} died with error {}", &id, e)
                        }
                        Ok(_) => (),
                    }) {
                    Ok(h) => {
                        println!("Spawned {}", worker_name);
                        h
                    }
                    Err(e) => {
                        eprintln!("{} failed to spawn with error {}", worker_name, e);
                        continue;
                    }
                };

            workers.push(worker_handle);
        }

        workers
    }

    pub fn join(workers: Vec<JoinHandle<()>>) {
        let _ = workers.into_iter().map(|worker| worker.join());
    }
}
