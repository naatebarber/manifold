use rand::distributions::uniform::SampleRange;
use std::{
    collections::VecDeque,
    error::Error,
    ops::Range,
    sync::Arc,
    thread::{self, available_parallelism, JoinHandle},
};
use zmq::{
    poll, Context, Socket,
    SocketType::{PUB, PULL, PUSH, SUB},
};

use crate::{
    manifold::{fc::Manifold, trainer::Hyper},
    Substrate,
};

use super::TrainChunk;

pub struct Neat {
    producer_sock: Socket,
    collector_sock: Socket,
    pub_sock: Socket,
    substrate: Arc<Substrate>,
    hyper: Arc<Hyper>,
    d_in: usize,
    d_out: usize,
    breadth: Range<usize>,
    depth: Range<usize>,
    workers: VecDeque<JoinHandle<()>>,
    sample_window: usize,
}

impl Neat {
    pub fn new(
        d_in: usize,
        d_out: usize,
        breadth: Range<usize>,
        depth: Range<usize>,
    ) -> Result<Neat, Box<dyn Error>> {
        let context = Context::new();
        let producer_sock = context.socket(PUSH)?;
        let collector_sock = context.socket(PULL)?;
        let pub_sock = context.socket(PUB)?;

        producer_sock.bind("tcp://127.0.0.1:12021")?;
        collector_sock.bind("tcp://127.0.0.1:12023")?;
        pub_sock.bind("tcp://127.0.0.1:12024")?;

        Ok(Neat {
            producer_sock,
            collector_sock,
            pub_sock,
            substrate: Substrate::blank().share(),
            hyper: Arc::new(Hyper::new()),
            d_in,
            d_out,
            breadth,
            depth,
            workers: VecDeque::new(),
            sample_window: 1000,
        })
    }

    pub fn with_hyper(&mut self, hyper: Hyper) -> &mut Self {
        self.hyper = Arc::new(hyper);
        self
    }

    pub fn with_arc_substrate(&mut self, substrate: Arc<Substrate>) -> &mut Self {
        self.substrate = substrate;
        self
    }

    pub fn with_sample_window(&mut self, sample_window: usize) -> &mut Self {
        self.sample_window = sample_window;
        self
    }

    pub fn worker(
        name: String,
        substrate: Arc<Substrate>,
        hyper: Arc<Hyper>,
        sample_window: usize,
    ) -> Result<(), Box<dyn Error>> {
        let context = Context::new();

        let worker_sock = context.socket(PULL)?;
        worker_sock.connect("tcp://127.0.0.1:12021")?;
        worker_sock.set_subscribe("arch".as_bytes())?;

        let result_sock = context.socket(PUSH)?;
        result_sock.connect("tcp://127.0.0.1:12023")?;

        let data_sock = context.socket(SUB)?;
        data_sock.connect("tcp://127.0.0.1:12021")?;
        data_sock.set_subscribe("xy".as_bytes())?;

        let mut sockets = [
            worker_sock.as_poll_item(zmq::POLLIN),
            data_sock.as_poll_item(zmq::POLLIN),
        ];

        let mut manifold: Option<Manifold> = None;
        let mut x_q: VecDeque<Vec<f64>> = VecDeque::new();
        let mut y_q: VecDeque<Vec<f64>> = VecDeque::new();

        loop {
            poll(&mut sockets, 10)?;

            if sockets[0].is_readable() {
                let msgb = worker_sock.recv_multipart(0)?;
                if msgb.len() < 1 {
                    println!("[{}] received bad architecture message.", name);
                }
                match Manifold::deserialize(&msgb[0]) {
                    Ok(m) => {
                        manifold = Some(m);
                    }
                    Err(_) => {
                        eprintln!("[{}] Received malformed binary architecture.", name);
                        manifold = None
                    }
                }
            }

            if sockets[1].is_readable() {
                let msgb = data_sock.recv_multipart(0)?;
                if msgb.len() < 2 {
                    println!("[{}] received bad training data message.", name);
                }

                match TrainChunk::deserialize(&msgb[1]) {
                    Ok(tc) => {
                        x_q.append(&mut VecDeque::from(tc.0));
                        y_q.append(&mut VecDeque::from(tc.1));

                        while x_q.len() > sample_window && y_q.len() > sample_window {
                            x_q.pop_front();
                            y_q.pop_front();
                        }
                    }
                    Err(_) => {
                        eprintln!("[{}] Received malformed train chunk.", name);
                    }
                }
            }

            if let Some(nn) = &mut manifold {
                // Train the model and evaluate performance.
                nn.set_substrate(substrate.clone())
                    .get_trainer()
                    .override_hyper((*hyper).clone())
                    .train(x_q.clone().into(), y_q.clone().into());
            }
        }
    }

    pub fn spawn_workers(&mut self) -> &mut Self {
        let cores: usize = available_parallelism().unwrap().into();

        for core in 0..cores {
            let worker_name = format!("plural-neat-worker-{}", core);
            let id = worker_name.clone();
            let substrate = self.substrate.clone();
            let hyper = self.hyper.clone();
            let sample_window = self.sample_window.clone();

            let worker_handle =
                match thread::Builder::new()
                    .name(worker_name.clone())
                    .spawn(move || {
                        match Neat::worker(id.clone(), substrate, hyper, sample_window) {
                            Err(e) => {
                                eprint!("{} died with error {}", &id, e)
                            }
                            Ok(_) => (),
                        }
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

            self.workers.push_back(worker_handle);
        }

        self
    }

    pub fn create_topologies(&mut self) -> Vec<Manifold> {
        (0..self.workers.len())
            .into_iter()
            .map(|_| {
                Manifold::dynamic(
                    self.d_in,
                    self.d_out,
                    self.breadth.clone(),
                    self.depth.clone(),
                )
            })
            .collect::<Vec<Manifold>>()
    }

    pub fn distribute_topologies(
        &mut self,
        manifolds: Vec<Manifold>,
    ) -> Result<(), Box<dyn Error>> {
        let binary_manifolds = manifolds.into_iter().filter_map(|mut m| m.serialize().ok());

        for bin in binary_manifolds {
            self.producer_sock.send_multipart([bin], 0)?
        }

        Ok(())
    }

    pub fn join(workers: Vec<JoinHandle<()>>) {
        let _ = workers.into_iter().map(|worker| worker.join());
    }
}
