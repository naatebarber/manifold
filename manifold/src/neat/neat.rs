use std::{
    collections::VecDeque,
    error::Error,
    ops::Range,
    sync::Arc,
    thread::{self, available_parallelism, JoinHandle},
    time::Duration,
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
    arch_epochs: usize,
    sample_window: usize,
    chunk_size: usize,
    retain: usize,
}

impl Neat {
    pub fn new(d_in: usize, d_out: usize) -> Result<Neat, Box<dyn Error>> {
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
            breadth: 4..12,
            depth: 4..12,
            workers: VecDeque::new(),
            sample_window: 1000,
            chunk_size: 10,
            arch_epochs: 100,
            retain: 1,
        })
    }

    pub fn set_breadth(&mut self, breadth: Range<usize>) -> &mut Self {
        self.breadth = breadth;
        self
    }

    pub fn set_depth(&mut self, depth: Range<usize>) -> &mut Self {
        self.depth = depth;
        self
    }

    pub fn set_hyper(&mut self, hyper: Hyper) -> &mut Self {
        self.hyper = Arc::new(hyper);
        self
    }

    pub fn set_arc_substrate(&mut self, substrate: Arc<Substrate>) -> &mut Self {
        self.substrate = substrate;
        self
    }

    pub fn set_chunk_size(&mut self, size: usize) -> &mut Self {
        self.chunk_size = size;
        self
    }

    pub fn set_sample_window(&mut self, sample_window: usize) -> &mut Self {
        self.sample_window = sample_window;
        self
    }

    pub fn set_arch_epochs(&mut self, epochs: usize) -> &mut Self {
        self.arch_epochs = epochs;
        self
    }

    pub fn set_retain(&mut self, retain: usize) -> &mut Self {
        self.retain = retain;
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

        let result_sock = context.socket(PUSH)?;
        result_sock.connect("tcp://127.0.0.1:12023")?;

        let data_sock = context.socket(SUB)?;
        data_sock.connect("tcp://127.0.0.1:12021")?;
        data_sock.set_subscribe("xy".as_bytes())?;
        data_sock.set_subscribe("kill".as_bytes())?;

        let mut sockets = [
            worker_sock.as_poll_item(zmq::POLLIN),
            data_sock.as_poll_item(zmq::POLLIN),
        ];

        let mut manifold: Option<Manifold> = None;
        let mut x_q: VecDeque<Vec<f64>> = VecDeque::new();
        let mut y_q: VecDeque<Vec<f64>> = VecDeque::new();

        let mut worker_losses: Vec<f64> = vec![];

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
                    }
                    Err(_) => {
                        eprintln!("[{}] Received malformed binary architecture.", name);
                        manifold = None
                    }
                }
            }

            if sockets[1].is_readable() {
                let msgb = data_sock.recv_multipart(0)?;
                let msg_type = String::from_utf8_lossy(&msgb[0]);

                let bad_msg = || println!("[{}] received bad training data message.", name);

                if msgb.len() < 2 {
                    bad_msg();
                    continue;
                }

                if msg_type == "xy" {
                    match TrainChunk::load(&msgb[1]) {
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
                } else if msg_type == "cmd" {
                    let cmd = String::from_utf8(msgb[1].clone())?;

                    match cmd.as_str() {
                        "dump" => {
                            if let Some(nn) = &mut manifold {
                                let bin_nn = nn.dump()?;
                                let result = [
                                    bin_nn,
                                    bincode::serialize(&worker_losses)?,
                                    name.as_bytes().to_vec(),
                                ];
                                result_sock.send_multipart(result, 0)?;
                            }
                        }
                        "kill" => return Ok(()),
                        _ => println!("[{}] Unknown command {}", name, cmd),
                    }
                } else {
                    bad_msg()
                }
            }

            if let Some(nn) = &mut manifold {
                // Train the model and evaluate performance.
                let mut trainer = nn.set_substrate(substrate.clone()).get_trainer();

                trainer
                    .override_hyper((*hyper).clone())
                    .train(x_q.clone().into(), y_q.clone().into());

                worker_losses.extend(trainer.losses.drain(..))
            }
        }
    }

    pub fn spawn_workers(&mut self) -> &mut Self {
        let cores: usize = available_parallelism().unwrap().into();

        for core in 0..cores {
            let worker_name = format!("manifold-neat-worker-{}", core);
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

    pub fn graceful_shutdown(&mut self) -> usize {
        let turns = self.workers.len();
        let mut turn = 0;
        let mut exited = 0;
        while turn < turns {
            if let Some(worker) = self.workers.pop_front() {
                if worker.is_finished() {
                    let _ = worker.join();
                    exited += 1;
                } else {
                    self.workers.push_back(worker);
                }
            } else {
                return exited;
            }

            turn += 1;
        }

        turns - exited
    }

    pub fn kill_workers(&mut self) -> Result<(), Box<dyn Error>> {
        let parts = ["cmd".as_bytes(), "kill".as_bytes()];
        self.pub_sock.send_multipart(parts, 0)?;
        let num_workers = self.workers.len();

        println!("[Done. Killing {} workers]", num_workers);

        let mut remaining_workers = self.graceful_shutdown();

        while remaining_workers > 0 {
            println!("[Waiting on {} workers...]", remaining_workers);
            thread::sleep(Duration::from_millis(500));
            remaining_workers = self.graceful_shutdown();
        }

        Ok(())
    }

    pub fn create_topologies(&mut self, exclude_retain: usize) -> Vec<Manifold> {
        (0..self.workers.len() - exclude_retain)
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
        let binary_manifolds = manifolds.into_iter().filter_map(|mut m| m.dump().ok());

        for bin in binary_manifolds {
            self.producer_sock.send_multipart([bin], 0)?
        }

        Ok(())
    }

    pub fn stream_data(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut tc = TrainChunk::new();
        tc.bulk(x, y);
        let bin_tc = tc.dump()?;

        self.pub_sock
            .send_multipart(["xy".as_bytes(), &bin_tc], 0)?;

        Ok(())
    }

    pub fn sift(
        &mut self,
        mut x_pool: VecDeque<Vec<f64>>,
        mut y_pool: VecDeque<Vec<f64>>,
    ) -> Result<Vec<Manifold>, Box<dyn Error>> {
        self.spawn_workers();
        let num_workers = self.workers.len();

        let mut splat: VecDeque<(String, Vec<f64>, Manifold)> = VecDeque::new();

        for epoch in 0..self.arch_epochs {
            println!("[neat epoch {}, worker count {}]", epoch, num_workers);

            let mut topologies = self.create_topologies(splat.len());

            // Continuously bake the elites against the rest
            if splat.len() > 0 {
                let mut splat_c = splat
                    .clone()
                    .drain(..)
                    .map(|v| v.2)
                    .collect::<Vec<Manifold>>();
                topologies.append(&mut splat_c);
            }

            match self.distribute_topologies(topologies) {
                Ok(()) => (),
                Err(_) => {
                    println!("Distribute topologies failed!");
                    continue;
                }
            }

            match self.stream_data(
                x_pool.drain(0..self.chunk_size).collect(),
                y_pool.drain(0..self.chunk_size).collect(),
            ) {
                Ok(()) => (),
                Err(_) => {
                    println!("Stream failed!");
                    continue;
                }
            }

            let mut workers_received = 0;

            while workers_received < num_workers {
                let msgb = match self.collector_sock.recv_multipart(0) {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        workers_received += 1;
                        continue;
                    }
                };

                let nn = match Manifold::load(&msgb[0]) {
                    Ok(m) => m,
                    _ => continue,
                };

                let losses: Vec<f64> = match bincode::deserialize(&msgb[1].clone()) {
                    Ok(l) => l,
                    _ => continue,
                };

                let name = match String::from_utf8(msgb[2].clone()) {
                    Ok(n) => n,
                    _ => continue,
                };

                if splat.len() > self.retain {
                    let current = (name, losses, nn);

                    // bake off!
                    for _ in 0..self.retain {
                        let splat_loss: f64 = splat[0].1.iter().fold(0., |a, v| a + *v);
                        let current_loss: f64 = current.1.iter().fold(0., |a, v| a + *v);

                        if splat_loss <= current_loss {
                            splat.rotate_left(1);
                        } else {
                            splat.pop_front();
                            splat.push_back(current);
                            break;
                        }
                    }
                } else {
                    splat.push_back((name, losses, nn))
                }
            }
        }

        let best = splat.into_iter().map(|x| x.2).collect::<Vec<Manifold>>();

        self.kill_workers()?;

        Ok(best)
    }
}
