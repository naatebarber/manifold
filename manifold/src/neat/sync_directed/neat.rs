use std::{
    collections::VecDeque,
    error::Error,
    ops::Range,
    sync::Arc,
    thread::{self, available_parallelism, JoinHandle},
    time::Duration,
};
use zmq::{
    Context, Socket,
    SocketType::{PUB, PULL, PUSH},
    DONTWAIT,
};

use crate::{
    manifold::{fc::Manifold, trainer::Hyper},
    util::timestamp,
    Substrate,
};

use super::super::TrainChunk;
use super::worker::worker;

pub type MultipartMessage = Vec<Vec<u8>>;

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
    chunks_per_generation: usize,
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
            chunks_per_generation: 100,
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

    pub fn set_chunks_per_generation(&mut self, cpg: usize) -> &mut Self {
        self.chunks_per_generation = cpg;
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

    pub fn spawn_workers(&mut self) -> &mut Self {
        let cores: usize = available_parallelism().unwrap().into();

        for core in 0..cores {
            let worker_name = format!("manifold-neat-worker-{}", core);
            let id = worker_name.clone();
            let substrate = self.substrate.clone();
            let hyper = self.hyper.clone();
            let sample_window = self.sample_window.clone();
            let chunk_size = self.chunk_size.clone();

            let worker_handle =
                match thread::Builder::new()
                    .name(worker_name.clone())
                    .spawn(move || {
                        match worker(id.clone(), substrate, hyper, sample_window, chunk_size) {
                            Err(e) => {
                                eprint!("[Worker '{}' died with error {}]", &id, e)
                            }
                            Ok(_) => (),
                        }
                    }) {
                    Ok(h) => {
                        println!("[Spawned worker '{}']", worker_name);
                        h
                    }
                    Err(e) => {
                        eprintln!(
                            "[Worker '{}' failed to spawn with error {}]",
                            worker_name, e
                        );
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

    pub fn assign_work(
        &mut self,
        payload: Vec<MultipartMessage>,
        timeout: u64,
    ) -> Result<Vec<MultipartMessage>, Box<dyn Error>> {
        for p in payload.into_iter() {
            self.producer_sock.send_multipart(&p, DONTWAIT)?;
        }

        let mut sockets = [self.collector_sock.as_poll_item(zmq::POLLIN)];

        let t = timestamp()?;
        let mut c = t.clone();

        let mut responses: Vec<MultipartMessage> = Vec::new();

        while t + timeout > c {
            zmq::poll(&mut sockets, 10)?;

            if sockets[0].is_readable() {
                let msgb = self.collector_sock.recv_multipart(0)?;
                responses.push(msgb);
            }

            c = timestamp()?;
        }

        Ok(responses)
    }

    pub fn notify_all(
        &mut self,
        payload: MultipartMessage,
        timeout: u64,
    ) -> Result<Vec<MultipartMessage>, Box<dyn Error>> {
        self.pub_sock.send_multipart(payload, 0)?;

        let mut sockets = [self.collector_sock.as_poll_item(zmq::POLLIN)];

        let t = timestamp()?;
        let mut c = t.clone();

        let mut responses: Vec<MultipartMessage> = Vec::new();

        while t + timeout > c {
            zmq::poll(&mut sockets, 10)?;

            if sockets[0].is_readable() {
                let msgb = self.collector_sock.recv_multipart(0)?;
                responses.push(msgb);
            }

            c = timestamp()?;
        }

        Ok(responses)
    }

    pub fn check_state(&mut self, state: &str, timeout: u64) -> Result<bool, Box<dyn Error>> {
        let msg = vec!["state".as_bytes().to_vec()];
        let responses = self.notify_all(msg, timeout)?;

        let states = responses
            .into_iter()
            .filter_map(|msgb| {
                if msgb.len() < 2 {
                    return None;
                }

                let msg_type = String::from_utf8(msgb[0].clone()).ok();
                let sent_state = String::from_utf8(msgb[1].clone()).ok();

                match (msg_type, sent_state) {
                    (Some(x), Some(y)) => {
                        if x == "state" {
                            return Some(y);
                        }
                        None
                    }
                    _ => None,
                }
            })
            .collect::<Vec<String>>();

        for s in states.iter() {
            if s != state {
                return Ok(false);
            }
        }

        Ok(true)
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

        thread::sleep(Duration::from_millis(500));

        let mut all_idle = false;
        while !all_idle {
            all_idle = self.check_state("idle", 100)?;
        }

        println!("[All workers idle]");

        let mut splat: VecDeque<(String, Vec<f64>, Manifold)> = VecDeque::new();

        for epoch in 0..self.arch_epochs {
            println!("[NEAT epoch {}, worker count {}]", epoch, num_workers);

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

            let num_topologies = topologies.len();
            println!("> created {} topologies", num_topologies);

            match self.distribute_topologies(topologies) {
                Ok(()) => (),
                Err(_) => {
                    println!("Distribute topologies failed!");
                    continue;
                }
            }

            let mut awaiting_data = false;
            while !awaiting_data {
                awaiting_data = self.check_state("awaiting_data", 100)?;
            }

            println!("> distributed {} topologies", num_topologies);

            for _ in 0..self.chunks_per_generation {
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
            }

            println!(
                "> streamed {} rows of training data to workers",
                self.chunk_size * self.chunks_per_generation
            );

            let mut trained = false;
            while !trained {
                trained = self.check_state("awaiting_data", 100)?;
            }
            println!("> workers trained");

            let payload = vec!["cmd".as_bytes().to_vec(), "dump".as_bytes().to_vec()];
            let train_results = self.notify_all(payload, 500)?;

            for msgb in train_results.into_iter() {
                let msg_type = match String::from_utf8(msgb[0].clone()) {
                    Ok(n) => n,
                    _ => continue,
                };

                if msg_type != "arch" {
                    continue;
                }

                let nn = match Manifold::load(&msgb[1]) {
                    Ok(m) => m,
                    _ => continue,
                };

                let losses: Vec<f64> = match bincode::deserialize(&msgb[2].clone()) {
                    Ok(l) => l,
                    _ => continue,
                };

                let name = match String::from_utf8(msgb[3].clone()) {
                    Ok(n) => n,
                    _ => continue,
                };

                let worker_loss = losses.iter().fold(0., |a, v| a + *v);

                if splat.len() > self.retain {
                    let current = (name, losses, nn);

                    // bake off!
                    for _ in 0..self.retain {
                        let splat_loss: f64 = splat[0].1.iter().fold(0., |a, v| a + *v);

                        if splat_loss <= worker_loss {
                            splat.rotate_left(1);
                        } else {
                            println!(
                                "> new elite arch from worker {} - LAYERS {:?} LOSS {}",
                                current.0, current.2.layers, worker_loss
                            );

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
