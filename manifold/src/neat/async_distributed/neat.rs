use std::{
    cell::RefCell,
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
};

use crate::{
    manifold::{fc::Manifold, trainer::Hyper},
    Substrate,
};

use super::super::{LiveDataset, NeatDataset, RollingDataset};
use super::worker::worker;

pub enum EvolutionStyle {
    MonteCarlo,
    Genetic,
}

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
    chunk_window: usize,
    chunk_size: usize,
    chunks_per_generation: usize,
    retain: usize,
    dataset: RefCell<Box<dyn NeatDataset>>,
    evolution_style: EvolutionStyle,
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
            chunk_window: 1000,
            chunk_size: 10,
            chunks_per_generation: 100,
            arch_epochs: 100,
            retain: 1,
            dataset: RefCell::new(Box::new(LiveDataset::new(|| (vec![], vec![])))),
            evolution_style: EvolutionStyle::MonteCarlo,
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

    pub fn set_chunk_window(&mut self, chunk_window: usize) -> &mut Self {
        self.chunk_window = chunk_window;
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

    pub fn set_rolling_dataset(&mut self, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) -> &mut Self {
        let rolling_dataset = RollingDataset::new(x, y);
        self.dataset = RefCell::new(Box::new(rolling_dataset));
        self
    }

    pub fn set_live_dataset(
        &mut self,
        generator: impl Fn() -> (Vec<f64>, Vec<f64>) + 'static,
    ) -> &mut Self {
        let live_dataset = LiveDataset::new(generator);
        self.dataset = RefCell::new(Box::new(live_dataset));
        self
    }

    pub fn set_evolution_style(&mut self, style: EvolutionStyle) -> &mut Self {
        self.evolution_style = style;
        self
    }

    pub fn spawn_workers(&mut self) -> &mut Self {
        let cores: usize = available_parallelism().unwrap().into();

        for core in 0..cores {
            let worker_name = format!("neat-worker-{}", core);
            let id = worker_name.clone();
            let substrate = self.substrate.clone();
            let hyper = self.hyper.clone();
            let chunk_window = self.chunk_window.clone();
            let chunks_per_generation = self.chunks_per_generation.clone();

            let worker_handle =
                match thread::Builder::new()
                    .name(worker_name.clone())
                    .spawn(move || {
                        match worker(
                            id.clone(),
                            substrate,
                            hyper,
                            chunk_window,
                            chunks_per_generation,
                        ) {
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
        let parts = ["kill".as_bytes()];
        let num_workers = self.workers.len();

        for _ in 0..num_workers {
            self.producer_sock.send_multipart(parts, 0)?;
        }

        println!("[Done. Killing {} workers]", num_workers);

        let mut remaining_workers = self.graceful_shutdown();

        while remaining_workers > 0 {
            println!("[Waiting on {} workers...]", remaining_workers);
            thread::sleep(Duration::from_millis(500));
            remaining_workers = self.graceful_shutdown();
        }

        Ok(())
    }

    pub fn push_chunk(&self) -> Result<(), Box<dyn Error>> {
        let mut d = self.dataset.borrow_mut();
        let mut tc = d.sample_chunk(self.chunk_size);
        let bin_tc = tc.dump()?;

        let msgb = ["xy".as_bytes().into(), bin_tc];
        self.pub_sock.send_multipart(msgb, 0)?;

        Ok(())
    }

    pub fn push_manifold(&self) -> Result<(), Box<dyn Error>> {
        let mut next_manifold = match self.evolution_style {
            EvolutionStyle::MonteCarlo | EvolutionStyle::Genetic => Manifold::dynamic(
                self.d_in,
                self.d_out,
                self.breadth.clone(),
                self.depth.clone(),
            ),
        };

        let bin_manifold = next_manifold.dump()?;

        let msgb = ["arch".as_bytes().into(), bin_manifold];
        self.producer_sock.send_multipart(msgb, 0)?;

        Ok(())
    }

    pub fn sift(&mut self) -> Result<Vec<Manifold>, Box<dyn Error>> {
        self.spawn_workers();

        let mut completed_archs = 0;
        let mut best: VecDeque<(Manifold, f64)> = VecDeque::new();

        let mut sockets = [self.collector_sock.as_poll_item(zmq::POLLIN)];

        loop {
            zmq::poll(&mut sockets, 10)?;

            if sockets[0].is_readable() {
                let msgb = self.collector_sock.recv_multipart(0)?;

                if msgb.len() < 2 {
                    continue;
                }

                if msgb[0] == "state".as_bytes() {
                    if msgb[1] == "idle".as_bytes() {
                        // Send manifold
                        match self.push_manifold() {
                            Err(e) => println!("[Failed to push an architecture] {}", e),
                            _ => (),
                        }
                    }
                    if msgb[1] == "awaiting_data".as_bytes() {
                        // If one worker needs data, generously give all workers data.
                        match self.push_chunk() {
                            Err(e) => println!("[Failed to stream a chunk] {}", e),
                            _ => (),
                        }
                    }
                    if msgb[1] == "training".as_bytes() {
                        // Desired state
                        continue;
                    }
                }

                if msgb[0] == "dump".as_bytes() {
                    if msgb.len() < 3 {
                        continue;
                    }

                    let manifold_binary = &msgb[1];
                    let losses_binary = &msgb[2];

                    let manifold = match Manifold::load(manifold_binary) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };

                    let losses: Vec<f64> = match bincode::deserialize(losses_binary) {
                        Ok(l) => l,
                        Err(_) => continue,
                    };

                    let avg_loss = losses.iter().fold(0., |a, v| a + *v);

                    if best.len() < self.retain {
                        best.push_back((manifold, avg_loss));
                        continue;
                    }

                    for _ in 0..best.len() {
                        let elite = match best.pop_front() {
                            Some(e) => e,
                            None => continue,
                        };

                        if elite.1 < avg_loss {
                            best.push_back(elite);
                        } else {
                            best.push_back((manifold, avg_loss));
                            break;
                        }
                    }

                    completed_archs += 1;
                }
            }

            if completed_archs >= self.arch_epochs {
                break;
            }
        }

        self.kill_workers()?;

        Ok(best.into_iter().map(|e| e.0).collect())
    }
}
